"""Cache de embeddings em disco: o sistema só lê; `sistema embeddings` é a única etapa que chama a API.

Cada texto vira um arquivo `dados/embeddings/<modelo>/<sha256 do texto>.json` com a lista do vetor. A leitura
(`vetor`, `matriz`) nunca importa cliente de modelo nem toca na rede: texto fora do cache levanta
`EmbeddingAusenteError` com a lista do que falta. `gerar` chama a OpenAI só para os ausentes, em lotes, e registra
cada lote no livro-razão de custos.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from sistema import custos
from sistema.config import CAMINHOS, ChaveAusenteError, ler_env

registro = logging.getLogger(__name__)

MODELO_PADRAO = "text-embedding-3-large"
TAMANHO_LOTE = 100
LIMITE_LISTA = 10
TAMANHO_AMOSTRA = 80


def _amostra(texto: str) -> str:
    plano = " ".join(texto.split())
    return plano if len(plano) <= TAMANHO_AMOSTRA else plano[: TAMANHO_AMOSTRA - 1] + "…"


class EmbeddingAusenteError(RuntimeError):
    """Textos sem embedding no cache; `textos_faltando` traz todos, a mensagem mostra os primeiros."""

    def __init__(self, textos_faltando: Iterable[str]) -> None:
        self.textos_faltando = list(textos_faltando)
        total = len(self.textos_faltando)
        linhas = [f"{total} texto(s) sem embedding no cache; gere com `uv run sistema embeddings` "
                  "(a única etapa que chama a API)."]
        linhas += [f"  - {_amostra(texto)}" for texto in self.textos_faltando[:LIMITE_LISTA]]
        if total > LIMITE_LISTA:
            linhas.append(f"  … e mais {total - LIMITE_LISTA}")
        super().__init__("\n".join(linhas))


# Pacote versionado com os vetores que o sistema usa (ver preparacao/empacotar_embeddings.py).
ARQUIVO_PACOTE = "vetores.npz"


class CacheEmbeddings:
    def __init__(self, pasta: Path = CAMINHOS.embeddings, modelo: str = MODELO_PADRAO) -> None:
        self.modelo = modelo
        self.pasta = Path(pasta) / modelo
        self._memoria: dict[str, np.ndarray] = {}
        self._pacote = Path(pasta) / ARQUIVO_PACOTE
        self._indice_pacote: dict[str, int] | None = None
        self._matriz_pacote: np.ndarray | None = None

    def _carregar_pacote(self) -> None:
        """Lê uma vez o pacote versionado (`vetores.npz`: hashes sha256 e matriz float32), se existir."""
        if self._indice_pacote is not None:
            return
        self._indice_pacote = {}
        if not self._pacote.exists():
            return
        with np.load(self._pacote, allow_pickle=False) as dados:
            hashes = [str(h) for h in dados["hashes"]]
            self._matriz_pacote = np.asarray(dados["matriz"], dtype=np.float32)
        self._indice_pacote = {h: i for i, h in enumerate(hashes)}

    def arquivo(self, texto: str) -> Path:
        return self.pasta / f"{hashlib.sha256(texto.encode('utf-8')).hexdigest()}.json"

    # --- leitura (sem rede) ------------------------------------------------------------------------------

    def _ler(self, texto: str) -> np.ndarray | None:
        """Vetor do cache, ou None se o arquivo não existe ou não tem um vetor válido."""
        if texto in self._memoria:
            return self._memoria[texto]
        self._carregar_pacote()
        linha = self._indice_pacote.get(hashlib.sha256(texto.encode("utf-8")).hexdigest())
        if linha is not None:
            vetor = self._matriz_pacote[linha]
            self._memoria[texto] = vetor
            return vetor
        try:
            bruto = json.loads(self.arquivo(texto).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError):
            registro.warning("arquivo de cache inválido para %r; conta como ausente", _amostra(texto))
            return None
        if not isinstance(bruto, list) or not bruto or not all(isinstance(x, (int, float)) for x in bruto):
            registro.warning("arquivo de cache sem vetor para %r; conta como ausente", _amostra(texto))
            return None
        vetor = np.asarray(bruto, dtype=np.float32)
        self._memoria[texto] = vetor
        return vetor

    def faltando(self, textos: Iterable[str]) -> list[str]:
        """Textos sem embedding no cache, sem repetição, na ordem em que aparecem."""
        return [texto for texto in dict.fromkeys(textos) if self._ler(texto) is None]

    def matriz(self, textos: Iterable[str]) -> np.ndarray:
        """Uma linha por texto, na ordem dada; levanta EmbeddingAusenteError se algum não está no cache."""
        textos = list(textos)
        vetores = [self._ler(texto) for texto in textos]
        ausentes = [texto for texto, vetor in zip(textos, vetores) if vetor is None]
        if ausentes:
            raise EmbeddingAusenteError(dict.fromkeys(ausentes))
        if not vetores:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(vetores)

    def vetor(self, texto: str) -> np.ndarray:
        return self.matriz([texto])[0]

    # --- geração (API) -----------------------------------------------------------------------------------

    def gerar(self, textos: Iterable[str], cliente=None, livro: custos.LivroRazao | None = None) -> int:
        """Gera e grava no cache os embeddings que faltam; devolve quantos foram gerados.

        Sem `cliente`, cria o da OpenAI com OPENAI_API_KEY (ChaveAusenteError sem ela). Nada a gerar não pede
        chave nem chama nada. Cada lote de até TAMANHO_LOTE textos é uma chamada e uma linha no livro-razão.
        """
        ausentes = self.faltando(textos)
        if not ausentes:
            return 0
        if cliente is None:
            chave = ler_env("OPENAI_API_KEY")
            if not chave:
                raise ChaveAusenteError(
                    f"OPENAI_API_KEY ausente: defina a chave no .env para gerar os {len(ausentes)} embedding(s) "
                    "que faltam no cache"
                )
            from openai import OpenAI

            cliente = OpenAI(api_key=chave)
        livro = livro if livro is not None else custos.LIVRO
        self.pasta.mkdir(parents=True, exist_ok=True)
        for inicio in range(0, len(ausentes), TAMANHO_LOTE):
            lote = ausentes[inicio:inicio + TAMANHO_LOTE]
            self._gerar_lote(lote, cliente, livro)
            registro.info("embeddings: %d de %d gerados", inicio + len(lote), len(ausentes))
        return len(ausentes)

    def _gerar_lote(self, lote: list[str], cliente, livro: custos.LivroRazao) -> None:
        comeco = time.perf_counter()
        try:
            resposta = cliente.embeddings.create(model=self.modelo, input=lote)
        except Exception as erro:
            self._anotar(livro, lote, comeco, None, custos.descrever_erro(erro))
            raise
        tokens = getattr(getattr(resposta, "usage", None), "total_tokens", None)
        uso = None if tokens is None else {"tokens_entrada": int(tokens), "tokens_saida": 0}
        itens = list(resposta.data)
        if len(itens) != len(lote):
            mensagem = (f"a API devolveu {len(itens)} vetor(es) para {len(lote)} texto(s); "
                        "nada deste lote foi gravado no cache")
            self._anotar(livro, lote, comeco, uso, mensagem)
            raise RuntimeError(mensagem)
        self._anotar(livro, lote, comeco, uso)
        # A API devolve `index` com a posição de cada texto no lote; sem ele, vale a ordem da resposta.
        ordenados = sorted(enumerate(itens), key=lambda par: getattr(par[1], "index", par[0]))
        for texto, (_, item) in zip(lote, ordenados):
            self._gravar(texto, item.embedding)

    def _anotar(self, livro, lote: list[str], comeco: float, uso: dict | None, erro: str | None = None) -> None:
        livro.registrar(
            etapa="embeddings", provedor="openai", modelo=self.modelo, n_textos=len(lote), uso=uso,
            latencia_ms=round((time.perf_counter() - comeco) * 1000, 1), sucesso=erro is None, erro=erro,
        )

    def _gravar(self, texto: str, vetor) -> None:
        valores = [float(x) for x in vetor]
        if not valores:
            raise RuntimeError(f"vetor vazio para {_amostra(texto)!r}")
        arquivo = self.arquivo(texto)
        temporario = arquivo.with_name(arquivo.name + ".tmp")
        temporario.write_text(json.dumps(valores), encoding="utf-8")
        os.replace(temporario, arquivo)
        self._memoria[texto] = np.asarray(valores, dtype=np.float32)


def textos_do_sistema() -> list[str]:
    """Todos os textos que o sistema lê do cache, sem repetição.

    Questões, texto dos 371 temas publicáveis (com os enriquecidos disponíveis), exemplos sintéticos e corpus v1
    (o corpus entra sempre, para a comparação de fontes poder rodar com e sem ele).
    """
    from sistema.arvore import ArvoreTemas
    from sistema.dados import carregar_corpus_v1, carregar_exemplos, carregar_questoes, carregar_temas_enriquecidos

    arvore = ArvoreTemas.carregar()
    enriquecidos = carregar_temas_enriquecidos()
    textos = list(carregar_questoes()["texto"])
    textos += [arvore.texto_tema(tema_id, enriquecidos) for tema_id in arvore.candidatos_possiveis()]
    textos += [exemplo["texto"] for exemplo in carregar_exemplos()]
    textos += [item["texto"] for item in carregar_corpus_v1()]
    return list(dict.fromkeys(textos))
