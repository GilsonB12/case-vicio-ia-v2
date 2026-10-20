"""Embeddings da OpenAI via LangChain, com cache em disco por texto."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from functools import lru_cache
from pathlib import Path

import tiktoken
from langchain_core.embeddings import Embeddings

from classificador import custos
from classificador.config import CAMINHOS, ChaveAusenteError, ler_env

registro = logging.getLogger(__name__)

MODELO_PADRAO = "text-embedding-3-large"
# Tokenizador dos modelos text-embedding-3.
CODIFICACAO = "cl100k_base"


@lru_cache(maxsize=1)
def _codificador() -> tiktoken.Encoding:
    return tiktoken.get_encoding(CODIFICACAO)


def contar_tokens(textos: list[str]) -> int | None:
    """Tokens de entrada de um lote; None se o tokenizador não carregar (o tiktoken baixa o arquivo na primeira vez)."""
    try:
        codificador = _codificador()
    except Exception as erro:
        registro.warning("tokenizador %s indisponível (%s); lote registrado com uso desconhecido", CODIFICACAO, erro)
        return None
    return sum(len(codificador.encode(texto, disallowed_special=())) for texto in textos)


class EmbeddingsComCache(Embeddings):
    """Lê do cache o que já foi calculado; chama a base só para o que falta."""

    def __init__(
        self, base: Embeddings | None, modelo: str, pasta_cache: Path, livro: custos.LivroRazao | None = None
    ) -> None:
        self.base = base
        self.modelo = modelo
        self.pasta = pasta_cache / "embeddings" / modelo
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.livro = livro if livro is not None else custos.LIVRO
        self.chamadas_base = 0
        self.acertos_cache = 0

    def _arquivo(self, texto: str) -> Path:
        chave = hashlib.sha256(f"{self.modelo}\n{texto}".encode("utf-8")).hexdigest()
        return self.pasta / f"{chave}.json"

    def _ler(self, texto: str) -> list[float] | None:
        arquivo = self._arquivo(texto)
        if not arquivo.exists():
            return None
        try:
            vetor = json.loads(arquivo.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            arquivo.unlink()
            return None
        if not isinstance(vetor, list) or not vetor or not all(isinstance(x, (int, float)) for x in vetor):
            arquivo.unlink()
            return None
        return [float(x) for x in vetor]

    def _gravar(self, texto: str, vetor: list[float]) -> None:
        self._arquivo(texto).write_text(json.dumps(vetor), encoding="utf-8")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resultado: list[list[float] | None] = [self._ler(t) for t in texts]
        self.acertos_cache += sum(1 for v in resultado if v is not None)
        faltantes = [i for i, v in enumerate(resultado) if v is None]
        if faltantes:
            if self.base is None:
                raise ChaveAusenteError(
                    f"{len(faltantes)} texto(s) sem embedding em cache e OPENAI_API_KEY ausente. "
                    "Defina a chave no .env ou use o cache commitado."
                )
            lote = [texts[i] for i in faltantes]
            inicio = time.perf_counter()
            try:
                novos = self.base.embed_documents(lote)
            except Exception as erro:
                self._anotar(lote, inicio, erro)
                raise
            self._anotar(lote, inicio)
            self.chamadas_base += len(faltantes)
            if len(novos) != len(faltantes):
                raise RuntimeError(
                    f"a base de embeddings devolveu {len(novos)} vetor(es) para {len(faltantes)} texto(s); "
                    "nada foi gravado no cache"
                )
            for indice, vetor in zip(faltantes, novos, strict=True):
                vetor = [float(x) for x in vetor]
                self._gravar(texts[indice], vetor)
                resultado[indice] = vetor
        return [v for v in resultado if v is not None]

    def _anotar(self, lote: list[str], inicio: float, erro: BaseException | None = None) -> None:
        """Uma linha no livro-razão para o lote que foi ao provedor; acerto de cache não gera linha."""
        tokens = None if erro is not None else contar_tokens(lote)
        uso = None if tokens is None else {
            "tokens_entrada": tokens, "tokens_saida": 0, "tokens_raciocinio": 0,
            "tokens_cache_leitura": 0, "tokens_cache_escrita": 0,
        }
        self.livro.registrar(
            etapa=custos.contexto_atual().etapa or "embeddings", provedor="openai", modelo=self.modelo,
            origem="api", tentativa=1, n_textos=len(lote), uso=uso,
            latencia_ms=round((time.perf_counter() - inicio) * 1000, 1), sucesso=erro is None,
            erro=None if erro is None else custos.descrever_erro(erro),
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def criar_embeddings(pasta_cache: Path = CAMINHOS.cache) -> EmbeddingsComCache:
    modelo = ler_env("MODELO_EMBEDDINGS", MODELO_PADRAO) or MODELO_PADRAO
    base: Embeddings | None = None
    if ler_env("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        base = OpenAIEmbeddings(model=modelo)
    return EmbeddingsComCache(base, modelo, pasta_cache)
