"""Livro-razão de custos: uma linha por invocação de modelo ou lote de embeddings, com tokens, latência e custo.

Cada chamada real ao provedor e cada acerto de cache de voto vira uma linha JSON em data/custos/chamadas.jsonl.
O custo sai da tabela de preços versionada em precos.json; `classificar custos` agrega o livro.
"""
from __future__ import annotations

import contextvars
import csv
import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from classificador.config import CAMINHOS, ler_env

registro = logging.getLogger(__name__)

TABELA_PRECOS: dict = json.loads(Path(__file__).with_name("precos.json").read_text(encoding="utf-8"))

CAMPOS_USO = ("tokens_entrada", "tokens_saida", "tokens_raciocinio", "tokens_cache_leitura", "tokens_cache_escrita")
CAMPOS = (
    "quando", "execucao_id", "etapa", "alvo", "id_questao", "provedor", "modelo", "configuracao", "origem",
    "tentativa", "hash_prompt", "tamanho_prompt_chars", "n_candidatos", "n_textos", *CAMPOS_USO,
    "latencia_ms", "custo_usd", "preco_versao", "sucesso", "erro", "tema_votado", "confianca",
)
VARIAVEL_EXECUCAO = "CLASSIFICADOR_EXECUCAO_ID"
TAMANHO_ERRO = 200
# O custo por questão soma estas etapas do alvo questoes: são as que chamam modelo para cada uma das 359.
ETAPAS_POR_QUESTAO = ("votos", "reparo")
SEM_ROTULO = "(sem)"


def agora_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --- preços e uso ----------------------------------------------------------------------------------------


def calcular_custo(modelo: str | None, uso: dict | None) -> float | None:
    """Custo em dólares pela tabela; None sem uso ou com modelo fora dela.

    Os tokens de raciocínio já vêm dentro de tokens_saida nos dois provedores e não entram de novo. Leitura e escrita
    de cache de prompt contam como entrada comum: a tabela não tem preço separado para elas.
    """
    if uso is None:
        return None
    precos = TABELA_PRECOS["modelos"].get(modelo)
    if precos is None:
        return None
    entrada = uso.get("tokens_entrada") or 0
    saida = uso.get("tokens_saida") or 0
    return (entrada * precos["entrada_por_milhao"] + saida * precos["saida_por_milhao"]) / 1_000_000


def normalizar_uso(usage_metadata: dict | None) -> dict | None:
    """Os cinco campos de uso do livro a partir do usage_metadata do LangChain; zero no que o provedor não informa."""
    if not usage_metadata:
        return None
    entrada = usage_metadata.get("input_token_details") or {}
    saida = usage_metadata.get("output_token_details") or {}
    return {
        "tokens_entrada": int(usage_metadata.get("input_tokens") or 0),
        "tokens_saida": int(usage_metadata.get("output_tokens") or 0),
        "tokens_raciocinio": int(saida.get("reasoning") or 0),
        "tokens_cache_leitura": int(entrada.get("cache_read") or 0),
        "tokens_cache_escrita": int(entrada.get("cache_creation") or 0),
    }


def descrever_erro(erro: BaseException) -> str:
    """Tipo e começo da mensagem do erro, para a linha do livro."""
    return f"{type(erro).__name__}: {str(erro)[:TAMANHO_ERRO]}"


# --- contexto da chamada e identificador da execução -----------------------------------------------------


@dataclass(frozen=True)
class Contexto:
    """Onde a chamada acontece: etapa do pipeline, alvo (questoes, corpus ou api) e questão."""

    etapa: str | None = None
    alvo: str | None = None
    id_questao: str | None = None


_CONTEXTO: contextvars.ContextVar[Contexto] = contextvars.ContextVar("contexto_de_custos", default=Contexto())


def contexto_atual() -> Contexto:
    return _CONTEXTO.get()


@contextmanager
def contexto(etapa: str | None = None, alvo: str | None = None, id_questao: str | None = None) -> Iterator[Contexto]:
    """Marca as linhas registradas dentro do bloco; o que não for passado herda do contexto de fora."""
    anterior = _CONTEXTO.get()
    atual = Contexto(
        anterior.etapa if etapa is None else etapa,
        anterior.alvo if alvo is None else alvo,
        anterior.id_questao if id_questao is None else id_questao,
    )
    token = _CONTEXTO.set(atual)
    try:
        yield atual
    finally:
        _CONTEXTO.reset(token)


def novo_execucao_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{os.getpid()}"


# Um por processo; a variável de ambiente, quando definida, vale na hora de registrar.
EXECUCAO_ID = ler_env(VARIAVEL_EXECUCAO) or novo_execucao_id()


def execucao_id() -> str:
    return ler_env(VARIAVEL_EXECUCAO) or EXECUCAO_ID


# --- livro-razão -----------------------------------------------------------------------------------------


class LivroRazao:
    """Arquivo JSONL só de acréscimo, com uma linha por invocação."""

    def __init__(self, caminho: Path = CAMINHOS.custos / "chamadas.jsonl") -> None:
        self.caminho = Path(caminho)
        # A API atende requisições em threads; a trava impede duas linhas misturadas no arquivo.
        self._trava = threading.Lock()

    def registrar(self, uso: dict | None = None, **campos) -> dict:
        """Acrescenta uma linha ao livro e a devolve.

        `uso`, no formato de normalizar_uso, preenche os cinco campos de tokens. Etapa, alvo e questão vêm do
        contexto, salvo se passados. Sem custo_usd explícito, acerto de cache custa 0.0 e chamada real custa pela
        tabela, ou None se o uso ou o preço do modelo for desconhecido.
        """
        desconhecidos = sorted(set(campos) - set(CAMPOS))
        if desconhecidos:
            raise TypeError(f"campo(s) desconhecido(s) no livro-razão: {', '.join(desconhecidos)}")
        atual = contexto_atual()
        linha: dict = dict.fromkeys(CAMPOS)
        linha.update(
            quando=agora_iso(), execucao_id=execucao_id(), etapa=atual.etapa, alvo=atual.alvo,
            id_questao=atual.id_questao, preco_versao=TABELA_PRECOS["versao"], sucesso=True,
        )
        if uso is not None:
            linha.update({campo: uso.get(campo) for campo in CAMPOS_USO})
        linha.update(campos)
        if "custo_usd" not in campos:
            linha["custo_usd"] = 0.0 if linha["origem"] == "cache" else calcular_custo(linha["modelo"], uso)
        texto = json.dumps(linha, ensure_ascii=False)
        with self._trava:
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
            with self.caminho.open("a", encoding="utf-8", newline="\n") as arquivo:
                arquivo.write(texto + "\n")
        return linha


LIVRO = LivroRazao()


# --- resumo ----------------------------------------------------------------------------------------------


def ler_livro(caminho: Path) -> tuple[list[dict], int]:
    """Linhas do livro e quantas estavam quebradas (uma gravação interrompida, por exemplo)."""
    if not caminho.exists():
        return [], 0
    linhas: list[dict] = []
    invalidas = 0
    with caminho.open(encoding="utf-8") as arquivo:
        for texto in arquivo:
            if not texto.strip():
                continue
            try:
                linha = json.loads(texto)
            except json.JSONDecodeError:
                invalidas += 1
                continue
            if isinstance(linha, dict):
                linhas.append(linha)
            else:
                invalidas += 1
    return linhas, invalidas


def _grupo_vazio() -> dict:
    return {"linhas": 0, "uso_desconhecido": 0, "custo_desconhecido": 0,
            "tokens_entrada": 0, "tokens_saida": 0, "tokens_raciocinio": 0, "custo_usd": 0.0}


def _somar(grupo: dict, linha: dict) -> None:
    grupo["linhas"] += 1
    if linha.get("tokens_entrada") is None:
        grupo["uso_desconhecido"] += 1
    for campo in ("tokens_entrada", "tokens_saida", "tokens_raciocinio"):
        grupo[campo] += linha.get(campo) or 0
    custo = linha.get("custo_usd")
    if custo is None:
        grupo["custo_desconhecido"] += 1
    else:
        grupo["custo_usd"] += custo


def _arredondar(grupo: dict) -> dict:
    return {**grupo, "custo_usd": round(grupo["custo_usd"], 8)}


def _custo(linhas: list[dict]) -> float:
    return sum(linha.get("custo_usd") or 0.0 for linha in linhas)


def _dividir(custo: float, quantidade: int | None) -> float | None:
    return None if not quantidade else round(custo / quantidade, 8)


def _contar_questoes(caminho: Path) -> int | None:
    if not caminho.exists():
        return None
    with caminho.open(encoding="utf-8") as arquivo:
        return sum(1 for linha in arquivo if linha.strip())


def _contar_publicadas(caminho: Path) -> int | None:
    if not caminho.exists():
        return None
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        return sum(1 for linha in csv.DictReader(arquivo) if linha.get("decisao") == "publicar")


AGRUPAMENTOS: dict[str, Callable[[dict], str | None]] = {
    "por_execucao": lambda linha: linha.get("execucao_id"),
    "por_etapa": lambda linha: linha.get("etapa"),
    "por_modelo": lambda linha: f"{linha.get('provedor') or SEM_ROTULO}/{linha.get('modelo') or SEM_ROTULO}",
    "por_origem": lambda linha: linha.get("origem"),
}


def resumir_custos(
    caminho: Path = CAMINHOS.custos / "chamadas.jsonl",
    painel: dict[str, float] | None = None,
    decisoes: Path = CAMINHOS.intermediarios / "decisoes_questoes.jsonl",
    publicacao: Path = CAMINHOS.publicacao,
    saida: Path | None = CAMINHOS.custos / "custos_resumo.json",
) -> dict:
    """Agrega o livro por execução, etapa, modelo e origem e grava o resumo em `saida`.

    Linha sem uso conta em uso_desconhecido e entra com zero tokens; linha sem custo entra com zero na soma e conta
    em custo_desconhecido. O custo por questão é o de votos e reparo do alvo questoes, dividido pelas questões das
    decisões e pelas publicadas de publicacao.csv. `painel` traz o valor que cada provedor cobrou, informado à mão,
    e é comparado com o custo calculado das chamadas reais. Sem nenhuma linha, nada é gravado.
    """
    linhas, invalidas = ler_livro(caminho)
    resumo: dict = {"livro": str(caminho), "preco_versao": TABELA_PRECOS["versao"],
                    "linhas": len(linhas), "linhas_invalidas": invalidas}
    if not linhas:
        return resumo

    total = _grupo_vazio()
    grupos: dict[str, dict[str, dict]] = {nome: {} for nome in AGRUPAMENTOS}
    for linha in linhas:
        _somar(total, linha)
        for nome, chave in AGRUPAMENTOS.items():
            _somar(grupos[nome].setdefault(str(chave(linha) or SEM_ROTULO), _grupo_vazio()), linha)

    custo_questoes = _custo([l for l in linhas if l.get("etapa") in ETAPAS_POR_QUESTAO and l.get("alvo") == "questoes"])
    questoes = _contar_questoes(decisoes)
    publicadas = _contar_publicadas(publicacao)

    comparacao: dict[str, dict] = {}
    for provedor, informado in (painel or {}).items():
        reais = [l for l in linhas if l.get("origem") == "api" and l.get("provedor") == provedor]
        calculado = round(_custo(reais), 8)
        diferenca = round(informado - calculado, 8)
        comparacao[provedor] = {
            "informado": informado,
            "calculado": calculado,
            "diferenca": diferenca,
            "diferenca_percentual": None if not informado else round(100 * diferenca / informado, 2),
            "linhas_sem_custo": sum(1 for l in reais if l.get("custo_usd") is None),
        }

    resumo.update(
        gerado_em=agora_iso(),
        total=_arredondar(total),
        **{nome: {chave: _arredondar(g) for chave, g in por.items()} for nome, por in grupos.items()},
        por_questao={
            "etapas": list(ETAPAS_POR_QUESTAO),
            "alvo": "questoes",
            "custo_usd": round(custo_questoes, 8),
            "questoes": questoes,
            "custo_por_questao": _dividir(custo_questoes, questoes),
            "publicadas": publicadas,
            "custo_por_publicada": _dividir(custo_questoes, publicadas),
        },
        painel=comparacao,
    )
    if saida is not None:
        saida.parent.mkdir(parents=True, exist_ok=True)
        saida.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        registro.info("resumo de custos gravado em %s", saida)
    return resumo


COLUNAS = (
    ("linhas", "linhas"),
    ("uso_desconhecido", "uso desconhecido"),
    ("tokens_entrada", "entrada"),
    ("tokens_saida", "saída"),
    ("tokens_raciocinio", "raciocínio"),
    ("custo_usd", "custo US$"),
)


def _tabela(titulo: str, grupos: dict[str, dict]) -> list[str]:
    largura = max(len(titulo), *(len(rotulo) for rotulo in grupos))
    larguras = [max(len(nome), 10) for _, nome in COLUNAS]
    linhas = [f"{titulo:<{largura}}" + "".join(f"  {nome:>{w}}" for (_, nome), w in zip(COLUNAS, larguras))]
    for rotulo, grupo in grupos.items():
        valores = [f"{grupo[campo]:.6f}" if campo == "custo_usd" else str(grupo[campo]) for campo, _ in COLUNAS]
        linhas.append(f"{rotulo:<{largura}}" + "".join(f"  {valor:>{w}}" for valor, w in zip(valores, larguras)))
    return linhas


def formatar_resumo(resumo: dict) -> str:
    """Tabela em texto simples para o terminal."""
    if not resumo["linhas"]:
        return f"Nenhum registro no livro-razão em {resumo['livro']}."
    saida = [f"Livro-razão: {resumo['livro']}"]
    cabecalho = f"{resumo['linhas']} linha(s), preços da tabela de {resumo['preco_versao']}"
    if resumo["linhas_invalidas"]:
        cabecalho += f"; {resumo['linhas_invalidas']} linha(s) quebrada(s) ignorada(s)"
    saida.append(cabecalho)
    for titulo, chave in (("Por execução", "por_execucao"), ("Por etapa", "por_etapa"),
                          ("Por modelo", "por_modelo"), ("Por origem", "por_origem")):
        saida += ["", *_tabela(titulo, resumo[chave])]
    saida += ["", *_tabela("Total", {"total": resumo["total"]})]
    if resumo["total"]["custo_desconhecido"]:
        saida.append(f"{resumo['total']['custo_desconhecido']} linha(s) sem custo calculado (uso desconhecido ou "
                     "modelo fora da tabela) entram com zero na soma.")

    unitario = resumo["por_questao"]
    saida.append("")
    if unitario["custo_por_questao"] is None:
        saida.append("Custo por questão: indisponível (sem as decisões das questões)")
    else:
        saida.append(f"Custo por questão (votos e reparo do alvo questoes): US$ {unitario['custo_por_questao']:.6f} "
                     f"em {unitario['questoes']} questões, US$ {unitario['custo_usd']:.6f} no total")
    if unitario["custo_por_publicada"] is None:
        saida.append("Custo por publicada: indisponível (sem publicacao.csv ou sem publicadas)")
    else:
        saida.append(f"Custo por publicada: US$ {unitario['custo_por_publicada']:.6f} em {unitario['publicadas']} publicadas")

    if resumo["painel"]:
        saida += ["", "Painel do provedor contra o custo calculado das chamadas reais"]
        largura = max(len(provedor) for provedor in resumo["painel"])
        for provedor, valores in resumo["painel"].items():
            percentual = "" if valores["diferenca_percentual"] is None else f" ({valores['diferenca_percentual']:+.1f}%)"
            saida.append(f"  {provedor:<{largura}}  informado {valores['informado']:.6f}  calculado "
                         f"{valores['calculado']:.6f}  diferença {valores['diferenca']:+.6f}{percentual}")
    return "\n".join(saida)
