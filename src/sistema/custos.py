"""Livro-razão de custos: uma linha por chamada paga a um provedor, com tokens, latência e custo.

No sistema só `sistema embeddings` chama API (um lote de embeddings por linha). As linhas ficam em
dados/custos/chamadas.jsonl; o custo sai da tabela de preços versionada em precos.json; `sistema custos` agrega o livro.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sistema.config import CAMINHOS, ler_env

registro = logging.getLogger(__name__)

TABELA_PRECOS: dict = json.loads(Path(__file__).with_name("precos.json").read_text(encoding="utf-8"))

CAMPOS_USO = ("tokens_entrada", "tokens_saida")
CAMPOS = (
    "quando", "execucao_id", "etapa", "provedor", "modelo", "n_textos", *CAMPOS_USO,
    "latencia_ms", "custo_usd", "preco_versao", "sucesso", "erro",
)
VARIAVEL_EXECUCAO = "SISTEMA_EXECUCAO_ID"
TAMANHO_ERRO = 200
SEM_ROTULO = "(sem)"


def agora_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --- preços ----------------------------------------------------------------------------------------------


def calcular_custo(modelo: str | None, uso: dict | None) -> float | None:
    """Custo em dólares pela tabela; None sem uso ou com modelo fora dela. Embedding só tem preço de entrada."""
    if uso is None:
        return None
    precos = TABELA_PRECOS["modelos"].get(modelo)
    if precos is None:
        return None
    entrada = uso.get("tokens_entrada") or 0
    saida = uso.get("tokens_saida") or 0
    return (entrada * precos["entrada_por_milhao"] + saida * precos["saida_por_milhao"]) / 1_000_000


def descrever_erro(erro: BaseException) -> str:
    """Tipo e começo da mensagem do erro, para a linha do livro."""
    return f"{type(erro).__name__}: {str(erro)[:TAMANHO_ERRO]}"


# --- identificador da execução ---------------------------------------------------------------------------


def novo_execucao_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{os.getpid()}"


# Um por processo; a variável de ambiente, quando definida, vale na hora de registrar.
EXECUCAO_ID = ler_env(VARIAVEL_EXECUCAO) or novo_execucao_id()


def execucao_id() -> str:
    return ler_env(VARIAVEL_EXECUCAO) or EXECUCAO_ID


# --- livro-razão -----------------------------------------------------------------------------------------


class LivroRazao:
    """Arquivo JSONL só de acréscimo, com uma linha por chamada."""

    def __init__(self, caminho: Path = CAMINHOS.custos / "chamadas.jsonl") -> None:
        self.caminho = Path(caminho)
        # Trava para duas threads não misturarem linhas no arquivo.
        self._trava = threading.Lock()

    def registrar(self, uso: dict | None = None, **campos) -> dict:
        """Acrescenta uma linha ao livro e a devolve.

        `uso` (`tokens_entrada`, `tokens_saida`) preenche os campos de tokens. Sem `custo_usd` explícito, o custo
        sai da tabela, ou fica None se o uso ou o preço do modelo for desconhecido.
        """
        desconhecidos = sorted(set(campos) - set(CAMPOS))
        if desconhecidos:
            raise TypeError(f"campo(s) desconhecido(s) no livro-razão: {', '.join(desconhecidos)}")
        linha: dict = dict.fromkeys(CAMPOS)
        linha.update(quando=agora_iso(), execucao_id=execucao_id(), preco_versao=TABELA_PRECOS["versao"], sucesso=True)
        if uso is not None:
            linha.update({campo: uso.get(campo) for campo in CAMPOS_USO})
        linha.update(campos)
        if "custo_usd" not in campos:
            linha["custo_usd"] = calcular_custo(linha["modelo"], uso)
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
    caminho = Path(caminho)
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
    return {"linhas": 0, "uso_desconhecido": 0, "custo_desconhecido": 0, "n_textos": 0,
            "tokens_entrada": 0, "tokens_saida": 0, "custo_usd": 0.0}


def _somar(grupo: dict, linha: dict) -> None:
    grupo["linhas"] += 1
    if linha.get("tokens_entrada") is None:
        grupo["uso_desconhecido"] += 1
    for campo in ("n_textos", "tokens_entrada", "tokens_saida"):
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


AGRUPAMENTOS: dict[str, Callable[[dict], str | None]] = {
    "por_execucao": lambda linha: linha.get("execucao_id"),
    "por_etapa": lambda linha: linha.get("etapa"),
    "por_modelo": lambda linha: f"{linha.get('provedor') or SEM_ROTULO}/{linha.get('modelo') or SEM_ROTULO}",
}


def resumir_custos(
    caminho: Path = CAMINHOS.custos / "chamadas.jsonl",
    painel: dict[str, float] | None = None,
    saida: Path | None = CAMINHOS.custos / "custos_resumo.json",
) -> dict:
    """Agrega o livro por execução, etapa e modelo e grava o resumo em `saida`.

    Linha sem uso conta em uso_desconhecido e entra com zero tokens; linha sem custo entra com zero na soma e conta
    em custo_desconhecido. `painel` traz o valor que cada provedor cobrou, informado à mão, e é comparado com o
    custo calculado das suas linhas. Sem nenhuma linha, nada é gravado.
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

    comparacao: dict[str, dict] = {}
    for provedor, informado in (painel or {}).items():
        do_provedor = [l for l in linhas if l.get("provedor") == provedor]
        calculado = round(_custo(do_provedor), 8)
        diferenca = round(informado - calculado, 8)
        comparacao[provedor] = {
            "informado": informado,
            "calculado": calculado,
            "diferenca": diferenca,
            "diferenca_percentual": None if not informado else round(100 * diferenca / informado, 2),
            "linhas_sem_custo": sum(1 for l in do_provedor if l.get("custo_usd") is None),
        }

    resumo.update(
        gerado_em=agora_iso(),
        total=_arredondar(total),
        **{nome: {chave: _arredondar(g) for chave, g in por.items()} for nome, por in grupos.items()},
        painel=comparacao,
    )
    if saida is not None:
        saida = Path(saida)
        saida.parent.mkdir(parents=True, exist_ok=True)
        saida.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        registro.info("resumo de custos gravado em %s", saida)
    return resumo


COLUNAS = (
    ("linhas", "linhas"),
    ("uso_desconhecido", "uso desconhecido"),
    ("n_textos", "textos"),
    ("tokens_entrada", "entrada"),
    ("tokens_saida", "saída"),
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
    for titulo, chave in (("Por execução", "por_execucao"), ("Por etapa", "por_etapa"), ("Por modelo", "por_modelo")):
        saida += ["", *_tabela(titulo, resumo[chave])]
    saida += ["", *_tabela("Total", {"total": resumo["total"]})]
    if resumo["total"]["custo_desconhecido"]:
        saida.append(f"{resumo['total']['custo_desconhecido']} linha(s) sem custo calculado (falha, uso desconhecido "
                     "ou modelo fora da tabela) entram com zero na soma.")

    if resumo["painel"]:
        saida += ["", "Painel do provedor contra o custo calculado do livro"]
        largura = max(len(provedor) for provedor in resumo["painel"])
        for provedor, valores in resumo["painel"].items():
            percentual = "" if valores["diferenca_percentual"] is None else f" ({valores['diferenca_percentual']:+.1f}%)"
            saida.append(f"  {provedor:<{largura}}  informado {valores['informado']:.6f}  calculado "
                         f"{valores['calculado']:.6f}  diferença {valores['diferenca']:+.6f}{percentual}")
    return "\n".join(saida)
