"""Caminhos do projeto e leitura de variáveis de ambiente."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[2]
load_dotenv(RAIZ / ".env")

SEMENTE = 42


@dataclass(frozen=True)
class Caminhos:
    anexos: Path
    questoes: Path
    temas: Path
    corpus: Path
    intermediarios: Path
    cache: Path
    custos: Path
    ancora_planilha: Path
    ancora_manual: Path
    publicacao: Path
    previsao: Path
    prompts: Path


CAMINHOS = Caminhos(
    anexos=RAIZ / "anexos",
    questoes=RAIZ / "anexos" / "questoes.jsonl",
    temas=RAIZ / "anexos" / "temas.csv",
    corpus=RAIZ / "data" / "corpus_referencia.jsonl",
    intermediarios=RAIZ / "data" / "intermediarios",
    cache=RAIZ / "data" / "cache",
    custos=RAIZ / "data" / "custos",
    ancora_planilha=RAIZ / "data" / "ancora_para_rotular.csv",
    ancora_manual=RAIZ / "data" / "ancora_manual.csv",
    publicacao=RAIZ / "publicacao.csv",
    previsao=RAIZ / "previsao.txt",
    prompts=RAIZ / "src" / "classificador" / "prompts",
)


def ler_env(nome: str, padrao: str | None = None) -> str | None:
    """Lê uma variável de ambiente; vazio conta como ausente."""
    valor = os.environ.get(nome)
    if valor is None or valor.strip() == "":
        return padrao
    return valor.strip()


class ChaveAusenteError(RuntimeError):
    """Chave de API ausente e sem cache para atender a chamada."""
