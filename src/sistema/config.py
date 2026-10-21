"""Caminhos do projeto, semente e leitura de variáveis de ambiente."""
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
    """Anexos do case em anexos/, artefatos versionados em dados/, entregáveis na raiz."""

    raiz: Path
    anexos: Path
    questoes: Path
    temas: Path
    folha: Path
    dados: Path
    temas_enriquecidos: Path
    exemplos: Path
    dicionario_leis: Path
    dicionario_ajustes: Path
    termos_fora: Path
    corpus_v1: Path
    embeddings: Path
    intermediarios: Path
    custos: Path
    publicacao: Path
    previsao: Path


def _caminhos(raiz: Path) -> Caminhos:
    anexos = raiz / "anexos"
    dados = raiz / "dados"
    return Caminhos(
        raiz=raiz,
        anexos=anexos,
        questoes=anexos / "questoes.jsonl",
        temas=anexos / "temas.csv",
        folha=anexos / "folha_verdade.csv",
        dados=dados,
        temas_enriquecidos=dados / "temas_enriquecidos.jsonl",
        exemplos=dados / "exemplos.jsonl",
        dicionario_leis=dados / "dicionario_leis.json",
        dicionario_ajustes=dados / "dicionario_leis_ajustes.json",
        termos_fora=dados / "termos_fora.json",
        corpus_v1=dados / "fontes" / "corpus_v1.jsonl",
        embeddings=dados / "embeddings",
        intermediarios=dados / "intermediarios",
        custos=dados / "custos",
        publicacao=raiz / "publicacao.csv",
        previsao=raiz / "previsao.txt",
    )


CAMINHOS = _caminhos(RAIZ)


def ler_env(nome: str, padrao: str | None = None) -> str | None:
    """Lê uma variável de ambiente; vazio conta como ausente."""
    valor = os.environ.get(nome)
    if valor is None or valor.strip() == "":
        return padrao
    return valor.strip()


class ChaveAusenteError(RuntimeError):
    """Chave de API ausente para uma etapa que precisa chamar o provedor."""
