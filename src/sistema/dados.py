"""Carga e limpeza das questões, dos temas e do corpus."""
from __future__ import annotations

import html as modulo_html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from classificador.config import CAMINHOS

_ESPACOS = re.compile(r"\s+")


def limpar_html(texto_html: str) -> str:
    """Remove tags, entidades e os '\\n' e '\\t' literais que vêm do banco; colapsa espaços."""
    if not texto_html:
        return ""
    sem_tags = BeautifulSoup(texto_html, "html.parser").get_text(" ")
    sem_entidades = modulo_html.unescape(sem_tags).replace("\xa0", " ")
    sem_barra_n = sem_entidades.replace("\\n", " ").replace("\\t", " ")
    return _ESPACOS.sub(" ", sem_barra_n).strip()


def montar_texto(enunciado_html: str, comando: str, alternativas: dict[str, str]) -> str:
    """Texto que os modelos e os embeddings recebem: enunciado, comando e alternativas."""
    partes = [limpar_html(enunciado_html)]
    comando_limpo = limpar_html(comando or "")
    if comando_limpo:
        partes.append(f"Comando: {comando_limpo}")
    if alternativas:
        partes.append("Alternativas:")
        for letra, texto in alternativas.items():
            partes.append(f"{letra}) {limpar_html(texto)}")
    return "\n".join(partes)


def ler_jsonl(caminho: Path) -> list[dict]:
    with open(caminho, encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def escrever_jsonl(caminho: Path, linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as arquivo:
        for linha in linhas:
            arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")


def carregar_questoes(caminho: Path = CAMINHOS.questoes) -> pd.DataFrame:
    linhas = ler_jsonl(caminho)
    df = pd.DataFrame(linhas)
    df["comando"] = df["comando"].fillna("")
    df["disciplina_origem"] = df["disciplina_origem"].fillna("")
    df["assunto_origem"] = df["assunto_origem"].fillna("")
    df["texto"] = [
        montar_texto(e, c, a)
        for e, c, a in zip(df["enunciado_html"], df["comando"], df["alternativas"])
    ]
    return df


def carregar_temas(caminho: Path = CAMINHOS.temas) -> pd.DataFrame:
    df = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    df["tema_id"] = df["tema_id"].astype(int)
    df["parent_id"] = pd.to_numeric(df["parent_id"].replace("", np.nan)).astype("Int64")
    df["nivel"] = df["nivel"].astype(int)
    df["ativo"] = df["status"].str.strip().str.lower() == "active"
    return df


def carregar_corpus(caminho: Path = CAMINHOS.corpus) -> pd.DataFrame:
    linhas = ler_jsonl(caminho)
    df = pd.DataFrame(linhas)
    if "comando" not in df.columns:
        df["comando"] = ""
    df["comando"] = df["comando"].fillna("")
    df["tema_id"] = df["tema_id"].astype(int)
    df["texto"] = [
        montar_texto(e, c, a)
        for e, c, a in zip(df["enunciado"], df["comando"], df["alternativas"])
    ]
    return df
