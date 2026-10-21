"""Carga e limpeza das questões, dos temas, da folha-verdade e das fontes de conhecimento em dados/."""
from __future__ import annotations

import html as modulo_html
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from sistema.config import CAMINHOS

if TYPE_CHECKING:
    from sistema.arvore import ArvoreTemas

_ESPACOS = re.compile(r"\s+")

COLUNAS_QUESTOES = ["id", "texto", "enunciado_html", "comando", "alternativas", "disciplina_origem", "assunto_origem"]
COLUNAS_FOLHA = ["id", "tema_central_id", "extras_ids", "fora"]


# --- texto -----------------------------------------------------------------------------------------------


def limpar_html(texto_html: str | None) -> str:
    """Remove tags, entidades e os '\\n' e '\\t' literais que vêm do banco; colapsa espaços."""
    if not texto_html:
        return ""
    sem_tags = BeautifulSoup(texto_html, "html.parser").get_text(" ")
    sem_entidades = modulo_html.unescape(sem_tags).replace("\xa0", " ")
    sem_barra_n = sem_entidades.replace("\\n", " ").replace("\\t", " ")
    return _ESPACOS.sub(" ", sem_barra_n).strip()


def montar_texto(enunciado_html: str, comando: str, alternativas: Mapping[str, str] | None) -> str:
    """Texto da questão para busca e embeddings: enunciado, comando e alternativas (nunca disciplina nem assunto)."""
    partes = [limpar_html(enunciado_html)]
    comando_limpo = limpar_html(comando or "")
    if comando_limpo:
        partes.append(f"Comando: {comando_limpo}")
    if alternativas:
        partes.append("Alternativas:")
        for letra, texto in alternativas.items():
            partes.append(f"{letra}) {limpar_html(texto)}")
    return "\n".join(partes)


# --- JSONL -----------------------------------------------------------------------------------------------


def ler_jsonl(caminho: Path) -> list[dict]:
    with open(caminho, encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def escrever_jsonl(caminho: Path, linhas: list[dict]) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="\n") as arquivo:
        for linha in linhas:
            arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")


# --- anexos ----------------------------------------------------------------------------------------------


def carregar_questoes(caminho: Path = CAMINHOS.questoes) -> pd.DataFrame:
    """As 659 questões, com o texto montado de enunciado, comando e alternativas."""
    df = pd.DataFrame(ler_jsonl(caminho))
    for coluna in ("comando", "disciplina_origem", "assunto_origem"):
        df[coluna] = df[coluna].fillna("")
    df["alternativas"] = [a if isinstance(a, dict) else {} for a in df["alternativas"]]
    df["texto"] = [
        montar_texto(e, c, a)
        for e, c, a in zip(df["enunciado_html"], df["comando"], df["alternativas"])
    ]
    return df[COLUNAS_QUESTOES]


def carregar_temas(caminho: Path = CAMINHOS.temas) -> pd.DataFrame:
    df = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    df["tema_id"] = df["tema_id"].astype(int)
    df["parent_id"] = pd.to_numeric(df["parent_id"].replace("", np.nan)).astype("Int64")
    df["nivel"] = df["nivel"].astype(int)
    df["ativo"] = df["status"].str.strip().str.lower() == "active"
    return df


def carregar_folha(caminho: Path = CAMINHOS.folha) -> pd.DataFrame:
    """Folha-verdade: `tema_central_id` inteiro (None quando fora), `extras_ids` lista de inteiros, `fora` booleano."""
    bruto = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    linhas = []
    for registro in bruto.to_dict("records"):
        id_questao = registro["id"].strip()
        marcador = registro["fora_da_taxonomia"].strip()
        central = registro["tema_central_id"].strip()
        if marcador not in ("0", "1"):
            raise ValueError(f"{id_questao}: fora_da_taxonomia deve ser 0 ou 1, veio {marcador!r}")
        fora = marcador == "1"
        if fora == bool(central):
            estado = "preenchido" if fora else "vazio"
            raise ValueError(f"{id_questao}: tema central {estado} com fora_da_taxonomia={marcador}")
        linhas.append({
            "id": id_questao,
            "tema_central_id": int(central) if central else None,
            "extras_ids": [int(extra) for extra in registro["extras_ids"].split(";") if extra.strip()],
            "fora": fora,
        })
    df = pd.DataFrame(linhas, columns=COLUNAS_FOLHA)
    # dtype object guarda int e None; o padrão do pandas viraria float com NaN.
    df["tema_central_id"] = pd.Series([linha["tema_central_id"] for linha in linhas], index=df.index, dtype=object)
    df["fora"] = df["fora"].astype(bool)
    return df


def _vazio(valor: Any) -> bool:
    return valor is None or (isinstance(valor, float) and np.isnan(valor))


def alvos_aceitos(linha: Mapping, arvore: "ArvoreTemas") -> set[int]:
    """Temas em que publicar conta como certo: central, extras e pai ou filho direto de cada um.

    Vazio quando a questão é fora da taxonomia. As raízes de disciplina ficam de fora: não são publicáveis.
    """
    if bool(linha["fora"]):
        return set()
    verdadeiros = [linha["tema_central_id"], *linha["extras_ids"]]
    alvos: set[int] = set()
    for tema_id in verdadeiros:
        if _vazio(tema_id):
            continue
        tema_id = int(tema_id)
        alvos.add(tema_id)
        pai = arvore.pai(tema_id)
        if pai is not None:
            alvos.add(pai)
        alvos.update(arvore.filhos(tema_id))
    return {tema_id for tema_id in alvos if not arvore.eh_raiz(tema_id)}


def acerto_regua(tema_id: int | None, linha: Mapping, arvore: "ArvoreTemas") -> bool:
    """Publicar `tema_id` nesta questão da folha está certo pela régua? Falso quando a questão é fora."""
    if _vazio(tema_id) or bool(linha["fora"]):
        return False
    return int(tema_id) in alvos_aceitos(linha, arvore)


# --- fontes de conhecimento em dados/ --------------------------------------------------------------------


def carregar_temas_enriquecidos(caminho: Path = CAMINHOS.temas_enriquecidos) -> dict[int, dict]:
    """Temas enriquecidos por `tema_id`; dicionário vazio enquanto o arquivo não existe."""
    caminho = Path(caminho)
    if not caminho.exists():
        return {}
    enriquecidos: dict[int, dict] = {}
    for linha in ler_jsonl(caminho):
        tema_id = int(linha["tema_id"])
        if tema_id in enriquecidos:
            raise ValueError(f"tema {tema_id} aparece mais de uma vez em {caminho}")
        enriquecidos[tema_id] = {**linha, "tema_id": tema_id}
    return enriquecidos


def carregar_exemplos(caminho: Path = CAMINHOS.exemplos) -> list[dict]:
    """Exemplos sintéticos (`id, tema_id, fora, texto, fonte`); lista vazia enquanto o arquivo não existe.

    `tema_id` é nulo exatamente nos exemplos fora da taxonomia; sem o campo `fora`, ele sai do `tema_id`.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        return []
    exemplos: list[dict] = []
    vistos: set[str] = set()
    for linha in ler_jsonl(caminho):
        id_exemplo = str(linha["id"])
        if id_exemplo in vistos:
            raise ValueError(f"exemplo {id_exemplo} aparece mais de uma vez em {caminho}")
        vistos.add(id_exemplo)
        bruto = linha.get("tema_id")
        tema_id = None if bruto in (None, "") else int(bruto)
        fora = bool(linha.get("fora", tema_id is None))
        if fora != (tema_id is None):
            raise ValueError(f"exemplo {id_exemplo}: fora={fora} com tema_id={tema_id}")
        exemplos.append({"id": id_exemplo, "tema_id": tema_id, "fora": fora, "texto": linha["texto"],
                         "fonte": linha.get("fonte")})
    return exemplos


def carregar_corpus_v1(caminho: Path = CAMINHOS.corpus_v1) -> list[dict]:
    """Corpus da etapa 1 (`id, tema_id, texto`), com o texto montado como o das questões."""
    return [
        {
            "id": str(linha["id"]),
            "tema_id": int(linha["tema_id"]),
            "texto": montar_texto(linha.get("enunciado") or "", linha.get("comando") or "", linha.get("alternativas")),
        }
        for linha in ler_jsonl(caminho)
    ]
