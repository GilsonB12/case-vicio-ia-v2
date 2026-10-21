"""Índice lexical BM25 (Okapi, `rank-bm25`) sobre os textos dos temas.

Tokens: minúsculas, sem acento, só letras e dígitos, sem stopwords do português e sem tokens de um caractere.
Os rótulos de estrutura que o próprio sistema põe nos textos ("Comando:", "Alternativas:" nas questões;
"Leis:", "Artigos:", "Termos:", "Sinônimos:" nos temas) saem antes: senão "Alternativas:", presente em toda
questão, daria pontos ao tema de penas alternativas em todas elas.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Hashable, Iterable

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS: frozenset[str] = frozenset("""
    a o as os um uma uns umas
    de do da dos das em no na nos nas num numa por pelo pela pelos pelas para pra pro com sem sob ao aos
    e ou nem mas porem que se como qual quais quando onde cujo cuja cujos cujas quem
    nao sim ja ainda tambem so apenas muito mais menos bem entre ate sobre apos ante perante desde contra
    este esta estes estas esse essa esses essas isto isso aquele aquela aqueles aquelas aquilo
    ele ela eles elas lhe lhes seu sua seus suas nosso nossa meu minha voce me te vos
    ser sao foi era sera sido sendo estar estao estava ha havia haver tem ter tinha teve pode podem
    deve devem caso todo toda todos todas outro outra outros outras mesmo mesma qualquer cada tal tais
""".split())

_ROTULOS = re.compile(r"(?m)^[ \t]*(?:Comando|Alternativas|Leis|Artigos|Termos|Sinônimos):")
_ORDINAIS = str.maketrans({"º": " ", "ª": " ", "°": " "})
_PALAVRA = re.compile(r"[a-z0-9]+")


def palavras(texto: str | None) -> list[str]:
    """Palavras normalizadas de um texto, na ordem, com repetição e com as stopwords: minúsculas, sem acento, só
    letras e dígitos (qualquer outro caractere, hífen inclusive, separa palavras), sem os rótulos de estrutura."""
    if not texto:
        return []
    sem_rotulos = _ROTULOS.sub(" ", texto).translate(_ORDINAIS)
    sem_acento = unicodedata.normalize("NFKD", sem_rotulos.lower()).encode("ascii", "ignore").decode()
    return _PALAVRA.findall(sem_acento)


def tokenizar(texto: str | None) -> list[str]:
    """Tokens normalizados de um texto, na ordem, com repetição: as `palavras` sem stopwords e sem as de um
    caractere."""
    return [token for token in palavras(texto) if len(token) > 1 and token not in STOPWORDS]


class IndiceBM25:
    """BM25 Okapi sobre `textos`, cada um com o seu id.

    `pontuacoes` reproduz `BM25Okapi.get_scores` (soma por ocorrência do termo na consulta) com a contribuição
    de cada termo calculada uma vez e guardada: as mesmas pontuações, sem percorrer os documentos a cada consulta.
    """

    def __init__(self, ids: Iterable[Hashable], textos: Iterable[str]) -> None:
        self.ids = list(ids)
        textos = list(textos)
        if len(self.ids) != len(textos):
            raise ValueError(f"{len(self.ids)} ids para {len(textos)} textos")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("ids repetidos no índice BM25")
        tokenizados = [tokenizar(texto) for texto in textos]
        if not any(tokenizados):
            raise ValueError("nenhum texto do índice BM25 tem token depois da normalização")
        self.bm25 = BM25Okapi(tokenizados)
        tamanhos = np.asarray(self.bm25.doc_len, dtype=float)
        self._saturacao = self.bm25.k1 * (1 - self.bm25.b + self.bm25.b * tamanhos / self.bm25.avgdl)
        self._por_termo: dict[str, np.ndarray | None] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def _contribuicao(self, termo: str) -> np.ndarray | None:
        if termo not in self._por_termo:
            idf = self.bm25.idf.get(termo) or 0
            if not idf:
                self._por_termo[termo] = None
            else:
                frequencias = np.array([doc.get(termo) or 0 for doc in self.bm25.doc_freqs], dtype=float)
                self._por_termo[termo] = idf * frequencias * (self.bm25.k1 + 1) / (frequencias + self._saturacao)
        return self._por_termo[termo]

    def pontuacoes(self, texto: str) -> np.ndarray:
        """Pontuação BM25 bruta de cada documento para a consulta, na ordem de `ids`."""
        total = np.zeros(len(self.ids))
        for termo, vezes in Counter(tokenizar(texto)).items():
            contribuicao = self._contribuicao(termo)
            if contribuicao is not None:
                total += vezes * contribuicao
        return total

    def buscar(self, texto: str, k: int) -> list[tuple[Hashable, float]]:
        """Os `k` documentos de maior pontuação (bruta), em ordem decrescente; pontuação zero fica de fora."""
        if k <= 0:
            return []
        pontos = self.pontuacoes(texto)
        ordem = np.argsort(-pontos, kind="stable")[:k]
        return [(self.ids[i], float(pontos[i])) for i in ordem if pontos[i] > 0]

