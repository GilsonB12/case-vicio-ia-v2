"""Dicionário de termos técnicos por tema, casado como frase inteira no texto da questão.

Cada tema publicável tem as expressões de `termos` e `sinonimos` do seu tema enriquecido; o duplicado herda as do
original, como em `ArvoreTemas.texto_tema`. Expressões e texto são normalizados do mesmo jeito: as palavras de
`indice_bm25.palavras` (minúsculas, sem acento, só letras e dígitos, com as stopwords) separadas por um espaço.
O hífen separa palavras, então "co-autoria" casa "co autoria"; a expressão com hífen também vale sem ele
("coautoria"), e o texto é lido das duas formas, então "co-autoria" na questão casa a expressão "coautoria". As
grafias com e sem hífen da mesma expressão contam como uma só.

O casamento é de frase inteira por fronteira de palavra: a questão vira a sua sequência de palavras e cada
expressão é procurada como subsequência contígua ("tipo" não casa "tipologia"). Expressão de uma palavra com menos
de `MIN_LETRAS` letras fica de fora (casa demais).

Peso de cada expressão: idf = log(1 + N / n), N = temas do índice, n = temas que têm a expressão; × 1,5 quando
ela tem duas ou mais palavras.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from sistema.indice_bm25 import palavras

if TYPE_CHECKING:
    from sistema.arvore import ArvoreTemas

CAMPOS = ("termos", "sinonimos")
MIN_LETRAS = 5
FATOR_MULTIPALAVRA = 1.5

_HIFEN = re.compile("[-‐‑]")  # hífen comum, hífen e hífen inseparável

Pontos = tuple[float, int, float]  # (soma dos pesos, expressões casadas, maior peso casado)


def normalizar(texto: str | None) -> str:
    """Palavras normalizadas (`indice_bm25.palavras`) separadas por um espaço; o hífen vira espaço."""
    return " ".join(palavras(texto))


def _sem_hifen(texto: str) -> str:
    return _HIFEN.sub("", texto)


def chave(expressao: str | None) -> str:
    """Forma canônica da expressão: normalizada com o hífen tirado (as grafias "co-autoria" e "coautoria" são a
    mesma expressão)."""
    return normalizar(_sem_hifen(expressao or ""))


def formas(expressao: str | None) -> tuple[str, ...]:
    """Formas normalizadas em que a expressão casa: com o hífen como espaço e, se ela tem hífen, sem ele."""
    com_espaco = normalizar(expressao)
    if not com_espaco:
        return ()
    return tuple(dict.fromkeys((com_espaco, chave(expressao))))


def _aceita(canonica: str) -> bool:
    return bool(canonica) and (" " in canonica or len(canonica) >= MIN_LETRAS)


class IndiceTermos:
    """Expressões técnicas de cada tema publicável, com peso idf, e o casamento delas num texto.

    `temas_ids`: os temas do índice (os publicáveis entre eles); sem ele, `arvore.candidatos_possiveis()`.
    """

    def __init__(self, enriquecidos: Mapping[int, dict], arvore: "ArvoreTemas",
                 temas_ids: Iterable[int] | None = None) -> None:
        if temas_ids is None:
            ids = arvore.candidatos_possiveis()
        else:
            ids = sorted({int(t) for t in temas_ids if arvore.publicavel(int(t))})
        self.temas_ids: list[int] = list(ids)

        por_tema: dict[int, set[str]] = {}
        formas_da_chave: dict[str, set[str]] = {}
        for tema_id in self.temas_ids:
            enriquecido = enriquecidos.get(arvore.original_de(tema_id)) or enriquecidos.get(tema_id) or {}
            canonicas: set[str] = set()
            for campo in CAMPOS:
                for expressao in enriquecido.get(campo) or []:
                    canonica = chave(str(expressao))
                    if not _aceita(canonica):
                        continue
                    canonicas.add(canonica)
                    formas_da_chave.setdefault(canonica, set()).update(formas(str(expressao)))
            if canonicas:
                por_tema[tema_id] = canonicas

        contagem = Counter(c for canonicas in por_tema.values() for c in canonicas)
        n_temas = len(self.temas_ids)
        self.expressoes: list[str] = sorted(contagem)
        posicao = {c: i for i, c in enumerate(self.expressoes)}
        self.pesos: list[float] = [
            math.log(1 + n_temas / contagem[c]) * (FATOR_MULTIPALAVRA if " " in c else 1.0) for c in self.expressoes
        ]
        self._temas: list[list[int]] = [[] for _ in self.expressoes]
        for tema_id, canonicas in por_tema.items():
            for c in canonicas:
                self._temas[posicao[c]].append(tema_id)
        self._por_tema: dict[int, list[str]] = {t: sorted(c) for t, c in por_tema.items()}

        # forma → expressões; primeira palavra → tamanhos (em palavras) das formas que começam com ela
        self._forma: dict[str, tuple[int, ...]] = {}
        tamanhos: dict[str, set[int]] = {}
        for canonica, formas_da_expressao in formas_da_chave.items():
            for forma in formas_da_expressao:
                self._forma[forma] = (*self._forma.get(forma, ()), posicao[canonica])
                partes = forma.split(" ")
                tamanhos.setdefault(partes[0], set()).add(len(partes))
        self._tamanhos: dict[str, tuple[int, ...]] = {p: tuple(sorted(t)) for p, t in tamanhos.items()}

    def __len__(self) -> int:
        return len(self.expressoes)

    @property
    def peso_maximo(self) -> float:
        """Maior peso de expressão do índice (0 sem expressão)."""
        return max(self.pesos, default=0.0)

    def peso(self, expressao: str) -> float:
        """Peso da expressão (em qualquer grafia); 0 se ela não está no índice."""
        canonica = chave(expressao)
        achadas = self._forma.get(canonica) or self._forma.get(normalizar(expressao)) or ()
        return max((self.pesos[i] for i in achadas), default=0.0)

    def expressoes_do_tema(self, tema_id: int) -> list[str]:
        """Expressões canônicas do tema, em ordem alfabética."""
        return list(self._por_tema.get(int(tema_id), []))

    def _casadas(self, texto: str) -> set[int]:
        leituras = [palavras(texto)]
        if _HIFEN.search(texto):
            leituras.append(palavras(_sem_hifen(texto)))
        achadas: set[int] = set()
        for sequencia in leituras:
            total = len(sequencia)
            for inicio, primeira in enumerate(sequencia):
                tamanhos = self._tamanhos.get(primeira)
                if tamanhos is None:
                    continue
                for tamanho in tamanhos:
                    if inicio + tamanho > total:
                        break
                    indices = self._forma.get(" ".join(sequencia[inicio:inicio + tamanho]))
                    if indices:
                        achadas.update(indices)
        return achadas

    def pontuar(self, texto: str | None) -> dict[int, Pontos]:
        """`{tema_id: (soma dos pesos, expressões casadas, maior peso casado)}` dos temas com alguma expressão no
        texto; cada expressão conta uma vez, por mais que se repita."""
        if not texto or not self.expressoes:
            return {}
        acumulado: dict[int, list] = {}
        for indice in sorted(self._casadas(texto)):
            peso = self.pesos[indice]
            for tema_id in self._temas[indice]:
                atual = acumulado.get(tema_id)
                if atual is None:
                    acumulado[tema_id] = [peso, 1, peso]
                else:
                    atual[0] += peso
                    atual[1] += 1
                    atual[2] = max(atual[2], peso)
        return {tema_id: (float(s), int(n), float(m)) for tema_id, (s, n, m) in acumulado.items()}
