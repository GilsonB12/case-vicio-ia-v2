"""Índice vetorial por cosseno (NumPy), para os temas e para os exemplos rotulados do kNN."""
from __future__ import annotations

from collections.abc import Collection, Hashable, Iterable
from typing import Any

import numpy as np


def _normalizar_linhas(matriz: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    return np.divide(matriz, normas, out=np.zeros_like(matriz), where=normas > 0)


class IndiceVetorial:
    """Linhas normalizadas uma vez; a busca é um produto pela consulta normalizada.

    Cada linha tem um id único e um rótulo (o tema do exemplo, ou None quando ele é fora da taxonomia).
    Linha ou consulta de norma zero tem similaridade zero com tudo.
    """

    def __init__(self, ids: Iterable[Hashable], matriz: Any, rotulos: Iterable[Any]) -> None:
        self.ids = list(ids)
        self.rotulos = list(rotulos)
        matriz = np.asarray(matriz, dtype=np.float32)
        if not self.ids and matriz.size == 0:
            matriz = matriz.reshape(0, matriz.shape[1] if matriz.ndim == 2 else 0)
        if matriz.ndim != 2 or matriz.shape[0] != len(self.ids):
            raise ValueError(f"matriz {matriz.shape} não tem uma linha por id ({len(self.ids)} ids)")
        if len(self.rotulos) != len(self.ids):
            raise ValueError(f"{len(self.rotulos)} rótulos para {len(self.ids)} ids")
        self._posicao = {id_: i for i, id_ in enumerate(self.ids)}
        if len(self._posicao) != len(self.ids):
            raise ValueError("ids repetidos no índice vetorial")
        self._matriz = _normalizar_linhas(matriz)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dimensao(self) -> int:
        return self._matriz.shape[1]

    def posicao(self, id_: Hashable) -> int:
        return self._posicao[id_]

    def similaridades(self, vetor: Any) -> np.ndarray:
        """Cosseno da consulta com cada linha, na ordem de `ids`."""
        if not self.ids:
            return np.zeros(0, dtype=np.float32)
        consulta = np.asarray(vetor, dtype=np.float32).ravel()
        if consulta.shape[0] != self.dimensao:
            raise ValueError(f"consulta de dimensão {consulta.shape[0]}; o índice tem {self.dimensao}")
        norma = float(np.linalg.norm(consulta))
        if norma == 0:
            return np.zeros(len(self.ids), dtype=np.float32)
        return self._matriz @ (consulta / norma)

    def buscar(self, vetor: Any, k: int, excluir: Collection[Hashable] = frozenset()) -> list[tuple[Hashable, Any, float]]:
        """Os `k` mais similares, `(id, rotulo, similaridade)` em ordem decrescente, sem os ids de `excluir`.

        Empate mantém a ordem de inserção.
        """
        if k <= 0 or not self.ids:
            return []
        similaridades = self.similaridades(vetor)
        resultado: list[tuple[Hashable, Any, float]] = []
        for i in np.argsort(-similaridades, kind="stable"):
            id_ = self.ids[i]
            if id_ in excluir:
                continue
            resultado.append((id_, self.rotulos[i], float(similaridades[i])))
            if len(resultado) == k:
                break
        return resultado
