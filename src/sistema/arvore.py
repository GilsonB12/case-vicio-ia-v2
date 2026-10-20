"""Árvore de temas: navegação, canonicalização de duplicatas e régua de acerto."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import pandas as pd

from classificador.dados import carregar_temas

EXCECOES_CANONICAS: dict[int, int] = {
    6735: 5543,  # "Provas em Espécie" -> "Provas em Espécies"
    6736: 5560,  # "Recursos (Parte 03) - Espécies" -> "Recursos em Espécies"
    6722: 4696,  # "Estatuto do Idoso - Crimes" -> "Estaturo de Idoso - Crimes"
}

_PARTE = re.compile(r"\(\s*parte\s*\d+\s*\)")
_CONTAGEM = re.compile(r"\s*\d+\s+quest(?:ões|oes)(?: no source)?\.?", re.IGNORECASE)
_PATH = re.compile(r"\s*Path:.*$", re.DOTALL)


def normalizar_titulo(titulo: str) -> str:
    """Minúsculo, sem acento, sem sufixo '(Parte N)', espaços colapsados."""
    sem_acento = unicodedata.normalize("NFKD", titulo).encode("ascii", "ignore").decode()
    sem_parte = _PARTE.sub("", sem_acento.lower())
    return re.sub(r"\s+", " ", sem_parte).strip()


class ArvoreTemas:
    def __init__(self, temas: pd.DataFrame) -> None:
        self._df = temas.set_index("tema_id", drop=False)
        self._filhos: dict[int, list[int]] = defaultdict(list)
        for tema_id, pai in zip(self._df["tema_id"], self._df["parent_id"]):
            if pd.notna(pai):
                self._filhos[int(pai)].append(int(tema_id))
        self._canonicos = self._montar_canonicos()

    @classmethod
    def carregar(cls) -> "ArvoreTemas":
        return cls(carregar_temas())

    def _linha(self, tema_id: int) -> pd.Series:
        try:
            return self._df.loc[int(tema_id)]
        except KeyError as erro:
            raise KeyError(f"tema {tema_id} não existe") from erro

    def _montar_canonicos(self) -> dict[int, int]:
        """Temas sem descrição são duplicatas; casam por título normalizado ou pela tabela de exceções."""
        por_titulo: dict[str, list[int]] = defaultdict(list)
        for tema_id, titulo, descricao in zip(self._df["tema_id"], self._df["titulo"], self._df["descricao"]):
            if descricao.strip():
                por_titulo[normalizar_titulo(titulo)].append(int(tema_id))
        mapa: dict[int, int] = {}
        for tema_id, titulo, descricao, nivel in zip(
            self._df["tema_id"], self._df["titulo"], self._df["descricao"], self._df["nivel"]
        ):
            tema_id = int(tema_id)
            if descricao.strip() or nivel == 0:
                continue
            if tema_id in EXCECOES_CANONICAS:
                mapa[tema_id] = EXCECOES_CANONICAS[tema_id]
                continue
            candidatos = por_titulo.get(normalizar_titulo(titulo), [])
            if len(candidatos) != 1:
                raise ValueError(f"tema duplicado {tema_id} '{titulo}' sem canônico único: {candidatos}")
            mapa[tema_id] = candidatos[0]
        return mapa

    def existe(self, tema_id: int) -> bool:
        return int(tema_id) in self._df.index

    def titulo(self, tema_id: int) -> str:
        return str(self._linha(tema_id)["titulo"])

    def caminho(self, tema_id: int) -> str:
        return str(self._linha(tema_id)["caminho"])

    def descricao(self, tema_id: int) -> str:
        return str(self._linha(tema_id)["descricao"])

    def nivel(self, tema_id: int) -> int:
        return int(self._linha(tema_id)["nivel"])

    def disciplina(self, tema_id: int) -> str:
        return str(self._linha(tema_id)["disciplina"])

    def ativo(self, tema_id: int) -> bool:
        return bool(self._linha(tema_id)["ativo"])

    def pai(self, tema_id: int) -> int | None:
        valor = self._linha(tema_id)["parent_id"]
        return None if pd.isna(valor) else int(valor)

    def filhos(self, tema_id: int) -> list[int]:
        return list(self._filhos.get(int(tema_id), []))

    def irmaos(self, tema_id: int) -> list[int]:
        pai = self.pai(tema_id)
        if pai is None:
            return []
        return [f for f in self.filhos(pai) if f != int(tema_id)]

    def eh_raiz(self, tema_id: int) -> bool:
        return self.nivel(tema_id) == 0

    def ancestral_nivel1(self, tema_id: int) -> int:
        nivel = self.nivel(tema_id)
        if nivel == 0:
            raise ValueError(f"tema {tema_id} é raiz de disciplina e não tem ancestral de nível 1")
        if nivel == 1:
            return int(tema_id)
        pai = self.pai(tema_id)
        assert pai is not None
        return pai

    def mesmo_ramo(self, a: int, b: int) -> bool:
        if self.eh_raiz(a) or self.eh_raiz(b):
            return False
        return self.ancestral_nivel1(a) == self.ancestral_nivel1(b)

    def eh_duplicado(self, tema_id: int) -> bool:
        return int(tema_id) in self._canonicos

    def canonico(self, tema_id: int) -> int:
        return self._canonicos.get(int(tema_id), int(tema_id))

    def duplicados(self) -> dict[int, int]:
        return dict(self._canonicos)

    def publicavel(self, tema_id: int) -> bool:
        return self.existe(tema_id) and self.ativo(tema_id) and not self.eh_raiz(tema_id) and not self.eh_duplicado(tema_id)

    def candidatos_possiveis(self) -> list[int]:
        return sorted(int(t) for t in self._df["tema_id"] if self.publicavel(int(t)))

    def documento(self, tema_id: int) -> str:
        descricao = _PATH.sub("", _CONTAGEM.sub("", self.descricao(tema_id))).strip()
        return f"{self.caminho(tema_id)}\n{descricao}" if descricao else self.caminho(tema_id)

    def acerto(self, decidido: int, verdadeiro: int) -> bool:
        """Régua da Vício: igual, pai ou filho do tema verdadeiro."""
        decidido, verdadeiro = int(decidido), int(verdadeiro)
        if decidido == verdadeiro:
            return True
        return self.pai(decidido) == verdadeiro or self.pai(verdadeiro) == decidido
