"""Árvore de temas: navegação, relação duplicado → original, texto do tema e régua de acerto."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any, Mapping

import pandas as pd

from sistema.dados import carregar_temas

# Duplicados cujo título não casa com o do original depois de normalizado.
EXCECOES_ORIGINAIS: dict[int, int] = {
    6735: 5543,  # "Provas em Espécie" -> "Provas em Espécies"
    6736: 5560,  # "Recursos (Parte 03) - Espécies" -> "Recursos em Espécies"
    6722: 4696,  # "Estatuto do Idoso - Crimes" -> "Estaturo de Idoso - Crimes"
}

_PARTE = re.compile(r"\(\s*parte\s*\d+\s*\)")
_CONTAGEM = re.compile(r"\s*\d+\s+quest(?:ões|oes)(?: no source)?\.?", re.IGNORECASE)
_PATH = re.compile(r"\s*Path:.*$", re.DOTALL)
_PREFIXO_ARTIGO = re.compile(r"^\s*art(?:igo)?s?\.?\s*", re.IGNORECASE)


def normalizar_titulo(titulo: str) -> str:
    """Minúsculo, sem acento, sem sufixo '(Parte N)', espaços colapsados."""
    sem_acento = unicodedata.normalize("NFKD", titulo).encode("ascii", "ignore").decode()
    sem_parte = _PARTE.sub("", sem_acento.lower())
    return re.sub(r"\s+", " ", sem_parte).strip()


def _texto(item: Any) -> str:
    return str(item or "").strip()


def _formatar_lei(item: Any) -> str:
    """`{"numero": "11.340/2006", "nome": "Lei Maria da Penha"}` vira "Lei Maria da Penha (11.340/2006)"."""
    if not isinstance(item, Mapping):
        return _texto(item)
    numero, nome = _texto(item.get("numero")), _texto(item.get("nome"))
    return f"{nome} ({numero})" if nome and numero else nome or numero


def _formatar_artigo(item: Any) -> str:
    """`{"diploma": "CP", "artigo": "157"}` vira "CP, art. 157"."""
    if not isinstance(item, Mapping):
        return _texto(item)
    diploma = _texto(item.get("diploma"))
    artigo = _PREFIXO_ARTIGO.sub("", _texto(item.get("artigo")))
    if not artigo:
        return diploma
    return f"{diploma}, art. {artigo}" if diploma else f"art. {artigo}"


# Partes do tema enriquecido que entram no texto do tema, na ordem, depois do caminho e do resumo.
_PARTES_ENRIQUECIDAS = (
    ("Leis", "leis", _formatar_lei),
    ("Artigos", "artigos", _formatar_artigo),
    ("Termos", "termos", _texto),
    ("Sinônimos", "sinonimos", _texto),
)


class ArvoreTemas:
    def __init__(self, temas: pd.DataFrame) -> None:
        self._df = temas.set_index("tema_id", drop=False)
        # Um registro por tema em dicionário: a validação cruzada consulta nível e pai centenas de milhares de
        # vezes, e `DataFrame.loc` por linha custa dezenas de microssegundos.
        self._registros: dict[int, dict[str, Any]] = {
            int(registro["tema_id"]): {
                **registro,
                "parent_id": None if pd.isna(registro["parent_id"]) else int(registro["parent_id"]),
                "nivel": int(registro["nivel"]),
                "ativo": bool(registro["ativo"]),
            }
            for registro in self._df.to_dict("records")
        }
        self._filhos: dict[int, list[int]] = defaultdict(list)
        for tema_id, pai in zip(self._df["tema_id"], self._df["parent_id"]):
            if pd.notna(pai):
                self._filhos[int(pai)].append(int(tema_id))
        self._originais = self._montar_originais()

    @classmethod
    def carregar(cls) -> "ArvoreTemas":
        return cls(carregar_temas())

    def _linha(self, tema_id: int) -> dict[str, Any]:
        try:
            return self._registros[int(tema_id)]
        except KeyError as erro:
            raise KeyError(f"tema {tema_id} não existe") from erro

    def _montar_originais(self) -> dict[int, int]:
        """Temas sem descrição são duplicados; casam com o original pelo título normalizado ou pela tabela de exceções."""
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
            if tema_id in EXCECOES_ORIGINAIS:
                mapa[tema_id] = EXCECOES_ORIGINAIS[tema_id]
                continue
            candidatos = por_titulo.get(normalizar_titulo(titulo), [])
            if len(candidatos) != 1:
                raise ValueError(f"tema duplicado {tema_id} '{titulo}' sem original único: {candidatos}")
            mapa[tema_id] = candidatos[0]
        return mapa

    # --- atributos ---------------------------------------------------------------------------------------

    def existe(self, tema_id: int) -> bool:
        return int(tema_id) in self._registros

    def titulo(self, tema_id: int) -> str:
        return str(self._linha(tema_id)["titulo"])

    def caminho(self, tema_id: int) -> str:
        return str(self._linha(tema_id)["caminho"])

    def descricao(self, tema_id: int) -> str:
        return str(self._linha(tema_id)["descricao"])

    def descricao_limpa(self, tema_id: int) -> str:
        """Descrição original sem a frase de contagem de questões e sem o 'Path:' que repete o caminho."""
        return _PATH.sub("", _CONTAGEM.sub("", self.descricao(tema_id))).strip()

    def nivel(self, tema_id: int) -> int:
        return int(self._linha(tema_id)["nivel"])

    def disciplina(self, tema_id: int) -> str:
        return str(self._linha(tema_id)["disciplina"])

    def ativo(self, tema_id: int) -> bool:
        return bool(self._linha(tema_id)["ativo"])

    # --- navegação ---------------------------------------------------------------------------------------

    def pai(self, tema_id: int) -> int | None:
        return self._linha(tema_id)["parent_id"]

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

    # --- duplicados e publicáveis ------------------------------------------------------------------------

    def eh_duplicado(self, tema_id: int) -> bool:
        return int(tema_id) in self._originais

    def original_de(self, tema_id: int) -> int:
        """O original de um duplicado (a quem ele copia o conteúdo enriquecido), ou o próprio tema."""
        return self._originais.get(int(tema_id), int(tema_id))

    def duplicados(self) -> dict[int, int]:
        """Duplicado → original."""
        return dict(self._originais)

    def publicavel(self, tema_id: int) -> bool:
        """Ativo e fora das raízes de disciplina; os duplicados são publicáveis (a equipe os usa na folha)."""
        return self.existe(tema_id) and self.ativo(tema_id) and not self.eh_raiz(tema_id)

    def candidatos_possiveis(self) -> list[int]:
        return sorted(int(t) for t in self._df["tema_id"] if self.publicavel(int(t)))

    # --- texto do tema -----------------------------------------------------------------------------------

    def texto_tema(self, tema_id: int, enriquecidos: Mapping[int, dict]) -> str:
        """Texto do tema para BM25 e embedding.

        Com tema enriquecido: caminho, resumo, leis, artigos, termos e sinônimos. Sem ele: caminho e descrição
        original sem a frase de contagem. O duplicado usa o conteúdo do original (o enriquecido do original, ou
        o próprio se só ele existir) com o seu próprio caminho.
        """
        tema_id = int(tema_id)
        original = self.original_de(tema_id)
        caminho = self.caminho(tema_id)
        enriquecido = enriquecidos.get(original) or enriquecidos.get(tema_id)
        if not enriquecido:
            descricao = self.descricao_limpa(original)
            return f"{caminho}\n{descricao}" if descricao else caminho
        partes = [caminho]
        resumo = _texto(enriquecido.get("resumo"))
        if resumo:
            partes.append(resumo)
        for rotulo, chave, formatar in _PARTES_ENRIQUECIDAS:
            itens = [formatar(item) for item in enriquecido.get(chave) or []]
            itens = [item for item in itens if item]
            if itens:
                partes.append(f"{rotulo}: {'; '.join(itens)}")
        return "\n".join(partes)

    # --- régua -------------------------------------------------------------------------------------------

    def acerto(self, decidido: int, verdadeiro: int) -> bool:
        """Régua da Vício: igual, pai ou filho do tema verdadeiro."""
        decidido, verdadeiro = int(decidido), int(verdadeiro)
        if decidido == verdadeiro:
            return True
        return self.pai(decidido) == verdadeiro or self.pai(verdadeiro) == decidido
