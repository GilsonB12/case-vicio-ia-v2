"""Contexto da rodada: tudo o que o sistema lê de anexos/ e dados/, carregado uma vez, sem rede.

Os embeddings vêm só do cache; texto sem vetor levanta `EmbeddingAusenteError` com a lista do que falta (gere com
`uv run sistema embeddings`). O índice kNN rotulado muda a cada dobra da validação cruzada e sai de
`Contexto.indice_rotulado`, com as questões da folha de treino, os exemplos sintéticos e, se pedido, o corpus.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sistema import dicionario as modulo_dicionario
from sistema.arvore import ArvoreTemas
from sistema.config import CAMINHOS, Caminhos
from sistema.dados import (
    carregar_corpus_v1,
    carregar_exemplos,
    carregar_folha,
    carregar_questoes,
    carregar_temas,
    carregar_temas_enriquecidos,
)
from sistema.embeddings import CacheEmbeddings, EmbeddingAusenteError
from sistema.indice_bm25 import IndiceBM25
from sistema.leis import extrair_chaves
from sistema.vizinhos import IndiceVetorial

registro = logging.getLogger(__name__)

# Peso de cada origem de vizinho rotulado; o do corpus multiplica a similaridade nas características do kNN.
PESO_FOLHA = 1.0
PESO_EXEMPLO = 1.0
PESO_CORPUS = 0.5

ORIGEM_QUESTAO = "questao"
ORIGEM_EXEMPLO = "exemplo"
ORIGEM_CORPUS = "corpus"


@dataclass(eq=False)
class Contexto:
    """Árvore, fontes de conhecimento, questões, folha, exemplos, corpus, índices dos temas e embeddings.

    `temas_ids` é a ordem de `textos_temas`, das linhas de `matriz_temas` e dos índices dos temas (por padrão,
    `arvore.candidatos_possiveis()`). `vetores` tem o embedding de cada questão, exemplo e item do corpus, por id.
    """

    arvore: ArvoreTemas
    enriquecidos: dict[int, dict]
    dicionario: dict
    termos_fora: list[str]
    questoes: pd.DataFrame
    folha: pd.DataFrame
    exemplos: list[dict]
    corpus: list[dict]
    cache: CacheEmbeddings | None
    temas_ids: list[int]
    textos_temas: list[str]
    matriz_temas: np.ndarray = field(repr=False)
    vetores: dict[str, np.ndarray] = field(repr=False)
    indice_bm25: IndiceBM25 = field(init=False, repr=False)
    indice_temas: IndiceVetorial = field(init=False, repr=False)
    publicaveis: frozenset[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.temas_ids = [int(tema_id) for tema_id in self.temas_ids]
        if len(self.textos_temas) != len(self.temas_ids):
            raise ValueError(f"{len(self.textos_temas)} textos para {len(self.temas_ids)} temas")
        self.indice_bm25 = IndiceBM25(self.temas_ids, self.textos_temas)
        self.indice_temas = IndiceVetorial(self.temas_ids, self.matriz_temas, self.temas_ids)
        self.publicaveis = frozenset(t for t in self.temas_ids if self.arvore.publicavel(t))

        self._itens: dict[str, dict] = {}
        fontes = (
            (ORIGEM_QUESTAO, self.questoes.to_dict("records")),
            (ORIGEM_EXEMPLO, self.exemplos),
            (ORIGEM_CORPUS, self.corpus),
        )
        for origem, linhas in fontes:
            for linha in linhas:
                id_item = str(linha["id"])
                if id_item in self._itens:
                    raise ValueError(f"id {id_item} repetido entre questões, exemplos e corpus")
                self._itens[id_item] = {
                    "id": id_item,
                    "texto": linha["texto"],
                    "disciplina_origem": linha.get("disciplina_origem") or "",
                    "origem": origem,
                    "tema_id": linha.get("tema_id"),
                }
        sem_vetor = [id_item for id_item in self._itens if id_item not in self.vetores]
        if sem_vetor:
            raise ValueError(f"{len(sem_vetor)} item(ns) sem vetor no contexto, por exemplo {sem_vetor[:5]}")

        self._folha: dict[str, dict] = {str(linha["id"]): linha for linha in self.folha.to_dict("records")}
        fora_das_questoes = [
            id_q for id_q in self._folha if self._itens.get(id_q, {}).get("origem") != ORIGEM_QUESTAO
        ]
        if fora_das_questoes:
            raise ValueError(f"questões da folha que não estão nas questões: {fora_das_questoes[:5]}")
        self._chaves: dict[str, Counter] = {}

    # --- itens (questões, exemplos, corpus) --------------------------------------------------------------

    @property
    def com_corpus(self) -> bool:
        return bool(self.corpus)

    @property
    def ids_folha(self) -> list[str]:
        return list(self._folha)

    def item(self, id_item: str) -> dict:
        """`id, texto, disciplina_origem` (vazia fora das questões), `origem` e `tema_id` (exemplo e corpus)."""
        try:
            return dict(self._itens[id_item])
        except KeyError as erro:
            raise KeyError(f"{id_item} não é questão, exemplo nem item do corpus deste contexto") from erro

    def texto(self, id_item: str) -> str:
        return self.item(id_item)["texto"]

    def disciplina_origem(self, id_item: str) -> str:
        return self.item(id_item)["disciplina_origem"]

    def vetor(self, id_item: str) -> np.ndarray:
        return self.vetores[id_item]

    def chaves(self, id_item: str) -> Counter:
        """Leis, artigos e súmulas do texto (`sistema.leis.extrair_chaves`), extraídos uma vez."""
        if id_item not in self._chaves:
            self._chaves[id_item] = extrair_chaves(self.texto(id_item))
        return self._chaves[id_item]

    def linha_folha(self, id_questao: str) -> dict:
        return self._folha[id_questao]

    # --- kNN rotulado ------------------------------------------------------------------------------------

    def indice_rotulado(
        self, ids_folha_treino: Iterable[str], com_corpus: bool = False
    ) -> tuple[IndiceVetorial, dict[str, float]]:
        """Índice kNN da rodada e o peso de cada vizinho por id.

        Entram as questões da folha de treino (rótulo = tema central, None quando fora; peso 1), os exemplos
        sintéticos (rótulo = tema, None quando fora; peso 1) e, com `com_corpus`, o corpus da etapa 1 (rótulo =
        tema; peso 0,5). Ordem: folha (na ordem da folha), exemplos, corpus.
        """
        treino = set(ids_folha_treino)
        desconhecidos = sorted(treino - set(self._folha))
        if desconhecidos:
            raise ValueError(f"ids fora da folha-verdade no treino: {desconhecidos[:5]}")
        if com_corpus and not self.corpus:
            raise ValueError("índice com corpus pedido, mas o contexto foi carregado sem o corpus "
                             "(use carregar_contexto(com_corpus=True))")
        ids: list[str] = []
        rotulos: list[int | None] = []
        pesos: dict[str, float] = {}
        for id_questao, linha in self._folha.items():
            if id_questao in treino:
                ids.append(id_questao)
                rotulos.append(None if linha["fora"] else int(linha["tema_central_id"]))
                pesos[id_questao] = PESO_FOLHA
        for exemplo in self.exemplos:
            ids.append(str(exemplo["id"]))
            rotulos.append(None if exemplo["tema_id"] is None else int(exemplo["tema_id"]))
            pesos[str(exemplo["id"])] = PESO_EXEMPLO
        if com_corpus:
            for item in self.corpus:
                ids.append(str(item["id"]))
                rotulos.append(int(item["tema_id"]))
                pesos[str(item["id"])] = PESO_CORPUS
        if ids:
            matriz = np.vstack([self.vetores[id_item] for id_item in ids])
        else:
            matriz = np.zeros((0, self.indice_temas.dimensao), dtype=np.float32)
        return IndiceVetorial(ids, matriz, rotulos), pesos


# --- carga -----------------------------------------------------------------------------------------------


def _ler_termos_fora(caminho: Path) -> list[str]:
    """Lista de `dados/termos_fora.json` (`{"termos": [...]}` ou a lista direto); vazia sem o arquivo."""
    caminho = Path(caminho)
    if not caminho.exists():
        return []
    bruto = json.loads(caminho.read_text(encoding="utf-8"))
    termos = bruto.get("termos", []) if isinstance(bruto, dict) else bruto
    return [str(termo) for termo in termos]


def _carregar_dicionario(caminhos: Caminhos, enriquecidos: dict[int, dict]) -> dict:
    """`dados/dicionario_leis.json`; sem ele, o mesmo dicionário montado em memória dos enriquecidos e ajustes."""
    if caminhos.dicionario_leis.exists():
        return modulo_dicionario.carregar(caminhos.dicionario_leis)
    registro.warning("%s não existe; dicionário montado em memória (gere o arquivo com `uv run sistema "
                     "dicionario`)", caminhos.dicionario_leis)
    return modulo_dicionario.construir(enriquecidos, modulo_dicionario.ler_ajustes(caminhos.dicionario_ajustes))


def carregar_contexto(com_corpus: bool = False, caminhos: Caminhos | None = None) -> Contexto:
    """Lê anexos, fontes de conhecimento e embeddings (só do cache) uma vez.

    `com_corpus` carrega também o corpus da etapa 1 (e exige os embeddings dele). Falta de embedding levanta
    `EmbeddingAusenteError` com todos os textos que faltam, antes de montar qualquer índice.
    """
    caminhos = caminhos or CAMINHOS
    arvore = ArvoreTemas(carregar_temas(caminhos.temas))
    enriquecidos = carregar_temas_enriquecidos(caminhos.temas_enriquecidos)
    questoes = carregar_questoes(caminhos.questoes)
    folha = carregar_folha(caminhos.folha)
    exemplos = carregar_exemplos(caminhos.exemplos)
    corpus = carregar_corpus_v1(caminhos.corpus_v1) if com_corpus else []
    cache = CacheEmbeddings(pasta=caminhos.embeddings)

    temas_ids = arvore.candidatos_possiveis()
    textos_temas = [arvore.texto_tema(tema_id, enriquecidos) for tema_id in temas_ids]
    textos_por_id = {str(i): t for i, t in zip(questoes["id"], questoes["texto"])}
    textos_por_id.update({str(e["id"]): e["texto"] for e in exemplos})
    textos_por_id.update({str(c["id"]): c["texto"] for c in corpus})

    faltando = cache.faltando([*textos_temas, *textos_por_id.values()])
    if faltando:
        raise EmbeddingAusenteError(faltando)
    matriz_itens = cache.matriz(textos_por_id.values())
    vetores = {id_item: matriz_itens[i] for i, id_item in enumerate(textos_por_id)}

    return Contexto(
        arvore=arvore,
        enriquecidos=enriquecidos,
        dicionario=_carregar_dicionario(caminhos, enriquecidos),
        termos_fora=_ler_termos_fora(caminhos.termos_fora),
        questoes=questoes,
        folha=folha,
        exemplos=exemplos,
        corpus=corpus,
        cache=cache,
        temas_ids=temas_ids,
        textos_temas=textos_temas,
        matriz_temas=cache.matriz(textos_temas),
        vetores=vetores,
    )
