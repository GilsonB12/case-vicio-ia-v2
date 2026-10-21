"""Candidatos por questão: união de quatro buscas, mais pai e filhos, com teto.

1. Dicionário: temas apontados pelas leis, artigos e súmulas da questão (`sistema.dicionario.pontuar`).
2. BM25 do texto da questão contra os textos dos temas.
3. Cosseno do embedding da questão com o embedding de cada tema.
4. kNN rotulado: vizinhos entre os exemplos rotulados da rodada; cada um traz o seu tema (rótulo None é ignorado).

Depois entram o pai (nível 1) de todo candidato de nível 2 e os filhos de todo tema de nível 1 que veio pelo
dicionário. Só temas publicáveis do contexto. Cada busca tem a pontuação normalizada em 0 a 1 dentro da questão
(dividida pela maior); o pai herda a maior pontuação dos filhos que o trouxeram, e o filho do pai, a do pai no
dicionário. Ordem: número de origens, maior pontuação, tema_id; corte no teto.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from sistema.dicionario import pontuar
from sistema.vizinhos import IndiceVetorial

if TYPE_CHECKING:
    from sistema.contexto import Contexto

ORIGEM_DICIONARIO = "dicionario"
ORIGEM_BM25 = "bm25"
ORIGEM_EMBEDDING = "embedding"
ORIGEM_KNN = "knn"
ORIGEM_PAI = "pai"
ORIGEM_FILHO_DO_PAI = "filho_do_pai"
ORIGENS_BUSCA = (ORIGEM_DICIONARIO, ORIGEM_BM25, ORIGEM_EMBEDDING, ORIGEM_KNN)

K_BM25 = 10
K_EMB = 10
K_KNN = 15
TETO = 30

Vizinho = tuple[Hashable, Any, float]  # (id, rótulo, similaridade)


@dataclass
class Candidato:
    tema_id: int
    origens: set[str]
    pontuacao: float
    por_origem: dict[str, float] = field(default_factory=dict)

    @property
    def n_buscas(self) -> int:
        """Quantas das quatro buscas trouxeram o tema (pai e filho do pai não contam)."""
        return sum(origem in self.origens for origem in ORIGENS_BUSCA)


def _normalizar(pontos: Mapping[int, float]) -> dict[int, float]:
    if not pontos:
        return {}
    maior = max(pontos.values())
    if maior <= 0:
        return {tema_id: 0.0 for tema_id in pontos}
    return {tema_id: min(1.0, max(0.0, valor / maior)) for tema_id, valor in pontos.items()}


def _melhores(ids: list[int], valores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Os `k` maiores valores, em ordem decrescente (empate mantém a ordem de `ids`), como nas buscas dos índices."""
    if k <= 0 or len(ids) == 0:
        return []
    return [(ids[i], float(valores[i])) for i in np.argsort(-np.asarray(valores), kind="stable")[:k]]


def buscar_vizinhos(id_item: Hashable, vetor: Any, indice_rotulado: IndiceVetorial, k: int = K_KNN) -> list[Vizinho]:
    """Os `k` vizinhos rotulados mais próximos, sem o próprio item (que pode estar no índice)."""
    return indice_rotulado.buscar(vetor, k, excluir=frozenset({id_item}))


def gerar(
    questao: Mapping[str, Any],
    chaves: Counter,
    vetor_questao: Any,
    contexto: "Contexto",
    indice_rotulado: IndiceVetorial,
    k_bm25: int = K_BM25,
    k_emb: int = K_EMB,
    k_knn: int = K_KNN,
    teto: int = TETO,
    *,
    peso_por_id: Mapping[Hashable, float] | None = None,
    vizinhos: list[Vizinho] | None = None,
    pontos_bm25: np.ndarray | None = None,
    cossenos: np.ndarray | None = None,
) -> list[Candidato]:
    """Candidatos da questão (`id` e `texto`), em ordem, no máximo `teto`.

    `vizinhos` já buscados (por `buscar_vizinhos`) evitam uma segunda busca; deles entram os `k_knn` primeiros.
    `peso_por_id` multiplica a similaridade de cada vizinho (o corpus pesa 0,5). `pontos_bm25` e `cossenos`
    (BM25 bruto e cosseno com cada tema, na ordem de `contexto.temas_ids`) reaproveitam as partes que não mudam
    entre as dobras (`Contexto.pontuacoes_bm25` e `Contexto.cossenos_temas`); sem eles, as buscas são feitas aqui.
    """
    arvore = contexto.arvore
    publicaveis = contexto.publicaveis
    pesos = peso_por_id or {}

    pontos_dicionario, _ = pontuar(chaves, contexto.dicionario)
    if pontos_bm25 is None:
        achados_bm25 = contexto.indice_bm25.buscar(questao["texto"], k_bm25)
    else:
        achados_bm25 = [(t, p) for t, p in _melhores(contexto.temas_ids, pontos_bm25, k_bm25) if p > 0]
    if cossenos is None:
        achados_emb = [(t, s) for t, _, s in contexto.indice_temas.buscar(vetor_questao, k_emb)]
    else:
        achados_emb = _melhores(contexto.temas_ids, cossenos, k_emb)
    brutos: dict[str, dict[int, float]] = {
        ORIGEM_DICIONARIO: {t: p for t, p in pontos_dicionario.items() if p > 0 and t in publicaveis},
        ORIGEM_BM25: {t: p for t, p in achados_bm25 if t in publicaveis},
        ORIGEM_EMBEDDING: {t: s for t, s in achados_emb if s > 0 and t in publicaveis},
    }
    if vizinhos is None:
        vizinhos = buscar_vizinhos(questao["id"], vetor_questao, indice_rotulado, k_knn)
    knn: dict[int, float] = {}
    for id_vizinho, rotulo, similaridade in vizinhos[:k_knn]:
        if rotulo is None or int(rotulo) not in publicaveis:
            continue
        valor = similaridade * pesos.get(id_vizinho, 1.0)
        if valor > 0:
            knn[int(rotulo)] = max(knn.get(int(rotulo), 0.0), valor)
    brutos[ORIGEM_KNN] = knn

    por_tema: dict[int, dict[str, float]] = {}
    normalizados = {origem: _normalizar(pontos) for origem, pontos in brutos.items()}
    for origem, pontos in normalizados.items():
        for tema_id, valor in pontos.items():
            por_tema.setdefault(tema_id, {})[origem] = valor

    def acrescentar(tema_id: int, origem: str, valor: float) -> None:
        origens = por_tema.setdefault(tema_id, {})
        origens[origem] = max(origens.get(origem, 0.0), valor)

    for tema_id in list(por_tema):
        if arvore.nivel(tema_id) >= 2:
            pai = arvore.pai(tema_id)
            if pai is not None and pai in publicaveis:
                acrescentar(pai, ORIGEM_PAI, max(por_tema[tema_id].values()))
    for tema_id, valor in normalizados[ORIGEM_DICIONARIO].items():
        if arvore.nivel(tema_id) == 1:
            for filho in arvore.filhos(tema_id):
                if filho in publicaveis:
                    acrescentar(filho, ORIGEM_FILHO_DO_PAI, valor)

    candidatos = [
        Candidato(tema_id=tema_id, origens=set(origens), pontuacao=max(origens.values()), por_origem=dict(origens))
        for tema_id, origens in por_tema.items()
    ]
    candidatos.sort(key=lambda c: (-len(c.origens), -c.pontuacao, c.tema_id))
    return candidatos[:teto]


def gerar_lote(
    ids: Iterable[str],
    contexto: "Contexto",
    indice_rotulado: IndiceVetorial,
    peso_por_id: Mapping[Hashable, float] | None = None,
    k_bm25: int = K_BM25,
    k_emb: int = K_EMB,
    k_knn: int = K_KNN,
    teto: int = TETO,
) -> tuple[dict[str, list[Candidato]], dict[str, list[Vizinho]]]:
    """Candidatos e vizinhos de cada questão (ou exemplo, ou item do corpus) do contexto, por id.

    Os vizinhos voltam para `caracteristicas.tabela_pares` usar sem buscar de novo. BM25 e cosseno com os temas
    vêm do contexto, calculados uma vez por item: entre as dobras da validação só o kNN é refeito.
    """
    candidatos_por_id: dict[str, list[Candidato]] = {}
    vizinhos_por_id: dict[str, list[Vizinho]] = {}
    for id_item in ids:
        item = contexto.item(id_item)
        vetor = contexto.vetor(id_item)
        vizinhos = buscar_vizinhos(id_item, vetor, indice_rotulado, k_knn)
        vizinhos_por_id[id_item] = vizinhos
        candidatos_por_id[id_item] = gerar(
            item, contexto.chaves(id_item), vetor, contexto, indice_rotulado,
            k_bm25=k_bm25, k_emb=k_emb, k_knn=k_knn, teto=teto, peso_por_id=peso_por_id, vizinhos=vizinhos,
            pontos_bm25=contexto.pontuacoes_bm25(id_item), cossenos=contexto.cossenos_temas(id_item),
        )
    return candidatos_por_id, vizinhos_por_id
