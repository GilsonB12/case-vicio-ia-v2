"""Características por par (questão, tema candidato), para o modelo de pares (spec, seção 6).

| coluna | definição |
|---|---|
| `bm25` | pontuação BM25 do tema dividida pela maior pontuação da questão entre todos os temas (0 a 1) |
| `cos_tema` | cosseno entre o embedding da questão e o do texto do tema |
| `knn_max` | maior similaridade × peso (corpus 0,5) entre os vizinhos rotulados com o tema |
| `knn_media3` | soma das 3 maiores similaridades × peso do tema dividida por 3 (vizinho que falta conta zero) |
| `knn_n10` | soma dos pesos dos vizinhos do tema entre os 10 mais próximos, dividida por 10 (0 a 1) |
| `knn_folha_max` | maior similaridade entre os vizinhos do tema que são questões da folha de treino |
| `dic_lei`, `dic_art` | soma dos pesos do dicionário das chaves da questão que apontam para o tema: `dic_art` com as chaves `art:`; `dic_lei` com as demais (`lei:`, `dl:`, `diploma:`, `sumula:`) |
| `nivel2` | 1 se o tema é subtema (nível 2) |
| `pai_de_candidato` | tema de nível 1: maior `cos_tema` entre os filhos que também são candidatos; 0 sem filho candidato |
| `n_buscas` | quantas das 4 buscas trouxeram o tema (0 a 4) |
| `disciplina_bate` | 1 se a `disciplina_origem` normalizada é a disciplina do tema |
| `prior` | log(1 + questões do tema na fonte, tirado da descrição original) dividido pelo maior entre os temas; 0 sem contagem |

Os vizinhos são os até 15 já buscados para a questão (`candidatos.gerar_lote`); par sem vizinho do tema tem as
`knn_*` zeradas. Similaridade negativa conta zero.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Hashable, Iterable, Mapping
from typing import TYPE_CHECKING

import pandas as pd

from sistema.candidatos import K_KNN, Candidato, Vizinho, buscar_vizinhos
from sistema.dicionario import pontuar
from sistema.vizinhos import IndiceVetorial

if TYPE_CHECKING:
    from sistema.arvore import ArvoreTemas
    from sistema.contexto import Contexto

CARACTERISTICAS = [
    "bm25", "cos_tema", "knn_max", "knn_media3", "knn_n10", "knn_folha_max", "dic_lei", "dic_art", "nivel2",
    "pai_de_candidato", "n_buscas", "disciplina_bate", "prior",
]
COLUNAS = ["id", "tema_id", *CARACTERISTICAS]

N_PROXIMOS = 10
N_MEDIA = 3

DISCIPLINAS = ("Penal", "Processo Penal", "Direito Penal - Legislação Especial", "Criminologia")
_PREFIXO_ARTIGO = "art:"
_QUESTOES_NA_FONTE = re.compile(r"(\d+)\s+quest(?:ões|oes)", re.IGNORECASE)


def _chave_disciplina(texto: str | None) -> str:
    sem_acento = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"\s*-\s*", " - ", sem_acento)).strip()


_APELIDOS_DISCIPLINA = {
    "penal": "Penal",
    "direito penal": "Penal",
    "processo penal": "Processo Penal",
    "direito processual penal": "Processo Penal",
    "direito penal - legislacao especial": "Direito Penal - Legislação Especial",
    "criminologia": "Criminologia",
}


def normalizar_disciplina(texto: str | None) -> str | None:
    """`disciplina_origem` ou disciplina da árvore como uma das quatro da árvore; None para as outras.

    "Penal" e "Direito Penal" viram Penal; "Processo Penal" e "Direito Processual Penal", Processo Penal;
    as militares, constitucional, civil, tributária, ambiental e vazias ficam sem disciplina.
    """
    return _APELIDOS_DISCIPLINA.get(_chave_disciplina(texto))


def prior_dos_temas(arvore: "ArvoreTemas", temas_ids: Iterable[int]) -> dict[int, float]:
    """log1p(questões na fonte) / maior valor entre `temas_ids`; o duplicado usa a descrição do original."""
    brutos: dict[int, float] = {}
    for tema_id in temas_ids:
        achado = _QUESTOES_NA_FONTE.search(arvore.descricao(arvore.original_de(tema_id)))
        brutos[int(tema_id)] = math.log1p(int(achado.group(1))) if achado else 0.0
    maior = max(brutos.values(), default=0.0)
    return {tema_id: (valor / maior if maior > 0 else 0.0) for tema_id, valor in brutos.items()}


def pontos_por_tipo(chaves: Counter, dicionario: dict) -> tuple[dict[int, float], dict[int, float]]:
    """(pontos das chaves de lei, diploma e súmula; pontos das chaves de artigo) por tema."""
    de_artigo = Counter({c: n for c, n in chaves.items() if c.startswith(_PREFIXO_ARTIGO)})
    de_lei = Counter({c: n for c, n in chaves.items() if not c.startswith(_PREFIXO_ARTIGO)})
    return pontuar(de_lei, dicionario)[0], pontuar(de_artigo, dicionario)[0]


def _knn_por_tema(
    vizinhos: list[Vizinho], peso_por_id: Mapping[Hashable, float], ids_folha_treino: Collection[str]
) -> dict[int, dict[str, float]]:
    valores: dict[int, list[float]] = defaultdict(list)
    proximos: dict[int, float] = defaultdict(float)
    da_folha: dict[int, float] = defaultdict(float)
    for posicao, (id_vizinho, rotulo, similaridade) in enumerate(vizinhos):
        if rotulo is None:
            continue
        tema_id = int(rotulo)
        peso = float(peso_por_id.get(id_vizinho, 1.0))
        similaridade = max(0.0, float(similaridade))
        valores[tema_id].append(similaridade * peso)
        if posicao < N_PROXIMOS:
            proximos[tema_id] += peso
        if id_vizinho in ids_folha_treino:
            da_folha[tema_id] = max(da_folha[tema_id], similaridade)
    resultado: dict[int, dict[str, float]] = {}
    for tema_id, lista in valores.items():
        maiores = sorted(lista, reverse=True)[:N_MEDIA]
        resultado[tema_id] = {
            "knn_max": maiores[0],
            "knn_media3": sum(maiores) / N_MEDIA,
            "knn_n10": proximos[tema_id] / N_PROXIMOS,
            "knn_folha_max": da_folha[tema_id],
        }
    return resultado


_KNN_ZERADO = {"knn_max": 0.0, "knn_media3": 0.0, "knn_n10": 0.0, "knn_folha_max": 0.0}


def tabela_pares(
    questoes_ids: Iterable[str],
    candidatos_por_id: Mapping[str, list[Candidato]],
    contexto: "Contexto",
    indice_rotulado: IndiceVetorial,
    peso_por_id: Mapping[Hashable, float],
    ids_folha_treino: Collection[str],
    vizinhos_por_id: Mapping[str, list[Vizinho]] | None = None,
    k_knn: int = K_KNN,
) -> pd.DataFrame:
    """Uma linha por (questão, candidato), na ordem das questões e dos candidatos; colunas em `COLUNAS`.

    `questoes_ids` podem ser questões, exemplos ou itens do corpus do contexto. Sem `vizinhos_por_id` (ou sem a
    questão nele), os `k_knn` vizinhos são buscados uma vez por questão em `indice_rotulado`, sem ela mesma.
    BM25 e cosseno com os temas vêm do contexto (`pontuacoes_bm25`, `cossenos_temas`), calculados uma vez por item.
    """
    arvore = contexto.arvore
    posicao_tema = {tema_id: i for i, tema_id in enumerate(contexto.temas_ids)}
    prior = prior_dos_temas(arvore, contexto.temas_ids)
    disciplina_tema = {tema_id: normalizar_disciplina(arvore.disciplina(tema_id)) for tema_id in contexto.temas_ids}
    folha_treino = set(ids_folha_treino)
    linhas: list[dict] = []
    for id_questao in questoes_ids:
        candidatos = candidatos_por_id.get(id_questao) or []
        if not candidatos:
            continue
        vetor = contexto.vetor(id_questao)
        bm25 = contexto.pontuacoes_bm25(id_questao)
        maior_bm25 = float(bm25.max()) if bm25.size else 0.0
        cossenos = contexto.cossenos_temas(id_questao)
        if vizinhos_por_id is not None and id_questao in vizinhos_por_id:
            vizinhos = list(vizinhos_por_id[id_questao])[:k_knn]
        else:
            vizinhos = buscar_vizinhos(id_questao, vetor, indice_rotulado, k_knn)
        knn = _knn_por_tema(vizinhos, peso_por_id, folha_treino)
        dic_lei, dic_art = pontos_por_tipo(contexto.chaves(id_questao), contexto.dicionario)
        disciplina = normalizar_disciplina(contexto.disciplina_origem(id_questao))
        ids_candidatos = {c.tema_id for c in candidatos}

        def cos(tema_id: int) -> float:
            return float(cossenos[posicao_tema[tema_id]])

        for candidato in candidatos:
            tema_id = int(candidato.tema_id)
            nivel = arvore.nivel(tema_id)
            filhos_candidatos = [f for f in arvore.filhos(tema_id) if f in ids_candidatos] if nivel == 1 else []
            linhas.append({
                "id": id_questao,
                "tema_id": tema_id,
                "bm25": float(bm25[posicao_tema[tema_id]]) / maior_bm25 if maior_bm25 > 0 else 0.0,
                "cos_tema": cos(tema_id),
                **knn.get(tema_id, _KNN_ZERADO),
                "dic_lei": float(dic_lei.get(tema_id, 0.0)),
                "dic_art": float(dic_art.get(tema_id, 0.0)),
                "nivel2": 1.0 if nivel >= 2 else 0.0,
                "pai_de_candidato": max((cos(f) for f in filhos_candidatos), default=0.0),
                "n_buscas": float(candidato.n_buscas),
                "disciplina_bate": 1.0 if disciplina is not None and disciplina == disciplina_tema[tema_id] else 0.0,
                "prior": prior.get(tema_id, 0.0),
            })
    tabela = pd.DataFrame(linhas, columns=COLUNAS)
    tabela["tema_id"] = tabela["tema_id"].astype("int64")
    tabela[CARACTERISTICAS] = tabela[CARACTERISTICAS].astype("float64")
    return tabela
