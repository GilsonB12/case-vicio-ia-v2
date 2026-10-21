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
| `n_buscas` | quantas das 5 buscas trouxeram o tema (0 a 5) |
| `disciplina_bate` | 1 se a `disciplina_origem` normalizada é a disciplina do tema |
| `prior` | log(1 + questões do tema na fonte, tirado da descrição original) dividido pelo maior entre os temas; 0 sem contagem |
| `termos_soma` | soma dos pesos das expressões técnicas do tema casadas na questão (`Contexto.pontuacoes_termos`) dividida pela maior soma entre os candidatos da questão; 0 se nenhum candidato casa |
| `termos_n` | expressões do tema casadas, cortado em 5 e dividido por 5 |
| `termos_max` | maior peso casado do tema dividido pelo maior peso do dicionário de termos |
| `voto_bm25`, `voto_cos`, `voto_dic`, `voto_knn`, `voto_termos` | 1 se o tema é o top-1 daquela busca (`topos_das_buscas`) |
| `votos` | soma dos cinco votos (0 a 5) |
| `votos_ramo` | quantas das cinco buscas têm o top-1 no mesmo ramo (ancestral de nível 1) do tema (0 a 5) |

Os vizinhos são os até 15 já buscados para a questão (`candidatos.gerar_lote`); par sem vizinho do tema tem as
`knn_*` zeradas. Similaridade negativa conta zero. Os votos reaproveitam o BM25, o cosseno e os termos do contexto,
o dicionário e os mesmos vizinhos: nenhuma busca nova.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Hashable, Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sistema.candidatos import K_KNN, Candidato, Vizinho, buscar_vizinhos
from sistema.dicionario import pontuar
from sistema.vizinhos import IndiceVetorial

if TYPE_CHECKING:
    from sistema.arvore import ArvoreTemas
    from sistema.contexto import Contexto

CARACTERISTICAS = [
    "bm25", "cos_tema", "knn_max", "knn_media3", "knn_n10", "knn_folha_max", "dic_lei", "dic_art", "nivel2",
    "pai_de_candidato", "n_buscas", "disciplina_bate", "prior", "termos_soma", "termos_n", "termos_max",
    "voto_bm25", "voto_cos", "voto_dic", "voto_knn", "voto_termos", "votos", "votos_ramo",
]
COLUNAS = ["id", "tema_id", *CARACTERISTICAS]

BUSCAS_VOTO = ("bm25", "cos", "dic", "knn", "termos")  # `voto_<busca>` de cada uma

N_PROXIMOS = 10
N_MEDIA = 3
N_TERMOS = 5  # `termos_n` satura neste número de expressões casadas

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


def caracteristicas_termos(pontos: tuple[float, int, float] | None, maior_soma: float,
                           peso_maximo: float) -> dict[str, float]:
    """`termos_soma`, `termos_n` e `termos_max` de um par, a partir dos pontos do tema (`IndiceTermos.pontuar`)."""
    if not pontos:
        return {"termos_soma": 0.0, "termos_n": 0.0, "termos_max": 0.0}
    soma, quantas, maior_peso = pontos
    return {
        "termos_soma": float(soma) / maior_soma if maior_soma > 0 else 0.0,
        "termos_n": min(int(quantas), N_TERMOS) / N_TERMOS,
        "termos_max": float(maior_peso) / peso_maximo if peso_maximo > 0 else 0.0,
    }


def _maior_positivo(pontos: Mapping[int, float], publicaveis: Collection[int]) -> int | None:
    """Tema publicável de maior pontuação positiva; empate fica com o menor id; None se nenhum pontua."""
    melhores = [(int(t), float(v)) for t, v in pontos.items() if v > 0 and int(t) in publicaveis]
    return min(melhores, key=lambda par: (-par[1], par[0]))[0] if melhores else None


def _maior_no_vetor(valores: np.ndarray, temas_ids: list[int], mascara: np.ndarray) -> int | None:
    """Tema publicável (`mascara`) de maior valor positivo no vetor alinhado a `temas_ids`; empate, menor posição."""
    if not len(valores):
        return None
    filtrados = np.where(mascara, np.asarray(valores, dtype=float), -np.inf)
    posicao = int(np.argmax(filtrados))
    return int(temas_ids[posicao]) if filtrados[posicao] > 0 else None


def topo_knn(vizinhos: Iterable[Vizinho], peso_por_id: Mapping[Hashable, float],
             publicaveis: Collection[int]) -> int | None:
    """Tema com a maior soma de similaridade × peso entre os vizinhos rotulados não-fora (similaridade negativa
    conta zero); empate fica com o menor id."""
    somas: dict[int, float] = defaultdict(float)
    for id_vizinho, rotulo, similaridade in vizinhos:
        if rotulo is None:
            continue
        somas[int(rotulo)] += max(0.0, float(similaridade)) * float(peso_por_id.get(id_vizinho, 1.0))
    return _maior_positivo(somas, publicaveis)


def topos_das_buscas(
    id_item: str,
    contexto: "Contexto",
    vizinhos: Iterable[Vizinho],
    peso_por_id: Mapping[Hashable, float],
    mascara_publicaveis: np.ndarray | None = None,
) -> dict[str, int | None]:
    """Top-1 de cada uma das cinco buscas para o item, só entre temas publicáveis (None quando nada pontua).

    `bm25` e `cos`: maior BM25 e maior cosseno com os temas (do contexto, calculados uma vez por item); `dic`:
    maior soma de pesos do dicionário (`dicionario.pontuar`); `knn`: `topo_knn` dos vizinhos dados; `termos`: maior
    soma de pesos das expressões técnicas casadas (`Contexto.pontuacoes_termos`). Empate fica com o menor id do tema.
    """
    temas_ids = contexto.temas_ids
    if mascara_publicaveis is None:
        mascara_publicaveis = np.array([t in contexto.publicaveis for t in temas_ids], dtype=bool)
    pontos_dicionario, _ = pontuar(contexto.chaves(id_item), contexto.dicionario)
    return {
        "bm25": _maior_no_vetor(contexto.pontuacoes_bm25(id_item), temas_ids, mascara_publicaveis),
        "cos": _maior_no_vetor(contexto.cossenos_temas(id_item), temas_ids, mascara_publicaveis),
        "dic": _maior_positivo(pontos_dicionario, contexto.publicaveis),
        "knn": topo_knn(vizinhos, peso_por_id, contexto.publicaveis),
        "termos": _maior_positivo({t: p[0] for t, p in contexto.pontuacoes_termos(id_item).items()},
                                  contexto.publicaveis),
    }


def votos_do_tema(tema_id: int, topos: Mapping[str, int | None], arvore: "ArvoreTemas") -> dict[str, float]:
    """`voto_<busca>`, `votos` e `votos_ramo` do par (questão, tema) a partir dos top-1 das buscas."""
    ramo = _ramo(arvore, tema_id)
    votos = {f"voto_{busca}": 1.0 if topos.get(busca) == int(tema_id) else 0.0 for busca in BUSCAS_VOTO}
    votos["votos"] = float(sum(votos.values()))
    votos["votos_ramo"] = float(sum(
        1 for busca in BUSCAS_VOTO
        if topos.get(busca) is not None and ramo is not None and _ramo(arvore, topos[busca]) == ramo
    ))
    return votos


def _ramo(arvore: "ArvoreTemas", tema_id: int) -> int | None:
    """Ancestral de nível 1 do tema; None para a raiz de disciplina."""
    return None if arvore.eh_raiz(int(tema_id)) else arvore.ancestral_nivel1(int(tema_id))


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
    BM25, cosseno e termos técnicos vêm do contexto (`pontuacoes_bm25`, `cossenos_temas`, `pontuacoes_termos`),
    calculados uma vez por item.
    """
    arvore = contexto.arvore
    posicao_tema = {tema_id: i for i, tema_id in enumerate(contexto.temas_ids)}
    prior = prior_dos_temas(arvore, contexto.temas_ids)
    disciplina_tema = {tema_id: normalizar_disciplina(arvore.disciplina(tema_id)) for tema_id in contexto.temas_ids}
    folha_treino = set(ids_folha_treino)
    mascara_publicaveis = np.array([t in contexto.publicaveis for t in contexto.temas_ids], dtype=bool)
    peso_maximo_termos = contexto.indice_termos.peso_maximo
    linhas: list[dict] = []
    for id_questao in questoes_ids:
        candidatos = candidatos_por_id.get(id_questao) or []
        if not candidatos:
            continue
        vetor = contexto.vetor(id_questao)
        bm25 = contexto.pontuacoes_bm25(id_questao)
        maior_bm25 = float(bm25.max()) if bm25.size else 0.0
        cossenos = contexto.cossenos_temas(id_questao)
        termos = contexto.pontuacoes_termos(id_questao)
        maior_soma_termos = max((termos[c.tema_id][0] for c in candidatos if c.tema_id in termos), default=0.0)
        if vizinhos_por_id is not None and id_questao in vizinhos_por_id:
            vizinhos = list(vizinhos_por_id[id_questao])[:k_knn]
        else:
            vizinhos = buscar_vizinhos(id_questao, vetor, indice_rotulado, k_knn)
        knn = _knn_por_tema(vizinhos, peso_por_id, folha_treino)
        topos = topos_das_buscas(id_questao, contexto, vizinhos, peso_por_id, mascara_publicaveis)
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
                **caracteristicas_termos(termos.get(tema_id), maior_soma_termos, peso_maximo_termos),
                **votos_do_tema(tema_id, topos, arvore),
            })
    tabela = pd.DataFrame(linhas, columns=COLUNAS)
    tabela["tema_id"] = tabela["tema_id"].astype("int64")
    tabela[CARACTERISTICAS] = tabela[CARACTERISTICAS].astype("float64")
    return tabela
