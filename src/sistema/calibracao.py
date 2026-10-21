"""Validação cruzada na folha-verdade, grade de limiares, escada e comparação de fontes (spec, seção 8).

Uma rodada (`treinar_rodada`) usa só os rótulos de treino que recebe: monta o índice kNN com a folha de treino, os
exemplos sintéticos e, se ligado, o corpus; gera candidatos e características desses itens; treina o modelo de
pares (folha de treino com peso 1, exemplos com tema e corpus com peso 0,5) e o detector de fora (folha de treino
com peso 1, exemplos fora como positivos e exemplos com tema como negativos, peso 0,5).

O segundo estágio (`ModeloConfianca`) aprende com previsões fora da amostra do modelo de pares: dentro da rodada,
uma validação interna de `N_DOBRAS_CONFIANCA` dobras nas questões de treino da folha refaz índice e modelo de
pares nas outras dobras internas e prevê as opções (`tema` e `ramo`) das questões da dobra interna, com o rótulo
da régua (`opcoes_fora_da_amostra`; questões fora da taxonomia não entram). `rodar_dobra` aplica a rodada nas
questões de teste da dobra, que não entram em nada do treino.

As dobras da validação rodam em paralelo (`N_TAREFAS` processos; -1 = todos os núcleos; variável de ambiente
`SISTEMA_TAREFAS`): cada dobra é independente e tem sementes fixas, então o resultado é o mesmo com qualquer número
de processos. Antes de distribuir, o contexto calcula uma vez as chaves, o BM25 e os cossenos de todos os itens e
vai uma vez para cada processo (sem o cache de embeddings, que os processos não leem). Se o grupo de processos
quebrar, as dobras rodam em série.

A grade varre o limiar da confiança do segundo estágio. Com repetições, cada questão da folha aparece uma vez por
repetição: a precisão de Jeffreys e o intervalo Beta usam as contagens divididas pelo número de repetições
(n efetivo, que pode ser fracionário).
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import copy
import logging
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from sistema.candidatos import Candidato, Vizinho, gerar_lote
from sistema.caracteristicas import normalizar_disciplina, tabela_pares
from sistema.config import SEMENTE
from sistema.dados import acerto_regua, alvos_aceitos
from sistema.decisao import (
    FORA,
    OPCAO_RAMO,
    OPCAO_TEMA,
    PUBLICAR,
    Decisao,
    Limiares,
    Opcao,
    decidir_listas,
    opcoes_de_tabela,
)
from sistema.modelo import (
    COLUNAS_OPCOES,
    DetectorFora,
    ModeloConfianca,
    ModeloPares,
    agregar_ramo,
    gatilhos_fora,
    pesos_por_linha,
    rotulos_pares,
    tabela_opcoes,
    tabela_questoes,
    temas_positivos,
)

if TYPE_CHECKING:
    from sistema.arvore import ArvoreTemas
    from sistema.contexto import Contexto

N_DOBRAS = 5
REPETICOES = 3
GRADE_PUBLICAR = tuple(round(0.50 + 0.01 * i, 2) for i in range(50))  # confiança de 0,50 a 0,99
GRADE_FORA = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_SORTEIOS = 10_000
CONFIANCA = 0.95

PESO_FOLHA = 1.0
PESO_EXEMPLO = 0.5
PESO_CORPUS = 0.5

N_DOBRAS_CONFIANCA = 5
N_TAREFAS = int(os.environ.get("SISTEMA_TAREFAS", "-1"))  # processos para as dobras da validação

registro = logging.getLogger(__name__)

PRECISAO_ALVO = 0.97
LIMITE_INFERIOR_ALVO = 0.95
PRECISAO_REGUA = 0.95
N_EFETIVO_MINIMO = 10
GANHO_MINIMO_CORPUS = 0.03

ESTRATO_FORA = "fora"
ESTRATO_OUTROS = "outros"

ERRO_IRMAO = "irmao"
ERRO_OUTRO_RAMO = "outro_ramo"
ERRO_FORA_PUBLICADA = "fora_publicada"
TIPOS_ERRO = (ERRO_IRMAO, ERRO_OUTRO_RAMO, ERRO_FORA_PUBLICADA)

_TOLERANCIA = 1e-9


# --- dobras ------------------------------------------------------------------------------------------------


def estratos(folha: pd.DataFrame, arvore: "ArvoreTemas") -> list[str]:
    """Estrato de cada questão da folha: "fora" ou a disciplina do tema central; estratos com menos membros
    que dobras viram "outros" (e "outros" pequeno demais se junta ao maior estrato) em `dobras`."""
    resultado: list[str] = []
    for linha in folha.to_dict("records"):
        if bool(linha["fora"]) or linha["tema_central_id"] is None:
            resultado.append(ESTRATO_FORA)
        else:
            disciplina = arvore.disciplina(int(linha["tema_central_id"]))
            resultado.append(normalizar_disciplina(disciplina) or disciplina)
    return resultado


def _juntar_pequenos(rotulos: list[str], minimo: int) -> list[str]:
    contagem = pd.Series(rotulos).value_counts()
    pequenos = set(contagem[contagem < minimo].index)
    rotulos = [ESTRATO_OUTROS if r in pequenos else r for r in rotulos]
    contagem = pd.Series(rotulos).value_counts()
    if ESTRATO_OUTROS in contagem and contagem[ESTRATO_OUTROS] < minimo and len(contagem) > 1:
        maior = contagem.drop(ESTRATO_OUTROS).sort_values(ascending=False, kind="stable").index[0]
        rotulos = [maior if r == ESTRATO_OUTROS else r for r in rotulos]
    return rotulos


def dobras(
    folha: pd.DataFrame,
    n_dobras: int = N_DOBRAS,
    repeticoes: int = REPETICOES,
    semente: int = SEMENTE,
    arvore: "ArvoreTemas | None" = None,
) -> list[tuple[list[str], list[str]]]:
    """`repeticoes` × `n_dobras` pares (ids de treino, ids de teste), estratificados por (fora, disciplina).

    `StratifiedKFold` embaralhado, semente `semente + repetição`; ids na ordem da folha.
    """
    if arvore is None:
        from sistema.arvore import ArvoreTemas

        arvore = ArvoreTemas.carregar()
    ids = [str(i) for i in folha["id"]]
    if len(ids) < n_dobras:
        raise ValueError(f"folha com {len(ids)} questões não dá {n_dobras} dobras")
    rotulos = _juntar_pequenos(estratos(folha, arvore), n_dobras)
    resultado: list[tuple[list[str], list[str]]] = []
    for repeticao in range(repeticoes):
        divisor = StratifiedKFold(n_splits=n_dobras, shuffle=True, random_state=semente + repeticao)
        for treino, teste in divisor.split(np.zeros(len(ids)), rotulos):
            resultado.append(([ids[i] for i in sorted(treino)], [ids[i] for i in sorted(teste)]))
    return resultado


# --- rodada: treino e aplicação ---------------------------------------------------------------------------


@dataclass
class Aplicacao:
    """Tudo o que uma rodada calcula para um conjunto de ids.

    `opcoes`: `modelo.tabela_opcoes` com a coluna `confianca` do segundo estágio (sem o modelo de confiança, como
    na validação interna, a coluna fica de fora).
    """

    ids: list[str]
    candidatos: dict[str, list[Candidato]]
    vizinhos: dict[str, list[Vizinho]]
    pares: pd.DataFrame
    prob: pd.DataFrame
    ramos: pd.DataFrame
    tabela_q: pd.DataFrame
    p_fora: pd.Series
    opcoes: pd.DataFrame

    def listas(self, id_item: str) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """(tema, p) e (pai, p_ramo) da questão, em ordem decrescente."""
        return self._por_id_prob.get(id_item, []), self._por_id_ramo.get(id_item, [])

    def opcoes_de(self, id_item: str) -> list[Opcao]:
        """As opções da questão (`tema` e, se houver, `ramo`) com a confiança do segundo estágio."""
        return self._por_id_opcoes.get(id_item, [])

    def linhas_opcoes(self, id_item: str) -> list[dict]:
        """As linhas de `opcoes` da questão, com as características e a confiança."""
        return self._linhas_opcoes.get(id_item, [])

    def __post_init__(self) -> None:
        self._por_id_prob = _agrupar(self.prob, "p")
        self._por_id_ramo = _agrupar(self.ramos, "p_ramo")
        self._linhas_opcoes: dict[str, list[dict]] = {}
        for linha in self.opcoes.to_dict("records"):
            self._linhas_opcoes.setdefault(str(linha["id"]), []).append(linha)
        self._por_id_opcoes: dict[str, list[Opcao]] = {}
        if "confianca" in self.opcoes:
            self._por_id_opcoes = {
                id_q: [Opcao(str(linha["opcao"]), int(linha["tema_id"]), float(linha["confianca"]),
                             int(linha["votos"])) for linha in linhas]
                for id_q, linhas in self._linhas_opcoes.items()
            }


def _agrupar(tabela: pd.DataFrame, coluna: str) -> dict[str, list[tuple[int, float]]]:
    grupos: dict[str, list[tuple[int, float]]] = {}
    for id_q, tema_id, valor in zip(tabela["id"], tabela["tema_id"], tabela[coluna]):
        grupos.setdefault(str(id_q), []).append((int(tema_id), float(valor)))
    return {id_q: sorted(lista, key=lambda par: (-par[1], par[0])) for id_q, lista in grupos.items()}


@dataclass
class Rodada:
    """Índice, modelo de pares, detector e modelo de confiança treinados com `ids_treino` da folha (mais exemplos e
    corpus); `ids_confianca` são as questões da folha cujas opções fora da amostra treinaram a confiança."""

    contexto: "Contexto"
    com_corpus: bool
    ids_treino: list[str]
    indice: object
    pesos_vizinhos: dict[Hashable, float]
    modelo: ModeloPares
    detector: DetectorFora | None
    ids_pares: list[str] = field(default_factory=list)
    ids_detector: list[str] = field(default_factory=list)
    confianca: ModeloConfianca | None = None
    ids_confianca: list[str] = field(default_factory=list)

    def aplicar(self, ids: Iterable[str]) -> Aplicacao:
        """Candidatos, características, `p`, `p_ramo`, `p_fora` e opções com confiança dos ids, com o índice e os
        modelos da rodada (sem detector, `p_fora` é zero; sem modelo de confiança, as opções ficam sem ela)."""
        return _aplicar(self.contexto, list(ids), self.indice, self.pesos_vizinhos, self.ids_treino, self.modelo,
                        self.detector, self.confianca)


def _aplicar(
    contexto: "Contexto",
    ids: list[str],
    indice,
    pesos_vizinhos: Mapping[Hashable, float],
    ids_treino: Sequence[str],
    modelo: ModeloPares,
    detector: DetectorFora | None,
    confianca: ModeloConfianca | None = None,
) -> Aplicacao:
    candidatos, vizinhos = gerar_lote(ids, contexto, indice, pesos_vizinhos)
    pares = tabela_pares(ids, candidatos, contexto, indice, pesos_vizinhos, set(ids_treino), vizinhos_por_id=vizinhos)
    prob = modelo.probabilidades(pares)
    ramos = agregar_ramo(prob, contexto.arvore)
    tabela_q = tabela_questoes(ids, prob, pares, vizinhos, contexto)
    if detector is not None:
        p_fora = detector.p_fora(tabela_q)
    else:
        p_fora = pd.Series(0.0, index=pd.Index(tabela_q["id"].astype(str), name="id"), name="p_fora")
    opcoes = tabela_opcoes(prob, ramos, pares, contexto)
    if confianca is not None:
        opcoes["confianca"] = confianca.confianca(opcoes)
    return Aplicacao(ids, candidatos, vizinhos, pares, prob, ramos, tabela_q, p_fora, opcoes)


def _treinar_base(contexto: "Contexto", ids_treino: list[str], com_corpus: bool, com_detector: bool = True) -> Rodada:
    """Índice, modelo de pares e (com `com_detector`) detector com os rótulos de `ids_treino`; sem confiança."""
    indice, pesos_vizinhos = contexto.indice_rotulado(ids_treino, com_corpus=com_corpus)
    exemplos_tema = [str(e["id"]) for e in contexto.exemplos if e["tema_id"] is not None]
    exemplos_fora = [str(e["id"]) for e in contexto.exemplos if e["tema_id"] is None]
    corpus = [str(c["id"]) for c in contexto.corpus] if com_corpus else []

    peso_pares = {**{i: PESO_FOLHA for i in ids_treino}, **{i: PESO_EXEMPLO for i in exemplos_tema},
                  **{i: PESO_CORPUS for i in corpus}}
    ids_pares = [*ids_treino, *exemplos_tema, *corpus]
    ids_detector = [*ids_treino, *exemplos_fora, *exemplos_tema] if com_detector else []
    ids_todos = list(dict.fromkeys([*ids_pares, *ids_detector]))

    modelo = ModeloPares()
    candidatos, vizinhos = gerar_lote(ids_todos, contexto, indice, pesos_vizinhos)
    pares_todos = tabela_pares(ids_todos, candidatos, contexto, indice, pesos_vizinhos, set(ids_treino),
                               vizinhos_por_id=vizinhos)
    pares = pares_todos[pares_todos["id"].isin(set(ids_pares))].reset_index(drop=True)
    positivos = {i: temas_positivos(contexto, i) for i in ids_pares}
    modelo.treinar(pares, rotulos_pares(pares, positivos), pesos_por_linha(pares, peso_pares))
    if not com_detector:
        return Rodada(contexto, com_corpus, ids_treino, indice, pesos_vizinhos, modelo, None, ids_pares, [])

    pares_detector = pares_todos[pares_todos["id"].isin(set(ids_detector))].reset_index(drop=True)
    prob_detector = modelo.probabilidades(pares_detector)
    tabela_q = tabela_questoes(ids_detector, prob_detector, pares_detector, vizinhos, contexto)
    rotulos_fora, pesos_fora = [], []
    for id_item in ids_detector:
        item = contexto.item(id_item)
        if item["origem"] == "questao":
            rotulos_fora.append(int(bool(contexto.linha_folha(id_item)["fora"])))
            pesos_fora.append(PESO_FOLHA)
        else:
            rotulos_fora.append(int(item["tema_id"] is None))
            pesos_fora.append(PESO_EXEMPLO)
    detector = DetectorFora().treinar(tabela_q, rotulos_fora, pesos_fora)
    return Rodada(contexto, com_corpus, ids_treino, indice, pesos_vizinhos, modelo, detector, ids_pares,
                  ids_detector)


COLUNAS_OPCOES_TREINO = ["dobra_interna", "acerto"]


def opcoes_fora_da_amostra(
    contexto: "Contexto",
    ids_treino: Iterable[str],
    com_corpus: bool,
    n_dobras: int = N_DOBRAS_CONFIANCA,
    semente: int = SEMENTE,
) -> pd.DataFrame:
    """Opções (`modelo.tabela_opcoes`) das questões de `ids_treino`, previstas fora da amostra, com o rótulo.

    Validação interna de `n_dobras` dobras estratificadas nas questões de treino da folha: em cada uma, índice e
    modelo de pares são refeitos só com as outras dobras internas (mais exemplos e corpus) e preveem as opções das
    questões da dobra interna. `acerto` = a opção está certa pela régua; questões fora da taxonomia não entram.
    Com menos de 2 questões de treino, a tabela sai vazia.
    """
    ids_treino = [str(i) for i in ids_treino]
    vazia = pd.DataFrame(columns=[*COLUNAS_OPCOES, *COLUNAS_OPCOES_TREINO])
    n = min(int(n_dobras), len(ids_treino))
    if n < 2:
        return vazia
    treino = set(ids_treino)
    folha = contexto.folha[contexto.folha["id"].astype(str).isin(treino)].reset_index(drop=True)
    partes: list[pd.DataFrame] = []
    for numero, (dentro, fora_da_dobra) in enumerate(dobras(folha, n, 1, semente, arvore=contexto.arvore)):
        teste = [i for i in fora_da_dobra if not bool(contexto.linha_folha(i)["fora"])]
        if not teste:
            continue
        base = _treinar_base(contexto, dentro, com_corpus, com_detector=False)
        opcoes = base.aplicar(teste).opcoes
        opcoes["dobra_interna"] = numero
        opcoes["acerto"] = [
            int(acerto_regua(int(t), contexto.linha_folha(str(i)), contexto.arvore))
            for i, t in zip(opcoes["id"], opcoes["tema_id"])
        ]
        partes.append(opcoes)
    return pd.concat(partes, ignore_index=True) if partes else vazia


def treinar_rodada(contexto: "Contexto", ids_treino: Iterable[str], com_corpus: bool,
                   n_dobras_confianca: int = N_DOBRAS_CONFIANCA) -> Rodada:
    """Treina uma rodada só com os rótulos de `ids_treino` (folha), os exemplos e, com `com_corpus`, o corpus.

    Primeiro a validação interna (`opcoes_fora_da_amostra`) e o `ModeloConfianca` sobre ela; depois o índice, o
    modelo de pares e o detector da rodada com todas as questões de treino.
    """
    ids_treino = [str(i) for i in ids_treino]
    opcoes = opcoes_fora_da_amostra(contexto, ids_treino, com_corpus, n_dobras_confianca)
    confianca = ModeloConfianca().treinar(opcoes, opcoes["acerto"].astype(int).to_numpy())
    rodada = _treinar_base(contexto, ids_treino, com_corpus)
    rodada.confianca = confianca
    rodada.ids_confianca = list(dict.fromkeys(str(i) for i in opcoes["id"]))
    return rodada


def treinar_final(contexto: "Contexto", com_corpus: bool) -> tuple[ModeloPares, DetectorFora, ModeloConfianca]:
    """Modelo de pares, detector e confiança treinados com os 100 rótulos da folha (mais exemplos e corpus se
    ligado)."""
    rodada = treinar_rodada(contexto, contexto.ids_folha, com_corpus)
    return rodada.modelo, rodada.detector, rodada.confianca


COLUNAS_DOBRA = [
    "id", "p_max", "tema_max", "p_ramo_max", "pai_max", "p_fora", "candidatos_prob", "ramos", "opcoes",
    "confianca_tema", "confianca_ramo", "votos", "votos_ramo", "gatilhos", "fora", "tema_central_id", "extras_ids",
    "alvos",
]


def resultados_teste(contexto: "Contexto", aplicacao: Aplicacao, ids_teste: Iterable[str]) -> pd.DataFrame:
    """Uma linha por questão de teste com o que a decisão precisa e os alvos aceitos pela régua."""
    linhas: list[dict] = []
    for id_q in ids_teste:
        temas, ramos = aplicacao.listas(id_q)
        linha_folha = contexto.linha_folha(id_q)
        opcoes = aplicacao.opcoes_de(id_q)
        por_opcao = {linha["opcao"]: linha for linha in aplicacao.linhas_opcoes(id_q)}
        do_tema = por_opcao.get(OPCAO_TEMA, {})
        linhas.append({
            "id": id_q,
            "p_max": temas[0][1] if temas else 0.0,
            "tema_max": temas[0][0] if temas else None,
            "p_ramo_max": ramos[0][1] if ramos else 0.0,
            "pai_max": ramos[0][0] if ramos else None,
            "p_fora": float(aplicacao.p_fora.get(id_q, 0.0)),
            "candidatos_prob": temas,
            "ramos": ramos,
            "opcoes": opcoes,
            "confianca_tema": next((o.confianca for o in opcoes if o.opcao == OPCAO_TEMA), None),
            "confianca_ramo": next((o.confianca for o in opcoes if o.opcao == OPCAO_RAMO), None),
            "votos": int(do_tema.get("votos", 0)),
            "votos_ramo": int(do_tema.get("votos_ramo", 0)),
            "gatilhos": gatilhos_fora(contexto, id_q),
            "fora": bool(linha_folha["fora"]),
            "tema_central_id": linha_folha["tema_central_id"],
            "extras_ids": list(linha_folha["extras_ids"]),
            "alvos": sorted(alvos_aceitos(linha_folha, contexto.arvore)),
        })
    return pd.DataFrame(linhas, columns=COLUNAS_DOBRA)


def rodar_dobra(contexto: "Contexto", ids_treino: Iterable[str], ids_teste: Iterable[str],
                com_corpus: bool) -> pd.DataFrame:
    """Treina com `ids_treino` e devolve, por questão de teste, `COLUNAS_DOBRA`."""
    ids_treino, ids_teste = [str(i) for i in ids_treino], [str(i) for i in ids_teste]
    if set(ids_treino) & set(ids_teste):
        repetidas = sorted(set(ids_treino) & set(ids_teste))
        raise ValueError("questões de teste no treino da dobra: " + ", ".join(repetidas))
    rodada = treinar_rodada(contexto, ids_treino, com_corpus)
    aplicacao = rodada.aplicar(ids_teste)
    return resultados_teste(contexto, aplicacao, ids_teste)


# --- grade de limiares -------------------------------------------------------------------------------------


def precisao_jeffreys(acertos: np.ndarray | int, publicadas: np.ndarray | int) -> np.ndarray:
    acertos, publicadas = np.asarray(acertos, dtype=float), np.asarray(publicadas, dtype=float)
    return (acertos + 0.5) / (publicadas + 1.0)


def intervalo_beta(acertos, publicadas, sorteios: int = N_SORTEIOS, semente: int = SEMENTE,
                   confianca: float = CONFIANCA) -> tuple[np.ndarray, np.ndarray]:
    """Limites do intervalo de `confianca` da Beta(acertos + 0,5, erros + 0,5), por simulação.

    As contagens podem ser fracionárias (n efetivo, com repetições). Cada par (acertos, publicadas) distinto é
    sorteado com um gerador novo na mesma semente: pares iguais têm o mesmo intervalo em qualquer linha da grade,
    e o ruído da simulação não desempata limiares.
    """
    acertos = np.atleast_1d(np.asarray(acertos, dtype=float)).round(9)
    publicadas = np.atleast_1d(np.asarray(publicadas, dtype=float)).round(9)
    cauda = (1 - confianca) / 2
    limites: dict[tuple[float, float], tuple[float, float]] = {}
    for par in set(zip(acertos.tolist(), publicadas.tolist())):
        certos, total = par
        amostras = np.random.default_rng(semente).beta(certos + 0.5, max(total - certos, 0.0) + 0.5, size=sorteios)
        limites[par] = (float(np.quantile(amostras, cauda)), float(np.quantile(amostras, 1 - cauda)))
    pares = list(zip(acertos.tolist(), publicadas.tolist()))
    return np.array([limites[par][0] for par in pares]), np.array([limites[par][1] for par in pares])


def _como_tabela(resultados) -> pd.DataFrame:
    if isinstance(resultados, pd.DataFrame):
        return resultados
    resultados = [r for r in resultados if len(r)]
    if not resultados:
        return pd.DataFrame(columns=COLUNAS_DOBRA)
    return pd.concat(resultados, ignore_index=True)


def _linha_folha(linha: Mapping) -> dict:
    return {"fora": bool(linha["fora"]), "tema_central_id": linha["tema_central_id"],
            "extras_ids": list(linha["extras_ids"])}


def _ordenados(registros: list[dict]) -> list[dict]:
    """Cópia dos registros com candidatos e ramos em ordem decrescente de probabilidade (desempate pelo id)."""
    def ordenar(lista):
        return sorted(((int(t), float(p)) for t, p in lista), key=lambda par: (-par[1], par[0]))

    return [{**r, "candidatos_prob": ordenar(r["candidatos_prob"]), "ramos": ordenar(r["ramos"]),
             "opcoes": opcoes_de_tabela(r.get("opcoes") or [])} for r in registros]


def decidir_resultado(linha: Mapping, limiares: Limiares, arvore: "ArvoreTemas", com_extras: bool = True) -> Decisao:
    """A decisão de uma questão de teste (linha de `rodar_dobra`) com os limiares dados."""
    return decidir_listas(str(linha["id"]), linha["candidatos_prob"], linha["opcoes"], linha["p_fora"], limiares,
                          arvore, linha["gatilhos"], com_extras=com_extras, ordenados=True)


def _repeticoes(tabela: pd.DataFrame, repeticoes: int | None) -> int:
    if repeticoes is not None:
        return max(1, int(repeticoes))
    if "repeticao" in tabela and len(tabela):
        return max(1, int(tabela["repeticao"].nunique()))
    return 1


def grade(
    resultados_das_dobras,
    arvore: "ArvoreTemas",
    grade_publicar: Sequence[float] = GRADE_PUBLICAR,
    grade_fora: Sequence[float] = GRADE_FORA,
    semente: int = SEMENTE,
    repeticoes: int | None = None,
) -> pd.DataFrame:
    """Para cada par (publicar, fora): decisões das questões de teste agrupadas, acertos pela régua e intervalos.

    `publicar` é o limiar da confiança do segundo estágio. Com `repeticoes` (sem ele, o número de valores da coluna
    `repeticao`, ou 1), cada questão aparece uma vez por repetição: `publicadas_efetivas` e `acertos_efetivos`
    são as contagens divididas pelas repetições, e a precisão de Jeffreys e o intervalo Beta usam essas contagens.

    Colunas: `publicar, fora, n, publicadas, acertos, publicadas_efetivas, acertos_efetivos, revisadas,
    fora_decididas, cobertura, precisao, limite_inferior, limite_superior, fora_certos, fora_errados,
    fora_perdidos`.
    """
    tabela = _como_tabela(resultados_das_dobras)
    n_repeticoes = _repeticoes(tabela, repeticoes)
    registros = _ordenados(tabela.to_dict("records"))
    linhas_folha = [_linha_folha(r) for r in registros]
    acertou: dict[tuple[int, int], bool] = {}
    linhas: list[dict] = []
    for limiar_fora in grade_fora:
        for limiar in grade_publicar:
            limiares = Limiares(publicar=float(limiar), fora=float(limiar_fora))
            contagem = dict(publicadas=0, acertos=0, revisadas=0, fora_decididas=0, fora_certos=0, fora_errados=0,
                            fora_perdidos=0)
            for posicao, (registro, linha_folha) in enumerate(zip(registros, linhas_folha)):
                decisao = decidir_resultado(registro, limiares, arvore, com_extras=False)
                if decisao.decisao == PUBLICAR:
                    contagem["publicadas"] += 1
                    chave = (posicao, decisao.tema_central_id)
                    if chave not in acertou:
                        acertou[chave] = acerto_regua(decisao.tema_central_id, linha_folha, arvore)
                    contagem["acertos"] += acertou[chave]
                    contagem["fora_perdidos"] += linha_folha["fora"]
                elif decisao.decisao == FORA:
                    contagem["fora_decididas"] += 1
                    contagem["fora_certos"] += linha_folha["fora"]
                    contagem["fora_errados"] += not linha_folha["fora"]
                else:
                    contagem["revisadas"] += 1
            linhas.append({"publicar": float(limiar), "fora": float(limiar_fora), "n": len(registros), **contagem})
    resultado = pd.DataFrame(linhas)
    if resultado.empty:
        return resultado
    resultado["cobertura"] = resultado["publicadas"] / resultado["n"].where(resultado["n"] > 0, 1)
    resultado["publicadas_efetivas"] = resultado["publicadas"] / n_repeticoes
    resultado["acertos_efetivos"] = resultado["acertos"] / n_repeticoes
    resultado["precisao"] = precisao_jeffreys(resultado["acertos_efetivos"], resultado["publicadas_efetivas"])
    baixo, alto = intervalo_beta(resultado["acertos_efetivos"], resultado["publicadas_efetivas"], semente=semente)
    resultado["limite_inferior"], resultado["limite_superior"] = baixo, alto
    ordem = ["publicar", "fora", "n", "publicadas", "acertos", "publicadas_efetivas", "acertos_efetivos", "revisadas",
             "fora_decididas", "cobertura", "precisao", "limite_inferior", "limite_superior", "fora_certos",
             "fora_errados", "fora_perdidos"]
    return resultado[ordem]


# --- escada ------------------------------------------------------------------------------------------------


def escolher(tabela_grade: pd.DataFrame) -> tuple[Limiares, int]:
    """Par de limiares pela escada e o degrau usado.

    `precisao` é a precisão suavizada de Jeffreys sobre as contagens efetivas (uma por questão, não por repetição).
    Degrau 0: precisão ≥ 0,97 e limite inferior ≥ 0,95. Degrau 1: só limite inferior ≥ 0,95. Degrau 2: precisão
    ≥ 0,97 com pelo menos `N_EFETIVO_MINIMO` publicações efetivas. Degrau 3: precisão ≥ 0,95 (a régua) com pelo menos
    `N_EFETIVO_MINIMO` publicações efetivas. Em todos, maior cobertura. Degrau 4, quando nenhum limiar atinge a régua:
    o de maior precisão (o `previsao.txt` diz que a régua não foi atingida). Nenhum degrau aceita, para publicar,
    um limiar com precisão abaixo da régua só porque o intervalo é mais estreito; com 100 rótulos o limite inferior
    nunca chega a 0,95, e maximizá-lo empurrava para limiares com precisão de 0,91.
    Empates: menor limiar de fora, depois maior limiar de publicar (o mais conservador).
    """
    if tabela_grade.empty:
        raise ValueError("grade vazia")
    g = tabela_grade
    media_ok = g["precisao"] >= PRECISAO_ALVO - _TOLERANCIA
    regua_ok = g["precisao"] >= PRECISAO_REGUA - _TOLERANCIA
    inferior_ok = g["limite_inferior"] >= LIMITE_INFERIOR_ALVO - _TOLERANCIA
    if "publicadas_efetivas" in g:
        n_ok = g["publicadas_efetivas"] >= N_EFETIVO_MINIMO - _TOLERANCIA
    else:
        n_ok = pd.Series(True, index=g.index)
    ordem = (["cobertura", "fora", "publicar"], [False, True, False])
    for degrau, filtro in ((0, media_ok & inferior_ok), (1, inferior_ok), (2, media_ok & n_ok), (3, regua_ok & n_ok)):
        candidatos = g[filtro]
        if len(candidatos):
            melhor = candidatos.sort_values(ordem[0], ascending=ordem[1], kind="stable").iloc[0]
            return Limiares(publicar=float(melhor["publicar"]), fora=float(melhor["fora"])), degrau
    melhor = g.sort_values(["precisao", "cobertura", "fora", "publicar"],
                           ascending=[False, False, True, False], kind="stable").iloc[0]
    return Limiares(publicar=float(melhor["publicar"]), fora=float(melhor["fora"])), 4


def linha_da_grade(tabela_grade: pd.DataFrame, limiares: Limiares) -> dict:
    escolhida = tabela_grade[np.isclose(tabela_grade["publicar"], limiares.publicar)
                             & np.isclose(tabela_grade["fora"], limiares.fora)]
    if escolhida.empty:
        raise KeyError(f"limiares {limiares} fora da grade")
    return {chave: (valor.item() if hasattr(valor, "item") else valor)
            for chave, valor in escolhida.iloc[0].to_dict().items()}


def resumir_grade(tabela_grade: pd.DataFrame, limiares: Limiares, passo: float = 0.05) -> pd.DataFrame:
    """Linhas da grade no limiar de fora escolhido, de `passo` em `passo` no de publicar, mais a escolhida."""
    no_fora = tabela_grade[np.isclose(tabela_grade["fora"], limiares.fora)]
    multiplos = np.isclose((no_fora["publicar"] / passo).round() * passo, no_fora["publicar"])
    escolhida = np.isclose(no_fora["publicar"], limiares.publicar)
    return no_fora[multiplos | escolhida].reset_index(drop=True)


# --- validação ----------------------------------------------------------------------------------------------


def tipo_erro(tema_id: int | None, linha_folha: Mapping, arvore: "ArvoreTemas") -> str | None:
    """Tipo do erro de uma publicação errada: fora publicada, irmão (mesmo ramo de um alvo) ou outro ramo."""
    if bool(linha_folha["fora"]):
        return ERRO_FORA_PUBLICADA
    if acerto_regua(tema_id, linha_folha, arvore):
        return None
    verdadeiros = [linha_folha["tema_central_id"], *linha_folha["extras_ids"]]
    if any(t is not None and arvore.mesmo_ramo(int(tema_id), int(t)) for t in verdadeiros):
        return ERRO_IRMAO
    return ERRO_OUTRO_RAMO


def avaliar(resultados: pd.DataFrame, limiares: Limiares, arvore: "ArvoreTemas") -> pd.DataFrame:
    """Decisão, acerto e tipo de erro de cada questão de teste das dobras com os limiares escolhidos."""
    linhas: list[dict] = []
    for registro in _ordenados(resultados.to_dict("records")):
        decisao = decidir_resultado(registro, limiares, arvore)
        linha_folha = _linha_folha(registro)
        publicada = decisao.decisao == PUBLICAR
        if publicada:
            acerto = acerto_regua(decisao.tema_central_id, linha_folha, arvore)
        elif decisao.decisao == FORA:
            acerto = linha_folha["fora"]
        else:
            acerto = None
        linhas.append({
            "repeticao": registro.get("repeticao"),
            "dobra": registro.get("dobra"),
            "id": registro["id"],
            "decisao": decisao.decisao,
            "tema_central_id": decisao.tema_central_id,
            "extras_ids": ";".join(map(str, decisao.extras_ids)),
            "confianca": decisao.confianca,
            "confianca_tema": registro.get("confianca_tema"),
            "confianca_ramo": registro.get("confianca_ramo"),
            "votos": registro.get("votos"),
            "votos_ramo": registro.get("votos_ramo"),
            "acerto": acerto,
            "tipo_erro": tipo_erro(decisao.tema_central_id, linha_folha, arvore) if publicada and not acerto else None,
            "p_max": registro["p_max"],
            "p_fora": registro["p_fora"],
            "verdade_central": linha_folha["tema_central_id"],
            "verdade_extras": ";".join(map(str, linha_folha["extras_ids"])),
            "verdade_fora": linha_folha["fora"],
        })
    avaliacao = pd.DataFrame(linhas)
    for coluna in ("tema_central_id", "verdade_central"):
        if coluna in avaliacao:
            avaliacao[coluna] = pd.array([None if pd.isna(v) else int(v) for v in avaliacao[coluna]], dtype="Int64")
    return avaliacao


def matriz_erros(avaliacao: pd.DataFrame) -> dict[str, int]:
    erros = avaliacao["tipo_erro"].dropna() if len(avaliacao) else pd.Series([], dtype=object)
    return {tipo: int((erros == tipo).sum()) for tipo in TIPOS_ERRO}


def resumo_validacao(validacao: Mapping) -> dict:
    """Números de uma validação para comparar fontes e gravar em JSON."""
    escolhida = validacao["escolhida"]
    return {
        "com_corpus": bool(validacao["com_corpus"]),
        "publicar": float(validacao["limiares"].publicar),
        "fora": float(validacao["limiares"].fora),
        "degrau": int(validacao["degrau"]),
        "precisao": float(escolhida["precisao"]),
        "limite_inferior": float(escolhida["limite_inferior"]),
        "limite_superior": float(escolhida["limite_superior"]),
        "cobertura": float(escolhida["cobertura"]),
        "publicadas": int(escolhida["publicadas"]),
        "acertos": int(escolhida["acertos"]),
        "n": int(escolhida["n"]),
        "repeticoes": int(validacao["repeticoes"]),
        "n_efetivo": float(escolhida["publicadas_efetivas"]),
        "acertos_efetivos": float(escolhida["acertos_efetivos"]),
        "erros": dict(validacao["erros"]),
    }


def aquecer(contexto: "Contexto") -> None:
    """Calcula as chaves, o BM25 e os cossenos de todos os itens do contexto (as partes que não mudam entre as
    dobras), para os processos da validação receberem o contexto já com elas."""
    for id_item in contexto.vetores:
        contexto.chaves(id_item)
        contexto.pontuacoes_bm25(id_item)
        contexto.cossenos_temas(id_item)


_CONTEXTO_DO_PROCESSO: "Contexto | None" = None


def _instalar_contexto(dados: bytes) -> None:
    """Inicializador de cada processo da validação: o contexto chega uma vez, já serializado."""
    global _CONTEXTO_DO_PROCESSO
    _CONTEXTO_DO_PROCESSO = pickle.loads(dados)


def _rodar_dobra_no_processo(ids_treino: list[str], ids_teste: list[str], com_corpus: bool) -> pd.DataFrame:
    return rodar_dobra(_CONTEXTO_DO_PROCESSO, ids_treino, ids_teste, com_corpus)


def numero_de_processos(n_tarefas: int | None, n_dobras: int) -> int:
    """Processos para `n_dobras` dobras: `n_tarefas` (None = `N_TAREFAS`; negativo conta a partir do total de
    núcleos, como no joblib: -1 = todos), no máximo um por dobra e pelo menos 1."""
    n_tarefas = N_TAREFAS if n_tarefas is None else int(n_tarefas)
    if n_tarefas < 0:
        n_tarefas = max(1, (os.cpu_count() or 1) + 1 + n_tarefas)
    return max(1, min(n_tarefas, n_dobras))


def rodar_dobras(contexto: "Contexto", lista_dobras: Sequence[tuple[list[str], list[str]]], com_corpus: bool,
                 n_tarefas: int | None = None) -> list[pd.DataFrame]:
    """`rodar_dobra` de cada (treino, teste), na ordem dada; com mais de um processo, em paralelo."""
    lista_dobras = list(lista_dobras)
    processos = numero_de_processos(n_tarefas, len(lista_dobras))
    if processos > 1:
        aquecer(contexto)
        enviado = copy.copy(contexto)
        enviado.cache = None
        dados = pickle.dumps(enviado, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            with ProcessPoolExecutor(max_workers=processos, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_instalar_contexto, initargs=(dados,)) as executor:
                return list(executor.map(_rodar_dobra_no_processo, [t for t, _ in lista_dobras],
                                         [t for _, t in lista_dobras], [com_corpus] * len(lista_dobras)))
        except BrokenProcessPool as erro:
            registro.warning("processos da validação quebraram (%s); as dobras rodam em série", erro)
    return [rodar_dobra(contexto, treino, teste, com_corpus) for treino, teste in lista_dobras]


def validar(contexto: "Contexto", com_corpus: bool, n_dobras: int = N_DOBRAS, repeticoes: int = REPETICOES,
            semente: int = SEMENTE, n_tarefas: int | None = None) -> dict:
    """Validação cruzada na folha: dobras, grade, escolha pela escada, previsão e matriz de tipos de erro."""
    lista_dobras = dobras(contexto.folha, n_dobras, repeticoes, semente, arvore=contexto.arvore)
    partes: list[pd.DataFrame] = []
    registro_dobras: list[dict] = []
    por_dobra = rodar_dobras(contexto, lista_dobras, com_corpus, n_tarefas)
    for numero, ((treino, teste), resultado) in enumerate(zip(lista_dobras, por_dobra)):
        repeticao, dobra = divmod(numero, n_dobras)
        resultado.insert(0, "dobra", dobra)
        resultado.insert(0, "repeticao", repeticao)
        partes.append(resultado)
        registro_dobras.append({"repeticao": repeticao, "dobra": dobra, "treino": treino, "teste": teste})
    resultados = _como_tabela(partes)
    tabela_grade = grade(resultados, contexto.arvore, semente=semente, repeticoes=repeticoes)
    limiares, degrau = escolher(tabela_grade)
    escolhida = linha_da_grade(tabela_grade, limiares)
    avaliacao = avaliar(resultados, limiares, contexto.arvore)
    validacao = {
        "com_corpus": bool(com_corpus),
        "n_dobras": n_dobras,
        "repeticoes": repeticoes,
        "dobras": registro_dobras,
        "resultados": resultados,
        "grade": tabela_grade,
        "limiares": limiares,
        "degrau": degrau,
        "escolhida": escolhida,
        "precisao": float(escolhida["precisao"]),
        "precisao_intervalo": [float(escolhida["limite_inferior"]), float(escolhida["limite_superior"])],
        "cobertura": float(escolhida["cobertura"]),
        "n_efetivo": float(escolhida["publicadas_efetivas"]),
        "avaliacao": avaliacao,
        "erros": matriz_erros(avaliacao),
    }
    validacao["resumo"] = resumo_validacao(validacao)
    return validacao


# --- comparação de fontes ----------------------------------------------------------------------------------


def regra_tres_pontos(sem: Mapping, com: Mapping) -> tuple[bool, str]:
    """Corpus aprovado se a cobertura sobe pelo menos 0,03 sem baixar a precisão prevista nem o limite inferior."""
    ganho = float(com["cobertura"]) - float(sem["cobertura"])
    problemas = []
    if ganho < GANHO_MINIMO_CORPUS - _TOLERANCIA:
        problemas.append(f"ganho de cobertura de {100 * ganho:+.1f} pontos, menos que 3".replace(".", ","))
    if float(com["precisao"]) < float(sem["precisao"]) - _TOLERANCIA:
        problemas.append("precisão prevista menor")
    if float(com["limite_inferior"]) < float(sem["limite_inferior"]) - _TOLERANCIA:
        problemas.append("limite inferior menor")
    if problemas:
        return False, "; ".join(problemas)
    return True, (f"ganho de cobertura de {100 * ganho:+.1f} pontos sem perder precisão nem limite "
                  "inferior").replace(".", ",")


def comparar_fontes(ctx_sem: "Contexto", ctx_com: "Contexto", **opcoes) -> dict:
    """Validação sem (A) e com (B) o corpus da etapa 1 e a regra dos 3 pontos.

    `validacoes` guarda as duas validações completas (para `tudo` não repetir a da fonte escolhida); o resto do
    dicionário vai para `comparacao_fontes.json`.
    """
    sem = validar(ctx_sem, com_corpus=False, **opcoes)
    com = validar(ctx_com, com_corpus=True, **opcoes)
    aprovado, motivo = regra_tres_pontos(sem["resumo"], com["resumo"])
    return {
        "regra": "o corpus fica se a cobertura prevista subir pelo menos 3 pontos com precisão prevista e limite "
                 "inferior não menores que os da versão sem corpus",
        "sem_corpus": sem["resumo"],
        "com_corpus": com["resumo"],
        "ganho_cobertura": com["resumo"]["cobertura"] - sem["resumo"]["cobertura"],
        "corpus_aprovado": aprovado,
        "motivo": motivo,
        "validacoes": {"sem_corpus": sem, "com_corpus": com},
    }
