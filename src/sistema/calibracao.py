"""Validação cruzada na folha-verdade, grade de limiares, escada e comparação de fontes (spec, seção 8).

Uma rodada (`treinar_rodada`) usa só os rótulos de treino que recebe: monta o índice kNN com a folha de treino, os
exemplos sintéticos e, se ligado, o corpus; gera candidatos e características desses itens; treina o modelo de
pares (folha de treino com peso 1, exemplos com tema e corpus com peso 0,5) e o detector de fora (folha de treino
com peso 1, exemplos fora como positivos e exemplos com tema como negativos, peso 0,5). `rodar_dobra` aplica a
rodada nas questões de teste da dobra, que não entram em nada do treino.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from sistema.candidatos import Candidato, Vizinho, gerar_lote
from sistema.caracteristicas import normalizar_disciplina, tabela_pares
from sistema.config import SEMENTE
from sistema.dados import acerto_regua, alvos_aceitos
from sistema.decisao import FORA, PUBLICAR, Decisao, Limiares, decidir_listas
from sistema.modelo import (
    DetectorFora,
    ModeloPares,
    agregar_ramo,
    gatilhos_fora,
    pesos_por_linha,
    rotulos_pares,
    tabela_questoes,
    temas_positivos,
)

if TYPE_CHECKING:
    from sistema.arvore import ArvoreTemas
    from sistema.contexto import Contexto

N_DOBRAS = 5
REPETICOES = 3
GRADE_PUBLICAR = tuple(round(0.30 + 0.01 * i, 2) for i in range(66))  # 0,30 a 0,95
GRADE_FORA = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_SORTEIOS = 10_000
CONFIANCA = 0.95

PESO_FOLHA = 1.0
PESO_EXEMPLO = 0.5
PESO_CORPUS = 0.5

PRECISAO_ALVO = 0.97
LIMITE_INFERIOR_ALVO = 0.95
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
    """Tudo o que uma rodada calcula para um conjunto de ids."""

    ids: list[str]
    candidatos: dict[str, list[Candidato]]
    vizinhos: dict[str, list[Vizinho]]
    pares: pd.DataFrame
    prob: pd.DataFrame
    ramos: pd.DataFrame
    tabela_q: pd.DataFrame
    p_fora: pd.Series

    def listas(self, id_item: str) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """(tema, p) e (pai, p_ramo) da questão, em ordem decrescente."""
        return self._por_id_prob.get(id_item, []), self._por_id_ramo.get(id_item, [])

    def __post_init__(self) -> None:
        self._por_id_prob = _agrupar(self.prob, "p")
        self._por_id_ramo = _agrupar(self.ramos, "p_ramo")


def _agrupar(tabela: pd.DataFrame, coluna: str) -> dict[str, list[tuple[int, float]]]:
    grupos: dict[str, list[tuple[int, float]]] = {}
    for id_q, tema_id, valor in zip(tabela["id"], tabela["tema_id"], tabela[coluna]):
        grupos.setdefault(str(id_q), []).append((int(tema_id), float(valor)))
    return {id_q: sorted(lista, key=lambda par: (-par[1], par[0])) for id_q, lista in grupos.items()}


@dataclass
class Rodada:
    """Índice, modelo de pares e detector treinados com `ids_treino` da folha (mais exemplos e corpus)."""

    contexto: "Contexto"
    com_corpus: bool
    ids_treino: list[str]
    indice: object
    pesos_vizinhos: dict[Hashable, float]
    modelo: ModeloPares
    detector: DetectorFora
    ids_pares: list[str] = field(default_factory=list)
    ids_detector: list[str] = field(default_factory=list)

    def aplicar(self, ids: Iterable[str]) -> Aplicacao:
        """Candidatos, características, `p`, `p_ramo` e `p_fora` dos ids, com o índice e os modelos da rodada."""
        return _aplicar(self.contexto, list(ids), self.indice, self.pesos_vizinhos, self.ids_treino, self.modelo,
                        self.detector)


def _aplicar(
    contexto: "Contexto",
    ids: list[str],
    indice,
    pesos_vizinhos: Mapping[Hashable, float],
    ids_treino: Sequence[str],
    modelo: ModeloPares,
    detector: DetectorFora,
) -> Aplicacao:
    candidatos, vizinhos = gerar_lote(ids, contexto, indice, pesos_vizinhos)
    pares = tabela_pares(ids, candidatos, contexto, indice, pesos_vizinhos, set(ids_treino), vizinhos_por_id=vizinhos)
    prob = modelo.probabilidades(pares)
    ramos = agregar_ramo(prob, contexto.arvore)
    tabela_q = tabela_questoes(ids, prob, pares, vizinhos, contexto)
    p_fora = detector.p_fora(tabela_q)
    return Aplicacao(ids, candidatos, vizinhos, pares, prob, ramos, tabela_q, p_fora)


def treinar_rodada(contexto: "Contexto", ids_treino: Iterable[str], com_corpus: bool) -> Rodada:
    """Treina uma rodada só com os rótulos de `ids_treino` (folha), os exemplos e, com `com_corpus`, o corpus."""
    ids_treino = [str(i) for i in ids_treino]
    indice, pesos_vizinhos = contexto.indice_rotulado(ids_treino, com_corpus=com_corpus)
    exemplos_tema = [str(e["id"]) for e in contexto.exemplos if e["tema_id"] is not None]
    exemplos_fora = [str(e["id"]) for e in contexto.exemplos if e["tema_id"] is None]
    corpus = [str(c["id"]) for c in contexto.corpus] if com_corpus else []

    peso_pares = {**{i: PESO_FOLHA for i in ids_treino}, **{i: PESO_EXEMPLO for i in exemplos_tema},
                  **{i: PESO_CORPUS for i in corpus}}
    ids_pares = [*ids_treino, *exemplos_tema, *corpus]
    ids_detector = [*ids_treino, *exemplos_fora, *exemplos_tema]
    ids_todos = list(dict.fromkeys([*ids_pares, *ids_detector]))

    modelo = ModeloPares()
    candidatos, vizinhos = gerar_lote(ids_todos, contexto, indice, pesos_vizinhos)
    pares_todos = tabela_pares(ids_todos, candidatos, contexto, indice, pesos_vizinhos, set(ids_treino),
                               vizinhos_por_id=vizinhos)
    pares = pares_todos[pares_todos["id"].isin(set(ids_pares))].reset_index(drop=True)
    positivos = {i: temas_positivos(contexto, i) for i in ids_pares}
    modelo.treinar(pares, rotulos_pares(pares, positivos), pesos_por_linha(pares, peso_pares))

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


def treinar_final(contexto: "Contexto", com_corpus: bool) -> tuple[ModeloPares, DetectorFora]:
    """Modelo de pares e detector treinados com os 100 rótulos da folha (mais exemplos e corpus se ligado)."""
    rodada = treinar_rodada(contexto, contexto.ids_folha, com_corpus)
    return rodada.modelo, rodada.detector


COLUNAS_DOBRA = [
    "id", "p_max", "tema_max", "p_ramo_max", "pai_max", "p_fora", "candidatos_prob", "ramos", "gatilhos",
    "fora", "tema_central_id", "extras_ids", "alvos",
]


def resultados_teste(contexto: "Contexto", aplicacao: Aplicacao, ids_teste: Iterable[str]) -> pd.DataFrame:
    """Uma linha por questão de teste com o que a decisão precisa e os alvos aceitos pela régua."""
    linhas: list[dict] = []
    for id_q in ids_teste:
        temas, ramos = aplicacao.listas(id_q)
        linha_folha = contexto.linha_folha(id_q)
        linhas.append({
            "id": id_q,
            "p_max": temas[0][1] if temas else 0.0,
            "tema_max": temas[0][0] if temas else None,
            "p_ramo_max": ramos[0][1] if ramos else 0.0,
            "pai_max": ramos[0][0] if ramos else None,
            "p_fora": float(aplicacao.p_fora.get(id_q, 0.0)),
            "candidatos_prob": temas,
            "ramos": ramos,
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

    Cada par (acertos, publicadas) distinto é sorteado com um gerador novo na mesma semente: pares iguais têm o
    mesmo intervalo em qualquer linha da grade, e o ruído da simulação não desempata limiares.
    """
    acertos = np.atleast_1d(np.asarray(acertos, dtype=int))
    publicadas = np.atleast_1d(np.asarray(publicadas, dtype=int))
    cauda = (1 - confianca) / 2
    limites: dict[tuple[int, int], tuple[float, float]] = {}
    for par in set(zip(acertos.tolist(), publicadas.tolist())):
        certos, total = par
        amostras = np.random.default_rng(semente).beta(certos + 0.5, total - certos + 0.5, size=sorteios)
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

    return [{**r, "candidatos_prob": ordenar(r["candidatos_prob"]), "ramos": ordenar(r["ramos"])} for r in registros]


def decidir_resultado(linha: Mapping, limiares: Limiares, arvore: "ArvoreTemas", com_extras: bool = True) -> Decisao:
    """A decisão de uma questão de teste (linha de `rodar_dobra`) com os limiares dados."""
    return decidir_listas(str(linha["id"]), linha["candidatos_prob"], linha["ramos"], linha["p_fora"], limiares,
                          arvore, linha["gatilhos"], com_extras=com_extras, ordenados=True)


def grade(
    resultados_das_dobras,
    arvore: "ArvoreTemas",
    grade_publicar: Sequence[float] = GRADE_PUBLICAR,
    grade_fora: Sequence[float] = GRADE_FORA,
    semente: int = SEMENTE,
) -> pd.DataFrame:
    """Para cada par (publicar, fora): decisões das questões de teste agrupadas, acertos pela régua e intervalos.

    Colunas: `publicar, fora, n, publicadas, acertos, revisadas, fora_decididas, cobertura, precisao,
    limite_inferior, limite_superior, fora_certos, fora_errados, fora_perdidos`.
    """
    tabela = _como_tabela(resultados_das_dobras)
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
    resultado["precisao"] = precisao_jeffreys(resultado["acertos"], resultado["publicadas"])
    baixo, alto = intervalo_beta(resultado["acertos"], resultado["publicadas"], semente=semente)
    resultado["limite_inferior"], resultado["limite_superior"] = baixo, alto
    ordem = ["publicar", "fora", "n", "publicadas", "acertos", "revisadas", "fora_decididas", "cobertura", "precisao",
             "limite_inferior", "limite_superior", "fora_certos", "fora_errados", "fora_perdidos"]
    return resultado[ordem]


# --- escada ------------------------------------------------------------------------------------------------


def escolher(tabela_grade: pd.DataFrame) -> tuple[Limiares, int]:
    """Par de limiares pela escada e o degrau usado.

    Degrau 0: precisão ≥ 0,97 e limite inferior ≥ 0,95, maior cobertura. Degrau 1: só limite inferior ≥ 0,95,
    maior cobertura. Degrau 2: só precisão ≥ 0,97, maior cobertura. Degrau 3: maior limite inferior, desempate pela
    maior cobertura. Empate depois disso: menor limiar de fora, depois maior limiar de publicar (o mais
    conservador entre os que publicam as mesmas questões).
    """
    if tabela_grade.empty:
        raise ValueError("grade vazia")
    g = tabela_grade
    media_ok = g["precisao"] >= PRECISAO_ALVO - _TOLERANCIA
    inferior_ok = g["limite_inferior"] >= LIMITE_INFERIOR_ALVO - _TOLERANCIA
    for degrau, filtro in ((0, media_ok & inferior_ok), (1, inferior_ok), (2, media_ok)):
        candidatos = g[filtro]
        if len(candidatos):
            melhor = candidatos.sort_values(["cobertura", "fora", "publicar"], ascending=[False, True, False],
                                            kind="stable").iloc[0]
            return Limiares(publicar=float(melhor["publicar"]), fora=float(melhor["fora"])), degrau
    melhor = g.sort_values(["limite_inferior", "cobertura", "fora", "publicar"],
                           ascending=[False, False, True, False], kind="stable").iloc[0]
    return Limiares(publicar=float(melhor["publicar"]), fora=float(melhor["fora"])), 3


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
        "erros": dict(validacao["erros"]),
    }


def validar(contexto: "Contexto", com_corpus: bool, n_dobras: int = N_DOBRAS, repeticoes: int = REPETICOES,
            semente: int = SEMENTE) -> dict:
    """Validação cruzada na folha: dobras, grade, escolha pela escada, previsão e matriz de tipos de erro."""
    lista_dobras = dobras(contexto.folha, n_dobras, repeticoes, semente, arvore=contexto.arvore)
    partes: list[pd.DataFrame] = []
    registro_dobras: list[dict] = []
    for numero, (treino, teste) in enumerate(lista_dobras):
        repeticao, dobra = divmod(numero, n_dobras)
        resultado = rodar_dobra(contexto, treino, teste, com_corpus)
        resultado.insert(0, "dobra", dobra)
        resultado.insert(0, "repeticao", repeticao)
        partes.append(resultado)
        registro_dobras.append({"repeticao": repeticao, "dobra": dobra, "treino": treino, "teste": teste})
    resultados = _como_tabela(partes)
    tabela_grade = grade(resultados, contexto.arvore, semente=semente)
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
