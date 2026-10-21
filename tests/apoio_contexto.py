"""Contexto pequeno montado à mão, com a árvore real (offline) e vetores escolhidos, para candidatos e características.

Cada tema do contexto tem um eixo próprio; o eixo FORA não pertence a tema nenhum. O cosseno entre uma questão e
um tema sai direto das coordenadas da questão.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sistema.arvore import ArvoreTemas
from sistema.contexto import Contexto

PATRIMONIO = 5157  # Penal > Dos Crimes Contra o Patrimônio (nível 1)
DISPOSICOES = 5158
FURTO = 5159
ROUBO = 5160
EXTORSAO = 5161
ESTELIONATO = 5166
MARIA_DA_PENHA = 4704  # Legislação Especial > Lei Maria da Penha (nível 1, sem filhos)
MARIA_DA_PENHA_DUP = 6724  # Penal > Lei Maria da Penha (duplicado de 4704)
DROGAS = 4708

EIXOS = {
    PATRIMONIO: 0, FURTO: 1, ROUBO: 2, EXTORSAO: 3, ESTELIONATO: 4, DISPOSICOES: 5,
    MARIA_DA_PENHA: 6, MARIA_DA_PENHA_DUP: 7, DROGAS: 8,
}
FORA = 9
DIMENSAO = 10

DICIONARIO = {
    "art:155:cp": {"nome": "CP, art. 155", "temas": {str(FURTO): 1.0}, "fora": False},
    "art:157:cp": {"nome": "CP, art. 157", "temas": {str(ROUBO): 1.0}, "fora": False},
    "diploma:cp": {"nome": "Código Penal", "temas": {str(PATRIMONIO): 0.25, str(FURTO): 0.25}, "fora": False},
    "lei:11340": {"nome": "Lei Maria da Penha", "temas": {str(MARIA_DA_PENHA): 1.0, str(MARIA_DA_PENHA_DUP): 0.3},
                  "fora": False},
    "lei:9999": {"nome": "lei de teste do patrimônio", "temas": {str(PATRIMONIO): 0.5}, "fora": False},
    "lei:1": {"nome": "lei que aponta para a raiz", "temas": {"5062": 1.0}, "fora": False},
    "diploma:cpm": {"nome": "Código Penal Militar", "temas": {}, "fora": True},
}


def vetor(**coordenadas: float) -> np.ndarray:
    """vetor(furto=1, roubo=0.5, fora=0.1): coordenadas pelo nome do tema (ou `fora`, o eixo sem tema)."""
    nomes = {
        "patrimonio": PATRIMONIO, "furto": FURTO, "roubo": ROUBO, "extorsao": EXTORSAO, "estelionato": ESTELIONATO,
        "disposicoes": DISPOSICOES, "penha": MARIA_DA_PENHA, "penha_dup": MARIA_DA_PENHA_DUP, "drogas": DROGAS,
    }
    v = np.zeros(DIMENSAO, dtype=np.float32)
    for nome, valor in coordenadas.items():
        v[FORA if nome == "fora" else EIXOS[nomes[nome]]] = valor
    return v


QUESTOES = [
    # id, texto, disciplina_origem, vetor
    ("Q0001", "Tício subtraiu coisa alheia móvel durante o repouso noturno, nos termos do art. 155 do CP. "
              "Também incide a Lei 11.340/2006?", "Direito Penal", vetor(furto=1, roubo=0.5)),
    ("Q0002", "Furto de energia elétrica.", "Penal", vetor(furto=1, fora=0.1)),
    ("Q0003", "Roubo com emprego de arma.", "Direito Penal", vetor(roubo=1, furto=0.2)),
    ("Q0004", "Crime militar de deserção, art. 187 do Código Penal Militar.", "Direito Penal Militar", vetor(fora=1)),
    ("Q0005", "Violência doméstica e familiar contra a mulher.", "Direito Penal - Legislação Especial",
     vetor(penha=1, penha_dup=0.3)),
]
FOLHA = [
    # id, central, extras, fora
    ("Q0002", FURTO, [], False),
    ("Q0003", ROUBO, [], False),
    ("Q0004", None, [], True),
]
EXEMPLOS = [
    {"id": "S0001", "tema_id": FURTO, "fora": False, "texto": "exemplo sintético de furto",
     "vetor": vetor(furto=1, patrimonio=0.3)},
    {"id": "S0002", "tema_id": None, "fora": True, "texto": "exemplo fora da taxonomia",
     "vetor": vetor(fora=1, furto=0.5)},
    {"id": "S0003", "tema_id": ESTELIONATO, "fora": False, "texto": "exemplo sintético de estelionato",
     "vetor": vetor(estelionato=1)},
]
CORPUS = [{"id": "C0001", "tema_id": FURTO, "texto": "questão do corpus sobre furto", "vetor": vetor(furto=1, roubo=0.2)}]


def montar(arvore: ArvoreTemas | None = None, com_corpus: bool = True) -> Contexto:
    arvore = arvore or ArvoreTemas.carregar()
    temas_ids = sorted(EIXOS)
    matriz_temas = np.vstack([np.eye(DIMENSAO, dtype=np.float32)[EIXOS[t]] for t in temas_ids])
    questoes = pd.DataFrame(
        [{"id": i, "texto": t, "disciplina_origem": d} for i, t, d, _ in QUESTOES]
    )
    folha = pd.DataFrame([{"id": i, "tema_central_id": c, "extras_ids": e, "fora": f} for i, c, e, f in FOLHA])
    folha["tema_central_id"] = pd.Series([c for _, c, _, _ in FOLHA], dtype=object)
    vetores = {i: v for i, _, _, v in QUESTOES}
    exemplos = [{k: v for k, v in e.items() if k != "vetor"} for e in EXEMPLOS]
    vetores.update({e["id"]: e["vetor"] for e in EXEMPLOS})
    corpus = []
    if com_corpus:
        corpus = [{k: v for k, v in c.items() if k != "vetor"} for c in CORPUS]
        vetores.update({c["id"]: c["vetor"] for c in CORPUS})
    return Contexto(
        arvore=arvore,
        enriquecidos={},
        dicionario=DICIONARIO,
        termos_fora=["militar"],
        questoes=questoes,
        folha=folha,
        exemplos=exemplos,
        corpus=corpus,
        cache=None,
        temas_ids=temas_ids,
        textos_temas=[arvore.texto_tema(t, {}) for t in temas_ids],
        matriz_temas=matriz_temas,
        vetores=vetores,
    )
