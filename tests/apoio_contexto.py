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


# --- contexto maior, para modelo, calibração e decisão ---------------------------------------------------

PALAVRAS = {
    PATRIMONIO: "crimes contra o patrimônio em geral",
    FURTO: "furto subtração de coisa alheia móvel",
    ROUBO: "roubo com violência ou grave ameaça",
    EXTORSAO: "extorsão constrangimento para obter vantagem",
    ESTELIONATO: "estelionato fraude induzindo alguém a erro",
    MARIA_DA_PENHA: "violência doméstica contra a mulher, Lei Maria da Penha",
    DROGAS: "tráfico de drogas e entorpecentes",
}
DISCIPLINA_ORIGEM = {
    PATRIMONIO: "Direito Penal", FURTO: "Direito Penal", ROUBO: "Direito Penal", EXTORSAO: "Direito Penal",
    ESTELIONATO: "Direito Penal", MARIA_DA_PENHA: "Direito Penal - Legislação Especial",
    DROGAS: "Direito Penal - Legislação Especial",
}
# Questões da folha por tema central (None = fora) e as que ficam fora da folha
FOLHA_MAIOR = {FURTO: 6, ROUBO: 6, EXTORSAO: 5, ESTELIONATO: 5, PATRIMONIO: 3, MARIA_DA_PENHA: 5, DROGAS: 5, None: 5}
FORA_DA_FOLHA_MAIOR = {FURTO: 2, ROUBO: 2, EXTORSAO: 2, ESTELIONATO: 2, MARIA_DA_PENHA: 2, DROGAS: 2, None: 2}
TEXTO_FORA = "crime militar de deserção previsto no Código Penal Militar, julgado pela justiça militar"


def _vetor_ruidoso(gerador: np.random.Generator, tema: int | None, ruido: float) -> np.ndarray:
    base = np.zeros(DIMENSAO, dtype=np.float32)
    if tema is None:
        base[FORA] = 1.0
    else:
        base[EIXOS[tema]] = 1.0
        if tema == PATRIMONIO:
            base[EIXOS[FURTO]] = 0.3
    return (base + gerador.normal(0.0, ruido, DIMENSAO)).astype(np.float32)


def dados_maiores(semente: int = 7, ruido: float = 0.25) -> dict:
    """Questões, folha, exemplos e corpus do contexto maior, com texto, disciplina e vetor de cada item.

    Temas da folha: os cinco do patrimônio (Penal) e Maria da Penha e Drogas (Legislação Especial); 5 fora (crime
    militar). Duas questões de roubo têm furto como extra. Exemplos: 3 por tema e 6 fora. Corpus: 1 por tema.
    """
    gerador = np.random.default_rng(semente)
    questoes, folha = [], []
    numero = 0

    def nova(tema, na_folha, extras=()):
        nonlocal numero
        numero += 1
        id_q = f"Q{numero:04d}"
        if tema is None:
            texto = f"{TEXTO_FORA}. Caso {numero}."
            disciplina = "Direito Penal Militar" if numero % 2 else "Direito Penal"
        else:
            texto = f"Questão sobre {PALAVRAS[tema]}. Caso {numero}."
            disciplina = DISCIPLINA_ORIGEM[tema]
        questoes.append({"id": id_q, "texto": texto, "disciplina_origem": disciplina,
                         "vetor": _vetor_ruidoso(gerador, tema, ruido)})
        if na_folha:
            folha.append({"id": id_q, "tema_central_id": tema, "extras_ids": list(extras), "fora": tema is None})

    for tema, n in FOLHA_MAIOR.items():
        for i in range(n):
            nova(tema, True, extras=(FURTO,) if tema == ROUBO and i < 2 else ())
    for tema, n in FORA_DA_FOLHA_MAIOR.items():
        for _ in range(n):
            nova(tema, False)

    exemplos, corpus = [], []
    for tema in [*EIXOS, None, None]:
        for i in range(3):
            id_e = f"S{len(exemplos) + 1:04d}"
            texto = (f"Exemplo sintético {id_e} de {TEXTO_FORA}" if tema is None
                     else f"Exemplo sintético {id_e}: {PALAVRAS.get(tema, 'disposições gerais do patrimônio')}")
            exemplos.append({"id": id_e, "tema_id": tema, "fora": tema is None, "texto": texto, "fonte": "teste",
                             "vetor": _vetor_ruidoso(gerador, tema, ruido)})
    for tema in PALAVRAS:
        id_c = f"C{len(corpus) + 1:04d}"
        corpus.append({"id": id_c, "tema_id": tema, "texto": f"Questão do corpus {id_c} sobre {PALAVRAS[tema]}",
                       "vetor": _vetor_ruidoso(gerador, tema, ruido)})
    return {"questoes": questoes, "folha": folha, "exemplos": exemplos, "corpus": corpus}


def montar_maior(arvore: ArvoreTemas | None = None, com_corpus: bool = True, semente: int = 7,
                 ruido: float = 0.25) -> Contexto:
    """Contexto com 40 questões na folha (35 de 7 temas e 5 fora), 14 fora dela, 33 exemplos e 7 do corpus."""
    arvore = arvore or ArvoreTemas.carregar()
    dados = dados_maiores(semente, ruido)
    temas_ids = sorted(EIXOS)
    matriz_temas = np.vstack([np.eye(DIMENSAO, dtype=np.float32)[EIXOS[t]] for t in temas_ids])
    questoes = pd.DataFrame([{k: q[k] for k in ("id", "texto", "disciplina_origem")} for q in dados["questoes"]])
    folha = pd.DataFrame(dados["folha"], columns=["id", "tema_central_id", "extras_ids", "fora"])
    folha["tema_central_id"] = pd.Series([f["tema_central_id"] for f in dados["folha"]], dtype=object)
    vetores = {q["id"]: q["vetor"] for q in dados["questoes"]}
    vetores.update({e["id"]: e["vetor"] for e in dados["exemplos"]})
    corpus = []
    if com_corpus:
        corpus = [{k: v for k, v in c.items() if k != "vetor"} for c in dados["corpus"]]
        vetores.update({c["id"]: c["vetor"] for c in dados["corpus"]})
    return Contexto(
        arvore=arvore,
        enriquecidos={},
        dicionario=DICIONARIO,
        termos_fora=["militar", "código penal militar", "direito tributário"],
        questoes=questoes,
        folha=folha,
        exemplos=[{k: v for k, v in e.items() if k != "vetor"} for e in dados["exemplos"]],
        corpus=corpus,
        cache=None,
        temas_ids=temas_ids,
        textos_temas=[arvore.texto_tema(t, {}) for t in temas_ids],
        matriz_temas=matriz_temas,
        vetores=vetores,
    )
