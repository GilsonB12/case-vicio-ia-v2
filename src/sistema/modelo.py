"""Modelo de pares (questão, tema), agregação pelo pai e detector de fora da taxonomia (spec, seção 7).

- `ModeloPares`: regressão logística sobre as colunas `CARACTERISTICAS` padronizadas; o C sai de uma validação
  cruzada interna agrupada por questão. Devolve `s` (probabilidade do par) e `p = s / Σ s` entre os candidatos da
  questão, que soma 1 e é comparável entre questões.
- `agregar_ramo`: `p_ramo(pai) = p_pai + Σ p dos filhos candidatos`, para os temas de nível 1 candidatos.
- `DetectorFora`: regressão logística separada sobre características da questão (`CARACTERISTICAS_FORA`, montadas
  por `tabela_questoes`), calibrada com Platt quando há exemplos das duas classes em número suficiente.
- `ModeloConfianca`: segundo estágio. Para cada questão, duas opções de publicação, o top-1 do modelo de pares
  (`tema`) e o ramo de nível 1 dele (`ramo`, só quando o top-1 é de nível 2), com `CARACTERISTICAS_CONFIANCA`
  montadas por `tabela_opcoes`; a regressão logística dá a probabilidade de a opção estar certa pela régua.

Tudo determinístico: sementes fixas e solver sem sorteio.
"""
from __future__ import annotations

import math
import re
import unicodedata
import weakref
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from sistema.caracteristicas import CARACTERISTICAS, normalizar_disciplina
from sistema.config import SEMENTE
from sistema.decisao import OPCAO_RAMO, OPCAO_TEMA
from sistema.dicionario import pontuar
from sistema.leis import peso_fora as calcular_peso_fora

if TYPE_CHECKING:
    from sistema.arvore import ArvoreTemas
    from sistema.candidatos import Vizinho
    from sistema.contexto import Contexto

GRADE_C = (0.1, 0.3, 1.0, 3.0)
C_PADRAO = 1.0
N_DOBRAS_C = 3
MAX_ITER = 2000

COLUNAS_PROBABILIDADE = ["id", "tema_id", "s", "p"]
COLUNAS_RAMO = ["id", "tema_id", "p_ramo"]


def _regressao(C: float = C_PADRAO) -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=MAX_ITER, C=C)


def _validar_tamanhos(tabela: pd.DataFrame, rotulos: Sequence, pesos: Sequence) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(rotulos, dtype=int).ravel()
    w = np.asarray(pesos, dtype=float).ravel()
    if len(y) != len(tabela) or len(w) != len(tabela):
        raise ValueError(f"{len(tabela)} linhas, {len(y)} rótulos e {len(w)} pesos: precisam ser iguais")
    if len(tabela) and not set(np.unique(y)) <= {0, 1}:
        raise ValueError("rótulos precisam ser 0 ou 1")
    return y, w


# --- pares (questão, tema) -------------------------------------------------------------------------------


class ModeloPares:
    """Regressão logística dos pares (questão, tema candidato): positivo = o tema é o central ou um extra.

    `class_weight="balanced"`, `max_iter=2000`; o `StandardScaler` é ajustado no treino e guardado. O C sai de
    `grade_c` por validação cruzada de 3 dobras agrupada por questão (`GroupKFold` embaralhado com semente fixa),
    pela menor log-loss ponderada nas dobras de validação; empate fica com o C menor. Sem as duas classes no
    treino, o modelo vira uma constante (a fração ponderada de positivos).
    """

    def __init__(self, grade_c: Iterable[float] = GRADE_C, semente: int = SEMENTE) -> None:
        self.grade_c = tuple(float(c) for c in grade_c)
        self.semente = semente
        self.escalador: StandardScaler | None = None
        self.regressao: LogisticRegression | None = None
        self.C: float | None = None
        self.perdas_c: dict[float, float] = {}
        self.constante: float | None = None

    @property
    def treinado(self) -> bool:
        return self.regressao is not None or self.constante is not None

    def _matriz(self, tabela: pd.DataFrame) -> np.ndarray:
        return tabela[CARACTERISTICAS].to_numpy(dtype=float)

    def _perdas_por_c(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, grupos: np.ndarray) -> dict[float, float]:
        """Log-loss média ponderada das dobras de validação para cada C; vazio se não dá para dobrar."""
        if len(np.unique(grupos)) < N_DOBRAS_C:
            return {}
        divisor = GroupKFold(n_splits=N_DOBRAS_C, shuffle=True, random_state=self.semente)
        dobras = [
            (treino, validacao) for treino, validacao in divisor.split(X, y, groups=grupos)
            if len(np.unique(y[treino])) == 2
        ]
        if not dobras:
            return {}
        perdas: dict[float, float] = {}
        for C in self.grade_c:
            soma, peso_total = 0.0, 0.0
            for treino, validacao in dobras:
                escalador = StandardScaler().fit(X[treino])
                regressao = _regressao(C).fit(escalador.transform(X[treino]), y[treino], sample_weight=w[treino])
                prob = regressao.predict_proba(escalador.transform(X[validacao]))
                peso = float(w[validacao].sum())
                soma += log_loss(y[validacao], prob, labels=[0, 1], sample_weight=w[validacao]) * peso
                peso_total += peso
            perdas[C] = soma / peso_total if peso_total > 0 else math.inf
        return perdas

    def treinar(self, tabela: pd.DataFrame, rotulos: Sequence, pesos: Sequence) -> "ModeloPares":
        """Treina com uma linha por par (`id`, `tema_id` e as `CARACTERISTICAS`), rótulo 0/1 e peso por linha."""
        y, w = _validar_tamanhos(tabela, rotulos, pesos)
        X = self._matriz(tabela)
        self.regressao, self.constante, self.perdas_c, self.C = None, None, {}, None
        self.escalador = StandardScaler().fit(X) if len(X) else None
        if len(np.unique(y)) < 2:
            self.constante = float(np.average(y, weights=w)) if len(y) and w.sum() > 0 else 0.0
            return self
        grupos = pd.factorize(tabela["id"].astype(str))[0]  # inteiros: o GroupKFold separa mais rápido
        self.perdas_c = self._perdas_por_c(X, y, w, grupos)
        self.C = min(self.perdas_c, key=lambda c: (self.perdas_c[c], c)) if self.perdas_c else C_PADRAO
        self.regressao = _regressao(self.C).fit(self.escalador.transform(X), y, sample_weight=w)
        return self

    def pontuar_pares(self, tabela: pd.DataFrame) -> np.ndarray:
        """`s` de cada linha: a probabilidade de o par ser positivo."""
        if not self.treinado:
            raise RuntimeError("ModeloPares ainda não foi treinado")
        if tabela.empty:
            return np.zeros(0)
        if self.regressao is None:
            return np.full(len(tabela), self.constante, dtype=float)
        return self.regressao.predict_proba(self.escalador.transform(self._matriz(tabela)))[:, 1]

    def probabilidades(self, tabela: pd.DataFrame) -> pd.DataFrame:
        """`id, tema_id, s, p` na ordem da tabela; `p = s / Σ s` entre os candidatos da questão (soma 1)."""
        s = self.pontuar_pares(tabela)
        prob = pd.DataFrame({
            "id": tabela["id"].astype(str).to_numpy(),
            "tema_id": tabela["tema_id"].astype("int64").to_numpy(),
            "s": s,
        })
        soma = prob.groupby("id", sort=False)["s"].transform("sum").to_numpy()
        quantos = prob.groupby("id", sort=False)["s"].transform("size").to_numpy()
        # soma zero (só o modelo constante em zero chega aqui): divide por igual entre os candidatos
        prob["p"] = np.divide(s, soma, out=1.0 / np.maximum(quantos, 1), where=soma > 0)
        return prob[COLUNAS_PROBABILIDADE]

    def pesos_legiveis(self) -> dict[str, float]:
        """Coeficiente de cada característica na escala padronizada (o tamanho compara a influência)."""
        if self.regressao is None:
            return {nome: 0.0 for nome in CARACTERISTICAS}
        return {nome: float(coef) for nome, coef in zip(CARACTERISTICAS, self.regressao.coef_[0])}

    def descrever(self) -> dict:
        """Resumo para `dados/intermediarios/pesos_modelo.json`."""
        return {
            "C": self.C,
            "perdas_validacao_por_C": {str(c): perda for c, perda in self.perdas_c.items()},
            "intercepto": float(self.regressao.intercept_[0]) if self.regressao is not None else None,
            "constante": self.constante,
            "pesos": self.pesos_legiveis(),
            "media": dict(zip(CARACTERISTICAS, map(float, self.escalador.mean_))) if self.escalador else None,
            "escala": dict(zip(CARACTERISTICAS, map(float, self.escalador.scale_))) if self.escalador else None,
        }


def agregar_ramo(prob: pd.DataFrame, arvore: "ArvoreTemas") -> pd.DataFrame:
    """`id, tema_id, p_ramo` de cada tema de nível 1 candidato: `p_pai + Σ p` dos seus filhos candidatos."""
    linhas: list[dict] = []
    por_questao: dict[str, dict[int, float]] = {}
    for id_q, tema_id, p in zip(prob["id"], prob["tema_id"], prob["p"]):
        por_questao.setdefault(str(id_q), {})[int(tema_id)] = float(p)
    for id_q, p_por_tema in por_questao.items():
        for tema_id, p in p_por_tema.items():
            if arvore.nivel(tema_id) != 1:
                continue
            filhos = sum(p_por_tema[f] for f in arvore.filhos(tema_id) if f in p_por_tema)
            linhas.append({"id": id_q, "tema_id": tema_id, "p_ramo": p + filhos})
    ramos = pd.DataFrame(linhas, columns=COLUNAS_RAMO)
    ramos["tema_id"] = ramos["tema_id"].astype("int64")
    ramos["p_ramo"] = ramos["p_ramo"].astype(float)
    return ramos


# --- características da questão para o detector de fora ---------------------------------------------------

CARACTERISTICAS_FORA = [
    "p_max", "s_max", "cos_max", "knn_max", "fracao_vizinhos_fora", "peso_fora", "termos_fora", "disciplina_fora",
]
COLUNAS_FORA = ["id", *CARACTERISTICAS_FORA]
N_VIZINHOS_FORA = 10
TERMOS_SATURAM = 5


def normalizar_texto(texto: str | None) -> str:
    """Minúsculas, sem acento, espaços colapsados."""
    sem_acento = unicodedata.normalize("NFKD", str(texto or "").lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento).strip()


def termos_encontrados(texto: str, termos: Iterable[str]) -> dict[str, int]:
    """Quantas vezes cada termo aparece no texto normalizado, como palavra inteira; só os que aparecem."""
    normal = normalizar_texto(texto)
    achados: dict[str, int] = {}
    for termo in termos:
        termo_normal = normalizar_texto(termo)
        if not termo_normal:
            continue
        n = len(re.findall(rf"(?<![a-z0-9]){re.escape(termo_normal)}(?![a-z0-9])", normal))
        if n:
            achados[termo] = n
    return achados


_TERMOS_POR_CONTEXTO: "weakref.WeakKeyDictionary[Contexto, dict[str, dict[str, int]]]" = weakref.WeakKeyDictionary()


def termos_do_item(contexto: "Contexto", id_item: str) -> dict[str, int]:
    """`termos_encontrados` do texto do item com os termos do contexto, calculado uma vez por item."""
    memoria = _TERMOS_POR_CONTEXTO.setdefault(contexto, {})
    if id_item not in memoria:
        memoria[id_item] = termos_encontrados(contexto.texto(id_item), contexto.termos_fora)
    return memoria[id_item]


def disciplina_fora(disciplina_origem: str | None) -> bool:
    """`disciplina_origem` preenchida que não é nenhuma das quatro disciplinas da árvore.

    Vazia conta como não fora: exemplos sintéticos e corpus não têm disciplina, e a característica não pode
    separar exemplo de questão.
    """
    return bool(str(disciplina_origem or "").strip()) and normalizar_disciplina(disciplina_origem) is None


def chaves_fora(chaves, dicionario: Mapping) -> list[str]:
    """Chaves da questão marcadas `fora` no dicionário que contam no peso de fora (a CF só sozinha)."""
    return sorted(
        chave for chave in chaves
        if dicionario.get(chave, {}).get("fora") and calcular_peso_fora(chaves, {chave}) > 0
    )


def gatilhos_fora(contexto: "Contexto", id_item: str) -> list[str]:
    """O que na questão aponta para fora da taxonomia: chaves fora, termos de outras disciplinas, disciplina."""
    gatilhos: list[str] = []
    chaves = chaves_fora(contexto.chaves(id_item), contexto.dicionario)
    if chaves:
        gatilhos.append("chaves " + ", ".join(chaves))
    termos = termos_do_item(contexto, id_item)
    if termos:
        gatilhos.append("termos " + ", ".join(sorted(termos)))
    disciplina = contexto.disciplina_origem(id_item)
    if disciplina_fora(disciplina):
        gatilhos.append(f"disciplina de origem {disciplina.strip()}")
    return gatilhos


def tabela_questoes(
    ids: Iterable[str],
    prob: pd.DataFrame,
    pares: pd.DataFrame,
    vizinhos_por_id: Mapping[str, list["Vizinho"]],
    contexto: "Contexto",
) -> pd.DataFrame:
    """Uma linha por questão (ou exemplo) com `COLUNAS_FORA`.

    `p_max`, `s_max`: maior `p` e maior `s` entre os candidatos; `cos_max`, `knn_max`: maiores `cos_tema` e
    `knn_max` dos pares; `fracao_vizinhos_fora`: fração dos até 10 vizinhos mais próximos com rótulo None;
    `peso_fora`: de `dicionario.pontuar`; `termos_fora`: ocorrências dos termos de `dados/termos_fora.json` / 5,
    no máximo 1; `disciplina_fora`: 1 se a disciplina de origem preenchida não é nenhuma das quatro. Sem
    candidato, as quatro primeiras ficam zero.
    """
    ids = list(ids)
    maiores_prob = prob.groupby("id", sort=False)[["p", "s"]].max() if len(prob) else None
    maiores_pares = pares.groupby("id", sort=False)[["cos_tema", "knn_max"]].max() if len(pares) else None
    linhas: list[dict] = []
    for id_item in ids:
        vizinhos = list(vizinhos_por_id.get(id_item) or [])[:N_VIZINHOS_FORA]
        termos = sum(termos_do_item(contexto, id_item).values())
        linha = {
            "id": id_item,
            "p_max": 0.0, "s_max": 0.0, "cos_max": 0.0, "knn_max": 0.0,
            "fracao_vizinhos_fora": (sum(r is None for _, r, _ in vizinhos) / len(vizinhos)) if vizinhos else 0.0,
            "peso_fora": float(pontuar(contexto.chaves(id_item), contexto.dicionario)[1]),
            "termos_fora": min(1.0, termos / TERMOS_SATURAM),
            "disciplina_fora": 1.0 if disciplina_fora(contexto.disciplina_origem(id_item)) else 0.0,
        }
        if maiores_prob is not None and id_item in maiores_prob.index:
            linha["p_max"] = float(maiores_prob.at[id_item, "p"])
            linha["s_max"] = float(maiores_prob.at[id_item, "s"])
        if maiores_pares is not None and id_item in maiores_pares.index:
            linha["cos_max"] = float(maiores_pares.at[id_item, "cos_tema"])
            linha["knn_max"] = float(maiores_pares.at[id_item, "knn_max"])
        linhas.append(linha)
    tabela = pd.DataFrame(linhas, columns=COLUNAS_FORA)
    tabela[CARACTERISTICAS_FORA] = tabela[CARACTERISTICAS_FORA].astype(float)
    return tabela


# --- detector de fora -------------------------------------------------------------------------------------

MIN_POR_CLASSE_PLATT = 6  # Platt com 3 dobras estratificadas pede ao menos 2 de cada classe por dobra
N_DOBRAS_PLATT = 3


class DetectorFora:
    """Regressão logística (`class_weight="balanced"`) sobre `CARACTERISTICAS_FORA` padronizadas, com Platt.

    Com pelo menos `MIN_POR_CLASSE_PLATT` exemplos de cada classe, a probabilidade sai de
    `CalibratedClassifierCV(method="sigmoid", cv=3)`; com menos, da própria regressão; sem positivos (ou sem
    negativos), é constante. A regressão no treino inteiro fica guardada para os pesos legíveis.
    """

    def __init__(self) -> None:
        self.escalador: StandardScaler | None = None
        self.regressao: LogisticRegression | None = None
        self.calibrado: CalibratedClassifierCV | None = None
        self.constante: float | None = None
        self.n_positivos = 0
        self.n_negativos = 0

    @property
    def treinado(self) -> bool:
        return self.regressao is not None or self.constante is not None

    @property
    def calibracao(self) -> str:
        if self.calibrado is not None:
            return "platt"
        return "sem calibração" if self.regressao is not None else "constante"

    def treinar(self, tabela_q: pd.DataFrame, rotulos_fora: Sequence, pesos: Sequence) -> "DetectorFora":
        y, w = _validar_tamanhos(tabela_q, rotulos_fora, pesos)
        X = tabela_q[CARACTERISTICAS_FORA].to_numpy(dtype=float)
        self.regressao, self.calibrado, self.constante = None, None, None
        self.n_positivos, self.n_negativos = int(y.sum()), int(len(y) - y.sum())
        self.escalador = StandardScaler().fit(X) if len(X) else None
        if self.n_positivos == 0 or self.n_negativos == 0:
            self.constante = float(np.average(y, weights=w)) if len(y) and w.sum() > 0 else 0.0
            return self
        Xp = self.escalador.transform(X)
        self.regressao = _regressao().fit(Xp, y, sample_weight=w)
        if min(self.n_positivos, self.n_negativos) >= MIN_POR_CLASSE_PLATT:
            self.calibrado = CalibratedClassifierCV(_regressao(), method="sigmoid", cv=N_DOBRAS_PLATT)
            self.calibrado.fit(Xp, y, sample_weight=w)
        return self

    def p_fora(self, tabela_q: pd.DataFrame) -> pd.Series:
        """Probabilidade de cada questão ser fora da taxonomia, indexada pelo id."""
        if not self.treinado:
            raise RuntimeError("DetectorFora ainda não foi treinado")
        indice = pd.Index(tabela_q["id"].astype(str), name="id")
        if tabela_q.empty:
            return pd.Series([], index=indice, dtype=float, name="p_fora")
        if self.constante is not None:
            valores = np.full(len(tabela_q), self.constante, dtype=float)
        else:
            Xp = self.escalador.transform(tabela_q[CARACTERISTICAS_FORA].to_numpy(dtype=float))
            modelo = self.calibrado if self.calibrado is not None else self.regressao
            valores = modelo.predict_proba(Xp)[:, 1]
        return pd.Series(valores, index=indice, dtype=float, name="p_fora")

    def pesos_legiveis(self) -> dict[str, float]:
        if self.regressao is None:
            return {nome: 0.0 for nome in CARACTERISTICAS_FORA}
        return {nome: float(coef) for nome, coef in zip(CARACTERISTICAS_FORA, self.regressao.coef_[0])}

    def descrever(self) -> dict:
        return {
            "calibracao": self.calibracao,
            "n_positivos": self.n_positivos,
            "n_negativos": self.n_negativos,
            "intercepto": float(self.regressao.intercept_[0]) if self.regressao is not None else None,
            "constante": self.constante,
            "pesos": self.pesos_legiveis(),
        }


# --- rótulos dos pares --------------------------------------------------------------------------------------


def temas_positivos(contexto: "Contexto", id_item: str) -> set[int]:
    """Temas que contam como positivos num par do item: central e extras (questão da folha) ou o tema do exemplo.

    Pai e filho não contam (o pai é tratado na decisão); questão ou exemplo fora não tem positivo.
    """
    item = contexto.item(id_item)
    if item["origem"] == "questao":
        linha = contexto.linha_folha(id_item)
        if linha["fora"] or linha["tema_central_id"] is None:
            return set()
        return {int(linha["tema_central_id"]), *(int(e) for e in linha["extras_ids"])}
    return set() if item["tema_id"] is None else {int(item["tema_id"])}


def rotulos_pares(pares: pd.DataFrame, positivos: Mapping[str, Collection[int]]) -> np.ndarray:
    """1 quando o tema do par está entre os positivos do item."""
    return np.array(
        [int(int(t) in positivos.get(str(i), ())) for i, t in zip(pares["id"], pares["tema_id"])], dtype=int
    )


def pesos_por_linha(tabela: pd.DataFrame, peso_por_id: Mapping[Hashable, float]) -> np.ndarray:
    return np.array([float(peso_por_id[str(i)]) for i in tabela["id"]], dtype=float)


# --- segundo estágio: confiança por opção de publicação ----------------------------------------------------

CARACTERISTICAS_CONFIANCA = ["p", "s", "votos", "votos_ramo", "nivel2_opcao", "knn_folha_max", "tem_lei"]
COLUNAS_OPCOES = ["id", "opcao", "tema_id", *CARACTERISTICAS_CONFIANCA]
C_CONFIANCA = 1.0


def tem_lei(contexto: "Contexto", id_item: str) -> bool:
    """A questão tem alguma chave de lei, artigo ou súmula que aponta para algum tema publicável."""
    pontos, _ = pontuar(contexto.chaves(id_item), contexto.dicionario)
    return any(valor > 0 and int(tema_id) in contexto.publicaveis for tema_id, valor in pontos.items())


def tabela_opcoes(prob: pd.DataFrame, ramos: pd.DataFrame, pares: pd.DataFrame, contexto: "Contexto") -> pd.DataFrame:
    """Uma ou duas linhas por questão com candidatos, na ordem de `prob`; colunas `COLUNAS_OPCOES`.

    - `tema`: o top-1 do modelo de pares (maior `p`; empate, menor id). `p` e `s` são os dele.
    - `ramo`: o pai de nível 1 do top-1, só quando o top-1 é de nível 2 e o pai é candidato; `p` = `p_ramo` do pai.

    `votos` e `votos_ramo` são os do par (questão, tema da opção); `s` e `knn_folha_max` são sempre os do par do
    top-1; `nivel2_opcao` é 1 se o tema da opção é de nível 2; `tem_lei` é da questão (`tem_lei`).
    """
    arvore = contexto.arvore
    par = {
        (str(i), int(t)): (float(v), float(r), float(k))
        for i, t, v, r, k in zip(pares["id"], pares["tema_id"], pares["votos"], pares["votos_ramo"],
                                 pares["knn_folha_max"])
    }
    p_ramo = {(str(i), int(t)): float(v) for i, t, v in zip(ramos["id"], ramos["tema_id"], ramos["p_ramo"])}
    topo: dict[str, tuple[int, float, float]] = {}
    for id_q, tema_id, s, p in zip(prob["id"], prob["tema_id"], prob["s"], prob["p"]):
        id_q, tema_id, p = str(id_q), int(tema_id), float(p)
        atual = topo.get(id_q)
        if atual is None or (-p, tema_id) < (-atual[2], atual[0]):
            topo[id_q] = (tema_id, float(s), p)
    linhas: list[dict] = []
    for id_q, (tema_id, s, p) in topo.items():
        votos, votos_ramo, knn_folha = par.get((id_q, tema_id), (0.0, 0.0, 0.0))
        lei = 1.0 if tem_lei(contexto, id_q) else 0.0
        nivel2 = arvore.nivel(tema_id) >= 2
        linhas.append({"id": id_q, "opcao": OPCAO_TEMA, "tema_id": tema_id, "p": p, "s": s, "votos": votos,
                       "votos_ramo": votos_ramo, "nivel2_opcao": 1.0 if nivel2 else 0.0, "knn_folha_max": knn_folha,
                       "tem_lei": lei})
        pai = arvore.pai(tema_id) if nivel2 else None
        if pai is not None and (id_q, int(pai)) in p_ramo:
            votos_pai, votos_ramo_pai, _ = par.get((id_q, int(pai)), (0.0, votos_ramo, 0.0))
            linhas.append({"id": id_q, "opcao": OPCAO_RAMO, "tema_id": int(pai), "p": p_ramo[(id_q, int(pai))],
                           "s": s, "votos": votos_pai, "votos_ramo": votos_ramo_pai, "nivel2_opcao": 0.0,
                           "knn_folha_max": knn_folha, "tem_lei": lei})
    tabela = pd.DataFrame(linhas, columns=COLUNAS_OPCOES)
    tabela["tema_id"] = tabela["tema_id"].astype("int64")
    tabela[CARACTERISTICAS_CONFIANCA] = tabela[CARACTERISTICAS_CONFIANCA].astype(float)
    return tabela


class ModeloConfianca:
    """Segundo estágio: probabilidade de uma opção de publicação (`tabela_opcoes`) estar certa pela régua.

    `LogisticRegression(C=1, class_weight=None)` sobre `CARACTERISTICAS_CONFIANCA` padronizadas, treinada com
    opções de previsões fora da amostra do modelo de pares (`calibracao.opcoes_fora_da_amostra`). Sem as duas
    classes no treino, devolve a fração de acertos (0 sem nenhuma opção).
    """

    def __init__(self, C: float = C_CONFIANCA) -> None:
        self.C = float(C)
        self.escalador: StandardScaler | None = None
        self.regressao: LogisticRegression | None = None
        self.constante: float | None = None
        self.n_opcoes = 0
        self.n_certas = 0

    @property
    def treinado(self) -> bool:
        return self.regressao is not None or self.constante is not None

    def treinar(self, opcoes: pd.DataFrame, rotulos: Sequence) -> "ModeloConfianca":
        """Uma linha por opção (`CARACTERISTICAS_CONFIANCA`) e rótulo 1 quando a opção está certa."""
        y, _ = _validar_tamanhos(opcoes, rotulos, np.ones(len(opcoes)))
        X = opcoes[CARACTERISTICAS_CONFIANCA].to_numpy(dtype=float)
        self.regressao, self.constante = None, None
        self.n_opcoes, self.n_certas = int(len(y)), int(y.sum())
        self.escalador = StandardScaler().fit(X) if len(X) else None
        if len(np.unique(y)) < 2:
            self.constante = float(y.mean()) if len(y) else 0.0
            return self
        self.regressao = LogisticRegression(C=self.C, class_weight=None, max_iter=MAX_ITER)
        self.regressao.fit(self.escalador.transform(X), y)
        return self

    def confianca(self, opcoes: pd.DataFrame) -> np.ndarray:
        """Probabilidade de acerto de cada opção, na ordem da tabela."""
        if not self.treinado:
            raise RuntimeError("ModeloConfianca ainda não foi treinado")
        if opcoes.empty:
            return np.zeros(0)
        if self.regressao is None:
            return np.full(len(opcoes), self.constante, dtype=float)
        X = self.escalador.transform(opcoes[CARACTERISTICAS_CONFIANCA].to_numpy(dtype=float))
        return self.regressao.predict_proba(X)[:, 1]

    def pesos_legiveis(self) -> dict[str, float]:
        if self.regressao is None:
            return {nome: 0.0 for nome in CARACTERISTICAS_CONFIANCA}
        return {nome: float(coef) for nome, coef in zip(CARACTERISTICAS_CONFIANCA, self.regressao.coef_[0])}

    def descrever(self) -> dict:
        """Resumo para `pesos_modelo.json`: pesos na escala padronizada, intercepto e o tamanho do treino."""
        return {
            "C": self.C,
            "n_opcoes": self.n_opcoes,
            "n_certas": self.n_certas,
            "intercepto": float(self.regressao.intercept_[0]) if self.regressao is not None else None,
            "constante": self.constante,
            "pesos": self.pesos_legiveis(),
            "media": (dict(zip(CARACTERISTICAS_CONFIANCA, map(float, self.escalador.mean_)))
                      if self.escalador is not None else None),
            "escala": (dict(zip(CARACTERISTICAS_CONFIANCA, map(float, self.escalador.scale_)))
                       if self.escalador is not None else None),
        }
