"""Decisão por questão a partir de `p` dos candidatos, `p_ramo` dos pais e `p_fora` (spec, seção 9).

1. `p_fora ≥ τ_fora` → `fora`, motivo com o que disparou (chaves fora, termos, disciplina).
2. Tema de maior `p` com `p ≥ τ` → `publicar` nele.
3. Senão, pai de nível 1 de maior `p_ramo` com `p_ramo ≥ τ` → `publicar` no pai.
4. Senão → `revisar` com o tema de maior `p`; motivo "probabilidade X abaixo do limiar Y", ou "divergência
   entre ramos" quando os dois maiores `p` estão em ramos diferentes e somam mais de 0,8.
5. Extras (publicar e revisar): até 3 outros candidatos com `p ≥ τ_extra`, em ordem de `p`, fora do central, do
   pai e dos filhos dele. Confiança: `p` do central (ou `p_ramo` no pai), com 2 casas; em `fora`, `p_fora`.

Questão sem nenhum candidato não tem tema para sugerir: sai `fora` (regra 1) com o motivo dito.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sistema.arvore import ArvoreTemas

PUBLICAR = "publicar"
REVISAR = "revisar"
FORA = "fora"
DECISOES = (PUBLICAR, REVISAR, FORA)

EXTRA_PADRAO = 0.15
N_EXTRAS = 3
SOMA_DIVERGENCIA = 0.8


@dataclass(frozen=True)
class Limiares:
    publicar: float
    fora: float
    extra: float = EXTRA_PADRAO


@dataclass
class Decisao:
    id: str
    decisao: str
    tema_central_id: int | None
    extras_ids: list[int] = field(default_factory=list)
    confianca: float = 0.0
    motivo: str = ""
    regra: int = 0


def numero(valor: float) -> str:
    """Duas casas com vírgula, para o texto do motivo."""
    return f"{valor:.2f}".replace(".", ",")


def _ordenar(pares: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    return sorted(((int(t), float(p)) for t, p in pares), key=lambda par: (-par[1], par[0]))


def _de_tabela(tabela: pd.DataFrame | Sequence | None, coluna: str) -> list[tuple[int, float]]:
    if tabela is None:
        return []
    if isinstance(tabela, pd.DataFrame):
        return list(zip(tabela["tema_id"].astype(int), tabela[coluna].astype(float)))
    return list(tabela)


def _extras(central: int, temas: list[tuple[int, float]], limiar: float, arvore: "ArvoreTemas") -> list[int]:
    excluidos = {central, *arvore.filhos(central)}
    pai = arvore.pai(central)
    if pai is not None:
        excluidos.add(pai)
    return [t for t, p in temas if p >= limiar and t not in excluidos][:N_EXTRAS]


def _motivo_revisar(temas: list[tuple[int, float]], limiar: float, arvore: "ArvoreTemas") -> str:
    (t1, p1), *resto = temas
    if resto:
        t2, p2 = resto[0]
        if p1 + p2 > SOMA_DIVERGENCIA and arvore.ancestral_nivel1(t1) != arvore.ancestral_nivel1(t2):
            return f"divergência entre ramos: {t1} ({numero(p1)}) e {t2} ({numero(p2)})"
    return f"probabilidade {numero(p1)} abaixo do limiar {numero(limiar)}"


def _motivo_fora(p_fora: float, gatilhos: Sequence[str], p_max: float | None) -> str:
    if gatilhos:
        return f"fora da taxonomia (p_fora {numero(p_fora)}): " + "; ".join(gatilhos)
    if p_max is None:
        return f"fora da taxonomia (p_fora {numero(p_fora)}): nenhum tema candidato"
    return f"fora da taxonomia (p_fora {numero(p_fora)}): tema mais provável só com {numero(p_max)}"


def decidir_listas(
    id_questao: str,
    temas: Sequence[tuple[int, float]],
    ramos: Sequence[tuple[int, float]],
    p_fora: float,
    limiares: Limiares,
    arvore: "ArvoreTemas",
    gatilhos: Sequence[str] = (),
    com_extras: bool = True,
    ordenados: bool = False,
) -> Decisao:
    """`decidir` com `(tema_id, p)` e `(pai, p_ramo)` em listas; `ordenados` diz que já vêm em ordem de `p`.

    `com_extras=False` pula os extras (a grade de limiares só conta o central).
    """
    if not ordenados:
        temas, ramos = _ordenar(temas), _ordenar(ramos)
    p_fora = float(p_fora)
    if p_fora >= limiares.fora or not temas:
        motivo = _motivo_fora(p_fora, gatilhos, temas[0][1] if temas else None)
        return Decisao(id_questao, FORA, None, [], round(p_fora, 2), motivo, 1)
    tema_max, p_max = temas[0]
    if p_max >= limiares.publicar:
        central, confianca, decisao, motivo, regra = tema_max, p_max, PUBLICAR, "", 2
    elif ramos and ramos[0][1] >= limiares.publicar:
        (central, confianca), decisao, motivo, regra = ramos[0], PUBLICAR, "", 3
    else:
        central, confianca, decisao, regra = tema_max, p_max, REVISAR, 4
        motivo = _motivo_revisar(temas, limiares.publicar, arvore)
    extras = _extras(central, temas, limiares.extra, arvore) if com_extras else []
    return Decisao(id_questao, decisao, int(central), extras, round(float(confianca), 2), motivo, regra)


def decidir(
    id_questao: str,
    prob_q: pd.DataFrame,
    p_ramo_q: pd.DataFrame,
    p_fora: float,
    limiares: Limiares,
    arvore: "ArvoreTemas",
    gatilhos: Sequence[str] = (),
) -> Decisao:
    """Decisão de uma questão: `prob_q` com `tema_id, p` dos candidatos; `p_ramo_q` com `tema_id, p_ramo` dos pais.

    `gatilhos` (de `modelo.gatilhos_fora`) entram no motivo quando a decisão é `fora`.
    """
    return decidir_listas(id_questao, _de_tabela(prob_q, "p"), _de_tabela(p_ramo_q, "p_ramo"), p_fora, limiares,
                          arvore, gatilhos)
