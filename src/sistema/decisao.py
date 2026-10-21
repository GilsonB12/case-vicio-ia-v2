"""Decisão por questão a partir de `p` dos candidatos, das opções do segundo estágio e de `p_fora` (spec, seção 9).

As opções (`Opcao`) vêm do `ModeloConfianca`: o top-1 do modelo de pares (`tema`) e, quando ele é de nível 2, o
ramo de nível 1 dele (`ramo`), cada uma com a confiança (probabilidade de estar certa pela régua).

1. `p_fora ≥ τ_fora` → `fora`, motivo com o que disparou (chaves fora, termos, disciplina).
2. Opção de maior confiança é o `tema` e a confiança ≥ `τ` → `publicar` no top-1.
3. Opção de maior confiança é o `ramo` e a confiança ≥ `τ` → `publicar` no pai (empate de confiança fica com o
   `tema`, mais específico).
4. Senão → `revisar` com o top-1 do modelo de pares; motivo "confiança X abaixo do limiar Y" (X é a da melhor
   opção), ou "buscas divergem" quando no máximo uma das cinco buscas tem o top-1 como primeiro.
5. Extras (publicar e revisar): até 3 outros candidatos com `p ≥ τ_extra`, em ordem de `p`, fora do central, do
   pai e dos filhos dele. Confiança: a do segundo estágio para o central (em `revisar`, a da opção `tema`), com 2
   casas; em `fora`, `p_fora`.

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
VOTOS_DIVERGENCIA = 1  # com no máximo este número de buscas no top-1, o motivo de revisar é "buscas divergem"
N_BUSCAS = 5  # dicionário de leis, BM25, cosseno com o tema, kNN e termos técnicos

OPCAO_TEMA = "tema"
OPCAO_RAMO = "ramo"


@dataclass(frozen=True)
class Limiares:
    publicar: float
    fora: float
    extra: float = EXTRA_PADRAO


@dataclass(frozen=True)
class Opcao:
    """Uma opção de publicação e a confiança do segundo estágio; `votos`: buscas com o tema como top-1."""

    opcao: str
    tema_id: int
    confianca: float
    votos: int = 0


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


def opcoes_de_tabela(tabela: pd.DataFrame | Sequence[Opcao] | None) -> list[Opcao]:
    """`Opcao` de cada linha (`opcao`, `tema_id`, `confianca` e, se houver, `votos`) de uma tabela de opções."""
    if tabela is None:
        return []
    if not isinstance(tabela, pd.DataFrame):
        return list(tabela)
    votos = tabela["votos"] if "votos" in tabela else [0] * len(tabela)
    return [Opcao(str(o), int(t), float(c), int(v))
            for o, t, c, v in zip(tabela["opcao"], tabela["tema_id"], tabela["confianca"], votos)]


def melhor_opcao(opcoes: Sequence[Opcao]) -> Opcao | None:
    """A de maior confiança; empate fica com o `tema`."""
    if not opcoes:
        return None
    return max(opcoes, key=lambda o: (float(o.confianca), o.opcao == OPCAO_TEMA))


def _extras(central: int, temas: list[tuple[int, float]], limiar: float, arvore: "ArvoreTemas") -> list[int]:
    excluidos = {central, *arvore.filhos(central)}
    pai = arvore.pai(central)
    if pai is not None:
        excluidos.add(pai)
    return [t for t, p in temas if p >= limiar and t not in excluidos][:N_EXTRAS]


def _motivo_revisar(tema: int, opcao_tema: Opcao | None, melhor: Opcao | None, limiar: float) -> str:
    confianca = float(melhor.confianca) if melhor is not None else 0.0
    no_ramo = f" (ramo {melhor.tema_id})" if melhor is not None and melhor.opcao == OPCAO_RAMO else ""
    abaixo = f"confiança {numero(confianca)}{no_ramo} abaixo do limiar {numero(limiar)}"
    votos = int(opcao_tema.votos) if opcao_tema is not None else 0
    if votos <= VOTOS_DIVERGENCIA:
        return f"buscas divergem: {votos} de {N_BUSCAS} buscas têm o tema {tema} como primeiro; {abaixo}"
    return abaixo


def _motivo_fora(p_fora: float, gatilhos: Sequence[str], p_max: float | None) -> str:
    if gatilhos:
        return f"fora da taxonomia (p_fora {numero(p_fora)}): " + "; ".join(gatilhos)
    if p_max is None:
        return f"fora da taxonomia (p_fora {numero(p_fora)}): nenhum tema candidato"
    return f"fora da taxonomia (p_fora {numero(p_fora)}): tema mais provável só com {numero(p_max)}"


def decidir_listas(
    id_questao: str,
    temas: Sequence[tuple[int, float]],
    opcoes: Sequence[Opcao],
    p_fora: float,
    limiares: Limiares,
    arvore: "ArvoreTemas",
    gatilhos: Sequence[str] = (),
    com_extras: bool = True,
    ordenados: bool = False,
) -> Decisao:
    """`decidir` com `(tema_id, p)` em lista e as opções do segundo estágio; `ordenados` diz que os temas já vêm em
    ordem de `p`.

    `com_extras=False` pula os extras (a grade de limiares só conta o central).
    """
    if not ordenados:
        temas = _ordenar(temas)
    p_fora = float(p_fora)
    if p_fora >= limiares.fora or not temas:
        motivo = _motivo_fora(p_fora, gatilhos, temas[0][1] if temas else None)
        return Decisao(id_questao, FORA, None, [], round(p_fora, 2), motivo, 1)
    tema_max, _ = temas[0]
    opcao_tema = next((o for o in opcoes if o.opcao == OPCAO_TEMA), None)
    melhor = melhor_opcao(opcoes)
    if melhor is not None and float(melhor.confianca) >= limiares.publicar:
        central, confianca, decisao, motivo = melhor.tema_id, melhor.confianca, PUBLICAR, ""
        regra = 2 if melhor.opcao == OPCAO_TEMA else 3
    else:
        central, decisao, regra = tema_max, REVISAR, 4
        confianca = opcao_tema.confianca if opcao_tema is not None else 0.0
        motivo = _motivo_revisar(tema_max, opcao_tema, melhor, limiares.publicar)
    extras = _extras(central, temas, limiares.extra, arvore) if com_extras else []
    return Decisao(id_questao, decisao, int(central), extras, round(float(confianca), 2), motivo, regra)


def decidir(
    id_questao: str,
    prob_q: pd.DataFrame,
    opcoes_q: pd.DataFrame | Sequence[Opcao],
    p_fora: float,
    limiares: Limiares,
    arvore: "ArvoreTemas",
    gatilhos: Sequence[str] = (),
) -> Decisao:
    """Decisão de uma questão: `prob_q` com `tema_id, p` dos candidatos; `opcoes_q` com `opcao, tema_id,
    confianca, votos` das opções do segundo estágio (ou a lista de `Opcao`).

    `gatilhos` (de `modelo.gatilhos_fora`) entram no motivo quando a decisão é `fora`.
    """
    return decidir_listas(id_questao, _de_tabela(prob_q, "p"), opcoes_de_tabela(opcoes_q), p_fora, limiares,
                          arvore, gatilhos)
