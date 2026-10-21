"""knn_folha_max transfere da folha para as 559? Compara o valor do top-1 fora da amostra (dobras) com o das 559
no modelo treinado com os 100 rótulos. Uso: uv run python scripts/diagnostico_transferencia.py"""
from __future__ import annotations

import sys

import numpy as np

from sistema.calibracao import dobras, treinar_rodada
from sistema.contexto import carregar_contexto
from sistema.dados import carregar_folha


def top1_knn(aplicacao, ids):
    pares = aplicacao.pares.set_index(["id", "tema_id"])
    valores = []
    for i in ids:
        lista, _ = aplicacao.listas(i)
        if lista:
            valores.append(float(pares.loc[(i, int(lista[0][0])), "knn_folha_max"]))
    return np.array(valores)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ctx = carregar_contexto(com_corpus=False)
    folha = carregar_folha()
    oof = []
    for ids_treino, ids_teste in dobras(folha, n_dobras=5, repeticoes=1, arvore=ctx.arvore):
        rodada = treinar_rodada(ctx, ids_treino, False)
        oof.append(top1_knn(rodada.aplicar(ids_teste), ids_teste))
    oof = np.concatenate(oof)
    fora_folha = [i for i in ctx.questoes.id if i not in set(folha.id)]
    final = treinar_rodada(ctx, list(folha.id), False)
    v559 = top1_knn(final.aplicar(fora_folha), fora_folha)
    q = lambda v: np.round(np.quantile(v, [0.25, 0.5, 0.75, 0.9]), 3)
    print(f"knn_folha_max do top-1, folha fora da amostra: média {oof.mean():.3f}, quantis 25/50/75/90 {q(oof)}, zeros {np.mean(oof == 0):.0%}")
    print(f"knn_folha_max do top-1, 559 com o modelo final: média {v559.mean():.3f}, quantis 25/50/75/90 {q(v559)}, zeros {np.mean(v559 == 0):.0%}")


if __name__ == "__main__":
    main()
