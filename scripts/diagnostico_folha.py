"""Diagnóstico fora da amostra na folha-verdade: onde o sistema perde a resposta certa.

Para uma repetição de 5 dobras, mede em cada questão de teste da árvore (não fora):
- recall dos candidatos: algum alvo aceito (central, extra ou pai/filho deles) está entre os candidatos?
- posição do primeiro alvo aceito na ordenação por probabilidade (top-1, top-3, top-5);
- distribuição de p_max e da confiança do segundo estágio quando o top-1 está certo e quando está errado.

Uso: uv run python scripts/diagnostico_folha.py [--com-corpus]
"""
from __future__ import annotations

import argparse
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from sistema.calibracao import dobras, rodar_dobra  # noqa: E402
from sistema.contexto import carregar_contexto  # noqa: E402
from sistema.dados import alvos_aceitos, carregar_folha  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--com-corpus", action="store_true")
    args = parser.parse_args()
    ctx = carregar_contexto(com_corpus=args.com_corpus)
    folha = carregar_folha()
    linhas = {r["id"]: r for r in folha.to_dict("records")}
    arv = ctx.arvore

    posicoes, pmax_certo, pmax_errado, recall, n_cand = [], [], [], [], []
    conf_certo, conf_errado = [], []
    for ids_treino, ids_teste in dobras(folha, n_dobras=5, repeticoes=1, arvore=arv):
        res = rodar_dobra(ctx, ids_treino, ids_teste, args.com_corpus)
        for r in res.to_dict("records"):
            linha = linhas[r["id"]]
            if linha["fora"]:
                continue
            alvos = alvos_aceitos(linha, arv)
            cand = [int(t) for t, _ in r["candidatos_prob"]]
            n_cand.append(len(cand))
            recall.append(any(t in alvos for t in cand))
            pos = next((i for i, t in enumerate(cand) if t in alvos), None)
            posicoes.append(pos)
            (pmax_certo if pos == 0 else pmax_errado).append(r["p_max"])
            (conf_certo if pos == 0 else conf_errado).append(r["confianca_tema"])

    n = len(posicoes)
    def taxa(k):
        return sum(1 for p in posicoes if p is not None and p < k) / n
    print(f"questões da árvore avaliadas: {n} | candidatos por questão: média {np.mean(n_cand):.1f}")
    print(f"recall dos candidatos (algum alvo aceito na lista): {sum(recall)}/{n} = {sum(recall)/n:.0%}")
    print(f"top-1: {taxa(1):.0%} | top-3: {taxa(3):.0%} | top-5: {taxa(5):.0%} | top-10: {taxa(10):.0%}")
    q = lambda v: np.round(np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]), 2) if v else []
    print(f"p_max quando top-1 certo (n={len(pmax_certo)}): quantis 10/25/50/75/90 = {q(pmax_certo)}")
    print(f"p_max quando top-1 errado (n={len(pmax_errado)}): quantis 10/25/50/75/90 = {q(pmax_errado)}")
    for limiar in (0.2, 0.3, 0.4, 0.5, 0.6):
        c = sum(1 for p in pmax_certo if p >= limiar); e = sum(1 for p in pmax_errado if p >= limiar)
        print(f"  p_max >= {limiar:.1f}: publicaria {c + e} ({(c + e)/n:.0%}), certas {c}, precisão {c/max(c+e,1):.0%}")
    print(f"confiança do tema quando top-1 certo (n={len(conf_certo)}): quantis 10/25/50/75/90 = {q(conf_certo)}")
    print(f"confiança do tema quando top-1 errado (n={len(conf_errado)}): quantis 10/25/50/75/90 = {q(conf_errado)}")
    for limiar in (0.6, 0.7, 0.8, 0.9):
        c = sum(1 for p in conf_certo if p >= limiar); e = sum(1 for p in conf_errado if p >= limiar)
        print(f"  confiança >= {limiar:.1f}: publicaria {c + e} ({(c + e)/n:.0%}), certas {c}, "
              f"precisão {c/max(c+e,1):.0%}")


if __name__ == "__main__":
    main()
