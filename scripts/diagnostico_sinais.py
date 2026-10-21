"""Sinais candidatos a confiança, medidos fora da amostra na folha-verdade (uma repetição de 5 dobras)."""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from sistema.calibracao import dobras, rodar_dobra  # noqa: E402
from sistema.contexto import carregar_contexto  # noqa: E402
from sistema.dados import alvos_aceitos, carregar_folha  # noqa: E402
from sistema.dicionario import pontuar  # noqa: E402


def ramo(arv, t):
    return t if arv.nivel(t) == 1 else arv.pai(t)


def main() -> None:
    ctx = carregar_contexto(com_corpus=False)
    folha = carregar_folha()
    linhas = {r["id"]: r for r in folha.to_dict("records")}
    arv = ctx.arvore
    reg = []
    for ids_treino, ids_teste in dobras(folha, n_dobras=5, repeticoes=1, arvore=arv):
        res = rodar_dobra(ctx, ids_treino, ids_teste, False)
        for r in res.to_dict("records"):
            linha = linhas[r["id"]]
            if linha["fora"]:
                continue
            alvos = alvos_aceitos(linha, arv)
            cp = [(int(t), float(p)) for t, p in r["candidatos_prob"]]
            t1, p1 = cp[0]; p2 = cp[1][1] if len(cp) > 1 else 0.0
            r1 = ramo(arv, t1)
            p_ramo1 = sum(p for t, p in cp if ramo(arv, t) == r1)
            # rankers independentes
            bm = ctx.pontuacoes_bm25(r["id"]); co = ctx.cossenos_temas(r["id"])
            top_bm = int(ctx.temas_ids[int(np.argmax(bm))]) if np.max(bm) > 0 else None
            top_co = int(ctx.temas_ids[int(np.argmax(co))])
            dic, _ = pontuar(ctx.chaves(r["id"]), ctx.dicionario)
            top_dic = max(dic, key=dic.get) if dic else None
            tops = [top_bm, top_co, top_dic]
            conc_tema = sum(1 for x in tops if x == t1)
            conc_ramo = sum(1 for x in tops if x is not None and ramo(arv, x) == r1)
            reg.append(dict(ok_tema=t1 in alvos, ok_ramo=r1 in alvos or any(arv.pai(a) == r1 for a in alvos if arv.nivel(a) == 2),
                            p1=p1, margem=p1 - p2, razao=p1 / max(p2, 1e-9), p_ramo=p_ramo1,
                            conc_tema=conc_tema, conc_ramo=conc_ramo, tem_dic=top_dic is not None))
    n = len(reg)
    print(f"n={n} | acerto top-1 no tema: {sum(r['ok_tema'] for r in reg)/n:.0%} | publicando o ramo (nível 1) do top-1: {sum(r['ok_ramo'] for r in reg)/n:.0%}")
    for chave in ("conc_tema", "conc_ramo"):
        print(f"\nconcordância {chave} (quantas de BM25, cosseno-tema e dicionário apontam o mesmo {'tema' if chave=='conc_tema' else 'ramo'} do top-1):")
        for v in range(4):
            g = [r for r in reg if r[chave] == v]
            if g:
                print(f"  {v}: n={len(g):3d} | acerto tema {sum(r['ok_tema'] for r in g)/len(g):.0%} | acerto ramo {sum(r['ok_ramo'] for r in g)/len(g):.0%}")
    for nome in ("margem", "razao", "p_ramo"):
        v = np.array([r[nome] for r in reg]); ok = np.array([r["ok_tema"] for r in reg]); okr = np.array([r["ok_ramo"] for r in reg])
        corte = np.quantile(v, [0.5, 0.75, 0.9])
        print(f"\n{nome}: quantis 50/75/90 = {np.round(corte, 3)}")
        for c in corte:
            m = v >= c
            print(f"  >= {c:.3f}: n={m.sum():3d} | acerto tema {ok[m].mean():.0%} | acerto ramo {okr[m].mean():.0%}")


if __name__ == "__main__":
    main()
