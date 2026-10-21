"""Sinais candidatos a confiança, medidos fora da amostra na folha-verdade (uma repetição de 5 dobras).

Mostra o acerto do top-1 do modelo de pares (no tema e no ramo de nível 1) por concordância entre as buscas
independentes: `votos` (quantas de BM25, cosseno questão×tema, dicionário de leis, kNN e termos técnicos têm o
top-1 como primeiro)
e `votos_ramo` (quantas têm o primeiro no mesmo ramo); a concordância antiga só com as três primeiras; e a
confiança do segundo estágio (`ModeloConfianca`) quando acerta e quando erra.

Uso: uv run python scripts/diagnostico_sinais.py [--com-corpus]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from sistema.calibracao import dobras, rodar_dobra  # noqa: E402
from sistema.contexto import carregar_contexto  # noqa: E402
from sistema.dados import acerto_regua, carregar_folha  # noqa: E402
from sistema.decisao import melhor_opcao  # noqa: E402
from sistema.dicionario import pontuar  # noqa: E402


def ramo(arv, t):
    return t if arv.nivel(t) == 1 else arv.pai(t)


def taxa(grupo, chave):
    return sum(r[chave] for r in grupo) / len(grupo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--com-corpus", action="store_true")
    args = parser.parse_args()
    ctx = carregar_contexto(com_corpus=args.com_corpus)
    folha = carregar_folha()
    linhas = {r["id"]: r for r in folha.to_dict("records")}
    arv = ctx.arvore
    reg = []
    for ids_treino, ids_teste in dobras(folha, n_dobras=5, repeticoes=1, arvore=arv):
        res = rodar_dobra(ctx, ids_treino, ids_teste, args.com_corpus)
        for r in res.to_dict("records"):
            linha = linhas[r["id"]]
            if linha["fora"]:
                continue
            t1 = int(r["tema_max"])
            r1 = ramo(arv, t1)
            # concordância antiga, só com as três buscas sem o kNN
            bm = ctx.pontuacoes_bm25(r["id"])
            co = ctx.cossenos_temas(r["id"])
            top_bm = int(ctx.temas_ids[int(np.argmax(bm))]) if np.max(bm) > 0 else None
            top_co = int(ctx.temas_ids[int(np.argmax(co))])
            dic, _ = pontuar(ctx.chaves(r["id"]), ctx.dicionario)
            top_dic = min(dic, key=lambda t: (-dic[t], t)) if dic else None
            conc3 = sum(1 for x in (top_bm, top_co, top_dic) if x == t1)
            melhor = melhor_opcao(r["opcoes"])
            reg.append(dict(
                ok_tema=acerto_regua(t1, linha, arv), ok_ramo=acerto_regua(r1, linha, arv),
                votos=int(r["votos"]), votos_ramo=int(r["votos_ramo"]), conc3=conc3,
                conf_tema=float(r["confianca_tema"]), conf_max=float(melhor.confianca),
                ok_melhor=acerto_regua(melhor.tema_id, linha, arv), melhor_ramo=melhor.opcao == "ramo",
            ))
    n = len(reg)
    print(f"n={n} | acerto top-1 no tema: {taxa(reg, 'ok_tema'):.0%} | publicando o ramo (nível 1) do top-1: "
          f"{taxa(reg, 'ok_ramo'):.0%}")
    titulos = {
        "votos": "votos (quantas de BM25, cosseno-tema, dicionário, kNN e termos têm o top-1 como primeiro)",
        "votos_ramo": "votos_ramo (quantas das cinco têm o primeiro no mesmo ramo do top-1)",
        "conc3": "concordância antiga (BM25, cosseno-tema e dicionário, sem o kNN)",
    }
    for chave, titulo in titulos.items():
        print(f"\n{titulo}:")
        for v in range(6):
            g = [r for r in reg if r[chave] == v]
            if g:
                print(f"  {v}: n={len(g):3d} | acerto tema {taxa(g, 'ok_tema'):.0%} | acerto ramo "
                      f"{taxa(g, 'ok_ramo'):.0%}")
    q = lambda v: np.round(np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]), 2) if v else []  # noqa: E731
    certos = [r["conf_tema"] for r in reg if r["ok_tema"]]
    errados = [r["conf_tema"] for r in reg if not r["ok_tema"]]
    print(f"\nconfiança do tema quando o top-1 acerta (n={len(certos)}): quantis 10/25/50/75/90 = {q(certos)}")
    print(f"confiança do tema quando o top-1 erra (n={len(errados)}): quantis 10/25/50/75/90 = {q(errados)}")
    print("\nmelhor opção (tema ou ramo) pela confiança, publicando acima do corte:")
    for corte in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        g = [r for r in reg if r["conf_max"] >= corte]
        if g:
            print(f"  >= {corte:.2f}: publicaria {len(g):3d} ({len(g) / n:.0%}), certas {sum(r['ok_melhor'] for r in g)}, "
                  f"precisão {taxa(g, 'ok_melhor'):.0%}, no ramo {sum(r['melhor_ramo'] for r in g)}")


if __name__ == "__main__":
    main()
