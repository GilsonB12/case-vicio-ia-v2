"""Consolida as saídas dos lotes de enriquecimento (geradas por modelos de linguagem, antes da decisão) em
dados/temas_enriquecidos.jsonl e dados/exemplos.jsonl, validando cada linha e listando o que faltou.

Uso: uv run python preparacao/consolidar_enriquecimento.py <pasta_dos_lotes> [--fora <exemplos_fora.jsonl>]

Cada lote é um .jsonl com uma linha por tema no formato do prompt em preparacao/prompts/enriquecer_temas.md.
Lotes "enriq_lote_NN.jsonl" vêm do Codex (gpt-6-astra); lotes "enriq_lote_NN_o.jsonl" vêm de Claude Opus 5.5; quando
os dois existem para o mesmo número, vale o que tiver mais linhas válidas. Linhas inválidas são contadas e os temas
sem linha válida saem em <pasta>/faltando.json para uma rodada de reparo.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = Path(__file__).resolve().parents[1]
CHAVES = ("tema_id", "resumo", "leis", "artigos", "termos", "sinonimos", "exemplos")


def ler_lote(caminho: Path) -> tuple[dict[int, dict], int]:
    validas: dict[int, dict] = {}
    invalidas = 0
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip().strip("`")
        if not linha.startswith("{") or "marcador" in linha[:40]:
            continue
        try:
            d = json.loads(linha)
        except json.JSONDecodeError:
            invalidas += 1
            continue
        if any(k not in d for k in CHAVES) or not isinstance(d["tema_id"], int):
            invalidas += 1
            continue
        if not isinstance(d["exemplos"], list) or len(d["exemplos"]) < 2 or len(d.get("termos", [])) < 8:
            invalidas += 1
            continue
        validas[int(d["tema_id"])] = d
    return validas, invalidas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pasta")
    parser.add_argument("--fora", default=None, help="jsonl com exemplos fora da taxonomia")
    args = parser.parse_args()
    pasta = Path(args.pasta)

    temas = pd.read_csv(RAIZ / "anexos" / "temas.csv")
    originais = temas[(temas.status == "active") & (temas.nivel > 0) & ~temas.tema_id.between(6716, 6738)]
    esperados = set(int(t) for t in originais.tema_id)

    por_lote: dict[str, tuple[dict[int, dict], str]] = {}
    for caminho in sorted(pasta.glob("enriq_lote_*.jsonl")):
        m = re.match(r"enriq_lote_(\d+)(_o)?\.jsonl", caminho.name)
        if not m:
            continue
        numero, sufixo = m.group(1), m.group(2)
        modelo = "Claude Opus 5.5 (via Claude Code)" if sufixo else "Codex CLI, gpt-6-astra"
        validas, invalidas = ler_lote(caminho)
        print(f"{caminho.name}: {len(validas)} válidas, {invalidas} inválidas")
        atual = por_lote.get(numero)
        if atual is None or len(validas) > len(atual[0]):
            por_lote[numero] = (validas, modelo)

    enriquecidos: dict[int, dict] = {}
    for numero, (validas, modelo) in por_lote.items():
        for tema_id, d in validas.items():
            if tema_id in esperados and tema_id not in enriquecidos:
                d["fonte"] = {"modelo": modelo, "data": str(date.today()), "lote": numero, "revisado": False}
                enriquecidos[tema_id] = d

    faltando = sorted(esperados - set(enriquecidos))
    print(f"\ntemas enriquecidos: {len(enriquecidos)} de {len(esperados)}; faltando: {len(faltando)}")
    (pasta / "faltando.json").write_text(json.dumps(faltando), encoding="utf-8")

    saida_temas = RAIZ / "dados" / "temas_enriquecidos.jsonl"
    with saida_temas.open("w", encoding="utf-8", newline="\n") as f:
        for tema_id in sorted(enriquecidos):
            d = enriquecidos[tema_id]
            registro = {k: d[k] for k in CHAVES if k != "exemplos"}
            registro["exemplos_enunciado"] = d["exemplos"]
            registro["fonte"] = d["fonte"]
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    saida_ex = RAIZ / "dados" / "exemplos.jsonl"
    n = 0
    with saida_ex.open("w", encoding="utf-8", newline="\n") as f:
        for tema_id in sorted(enriquecidos):
            d = enriquecidos[tema_id]
            for ex in d["exemplos"]:
                if not isinstance(ex, str) or len(ex.split()) < 20:
                    continue
                n += 1
                f.write(json.dumps({"id": f"S{n:04d}", "tema_id": tema_id, "fora": False, "texto": ex.strip(),
                                    "fonte": d["fonte"]}, ensure_ascii=False) + "\n")
        if args.fora:
            for linha in Path(args.fora).read_text(encoding="utf-8").splitlines():
                linha = linha.strip()
                if not linha.startswith("{"):
                    continue
                d = json.loads(linha)
                n += 1
                f.write(json.dumps({"id": f"S{n:04d}", "tema_id": None, "fora": True, "texto": d["texto"].strip(),
                                    "disciplina": d.get("disciplina"),
                                    "fonte": {"modelo": "Claude Fable 5.1 (via Claude Code)", "data": str(date.today()),
                                              "lote": "fora", "revisado": False}}, ensure_ascii=False) + "\n")
    print(f"exemplos gravados: {n} em {saida_ex}")
    print(f"temas gravados em {saida_temas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
