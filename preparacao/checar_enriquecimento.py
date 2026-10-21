"""Checagem de consistência do enriquecimento (gerado por modelo) contra a folha-verdade, sem modelo.

Para cada questão da folha com tema central, extrai as leis e os artigos citados na questão e confere se o tema
enriquecido do rótulo da equipe (ou o pai dele) cita a mesma lei. Grava as divergências em
dados/intermediarios/revisao_enriquecimento.json. Uso: uv run python preparacao/checar_enriquecimento.py
"""
from __future__ import annotations

import json
import sys

from sistema.arvore import ArvoreTemas
from sistema.config import CAMINHOS
from sistema.dados import carregar_folha, carregar_questoes, carregar_temas_enriquecidos
from sistema.leis import extrair_chaves, normalizar_lei


def leis_do_tema(tema_id: int, enriquecidos: dict, arvore: ArvoreTemas) -> set[str]:
    chaves: set[str] = set()
    for t in {tema_id, arvore.original_de(tema_id), arvore.pai(tema_id) or tema_id}:
        for lei in (enriquecidos.get(arvore.original_de(t)) or {}).get("leis", []):
            numero = lei.get("numero") if isinstance(lei, dict) else str(lei)
            if numero:
                chaves.add("lei:" + normalizar_lei(numero))
    return chaves


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    arvore = ArvoreTemas.carregar()
    enriquecidos = carregar_temas_enriquecidos()
    textos = dict(zip(carregar_questoes()["id"], carregar_questoes()["texto"]))
    com_lei = coerentes = 0
    divergencias = []
    for linha in carregar_folha().to_dict("records"):
        if linha["fora"] or not linha["tema_central_id"]:
            continue
        citadas = {k for k in extrair_chaves(textos[linha["id"]]) if k.startswith(("lei:", "dl:"))}
        citadas = {"lei:" + k.split(":", 1)[1] if k.startswith("dl:") else k for k in citadas}
        if not citadas:
            continue
        com_lei += 1
        alvos = [linha["tema_central_id"], *linha["extras_ids"]]
        do_tema = set().union(*(leis_do_tema(int(t), enriquecidos, arvore) for t in alvos))
        if citadas & do_tema:
            coerentes += 1
        else:
            divergencias.append({"id": linha["id"], "tema": int(linha["tema_central_id"]),
                                 "caminho": arvore.caminho(int(linha["tema_central_id"])),
                                 "leis_na_questao": sorted(citadas), "leis_no_tema": sorted(do_tema)})
    saida = CAMINHOS.intermediarios / "revisao_enriquecimento.json"
    saida.write_text(json.dumps({"questoes_com_lei": com_lei, "coerentes": coerentes, "divergencias": divergencias},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"questões da folha que citam lei: {com_lei}; o tema da equipe (central ou extra, ou o pai) cita a mesma lei "
          f"no enriquecimento: {coerentes} ({coerentes / max(com_lei, 1):.0%}); divergências em {saida}")
    for d in divergencias[:12]:
        print(f"  {d['id']} {d['caminho'][:55]} | questão {d['leis_na_questao']} | tema {d['leis_no_tema'][:4]}")


if __name__ == "__main__":
    main()
