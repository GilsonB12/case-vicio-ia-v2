"""Dicionário lei-tema: chave normalizada de lei, artigo, diploma ou súmula -> temas que ela indica.

Construído por script (``sistema dicionario``) a partir de ``dados/temas_enriquecidos.jsonl``: cada lei
em ``leis`` vira ``lei:NNNN`` (ou ``dl:``/``diploma:``) e cada artigo em ``artigos`` vira ``art:N:diploma``,
apontando para o tema, com peso 1/n sobre os n temas que citam a chave. Depois entram as chaves de outras
disciplinas (``CHAVES_FORA``, marcadas ``fora``) e, por último, ``dados/dicionario_leis_ajustes.json``,
revisado à mão, que substitui por inteiro a entrada de cada chave que traz.

Formato em memória: ``{chave: {"nome": str, "temas": {tema_id: peso}, "fora": bool}}``, com ``tema_id``
inteiro. No arquivo os ids viram texto (JSON), e as chaves começadas por ``_`` são metadados.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from sistema.leis import CHAVES_FORA, NOMES, extrair_chaves, peso_fora, sufixo_artigo

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_DICIONARIO = RAIZ / "dados" / "dicionario_leis.json"
CAMINHO_AJUSTES = RAIZ / "dados" / "dicionario_leis_ajustes.json"
CAMINHO_ENRIQUECIDOS = RAIZ / "dados" / "temas_enriquecidos.jsonl"

_PREFIXOS_LEI = ("lei:", "dl:", "diploma:")


def _texto_lei(entrada: object) -> str:
    """Texto que o extrator entende para uma lei do tema: "11.340/2006" vira "Lei 11.340/2006"."""
    if isinstance(entrada, dict):
        numero = str(entrada.get("numero") or "").strip()
        nome = str(entrada.get("nome") or "").strip()
        if numero[:1].isdigit():
            decreto = "decreto" in f"{entrada.get('tipo') or ''} {nome}".lower()
            numero = f"{'Decreto-Lei' if decreto else 'Lei'} {numero}"
        return f"{numero} {nome}".strip()
    if isinstance(entrada, str):
        texto = entrada.strip()
        return f"Lei {texto}" if texto[:1].isdigit() else texto
    return ""


def _nome_lei(entrada: object, texto: str) -> str:
    if isinstance(entrada, dict) and entrada.get("nome"):
        return str(entrada["nome"]).strip()
    return texto


def _texto_artigo(entrada: object) -> str:
    """Texto de um artigo do tema: "157 do CP", {"artigo": "33", "diploma": "Lei 11.343/2006"}..."""
    if isinstance(entrada, dict):
        numero = str(entrada.get("artigo") or entrada.get("numero") or entrada.get("art") or "").strip()
        diploma = str(entrada.get("diploma") or entrada.get("lei") or "").strip()
        if diploma[:1].isdigit():
            diploma = f"Lei {diploma}"
        texto = f"{numero} do {diploma}" if diploma else numero
    elif isinstance(entrada, str):
        texto = entrada.strip()
    else:
        return ""
    if texto and not re.match(r"(?i)art", texto):
        texto = f"art. {texto}"
    return texto


def _chaves_do_tema(tema: dict) -> list[tuple[str, str]]:
    """(chave, nome) de cada lei e artigo do tema enriquecido, sem repetição."""
    achadas: dict[str, str] = {}
    for entrada in tema.get("leis") or []:
        texto = _texto_lei(entrada)
        for chave in extrair_chaves(texto):
            if chave.startswith(_PREFIXOS_LEI):
                achadas.setdefault(chave, _nome_lei(entrada, texto))
    leis_do_tema = [chave for chave in achadas if chave.startswith(_PREFIXOS_LEI)]
    padrao = sufixo_artigo(leis_do_tema[0]) if leis_do_tema else None
    for entrada in tema.get("artigos") or []:
        texto = _texto_artigo(entrada)
        for chave in extrair_chaves(texto, diploma_padrao=padrao):
            # artigo sem diploma ("art:14") é ambíguo demais para apontar tema
            if chave.startswith("art:") and chave.count(":") >= 2:
                achadas.setdefault(chave, texto)
    return list(achadas.items())


def _entrada_ajuste(chave: str, entrada: object, anterior: dict | None) -> dict:
    """Valida e normaliza uma entrada de ajuste; campos ausentes ficam com o padrão."""
    if not isinstance(entrada, dict):
        raise ValueError(f"ajuste de {chave!r}: a entrada precisa ser um objeto, veio {entrada!r}")
    temas_brutos = entrada.get("temas", {})
    if not isinstance(temas_brutos, dict):
        raise ValueError(f"ajuste de {chave!r}: 'temas' precisa ser um objeto {{tema_id: peso}}")
    temas: dict[int, float] = {}
    for tema_id, peso in temas_brutos.items():
        try:
            temas[int(tema_id)] = float(peso)
        except (TypeError, ValueError) as erro:
            raise ValueError(f"ajuste de {chave!r}: tema {tema_id!r} com peso {peso!r} inválido") from erro
    fora = entrada.get("fora", chave in CHAVES_FORA)
    if not isinstance(fora, bool):
        raise ValueError(f"ajuste de {chave!r}: 'fora' precisa ser true ou false, veio {fora!r}")
    nome = entrada.get("nome") or (anterior or {}).get("nome") or NOMES.get(chave) or chave
    return {"nome": str(nome), "temas": dict(sorted(temas.items())), "fora": fora}


def construir(enriquecidos: dict[int, dict], ajustes: dict) -> dict:
    """Monta o dicionário: temas enriquecidos (peso 1/n), chaves fora por padrão, ajustes por cima."""
    citantes: dict[str, set[int]] = defaultdict(set)
    nomes: dict[str, str] = {}
    for tema_id, tema in enriquecidos.items():
        for chave, nome in _chaves_do_tema(tema):
            citantes[chave].add(int(tema_id))
            nomes.setdefault(chave, nome)
    dicionario: dict[str, dict] = {}
    for chave, temas in citantes.items():
        peso = 1.0 / len(temas)
        dicionario[chave] = {"nome": nomes[chave], "temas": {t: peso for t in sorted(temas)}, "fora": False}
    for chave in CHAVES_FORA:
        dicionario.setdefault(chave, {"nome": NOMES.get(chave, chave), "temas": {}, "fora": True})["fora"] = True
    for chave, entrada in ajustes.items():
        if chave.startswith("_"):
            continue
        dicionario[chave] = _entrada_ajuste(chave, entrada, dicionario.get(chave))
    return dict(sorted(dicionario.items()))


def pontuar(chaves: Counter, dicionario: dict) -> tuple[dict[int, float], float]:
    """Pontos por tema (soma de peso × contagem das chaves) e ``peso_fora`` da questão.

    ``peso_fora`` soma as contagens das chaves marcadas ``fora`` no dicionário; a CF só conta quando é o
    único diploma do texto (regra de ``sistema.leis.peso_fora``).
    """
    pontos: dict[int, float] = defaultdict(float)
    marcadas_fora = set()
    for chave, contagem in chaves.items():
        entrada = dicionario.get(chave)
        if entrada is None or contagem <= 0:
            continue
        for tema_id, peso in entrada["temas"].items():
            pontos[int(tema_id)] += peso * contagem
        if entrada.get("fora"):
            marcadas_fora.add(chave)
    return dict(pontos), peso_fora(chaves, marcadas_fora)


def salvar(dicionario: dict, caminho: Path = CAMINHO_DICIONARIO, fonte: dict | None = None) -> None:
    saida: dict[str, object] = {"_fonte": fonte} if fonte else {}
    for chave, entrada in sorted(dicionario.items()):
        saida[chave] = {
            "nome": entrada["nome"],
            "temas": {str(t): round(float(p), 6) for t, p in sorted(entrada["temas"].items())},
            "fora": bool(entrada["fora"]),
        }
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(saida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def carregar(caminho: Path = CAMINHO_DICIONARIO) -> dict:
    bruto = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return {
        chave: {
            "nome": entrada.get("nome", chave),
            "temas": {int(t): float(p) for t, p in entrada.get("temas", {}).items()},
            "fora": bool(entrada.get("fora", False)),
        }
        for chave, entrada in bruto.items()
        if not chave.startswith("_")
    }


def ler_enriquecidos(caminho: Path = CAMINHO_ENRIQUECIDOS) -> dict[int, dict]:
    with open(caminho, encoding="utf-8") as arquivo:
        linhas = [json.loads(linha) for linha in arquivo if linha.strip()]
    return {int(linha["tema_id"]): linha for linha in linhas}


def ler_ajustes(caminho: Path = CAMINHO_AJUSTES) -> dict:
    caminho = Path(caminho)
    if not caminho.exists():
        return {}
    return json.loads(caminho.read_text(encoding="utf-8"))


def _relativo(caminho: Path) -> str:
    try:
        return caminho.resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return str(caminho)


def main(argumentos: list[str] | None = None) -> int:
    """``sistema dicionario``: reconstrói ``dados/dicionario_leis.json``. Devolve o código de saída."""
    parser = argparse.ArgumentParser(
        prog="sistema dicionario",
        description="Reconstrói o dicionário lei-tema a partir dos temas enriquecidos e dos ajustes revisados.",
    )
    parser.add_argument("--enriquecidos", type=Path, default=CAMINHO_ENRIQUECIDOS)
    parser.add_argument("--ajustes", type=Path, default=CAMINHO_AJUSTES)
    parser.add_argument("--saida", type=Path, default=CAMINHO_DICIONARIO)
    args = parser.parse_args([] if argumentos is None else argumentos)

    if not args.enriquecidos.exists():
        print(
            f"Arquivo de temas enriquecidos não encontrado: {args.enriquecidos}\n"
            "Gere-o na preparação (preparacao/README.md) e rode `sistema dicionario` de novo.",
            file=sys.stderr,
        )
        return 1
    if not args.ajustes.exists():
        print(f"Sem ajustes em {args.ajustes}: o dicionário sai só dos temas enriquecidos.")
    dicionario = construir(ler_enriquecidos(args.enriquecidos), ler_ajustes(args.ajustes))
    fonte = {
        "gerado_por": "sistema dicionario",
        "temas_enriquecidos": _relativo(args.enriquecidos),
        "ajustes": _relativo(args.ajustes),
        "data": date.today().isoformat(),
    }
    salvar(dicionario, args.saida, fonte=fonte)
    com_temas = sum(1 for entrada in dicionario.values() if entrada["temas"])
    fora = sum(1 for entrada in dicionario.values() if entrada["fora"])
    print(f"Dicionário com {len(dicionario)} chaves ({com_temas} com temas, {fora} fora) gravado em {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
