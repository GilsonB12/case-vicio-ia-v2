"""Linha de comando: `uv run sistema <subcomando>`."""
from __future__ import annotations

import argparse
import importlib
import logging

from sistema import custos
from sistema.embeddings import CacheEmbeddings, textos_do_sistema


def ler_valor_painel(texto: str) -> tuple[str, float]:
    """PROVEDOR=VALOR em dólares, como no painel do provedor; aceita vírgula decimal."""
    provedor, separador, valor = texto.partition("=")
    if not separador or not provedor.strip():
        raise argparse.ArgumentTypeError(f"use PROVEDOR=VALOR, por exemplo openai=0.70 (recebido {texto!r})")
    try:
        return provedor.strip(), float(valor.strip().replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError(f"valor do painel não é um número: {valor!r}") from None


def _montar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sistema",
        description="Classificação de questões jurídicas em temas, sem modelo de linguagem na decisão (case Vício).",
    )
    sub = parser.add_subparsers(dest="comando", required=True, metavar="subcomando")

    p = sub.add_parser("tudo", help="regenera publicacao.csv e previsao.txt a partir dos anexos e de dados/, sem rede",
                       description="Sem --com-corpus nem --sem-corpus, a validação roda com e sem o corpus da etapa 1, "
                                   "registra o veredito da regra dos 3 pontos e usa a fonte escolhida à mão "
                                   "(com corpus; ver pipeline.USAR_CORPUS_NA_ENTREGA).")
    fonte = p.add_mutually_exclusive_group()
    fonte.add_argument("--com-corpus", dest="com_corpus", action="store_const", const=True, default=None,
                       help="usa o corpus da etapa 1 como vizinho rotulado, sem comparar fontes")
    fonte.add_argument("--sem-corpus", dest="com_corpus", action="store_const", const=False,
                       help="não usa o corpus da etapa 1, sem comparar fontes")

    sub.add_parser("embeddings", help="grava no cache os embeddings que faltam (a única etapa que chama a API)")

    # Sem ajuda própria: tudo depois de `dicionario`, inclusive --help, vai para sistema.dicionario.main.
    sub.add_parser("dicionario", add_help=False,
                   help="reconstrói dados/dicionario_leis.json a partir dos temas enriquecidos e dos ajustes")

    p = sub.add_parser("avaliar-folha", help="validação cruzada na folha-verdade e confusão por tipo de erro")
    p.add_argument("--com-corpus", action="store_true", help="usa o corpus da etapa 1 como vizinho rotulado")

    sub.add_parser("comparar-fontes", help="validação com e sem o corpus da etapa 1 e a regra dos 3 pontos")

    p = sub.add_parser("custos", help="resume o livro-razão de custos por execução, etapa e modelo",
                       description="Lê dados/custos/chamadas.jsonl, imprime o resumo e grava custos_resumo.json ao lado.")
    p.add_argument("--painel", action="append", type=ler_valor_painel, default=[], metavar="PROVEDOR=USD",
                   help="valor cobrado no painel do provedor, para comparar com o calculado; repetível, "
                        "por exemplo --painel openai=0.70")
    return parser


def _embeddings() -> int:
    cache = CacheEmbeddings()
    textos = list(dict.fromkeys(textos_do_sistema()))
    ausentes = cache.faltando(textos)
    print(f"{len(textos)} texto(s) do sistema; {len(textos) - len(ausentes)} já no cache {cache.pasta}; "
          f"{len(ausentes)} a gerar")
    if ausentes:
        gerados = cache.gerar(ausentes)
        print(f"{gerados} embedding(s) gravado(s); chamadas registradas em {custos.LIVRO.caminho}")
    return 0


def _dicionario(resto: list[str]) -> int:
    try:
        modulo = importlib.import_module("sistema.dicionario")
    except ModuleNotFoundError as erro:
        if erro.name != "sistema.dicionario":
            raise
        raise SystemExit("sistema dicionario indisponível: o módulo sistema.dicionario ainda não existe "
                         "(chega na tarefa T2)") from None
    principal = getattr(modulo, "main", None)
    if principal is None:
        raise SystemExit("sistema dicionario indisponível: sistema.dicionario não tem a função main")
    resultado = principal(resto)
    return 0 if resultado is None else int(resultado)


def _custos(painel: list[tuple[str, float]]) -> int:
    livro = custos.LIVRO.caminho
    resumo = custos.resumir_custos(livro, painel=dict(painel), saida=livro.with_name("custos_resumo.json"))
    print(custos.formatar_resumo(resumo))
    return 0


def main(argumentos: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _montar_parser()
    args, resto = parser.parse_known_args(argumentos)
    if args.comando == "dicionario":
        return _dicionario(resto)
    if resto:
        parser.error(f"argumento(s) não reconhecido(s): {' '.join(resto)}")
    if args.comando in ("tudo", "avaliar-folha", "comparar-fontes"):
        from sistema import pipeline

        if args.comando == "tudo":
            pipeline.tudo(com_corpus=args.com_corpus)
        elif args.comando == "avaliar-folha":
            pipeline.avaliar_folha(com_corpus=args.com_corpus)
        else:
            pipeline.comparar()
        return 0
    if args.comando == "embeddings":
        return _embeddings()
    if args.comando == "custos":
        return _custos(args.painel)
    parser.error(f"subcomando desconhecido: {args.comando}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
