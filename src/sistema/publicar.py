"""Saídas do case: `publicacao.csv` e `previsao.txt`, no formato de `anexos/LEIA-ME.md`."""
from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from sistema.decisao import DECISOES, FORA, PUBLICAR, REVISAR, Decisao

if TYPE_CHECKING:
    from sistema.arvore import ArvoreTemas

COLUNAS_PUBLICACAO = ["id", "decisao", "tema_central_id", "extras_ids", "confianca", "motivo"]


def linha_publicacao(decisao: Decisao) -> dict[str, str]:
    """Uma decisão como linha do CSV: extras separados por `;`, confiança com 2 casas, central vazio em `fora`."""
    return {
        "id": decisao.id,
        "decisao": decisao.decisao,
        "tema_central_id": "" if decisao.tema_central_id is None else str(int(decisao.tema_central_id)),
        "extras_ids": ";".join(str(int(t)) for t in decisao.extras_ids),
        "confianca": f"{float(decisao.confianca):.2f}",
        "motivo": decisao.motivo or "",
    }


def escrever_publicacao(decisoes: Iterable[Decisao], caminho: Path) -> pd.DataFrame:
    """Grava `publicacao.csv` (UTF-8, vírgula, aspas quando precisa), uma linha por decisão, na ordem dada."""
    linhas = [linha_publicacao(decisao) for decisao in decisoes]
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_PUBLICACAO, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(linhas)
    return pd.DataFrame(linhas, columns=COLUNAS_PUBLICACAO)


def ler_publicacao(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho, dtype=str, keep_default_na=False)


def _inteiro(texto: str) -> int | None:
    try:
        return int(str(texto).strip())
    except ValueError:
        return None


def validar_publicacao(df: pd.DataFrame, ids_esperados: Sequence[str], arvore: "ArvoreTemas") -> list[str]:
    """Erros do CSV contra o LEIA-ME; lista vazia quando está válido.

    Confere colunas, ids (faltando, a mais, repetidos, ordem), decisão, central obrigatório em publicar e revisar e
    vazio em fora, temas publicáveis (central e extras), confiança entre 0 e 1 e motivo em revisar e fora.
    """
    erros: list[str] = []
    if list(df.columns) != COLUNAS_PUBLICACAO:
        return [f"colunas {list(df.columns)}; o LEIA-ME pede {COLUNAS_PUBLICACAO}"]
    df = df.fillna("").astype(str)
    ids = [i.strip() for i in df["id"]]
    esperados = [str(i) for i in ids_esperados]
    faltando = sorted(set(esperados) - set(ids))
    a_mais = sorted(set(ids) - set(esperados))
    repetidos = sorted(i for i, n in Counter(ids).items() if n > 1)
    if faltando:
        erros.append(f"{len(faltando)} id(s) faltando: {', '.join(faltando[:10])}")
    if a_mais:
        erros.append(f"{len(a_mais)} id(s) a mais: {', '.join(a_mais[:10])}")
    if repetidos:
        erros.append(f"id(s) repetido(s): {', '.join(repetidos[:10])}")
    if not (faltando or a_mais or repetidos) and ids != esperados:
        erros.append("as linhas não estão na ordem das questões")
    for linha in df.to_dict("records"):
        id_q, decisao = linha["id"].strip(), linha["decisao"].strip()
        central, motivo = linha["tema_central_id"].strip(), linha["motivo"].strip()
        if decisao not in DECISOES:
            erros.append(f"{id_q}: decisão inválida {decisao!r} (use {', '.join(DECISOES)})")
            continue
        if decisao in (PUBLICAR, REVISAR):
            tema = _inteiro(central)
            if not central:
                erros.append(f"{id_q}: tema central ausente em {decisao}")
            elif tema is None or not arvore.publicavel(tema):
                erros.append(f"{id_q}: tema central {central} não é publicável")
        elif central:
            erros.append(f"{id_q}: tema central {central} presente em fora (precisa ficar vazio)")
        for extra in [e.strip() for e in linha["extras_ids"].split(";") if e.strip()]:
            tema = _inteiro(extra)
            if tema is None or not arvore.publicavel(tema):
                erros.append(f"{id_q}: extra {extra} não é publicável")
        try:
            confianca = float(linha["confianca"])
            if not 0 <= confianca <= 1:
                erros.append(f"{id_q}: confiança {linha['confianca']} fora de 0 a 1")
        except ValueError:
            erros.append(f"{id_q}: confiança {linha['confianca']!r} não é número")
        if decisao in (REVISAR, FORA) and not motivo:
            erros.append(f"{id_q}: motivo ausente em {decisao}")
    return erros


# --- previsao.txt -------------------------------------------------------------------------------------------


def _faixa(intervalo: Sequence[float]) -> str:
    baixo, alto = intervalo
    return f"{float(baixo):.2f}-{float(alto):.2f}"


def _pct(valor: float) -> str:
    return f"{100 * float(valor):.1f}%".replace(".", ",")


def _num(valor: float) -> str:
    return f"{float(valor):.2f}".replace(".", ",")


def linhas_previsao(resumo: dict) -> list[str]:
    """As 4 linhas exigidas pelo LEIA-ME, com 2 casas."""
    return [
        f"precisao_prevista: {float(resumo['precisao']):.2f}",
        f"precisao_intervalo: {_faixa(resumo['precisao_intervalo'])}",
        f"cobertura_prevista: {float(resumo['cobertura']):.2f}",
        f"cobertura_intervalo: {_faixa(resumo['cobertura_intervalo'])}",
    ]


_DEGRAUS = {
    0: "degrau 0 (precisão prevista de pelo menos 0,97 e limite inferior de pelo menos 0,95, com a maior cobertura)",
    1: "degrau 1 (nenhum par de limiares chegou ao degrau 0; vale só o limite inferior de pelo menos 0,95, com a "
       "maior cobertura)",
    2: "degrau 2 (nenhum par chegou aos degraus 0 e 1; vale só a precisão prevista de pelo menos 0,97, com a maior "
       "cobertura)",
    3: "degrau 3 (nenhum par chegou aos degraus 0 a 2; fica o par de maior limite inferior)",
}


def texto_previsao(resumo: dict) -> str:
    """Explicação em português de como os números de `previsao.txt` foram obtidos."""
    n_rotulos = resumo.get("n_rotulos", 100)
    n_dobras = resumo.get("n_dobras", 5)
    repeticoes = resumo.get("repeticoes", 3)
    total = n_dobras * repeticoes
    n_teste = resumo.get("n_teste_dobras", n_rotulos * repeticoes)
    n_avaliadas = resumo.get("n_avaliadas", 559)
    lim = resumo["limiares"]
    publicadas_dobras = resumo.get("publicadas_dobras")
    acertos_dobras = resumo.get("acertos_dobras")
    n_internas = resumo.get("n_dobras_confianca", 5)
    n_efetivo = resumo.get("n_efetivo")
    if n_efetivo is None and publicadas_dobras is not None:
        n_efetivo = publicadas_dobras / max(int(repeticoes), 1)
    partes = [
        "Como chegamos nesses números",
        "",
        f"A folha-verdade ({n_rotulos} questões rotuladas pela equipe) é a única fonte de avaliação e de "
        "calibração: a precisão e os limiares são medidos só contra esses rótulos. Os exemplos sintéticos "
        "ajudam a treinar, mas nunca contam como acerto.",
        "",
        f"Validação cruzada: {n_dobras} dobras estratificadas por (fora da taxonomia, disciplina do tema central), "
        f"repetidas {repeticoes} vezes com sementes diferentes, {total} avaliações ao todo. Em cada dobra, o índice "
        "de vizinhos rotulados, o modelo de pares, o detector de fora e o modelo de confiança são refeitos só com as "
        "questões de treino da dobra (mais os exemplos sintéticos, e o corpus quando ligado); as questões de teste "
        "da dobra não entram em nada do treino.",
        "",
        "Confiança: um segundo modelo (regressão logística) dá a chance de publicar certo em duas opções por "
        "questão, o tema mais provável do modelo de pares e o ramo de nível 1 dele, a partir da probabilidade, da "
        "concordância entre as quatro buscas independentes (BM25, cosseno com o tema, dicionário de leis e vizinhos "
        "rotulados), do nível do tema, da similaridade com a folha de treino e da presença de lei citada. Ele "
        f"aprende com previsões fora da amostra: dentro de cada treino, uma validação interna de {n_internas} dobras "
        "refaz o "
        "modelo de pares sem as questões que vai prever. Publica-se a opção de maior confiança quando ela passa do "
        "limiar; é essa confiança que vai na coluna confianca do publicacao.csv.",
        "",
        f"Limiares: a grade cruza o limiar da confiança (0,50 a 0,99, de 0,01 em 0,01) com o limiar de fora (0,3 a "
        "0,9). Para cada par, as publicadas e os acertos das dobras são somados; como cada questão aparece uma vez "
        f"por repetição, as contagens são divididas pelas {repeticoes} repetições (n efetivo) antes da média de "
        "Jeffreys (acertos + 0,5) / (publicadas + 1) e do intervalo de 95% por 10.000 sorteios da Beta. A escolha "
        "segue a "
        f"escada: {_DEGRAUS.get(int(resumo['degrau']), 'degrau ' + str(resumo['degrau']))}. Ficou no "
        f"degrau {resumo['degrau']}: publicar a partir de {_num(lim['publicar'])}, fora a partir de "
        f"{_num(lim['fora'])}, extras a partir de {_num(lim.get('extra', 0.15))}.",
        "",
    ]
    contagem_dobras = ""
    if publicadas_dobras is not None and acertos_dobras is not None:
        efetivo = f"; n efetivo {_num(n_efetivo)} publicadas" if n_efetivo is not None else ""
        contagem_dobras = (f" ({acertos_dobras} certas em {publicadas_dobras} publicadas, somando as {n_teste} "
                           f"questões de teste das dobras{efetivo})")
    partes += [
        f"Precisão prevista: a precisão agrupada das dobras no par de limiares escolhido{contagem_dobras}, "
        f"{_num(resumo['precisao'])}, com intervalo {_faixa(resumo['precisao_intervalo']).replace('.', ',')}. "
        "Acerto pela régua completa: tema central, extra, ou pai ou filho de um deles; publicar uma questão fora "
        "da taxonomia conta como erro.",
        "",
        f"Cobertura prevista: a fração publicada observada nas {n_avaliadas} questões fora da folha, com o modelo "
        f"final treinado nos {n_rotulos} rótulos e os mesmos limiares: {resumo.get('publicadas_avaliadas', '?')} "
        f"publicadas, {_num(resumo['cobertura'])}; o intervalo é ±0,01 em torno dela. Como conferência, a cobertura "
        f"das dobras no mesmo par foi {_num(resumo['cobertura_dobras'])}.",
        "",
    ]
    comparacao = resumo.get("comparacao")
    if comparacao:
        sem, com = comparacao["sem_corpus"], comparacao["com_corpus"]
        partes += [
            "Comparação de fontes: a validação rodou sem e com o corpus da etapa 1 como vizinho rotulado. Regra "
            "fixada antes de ver os números: o corpus fica se a cobertura das dobras subir pelo menos 3 pontos sem "
            "baixar a precisão prevista nem o limite inferior. Sem corpus: precisão "
            f"{_num(sem['precisao'])} (limite inferior {_num(sem['limite_inferior'])}), cobertura "
            f"{_pct(sem['cobertura'])}. Com corpus: precisão {_num(com['precisao'])} (limite inferior "
            f"{_num(com['limite_inferior'])}), cobertura {_pct(com['cobertura'])}. Resultado: "
            f"{'corpus aprovado' if comparacao['corpus_aprovado'] else 'corpus fora'} ({comparacao['motivo']}).",
            "",
        ]
    else:
        fonte = "com o corpus da etapa 1" if resumo.get("com_corpus") else "sem o corpus da etapa 1"
        partes += [f"Comparação de fontes: não rodou nesta execução; a fonte foi escolhida na linha de comando "
                   f"({fonte}).", ""]
    partes += [
        "O que pode deixar a previsão otimista:",
        f"- são só {n_rotulos} rótulos; o intervalo da precisão é largo e cada erro pesa;",
        "- o limiar foi escolhido na mesma folha que mede a precisão, o que favorece o par escolhido;",
        f"- as {repeticoes} repetições reaproveitam as mesmas {n_rotulos} questões; o intervalo usa o n efetivo "
        "(contagens divididas pelas repetições), mas as dobras ainda compartilham questões de treino;",
        "- os exemplos sintéticos entram como vizinhos rotulados e no treino; se eles forem mais fáceis que as "
        "questões reais, o modelo parece melhor do que é.",
    ]
    n_recentes = int(resumo.get("n_recentes") or 0)
    if n_recentes:
        partes[-1] = partes[-1][:-1] + ";"
        partes.append(f"- as {n_avaliadas} avaliadas incluem {n_recentes} questões de concursos recentes, de fora do "
                      "banco de onde saiu a folha; se elas forem diferentes, a precisão pode cair.")
    return "\n".join(partes) + "\n"


def escrever_previsao(caminho: Path, resumo: dict) -> str:
    """Grava `previsao.txt`: as 4 linhas do LEIA-ME e depois a explicação. Devolve o texto gravado."""
    texto = "\n".join(linhas_previsao(resumo)) + "\n\n" + texto_previsao(resumo)
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(texto, encoding="utf-8", newline="\n")
    return texto
