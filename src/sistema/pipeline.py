"""`sistema tudo`, `avaliar-folha` e `comparar-fontes`: do contexto às saídas, sem rede e sem modelo de linguagem.

`tudo` carrega o contexto (embeddings só do cache), escolhe a fonte de vizinhos (regra dos 3 pontos, quando não
vem da linha de comando), valida na folha, treina o modelo final com os 100 rótulos, decide as 659 questões e
grava `publicacao.csv`, `previsao.txt` e os intermediários em `dados/intermediarios/`.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from sistema import config
from sistema.calibracao import (
    N_DOBRAS_CONFIANCA,
    Aplicacao,
    comparar_fontes,
    resumir_grade,
    treinar_rodada,
    validar,
)
from sistema.config import Caminhos
from sistema.contexto import Contexto, carregar_contexto
from sistema.dados import escrever_jsonl
from sistema.decisao import DECISOES, PUBLICAR, Decisao, decidir_listas
from sistema.modelo import gatilhos_fora
from sistema.publicar import escrever_previsao, escrever_publicacao, ler_publicacao, validar_publicacao

MARGEM_COBERTURA = 0.01
PRIMEIRA_RECENTE = 360  # Q0360 a Q0659 são de concursos recentes (LEIA-ME)


class PublicacaoInvalidaError(RuntimeError):
    """O `publicacao.csv` gerado não passou na validação contra o LEIA-ME."""


def _numero_da_questao(id_questao: str) -> int:
    digitos = "".join(c for c in str(id_questao) if c.isdigit())
    return int(digitos) if digitos else 0


def _caminhos(caminhos: Caminhos | None) -> Caminhos:
    return caminhos or config.CAMINHOS


def _json(valor):
    """Converte o que vier do pandas e do NumPy para tipos que o `json` grava."""
    if isinstance(valor, Mapping):
        return {str(k): _json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, set)):
        return [_json(v) for v in valor]
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, float) and np.isnan(valor):
        return None
    return valor


def gravar_json(caminho: Path, dados) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(_json(dados), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def decidir_todas(contexto: Contexto, aplicacao: Aplicacao, limiares) -> list[Decisao]:
    """Decisão de cada questão, na ordem das questões do contexto."""
    decisoes = []
    for id_q in contexto.questoes["id"]:
        id_q = str(id_q)
        temas, _ = aplicacao.listas(id_q)
        decisoes.append(decidir_listas(id_q, temas, aplicacao.opcoes_de(id_q), float(aplicacao.p_fora.get(id_q, 0.0)),
                                       limiares, contexto.arvore, gatilhos_fora(contexto, id_q), ordenados=True))
    return decisoes


# --- intermediários -----------------------------------------------------------------------------------------


def _gravar_intermediarios(pasta: Path, contexto: Contexto, aplicacao: Aplicacao, validacao: dict,
                           comparacao: dict | None, rodada, resumo: dict) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    escrever_jsonl(pasta / "candidatos.jsonl", [
        {"id": id_q, "candidatos": [
            {"tema_id": c.tema_id, "origens": sorted(c.origens), "pontuacao": round(float(c.pontuacao), 6)}
            for c in aplicacao.candidatos.get(id_q, [])
        ]}
        for id_q in aplicacao.ids
    ])
    aplicacao.pares.to_csv(pasta / "caracteristicas.csv", index=False, float_format="%.6f")
    linhas_prob = []
    tabela_q = aplicacao.tabela_q.set_index("id")
    s_por_par = {(str(i), int(t)): float(s) for i, t, s in
                 zip(aplicacao.prob["id"], aplicacao.prob["tema_id"], aplicacao.prob["s"])}
    for id_q in aplicacao.ids:
        temas, ramos = aplicacao.listas(id_q)
        linhas_prob.append({
            "id": id_q,
            "temas": [{"tema_id": t, "s": round(s_por_par[(id_q, t)], 6), "p": round(p, 6)} for t, p in temas],
            "ramos": [{"tema_id": t, "p_ramo": round(p, 6)} for t, p in ramos],
            "opcoes": [{k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
                        for k, v in linha.items() if k != "id"} for linha in aplicacao.linhas_opcoes(id_q)],
            "p_fora": round(float(aplicacao.p_fora.get(id_q, 0.0)), 6),
            "caracteristicas_fora": {k: round(float(v), 6) for k, v in tabela_q.loc[id_q].items()},
        })
    escrever_jsonl(pasta / "probabilidades.jsonl", linhas_prob)
    validacao["avaliacao"].to_csv(pasta / "avaliacao_folha.csv", index=False)
    validacao["grade"].to_csv(pasta / "grade.csv", index=False, float_format="%.6f")
    gravar_json(pasta / "limiares.json", {**asdict(validacao["limiares"]), "degrau": validacao["degrau"],
                                          "com_corpus": validacao["com_corpus"]})
    if comparacao is not None:
        gravar_json(pasta / "comparacao_fontes.json", {k: v for k, v in comparacao.items() if k != "validacoes"})
    else:
        gravar_json(pasta / "comparacao_fontes.json", {
            "executada": False,
            "motivo": "fonte escolhida na linha de comando",
            "com_corpus": validacao["com_corpus"],
            "resumo": validacao["resumo"],
        })
    gravar_json(pasta / "pesos_modelo.json", {"pares": rodada.modelo.descrever(), "fora": rodada.detector.descrever(),
                                              "confianca": rodada.confianca.descrever()})
    gravar_json(pasta / "resumo.json", resumo)


# --- comandos ------------------------------------------------------------------------------------------------


def tudo(com_corpus: bool | None = None, caminhos: Caminhos | None = None, imprimir: bool = True) -> dict:
    """Regenera `publicacao.csv` e `previsao.txt`; `com_corpus=None` decide o corpus pela regra dos 3 pontos."""
    inicio = time.perf_counter()
    caminhos = _caminhos(caminhos)
    comparacao = None
    if com_corpus is None:
        contexto = carregar_contexto(com_corpus=True, caminhos=caminhos)
        comparacao = comparar_fontes(contexto, contexto)
        usar_corpus = bool(comparacao["corpus_aprovado"])
        validacao = comparacao["validacoes"]["com_corpus" if usar_corpus else "sem_corpus"]
    else:
        usar_corpus = bool(com_corpus)
        contexto = carregar_contexto(com_corpus=usar_corpus, caminhos=caminhos)
        validacao = validar(contexto, usar_corpus)
    limiares = validacao["limiares"]

    rodada = treinar_rodada(contexto, contexto.ids_folha, usar_corpus)
    ids_questoes = [str(i) for i in contexto.questoes["id"]]
    aplicacao = rodada.aplicar(ids_questoes)
    decisoes = decidir_todas(contexto, aplicacao, limiares)

    escrever_publicacao(decisoes, caminhos.publicacao)
    erros = validar_publicacao(ler_publicacao(caminhos.publicacao), ids_questoes, contexto.arvore)
    if erros:
        raise PublicacaoInvalidaError(f"{len(erros)} erro(s) no publicacao.csv: " + "; ".join(erros[:10]))

    folha = set(contexto.ids_folha)
    avaliadas = [d for d in decisoes if d.id not in folha]
    publicadas_avaliadas = sum(d.decisao == PUBLICAR for d in avaliadas)
    cobertura = publicadas_avaliadas / len(avaliadas) if avaliadas else 0.0
    escolhida = validacao["escolhida"]
    resumo = {
        "precisao": validacao["precisao"],
        "precisao_intervalo": validacao["precisao_intervalo"],
        "cobertura": cobertura,
        "cobertura_intervalo": [max(0.0, cobertura - MARGEM_COBERTURA), min(1.0, cobertura + MARGEM_COBERTURA)],
        "cobertura_dobras": validacao["cobertura"],
        "publicadas_dobras": int(escolhida["publicadas"]),
        "acertos_dobras": int(escolhida["acertos"]),
        "n_teste_dobras": int(escolhida["n"]),
        "n_efetivo": float(escolhida["publicadas_efetivas"]),
        "acertos_efetivos": float(escolhida["acertos_efetivos"]),
        "degrau": validacao["degrau"],
        "limiares": asdict(limiares),
        "com_corpus": usar_corpus,
        "n_rotulos": len(folha),
        "n_dobras": validacao["n_dobras"],
        "n_dobras_confianca": N_DOBRAS_CONFIANCA,
        "repeticoes": validacao["repeticoes"],
        "n_questoes": len(decisoes),
        "n_avaliadas": len(avaliadas),
        "n_recentes": sum(_numero_da_questao(d.id) >= PRIMEIRA_RECENTE for d in avaliadas),
        "publicadas_avaliadas": publicadas_avaliadas,
        "contagens": {d: sum(x.decisao == d for x in decisoes) for d in DECISOES},
        "contagens_avaliadas": {d: sum(x.decisao == d for x in avaliadas) for d in DECISOES},
        "erros_dobras": validacao["erros"],
        "comparacao": ({k: v for k, v in comparacao.items() if k != "validacoes"} if comparacao else None),
    }
    escrever_previsao(caminhos.previsao, resumo)
    _gravar_intermediarios(caminhos.intermediarios, contexto, aplicacao, validacao, comparacao, rodada, resumo)
    resumo["tempo_segundos"] = round(time.perf_counter() - inicio, 1)
    gravar_json(caminhos.intermediarios / "resumo.json", resumo)
    if imprimir:
        print(formatar_resumo(resumo, caminhos))
    return resumo


def avaliar_folha(com_corpus: bool = False, caminhos: Caminhos | None = None, imprimir: bool = True) -> dict:
    """Só a validação cruzada na folha: grade resumida e matriz de tipos de erro."""
    inicio = time.perf_counter()
    contexto = carregar_contexto(com_corpus=com_corpus, caminhos=_caminhos(caminhos))
    validacao = validar(contexto, com_corpus)
    if imprimir:
        print(formatar_validacao(validacao))
        print(f"\nTempo: {time.perf_counter() - inicio:.1f} s")
    return validacao


def comparar(caminhos: Caminhos | None = None, imprimir: bool = True) -> dict:
    """Validação sem e com o corpus da etapa 1 e a regra dos 3 pontos."""
    inicio = time.perf_counter()
    contexto = carregar_contexto(com_corpus=True, caminhos=_caminhos(caminhos))
    comparacao = comparar_fontes(contexto, contexto)
    if imprimir:
        print(formatar_comparacao(comparacao))
        print(f"\nTempo: {time.perf_counter() - inicio:.1f} s")
    return comparacao


# --- texto para o terminal -----------------------------------------------------------------------------------


def _n(valor: float) -> str:
    return f"{float(valor):.3f}"


def formatar_validacao(validacao: Mapping) -> str:
    lim = validacao["limiares"]
    fonte = "com corpus" if validacao["com_corpus"] else "sem corpus"
    linhas = [
        f"Validação cruzada na folha ({fonte}): {len(validacao['dobras'])} dobras, "
        f"{len(validacao['resultados'])} avaliações de teste",
        f"Escolha: confiança ≥ {lim.publicar:.2f}, fora ≥ {lim.fora:.2f} (degrau {validacao['degrau']}); "
        f"precisão {_n(validacao['precisao'])} [{_n(validacao['precisao_intervalo'][0])}, "
        f"{_n(validacao['precisao_intervalo'][1])}] com n efetivo {validacao['n_efetivo']:.1f}, "
        f"cobertura {_n(validacao['cobertura'])}",
        "",
        f"Grade no limiar de fora {lim.fora:.2f}:",
    ]
    resumo = resumir_grade(validacao["grade"], lim)
    colunas = ["publicar", "publicadas", "acertos", "publicadas_efetivas", "cobertura", "precisao", "limite_inferior",
               "fora_certos", "fora_errados", "fora_perdidos"]
    linhas.append(resumo[colunas].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    erros = validacao["erros"]
    linhas += [
        "",
        "Publicadas erradas por tipo: "
        f"irmão (mesmo ramo, alvo não aceito) {erros['irmao']}, outro ramo {erros['outro_ramo']}, "
        f"fora publicada {erros['fora_publicada']}",
    ]
    return "\n".join(linhas)


COLUNAS_COMPARACAO = ("publicar", "fora", "degrau", "precisao", "limite_inferior", "cobertura", "publicadas",
                      "acertos", "n_efetivo")


def formatar_comparacao(comparacao: Mapping) -> str:
    sem, com = comparacao["sem_corpus"], comparacao["com_corpus"]
    tabela = pd.DataFrame([
        {"fonte": "A sem corpus", **{k: sem[k] for k in COLUNAS_COMPARACAO}},
        {"fonte": "B com corpus", **{k: com[k] for k in COLUNAS_COMPARACAO}},
    ])
    veredito = "aprovado" if comparacao["corpus_aprovado"] else "fora"
    return "\n".join([
        "Comparação de fontes (regra dos 3 pontos)",
        tabela.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
        f"Corpus {veredito}: {comparacao['motivo']}",
    ])


def formatar_resumo(resumo: Mapping, caminhos: Caminhos) -> str:
    contagens = resumo["contagens"]
    lim = resumo["limiares"]
    return "\n".join([
        f"Tempo total: {resumo['tempo_segundos']:.1f} s",
        f"Decisões nas {resumo['n_questoes']} questões: " + ", ".join(f"{d} {contagens[d]}" for d in DECISOES),
        f"Fonte: {'com' if resumo['com_corpus'] else 'sem'} corpus; limiares confiança {lim['publicar']:.2f}, "
        f"fora {lim['fora']:.2f} (degrau {resumo['degrau']})",
        f"Precisão prevista {resumo['precisao']:.2f} ({resumo['precisao_intervalo'][0]:.2f}-"
        f"{resumo['precisao_intervalo'][1]:.2f}, n efetivo {resumo['n_efetivo']:.1f}); "
        f"cobertura prevista {resumo['cobertura']:.2f} nas "
        f"{resumo['n_avaliadas']} fora da folha (dobras: {resumo['cobertura_dobras']:.2f})",
        f"Gravados {caminhos.publicacao}, {caminhos.previsao} e {caminhos.intermediarios}",
    ])
