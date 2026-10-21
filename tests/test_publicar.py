import re

import pandas as pd
import pytest

from apoio_contexto import DROGAS, FURTO, PATRIMONIO, ROUBO
from sistema.arvore import ArvoreTemas
from sistema.dados import carregar_temas
from sistema.decisao import Decisao
from sistema.publicar import (
    COLUNAS_PUBLICACAO,
    escrever_previsao,
    escrever_publicacao,
    ler_publicacao,
    linhas_previsao,
    validar_publicacao,
)

IDS = ["Q0001", "Q0002", "Q0003", "Q0004"]
DECISOES = [
    Decisao("Q0001", "publicar", ROUBO, [DROGAS, FURTO], 0.9712, "", 2),
    Decisao("Q0002", "publicar", PATRIMONIO, [], 0.8, "", 3),
    Decisao("Q0003", "revisar", FURTO, [], 0.4, "confiança 0,40 abaixo do limiar 0,80", 4),
    Decisao("Q0004", "fora", None, [], 0.93, "fora da taxonomia (p_fora 0,93): termos militar, cpm", 1),
]


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


@pytest.fixture
def publicacao(tmp_path):
    caminho = tmp_path / "publicacao.csv"
    escrever_publicacao(DECISOES, caminho)
    return caminho


def test_formato_do_leia_me(publicacao):
    texto = publicacao.read_text(encoding="utf-8")
    linhas = texto.splitlines()
    assert linhas[0] == "id,decisao,tema_central_id,extras_ids,confianca,motivo"
    assert linhas[1] == f"Q0001,publicar,{ROUBO},{DROGAS};{FURTO},0.97,"
    assert linhas[2] == f"Q0002,publicar,{PATRIMONIO},,0.80,"
    assert linhas[3] == f'Q0003,revisar,{FURTO},,0.40,"confiança 0,40 abaixo do limiar 0,80"'
    assert linhas[4].startswith("Q0004,fora,,,0.93,")
    assert len(linhas) == 1 + len(DECISOES)
    df = ler_publicacao(publicacao)
    assert list(df.columns) == COLUNAS_PUBLICACAO
    assert df.loc[2, "motivo"] == "confiança 0,40 abaixo do limiar 0,80"


def test_publicacao_valida_nao_tem_erro(publicacao, arvore):
    assert validar_publicacao(ler_publicacao(publicacao), IDS, arvore) == []


def com(df, indice, **valores):
    df = df.copy()
    for coluna, valor in valores.items():
        df.loc[indice, coluna] = valor
    return df


@pytest.mark.parametrize("alterar, trecho", [
    (lambda df: df.iloc[:-1], "faltando"),
    (lambda df: pd.concat([df, df.iloc[[0]].assign(id="Q9999")], ignore_index=True), "a mais"),
    (lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True), "repetido"),
    (lambda df: df.iloc[[1, 0, 2, 3]].reset_index(drop=True), "ordem"),
    (lambda df: com(df, 0, decisao="talvez"), "decisão inválida"),
    (lambda df: com(df, 0, tema_central_id=""), "central ausente em publicar"),
    (lambda df: com(df, 2, tema_central_id=""), "central ausente em revisar"),
    (lambda df: com(df, 3, tema_central_id=str(FURTO)), "presente em fora"),
    (lambda df: com(df, 0, tema_central_id="5062"), "não é publicável"),  # raiz de Penal
    (lambda df: com(df, 0, tema_central_id="1"), "não é publicável"),  # não existe
    (lambda df: com(df, 0, extras_ids="5062"), "extra 5062 não é publicável"),
    (lambda df: com(df, 2, motivo=""), "motivo ausente em revisar"),
    (lambda df: com(df, 3, motivo=" "), "motivo ausente em fora"),
    (lambda df: com(df, 0, confianca="1.20"), "fora de 0 a 1"),
    (lambda df: com(df, 0, confianca="alta"), "não é número"),
    (lambda df: df.drop(columns=["motivo"]), "colunas"),
])
def test_validar_publicacao_pega_cada_erro(publicacao, arvore, alterar, trecho):
    erros = validar_publicacao(alterar(ler_publicacao(publicacao)), IDS, arvore)
    assert erros and any(trecho in erro for erro in erros), erros


def test_tema_inativo_nao_e_publicavel(arvore):
    temas = carregar_temas()
    inativos = [int(t) for t, ativo in zip(temas["tema_id"], temas["ativo"]) if not ativo]
    assert inativos
    df = pd.DataFrame([{"id": "Q0001", "decisao": "publicar", "tema_central_id": str(inativos[0]),
                        "extras_ids": "", "confianca": "0.90", "motivo": ""}])
    assert any("não é publicável" in erro for erro in validar_publicacao(df, ["Q0001"], arvore))


RESUMO = {
    "precisao": 0.9734, "precisao_intervalo": [0.9411, 0.9912], "cobertura": 0.5527,
    "cobertura_intervalo": [0.5427, 0.5627], "cobertura_dobras": 0.51, "publicadas_dobras": 153,
    "acertos_dobras": 149, "n_teste_dobras": 300, "degrau": 1,
    "limiares": {"publicar": 0.72, "fora": 0.6, "extra": 0.15}, "com_corpus": False, "n_rotulos": 100,
    "n_dobras": 5, "repeticoes": 3, "n_avaliadas": 559, "publicadas_avaliadas": 309, "n_recentes": 300,
    "comparacao": {
        "sem_corpus": {"precisao": 0.9734, "limite_inferior": 0.9411, "cobertura": 0.51},
        "com_corpus": {"precisao": 0.97, "limite_inferior": 0.93, "cobertura": 0.53},
        "corpus_aprovado": False, "motivo": "ganho de cobertura de +2,0 pontos, menos que 3",
    },
}


def test_previsao_tem_as_4_linhas_no_formato_exato(tmp_path):
    caminho = tmp_path / "previsao.txt"
    texto = escrever_previsao(caminho, RESUMO)
    linhas = caminho.read_text(encoding="utf-8").splitlines()
    assert linhas[:4] == [
        "precisao_prevista: 0.97",
        "precisao_intervalo: 0.94-0.99",
        "cobertura_prevista: 0.55",
        "cobertura_intervalo: 0.54-0.56",
    ]
    padroes = [r"precisao_prevista: \d\.\d\d", r"precisao_intervalo: \d\.\d\d-\d\.\d\d",
               r"cobertura_prevista: \d\.\d\d", r"cobertura_intervalo: \d\.\d\d-\d\.\d\d"]
    assert all(re.fullmatch(padrao, linha) for padrao, linha in zip(padroes, linhas[:4]))
    assert linhas[4] == ""  # o texto livre vem depois de uma linha em branco
    assert texto == caminho.read_text(encoding="utf-8")


def test_previsao_explica_como_chegou_nos_numeros(tmp_path):
    texto = escrever_previsao(tmp_path / "previsao.txt", RESUMO)
    explicacao = texto.split("\n\n", 1)[1]
    for trecho in ("folha-verdade", "15 avaliações", "degrau 1", "Jeffreys", "559", "±0,01", "0,51",
                   "149 certas em 153 publicadas", "corpus fora", "otimista", "sintéticos",
                   "mesma folha", "300 questões de concursos recentes", "n efetivo 51,00 publicadas",
                   "divididas pelas 3 repetições", "confiança (0,50 a 0,99", "validação interna de 5 dobras"):
        assert trecho in explicacao, trecho
    com_n = escrever_previsao(tmp_path / "outra.txt", {**RESUMO, "n_efetivo": 50.5, "n_dobras_confianca": 3})
    assert "n efetivo 50,50 publicadas" in com_n and "validação interna de 3 dobras" in com_n


def test_previsao_sem_comparacao_diz_a_fonte(tmp_path):
    texto = escrever_previsao(tmp_path / "previsao.txt", {**RESUMO, "comparacao": None, "com_corpus": True,
                                                          "n_recentes": 0})
    assert "não rodou" in texto and "com o corpus" in texto
    assert "concursos recentes" not in texto and texto.rstrip().endswith(".")


def test_linhas_previsao_arredondam_com_duas_casas():
    assert linhas_previsao({**RESUMO, "precisao": 0.965, "cobertura": 1.0, "cobertura_intervalo": [0.99, 1.0]})[2:] \
        == ["cobertura_prevista: 1.00", "cobertura_intervalo: 0.99-1.00"]
