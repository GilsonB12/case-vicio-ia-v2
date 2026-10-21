import warnings
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from apoio_contexto import DROGAS, FURTO, MARIA_DA_PENHA, PATRIMONIO, ROUBO, montar_maior
from sistema import calibracao
from sistema.arvore import ArvoreTemas
from sistema.calibracao import (
    COLUNAS_DOBRA,
    GRADE_FORA,
    GRADE_PUBLICAR,
    comparar_fontes,
    dobras,
    escolher,
    estratos,
    grade,
    intervalo_beta,
    opcoes_fora_da_amostra,
    precisao_jeffreys,
    regra_tres_pontos,
    rodar_dobra,
    tipo_erro,
    treinar_final,
    treinar_rodada,
    validar,
)
from sistema.decisao import Limiares, Opcao
from sistema.modelo import CARACTERISTICAS_CONFIANCA, DetectorFora, ModeloConfianca, ModeloPares


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


@pytest.fixture(scope="module")
def ctx(arvore):
    return montar_maior(arvore)


@pytest.fixture(scope="module")
def validacao(ctx):
    return validar(ctx, com_corpus=False)


# --- dobras ------------------------------------------------------------------------------------------------


def test_dobras_estratificadas_por_fora_e_disciplina(ctx):
    lista = dobras(ctx.folha, arvore=ctx.arvore)
    assert len(lista) == 15
    estrato = dict(zip(ctx.folha["id"], estratos(ctx.folha, ctx.arvore)))
    assert Counter(estrato.values()) == {"Penal": 25, "Direito Penal - Legislação Especial": 10, "fora": 5}
    for repeticao in range(3):
        testes = [teste for _, teste in lista[5 * repeticao:5 * repeticao + 5]]
        assert sorted(i for teste in testes for i in teste) == sorted(ctx.folha["id"])  # partição da folha
        for treino, teste in lista[5 * repeticao:5 * repeticao + 5]:
            assert not set(treino) & set(teste) and len(treino) + len(teste) == 40
            assert Counter(estrato[i] for i in teste) == {"Penal": 5, "Direito Penal - Legislação Especial": 2,
                                                          "fora": 1}
    # repetições com sementes diferentes embaralham de outro jeito; a mesma semente repete
    assert [t for _, t in lista[:5]] != [t for _, t in lista[5:10]]
    assert lista == dobras(ctx.folha, arvore=ctx.arvore)


def test_estrato_pequeno_vira_outros_sem_aviso(arvore):
    folha = pd.DataFrame({
        "id": [f"Q{i:04d}" for i in range(12)],
        "tema_central_id": pd.Series([FURTO] * 6 + [MARIA_DA_PENHA] * 2 + [None] * 4, dtype=object),
        "extras_ids": [[] for _ in range(12)],
        "fora": [False] * 8 + [True] * 4,
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lista = dobras(folha, n_dobras=5, repeticoes=1, arvore=arvore)
    assert len(lista) == 5
    assert sorted(i for _, teste in lista for i in teste) == list(folha["id"])


# --- vazamento ---------------------------------------------------------------------------------------------


@pytest.fixture
def espioes(ctx, monkeypatch):
    """Grava os ids de cada índice kNN montado e de cada treino de pares, detector e confiança."""
    vistos = {"indice": [], "pares": [], "detector": [], "confianca": []}
    indice_original = type(ctx).indice_rotulado
    treinar_pares, treinar_detector = ModeloPares.treinar, DetectorFora.treinar
    treinar_confianca = ModeloConfianca.treinar

    def indice_espiao(self, ids_folha_treino, com_corpus=False):
        indice, pesos = indice_original(self, ids_folha_treino, com_corpus)
        vistos["indice"].append(set(indice.ids))
        return indice, pesos

    def pares_espiao(self, tabela, rotulos, pesos):
        vistos["pares"].append(set(tabela["id"]))
        return treinar_pares(self, tabela, rotulos, pesos)

    def detector_espiao(self, tabela_q, rotulos, pesos):
        vistos["detector"].append(set(tabela_q["id"]))
        return treinar_detector(self, tabela_q, rotulos, pesos)

    def confianca_espiao(self, opcoes, rotulos):
        vistos["confianca"].append(opcoes.copy())
        return treinar_confianca(self, opcoes, rotulos)

    monkeypatch.setattr(type(ctx), "indice_rotulado", indice_espiao)
    monkeypatch.setattr(ModeloPares, "treinar", pares_espiao)
    monkeypatch.setattr(DetectorFora, "treinar", detector_espiao)
    monkeypatch.setattr(ModeloConfianca, "treinar", confianca_espiao)
    return vistos


def test_rodar_dobra_sem_vazamento(ctx, espioes):
    treino, teste = dobras(ctx.folha, arvore=ctx.arvore)[0]
    resultado = rodar_dobra(ctx, treino, teste, com_corpus=True)

    assert list(resultado.columns) == COLUNAS_DOBRA
    assert list(resultado["id"]) == teste
    # 5 dobras internas da confiança e a rodada: nenhuma questão de teste externo em índice nem treino nenhum
    assert len(espioes["indice"]) == len(espioes["pares"]) == 6
    for ids in [*espioes["indice"], *espioes["pares"], *espioes["detector"]]:
        assert not set(teste) & ids
    [opcoes_confianca] = espioes["confianca"]
    ids_confianca = set(opcoes_confianca["id"])
    assert not set(teste) & ids_confianca
    nao_fora = {i for i in treino if not ctx.linha_folha(i)["fora"]}
    assert ids_confianca == nao_fora  # todas as questões de treino da árvore, nenhuma fora da taxonomia
    # a rodada (último treino): o índice tem a folha de treino, os exemplos e o corpus; os pares, a folha de
    # treino, os exemplos com tema e o corpus; o detector, a folha de treino e todos os exemplos
    exemplos = {e["id"] for e in ctx.exemplos}
    exemplos_tema = {e["id"] for e in ctx.exemplos if e["tema_id"] is not None}
    corpus = {c["id"] for c in ctx.corpus}
    [ids_detector] = espioes["detector"]
    assert espioes["indice"][-1] == set(treino) | exemplos | corpus
    assert espioes["pares"][-1] == set(treino) | exemplos_tema | corpus
    assert ids_detector == set(treino) | exemplos


def test_validacao_interna_preve_cada_questao_sem_ela_no_treino(ctx, espioes):
    treino, _ = dobras(ctx.folha, arvore=ctx.arvore)[0]
    opcoes = opcoes_fora_da_amostra(ctx, treino, com_corpus=False)
    assert set(opcoes["dobra_interna"]) == {0, 1, 2, 3, 4}
    assert len(espioes["indice"]) == len(espioes["pares"]) == 5 and not espioes["detector"]
    for dobra, ids_indice, ids_pares in zip(range(5), espioes["indice"], espioes["pares"]):
        previstas = set(opcoes.loc[opcoes["dobra_interna"] == dobra, "id"])
        assert previstas and not previstas & ids_indice and not previstas & ids_pares
        assert (ids_indice & set(ctx.ids_folha)) == set(treino) - previstas - {
            i for i in treino if ctx.linha_folha(i)["fora"] and i not in ids_indice}
    # rótulo pela régua; o ramo só aparece com top-1 de nível 2
    assert set(opcoes["opcao"]) <= {"tema", "ramo"} and set(opcoes["acerto"]) <= {0, 1}
    for linha in opcoes.to_dict("records"):
        assert linha["acerto"] == calibracao.acerto_regua(linha["tema_id"], ctx.linha_folha(linha["id"]), ctx.arvore)
    assert (opcoes.groupby("id")["opcao"].apply(list).map(lambda o: o[0] == "tema")).all()
    assert set(CARACTERISTICAS_CONFIANCA) <= set(opcoes.columns)


def test_validacao_interna_com_poucas_questoes(ctx):
    assert opcoes_fora_da_amostra(ctx, ["Q0001"], com_corpus=False).empty
    rodada = treinar_rodada(ctx, ["Q0001"], com_corpus=False)
    assert rodada.confianca.treinado and rodada.confianca.constante == 0.0


def test_questao_de_teste_no_treino_e_erro(ctx):
    with pytest.raises(ValueError):
        rodar_dobra(ctx, ["Q0001", "Q0002"], ["Q0002"], com_corpus=False)


def test_rodar_dobra_resultado(ctx):
    treino, teste = dobras(ctx.folha, arvore=ctx.arvore)[0]
    resultado = rodar_dobra(ctx, treino, teste, com_corpus=False).set_index("id")
    for id_q, linha in resultado.iterrows():
        soma = sum(p for _, p in linha["candidatos_prob"])
        assert soma == pytest.approx(1.0)
        opcoes = linha["opcoes"]
        assert opcoes[0].opcao == "tema" and opcoes[0].tema_id == linha["tema_max"]
        assert all(0 <= o.confianca <= 1 for o in opcoes) and linha["confianca_tema"] == opcoes[0].confianca
        assert 0 <= linha["votos"] <= linha["votos_ramo"] <= 4
        if len(opcoes) == 2:
            assert opcoes[1].opcao == "ramo" and opcoes[1].tema_id == ctx.arvore.pai(linha["tema_max"])
        assert linha["p_max"] == pytest.approx(linha["candidatos_prob"][0][1])
        assert linha["tema_max"] == linha["candidatos_prob"][0][0]
        assert 0 <= linha["p_fora"] <= 1
        verdade = ctx.linha_folha(id_q)
        assert linha["fora"] == verdade["fora"]
        if not verdade["fora"]:
            assert verdade["tema_central_id"] in linha["alvos"]


# --- grade ---------------------------------------------------------------------------------------------------


def resultado(id_q, temas, p_fora, central, confianca, extras=(), fora=False, ramo=None, repeticao=0):
    """Linha de `rodar_dobra` à mão: `confianca` da opção tema; `ramo` = (pai, confiança) da opção ramo."""
    temas = sorted(temas, key=lambda par: -par[1])
    opcoes = [Opcao("tema", temas[0][0], confianca, 2)]
    if ramo is not None:
        opcoes.append(Opcao("ramo", ramo[0], ramo[1], 0))
    return {
        "id": id_q, "p_max": temas[0][1], "tema_max": temas[0][0], "p_ramo_max": 0.0, "pai_max": None,
        "p_fora": p_fora, "candidatos_prob": temas, "ramos": [], "opcoes": opcoes, "confianca_tema": confianca,
        "confianca_ramo": ramo[1] if ramo else None, "votos": 2, "votos_ramo": 2, "gatilhos": [], "fora": fora,
        "tema_central_id": central, "extras_ids": list(extras), "alvos": [], "repeticao": repeticao,
    }


@pytest.fixture
def resultados_a_mao():
    return pd.DataFrame([
        resultado("Q1", [(FURTO, 0.3), (ROUBO, 0.1)], 0.1, FURTO, 0.90),  # certo
        resultado("Q2", [(ROUBO, 0.25), (FURTO, 0.15)], 0.1, FURTO, 0.85),  # irmão: erro
        resultado("Q3", [(DROGAS, 0.5), (FURTO, 0.5)], 0.95, None, 0.99, fora=True),  # fora certo
        resultado("Q4", [(MARIA_DA_PENHA, 0.35), (DROGAS, 0.05)], 0.2, None, 0.95, fora=True),  # fora publicada
        resultado("Q5", [(FURTO, 0.3), (PATRIMONIO, 0.2), (DROGAS, 0.1)], 0.45, ROUBO, 0.60, extras=[DROGAS],
                  ramo=(PATRIMONIO, 0.90)),  # pai publicado pelo ramo: certo (pai de Roubo)
    ], columns=[*COLUNAS_DOBRA, "repeticao"])


def linha(tabela, publicar, fora):
    return tabela[np.isclose(tabela["publicar"], publicar) & np.isclose(tabela["fora"], fora)].iloc[0]


def test_grade_conta_publicadas_acertos_e_fora(resultados_a_mao, arvore):
    tabela = grade(resultados_a_mao, arvore)
    assert len(tabela) == len(GRADE_PUBLICAR) * len(GRADE_FORA)
    assert GRADE_PUBLICAR[0] == 0.50 and GRADE_PUBLICAR[-1] == 0.99 and len(GRADE_PUBLICAR) == 50

    # a grade varre a confiança, não o p: Q1, Q2, Q4 publicadas pela confiança; Q3 fora; Q5 pelo ramo (0,90)
    a = linha(tabela, 0.80, 0.9)
    assert (a["n"], a["publicadas"], a["acertos"]) == (5, 4, 2)
    assert (a["fora_certos"], a["fora_errados"], a["fora_perdidos"]) == (1, 0, 1)
    assert a["cobertura"] == pytest.approx(4 / 5)
    assert a["precisao"] == pytest.approx((2 + 0.5) / (4 + 1))
    assert (a["publicadas_efetivas"], a["acertos_efetivos"]) == (4, 2)  # uma repetição só

    b = linha(tabela, 0.95, 0.4)  # Q5 vira fora (0,45); Q1, Q2 abaixo de 0,95 revisadas; Q4 publicada (0,95)
    assert (b["publicadas"], b["acertos"], b["revisadas"]) == (1, 0, 2)
    assert (b["fora_certos"], b["fora_errados"], b["fora_perdidos"]) == (1, 1, 1)


def test_grade_com_repeticoes_usa_n_efetivo(resultados_a_mao, arvore):
    repetidos = pd.concat([resultados_a_mao.assign(repeticao=r) for r in range(3)], ignore_index=True)
    tabela = grade(repetidos, arvore, repeticoes=3)
    a = linha(tabela, 0.80, 0.9)
    assert (a["publicadas"], a["acertos"]) == (12, 6)  # contagens brutas: cada questão 3 vezes
    assert (a["publicadas_efetivas"], a["acertos_efetivos"]) == (4, 2)
    assert a["precisao"] == pytest.approx((2 + 0.5) / (4 + 1))  # igual à de uma repetição
    baixo, alto = intervalo_beta([2], [4])
    assert (a["limite_inferior"], a["limite_superior"]) == (baixo[0], alto[0])
    # contar as 3 cópias como independentes estreitaria o intervalo
    baixo_bruto, alto_bruto = intervalo_beta([6], [12])
    assert alto_bruto[0] - baixo_bruto[0] < a["limite_superior"] - a["limite_inferior"]
    # sem `repeticoes`, sai da coluna repeticao
    pd.testing.assert_frame_equal(grade(repetidos, arvore), tabela)
    # n efetivo fracionário: uma questão publicada em só 2 das 3 repetições conta 2/3
    parcial = repetidos[~((repetidos["id"] == "Q1") & (repetidos["repeticao"] == 2))]
    b = linha(grade(parcial, arvore, repeticoes=3), 0.80, 0.9)
    assert b["publicadas_efetivas"] == pytest.approx(11 / 3) and b["acertos_efetivos"] == pytest.approx(5 / 3)
    assert b["precisao"] == pytest.approx((5 / 3 + 0.5) / (11 / 3 + 1))


def test_grade_aceita_lista_de_dobras(resultados_a_mao, arvore):
    juntos = grade(resultados_a_mao, arvore)
    separados = grade([resultados_a_mao.iloc[:2], resultados_a_mao.iloc[2:]], arvore)
    pd.testing.assert_frame_equal(juntos, separados)


def test_jeffreys_e_intervalo_beta():
    assert precisao_jeffreys(41, 42) == pytest.approx(41.5 / 43)
    baixo, alto = intervalo_beta([41, 41, 0], [42, 42, 0])
    assert baixo[0] == baixo[1] and alto[0] == alto[1]  # mesmos números, mesmo intervalo
    assert baixo[0] == pytest.approx(stats.beta.ppf(0.025, 41.5, 1.5), abs=0.01)
    assert alto[0] == pytest.approx(stats.beta.ppf(0.975, 41.5, 1.5), abs=0.01)
    assert 0 < baixo[2] < 0.05 and 0.95 < alto[2] <= 1  # sem publicadas: Beta(0,5; 0,5)
    # determinístico
    assert intervalo_beta([41], [42]) == intervalo_beta([41], [42])
    # contagens fracionárias (n efetivo)
    baixo, alto = intervalo_beta([41 / 3], [42 / 3])
    assert baixo[0] == pytest.approx(stats.beta.ppf(0.025, 41 / 3 + 0.5, 1 / 3 + 0.5), abs=0.01)
    assert alto[0] == pytest.approx(stats.beta.ppf(0.975, 41 / 3 + 0.5, 1 / 3 + 0.5), abs=0.01)


def test_grade_tem_intervalo_em_volta_da_precisao(resultados_a_mao, arvore):
    tabela = grade(resultados_a_mao, arvore)
    assert (tabela["limite_inferior"] <= tabela["precisao"]).all()
    assert (tabela["precisao"] <= tabela["limite_superior"]).all()


# --- escada ----------------------------------------------------------------------------------------------------


def grade_a_mao(linhas):
    return pd.DataFrame(linhas, columns=["publicar", "fora", "cobertura", "precisao", "limite_inferior"])


def test_escada_degrau_0_maior_cobertura_entre_os_dois_criterios():
    g = grade_a_mao([
        (0.50, 0.5, 0.70, 0.96, 0.95),  # média abaixo de 0,97
        (0.60, 0.5, 0.60, 0.98, 0.955),  # degrau 0
        (0.70, 0.5, 0.50, 0.99, 0.97),  # degrau 0, cobertura menor
        (0.40, 0.5, 0.80, 0.98, 0.94),  # limite inferior abaixo
    ])
    assert escolher(g) == (Limiares(0.60, 0.5), 0)


def test_escada_degrau_1_so_limite_inferior():
    g = grade_a_mao([(0.50, 0.5, 0.70, 0.96, 0.951), (0.60, 0.5, 0.60, 0.965, 0.952), (0.40, 0.5, 0.9, 0.9, 0.8)])
    assert escolher(g) == (Limiares(0.50, 0.5), 1)


def test_escada_degrau_2_so_media():
    g = grade_a_mao([(0.50, 0.5, 0.70, 0.975, 0.90), (0.60, 0.5, 0.50, 0.99, 0.94), (0.40, 0.5, 0.9, 0.9, 0.8)])
    assert escolher(g) == (Limiares(0.50, 0.5), 2)


def test_escada_degrau_3_regua_com_maior_cobertura():
    g = grade_a_mao([
        (0.50, 0.5, 0.70, 0.91, 0.85),  # abaixo da régua, mesmo com o maior limite inferior
        (0.80, 0.5, 0.40, 0.955, 0.80),  # degrau 3
        (0.90, 0.5, 0.20, 0.96, 0.78),  # degrau 3, cobertura menor
    ])
    assert escolher(g) == (Limiares(0.80, 0.5), 3)


def test_escada_degrau_3_exige_n_efetivo_minimo():
    g = grade_a_mao([(0.80, 0.5, 0.40, 0.955, 0.80), (0.95, 0.5, 0.05, 0.96, 0.70)])
    g["publicadas_efetivas"] = [12, 4]
    assert escolher(g) == (Limiares(0.80, 0.5), 3)
    g["publicadas_efetivas"] = [8, 4]
    assert escolher(g)[1] == 4


def test_escada_degrau_4_maior_precisao_quando_nada_atinge_a_regua():
    g = grade_a_mao([
        (0.50, 0.5, 0.70, 0.90, 0.85),  # maior limite inferior: não é escolhido
        (0.80, 0.6, 0.30, 0.93, 0.75),
        (0.80, 0.4, 0.30, 0.93, 0.75),  # empata: menor limiar de fora
    ])
    assert escolher(g) == (Limiares(0.80, 0.4), 4)


def test_escada_empate_de_cobertura_fica_com_menor_fora_e_maior_publicar():
    g = grade_a_mao([
        (0.60, 0.5, 0.60, 0.98, 0.96), (0.65, 0.5, 0.60, 0.98, 0.96), (0.60, 0.3, 0.60, 0.98, 0.96),
        (0.62, 0.3, 0.60, 0.98, 0.96),
    ])
    assert escolher(g) == (Limiares(0.62, 0.3), 0)


# --- validação e comparação --------------------------------------------------------------------------------------


def test_validar(validacao, ctx):
    assert len(validacao["dobras"]) == 15
    assert len(validacao["resultados"]) == 3 * len(ctx.folha)
    assert set(validacao["resultados"]["repeticao"]) == {0, 1, 2}
    assert set(validacao["resultados"]["dobra"]) == {0, 1, 2, 3, 4}
    assert isinstance(validacao["limiares"], Limiares) and validacao["degrau"] in (0, 1, 2, 3)
    escolhida = validacao["escolhida"]
    assert validacao["precisao"] == pytest.approx(escolhida["precisao"])
    assert validacao["precisao_intervalo"][0] <= validacao["precisao"] <= validacao["precisao_intervalo"][1]
    assert validacao["cobertura"] == pytest.approx(escolhida["publicadas"] / escolhida["n"])
    assert set(validacao["erros"]) == {"irmao", "outro_ramo", "fora_publicada"}
    avaliacao = validacao["avaliacao"]
    assert len(avaliacao) == len(validacao["resultados"])
    publicadas = avaliacao[avaliacao["decisao"] == "publicar"]
    assert len(publicadas) == escolhida["publicadas"]
    assert int(publicadas["acerto"].sum()) == escolhida["acertos"]
    assert sum(validacao["erros"].values()) == len(publicadas) - escolhida["acertos"]
    resumo = validacao["resumo"]
    assert resumo["publicar"] == validacao["limiares"].publicar and resumo["com_corpus"] is False
    # n efetivo: as publicadas das 3 repetições divididas por 3
    assert resumo["n_efetivo"] == pytest.approx(escolhida["publicadas"] / 3)
    assert validacao["n_efetivo"] == pytest.approx(resumo["n_efetivo"])
    assert validacao["precisao"] == pytest.approx(precisao_jeffreys(escolhida["acertos"] / 3,
                                                                    escolhida["publicadas"] / 3))
    assert set(validacao["avaliacao"]["confianca_tema"].dropna().between(0, 1)) <= {True}


def test_validar_e_deterministica(validacao, ctx):
    de_novo = validar(ctx, com_corpus=False)
    pd.testing.assert_frame_equal(validacao["grade"], de_novo["grade"])
    assert validacao["limiares"] == de_novo["limiares"]


def test_tipo_de_erro(arvore):
    roubo = {"fora": False, "tema_central_id": ROUBO, "extras_ids": []}
    assert tipo_erro(FURTO, roubo, arvore) == "irmao"
    assert tipo_erro(DROGAS, roubo, arvore) == "outro_ramo"
    assert tipo_erro(PATRIMONIO, roubo, arvore) is None  # o pai é acerto
    assert tipo_erro(FURTO, {"fora": True, "tema_central_id": None, "extras_ids": []}, arvore) == "fora_publicada"


@pytest.mark.parametrize("com, aprovado", [
    ({"cobertura": 0.53, "precisao": 0.97, "limite_inferior": 0.95}, True),  # +3 pontos exatos
    ({"cobertura": 0.60, "precisao": 0.98, "limite_inferior": 0.96}, True),
    ({"cobertura": 0.52, "precisao": 0.98, "limite_inferior": 0.96}, False),  # só +2 pontos
    ({"cobertura": 0.60, "precisao": 0.969, "limite_inferior": 0.96}, False),  # precisão menor
    ({"cobertura": 0.60, "precisao": 0.98, "limite_inferior": 0.949}, False),  # limite inferior menor
])
def test_regra_dos_3_pontos(com, aprovado):
    sem = {"cobertura": 0.50, "precisao": 0.97, "limite_inferior": 0.95}
    resultado_regra, motivo = regra_tres_pontos(sem, com)
    assert resultado_regra is aprovado and motivo


def test_comparar_fontes(ctx, monkeypatch):
    chamadas = []
    original = calibracao.validar

    def validar_espiao(contexto, com_corpus, **opcoes):
        chamadas.append(com_corpus)
        return original(contexto, com_corpus, n_dobras=5, repeticoes=1)

    monkeypatch.setattr(calibracao, "validar", validar_espiao)
    comparacao = comparar_fontes(ctx, ctx)
    assert chamadas == [False, True]
    assert set(comparacao) >= {"sem_corpus", "com_corpus", "corpus_aprovado", "motivo", "regra", "validacoes"}
    esperado, _ = regra_tres_pontos(comparacao["sem_corpus"], comparacao["com_corpus"])
    assert comparacao["corpus_aprovado"] is esperado
    assert comparacao["validacoes"]["com_corpus"]["com_corpus"] is True


def test_treinar_final_usa_os_100_rotulos(ctx, monkeypatch):
    vistos = []
    original = ctx.indice_rotulado

    def espiao(ids_folha_treino, com_corpus=False):
        vistos.append(set(ids_folha_treino))
        return original(ids_folha_treino, com_corpus)

    monkeypatch.setattr(ctx, "indice_rotulado", espiao)
    modelo, detector, confianca = treinar_final(ctx, com_corpus=True)
    # as 5 dobras internas da confiança deixam cada uma um quinto de fora; a última rodada usa a folha toda
    assert len(vistos) == 6 and vistos[-1] == set(ctx.ids_folha)
    assert all(v < set(ctx.ids_folha) for v in vistos[:-1])
    assert set().union(*(set(ctx.ids_folha) - v for v in vistos[:-1])) == set(ctx.ids_folha)
    assert modelo.treinado and detector.treinado and confianca.treinado
