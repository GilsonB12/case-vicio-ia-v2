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
    precisao_jeffreys,
    regra_tres_pontos,
    rodar_dobra,
    tipo_erro,
    treinar_final,
    validar,
)
from sistema.decisao import Limiares
from sistema.modelo import DetectorFora, ModeloPares


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


def test_rodar_dobra_sem_vazamento(ctx, monkeypatch):
    treino, teste = dobras(ctx.folha, arvore=ctx.arvore)[0]
    vistos = {"indice": [], "pares": [], "detector": []}
    indice_original = type(ctx).indice_rotulado
    treinar_pares, treinar_detector = ModeloPares.treinar, DetectorFora.treinar

    def indice_espiao(self, ids_folha_treino, com_corpus=False):
        indice, pesos = indice_original(self, ids_folha_treino, com_corpus)
        vistos["indice"].append(list(indice.ids))
        return indice, pesos

    def pares_espiao(self, tabela, rotulos, pesos):
        vistos["pares"].append(set(tabela["id"]))
        return treinar_pares(self, tabela, rotulos, pesos)

    def detector_espiao(self, tabela_q, rotulos, pesos):
        vistos["detector"].append(set(tabela_q["id"]))
        return treinar_detector(self, tabela_q, rotulos, pesos)

    monkeypatch.setattr(type(ctx), "indice_rotulado", indice_espiao)
    monkeypatch.setattr(ModeloPares, "treinar", pares_espiao)
    monkeypatch.setattr(DetectorFora, "treinar", detector_espiao)
    resultado = rodar_dobra(ctx, treino, teste, com_corpus=True)

    assert list(resultado.columns) == COLUNAS_DOBRA
    assert list(resultado["id"]) == teste
    [ids_indice] = vistos["indice"]
    [ids_pares] = vistos["pares"]
    [ids_detector] = vistos["detector"]
    assert not set(teste) & set(ids_indice)
    assert not set(teste) & ids_pares
    assert not set(teste) & ids_detector
    # o índice tem a folha de treino, os exemplos e o corpus; os pares, a folha de treino, os exemplos com tema e
    # o corpus; o detector, a folha de treino e todos os exemplos
    exemplos = {e["id"] for e in ctx.exemplos}
    exemplos_tema = {e["id"] for e in ctx.exemplos if e["tema_id"] is not None}
    corpus = {c["id"] for c in ctx.corpus}
    assert set(ids_indice) == set(treino) | exemplos | corpus
    assert ids_pares == set(treino) | exemplos_tema | corpus
    assert ids_detector == set(treino) | exemplos


def test_questao_de_teste_no_treino_e_erro(ctx):
    with pytest.raises(ValueError):
        rodar_dobra(ctx, ["Q0001", "Q0002"], ["Q0002"], com_corpus=False)


def test_rodar_dobra_resultado(ctx):
    treino, teste = dobras(ctx.folha, arvore=ctx.arvore)[0]
    resultado = rodar_dobra(ctx, treino, teste, com_corpus=False).set_index("id")
    for id_q, linha in resultado.iterrows():
        soma = sum(p for _, p in linha["candidatos_prob"])
        assert soma == pytest.approx(1.0)
        assert linha["p_max"] == pytest.approx(linha["candidatos_prob"][0][1])
        assert linha["tema_max"] == linha["candidatos_prob"][0][0]
        assert 0 <= linha["p_fora"] <= 1
        verdade = ctx.linha_folha(id_q)
        assert linha["fora"] == verdade["fora"]
        if not verdade["fora"]:
            assert verdade["tema_central_id"] in linha["alvos"]


# --- grade ---------------------------------------------------------------------------------------------------


def resultado(id_q, temas, p_fora, central, extras=(), fora=False, ramos=(), arvore=None):
    temas = sorted(temas, key=lambda par: -par[1])
    ramos = sorted(ramos, key=lambda par: -par[1])
    return {
        "id": id_q, "p_max": temas[0][1], "tema_max": temas[0][0], "p_ramo_max": ramos[0][1] if ramos else 0.0,
        "pai_max": ramos[0][0] if ramos else None, "p_fora": p_fora, "candidatos_prob": temas, "ramos": ramos,
        "gatilhos": [], "fora": fora, "tema_central_id": central, "extras_ids": list(extras), "alvos": [],
    }


@pytest.fixture
def resultados_a_mao():
    return pd.DataFrame([
        resultado("Q1", [(FURTO, 0.9), (ROUBO, 0.1)], 0.1, FURTO),  # certo
        resultado("Q2", [(ROUBO, 0.85), (FURTO, 0.15)], 0.1, FURTO),  # irmão: erro
        resultado("Q3", [(DROGAS, 0.5), (FURTO, 0.5)], 0.95, None, fora=True),  # fora certo
        resultado("Q4", [(MARIA_DA_PENHA, 0.95), (DROGAS, 0.05)], 0.2, None, fora=True),  # fora publicada
        resultado("Q5", [(FURTO, 0.6), (PATRIMONIO, 0.3), (DROGAS, 0.1)], 0.45, ROUBO, extras=[DROGAS],
                  ramos=[(PATRIMONIO, 0.9)]),  # pai publicado pelo ramo: certo (pai de Roubo)
    ], columns=COLUNAS_DOBRA)


def linha(tabela, publicar, fora):
    return tabela[np.isclose(tabela["publicar"], publicar) & np.isclose(tabela["fora"], fora)].iloc[0]


def test_grade_conta_publicadas_acertos_e_fora(resultados_a_mao, arvore):
    tabela = grade(resultados_a_mao, arvore)
    assert len(tabela) == len(GRADE_PUBLICAR) * len(GRADE_FORA)
    assert GRADE_PUBLICAR[0] == 0.30 and GRADE_PUBLICAR[-1] == 0.95 and len(GRADE_PUBLICAR) == 66

    a = linha(tabela, 0.80, 0.9)  # Q1, Q2, Q4 publicadas; Q3 fora; Q5 pelo ramo (0,9)
    assert (a["n"], a["publicadas"], a["acertos"]) == (5, 4, 2)
    assert (a["fora_certos"], a["fora_errados"], a["fora_perdidos"]) == (1, 0, 1)
    assert a["cobertura"] == pytest.approx(4 / 5)
    assert a["precisao"] == pytest.approx((2 + 0.5) / (4 + 1))

    b = linha(tabela, 0.95, 0.4)  # Q5 vira fora (0,45); Q1, Q2 abaixo de 0,95 revisadas; Q4 publicada (0,95)
    assert (b["publicadas"], b["acertos"], b["revisadas"]) == (1, 0, 2)
    assert (b["fora_certos"], b["fora_errados"], b["fora_perdidos"]) == (1, 1, 1)


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


def test_escada_degrau_3_maior_limite_inferior_e_desempates():
    g = grade_a_mao([
        (0.50, 0.5, 0.70, 0.90, 0.85),
        (0.60, 0.6, 0.60, 0.92, 0.88),
        (0.70, 0.4, 0.60, 0.92, 0.88),  # empata no limite inferior e na cobertura: menor limiar de fora
        (0.80, 0.4, 0.40, 0.93, 0.88),  # mesmo limite inferior, cobertura menor
    ])
    assert escolher(g) == (Limiares(0.70, 0.4), 3)


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
    modelo, detector = treinar_final(ctx, com_corpus=True)
    assert vistos == [set(ctx.ids_folha)]
    assert modelo.treinado and detector.treinado
