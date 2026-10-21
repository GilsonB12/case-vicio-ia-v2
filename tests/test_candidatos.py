from collections import Counter

import numpy as np
import pytest

from apoio_contexto import (
    DIMENSAO,
    DISPOSICOES,
    DROGAS,
    ESTELIONATO,
    EXTORSAO,
    FURTO,
    MARIA_DA_PENHA,
    MARIA_DA_PENHA_DUP,
    PATRIMONIO,
    ROUBO,
    montar,
    vetor,
)
from sistema.arvore import ArvoreTemas
from sistema.candidatos import (
    ORIGEM_BM25,
    ORIGEM_DICIONARIO,
    ORIGEM_EMBEDDING,
    ORIGEM_FILHO_DO_PAI,
    ORIGEM_KNN,
    ORIGEM_PAI,
    ORIGEM_TERMOS,
    ORIGENS_BUSCA,
    K_TERMOS,
    Candidato,
    buscar_vizinhos,
    gerar,
    gerar_lote,
)
from sistema.vizinhos import IndiceVetorial

VAZIO = IndiceVetorial([], np.zeros((0, DIMENSAO)), [])
SEM_CHAVES = Counter()


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


@pytest.fixture
def ctx(arvore):
    return montar(arvore)


def indice(*linhas):
    """indice(("X1", vetor(...), tema), ...)"""
    ids = [linha[0] for linha in linhas]
    return IndiceVetorial(ids, np.vstack([linha[1] for linha in linhas]), [linha[2] for linha in linhas])


def por_tema(candidatos):
    return {c.tema_id: c for c in candidatos}


def questao(texto="", id_="QX"):
    return {"id": id_, "texto": texto}


# --- as buscas ------------------------------------------------------------------------------------


def test_uniao_das_quatro_buscas_mais_o_pai(ctx):
    knn = indice(("X1", vetor(extorsao=1), EXTORSAO))
    candidatos = gerar(questao("estelionato"), Counter({"lei:11340": 1}), vetor(roubo=1, extorsao=0.5), ctx, knn,
                       k_bm25=1, k_emb=1, k_knn=1)
    temas = por_tema(candidatos)
    assert set(temas) == {MARIA_DA_PENHA, MARIA_DA_PENHA_DUP, ESTELIONATO, ROUBO, EXTORSAO, PATRIMONIO}
    assert temas[MARIA_DA_PENHA].origens == {ORIGEM_DICIONARIO}
    assert temas[MARIA_DA_PENHA_DUP].origens == {ORIGEM_DICIONARIO}
    assert temas[ESTELIONATO].origens == {ORIGEM_BM25}
    assert temas[ROUBO].origens == {ORIGEM_EMBEDDING}
    assert temas[EXTORSAO].origens == {ORIGEM_KNN}
    assert temas[PATRIMONIO].origens == {ORIGEM_PAI}
    assert all(isinstance(c, Candidato) for c in candidatos)
    assert temas[EXTORSAO].n_buscas == 1 and temas[PATRIMONIO].n_buscas == 0


def test_pontuacao_de_cada_busca_normalizada_pela_maior_da_questao(ctx):
    candidatos = gerar(questao(), Counter({"lei:11340": 1}), vetor(roubo=1, extorsao=0.5), ctx, VAZIO,
                       k_bm25=0, k_emb=10, k_knn=0)
    temas = por_tema(candidatos)
    assert temas[MARIA_DA_PENHA].pontuacao == pytest.approx(1.0)
    assert temas[MARIA_DA_PENHA_DUP].pontuacao == pytest.approx(0.3)
    assert temas[ROUBO].por_origem[ORIGEM_EMBEDDING] == pytest.approx(1.0)
    assert temas[EXTORSAO].por_origem[ORIGEM_EMBEDDING] == pytest.approx(0.5, rel=1e-5)
    # temas de cosseno zero não entram pela busca de embedding
    assert set(temas) == {MARIA_DA_PENHA, MARIA_DA_PENHA_DUP, ROUBO, EXTORSAO, PATRIMONIO}
    assert all(0 <= c.pontuacao <= 1 for c in candidatos)


def test_knn_ignora_rotulo_none(ctx):
    knn = indice(("X1", vetor(fora=1), None), ("X2", vetor(fora=1, extorsao=0.5), EXTORSAO))
    assert gerar(questao(), SEM_CHAVES, vetor(fora=1), ctx, knn, k_bm25=0, k_emb=0, k_knn=1) == []
    temas = por_tema(gerar(questao(), SEM_CHAVES, vetor(fora=1), ctx, knn, k_bm25=0, k_emb=0, k_knn=2))
    assert set(temas) == {EXTORSAO, PATRIMONIO}
    assert temas[EXTORSAO].origens == {ORIGEM_KNN}


def test_knn_nao_ve_a_propria_questao(ctx):
    v = vetor(disposicoes=1)
    knn = indice(("QX", v, DISPOSICOES), ("X2", vetor(disposicoes=1, extorsao=1), EXTORSAO))
    temas = por_tema(gerar(questao(id_="QX"), SEM_CHAVES, v, ctx, knn, k_bm25=0, k_emb=0, k_knn=1))
    assert EXTORSAO in temas and DISPOSICOES not in temas
    assert [id_ for id_, _, _ in buscar_vizinhos("QX", v, knn, 5)] == ["X2"]


def test_vizinhos_ja_buscados_e_peso_do_corpus(ctx):
    vizinhos = [("C1", ESTELIONATO, 0.8), ("X1", EXTORSAO, 0.6)]
    temas = por_tema(gerar(questao(), SEM_CHAVES, vetor(fora=1), ctx, VAZIO, k_bm25=0, k_emb=0, k_knn=15,
                           vizinhos=vizinhos, peso_por_id={"C1": 0.5, "X1": 1.0}))
    assert temas[EXTORSAO].por_origem[ORIGEM_KNN] == pytest.approx(1.0)  # 0,6 é a maior depois do peso
    assert temas[ESTELIONATO].por_origem[ORIGEM_KNN] == pytest.approx(0.4 / 0.6)
    so_o_primeiro = por_tema(gerar(questao(), SEM_CHAVES, vetor(fora=1), ctx, VAZIO, k_bm25=0, k_emb=0, k_knn=1,
                                   vizinhos=vizinhos))
    assert ESTELIONATO in so_o_primeiro and EXTORSAO not in so_o_primeiro


# --- pai, filhos do pai, publicáveis ---------------------------------------------------------------------


def test_pai_de_candidato_de_nivel_2_herda_a_pontuacao_do_filho(ctx):
    temas = por_tema(gerar(questao(), Counter({"art:155:cp": 1}), vetor(fora=1), ctx, VAZIO,
                           k_bm25=0, k_emb=0, k_knn=0))
    assert set(temas) == {FURTO, PATRIMONIO}
    assert temas[FURTO].origens == {ORIGEM_DICIONARIO}
    assert temas[PATRIMONIO].origens == {ORIGEM_PAI}
    assert temas[PATRIMONIO].pontuacao == pytest.approx(1.0)


def test_filhos_do_pai_que_veio_pelo_dicionario(ctx, arvore):
    temas = por_tema(gerar(questao(), Counter({"lei:9999": 1}), vetor(fora=1), ctx, VAZIO,
                           k_bm25=0, k_emb=0, k_knn=0))
    filhos_no_contexto = {DISPOSICOES, FURTO, ROUBO, EXTORSAO, ESTELIONATO}
    assert set(temas) == {PATRIMONIO} | filhos_no_contexto
    assert temas[PATRIMONIO].origens == {ORIGEM_DICIONARIO}
    for filho in filhos_no_contexto:
        assert temas[filho].origens == {ORIGEM_FILHO_DO_PAI}
        assert temas[filho].pontuacao == pytest.approx(1.0)
    # filhos reais de 5157 fora do contexto (não publicáveis nele) não entram
    assert set(arvore.filhos(PATRIMONIO)) - filhos_no_contexto
    assert not (set(temas) & (set(arvore.filhos(PATRIMONIO)) - filhos_no_contexto))


def test_pai_vindo_de_outra_busca_nao_traz_os_filhos(ctx):
    temas = por_tema(gerar(questao(), SEM_CHAVES, vetor(patrimonio=1), ctx, VAZIO, k_bm25=0, k_emb=1, k_knn=0))
    assert set(temas) == {PATRIMONIO}
    assert temas[PATRIMONIO].origens == {ORIGEM_EMBEDDING}


def test_so_temas_publicaveis_do_contexto(ctx):
    # lei:1 aponta para 5062, a raiz de Penal; o kNN aponta para 5162, publicável mas fora deste contexto
    knn = indice(("X1", vetor(fora=1), 5162))
    assert gerar(questao(), Counter({"lei:1": 1}), vetor(fora=1), ctx, knn, k_bm25=0, k_emb=0, k_knn=1) == []


# --- ordem e teto ----------------------------------------------------------------------------------------


def test_ordem_por_numero_de_origens_e_pontuacao_e_corte_no_teto(ctx):
    knn, pesos = ctx.indice_rotulado({"Q0002", "Q0003"})
    chaves = Counter({"art:155:cp": 1, "lei:9999": 1, "lei:11340": 1})
    argumentos = dict(questao=questao("furto"), chaves=chaves, vetor_questao=vetor(furto=1, roubo=0.5),
                      contexto=ctx, indice_rotulado=knn, peso_por_id=pesos)
    completo = gerar(**argumentos, teto=1000)
    chaves_de_ordem = [(-len(c.origens), -c.pontuacao, c.tema_id) for c in completo]
    assert chaves_de_ordem == sorted(chaves_de_ordem)
    assert [c.tema_id for c in completo[:3]] == [FURTO, ROUBO, PATRIMONIO]
    assert completo[0].origens == {ORIGEM_DICIONARIO, ORIGEM_BM25, ORIGEM_EMBEDDING, ORIGEM_KNN, ORIGEM_FILHO_DO_PAI}
    assert completo[0].n_buscas == 4
    assert [c.tema_id for c in completo[3:]] == [MARIA_DA_PENHA, DISPOSICOES, EXTORSAO, ESTELIONATO,
                                                 MARIA_DA_PENHA_DUP]
    cortado = gerar(**argumentos, teto=3)
    assert [c.tema_id for c in cortado] == [FURTO, ROUBO, PATRIMONIO]
    assert len(gerar(**argumentos)) == len(completo) <= 30


def test_gerar_lote_usa_o_texto_as_chaves_e_o_vetor_do_contexto(ctx):
    knn, pesos = ctx.indice_rotulado(ctx.ids_folha, com_corpus=True)
    candidatos, vizinhos = gerar_lote(["Q0001", "Q0002", "S0001"], ctx, knn, pesos)
    assert set(candidatos) == set(vizinhos) == {"Q0001", "Q0002", "S0001"}
    assert all("Q0002" != id_ for id_, _, _ in vizinhos["Q0002"])
    assert all("S0001" != id_ for id_, _, _ in vizinhos["S0001"])
    esperado = gerar({"id": "Q0001", "texto": ctx.texto("Q0001")}, ctx.chaves("Q0001"), ctx.vetor("Q0001"), ctx, knn,
                     peso_por_id=pesos)
    assert [(c.tema_id, c.origens, c.pontuacao) for c in candidatos["Q0001"]] == \
        [(c.tema_id, c.origens, c.pontuacao) for c in esperado]
    # Q0001 cita o art. 155 do CP e a Lei 11.340: dicionário traz Furto e Maria da Penha
    temas = por_tema(candidatos["Q0001"])
    assert ORIGEM_DICIONARIO in temas[FURTO].origens and ORIGEM_DICIONARIO in temas[MARIA_DA_PENHA].origens


# --- quinta busca: dicionário de termos técnicos ------------------------------------------------------------


TERMOS = {
    FURTO: {"tema_id": FURTO, "termos": ["repouso noturno", "coisa alheia móvel"]},
    ROUBO: {"tema_id": ROUBO, "termos": ["grave ameaça", "coisa alheia móvel"]},
    MARIA_DA_PENHA: {"tema_id": MARIA_DA_PENHA, "termos": ["violência doméstica"]},
}


def test_quinta_busca_pelos_termos_tecnicos_normalizada_pela_maior_soma(arvore):
    ctx = montar(arvore, enriquecidos=TERMOS)
    texto = "Subtraiu coisa alheia móvel durante o repouso noturno."
    temas = por_tema(gerar(questao(texto), SEM_CHAVES, vetor(fora=1), ctx, VAZIO, k_bm25=0, k_emb=0, k_knn=0))
    assert set(temas) == {FURTO, ROUBO, PATRIMONIO}
    assert temas[FURTO].origens == temas[ROUBO].origens == {ORIGEM_TERMOS}
    assert temas[PATRIMONIO].origens == {ORIGEM_PAI}
    pontos = ctx.indice_termos.pontuar(texto)
    assert temas[FURTO].por_origem[ORIGEM_TERMOS] == pytest.approx(1.0)
    assert temas[ROUBO].por_origem[ORIGEM_TERMOS] == pytest.approx(pontos[ROUBO][0] / pontos[FURTO][0])
    assert 0 < temas[ROUBO].por_origem[ORIGEM_TERMOS] < 1
    assert temas[FURTO].n_buscas == 1
    # o duplicado herda os termos do original
    temas = por_tema(gerar(questao("violência doméstica"), SEM_CHAVES, vetor(fora=1), ctx, VAZIO,
                           k_bm25=0, k_emb=0, k_knn=0))
    assert set(temas) == {MARIA_DA_PENHA, MARIA_DA_PENHA_DUP}
    # nada casado, nada trazido
    assert gerar(questao("sem termo nenhum"), SEM_CHAVES, vetor(fora=1), ctx, VAZIO, k_bm25=0, k_emb=0, k_knn=0) == []


def test_quinta_busca_traz_os_5_de_maior_soma(arvore):
    seis = (DROGAS, DISPOSICOES, FURTO, ROUBO, EXTORSAO, ESTELIONATO)
    enriquecidos = {t: {"tema_id": t, "termos": [f"expressao numero {i}"]} for i, t in enumerate(seis)}
    ctx = montar(arvore, enriquecidos=enriquecidos)
    texto = " ".join(f"expressao numero {i}." for i in range(6))
    assert K_TERMOS == 5 and ORIGEM_TERMOS in ORIGENS_BUSCA and len(ORIGENS_BUSCA) == 5
    temas = por_tema(gerar(questao(texto), SEM_CHAVES, vetor(fora=1), ctx, VAZIO, k_bm25=0, k_emb=0, k_knn=0))
    # pesos iguais: empate fica com os menores ids, e Estelionato (o maior) fica de fora
    assert {t for t, c in temas.items() if ORIGEM_TERMOS in c.origens} == set(seis) - {ESTELIONATO}
    um = por_tema(gerar(questao(texto), SEM_CHAVES, vetor(fora=1), ctx, VAZIO, k_bm25=0, k_emb=0, k_knn=0,
                        k_termos=1))
    assert set(um) == {DROGAS}
    assert len(gerar(questao(texto), SEM_CHAVES, vetor(fora=1), ctx, VAZIO, k_bm25=0, k_emb=0, k_knn=0,
                     teto=2)) == 2


def test_gerar_lote_reaproveita_os_termos_do_contexto(arvore):
    ctx = montar(arvore, enriquecidos=TERMOS)
    knn, pesos = ctx.indice_rotulado(ctx.ids_folha, com_corpus=True)
    candidatos, _ = gerar_lote(["Q0001"], ctx, knn, pesos)
    esperado = gerar({"id": "Q0001", "texto": ctx.texto("Q0001")}, ctx.chaves("Q0001"), ctx.vetor("Q0001"), ctx, knn,
                     peso_por_id=pesos)
    assert [(c.tema_id, c.origens, c.pontuacao) for c in candidatos["Q0001"]] ==         [(c.tema_id, c.origens, c.pontuacao) for c in esperado]
    # Q0001: "coisa alheia móvel durante o repouso noturno" → Furto pelos termos (o BM25 não o traz)
    furto = por_tema(candidatos["Q0001"])[FURTO]
    assert furto.origens == {ORIGEM_DICIONARIO, ORIGEM_EMBEDDING, ORIGEM_KNN, ORIGEM_TERMOS, ORIGEM_FILHO_DO_PAI}
    assert furto.n_buscas == 4
