import math

import numpy as np
import pytest

from apoio_contexto import (
    DIMENSAO,
    DISPOSICOES,
    DROGAS,
    ESTELIONATO,
    FURTO,
    MARIA_DA_PENHA,
    MARIA_DA_PENHA_DUP,
    PATRIMONIO,
    ROUBO,
    montar,
)
from sistema.arvore import ArvoreTemas
from sistema.candidatos import Candidato, gerar_lote
from sistema.caracteristicas import (
    CARACTERISTICAS,
    COLUNAS,
    normalizar_disciplina,
    prior_dos_temas,
    tabela_pares,
    topo_knn,
    topos_das_buscas,
    votos_do_tema,
)
from sistema.vizinhos import IndiceVetorial

VAZIO = IndiceVetorial([], np.zeros((0, DIMENSAO)), [])

# Similaridades de Q0001 (furto 1, roubo 0,5) com os vizinhos rotulados do apoio_contexto
SIM_Q0002 = 1 / (math.sqrt(1.25) * math.sqrt(1.01))  # folha, Furto
SIM_Q0003 = 0.7 / (math.sqrt(1.25) * math.sqrt(1.04))  # folha, Roubo
SIM_S0001 = 1 / (math.sqrt(1.25) * math.sqrt(1.09))  # exemplo, Furto
SIM_C0001 = 1.1 / (math.sqrt(1.25) * math.sqrt(1.04))  # corpus, Furto, peso 0,5
COS_FURTO = 1 / math.sqrt(1.25)
COS_ROUBO = 0.5 / math.sqrt(1.25)


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


@pytest.fixture
def ctx(arvore):
    return montar(arvore)


CANDIDATOS_Q0001 = [
    Candidato(FURTO, {"dicionario", "bm25", "embedding", "knn"}, 1.0),
    Candidato(ROUBO, {"embedding", "knn"}, 0.5),
    Candidato(PATRIMONIO, {"dicionario", "pai"}, 1.0),
    Candidato(ESTELIONATO, {"filho_do_pai"}, 0.5),
    Candidato(MARIA_DA_PENHA, {"dicionario"}, 1.0),
]


def linhas_por_tema(tabela, id_questao="Q0001"):
    return {int(linha["tema_id"]): linha for linha in tabela[tabela["id"] == id_questao].to_dict("records")}


@pytest.fixture
def tabela_q0001(ctx):
    knn, pesos = ctx.indice_rotulado({"Q0002", "Q0003"}, com_corpus=True)
    tabela = tabela_pares(["Q0001"], {"Q0001": CANDIDATOS_Q0001}, ctx, knn, pesos, {"Q0002", "Q0003"})
    return linhas_por_tema(tabela), tabela


def test_colunas_e_uma_linha_por_par_na_ordem_dos_candidatos(tabela_q0001):
    _, tabela = tabela_q0001
    assert list(tabela.columns) == COLUNAS
    assert COLUNAS[:2] == ["id", "tema_id"]
    assert set(CARACTERISTICAS) == {
        "bm25", "cos_tema", "knn_max", "knn_media3", "knn_n10", "knn_folha_max", "dic_lei", "dic_art", "nivel2",
        "pai_de_candidato", "n_buscas", "disciplina_bate", "prior",
        "voto_bm25", "voto_cos", "voto_dic", "voto_knn", "votos", "votos_ramo",
    }
    assert list(tabela["tema_id"]) == [c.tema_id for c in CANDIDATOS_Q0001]
    contagens = ("n_buscas", "dic_lei", "dic_art", "votos", "votos_ramo")
    assert (tabela[[c for c in CARACTERISTICAS if c not in contagens]]
            .apply(lambda coluna: coluna.between(0, 1)).all().all())


def test_cos_tema(tabela_q0001):
    linhas, _ = tabela_q0001
    assert linhas[FURTO]["cos_tema"] == pytest.approx(COS_FURTO, rel=1e-5)
    assert linhas[ROUBO]["cos_tema"] == pytest.approx(COS_ROUBO, rel=1e-5)
    assert linhas[PATRIMONIO]["cos_tema"] == pytest.approx(0.0, abs=1e-7)


def test_knn_com_peso_do_corpus(tabela_q0001):
    linhas, _ = tabela_q0001
    furto = linhas[FURTO]
    assert furto["knn_max"] == pytest.approx(SIM_Q0002, rel=1e-5)  # o corpus (0,96 × 0,5) não vence
    assert furto["knn_media3"] == pytest.approx((SIM_Q0002 + SIM_S0001 + 0.5 * SIM_C0001) / 3, rel=1e-5)
    assert furto["knn_n10"] == pytest.approx((1 + 1 + 0.5) / 10)
    assert furto["knn_folha_max"] == pytest.approx(SIM_Q0002, rel=1e-5)
    roubo = linhas[ROUBO]
    assert roubo["knn_max"] == pytest.approx(SIM_Q0003, rel=1e-5)
    assert roubo["knn_media3"] == pytest.approx(SIM_Q0003 / 3, rel=1e-5)
    assert roubo["knn_n10"] == pytest.approx(0.1)


def test_par_sem_vizinho_do_tema_tem_knn_zerado(tabela_q0001):
    linhas, _ = tabela_q0001
    for tema_id in (PATRIMONIO, MARIA_DA_PENHA):
        assert [linhas[tema_id][c] for c in ("knn_max", "knn_media3", "knn_n10", "knn_folha_max")] == [0, 0, 0, 0]


def test_knn_folha_max_so_com_vizinhos_da_folha_de_treino(ctx):
    knn, pesos = ctx.indice_rotulado({"Q0003"})  # Q0002 (Furto) fica fora da folha de treino
    linhas = linhas_por_tema(tabela_pares(["Q0001"], {"Q0001": CANDIDATOS_Q0001}, ctx, knn, pesos, {"Q0003"}))
    assert linhas[FURTO]["knn_max"] == pytest.approx(SIM_S0001, rel=1e-5)
    assert linhas[FURTO]["knn_folha_max"] == 0
    assert linhas[ROUBO]["knn_folha_max"] == pytest.approx(SIM_Q0003, rel=1e-5)


def test_vizinhos_ja_buscados_e_so_os_10_primeiros_em_knn_n10(ctx):
    vizinhos = [(f"S{i:03d}", FURTO if i % 2 == 0 else ROUBO, 0.9 - 0.01 * i) for i in range(15)]
    linhas = linhas_por_tema(tabela_pares(["Q0001"], {"Q0001": CANDIDATOS_Q0001}, ctx, VAZIO, {}, set(),
                                          vizinhos_por_id={"Q0001": vizinhos}))
    assert linhas[FURTO]["knn_n10"] == pytest.approx(0.5)  # 0, 2, 4, 6, 8 entre os 10 primeiros
    assert linhas[ROUBO]["knn_n10"] == pytest.approx(0.5)
    assert linhas[FURTO]["knn_max"] == pytest.approx(0.9)
    assert linhas[ROUBO]["knn_media3"] == pytest.approx((0.89 + 0.87 + 0.85) / 3)
    so_cinco = linhas_por_tema(tabela_pares(["Q0001"], {"Q0001": CANDIDATOS_Q0001}, ctx, VAZIO, {}, set(),
                                            vizinhos_por_id={"Q0001": vizinhos}, k_knn=5))
    assert so_cinco[FURTO]["knn_n10"] == pytest.approx(0.3)


def test_bm25_dividido_pela_maior_pontuacao_da_questao(ctx, tabela_q0001):
    linhas, _ = tabela_q0001
    pontos = ctx.indice_bm25.pontuacoes(ctx.texto("Q0001"))
    assert pontos.max() > 0
    posicao = {t: i for i, t in enumerate(ctx.temas_ids)}
    # o texto da questão cita "Lei", que só aparece nos caminhos dos temas da Lei Maria da Penha
    assert linhas[MARIA_DA_PENHA]["bm25"] == pytest.approx(pontos[posicao[MARIA_DA_PENHA]] / pontos.max())
    assert linhas[MARIA_DA_PENHA]["bm25"] > 0
    assert linhas[FURTO]["bm25"] == 0


def test_dic_lei_e_dic_art(ctx, tabela_q0001):
    linhas, _ = tabela_q0001
    assert ctx.chaves("Q0001") == {"art:155:cp": 1, "diploma:cp": 1, "lei:11340": 1}
    assert (linhas[FURTO]["dic_art"], linhas[FURTO]["dic_lei"]) == (pytest.approx(1.0), pytest.approx(0.25))
    assert (linhas[PATRIMONIO]["dic_art"], linhas[PATRIMONIO]["dic_lei"]) == (0, pytest.approx(0.25))
    assert (linhas[MARIA_DA_PENHA]["dic_art"], linhas[MARIA_DA_PENHA]["dic_lei"]) == (0, pytest.approx(1.0))
    assert (linhas[ROUBO]["dic_art"], linhas[ROUBO]["dic_lei"]) == (0, 0)


def test_nivel2_n_buscas_e_pai_de_candidato(tabela_q0001):
    linhas, _ = tabela_q0001
    assert {t: linhas[t]["nivel2"] for t in linhas} == {
        FURTO: 1, ROUBO: 1, ESTELIONATO: 1, PATRIMONIO: 0, MARIA_DA_PENHA: 0,
    }
    assert {t: linhas[t]["n_buscas"] for t in linhas} == {
        FURTO: 4, ROUBO: 2, PATRIMONIO: 1, ESTELIONATO: 0, MARIA_DA_PENHA: 1,
    }
    # Patrimônio tem Furto, Roubo e Estelionato entre os candidatos: vale o maior cos_tema deles (Furto)
    assert linhas[PATRIMONIO]["pai_de_candidato"] == pytest.approx(COS_FURTO, rel=1e-5)
    assert linhas[MARIA_DA_PENHA]["pai_de_candidato"] == 0  # nível 1 sem filho candidato
    assert linhas[FURTO]["pai_de_candidato"] == 0  # nível 2


def test_disciplina_bate(ctx):
    knn, pesos = ctx.indice_rotulado(set())
    candidatos = {
        "Q0001": [Candidato(FURTO, {"knn"}, 1.0), Candidato(MARIA_DA_PENHA, {"knn"}, 1.0)],  # Direito Penal
        "Q0005": [Candidato(FURTO, {"knn"}, 1.0), Candidato(MARIA_DA_PENHA, {"knn"}, 1.0)],  # Legislação Especial
        "Q0004": [Candidato(FURTO, {"knn"}, 1.0)],  # Direito Penal Militar
        "S0001": [Candidato(FURTO, {"knn"}, 1.0)],  # exemplo sintético: sem disciplina de origem
    }
    tabela = tabela_pares(list(candidatos), candidatos, ctx, knn, pesos, set())
    bate = {(linha["id"], linha["tema_id"]): linha["disciplina_bate"] for linha in tabela.to_dict("records")}
    assert bate == {
        ("Q0001", FURTO): 1, ("Q0001", MARIA_DA_PENHA): 0,
        ("Q0005", FURTO): 0, ("Q0005", MARIA_DA_PENHA): 1,
        ("Q0004", FURTO): 0,
        ("S0001", FURTO): 0,
    }


def test_normalizar_disciplina():
    assert normalizar_disciplina("Direito Penal") == normalizar_disciplina("Penal") == "Penal"
    assert normalizar_disciplina("  direito  penal ") == "Penal"
    assert normalizar_disciplina("Processo Penal") == normalizar_disciplina("Direito Processual Penal") == \
        "Processo Penal"
    assert normalizar_disciplina("Direito Penal - Legislação Especial") == "Direito Penal - Legislação Especial"
    assert normalizar_disciplina("Direito Penal-Legislacao Especial") == "Direito Penal - Legislação Especial"
    assert normalizar_disciplina("Criminologia") == "Criminologia"
    for outra in ("Direito Penal Militar", "Direito Processual Penal Militar", "Direito Constitucional",
                  "Direito Civil - Legislação Especial", "Disciplina Unificada", "", None):
        assert normalizar_disciplina(outra) is None


def test_prior_log_da_contagem_normalizado(arvore, ctx, tabela_q0001):
    linhas, _ = tabela_q0001
    # no contexto pequeno, o maior é Dos Crimes Contra o Patrimônio, com 325 questões na fonte
    assert linhas[PATRIMONIO]["prior"] == pytest.approx(1.0)
    assert linhas[FURTO]["prior"] == pytest.approx(math.log1p(114) / math.log1p(325))
    assert linhas[MARIA_DA_PENHA]["prior"] == pytest.approx(math.log1p(18) / math.log1p(325))
    todos = prior_dos_temas(arvore, arvore.candidatos_possiveis())
    assert max(todos.values()) == pytest.approx(1.0)
    assert todos[6724] == todos[4704]  # duplicado usa a contagem do original
    assert todos[5163] == 0  # "0 questões no source"
    assert todos[5477] == 0  # Habeas Corpus: descrição sem contagem


def test_tabela_vazia_e_questao_sem_candidato(ctx):
    knn, pesos = ctx.indice_rotulado(set())
    vazia = tabela_pares(["Q0001"], {"Q0001": []}, ctx, knn, pesos, set())
    assert list(vazia.columns) == COLUNAS and vazia.empty


def test_com_candidatos_gerados_de_verdade(ctx):
    treino = {"Q0002", "Q0003", "Q0004"}
    knn, pesos = ctx.indice_rotulado(treino, com_corpus=True)
    candidatos, vizinhos = gerar_lote(["Q0001", "Q0005"], ctx, knn, pesos)
    tabela = tabela_pares(["Q0001", "Q0005"], candidatos, ctx, knn, pesos, treino, vizinhos_por_id=vizinhos)
    rebuscada = tabela_pares(["Q0001", "Q0005"], candidatos, ctx, knn, pesos, treino)
    assert tabela.equals(rebuscada)
    assert len(tabela) == sum(len(c) for c in candidatos.values())
    assert set(tabela["n_buscas"]) <= {0, 1, 2, 3, 4}
    assert DISPOSICOES in set(tabela.loc[tabela["id"] == "Q0001", "tema_id"])  # filho do pai que veio pelo dicionário


# --- votos das buscas ----------------------------------------------------------------------------------------


def votos_por_tema(tabela, id_questao):
    colunas = ("voto_bm25", "voto_cos", "voto_dic", "voto_knn", "votos", "votos_ramo")
    return {t: tuple(int(linha[c]) for c in colunas) for t, linha in linhas_por_tema(tabela, id_questao).items()}


def test_topos_das_buscas_num_caso_construido(ctx):
    knn, pesos = ctx.indice_rotulado({"Q0002", "Q0003"}, com_corpus=True)
    vizinhos = knn.buscar(ctx.vetor("Q0001"), 15, excluir=frozenset({"Q0001"}))
    # Q0001 (furto 1, roubo 0,5): cosseno, dicionário (art. 155 + CP) e vizinhos apontam Furto; o BM25 só acha a
    # palavra "Lei", que está no caminho do duplicado Penal > Lei Maria da Penha
    assert topos_das_buscas("Q0001", ctx, vizinhos, pesos) == {
        "bm25": MARIA_DA_PENHA_DUP, "cos": FURTO, "dic": FURTO, "knn": FURTO,
    }
    # Q0003 (roubo): ela mesma não é vizinha; os vizinhos de Furto (corpus × 0,5 + folha + exemplo) somam mais
    vizinhos = knn.buscar(ctx.vetor("Q0003"), 15, excluir=frozenset({"Q0003"}))
    assert topos_das_buscas("Q0003", ctx, vizinhos, pesos) == {"bm25": ROUBO, "cos": ROUBO, "dic": None,
                                                               "knn": FURTO}


def test_votos_e_votos_ramo_por_par(ctx):
    treino = {"Q0002", "Q0003"}
    knn, pesos = ctx.indice_rotulado(treino, com_corpus=True)
    q0001 = votos_por_tema(tabela_pares(["Q0001"], {"Q0001": CANDIDATOS_Q0001}, ctx, knn, pesos, treino), "Q0001")
    # (bm25, cos, dic, knn, votos, votos_ramo): três buscas no Furto; as três no ramo do patrimônio
    assert q0001 == {
        FURTO: (0, 1, 1, 1, 3, 3),
        ROUBO: (0, 0, 0, 0, 0, 3),
        PATRIMONIO: (0, 0, 0, 0, 0, 3),
        ESTELIONATO: (0, 0, 0, 0, 0, 3),
        MARIA_DA_PENHA: (0, 0, 0, 0, 0, 0),  # o top-1 do BM25 é o duplicado, que é outro ramo
    }
    candidatos = {"Q0003": [Candidato(ROUBO, {"bm25"}, 1.0), Candidato(FURTO, {"knn"}, 1.0),
                            Candidato(DROGAS, {"bm25"}, 0.1)]}
    q0003 = votos_por_tema(tabela_pares(["Q0003"], candidatos, ctx, knn, pesos, treino), "Q0003")
    assert q0003 == {ROUBO: (1, 1, 0, 0, 2, 3), FURTO: (0, 0, 0, 1, 1, 3), DROGAS: (0, 0, 0, 0, 0, 0)}


def test_topo_knn_soma_similaridade_ponderada_e_ignora_fora():
    publicaveis = {FURTO, ROUBO}
    vizinhos = [("C1", FURTO, 0.9), ("Q1", ROUBO, 0.8), ("S1", None, 0.99), ("S2", FURTO, 0.2),
                ("S3", ROUBO, -0.5), ("S4", 5062, 0.99)]  # 5062 é raiz: não publicável
    # Furto: 0,9 × 0,5 (corpus) + 0,2 = 0,65; Roubo: 0,8 + 0 (negativo conta zero) = 0,8
    assert topo_knn(vizinhos, {"C1": 0.5}, publicaveis) == ROUBO
    assert topo_knn(vizinhos, {}, publicaveis) == FURTO  # sem o peso do corpus, Furto soma 1,1
    assert topo_knn([("S1", None, 0.9)], {}, publicaveis) is None
    assert topo_knn([("A", ROUBO, 0.5), ("B", FURTO, 0.5)], {}, publicaveis) == FURTO  # empate: menor id


def test_votos_do_tema_sem_busca_nenhuma(arvore):
    vazio = {"bm25": None, "cos": None, "dic": None, "knn": None}
    assert votos_do_tema(FURTO, vazio, arvore) == {"voto_bm25": 0, "voto_cos": 0, "voto_dic": 0, "voto_knn": 0,
                                                    "votos": 0, "votos_ramo": 0}
    todos_no_pai = dict.fromkeys(vazio, PATRIMONIO)
    assert votos_do_tema(PATRIMONIO, todos_no_pai, arvore)["votos"] == 4
    assert votos_do_tema(ROUBO, todos_no_pai, arvore)["votos_ramo"] == 4
    assert votos_do_tema(ROUBO, todos_no_pai, arvore)["votos"] == 0
