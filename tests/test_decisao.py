import numpy as np
import pandas as pd
import pytest

from apoio_contexto import DROGAS, ESTELIONATO, EXTORSAO, FURTO, MARIA_DA_PENHA, PATRIMONIO, ROUBO
from sistema.arvore import ArvoreTemas
from sistema.decisao import (
    FORA,
    PUBLICAR,
    REVISAR,
    Decisao,
    Limiares,
    Opcao,
    decidir,
    decidir_listas,
    melhor_opcao,
    opcoes_de_tabela,
)

LIMIARES = Limiares(publicar=0.80, fora=0.60)


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


def prob(*pares):
    return pd.DataFrame([{"id": "Q1", "tema_id": t, "s": p, "p": p} for t, p in pares],
                        columns=["id", "tema_id", "s", "p"])


def opcoes(*linhas):
    """(opcao, tema_id, confianca, votos) de cada opção."""
    return pd.DataFrame([{"id": "Q1", "opcao": o, "tema_id": t, "confianca": c, "votos": v} for o, t, c, v in linhas],
                        columns=["id", "opcao", "tema_id", "confianca", "votos"])


def test_regra_1_fora_com_os_gatilhos_no_motivo(arvore):
    decisao = decidir("Q1", prob((FURTO, 0.9)), opcoes(("tema", FURTO, 0.99, 4)), 0.6, LIMIARES, arvore,
                      gatilhos=["chaves diploma:cpm", "termos militar"])
    assert isinstance(decisao, Decisao)
    assert (decisao.decisao, decisao.tema_central_id, decisao.extras_ids, decisao.regra) == (FORA, None, [], 1)
    assert decisao.confianca == 0.6
    assert "diploma:cpm" in decisao.motivo and "militar" in decisao.motivo and "0,60" in decisao.motivo


def test_regra_1_sem_gatilho_explica_pela_probabilidade(arvore):
    decisao = decidir("Q1", prob((FURTO, 0.3), (ROUBO, 0.2)), opcoes(("tema", FURTO, 0.5, 1)), 0.91, LIMIARES,
                      arvore)
    assert decisao.decisao == FORA and decisao.confianca == 0.91
    assert "0,30" in decisao.motivo


def test_regra_2_publica_o_top_1_pela_confianca_mesmo_com_p_baixo(arvore):
    # p do top-1 é 0,20 (o p normalizado é baixo com muitos candidatos); quem decide é a confiança
    decisao = decidir("Q1", prob((ROUBO, 0.20), (FURTO, 0.15), (DROGAS, 0.05)), opcoes(("tema", ROUBO, 0.85, 3)),
                      0.59, LIMIARES, arvore)
    assert (decisao.decisao, decisao.tema_central_id, decisao.regra) == (PUBLICAR, ROUBO, 2)
    assert decisao.confianca == 0.85 and decisao.motivo == ""


def test_regra_2_no_limiar_exato_publica(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.30), (FURTO, 0.20)), opcoes(("tema", ROUBO, 0.80, 2)), 0.0, LIMIARES,
                      arvore)
    assert decisao.decisao == PUBLICAR


def test_regra_3_escolhe_o_ramo_quando_a_confianca_do_ramo_e_maior(arvore):
    p = prob((FURTO, 0.30), (ROUBO, 0.28), (PATRIMONIO, 0.10), (MARIA_DA_PENHA, 0.15))
    o = opcoes(("tema", FURTO, 0.70, 1), ("ramo", PATRIMONIO, 0.92, 0))
    decisao = decidir("Q1", p, o, 0.1, LIMIARES, arvore)
    assert (decisao.decisao, decisao.tema_central_id, decisao.regra) == (PUBLICAR, PATRIMONIO, 3)
    assert decisao.confianca == 0.92 and decisao.motivo == ""
    # extras: os filhos do pai publicado ficam de fora; Maria da Penha (0,15) entra
    assert decisao.extras_ids == [MARIA_DA_PENHA]


def test_tema_fica_quando_a_confianca_dele_e_maior_que_a_do_ramo(arvore):
    o = opcoes(("tema", FURTO, 0.93, 3), ("ramo", PATRIMONIO, 0.90, 0))
    decisao = decidir("Q1", prob((FURTO, 0.30), (ROUBO, 0.28), (PATRIMONIO, 0.10)), o, 0.1, LIMIARES, arvore)
    assert (decisao.tema_central_id, decisao.regra, decisao.confianca) == (FURTO, 2, 0.93)


def test_empate_de_confianca_fica_com_o_tema(arvore):
    o = [Opcao("ramo", PATRIMONIO, 0.9, 0), Opcao("tema", FURTO, 0.9, 2)]
    assert melhor_opcao(o).opcao == "tema"
    decisao = decidir_listas("Q1", [(FURTO, 0.4), (PATRIMONIO, 0.1)], o, 0.0, LIMIARES, arvore)
    assert decisao.tema_central_id == FURTO and decisao.regra == 2


def test_regra_4_revisar_abaixo_do_limiar_com_o_top_1(arvore):
    o = opcoes(("tema", ROUBO, 0.62, 2), ("ramo", PATRIMONIO, 0.55, 0))
    decisao = decidir("Q1", prob((ROUBO, 0.32), (FURTO, 0.30), (PATRIMONIO, 0.08)), o, 0.2, LIMIARES, arvore)
    assert (decisao.decisao, decisao.tema_central_id, decisao.regra) == (REVISAR, ROUBO, 4)
    assert decisao.confianca == 0.62
    assert decisao.motivo == "confiança 0,62 abaixo do limiar 0,80"


def test_regra_4_com_o_ramo_melhor_mas_abaixo_do_limiar_revisa_o_top_1(arvore):
    o = opcoes(("tema", ROUBO, 0.40, 2), ("ramo", PATRIMONIO, 0.71, 0))
    decisao = decidir("Q1", prob((ROUBO, 0.32), (FURTO, 0.30), (PATRIMONIO, 0.08)), o, 0.2, LIMIARES, arvore)
    assert (decisao.decisao, decisao.tema_central_id) == (REVISAR, ROUBO)
    assert decisao.confianca == 0.40  # a chance de o central (o top-1) estar certo
    assert decisao.motivo == f"confiança 0,71 (ramo {PATRIMONIO}) abaixo do limiar 0,80"


@pytest.mark.parametrize("votos", [0, 1])
def test_regra_4_buscas_divergem_com_no_maximo_um_voto(arvore, votos):
    decisao = decidir("Q1", prob((ROUBO, 0.45), (DROGAS, 0.40), (FURTO, 0.15)), opcoes(("tema", ROUBO, 0.5, votos)),
                      0.2, LIMIARES, arvore)
    assert decisao.decisao == REVISAR and decisao.tema_central_id == ROUBO
    assert decisao.motivo.startswith("buscas divergem")
    assert f"{votos} de 4" in decisao.motivo and str(ROUBO) in decisao.motivo
    assert "confiança 0,50 abaixo do limiar 0,80" in decisao.motivo


def test_extras_ate_3_acima_do_limiar_sem_pai_nem_filhos(arvore):
    p = prob((ROUBO, 0.50), (PATRIMONIO, 0.20), (DROGAS, 0.16), (MARIA_DA_PENHA, 0.17), (ESTELIONATO, 0.15),
             (EXTORSAO, 0.15), (FURTO, 0.14))
    decisao = decidir("Q1", p, opcoes(("tema", ROUBO, 0.9, 3)), 0.0, Limiares(publicar=0.5, fora=0.9), arvore)
    assert decisao.tema_central_id == ROUBO
    # PATRIMONIO é o pai de Roubo: fora; Furto abaixo de 0,15; empate em 0,15 desempata pelo id
    assert decisao.extras_ids == [MARIA_DA_PENHA, DROGAS, EXTORSAO]
    assert PATRIMONIO not in decisao.extras_ids and len(decisao.extras_ids) == 3


def test_extras_de_um_central_de_nivel_1_excluem_os_filhos(arvore):
    p = prob((PATRIMONIO, 0.60), (FURTO, 0.20), (DROGAS, 0.20))
    decisao = decidir("Q1", p, opcoes(("tema", PATRIMONIO, 0.9, 2)), 0.0, Limiares(publicar=0.6, fora=0.9), arvore)
    assert decisao.tema_central_id == PATRIMONIO and decisao.regra == 2
    assert decisao.extras_ids == [DROGAS]


def test_extra_tambem_em_revisar(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.5), (DROGAS, 0.3), (FURTO, 0.2)), opcoes(("tema", ROUBO, 0.3, 2)), 0.0,
                      LIMIARES, arvore)
    # Furto é irmão de Roubo (não pai nem filho): pode ser extra
    assert decisao.decisao == REVISAR and decisao.extras_ids == [DROGAS, FURTO]


def test_sem_candidato_sai_fora_com_motivo(arvore):
    decisao = decidir("Q1", prob(), opcoes(), 0.1, LIMIARES, arvore)
    assert decisao.decisao == FORA and decisao.tema_central_id is None and decisao.motivo


def test_sem_opcoes_revisa_com_confianca_zero(arvore):
    decisao = decidir_listas("Q1", [(ROUBO, 0.6)], [], 0.0, LIMIARES, arvore)
    assert (decisao.decisao, decisao.tema_central_id, decisao.confianca) == (REVISAR, ROUBO, 0.0)
    assert decisao.motivo.startswith("buscas divergem")


def test_confianca_com_duas_casas(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.6), (FURTO, 0.4)), opcoes(("tema", ROUBO, 0.876543, 3)), 0.0, LIMIARES,
                      arvore)
    assert decisao.confianca == 0.88


def test_motivo_obrigatorio_em_revisar_e_fora_e_vazio_em_publicar(arvore):
    gerador = np.random.default_rng(3)
    temas = [FURTO, ROUBO, EXTORSAO, ESTELIONATO, PATRIMONIO, MARIA_DA_PENHA, DROGAS]
    for _ in range(300):
        n = int(gerador.integers(1, len(temas) + 1))
        escolhidos = list(gerador.choice(temas, size=n, replace=False))
        valores = gerador.dirichlet(np.ones(n))
        pares = sorted(zip(map(int, escolhidos), valores), key=lambda par: -par[1])
        topo = pares[0][0]
        lista = [Opcao("tema", topo, float(gerador.random()), int(gerador.integers(0, 5)))]
        if arvore.pai(topo) == PATRIMONIO:
            lista.append(Opcao("ramo", PATRIMONIO, float(gerador.random()), 0))
        limiares = Limiares(publicar=float(gerador.uniform(0.5, 0.99)), fora=float(gerador.uniform(0.3, 0.9)))
        decisao = decidir_listas("Q1", pares, lista, float(gerador.random()), limiares, arvore)
        if decisao.decisao in (REVISAR, FORA):
            assert decisao.motivo
        else:
            assert decisao.motivo == "" and decisao.tema_central_id is not None
            assert decisao.confianca >= round(limiares.publicar, 2) - 0.005
        if decisao.decisao == REVISAR:
            assert decisao.tema_central_id == topo
        assert 0 <= decisao.confianca <= 1
        assert decisao.tema_central_id not in decisao.extras_ids


def test_listas_e_tabelas_dao_a_mesma_decisao(arvore):
    pares = [(FURTO, 0.45), (ROUBO, 0.30), (PATRIMONIO, 0.10), (DROGAS, 0.15)]
    tabela = opcoes(("tema", FURTO, 0.70, 2), ("ramo", PATRIMONIO, 0.85, 0))
    por_tabela = decidir("Q1", prob(*pares), tabela, 0.1, LIMIARES, arvore)
    por_lista = decidir_listas("Q1", pares, opcoes_de_tabela(tabela), 0.1, LIMIARES, arvore)
    assert por_tabela == por_lista
    assert opcoes_de_tabela(tabela) == [Opcao("tema", FURTO, 0.70, 2), Opcao("ramo", PATRIMONIO, 0.85, 0)]
