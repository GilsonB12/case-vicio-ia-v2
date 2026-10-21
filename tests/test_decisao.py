import numpy as np
import pandas as pd
import pytest

from apoio_contexto import DROGAS, ESTELIONATO, EXTORSAO, FURTO, MARIA_DA_PENHA, PATRIMONIO, ROUBO
from sistema.arvore import ArvoreTemas
from sistema.decisao import FORA, PUBLICAR, REVISAR, Decisao, Limiares, decidir, decidir_listas

LIMIARES = Limiares(publicar=0.80, fora=0.60)


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


def prob(*pares):
    return pd.DataFrame([{"id": "Q1", "tema_id": t, "s": p, "p": p} for t, p in pares],
                        columns=["id", "tema_id", "s", "p"])


def ramos(*pares):
    return pd.DataFrame([{"id": "Q1", "tema_id": t, "p_ramo": p} for t, p in pares],
                        columns=["id", "tema_id", "p_ramo"])


def test_regra_1_fora_com_os_gatilhos_no_motivo(arvore):
    decisao = decidir("Q1", prob((FURTO, 0.9)), ramos(), 0.6, LIMIARES, arvore,
                      gatilhos=["chaves diploma:cpm", "termos militar"])
    assert isinstance(decisao, Decisao)
    assert (decisao.decisao, decisao.tema_central_id, decisao.extras_ids, decisao.regra) == (FORA, None, [], 1)
    assert decisao.confianca == 0.6
    assert "diploma:cpm" in decisao.motivo and "militar" in decisao.motivo and "0,60" in decisao.motivo


def test_regra_1_sem_gatilho_explica_pela_probabilidade(arvore):
    decisao = decidir("Q1", prob((FURTO, 0.3), (ROUBO, 0.2)), ramos(), 0.91, LIMIARES, arvore)
    assert decisao.decisao == FORA and decisao.confianca == 0.91
    assert "0,30" in decisao.motivo


def test_regra_2_publica_o_tema_de_maior_p(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.85), (FURTO, 0.1), (DROGAS, 0.05)), ramos(), 0.59, LIMIARES, arvore)
    assert (decisao.decisao, decisao.tema_central_id, decisao.regra) == (PUBLICAR, ROUBO, 2)
    assert decisao.confianca == 0.85 and decisao.motivo == ""


def test_regra_2_no_limiar_exato_publica(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.80), (FURTO, 0.20)), ramos(), 0.0, LIMIARES, arvore)
    assert decisao.decisao == PUBLICAR


def test_regra_3_publica_o_pai_quando_os_filhos_dividem(arvore):
    p = prob((FURTO, 0.45), (ROUBO, 0.30), (PATRIMONIO, 0.10), (MARIA_DA_PENHA, 0.15))
    decisao = decidir("Q1", p, ramos((PATRIMONIO, 0.85), (MARIA_DA_PENHA, 0.15)), 0.1, LIMIARES, arvore)
    assert (decisao.decisao, decisao.tema_central_id, decisao.regra) == (PUBLICAR, PATRIMONIO, 3)
    assert decisao.confianca == 0.85 and decisao.motivo == ""
    # extras: os filhos do pai publicado ficam de fora; Maria da Penha (0,15) entra
    assert decisao.extras_ids == [MARIA_DA_PENHA]


def test_regra_4_revisar_abaixo_do_limiar(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.62), (FURTO, 0.30), (PATRIMONIO, 0.08)), ramos((PATRIMONIO, 0.70)),
                      0.2, LIMIARES, arvore)
    assert (decisao.decisao, decisao.tema_central_id, decisao.regra) == (REVISAR, ROUBO, 4)
    assert decisao.confianca == 0.62
    assert decisao.motivo == "probabilidade 0,62 abaixo do limiar 0,80"


def test_regra_4_divergencia_entre_ramos(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.45), (DROGAS, 0.40), (FURTO, 0.15)), ramos(), 0.2, LIMIARES, arvore)
    assert decisao.decisao == REVISAR and decisao.tema_central_id == ROUBO
    assert decisao.motivo.startswith("divergência entre ramos")
    assert str(ROUBO) in decisao.motivo and str(DROGAS) in decisao.motivo


def test_regra_4_dois_maiores_no_mesmo_ramo_nao_e_divergencia(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.45), (FURTO, 0.44), (DROGAS, 0.11)), ramos(), 0.2,
                      Limiares(publicar=0.95, fora=0.6), arvore)
    assert decisao.motivo.startswith("probabilidade 0,45 abaixo do limiar 0,95")


def test_regra_4_divergencia_so_com_soma_acima_de_0_8(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.40), (DROGAS, 0.40), (FURTO, 0.20)), ramos(), 0.2, LIMIARES, arvore)
    assert decisao.motivo.startswith("probabilidade")


def test_extras_ate_3_acima_do_limiar_sem_pai_nem_filhos(arvore):
    p = prob((ROUBO, 0.50), (PATRIMONIO, 0.20), (DROGAS, 0.16), (MARIA_DA_PENHA, 0.17), (ESTELIONATO, 0.15),
             (EXTORSAO, 0.15), (FURTO, 0.14))
    decisao = decidir("Q1", p, ramos(), 0.0, Limiares(publicar=0.5, fora=0.9), arvore)
    assert decisao.tema_central_id == ROUBO
    # PATRIMONIO é o pai de Roubo: fora; Furto abaixo de 0,15; empate em 0,15 desempata pelo id
    assert decisao.extras_ids == [MARIA_DA_PENHA, DROGAS, EXTORSAO]
    assert PATRIMONIO not in decisao.extras_ids and len(decisao.extras_ids) == 3


def test_extras_de_um_central_de_nivel_1_excluem_os_filhos(arvore):
    p = prob((PATRIMONIO, 0.60), (FURTO, 0.20), (DROGAS, 0.20))
    decisao = decidir("Q1", p, ramos((PATRIMONIO, 0.8)), 0.0, Limiares(publicar=0.6, fora=0.9), arvore)
    assert decisao.tema_central_id == PATRIMONIO and decisao.regra == 2
    assert decisao.extras_ids == [DROGAS]


def test_extra_tambem_em_revisar(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.5), (DROGAS, 0.3), (FURTO, 0.2)), ramos(), 0.0, LIMIARES, arvore)
    # Furto é irmão de Roubo (não pai nem filho): pode ser extra
    assert decisao.decisao == REVISAR and decisao.extras_ids == [DROGAS, FURTO]


def test_sem_candidato_sai_fora_com_motivo(arvore):
    decisao = decidir("Q1", prob(), ramos(), 0.1, LIMIARES, arvore)
    assert decisao.decisao == FORA and decisao.tema_central_id is None and decisao.motivo


def test_confianca_com_duas_casas(arvore):
    decisao = decidir("Q1", prob((ROUBO, 0.876543), (FURTO, 0.123457)), ramos(), 0.0, LIMIARES, arvore)
    assert decisao.confianca == 0.88


def test_motivo_obrigatorio_em_revisar_e_fora_e_vazio_em_publicar(arvore):
    gerador = np.random.default_rng(3)
    temas = [FURTO, ROUBO, EXTORSAO, ESTELIONATO, PATRIMONIO, MARIA_DA_PENHA, DROGAS]
    for _ in range(300):
        n = int(gerador.integers(1, len(temas) + 1))
        escolhidos = list(gerador.choice(temas, size=n, replace=False))
        valores = gerador.dirichlet(np.ones(n))
        pares = list(zip(map(int, escolhidos), valores))
        filhos = sum(p for t, p in pares if arvore.pai(t) == PATRIMONIO)
        lista_ramos = [(PATRIMONIO, dict(pares)[PATRIMONIO] + filhos)] if PATRIMONIO in dict(pares) else []
        limiares = Limiares(publicar=float(gerador.uniform(0.3, 0.95)), fora=float(gerador.uniform(0.3, 0.9)))
        decisao = decidir_listas("Q1", pares, lista_ramos, float(gerador.random()), limiares, arvore)
        if decisao.decisao in (REVISAR, FORA):
            assert decisao.motivo
        else:
            assert decisao.motivo == "" and decisao.tema_central_id is not None
        assert 0 <= decisao.confianca <= 1
        assert decisao.tema_central_id not in decisao.extras_ids


def test_listas_e_tabelas_dao_a_mesma_decisao(arvore):
    pares = [(FURTO, 0.45), (ROUBO, 0.30), (PATRIMONIO, 0.10), (DROGAS, 0.15)]
    por_tabela = decidir("Q1", prob(*pares), ramos((PATRIMONIO, 0.85)), 0.1, LIMIARES, arvore)
    por_lista = decidir_listas("Q1", pares, [(PATRIMONIO, 0.85)], 0.1, LIMIARES, arvore)
    assert por_tabela == por_lista
