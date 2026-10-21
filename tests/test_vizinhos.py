import math

import numpy as np
import pytest

from sistema.vizinhos import IndiceVetorial


@pytest.fixture
def indice():
    matriz = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],   # norma 2: a normalização tira a escala
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 5.0],
    ])
    return IndiceVetorial(["a", "b", "c", "d"], matriz, [10, 20, None, 10])


def test_buscar_ordena_por_cosseno_e_devolve_id_rotulo_e_similaridade(indice):
    resultado = indice.buscar(np.array([3.0, 1.0, 0.0]), 3)
    assert [(id_, rotulo) for id_, rotulo, _ in resultado] == [("a", 10), ("c", None), ("b", 20)]
    norma = math.sqrt(10)
    esperado = [3 / norma, (3 + 1) / (norma * math.sqrt(2)), 1 / norma]
    assert [similaridade for _, _, similaridade in resultado] == pytest.approx(esperado, rel=1e-6)
    assert all(isinstance(similaridade, float) for _, _, similaridade in resultado)


def test_a_escala_da_consulta_e_das_linhas_nao_muda_a_similaridade(indice):
    uma = indice.buscar(np.array([0.0, 1.0, 0.0]), 1)
    outra = indice.buscar(np.array([0.0, 7.0, 0.0]), 1)
    assert uma[0][0] == outra[0][0] == "b"
    assert uma[0][2] == pytest.approx(1.0) and outra[0][2] == pytest.approx(1.0)


def test_excluir_tira_ids_e_completa_k_com_os_seguintes(indice):
    resultado = indice.buscar(np.array([3.0, 1.0, 0.0]), 3, excluir=frozenset({"c"}))
    assert [id_ for id_, _, _ in resultado] == ["a", "b", "d"]
    assert [id_ for id_, _, _ in indice.buscar(np.array([3.0, 1.0, 0.0]), 2, excluir={"a"})] == ["c", "b"]
    assert indice.buscar(np.array([1.0, 0.0, 0.0]), 2, excluir={"a", "b", "c", "d"}) == []


def test_k_maior_que_o_indice_e_k_zero(indice):
    assert len(indice.buscar(np.array([1.0, 1.0, 1.0]), 50)) == 4
    assert indice.buscar(np.array([1.0, 1.0, 1.0]), 0) == []


def test_empate_mantem_a_ordem_de_insercao():
    indice = IndiceVetorial(["x", "y", "z"], np.ones((3, 2)), [1, 2, 3])
    assert [id_ for id_, _, _ in indice.buscar(np.array([1.0, 1.0]), 3)] == ["x", "y", "z"]


def test_similaridades_alinhadas_aos_ids(indice):
    similaridades = indice.similaridades(np.array([0.0, 0.0, 1.0]))
    assert similaridades.shape == (4,)
    assert similaridades == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-7)
    assert indice.posicao("d") == 3
    assert len(indice) == 4


def test_linha_ou_consulta_nula_tem_similaridade_zero():
    indice = IndiceVetorial(["nula", "um"], np.array([[0.0, 0.0], [1.0, 0.0]]), [1, 2])
    assert indice.similaridades(np.array([1.0, 0.0])) == pytest.approx([0.0, 1.0])
    assert indice.similaridades(np.zeros(2)) == pytest.approx([0.0, 0.0])


def test_indice_vazio_nao_devolve_nada():
    vazio = IndiceVetorial([], np.zeros((0, 0)), [])
    assert vazio.buscar(np.array([1.0, 2.0]), 5) == []
    assert len(vazio) == 0


def test_valida_tamanhos_e_ids():
    with pytest.raises(ValueError):
        IndiceVetorial(["a", "b"], np.ones((3, 2)), [1, 2])
    with pytest.raises(ValueError):
        IndiceVetorial(["a", "b"], np.ones((2, 2)), [1])
    with pytest.raises(ValueError):
        IndiceVetorial(["a", "a"], np.ones((2, 2)), [1, 2])
    indice = IndiceVetorial(["a"], np.ones((1, 3)), [1])
    with pytest.raises(ValueError):
        indice.buscar(np.ones(2), 1)
