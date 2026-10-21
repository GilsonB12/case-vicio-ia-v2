import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from sistema.arvore import ArvoreTemas
from sistema.indice_bm25 import STOPWORDS, IndiceBM25, tokenizar


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


# --- tokenização -----------------------------------------------------------------------------------------


def test_tokenizar_minusculas_sem_acento_e_sem_stopwords():
    texto = "A Ação Penal PÚBLICA é promovida pelo Ministério Público, e não pela vítima."
    assert tokenizar(texto) == ["acao", "penal", "publica", "promovida", "ministerio", "publico", "vitima"]


def test_tokenizar_so_letras_e_digitos_e_tira_tokens_de_uma_letra():
    assert tokenizar("art. 157, § 2º, inciso I, do CP; alínea b) e Lei 11.343/06") == [
        "art", "157", "inciso", "cp", "alinea", "lei", "11", "343", "06",
    ]


def test_tokenizar_tira_os_rotulos_de_estrutura_do_texto():
    texto = "Furto de coisa alheia móvel.\nComando: Assinale a correta.\nAlternativas:\nA) furto\nB) roubo"
    tokens = tokenizar(texto)
    assert "comando" not in tokens and "alternativas" not in tokens
    assert tokens[:4] == ["furto", "coisa", "alheia", "movel"]
    # no meio da frase a palavra fica: "penas alternativas" é conteúdo, não rótulo
    assert tokenizar("As penas alternativas: conceito") == ["penas", "alternativas", "conceito"]
    temas = "Penal > Furto\nLeis: CP (2.848/1940)\nArtigos: CP, art. 155\nTermos: subtração\nSinônimos: furtar"
    assert tokenizar(temas) == ["penal", "furto", "cp", "848", "1940", "cp", "art", "155", "subtracao", "furtar"]


def test_stopwords_estao_sem_acento_e_em_minusculas():
    assert {"de", "da", "do", "que", "nao", "pelo", "ao", "sao"} <= STOPWORDS
    assert all(palavra == palavra.lower() and palavra.isascii() for palavra in STOPWORDS)


def test_tokenizar_texto_vazio():
    assert tokenizar("") == []
    assert tokenizar(None) == []


# --- índice ----------------------------------------------------------------------------------------------


def test_buscar_devolve_primeiro_o_tema_cujo_texto_e_igual_a_consulta(arvore):
    ids = arvore.candidatos_possiveis()
    textos = [arvore.texto_tema(tema_id, {}) for tema_id in ids]
    indice = IndiceBM25(ids, textos)
    for tema_id in (5159, 5160, 5166, 5169, 4704, 4697):
        primeiro, pontuacao = indice.buscar(arvore.texto_tema(tema_id, {}), 5)[0]
        assert primeiro == tema_id
        assert pontuacao > 0


def test_buscar_ordena_corta_em_k_e_devolve_a_pontuacao_bruta():
    ids = ["a", "b", "c", "d", "e"]
    textos = ["furto simples de coisa alheia", "roubo com violência", "furto qualificado furto noturno",
              "estupro de vulnerável", "homicídio qualificado"]
    indice = IndiceBM25(ids, textos)
    resultado = indice.buscar("furto noturno", 2)
    assert [id_ for id_, _ in resultado] == ["c", "a"]
    brutas = BM25Okapi([tokenizar(t) for t in textos]).get_scores(tokenizar("furto noturno"))
    assert resultado[0][1] == pytest.approx(brutas[2])
    assert resultado[1][1] == pytest.approx(brutas[0])
    assert resultado[0][1] > resultado[1][1]


def test_buscar_deixa_de_fora_temas_sem_pontuacao():
    indice = IndiceBM25(["a", "b", "c"], ["furto simples", "roubo majorado", "estupro"])
    assert [id_ for id_, _ in indice.buscar("furto", 10)] == ["a"]
    assert indice.buscar("de que o", 10) == []
    assert indice.buscar("furto", 0) == []


def test_pontuacoes_iguais_as_do_bm25okapi(arvore):
    ids = arvore.candidatos_possiveis()
    textos = [arvore.texto_tema(tema_id, {}) for tema_id in ids]
    indice = IndiceBM25(ids, textos)
    consulta = ("Tício subtraiu, mediante grave ameaça, o celular da vítima; depois, em concurso com Mévio, "
                "praticou furto qualificado. Comando: Assinale a correta. Alternativas: A) roubo B) furto")
    referencia = BM25Okapi([tokenizar(t) for t in textos]).get_scores(tokenizar(consulta))
    np.testing.assert_allclose(indice.pontuacoes(consulta), referencia, rtol=1e-9, atol=1e-12)
    assert indice.ids == ids


def test_indice_valida_entradas():
    with pytest.raises(ValueError):
        IndiceBM25(["a", "b"], ["só um texto"])
    with pytest.raises(ValueError):
        IndiceBM25(["a", "a"], ["furto", "roubo"])
    with pytest.raises(ValueError):
        IndiceBM25(["a"], ["de que o"])
