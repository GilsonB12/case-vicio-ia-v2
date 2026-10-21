import math
import random
import time

import pytest

from apoio_contexto import FURTO, MARIA_DA_PENHA, MARIA_DA_PENHA_DUP, PATRIMONIO, ROUBO
from sistema.arvore import ArvoreTemas
from sistema.termos import IndiceTermos, formas, normalizar

ERRO_DE_TIPO = 5224
CONCURSO = 5071


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


def indice(arvore, por_tema, temas_ids=None):
    """IndiceTermos com os `termos` de cada tema (e os `sinonimos`, quando o valor é um par de listas)."""
    enriquecidos = {}
    for tema_id, valor in por_tema.items():
        termos, sinonimos = valor if isinstance(valor, tuple) else (valor, [])
        enriquecidos[tema_id] = {"tema_id": tema_id, "termos": termos, "sinonimos": sinonimos}
    return IndiceTermos(enriquecidos, arvore, temas_ids=temas_ids or list(por_tema))


# --- normalização -------------------------------------------------------------------------------------------


def test_normalizar_minusculas_sem_acento_espacos_simples_e_hifen_como_espaco():
    assert normalizar("  Execução   da PENA ") == "execucao da pena"
    assert normalizar("Co-Autoria") == "co autoria"
    assert normalizar("provas cautelares, não repetíveis e antecipadas") == \
        "provas cautelares nao repetiveis e antecipadas"
    assert normalizar("") == "" and normalizar(None) == ""


def test_formas_da_expressao_com_hifen_incluem_a_forma_sem_espaco():
    assert formas("co-autoria") == ("co autoria", "coautoria")
    assert formas("relação autor-vítima") == ("relacao autor vitima", "relacao autorvitima")
    assert formas("Erro sobre a Pessoa") == ("erro sobre a pessoa",)
    assert formas("  ") == ()


# --- casamento de frase ---------------------------------------------------------------------------------------


def test_casa_a_frase_inteira_sem_acento_e_sem_caixa(arvore):
    termos = indice(arvore, {ERRO_DE_TIPO: ["erro de tipo", "erro sobre a pessoa"], CONCURSO: ["execução penal"]})
    assert set(termos.pontuar("Houve ERRO DE TIPO essencial.")) == {ERRO_DE_TIPO}
    assert set(termos.pontuar("o agente incidiu em erro sobre a pessoa")) == {ERRO_DE_TIPO}
    assert set(termos.pontuar("Compete ao juízo da EXECUCAO   PENAL")) == {CONCURSO}
    # palavras soltas ou em outra ordem não casam
    assert termos.pontuar("o tipo penal admite erro de fato") == {}
    assert termos.pontuar("erro quanto ao tipo") == {}


def test_casamento_respeita_a_fronteira_de_palavra(arvore):
    termos = indice(arvore, {ERRO_DE_TIPO: ["erro de tipo", "coautoria"]})
    assert termos.pontuar("erro de tipologia vitimológica") == {}
    assert termos.pontuar("houve coautorias diversas") == {}
    assert termos.pontuar("superErro de tipo") == {}
    assert set(termos.pontuar("(erro de tipo)")) == {ERRO_DE_TIPO}


def test_hifen_casa_com_espaco_e_sem_espaco(arvore):
    com_hifen = indice(arvore, {CONCURSO: ["co-autoria"]})
    for texto in ("Trata-se de co-autoria.", "Trata-se de coautoria.", "Trata-se de co autoria."):
        assert set(com_hifen.pontuar(texto)) == {CONCURSO}, texto
    # a expressão sem hífen casa o texto com hífen; "co autoria" são duas palavras que ela não tem
    sem_hifen = indice(arvore, {CONCURSO: ["coautoria"]})
    assert set(sem_hifen.pontuar("Trata-se de co-autoria.")) == {CONCURSO}
    assert set(sem_hifen.pontuar("Trata-se de coautoria.")) == {CONCURSO}
    assert sem_hifen.pontuar("Trata-se de co autoria.") == {}
    # as duas grafias são a mesma expressão
    ambas = indice(arvore, {CONCURSO: ["co-autoria", "coautoria"]})
    assert ambas.expressoes_do_tema(CONCURSO) == ["coautoria"]
    assert ambas.pontuar("coautoria e co-autoria")[CONCURSO][1] == 1


def test_expressao_de_uma_palavra_curta_fica_de_fora(arvore):
    termos = indice(arvore, {FURTO: ["dolo", "culpa", "art.", "ação"], ROUBO: ["dolo eventual"]})
    assert termos.expressoes_do_tema(FURTO) == ["culpa"]
    assert termos.pontuar("dolo e ação") == {}
    assert set(termos.pontuar("dolo eventual e culpa")) == {FURTO, ROUBO}


def test_termos_e_sinonimos_entram(arvore):
    termos = indice(arvore, {FURTO: (["subtração"], ["furto de uso"])})
    assert termos.expressoes_do_tema(FURTO) == ["furto de uso", "subtracao"]


# --- pesos ----------------------------------------------------------------------------------------------------


def test_peso_idf_e_multipalavra(arvore):
    temas = [FURTO, ROUBO, PATRIMONIO, ERRO_DE_TIPO]
    termos = indice(arvore, {FURTO: ["coisa alheia", "subtração"], ROUBO: ["coisa alheia", "violência"]},
                    temas_ids=temas)
    n = len(temas)
    assert termos.peso("coisa alheia") == pytest.approx(math.log(1 + n / 2) * 1.5)
    assert termos.peso("Subtração") == pytest.approx(math.log(1 + n / 1))
    assert termos.peso("não existe") == 0
    # o maior peso do índice: aqui o da expressão de duas palavras em dois temas, acima de log(1 + 4) de uma palavra
    assert termos.peso_maximo == pytest.approx(math.log(1 + n / 2) * 1.5)
    assert termos.peso_maximo > math.log(1 + n)


def test_pontuar_soma_conta_e_maior_peso(arvore):
    temas = [FURTO, ROUBO, PATRIMONIO, ERRO_DE_TIPO]
    termos = indice(arvore, {FURTO: ["coisa alheia", "subtração"], ROUBO: ["coisa alheia", "violência"]},
                    temas_ids=temas)
    pontos = termos.pontuar("subtração de coisa alheia; subtração repetida conta uma vez")
    comum, so_furto = termos.peso("coisa alheia"), termos.peso("subtração")
    assert set(pontos) == {FURTO, ROUBO}
    assert pontos[FURTO] == pytest.approx((comum + so_furto, 2, max(comum, so_furto)))
    assert pontos[ROUBO] == pytest.approx((comum, 1, comum))
    assert termos.pontuar("") == {} and termos.pontuar(None) == {}


def test_duplicado_herda_as_expressoes_do_original(arvore):
    assert arvore.original_de(MARIA_DA_PENHA_DUP) == MARIA_DA_PENHA
    enriquecidos = {MARIA_DA_PENHA: {"tema_id": MARIA_DA_PENHA, "termos": ["medida protetiva de urgência"]}}
    termos = IndiceTermos(enriquecidos, arvore, temas_ids=[MARIA_DA_PENHA, MARIA_DA_PENHA_DUP, FURTO])
    assert termos.expressoes_do_tema(MARIA_DA_PENHA_DUP) == ["medida protetiva de urgencia"]
    assert set(termos.pontuar("Cabe medida protetiva de urgência.")) == {MARIA_DA_PENHA, MARIA_DA_PENHA_DUP}
    assert termos.peso("medida protetiva de urgência") == pytest.approx(math.log(1 + 3 / 2) * 1.5)


def test_so_temas_publicaveis_e_indice_vazio(arvore):
    raiz = 5062
    assert not arvore.publicavel(raiz)
    termos = indice(arvore, {raiz: ["direito penal"], FURTO: ["furto noturno"]}, temas_ids=[raiz, FURTO])
    assert termos.temas_ids == [FURTO]
    peso = termos.peso("furto noturno")
    assert termos.pontuar("direito penal e furto noturno") == {FURTO: pytest.approx((peso, 1, peso))}
    vazio = IndiceTermos({}, arvore)
    assert vazio.temas_ids == arvore.candidatos_possiveis()
    assert vazio.pontuar("erro de tipo") == {} and vazio.peso_maximo == 0 and len(vazio) == 0


# --- tempo ----------------------------------------------------------------------------------------------------


def test_pontuar_659_questoes_longas_contra_371_temas_em_menos_de_2_segundos(arvore):
    sorteio = random.Random(11)
    letras = "abcdefghijlmnopqrstuvxz"
    vocabulario = ["".join(sorteio.choices(letras, k=sorteio.randint(3, 10))) for _ in range(3000)]
    temas = arvore.candidatos_possiveis()
    assert len(temas) == 371
    enriquecidos = {
        tema_id: {"tema_id": tema_id,
                  "termos": [" ".join(sorteio.choices(vocabulario, k=sorteio.choice((1, 2, 2, 3, 3, 4, 5, 7, 10))))
                             for _ in range(22)],
                  "sinonimos": [" ".join(sorteio.choices(vocabulario, k=3)) for _ in range(5)]}
        for tema_id in temas
    }
    termos = IndiceTermos(enriquecidos, arvore)
    # cada texto leva expressões de verdade no meio de 450 palavras soltas
    expressoes = termos.expressoes
    textos = [" ".join([*sorteio.choices(vocabulario, k=450), *sorteio.sample(expressoes, 5), "co-autoria"])
              for _ in range(659)]
    inicio = time.perf_counter()
    resultados = [termos.pontuar(texto) for texto in textos]
    assert time.perf_counter() - inicio < 2.0
    assert all(resultados)
