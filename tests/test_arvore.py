import pytest

from sistema.arvore import ArvoreTemas, normalizar_titulo

DUPLICADOS_ESPERADOS = {
    6716, 6717, 6718, 6719, 6720, 6721, 6722, 6723, 6724, 6725,
    6726, 6727, 6728, 6732, 6733, 6734, 6735, 6736, 6737, 6738,
}

ENRIQUECIDO_MARIA_DA_PENHA = {
    "tema_id": 4704,
    "resumo": "Violência doméstica e familiar contra a mulher.",
    "leis": [{"numero": "11.340/2006", "nome": "Lei Maria da Penha"}],
    "artigos": [{"diploma": "Lei 11.340", "artigo": "5"}, {"diploma": "CP", "artigo": "129"}],
    "termos": ["medida protetiva de urgência", "violência doméstica"],
    "sinonimos": ["violência contra a mulher"],
    "exemplos": ["Questão de exemplo que não entra no texto do tema."],
    "descricao_original": "Subtema de 'Direito Penal - Legislação Especial'.",
}

TEXTO_MARIA_DA_PENHA = (
    "Direito Penal - Legislação Especial > Lei Maria da Penha\n"
    "Violência doméstica e familiar contra a mulher.\n"
    "Leis: Lei Maria da Penha (11.340/2006)\n"
    "Artigos: Lei 11.340, art. 5; CP, art. 129\n"
    "Termos: medida protetiva de urgência; violência doméstica\n"
    "Sinônimos: violência contra a mulher"
)


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


def test_normalizar_titulo_tira_acento_parte_e_caixa():
    assert normalizar_titulo("Dos Crimes Contra a Administração Pública (Parte 1)") == "dos crimes contra a administracao publica"
    assert normalizar_titulo("Teoria da Pena (Parte 2)") == "teoria da pena"


def test_duplicados_sao_exatamente_os_vinte(arvore):
    mapa = arvore.duplicados()
    assert set(mapa) == DUPLICADOS_ESPERADOS
    for duplicado, original in mapa.items():
        assert arvore.eh_duplicado(duplicado)
        assert not arvore.eh_duplicado(original)
        assert arvore.ativo(original)
        assert not arvore.eh_raiz(original)


def test_original_de_duplicados_conhecidos(arvore):
    assert arvore.original_de(6724) == 4704
    assert arvore.original_de(6726) == 5199
    assert arvore.original_de(6727) == 5199
    assert arvore.original_de(6732) == 5480
    assert arvore.original_de(6735) == 5543
    assert arvore.original_de(6736) == 5560
    assert arvore.original_de(6722) == 4696
    assert arvore.original_de(6734) == 4703


def test_original_de_tema_comum_e_ele_mesmo(arvore):
    assert arvore.original_de(5159) == 5159
    assert arvore.original_de(4704) == 4704


def test_navegacao_basica(arvore):
    assert arvore.pai(5527) == 5524
    assert 5527 in arvore.filhos(5524)
    assert 5528 in arvore.irmaos(5527)
    assert 5527 not in arvore.irmaos(5527)
    assert arvore.pai(5062) is None
    assert arvore.eh_raiz(5465)
    assert arvore.nivel(5527) == 2
    assert arvore.disciplina(5527) == "Processo Penal"
    assert arvore.titulo(5159) == "Furto"
    assert arvore.caminho(6724) == "Penal > Lei Maria da Penha"
    assert arvore.existe(5159) and not arvore.existe(1)


def test_tema_inexistente_levanta_key_error(arvore):
    with pytest.raises(KeyError, match="1"):
        arvore.caminho(1)


def test_ancestral_nivel1(arvore):
    assert arvore.ancestral_nivel1(5527) == 5524
    assert arvore.ancestral_nivel1(5524) == 5524
    with pytest.raises(ValueError):
        arvore.ancestral_nivel1(5465)


def test_mesmo_ramo(arvore):
    assert arvore.mesmo_ramo(5527, 5528)
    assert arvore.mesmo_ramo(5527, 5524)
    assert not arvore.mesmo_ramo(5524, 5543)
    assert not arvore.mesmo_ramo(5543, 5590)


def test_acerto_segue_a_regua_da_vicio(arvore):
    assert arvore.acerto(5527, 5527)
    assert arvore.acerto(5524, 5527)
    assert arvore.acerto(5527, 5524)
    assert not arvore.acerto(5527, 5528)
    assert not arvore.acerto(5524, 5543)


def test_publicavel_e_ativo_fora_das_raizes_com_duplicados(arvore):
    assert arvore.publicavel(4704)
    assert arvore.publicavel(6724)
    assert arvore.publicavel(6734)
    assert not arvore.publicavel(5062)  # raiz
    assert not arvore.publicavel(5078)  # inativo
    assert not arvore.publicavel(1)  # inexistente


def test_candidatos_possiveis_sao_os_371(arvore):
    possiveis = arvore.candidatos_possiveis()
    assert len(possiveis) == 377 - 4 - 2 == 371
    assert possiveis == sorted(possiveis)
    assert DUPLICADOS_ESPERADOS <= set(possiveis)
    assert 6734 in possiveis
    assert not {5078, 5234, 4365, 4683, 5062, 5465} & set(possiveis)


def test_texto_tema_sem_enriquecido_tira_a_contagem(arvore):
    texto = arvore.texto_tema(5159, {})
    assert texto.startswith("Penal > Dos Crimes Contra o Patrimônio > Furto\n")
    assert "Subtema de 'Dos Crimes Contra o Patrimônio' (Penal)." in texto
    assert "questões" not in texto and "no source" not in texto
    assert "Path:" not in texto


def test_texto_tema_sem_enriquecido_de_duplicado_usa_a_descricao_do_original(arvore):
    texto = arvore.texto_tema(6724, {})
    original = arvore.texto_tema(4704, {})
    assert texto.startswith("Penal > Lei Maria da Penha\n")
    assert texto.split("\n", 1)[1] == original.split("\n", 1)[1]


def test_texto_tema_sem_descricao_nenhuma_e_so_o_caminho():
    import pandas as pd

    temas = pd.DataFrame({
        "tema_id": [1, 2], "parent_id": pd.array([None, 1], dtype="Int64"), "disciplina": ["Penal", "Penal"],
        "nivel": [0, 1], "titulo": ["Penal", "Tema"], "caminho": ["Penal", "Penal > Tema"],
        "descricao": ["Disciplina (root). 10 questões.", "3 questões no source."], "status": ["active", "active"],
        "ativo": [True, True],
    })
    assert ArvoreTemas(temas).texto_tema(2, {}) == "Penal > Tema"


def test_texto_tema_com_enriquecido(arvore):
    assert arvore.texto_tema(4704, {4704: ENRIQUECIDO_MARIA_DA_PENHA}) == TEXTO_MARIA_DA_PENHA


def test_texto_tema_de_duplicado_usa_o_enriquecido_do_original_com_o_proprio_caminho(arvore):
    texto = arvore.texto_tema(6724, {4704: ENRIQUECIDO_MARIA_DA_PENHA})
    assert texto == TEXTO_MARIA_DA_PENHA.replace(
        "Direito Penal - Legislação Especial > Lei Maria da Penha", "Penal > Lei Maria da Penha", 1)


def test_texto_tema_de_duplicado_sem_original_enriquecido_usa_o_proprio(arvore):
    proprio = {**ENRIQUECIDO_MARIA_DA_PENHA, "tema_id": 6724}
    texto = arvore.texto_tema(6724, {6724: proprio})
    assert texto.startswith("Penal > Lei Maria da Penha\nViolência doméstica")


def test_texto_tema_omite_partes_vazias_e_aceita_listas_de_texto(arvore):
    enriquecido = {"tema_id": 4368, "resumo": "  ", "leis": [], "artigos": ["art. 59 do CP"],
                   "termos": ["seletividade"], "sinonimos": []}
    assert arvore.texto_tema(4368, {4368: enriquecido}) == (
        "Criminologia > Criminologia no Estado Democrático de Direito\n"
        "Artigos: art. 59 do CP\n"
        "Termos: seletividade"
    )


def test_texto_tema_lei_sem_nome_ou_sem_numero(arvore):
    enriquecido = {"tema_id": 4704, "resumo": "", "leis": [{"numero": "11.340/2006", "nome": ""},
                                                           {"numero": "", "nome": "Estatuto do Idoso"}]}
    assert arvore.texto_tema(4704, {4704: enriquecido}).endswith("Leis: 11.340/2006; Estatuto do Idoso")
