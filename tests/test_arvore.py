import pytest

from classificador.arvore import ArvoreTemas, normalizar_titulo

DUPLICADOS_ESPERADOS = {
    6716, 6717, 6718, 6719, 6720, 6721, 6722, 6723, 6724, 6725,
    6726, 6727, 6728, 6732, 6733, 6734, 6735, 6736, 6737, 6738,
}


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


def test_normalizar_titulo_tira_acento_parte_e_caixa():
    assert normalizar_titulo("Dos Crimes Contra a Administração Pública (Parte 1)") == "dos crimes contra a administracao publica"
    assert normalizar_titulo("Teoria da Pena (Parte 2)") == "teoria da pena"


def test_duplicados_sao_exatamente_os_vinte(arvore):
    mapa = arvore.duplicados()
    assert set(mapa) == DUPLICADOS_ESPERADOS
    for origem, destino in mapa.items():
        assert not arvore.eh_duplicado(destino)
        assert arvore.ativo(destino)
        assert not arvore.eh_raiz(destino)


def test_canonicos_conhecidos(arvore):
    assert arvore.canonico(6724) == 4704
    assert arvore.canonico(6726) == 5199
    assert arvore.canonico(6727) == 5199
    assert arvore.canonico(6732) == 5480
    assert arvore.canonico(6735) == 5543
    assert arvore.canonico(6736) == 5560
    assert arvore.canonico(6722) == 4696
    assert arvore.canonico(6734) == 4703
    assert arvore.canonico(5159) == 5159


def test_navegacao_basica(arvore):
    assert arvore.pai(5527) == 5524
    assert 5527 in arvore.filhos(5524)
    assert 5528 in arvore.irmaos(5527)
    assert 5527 not in arvore.irmaos(5527)
    assert arvore.pai(5062) is None
    assert arvore.eh_raiz(5465)
    assert arvore.nivel(5527) == 2
    assert arvore.disciplina(5527) == "Processo Penal"


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


def test_publicavel_e_candidatos_possiveis(arvore):
    assert arvore.publicavel(4704)
    assert not arvore.publicavel(6724)
    assert not arvore.publicavel(5062)
    assert not arvore.publicavel(5078)
    possiveis = arvore.candidatos_possiveis()
    assert len(possiveis) == 377 - 4 - 2 - 20
    assert possiveis == sorted(possiveis)


def test_documento_sem_ruido_de_contagem(arvore):
    doc = arvore.documento(5159)
    assert doc.startswith("Penal > Dos Crimes Contra o Patrimônio > Furto")
    assert "no source" not in doc
    assert "Path:" not in doc
    assert arvore.documento(6724) == "Penal > Lei Maria da Penha"
