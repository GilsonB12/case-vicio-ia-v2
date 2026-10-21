import json

import pytest

from sistema import dados
from sistema.arvore import ArvoreTemas

COLUNAS_QUESTOES = ["id", "texto", "enunciado_html", "comando", "alternativas", "disciplina_origem", "assunto_origem"]


@pytest.fixture(scope="module")
def arvore():
    return ArvoreTemas.carregar()


@pytest.fixture(scope="module")
def folha():
    return dados.carregar_folha()


@pytest.fixture(scope="module")
def questoes():
    return dados.carregar_questoes()


def linha_da_folha(folha, id_questao):
    return folha.set_index("id").loc[id_questao]


# --- limpeza e texto -------------------------------------------------------------------------------------


def test_limpar_html_remove_tags_e_normaliza_espacos():
    html = "Texto <b>em negrito</b><br><br>linha &nbsp; nova  <i>it</i>"
    assert dados.limpar_html(html) == "Texto em negrito linha nova it"


def test_limpar_html_troca_barra_n_literal_por_espaco():
    assert dados.limpar_html("Ministério\\nPúblico") == "Ministério Público"


def test_limpar_html_troca_barra_t_literal_por_espaco():
    assert dados.limpar_html("15\\tdias") == "15 dias"


def test_limpar_html_vazio():
    assert dados.limpar_html("") == ""
    assert dados.limpar_html(None) == ""


def test_montar_texto_inclui_comando_e_alternativas():
    texto = dados.montar_texto(
        "Assinale a alternativa correta.",
        "Sobre furto.",
        {"A": " furto simples ", "B": "roubo"},
    )
    assert texto == (
        "Assinale a alternativa correta.\n"
        "Comando: Sobre furto.\n"
        "Alternativas:\n"
        "A) furto simples\n"
        "B) roubo"
    )


def test_montar_texto_sem_comando_omite_linha_de_comando():
    texto = dados.montar_texto("Enunciado", "", {"C": "Certo", "E": "Errado"})
    assert texto == "Enunciado\nAlternativas:\nC) Certo\nE) Errado"


# --- questões e temas ------------------------------------------------------------------------------------


def test_carregar_questoes_le_as_659_com_texto(questoes):
    assert len(questoes) == 659
    assert list(questoes.columns) == COLUNAS_QUESTOES
    assert questoes["id"].is_unique
    assert list(questoes["id"]) == [f"Q{numero:04d}" for numero in range(1, 660)]
    generica = questoes.set_index("id").loc["Q0090", "texto"]
    assert "Alternativas:" in generica and "A)" in generica
    assert all(isinstance(texto, str) and texto for texto in questoes["texto"])


def test_texto_da_questao_nao_leva_disciplina_nem_assunto_de_origem(questoes):
    primeira = questoes.set_index("id").loc["Q0001"]
    esperado = dados.montar_texto(primeira["enunciado_html"], primeira["comando"], primeira["alternativas"])
    assert primeira["texto"] == esperado
    assert primeira["assunto_origem"] == "Procedimento do Júri"
    assert "Procedimento do Júri" not in primeira["texto"]


def test_carregar_temas_tipos_e_ativos():
    df = dados.carregar_temas()
    assert len(df) == 377
    assert df["tema_id"].dtype.kind == "i"
    assert df["nivel"].dtype.kind == "i"
    assert df["parent_id"].isna().sum() == 4
    assert (~df["ativo"]).sum() == 2
    assert set(df.loc[~df["ativo"], "tema_id"]) == {5078, 5234}


def test_jsonl_ida_e_volta(tmp_path):
    caminho = tmp_path / "sub" / "x.jsonl"
    linhas = [{"id": "Q1", "v": [1, 2]}, {"id": "Q2", "v": None, "t": "ação"}]
    dados.escrever_jsonl(caminho, linhas)
    assert dados.ler_jsonl(caminho) == linhas
    assert "ação" in caminho.read_text(encoding="utf-8")


# --- folha-verdade e régua -------------------------------------------------------------------------------


def test_carregar_folha_tem_100_linhas_e_8_fora(folha):
    assert len(folha) == 100
    assert list(folha.columns) == ["id", "tema_central_id", "extras_ids", "fora"]
    assert folha["id"].is_unique
    assert folha["fora"].dtype == bool
    assert int(folha["fora"].sum()) == 8
    fora = folha[folha["fora"]]
    assert all(central is None for central in fora["tema_central_id"])
    assert all(extras == [] for extras in fora["extras_ids"])
    assert all(type(central) is int for central in folha.loc[~folha["fora"], "tema_central_id"])


def test_carregar_folha_converte_central_e_extras_em_inteiros(folha):
    linha = linha_da_folha(folha, "Q0012")
    assert linha["tema_central_id"] == 5562
    assert type(linha["tema_central_id"]) is int
    assert linha["extras_ids"] == [4701, 5569]
    assert all(type(extra) is int for extra in linha["extras_ids"])
    assert not linha["fora"]
    assert linha_da_folha(folha, "Q0005")["extras_ids"] == []


def test_folha_esta_toda_entre_as_359_do_banco(folha):
    assert all(int(id_questao[1:]) <= 359 for id_questao in folha["id"])


def test_carregar_folha_recusa_linha_incoerente(tmp_path):
    caminho = tmp_path / "folha.csv"
    caminho.write_text(
        "id,tema_central_id,tema_central_caminho,extras_ids,fora_da_taxonomia\nQ0001,5524,x,,1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Q0001"):
        dados.carregar_folha(caminho)


def test_alvos_aceitos_inclui_central_extras_e_pai_sem_raiz(folha, arvore):
    # Q0012: central 5562 (Apelação, filho de 5560), extras 4701 (nível 1, sem filhos) e 5569 (filho de 5560).
    # O pai de 4701 é a raiz da Legislação Especial, que não é publicável e fica de fora.
    assert dados.alvos_aceitos(linha_da_folha(folha, "Q0012"), arvore) == {5562, 5560, 4701, 5569}


def test_alvos_aceitos_inclui_os_filhos_do_central(folha, arvore):
    # Q0003: central 5560 (Recursos em Espécies, nível 1), extra 5562 (um dos seus filhos).
    alvos = dados.alvos_aceitos(linha_da_folha(folha, "Q0003"), arvore)
    assert alvos == {5560, *arvore.filhos(5560)}
    assert len(alvos) == 12
    assert 5465 not in alvos


def test_alvos_aceitos_de_questao_fora_e_vazio(folha, arvore):
    assert dados.alvos_aceitos(linha_da_folha(folha, "Q0002"), arvore) == set()


def test_alvos_aceitos_aceita_dicionario(arvore):
    linha = {"id": "X", "tema_central_id": 5527, "extras_ids": [], "fora": False}
    assert dados.alvos_aceitos(linha, arvore) == {5527, 5524}


def test_acerto_regua(folha, arvore):
    linha = linha_da_folha(folha, "Q0012")
    assert dados.acerto_regua(5562, linha, arvore)
    assert dados.acerto_regua(5560, linha, arvore)  # pai do central
    assert dados.acerto_regua(4701, linha, arvore)  # extra
    assert dados.acerto_regua(5569, linha, arvore)  # extra
    assert not dados.acerto_regua(5563, linha, arvore)  # irmão do central
    assert not dados.acerto_regua(4683, linha, arvore)  # raiz
    assert not dados.acerto_regua(None, linha, arvore)


def test_acerto_regua_e_falso_em_questao_fora(folha, arvore):
    linha = linha_da_folha(folha, "Q0002")
    assert not dados.acerto_regua(4368, linha, arvore)
    assert not dados.acerto_regua(None, linha, arvore)


# --- temas enriquecidos, exemplos e corpus ---------------------------------------------------------------


def test_temas_enriquecidos_ausentes_devolvem_dicionario_vazio(tmp_path):
    assert dados.carregar_temas_enriquecidos(tmp_path / "nao_existe.jsonl") == {}


def test_temas_enriquecidos_por_tema_id(tmp_path):
    caminho = tmp_path / "temas_enriquecidos.jsonl"
    dados.escrever_jsonl(caminho, [
        {"tema_id": 4704, "resumo": "Violência doméstica.", "leis": [], "artigos": [], "termos": ["medida protetiva"]},
        {"tema_id": "5159", "resumo": "Furto.", "leis": [], "artigos": [], "termos": []},
    ])
    enriquecidos = dados.carregar_temas_enriquecidos(caminho)
    assert set(enriquecidos) == {4704, 5159}
    assert enriquecidos[4704]["termos"] == ["medida protetiva"]
    assert enriquecidos[5159]["tema_id"] == 5159


def test_temas_enriquecidos_recusa_tema_repetido(tmp_path):
    caminho = tmp_path / "temas_enriquecidos.jsonl"
    dados.escrever_jsonl(caminho, [{"tema_id": 4704, "resumo": "a"}, {"tema_id": 4704, "resumo": "b"}])
    with pytest.raises(ValueError, match="4704"):
        dados.carregar_temas_enriquecidos(caminho)


def test_exemplos_ausentes_devolvem_lista_vazia(tmp_path):
    assert dados.carregar_exemplos(tmp_path / "nao_existe.jsonl") == []


def test_exemplos_rotulados_e_fora(tmp_path):
    caminho = tmp_path / "exemplos.jsonl"
    fonte = {"modelo": "x", "data": "2026-10-20"}
    dados.escrever_jsonl(caminho, [
        {"id": "S0001", "tema_id": 4704, "fora": False, "texto": "Sobre medidas protetivas.", "fonte": fonte},
        {"id": "S0002", "tema_id": None, "fora": True, "texto": "Sobre ICMS.", "fonte": fonte, "extra": 1},
        {"id": "S0003", "tema_id": "5159", "texto": "Sobre furto."},
    ])
    assert dados.carregar_exemplos(caminho) == [
        {"id": "S0001", "tema_id": 4704, "fora": False, "texto": "Sobre medidas protetivas.", "fonte": fonte},
        {"id": "S0002", "tema_id": None, "fora": True, "texto": "Sobre ICMS.", "fonte": fonte},
        {"id": "S0003", "tema_id": 5159, "fora": False, "texto": "Sobre furto.", "fonte": None},
    ]


def test_exemplos_recusam_fora_com_tema(tmp_path):
    caminho = tmp_path / "exemplos.jsonl"
    dados.escrever_jsonl(caminho, [{"id": "S0001", "tema_id": 4704, "fora": True, "texto": "x"}])
    with pytest.raises(ValueError, match="S0001"):
        dados.carregar_exemplos(caminho)


def test_exemplos_recusam_id_repetido(tmp_path):
    caminho = tmp_path / "exemplos.jsonl"
    dados.escrever_jsonl(caminho, [{"id": "S0001", "tema_id": 4704, "texto": "x"},
                                   {"id": "S0001", "tema_id": 5159, "texto": "y"}])
    with pytest.raises(ValueError, match="S0001"):
        dados.carregar_exemplos(caminho)


def test_carregar_corpus_v1_monta_texto_como_nas_questoes(tmp_path):
    caminho = tmp_path / "corpus.jsonl"
    caminho.write_text(json.dumps({
        "id": "C0001", "tema_id": 4704, "enunciado": "Sobre a <b>Lei Maria da Penha</b>",
        "comando": "Julgue.", "alternativas": {"C": "Certo", "E": "Errado"},
        "url": "https://exemplo", "site": "exemplo", "confianca_fonte": "alta",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    assert dados.carregar_corpus_v1(caminho) == [{
        "id": "C0001", "tema_id": 4704,
        "texto": "Sobre a Lei Maria da Penha\nComando: Julgue.\nAlternativas:\nC) Certo\nE) Errado",
    }]


def test_carregar_corpus_v1_real_tem_332_rotuladas():
    corpus = dados.carregar_corpus_v1()
    assert len(corpus) == 332
    assert all(set(item) == {"id", "tema_id", "texto"} for item in corpus)
    assert all(type(item["tema_id"]) is int and item["texto"] for item in corpus)
    assert len({item["id"] for item in corpus}) == 332
