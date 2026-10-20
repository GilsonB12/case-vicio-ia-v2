import json

import pandas as pd

from classificador import dados
from classificador.config import CAMINHOS


def test_limpar_html_remove_tags_e_normaliza_espacos():
    html = "Texto <b>em negrito</b><br><br>linha &nbsp; nova  <i>it</i>"
    assert dados.limpar_html(html) == "Texto em negrito linha nova it"


def test_limpar_html_troca_barra_n_literal_por_espaco():
    assert dados.limpar_html("Ministério\\nPúblico") == "Ministério Público"


def test_limpar_html_troca_barra_t_literal_por_espaco():
    assert dados.limpar_html("15\\tdias") == "15 dias"


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


def test_carregar_questoes_le_as_359_com_texto():
    df = dados.carregar_questoes()
    assert len(df) == 359
    assert df["id"].is_unique
    assert set(["id", "texto", "alternativas", "disciplina_origem", "assunto_origem"]) <= set(df.columns)
    generica = df.set_index("id").loc["Q0090", "texto"]
    assert "Alternativas:" in generica and "A)" in generica


def test_carregar_temas_tipos_e_ativos():
    df = dados.carregar_temas()
    assert len(df) == 377
    assert df["tema_id"].dtype.kind == "i"
    assert df["nivel"].dtype.kind == "i"
    assert df["parent_id"].isna().sum() == 4
    assert (~df["ativo"]).sum() == 2
    assert set(df.loc[~df["ativo"], "tema_id"]) == {5078, 5234}


def test_jsonl_ida_e_volta(tmp_path):
    caminho = tmp_path / "x.jsonl"
    linhas = [{"id": "Q1", "v": [1, 2]}, {"id": "Q2", "v": None}]
    dados.escrever_jsonl(caminho, linhas)
    assert dados.ler_jsonl(caminho) == linhas


def test_carregar_corpus_monta_texto(tmp_path):
    caminho = tmp_path / "corpus.jsonl"
    caminho.write_text(json.dumps({
        "id": "C0001", "tema_id": 4704, "enunciado": "Sobre a Lei Maria da Penha",
        "comando": "", "alternativas": {"C": "Certo", "E": "Errado"},
        "url": "https://exemplo", "site": "exemplo", "confianca_fonte": "alta",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    df = dados.carregar_corpus(caminho)
    assert list(df["id"]) == ["C0001"]
    assert df.loc[0, "texto"].startswith("Sobre a Lei Maria da Penha")
    assert df.loc[0, "tema_id"] == 4704
