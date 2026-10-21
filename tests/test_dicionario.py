"""Dicionário lei-tema: construção (peso 1/n), ajustes revisados, pontuação e os dados versionados."""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from sistema.dicionario import carregar, construir, ler_ajustes, ler_enriquecidos, main, pontuar, salvar
from sistema.leis import CHAVES_FORA, extrair_chaves

RAIZ = Path(__file__).resolve().parents[1]
AJUSTES = RAIZ / "dados" / "dicionario_leis_ajustes.json"
TERMOS_FORA = RAIZ / "dados" / "termos_fora.json"
TEMAS = RAIZ / "anexos" / "temas.csv"

LAVAGEM = {"numero": "9.613/1998", "nome": "Lei de Lavagem de Dinheiro"}


def _temas_csv() -> dict[int, dict]:
    with open(TEMAS, encoding="utf-8") as arquivo:
        return {int(linha["tema_id"]): linha for linha in csv.DictReader(arquivo)}


# --- construção ---------------------------------------------------------------------------------

def test_peso_e_um_sobre_o_numero_de_temas_que_citam_a_chave():
    enriquecidos = {
        4702: {"leis": [LAVAGEM], "artigos": ["art. 1º da Lei 9.613/1998"]},
        6723: {"leis": [LAVAGEM], "artigos": []},
        5160: {"leis": [], "artigos": ["art. 157 do CP"]},
    }
    dicionario = construir(enriquecidos, {})
    assert dicionario["lei:9613"] == {"nome": "Lei de Lavagem de Dinheiro", "temas": {4702: 0.5, 6723: 0.5}, "fora": False}
    assert dicionario["art:1:lei:9613"]["temas"] == {4702: 1.0}
    assert dicionario["art:157:cp"]["temas"] == {5160: 1.0}
    assert "diploma:cp" not in dicionario  # o artigo gera só a chave de artigo


def test_tres_temas_dividem_o_peso_em_tercos():
    enriquecidos = {tema: {"leis": ["Lei 8.072/1990"], "artigos": []} for tema in (4693, 5200, 5201)}
    assert construir(enriquecidos, {})["lei:8072"]["temas"] == pytest.approx({4693: 1 / 3, 5200: 1 / 3, 5201: 1 / 3})


def test_formatos_de_lei_e_de_artigo_aceitos():
    enriquecidos = {
        4708: {
            "leis": ["11.343/2006", {"numero": "Lei 10.826/2003"}, {"nome": "Estatuto da Criança e do Adolescente"}],
            "artigos": [
                {"artigo": "33", "diploma": "Lei 11.343/2006"},
                {"numero": "244-B", "diploma": "ECA"},
                "art. 35",  # sem diploma: usa a primeira lei do tema
                "40, inciso V",  # sem a palavra "art."
            ],
        },
    }
    dicionario = construir(enriquecidos, {})
    for chave in ("lei:11343", "lei:10826", "diploma:eca", "lei:8069", "art:33:lei:11343", "art:244-b:eca",
                  "art:35:lei:11343", "art:40:lei:11343"):
        assert dicionario[chave]["temas"] == {4708: 1.0}, chave


def test_codigo_por_decreto_lei_gera_as_duas_chaves():
    enriquecidos = {5160: {"leis": [{"numero": "2.848/1940", "nome": "Código Penal"}], "artigos": ["art. 157"]}}
    dicionario = construir(enriquecidos, {})
    assert dicionario["dl:2848"]["temas"] == {5160: 1.0}
    assert dicionario["diploma:cp"]["temas"] == {5160: 1.0}
    assert dicionario["art:157:cp"]["temas"] == {5160: 1.0}


def test_artigo_sem_diploma_em_tema_sem_lei_nao_entra():
    dicionario = construir({5177: {"leis": [], "artigos": ["art. 14"]}}, {})
    assert not any(chave.startswith("art:") for chave in dicionario)


def test_tema_sem_leis_nem_artigos_e_ids_em_texto():
    dicionario = construir({"4702": {"leis": [LAVAGEM]}, 5000: {"resumo": "sem campos de lei"}}, {})
    assert dicionario["lei:9613"]["temas"] == {4702: 1.0}


def test_chaves_fora_entram_por_padrao():
    dicionario = construir({}, {})
    assert set(dicionario) == CHAVES_FORA
    assert all(entrada["fora"] and entrada["temas"] == {} for entrada in dicionario.values())
    assert dicionario["diploma:cpm"]["nome"] == "Código Penal Militar"


def test_cf_citada_por_tema_continua_fora_e_aponta_para_o_tema():
    dicionario = construir({5525: {"leis": ["Constituição Federal"], "artigos": ["art. 5º, XXXVIII, da CF"]}}, {})
    assert dicionario["diploma:cf"] == {"nome": "Constituição Federal", "temas": {5525: 1.0}, "fora": True}
    assert dicionario["art:5:cf"]["fora"] is False


# --- ajustes ------------------------------------------------------------------------------------

def test_ajuste_sobrescreve_a_entrada_inteira():
    enriquecidos = {tema: {"leis": ["Lei 11.340/2006"]} for tema in (4704, 6724, 4709)}
    ajustes = {
        "_formato": "explicação ignorada",
        "lei:11340": {"nome": "Lei Maria da Penha", "temas": {"4704": 1.0, "6724": 0.3}, "fora": False},
    }
    entrada = construir(enriquecidos, ajustes)["lei:11340"]
    assert entrada == {"nome": "Lei Maria da Penha", "temas": {4704: 1.0, 6724: 0.3}, "fora": False}


def test_ajuste_acrescenta_chave_nova_e_zera_temas():
    enriquecidos = {5160: {"leis": ["Código Penal"]}}
    ajustes = {"sumula:v:11": {"temas": {"5100": 1}}, "diploma:cp": {"temas": {}}}
    dicionario = construir(enriquecidos, ajustes)
    assert dicionario["sumula:v:11"] == {"nome": "sumula:v:11", "temas": {5100: 1.0}, "fora": False}
    assert dicionario["diploma:cp"] == {"nome": "Código Penal", "temas": {}, "fora": False}
    assert "_formato" not in dicionario


def test_ajuste_sem_fora_mantem_as_chaves_fora_como_fora():
    dicionario = construir({}, {"diploma:ctn": {"temas": {"4695": 0.2}}, "lei:5172": {"fora": True}})
    assert dicionario["diploma:ctn"]["fora"] is True
    assert dicionario["lei:5172"]["fora"] is True


@pytest.mark.parametrize(
    "entrada",
    [{"temas": [4704]}, {"temas": {"abc": 1.0}}, {"temas": {"4704": "um"}}, {"fora": "sim"}, "lei:11340"],
)
def test_ajuste_malformado_levanta_erro(entrada):
    with pytest.raises(ValueError, match="lei:11340"):
        construir({}, {"lei:11340": entrada})


# --- pontuação ----------------------------------------------------------------------------------

@pytest.fixture
def dicionario_pequeno():
    return construir(
        {},
        {
            "lei:11340": {"temas": {"4704": 1.0, "6724": 0.3}},
            "art:16:lei:11340": {"temas": {"4704": 1.0}},
            "lei:11343": {"temas": {"4708": 1.0}},
            "lei:5172": {"temas": {}, "fora": True},
        },
    )


def test_pontuar_soma_peso_vezes_contagem(dicionario_pequeno):
    pontos, fora = pontuar(Counter({"lei:11340": 2, "art:16:lei:11340": 1, "lei:11343": 1}), dicionario_pequeno)
    assert pontos == pytest.approx({4704: 3.0, 6724: 0.6, 4708: 1.0})
    assert fora == 0.0


def test_pontuar_ignora_chaves_fora_do_dicionario_e_texto_vazio(dicionario_pequeno):
    assert pontuar(Counter({"art:999:cp": 3}), dicionario_pequeno) == ({}, 0.0)
    assert pontuar(Counter(), dicionario_pequeno) == ({}, 0.0)


def test_peso_fora_soma_as_contagens_das_chaves_fora(dicionario_pequeno):
    _, fora = pontuar(Counter({"diploma:cpm": 2, "diploma:clt": 1, "lei:5172": 1, "lei:11343": 1}), dicionario_pequeno)
    assert fora == 4.0


def test_cf_so_conta_como_fora_quando_e_o_unico_diploma(dicionario_pequeno):
    so_cf = extrair_chaves("Nos termos do art. 5º, LVI, da Constituição Federal, são inadmissíveis as provas ilícitas.")
    assert pontuar(so_cf, dicionario_pequeno)[1] == 1.0
    com_lei = extrair_chaves("A Constituição Federal e a Lei nº 11.340/2006 protegem a mulher.")
    pontos, fora = pontuar(com_lei, dicionario_pequeno)
    assert fora == 0.0
    assert pontos == pytest.approx({4704: 1.0, 6724: 0.3})


def test_pontuar_com_frase_real_da_lei_maria_da_penha(dicionario_pequeno):
    chaves = extrair_chaves("audiência prevista no Art. 16 da Lei nº 11.340/2006 (Lei Maria da Penha)")
    pontos, fora = pontuar(chaves, dicionario_pequeno)
    assert pontos == pytest.approx({4704: 3.0, 6724: 0.6})
    assert fora == 0.0


# --- arquivos -----------------------------------------------------------------------------------

def test_salvar_e_carregar_mantem_ids_inteiros(tmp_path):
    dicionario = construir({4702: {"leis": [LAVAGEM]}, 6723: {"leis": [LAVAGEM]}}, {})
    caminho = tmp_path / "dicionario_leis.json"
    salvar(dicionario, caminho, fonte={"gerado_por": "teste"})
    bruto = json.loads(caminho.read_text(encoding="utf-8"))
    assert bruto["_fonte"] == {"gerado_por": "teste"}
    assert bruto["lei:9613"]["temas"] == {"4702": 0.5, "6723": 0.5}
    assert carregar(caminho) == dicionario


def test_ler_enriquecidos_e_ajustes(tmp_path):
    caminho = tmp_path / "temas_enriquecidos.jsonl"
    caminho.write_text(
        json.dumps({"tema_id": 4702, "leis": [LAVAGEM]}, ensure_ascii=False) + "\n\n"
        + json.dumps({"tema_id": "6723", "leis": [LAVAGEM]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert set(ler_enriquecidos(caminho)) == {4702, 6723}
    assert ler_ajustes(tmp_path / "nao_existe.json") == {}


def test_main_sem_temas_enriquecidos_avisa_e_sai_com_1(tmp_path, capsys):
    faltando = tmp_path / "temas_enriquecidos.jsonl"
    codigo = main(["--enriquecidos", str(faltando), "--saida", str(tmp_path / "saida.json")])
    assert codigo == 1
    erro = capsys.readouterr().err
    assert "temas enriquecidos" in erro and str(faltando) in erro
    assert not (tmp_path / "saida.json").exists()


def test_main_grava_o_dicionario(tmp_path, capsys):
    enriquecidos = tmp_path / "temas_enriquecidos.jsonl"
    enriquecidos.write_text(
        "\n".join(json.dumps({"tema_id": t, "leis": ["Lei 11.340/2006"], "artigos": ["art. 16"]}) for t in (4704, 6724)),
        encoding="utf-8",
    )
    ajustes = tmp_path / "ajustes.json"
    ajustes.write_text(json.dumps({"lei:11340": {"temas": {"4704": 1.0}}}), encoding="utf-8")
    saida = tmp_path / "dicionario_leis.json"
    assert main(["--enriquecidos", str(enriquecidos), "--ajustes", str(ajustes), "--saida", str(saida)]) == 0
    dicionario = carregar(saida)
    assert dicionario["lei:11340"]["temas"] == {4704: 1.0}
    assert dicionario["art:16:lei:11340"]["temas"] == {4704: 0.5, 6724: 0.5}
    assert "diploma:cf" in dicionario
    assert str(saida) in capsys.readouterr().out


# --- dados versionados --------------------------------------------------------------------------

def test_ajustes_versionados_tem_formato_e_ids_validos():
    bruto = json.loads(AJUSTES.read_text(encoding="utf-8"))
    assert isinstance(bruto.pop("_formato"), str)
    temas = _temas_csv()
    for chave, entrada in bruto.items():
        assert set(entrada) <= {"nome", "temas", "fora"}, chave
        for tema_id in entrada.get("temas", {}):
            linha = temas[int(tema_id)]
            assert linha["status"] == "active" and linha["nivel"] != "0", (chave, tema_id)
    dicionario = construir({}, bruto)
    assert dicionario["lei:11340"]["temas"] == {4704: 1.0, 6724: 0.3}
    assert dicionario["lei:10826"]["temas"] == {4697: 1.0}
    assert dicionario["lei:11343"]["temas"] == {4708: 1.0}
    assert dicionario["lei:9099"]["temas"] == {4701: 1.0, 5539: 0.5}
    assert dicionario["lei:7210"]["temas"] == {4703: 1.0, 6734: 0.5}
    for generica in ("dl:2848", "diploma:cp", "diploma:cpp"):
        assert dicionario[generica]["temas"] == {} and dicionario[generica]["fora"] is False
    for fora in ("diploma:cpm", "diploma:cppm", "diploma:ctn", "diploma:clt", "diploma:cf"):
        assert dicionario[fora]["fora"] is True


def _dobrar(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if not unicodedata.combining(c))


def test_termos_fora_nao_colidem_com_temas_da_arvore():
    bruto = json.loads(TERMOS_FORA.read_text(encoding="utf-8"))
    assert bruto["_nota"]
    termos = bruto["termos"]
    assert termos == [termo.lower() for termo in termos]
    assert len(termos) == len(set(termos))
    assert {"militar", "código penal militar", "cpm", "cppm", "clt", "inss", "processo civil"} <= set(termos)
    proibidos = {"ambiental", "eleitoral", "ordem tributária", "trânsito", "consumidor", "tributário", "civil"}
    assert not proibidos & set(termos)
    caminhos = " | ".join(_dobrar(linha["caminho"]) for linha in _temas_csv().values())
    for termo in termos:
        base = _dobrar(termo)
        variantes = {base, base + "s", base[:-1] + "a", base[:-1] + "as", base[:-1] + "os", base[:-1] + "is"}
        for variante in variantes:
            assert not re.search(rf"(?<!\w){re.escape(variante)}(?!\w)", caminhos), (termo, variante)
