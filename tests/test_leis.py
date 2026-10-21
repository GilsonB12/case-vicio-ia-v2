"""Extração de leis, artigos, diplomas e súmulas, com frases reais de anexos/questoes.jsonl."""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from sistema.leis import (
    CHAVES_FORA,
    DIPLOMAS,
    extrair_chaves,
    normalizar_lei,
    peso_fora,
    sufixo_artigo,
)

QUESTOES = Path(__file__).resolve().parents[1] / "anexos" / "questoes.jsonl"


def _limpar(texto_html: str | None) -> str:
    """Mesma limpeza do sistema: sem tags, sem entidades, sem os '\\n' literais do banco."""
    if not texto_html:
        return ""
    texto = BeautifulSoup(texto_html, "html.parser").get_text(" ")
    texto = html.unescape(texto).replace("\xa0", " ").replace("\\n", " ").replace("\\t", " ")
    return re.sub(r"\s+", " ", texto).strip()


@lru_cache(maxsize=1)
def _textos() -> dict[str, str]:
    textos = {}
    with open(QUESTOES, encoding="utf-8") as arquivo:
        for linha in arquivo:
            if not linha.strip():
                continue
            questao = json.loads(linha)
            partes = [_limpar(questao["enunciado_html"]), _limpar(questao.get("comando"))]
            partes += [_limpar(valor) for valor in (questao.get("alternativas") or {}).values()]
            textos[questao["id"]] = "\n".join(partes)
    return textos


def texto(id_questao: str) -> str:
    return _textos()[id_questao]


def frase(id_questao: str, trecho: str) -> str:
    """Devolve o trecho depois de conferir que ele está, literalmente, no texto da questão."""
    assert trecho in texto(id_questao), f"{trecho!r} não está em {id_questao}"
    return trecho


def so(chaves: Counter, prefixo: str) -> Counter:
    return Counter({chave: n for chave, n in chaves.items() if chave.startswith(prefixo)})


# --- leis por número --------------------------------------------------------------------------

def test_lei_com_n_e_ano_curto():
    chaves = extrair_chaves(frase("Q0063", "Segundo a Lei nº 11.340/06, as medidas protetivas"))
    assert chaves == Counter({"lei:11340": 1})


def test_lei_8072_com_grau_e_ano_curto():
    chaves = extrair_chaves(frase("Q0560", "Lei n° 8.072/90 (Crimes Hediondos)."))
    assert chaves["lei:8072"] == 1


def test_variantes_de_numero_de_lei():
    assert extrair_chaves(frase("Q0484", "Acerca da Lei Federal n.º 9613/98 (lavagem de dinheiro)")) == Counter(
        {"lei:9613": 1}
    )
    assert extrair_chaves(frase("Q0168", "(art. 165, da Lei n o 14.597/2023)"))["lei:14597"] == 1
    assert extrair_chaves(frase("Q0261", "art. 12 da Lei n°9.609"))["lei:9609"] == 1
    assert extrair_chaves("Lei 8137/90 e Lei nº. 8.137/1990")["lei:8137"] == 2
    assert extrair_chaves("a lei nº 13.964/19")["lei:13964"] == 1


def test_lista_de_leis_no_plural():
    chaves = extrair_chaves(frase("Q0445", "nas Leis n.º 12.850/2013, n.º 13.869/2019 e n.º 9.613/1998"))
    assert so(chaves, "lei:") == Counter({"lei:12850": 1, "lei:13869": 1, "lei:9613": 1})


def test_decreto_lei():
    assert extrair_chaves("Decreto-Lei nº 2.848/1940") == Counter({"dl:2848": 1})
    assert extrair_chaves("Decreto-Lei nº 201/67")["dl:201"] == 1


def test_numero_de_decreto_lei_escrito_como_lei_vira_dl():
    assert extrair_chaves("o Código Penal (Lei n.° 2.848/1940)")["dl:2848"] == 1


def test_normalizar_lei():
    assert normalizar_lei("Lei nº 8.072/90") == "8072"
    assert normalizar_lei("11.340/2006") == "11340"
    assert normalizar_lei("n.º 9.613/1998") == "9613"
    assert normalizar_lei(7210) == "7210"
    with pytest.raises(ValueError):
        normalizar_lei("sem número")


# --- diplomas por nome e sigla ------------------------------------------------------------------

def test_codigo_penal_militar_nao_vira_codigo_penal():
    chaves = extrair_chaves(frase("Q0170", "corrupção passiva prevista no Código Penal Militar"))
    assert chaves == Counter({"diploma:cpm": 1})


def test_sigla_cpm():
    assert extrair_chaves(frase("Q0599", "No CPM, há crimes em que se procede somente mediante representação."))[
        "diploma:cpm"
    ] == 1


def test_eca_por_sigla_e_por_nome():
    chaves = extrair_chaves(frase("Q0574", "art. 244-B do ECA depende da prova"))
    assert chaves == Counter({"art:244-b:eca": 1, "diploma:eca": 1, "lei:8069": 1})
    por_nome = extrair_chaves(frase("Q0227", "artigo 240 do Estatuto da Criança e do Adolescente"))
    assert por_nome["art:240:eca"] == 1 and por_nome["diploma:eca"] == 1


def test_lei_de_execucao_penal_aponta_para_lep_e_7210():
    chaves = extrair_chaves(frase("Q0087", "art. 117 da Lei de Execução Penal."))
    assert chaves == Counter({"art:117:lep": 1, "diploma:lep": 1, "lei:7210": 1})


def test_lei_de_abuso_de_autoridade_aponta_para_as_duas_leis():
    chaves = extrair_chaves(frase("Q0657", "Lei de Abuso de Autoridade"))
    assert chaves == Counter({"lei:13869": 1, "lei:4898": 1})


def test_lei_maria_da_penha_por_numero_e_por_nome():
    chaves = extrair_chaves(frase("Q0036", "Art. 16 da Lei nº 11.340/2006 (Lei Maria da Penha)"))
    assert chaves == Counter({"art:16:lei:11340": 1, "lei:11340": 2})


def test_constituicao_estadual_e_constituicao_de_milicia_nao_sao_cf():
    assert "diploma:cf" not in extrair_chaves(frase("Q0537", "fixado na Constituição Estadual."))
    assert extrair_chaves(frase("Q0054", "constituição de milícia privada e autoaborto.")) == Counter()


def test_cf_por_nome_e_sigla():
    assert extrair_chaves(frase("Q0225", "de acordo com a Constituição Federal de 1988"))["diploma:cf"] == 1
    assert extrair_chaves(frase("Q0016", "Art. 5º, caput, da CRFB/88"))["art:5:cf"] == 1


@pytest.mark.parametrize(
    ("apelido", "esperadas"),
    [
        ("Código Penal", {"diploma:cp"}),
        ("CP", {"diploma:cp"}),
        ("Código de Processo Penal", {"diploma:cpp"}),
        ("CPP", {"diploma:cpp"}),
        ("Código Penal Militar", {"diploma:cpm"}),
        ("CPPM", {"diploma:cppm"}),
        ("Código de Processo Penal Militar", {"diploma:cppm"}),
        ("Constituição Federal", {"diploma:cf"}),
        ("CF", {"diploma:cf"}),
        ("CRFB", {"diploma:cf"}),
        ("CTN", {"diploma:ctn"}),
        ("CLT", {"diploma:clt"}),
        ("CDC", {"diploma:cdc", "lei:8078"}),
        ("ECA", {"diploma:eca", "lei:8069"}),
        ("LEP", {"diploma:lep", "lei:7210"}),
        ("Lei de Execução Penal", {"diploma:lep", "lei:7210"}),
        ("Lei de Drogas", {"lei:11343"}),
        ("Lei das Contravenções Penais", {"dl:3688"}),
        ("Estatuto do Desarmamento", {"lei:10826"}),
        ("Lei Maria da Penha", {"lei:11340"}),
        ("Lei de Lavagem de Dinheiro", {"lei:9613"}),
        ("Lei de Abuso de Autoridade", {"lei:13869", "lei:4898"}),
        ("Lei dos Crimes Hediondos", {"lei:8072"}),
        ("Lei de Organizações Criminosas", {"lei:12850"}),
        ("Juizados Especiais Criminais", {"lei:9099"}),
        ("Estatuto do Idoso", {"lei:10741"}),
        ("Lei de Tortura", {"lei:9455"}),
        ("Lei de Crimes Ambientais", {"lei:9605"}),
        ("Código de Trânsito Brasileiro", {"lei:9503"}),
        ("Lei de Interceptação Telefônica", {"lei:9296"}),
        ("Lei de Racismo", {"lei:7716"}),
        ("Lei Anticrime", {"lei:13964"}),
        ("Código Eleitoral", {"lei:4737"}),
        ("Lei Antiterrorismo", {"lei:13260"}),
    ],
)
def test_apelidos(apelido, esperadas):
    assert set(extrair_chaves(f"Conforme o {apelido}, julgue o item.")) == esperadas
    assert set(DIPLOMAS[apelido]) == esperadas


def test_apelido_sem_acento_e_em_minusculas():
    assert extrair_chaves(frase("Q0161", "art. 33 da Lei de Drogas"))["art:33:lei:11343"] == 1
    assert extrair_chaves("segundo a lei de execucao penal")["lei:7210"] == 1


def test_juizado_especial_federal_nao_e_9099():
    assert "lei:9099" not in extrair_chaves("nos Juizados Especiais Federais")


# --- artigos ----------------------------------------------------------------------------------

def test_artigo_com_paragrafo_e_cp():
    assert extrair_chaves("art. 157, § 2º, do CP") == Counter({"art:157:cp": 1, "diploma:cp": 1})
    real = extrair_chaves(frase("Q0171", "(Art. 157, §2º-A, I, do Código Penal)"))
    assert real == Counter({"art:157:cp": 1, "diploma:cp": 1})


def test_paragrafo_inciso_e_parenteses_nao_quebram():
    chaves = extrair_chaves(
        frase("Q0208", "art. 121, § 2.º, inciso II (homicídio qualificado por motivo fútil), do CP")
    )
    assert chaves == Counter({"art:121:cp": 1, "diploma:cp": 1})


def test_caput_e_combinado_com():
    chaves = extrair_chaves(frase("Q0336", "art. 155, caput, c/c art. 14, II, ambos do CP"))
    assert chaves == Counter({"art:155:cp": 1, "art:14:cp": 1, "diploma:cp": 1})


def test_intervalo_de_artigos_gera_so_as_pontas():
    chaves = extrair_chaves("Nos termos dos arts. 240 a 245 do CPP, a busca")
    assert chaves == Counter({"art:240:cpp": 1, "art:245:cpp": 1, "diploma:cpp": 1})
    real = extrair_chaves(frase("Q0536", "arts. 158-A a 158-F do Código de Processo Penal"))
    assert real == Counter({"art:158-a:cpp": 1, "art:158-f:cpp": 1, "diploma:cpp": 1})


def test_lista_de_artigos_com_caput():
    chaves = extrair_chaves(frase("Q0127", "Arts. 33, caput , e 35, ambos da Lei nº 11.343/2006"))
    assert so(chaves, "art:") == Counter({"art:33:lei:11343": 1, "art:35:lei:11343": 1})


def test_artigo_com_letra_separada_por_espacos():
    chaves = extrair_chaves(frase("Q0623", "artigos 158 - A e seguintes do CPP"))
    assert so(chaves, "art:") == Counter({"art:158-a:cpp": 1})


def test_travessao_de_artigo_transcrito_nao_vira_letra():
    chaves = extrair_chaves(frase("Q0499", "“Art. 13 - O resultado, de que depende a existência do crime"))
    assert chaves == Counter({"art:13": 1})


def test_ordinal_e_letra_depois_do_ordinal():
    assert extrair_chaves(frase("Q0016", "§ 1º do Art. 3º-B do CPP"))["art:3-b:cpp"] == 1
    assert extrair_chaves(frase("Q0020", "(artigo 5 o , inciso LXIII, CF)"))["art:5:cf"] == 1


def test_diploma_antes_do_artigo():
    chaves = extrair_chaves(frase("Q0065", "O crime de extorsão mediante sequestro (CP, art. 159)"))
    assert chaves == Counter({"art:159:cp": 1, "diploma:cp": 1})


def test_artigo_do_decreto_lei_usa_a_sigla_do_codigo():
    chaves = extrair_chaves(
        frase("Q0484", "art. 366 do Decreto-Lei nº 3.689, de 3 de outubro de 1941 (Código de Processo Penal)")
    )
    assert chaves == Counter({"art:366:cpp": 1, "dl:3689": 1, "diploma:cpp": 1})


def test_artigo_de_lei_especial_por_apelido():
    chaves = extrair_chaves(frase("Q0282", "o art. 14 do Estatuto do Desarmamento"))
    assert chaves == Counter({"art:14:lei:10826": 1, "lei:10826": 1})


def test_artigo_sem_diploma_usa_o_dominante_do_texto():
    # Q0036 cita a Lei 11.340 três vezes (número e duas vezes o nome) e o Código Penal uma vez;
    # as três menções a "audiência do Art. 16" sem diploma por perto ficam com a Lei Maria da Penha.
    chaves = extrair_chaves(texto("Q0036"))
    assert chaves["lei:11340"] == 3
    assert chaves["diploma:cp"] == 1
    assert chaves["art:129:cp"] == 1
    assert chaves["art:16:lei:11340"] == 5
    assert "art:16" not in chaves and "art:16:cp" not in chaves


def test_artigo_da_mesma_lei_usa_a_mencao_anterior():
    chaves = extrair_chaves(texto("Q0127"))
    assert chaves["art:35:lei:11343"] == 2
    assert chaves["art:69:cp"] == 1
    assert chaves["art:3-a:cpp"] == 1


def test_artigo_sem_diploma_nenhum_fica_sem_diploma():
    assert extrair_chaves("O art. 157 trata do roubo.") == Counter({"art:157": 1})


def test_diploma_padrao_quando_o_texto_nao_tem_nenhum():
    assert extrair_chaves("art. 33", diploma_padrao="lei:11343") == Counter({"art:33:lei:11343": 1})
    assert extrair_chaves("art. 33 do CP", diploma_padrao="lei:11343")["art:33:cp"] == 1


def test_sufixo_artigo():
    assert sufixo_artigo("diploma:cp") == "cp"
    assert sufixo_artigo("dl:2848") == "cp"
    assert sufixo_artigo("lei:7210") == "lep"
    assert sufixo_artigo("lei:11340") == "lei:11340"


# --- súmulas ----------------------------------------------------------------------------------

def test_sumula_vinculante():
    assert extrair_chaves("Súmula Vinculante 11") == Counter({"sumula:v:11": 1})
    real = extrair_chaves(frase("Q0557", "Dispõe a Súmula Vinculante 35 do STF"))
    assert real == Counter({"sumula:v:35": 1})


def test_sumula_com_tribunal():
    assert extrair_chaves(frase("Q0005", "incide no caso a súmula n.º 269 do STJ.")) == Counter({"sumula:stj:269": 1})
    assert extrair_chaves("Súmula 11 do Supremo Tribunal Federal") == Counter({"sumula:stf:11": 1})
    assert extrair_chaves("Súmula 7/STJ") == Counter({"sumula:stj:7": 1})


def test_sumula_sem_tribunal():
    assert extrair_chaves(frase("Q0225", "no enunciado da sua Súmula n.º 7.")) == Counter({"sumula:7": 1})


# --- vazios -----------------------------------------------------------------------------------

def test_texto_vazio_ou_sem_lei():
    assert extrair_chaves("") == Counter()
    assert extrair_chaves(None) == Counter()
    assert extrair_chaves(texto("Q0002")) == Counter()  # criminologia, sem lei nenhuma


def test_todas_as_questoes_rodam_e_as_chaves_tem_formato_valido():
    formato = re.compile(
        r"^(lei:\d+|dl:\d+|diploma:[a-z]+|art:\d+(-[a-z])?(:([a-z]+|lei:\d+|dl:\d+))?|sumula:((v|stf|stj|tst|tse|stm):)?\d+)$"
    )
    com_chave = 0
    for texto_questao in _textos().values():
        chaves = extrair_chaves(texto_questao)
        com_chave += bool(chaves)
        for chave, n in chaves.items():
            assert formato.match(chave), chave
            assert n >= 1
    assert com_chave > 200


# --- chaves fora --------------------------------------------------------------------------------

def test_chaves_fora():
    assert CHAVES_FORA == {"diploma:cpm", "diploma:cppm", "diploma:ctn", "diploma:clt", "diploma:cf"}


def test_peso_fora_conta_cpm_ctn_clt():
    assert peso_fora(extrair_chaves(frase("Q0416", "o CPM foi alterado para aumentar a pena"))) == 1.0
    assert peso_fora(Counter({"diploma:ctn": 2, "diploma:clt": 1})) == 3.0
    assert peso_fora(Counter({"diploma:cp": 3})) == 0.0


def test_cf_so_conta_como_fora_quando_e_o_unico_diploma():
    assert peso_fora(extrair_chaves("Nos termos do art. 5º da Constituição Federal")) == 1.0
    assert peso_fora(extrair_chaves(texto("Q0127"))) == 0.0  # CF com CP, CPP e Lei 11.343
    assert peso_fora(Counter({"diploma:cf": 1, "diploma:cpm": 1})) == 1.0  # só o CPM conta


def test_peso_fora_aceita_outro_conjunto_de_chaves():
    assert peso_fora(Counter({"lei:9605": 2, "diploma:cp": 1}), fora={"lei:9605"}) == 2.0
