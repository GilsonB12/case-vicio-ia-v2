import json
import re
from datetime import datetime, timedelta

import pytest

from classificador import custos
from classificador.custos import (
    Contexto,
    LivroRazao,
    calcular_custo,
    contexto,
    contexto_atual,
    formatar_resumo,
    normalizar_uso,
    resumir_custos,
)

USO = {"tokens_entrada": 1000, "tokens_saida": 500, "tokens_raciocinio": 200,
       "tokens_cache_leitura": 0, "tokens_cache_escrita": 0}


def uso(entrada, saida, raciocinio=0):
    return {"tokens_entrada": entrada, "tokens_saida": saida, "tokens_raciocinio": raciocinio,
            "tokens_cache_leitura": 0, "tokens_cache_escrita": 0}


def ler_livro(caminho):
    return [json.loads(linha) for linha in caminho.read_text(encoding="utf-8").splitlines()]


# --- tabela de preços e uso ------------------------------------------------------------------------------


def test_tabela_de_precos_tem_versao_e_os_modelos_do_projeto():
    assert custos.TABELA_PRECOS["versao"] == "2026-10-15"
    assert set(custos.TABELA_PRECOS["modelos"]) == {"claude-sonnet-5-5", "gpt-5", "gpt-5-mini", "text-embedding-3-large"}


def test_calcular_custo_de_modelo_conhecido_nao_soma_o_raciocinio_duas_vezes():
    # 1000 de entrada a US$ 2/M e 500 de saída a US$ 10/M; os 200 de raciocínio já estão nos 500 de saída.
    assert calcular_custo("claude-sonnet-5-5", USO) == pytest.approx(0.002 + 0.005)
    assert calcular_custo("gpt-5", USO) == pytest.approx(0.00125 + 0.005)


def test_calcular_custo_de_embedding_so_cobra_a_entrada():
    assert calcular_custo("text-embedding-3-large", uso(1_000_000, 0)) == pytest.approx(0.13)


def test_calcular_custo_de_modelo_desconhecido_ou_sem_uso_e_none():
    assert calcular_custo("modelo-x", USO) is None
    assert calcular_custo("gpt-5", None) is None


def test_normalizar_uso_completo():
    metadados = {
        "input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500,
        "input_token_details": {"cache_read": 1000, "cache_creation": 50},
        "output_token_details": {"reasoning": 120},
    }
    assert normalizar_uso(metadados) == {
        "tokens_entrada": 1200, "tokens_saida": 300, "tokens_raciocinio": 120,
        "tokens_cache_leitura": 1000, "tokens_cache_escrita": 50,
    }


def test_normalizar_uso_incompleto_poe_zero_no_que_falta():
    assert normalizar_uso({"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}) == uso(10, 2)
    incompleto = {"input_tokens": 10, "input_token_details": {"audio": 3}, "output_token_details": {}}
    assert normalizar_uso(incompleto) == uso(10, 0)


def test_normalizar_uso_ausente_e_none():
    assert normalizar_uso(None) is None
    assert normalizar_uso({}) is None


# --- contexto e identificador da execução ----------------------------------------------------------------


def test_contexto_aninhado_herda_o_que_nao_muda_e_restaura_o_anterior():
    assert contexto_atual() == Contexto()
    with contexto(etapa="votos", alvo="questoes"):
        assert contexto_atual() == Contexto("votos", "questoes", None)
        with contexto(id_questao="Q0001"):
            assert contexto_atual() == Contexto("votos", "questoes", "Q0001")
        assert contexto_atual() == Contexto("votos", "questoes", None)
        with contexto(etapa="reparo", id_questao="Q0002"):
            assert contexto_atual() == Contexto("reparo", "questoes", "Q0002")
        assert contexto_atual() == Contexto("votos", "questoes", None)
    assert contexto_atual() == Contexto()


def test_contexto_e_restaurado_mesmo_com_excecao():
    with pytest.raises(RuntimeError):
        with contexto(etapa="votos"):
            raise RuntimeError("quebrou")
    assert contexto_atual() == Contexto()


def test_execucao_id_tem_data_hora_e_pid_e_a_variavel_de_ambiente_sobrescreve(monkeypatch):
    assert re.fullmatch(r"\d{8}T\d{6}-\d+", custos.novo_execucao_id())
    monkeypatch.delenv("CLASSIFICADOR_EXECUCAO_ID", raising=False)
    assert custos.execucao_id() == custos.EXECUCAO_ID
    monkeypatch.setenv("CLASSIFICADOR_EXECUCAO_ID", "execucao-de-teste")
    assert custos.execucao_id() == "execucao-de-teste"


# --- livro-razão -----------------------------------------------------------------------------------------


def test_registrar_grava_uma_linha_json_com_todos_os_campos(tmp_path, monkeypatch):
    monkeypatch.setenv("CLASSIFICADOR_EXECUCAO_ID", "exec-1")
    caminho = tmp_path / "custos" / "chamadas.jsonl"
    livro = LivroRazao(caminho)
    with contexto(etapa="votos", alvo="questoes", id_questao="Q0001"):
        livro.registrar(
            provedor="anthropic", modelo="claude-sonnet-5-5", configuracao="effort-low_max4096_json_schema",
            origem="api", tentativa=1, hash_prompt="abc", tamanho_prompt_chars=900, n_candidatos=12,
            uso=USO, latencia_ms=850.0, sucesso=True, erro=None, tema_votado=5524, confianca=0.9,
        )
    [linha] = ler_livro(caminho)
    assert list(linha) == list(custos.CAMPOS)
    assert linha["execucao_id"] == "exec-1"
    assert (linha["etapa"], linha["alvo"], linha["id_questao"]) == ("votos", "questoes", "Q0001")
    assert (linha["provedor"], linha["modelo"], linha["origem"], linha["tentativa"]) == ("anthropic", "claude-sonnet-5-5", "api", 1)
    assert linha["configuracao"] == "effort-low_max4096_json_schema"
    assert (linha["hash_prompt"], linha["tamanho_prompt_chars"], linha["n_candidatos"]) == ("abc", 900, 12)
    assert {campo: linha[campo] for campo in custos.CAMPOS_USO} == USO
    assert linha["latencia_ms"] == 850.0
    assert linha["custo_usd"] == pytest.approx(0.007)
    assert linha["preco_versao"] == "2026-10-15"
    assert linha["n_textos"] is None
    assert (linha["sucesso"], linha["erro"], linha["tema_votado"], linha["confianca"]) == (True, None, 5524, 0.9)
    assert datetime.fromisoformat(linha["quando"]).utcoffset() == timedelta(0)


def test_registrar_acrescenta_linhas_e_cache_custa_zero_e_uso_desconhecido_fica_nulo(tmp_path):
    caminho = tmp_path / "chamadas.jsonl"
    livro = LivroRazao(caminho)
    livro.registrar(provedor="openai", modelo="gpt-5", origem="cache", tentativa=1)
    livro.registrar(provedor="openai", modelo="gpt-5", origem="api", tentativa=1, sucesso=False, erro="RuntimeError: 503")
    cache, falha = ler_livro(caminho)
    assert cache["custo_usd"] == 0.0
    assert cache["tokens_entrada"] is None and cache["latencia_ms"] is None
    assert falha["custo_usd"] is None
    assert (falha["sucesso"], falha["erro"]) == (False, "RuntimeError: 503")


def test_registrar_recusa_campo_desconhecido(tmp_path):
    with pytest.raises(TypeError, match="custo_total"):
        LivroRazao(tmp_path / "chamadas.jsonl").registrar(custo_total=1.0)


# --- resumo ----------------------------------------------------------------------------------------------


@pytest.fixture
def livro_pequeno(tmp_path, monkeypatch):
    """3 chamadas reais de modelo, 2 acertos de cache (um sem uso, outro com o uso do arquivo) e 1 lote de embeddings."""
    livro = LivroRazao(tmp_path / "custos" / "chamadas.jsonl")
    monkeypatch.setenv("CLASSIFICADOR_EXECUCAO_ID", "exec-1")
    with contexto(etapa="votos", alvo="questoes", id_questao="Q0001"):
        livro.registrar(provedor="anthropic", modelo="claude-sonnet-5-5", origem="api", tentativa=1,
                        uso=uso(1000, 500, 100))  # 0.002 + 0.005 = 0.007
        livro.registrar(provedor="openai", modelo="gpt-5", origem="api", tentativa=1,
                        uso=uso(2000, 1000, 400))  # 0.0025 + 0.01 = 0.0125
    with contexto(etapa="reparo", alvo="questoes", id_questao="Q0002"):
        livro.registrar(provedor="anthropic", modelo="claude-sonnet-5-5", origem="api", tentativa=1,
                        uso=uso(1000, 1000))  # 0.002 + 0.01 = 0.012
    livro.registrar(etapa="embeddings", provedor="openai", modelo="text-embedding-3-large", origem="api",
                    tentativa=1, n_textos=10, uso=uso(1_000_000, 0))  # 0.13
    monkeypatch.setenv("CLASSIFICADOR_EXECUCAO_ID", "exec-2")
    with contexto(etapa="votos", alvo="questoes", id_questao="Q0003"):
        livro.registrar(provedor="openai", modelo="gpt-5", origem="cache", tentativa=1)
        livro.registrar(provedor="anthropic", modelo="claude-sonnet-5-5", origem="cache", tentativa=1, uso=uso(800, 300))
    decisoes = tmp_path / "decisoes_questoes.jsonl"
    decisoes.write_text("".join(json.dumps({"id": f"Q000{i}"}) + "\n" for i in range(1, 5)), encoding="utf-8")
    publicacao = tmp_path / "publicacao.csv"
    publicacao.write_text(
        "id,decisao,tema_central_id,extras_ids,confianca,motivo\n"
        "Q0001,publicar,5524,,0.90,\nQ0002,revisar,5524,,0.50,baixa\nQ0003,publicar,5159,,0.95,\n",
        encoding="utf-8",
    )
    return livro, decisoes, publicacao


def resumir(livro_pequeno, tmp_path, painel=None):
    livro, decisoes, publicacao = livro_pequeno
    return resumir_custos(livro.caminho, painel=painel, decisoes=decisoes, publicacao=publicacao,
                          saida=tmp_path / "custos" / "custos_resumo.json")


def test_resumo_agrega_totais_e_por_origem(livro_pequeno, tmp_path):
    resumo = resumir(livro_pequeno, tmp_path)
    assert resumo["linhas"] == 6
    total = resumo["total"]
    assert total["linhas"] == 6
    assert total["uso_desconhecido"] == 1
    assert total["tokens_entrada"] == 1000 + 2000 + 1000 + 1_000_000 + 800
    assert total["tokens_saida"] == 500 + 1000 + 1000 + 0 + 300
    assert total["tokens_raciocinio"] == 500
    assert total["custo_usd"] == pytest.approx(0.1615)
    api = resumo["por_origem"]["api"]
    assert (api["linhas"], api["uso_desconhecido"]) == (4, 0)
    assert api["custo_usd"] == pytest.approx(0.1615)
    cache = resumo["por_origem"]["cache"]
    assert (cache["linhas"], cache["uso_desconhecido"], cache["tokens_entrada"], cache["custo_usd"]) == (2, 1, 800, 0.0)


def test_resumo_agrega_por_execucao_etapa_e_modelo(livro_pequeno, tmp_path):
    resumo = resumir(livro_pequeno, tmp_path)
    assert {chave: g["linhas"] for chave, g in resumo["por_execucao"].items()} == {"exec-1": 4, "exec-2": 2}
    assert resumo["por_execucao"]["exec-2"]["custo_usd"] == 0.0
    etapas = resumo["por_etapa"]
    assert {chave: g["linhas"] for chave, g in etapas.items()} == {"votos": 4, "reparo": 1, "embeddings": 1}
    assert etapas["votos"]["custo_usd"] == pytest.approx(0.0195)
    assert etapas["reparo"]["custo_usd"] == pytest.approx(0.012)
    modelos = resumo["por_modelo"]
    assert set(modelos) == {"anthropic/claude-sonnet-5-5", "openai/gpt-5", "openai/text-embedding-3-large"}
    assert modelos["anthropic/claude-sonnet-5-5"]["linhas"] == 3
    assert modelos["anthropic/claude-sonnet-5-5"]["custo_usd"] == pytest.approx(0.019)
    assert modelos["openai/text-embedding-3-large"]["custo_usd"] == pytest.approx(0.13)


def test_custo_por_questao_e_por_publicada_usam_votos_e_reparo_das_questoes(livro_pequeno, tmp_path):
    resumo = resumir(livro_pequeno, tmp_path)
    unitario = resumo["por_questao"]
    assert unitario["custo_usd"] == pytest.approx(0.007 + 0.0125 + 0.012)
    assert unitario["questoes"] == 4
    assert unitario["custo_por_questao"] == pytest.approx(0.0315 / 4)
    assert unitario["publicadas"] == 2
    assert unitario["custo_por_publicada"] == pytest.approx(0.0315 / 2)


def test_sem_decisoes_nem_publicacao_o_custo_unitario_fica_nulo(livro_pequeno, tmp_path):
    livro, _, _ = livro_pequeno
    resumo = resumir_custos(livro.caminho, decisoes=tmp_path / "nao_existe.jsonl", publicacao=tmp_path / "nao_existe.csv",
                            saida=None)
    assert resumo["por_questao"]["custo_por_questao"] is None
    assert resumo["por_questao"]["custo_por_publicada"] is None


def test_painel_compara_com_o_custo_calculado_das_chamadas_reais(livro_pequeno, tmp_path):
    resumo = resumir(livro_pequeno, tmp_path, painel={"anthropic": 0.02, "openai": 0.15})
    anthropic = resumo["painel"]["anthropic"]
    assert anthropic["informado"] == 0.02
    assert anthropic["calculado"] == pytest.approx(0.019)
    assert anthropic["diferenca"] == pytest.approx(0.001)
    openai = resumo["painel"]["openai"]
    assert openai["calculado"] == pytest.approx(0.0125 + 0.13)
    assert openai["diferenca"] == pytest.approx(0.15 - 0.1425)


def test_resumo_e_gravado_em_json(livro_pequeno, tmp_path):
    resumo = resumir(livro_pequeno, tmp_path, painel={"anthropic": 0.02})
    gravado = json.loads((tmp_path / "custos" / "custos_resumo.json").read_text(encoding="utf-8"))
    assert gravado == json.loads(json.dumps(resumo))


def test_linha_quebrada_no_livro_e_ignorada_sem_quebrar_a_soma(livro_pequeno, tmp_path):
    livro, _, _ = livro_pequeno
    with livro.caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write('{"quando": "interrompida\n')
    resumo = resumir(livro_pequeno, tmp_path)
    assert resumo["linhas"] == 6
    assert resumo["linhas_invalidas"] == 1
    assert resumo["total"]["custo_usd"] == pytest.approx(0.1615)


def test_livro_inexistente_nao_falha_e_nao_grava_resumo(tmp_path):
    saida = tmp_path / "custos_resumo.json"
    resumo = resumir_custos(tmp_path / "nao_existe.jsonl", saida=saida)
    assert resumo["linhas"] == 0
    assert not saida.exists()
    assert "nenhum registro" in formatar_resumo(resumo).lower()


def test_tabela_impressa_traz_as_secoes_e_o_painel(livro_pequeno, tmp_path):
    texto = formatar_resumo(resumir(livro_pequeno, tmp_path, painel={"anthropic": 0.02}))
    for trecho in ("Por execução", "Por etapa", "Por modelo", "Por origem", "Total", "exec-1",
                   "anthropic/claude-sonnet-5-5", "uso desconhecido", "Custo por questão", "Custo por publicada", "Painel"):
        assert trecho in texto
    assert "0.161500" in texto
