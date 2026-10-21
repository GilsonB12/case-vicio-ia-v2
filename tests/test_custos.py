import json
import re
from datetime import datetime, timedelta

import pytest

from sistema import custos
from sistema.config import CAMINHOS
from sistema.custos import LivroRazao, calcular_custo, formatar_resumo, resumir_custos

VARIAVEL = "SISTEMA_EXECUCAO_ID"


def uso(entrada, saida=0):
    return {"tokens_entrada": entrada, "tokens_saida": saida}


def ler_livro(caminho):
    return [json.loads(linha) for linha in caminho.read_text(encoding="utf-8").splitlines()]


# --- tabela de preços ------------------------------------------------------------------------------------


def test_tabela_de_precos_tem_versao_e_o_modelo_de_embeddings():
    assert custos.TABELA_PRECOS["versao"] == "2026-10-15"
    assert "text-embedding-3-large" in custos.TABELA_PRECOS["modelos"]


def test_calcular_custo_de_modelo_conhecido():
    # 1000 de entrada a US$ 1,25/M e 500 de saída a US$ 10/M.
    assert calcular_custo("gpt-5", uso(1000, 500)) == pytest.approx(0.00125 + 0.005)


def test_calcular_custo_de_embedding_so_cobra_a_entrada():
    assert calcular_custo("text-embedding-3-large", uso(1_000_000)) == pytest.approx(0.13)
    assert calcular_custo("text-embedding-3-large", {"tokens_entrada": 1_000_000}) == pytest.approx(0.13)


def test_calcular_custo_de_modelo_desconhecido_ou_sem_uso_e_none():
    assert calcular_custo("modelo-x", uso(10)) is None
    assert calcular_custo("text-embedding-3-large", None) is None


def test_descrever_erro_corta_a_mensagem():
    assert custos.descrever_erro(RuntimeError("503")) == "RuntimeError: 503"
    assert len(custos.descrever_erro(RuntimeError("x" * 1000))) == len("RuntimeError: ") + custos.TAMANHO_ERRO


# --- identificador da execução ---------------------------------------------------------------------------


def test_execucao_id_tem_data_hora_e_pid_e_a_variavel_de_ambiente_sobrescreve(monkeypatch):
    assert re.fullmatch(r"\d{8}T\d{6}-\d+", custos.novo_execucao_id())
    monkeypatch.delenv(VARIAVEL, raising=False)
    assert custos.execucao_id() == custos.EXECUCAO_ID
    monkeypatch.setenv(VARIAVEL, "execucao-de-teste")
    assert custos.execucao_id() == "execucao-de-teste"


# --- livro-razão -----------------------------------------------------------------------------------------


def test_livro_global_aponta_para_dados_custos_fora_dos_testes():
    assert LivroRazao().caminho == CAMINHOS.custos / "chamadas.jsonl"


def test_registrar_grava_uma_linha_json_com_todos_os_campos(tmp_path, monkeypatch):
    monkeypatch.setenv(VARIAVEL, "exec-1")
    caminho = tmp_path / "custos" / "chamadas.jsonl"
    livro = LivroRazao(caminho)
    devolvida = livro.registrar(etapa="embeddings", provedor="openai", modelo="text-embedding-3-large",
                                n_textos=100, uso=uso(1_000_000), latencia_ms=850.0)
    [linha] = ler_livro(caminho)
    assert linha == devolvida
    assert list(linha) == list(custos.CAMPOS)
    assert linha["execucao_id"] == "exec-1"
    assert (linha["etapa"], linha["provedor"], linha["modelo"], linha["n_textos"]) == (
        "embeddings", "openai", "text-embedding-3-large", 100)
    assert (linha["tokens_entrada"], linha["tokens_saida"]) == (1_000_000, 0)
    assert linha["latencia_ms"] == 850.0
    assert linha["custo_usd"] == pytest.approx(0.13)
    assert linha["preco_versao"] == "2026-10-15"
    assert (linha["sucesso"], linha["erro"]) == (True, None)
    assert datetime.fromisoformat(linha["quando"]).utcoffset() == timedelta(0)


def test_registrar_acrescenta_linhas_e_uso_desconhecido_fica_nulo(tmp_path):
    caminho = tmp_path / "chamadas.jsonl"
    livro = LivroRazao(caminho)
    livro.registrar(etapa="embeddings", provedor="openai", modelo="text-embedding-3-large", uso=uso(10))
    livro.registrar(etapa="embeddings", provedor="openai", modelo="text-embedding-3-large", sucesso=False,
                    erro="RuntimeError: 503")
    certa, falha = ler_livro(caminho)
    assert certa["custo_usd"] == pytest.approx(10 * 0.13 / 1_000_000)
    assert falha["tokens_entrada"] is None and falha["latencia_ms"] is None
    assert falha["custo_usd"] is None
    assert (falha["sucesso"], falha["erro"]) == (False, "RuntimeError: 503")


def test_registrar_aceita_custo_explicito(tmp_path):
    linha = LivroRazao(tmp_path / "chamadas.jsonl").registrar(provedor="openai", modelo="x", custo_usd=0.5)
    assert linha["custo_usd"] == 0.5


def test_registrar_recusa_campo_desconhecido_inclusive_os_dos_votos_da_etapa_1(tmp_path):
    livro = LivroRazao(tmp_path / "chamadas.jsonl")
    with pytest.raises(TypeError, match="custo_total"):
        livro.registrar(custo_total=1.0)
    for campo in ("tema_votado", "confianca", "n_candidatos", "id_questao", "hash_prompt"):
        with pytest.raises(TypeError, match=campo):
            livro.registrar(**{campo: 1})


# --- resumo ----------------------------------------------------------------------------------------------


@pytest.fixture
def livro_pequeno(tmp_path, monkeypatch):
    """Dois lotes de embeddings numa execução, uma falha e um lote noutra, e uma linha de outro provedor."""
    livro = LivroRazao(tmp_path / "custos" / "chamadas.jsonl")
    monkeypatch.setenv(VARIAVEL, "exec-1")
    livro.registrar(etapa="embeddings", provedor="openai", modelo="text-embedding-3-large", n_textos=100,
                    uso=uso(1_000_000))  # 0.13
    livro.registrar(etapa="embeddings", provedor="openai", modelo="text-embedding-3-large", n_textos=50,
                    uso=uso(500_000))  # 0.065
    monkeypatch.setenv(VARIAVEL, "exec-2")
    livro.registrar(etapa="embeddings", provedor="openai", modelo="text-embedding-3-large", n_textos=100,
                    sucesso=False, erro="RuntimeError: 503")  # uso e custo desconhecidos
    livro.registrar(etapa="embeddings", provedor="openai", modelo="text-embedding-3-large", n_textos=10,
                    uso=uso(100_000))  # 0.013
    livro.registrar(etapa="teste", provedor="anthropic", modelo="claude-sonnet-5-5", n_textos=1,
                    uso=uso(1000, 500))  # 0.002 + 0.005 = 0.007
    return livro


def resumir(livro, tmp_path, painel=None):
    return resumir_custos(livro.caminho, painel=painel, saida=tmp_path / "custos" / "custos_resumo.json")


def test_resumo_agrega_os_totais(livro_pequeno, tmp_path):
    resumo = resumir(livro_pequeno, tmp_path)
    assert resumo["linhas"] == 5
    total = resumo["total"]
    assert total["linhas"] == 5
    assert total["uso_desconhecido"] == 1
    assert total["custo_desconhecido"] == 1
    assert total["n_textos"] == 261
    assert total["tokens_entrada"] == 1_000_000 + 500_000 + 100_000 + 1000
    assert total["tokens_saida"] == 500
    assert total["custo_usd"] == pytest.approx(0.215)


def test_resumo_agrega_por_execucao_etapa_e_modelo(livro_pequeno, tmp_path):
    resumo = resumir(livro_pequeno, tmp_path)
    assert set(resumo) >= {"por_execucao", "por_etapa", "por_modelo"}
    assert "por_origem" not in resumo and "por_questao" not in resumo
    assert {chave: g["linhas"] for chave, g in resumo["por_execucao"].items()} == {"exec-1": 2, "exec-2": 3}
    assert resumo["por_execucao"]["exec-1"]["custo_usd"] == pytest.approx(0.195)
    etapas = resumo["por_etapa"]
    assert {chave: g["linhas"] for chave, g in etapas.items()} == {"embeddings": 4, "teste": 1}
    assert etapas["embeddings"]["custo_usd"] == pytest.approx(0.208)
    modelos = resumo["por_modelo"]
    assert set(modelos) == {"openai/text-embedding-3-large", "anthropic/claude-sonnet-5-5"}
    assert modelos["openai/text-embedding-3-large"]["n_textos"] == 260


def test_painel_compara_com_o_custo_calculado(livro_pequeno, tmp_path):
    resumo = resumir(livro_pequeno, tmp_path, painel={"openai": 0.21, "anthropic": 0.0})
    openai = resumo["painel"]["openai"]
    assert openai["informado"] == 0.21
    assert openai["calculado"] == pytest.approx(0.208)
    assert openai["diferenca"] == pytest.approx(0.002)
    assert openai["diferenca_percentual"] == pytest.approx(100 * 0.002 / 0.21, abs=0.01)
    assert openai["linhas_sem_custo"] == 1
    assert resumo["painel"]["anthropic"]["diferenca_percentual"] is None


def test_resumo_e_gravado_em_json(livro_pequeno, tmp_path):
    resumo = resumir(livro_pequeno, tmp_path, painel={"openai": 0.21})
    gravado = json.loads((tmp_path / "custos" / "custos_resumo.json").read_text(encoding="utf-8"))
    assert gravado == json.loads(json.dumps(resumo))


def test_linha_quebrada_no_livro_e_ignorada_sem_quebrar_a_soma(livro_pequeno, tmp_path):
    with livro_pequeno.caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write('{"quando": "interrompida\n')
    resumo = resumir(livro_pequeno, tmp_path)
    assert resumo["linhas"] == 5
    assert resumo["linhas_invalidas"] == 1
    assert resumo["total"]["custo_usd"] == pytest.approx(0.215)


def test_livro_inexistente_nao_falha_e_nao_grava_resumo(tmp_path):
    saida = tmp_path / "custos_resumo.json"
    resumo = resumir_custos(tmp_path / "nao_existe.jsonl", saida=saida)
    assert resumo["linhas"] == 0
    assert not saida.exists()
    assert "nenhum registro" in formatar_resumo(resumo).lower()


def test_tabela_impressa_traz_as_secoes_e_o_painel(livro_pequeno, tmp_path):
    texto = formatar_resumo(resumir(livro_pequeno, tmp_path, painel={"openai": 0.21}))
    for trecho in ("Por execução", "Por etapa", "Por modelo", "Total", "exec-1", "openai/text-embedding-3-large",
                   "uso desconhecido", "textos", "Painel", "sem custo calculado"):
        assert trecho in texto
    assert "0.215000" in texto
