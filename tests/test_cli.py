import json
import sys
from functools import partial
from types import SimpleNamespace

import pytest

from sistema import cli
from sistema.embeddings import CacheEmbeddings

SUBCOMANDOS = ("tudo", "embeddings", "dicionario", "avaliar-folha", "comparar-fontes", "custos")


def test_help_lista_os_subcomandos(capsys):
    with pytest.raises(SystemExit) as saida:
        cli.main(["--help"])
    assert saida.value.code == 0
    texto = capsys.readouterr().out
    for subcomando in SUBCOMANDOS:
        assert subcomando in texto


def test_sem_subcomando_e_erro_de_uso():
    with pytest.raises(SystemExit) as saida:
        cli.main([])
    assert saida.value.code == 2


@pytest.fixture
def pipeline_falso(monkeypatch):
    """Troca as funções do pipeline por espiões que só anotam a chamada."""
    from sistema import pipeline

    chamadas = []
    monkeypatch.setattr(pipeline, "tudo", lambda com_corpus: chamadas.append(("tudo", com_corpus)))
    monkeypatch.setattr(pipeline, "avaliar_folha", lambda com_corpus: chamadas.append(("avaliar-folha", com_corpus)))
    monkeypatch.setattr(pipeline, "comparar", lambda: chamadas.append(("comparar-fontes",)))
    return chamadas


@pytest.mark.parametrize("argumentos, esperado", [
    (["tudo"], ("tudo", None)),  # sem escolha: a regra dos 3 pontos decide o corpus
    (["tudo", "--com-corpus"], ("tudo", True)),
    (["tudo", "--sem-corpus"], ("tudo", False)),
    (["avaliar-folha"], ("avaliar-folha", False)),
    (["avaliar-folha", "--com-corpus"], ("avaliar-folha", True)),
    (["comparar-fontes"], ("comparar-fontes",)),
])
def test_subcomandos_do_pipeline(pipeline_falso, argumentos, esperado):
    assert cli.main(argumentos) == 0
    assert pipeline_falso == [esperado]


def test_tudo_nao_aceita_com_e_sem_corpus_juntos(pipeline_falso):
    with pytest.raises(SystemExit) as saida:
        cli.main(["tudo", "--com-corpus", "--sem-corpus"])
    assert saida.value.code == 2 and pipeline_falso == []


def test_argumento_desconhecido_e_erro_de_uso():
    with pytest.raises(SystemExit) as saida:
        cli.main(["tudo", "--nao-existe"])
    assert saida.value.code == 2


# --- dicionario ------------------------------------------------------------------------------------------


def test_dicionario_repassa_o_resto_dos_argumentos(monkeypatch):
    recebidos = []
    monkeypatch.setitem(sys.modules, "sistema.dicionario", SimpleNamespace(main=lambda argumentos: recebidos.append(argumentos)))
    assert cli.main(["dicionario", "--saida", "x.json", "--help"]) == 0
    assert recebidos == [["--saida", "x.json", "--help"]]


def test_dicionario_devolve_o_codigo_do_modulo(monkeypatch):
    monkeypatch.setitem(sys.modules, "sistema.dicionario", SimpleNamespace(main=lambda argumentos: 3))
    assert cli.main(["dicionario"]) == 3


def test_dicionario_sem_o_modulo_da_mensagem_clara(monkeypatch):
    monkeypatch.setitem(sys.modules, "sistema.dicionario", None)  # import levanta ModuleNotFoundError
    with pytest.raises(SystemExit) as saida:
        cli.main(["dicionario"])
    assert "sistema.dicionario" in str(saida.value.code)
    assert "ainda não existe" in str(saida.value.code)


def test_dicionario_sem_main_da_mensagem_clara(monkeypatch):
    monkeypatch.setitem(sys.modules, "sistema.dicionario", SimpleNamespace())
    with pytest.raises(SystemExit) as saida:
        cli.main(["dicionario"])
    assert "main" in str(saida.value.code)


# --- embeddings ------------------------------------------------------------------------------------------


@pytest.fixture
def embeddings_de_teste(monkeypatch, tmp_path, cliente_falso):
    """`sistema embeddings` com textos pequenos, cache em pasta temporária e o SDK da OpenAI trocado pelo dublê."""
    clientes = []

    def openai_falso(api_key):
        cliente = cliente_falso()
        clientes.append((api_key, cliente))
        return cliente

    monkeypatch.setattr(cli, "textos_do_sistema", lambda: ["questão um", "tema dois", "questão um", "exemplo três"])
    monkeypatch.setattr(cli, "CacheEmbeddings", partial(CacheEmbeddings, pasta=tmp_path / "embeddings"))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai_falso))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    return SimpleNamespace(pasta=tmp_path / "embeddings", clientes=clientes)


def test_embeddings_grava_cache_e_livro(embeddings_de_teste, livro_temporario, capsys):
    assert cli.main(["embeddings"]) == 0
    [(chave, cliente)] = embeddings_de_teste.clientes
    assert chave == "sk-teste"
    assert [lote for _, lote in cliente.lotes] == [["questão um", "tema dois", "exemplo três"]]
    cache = CacheEmbeddings(pasta=embeddings_de_teste.pasta)
    assert cache.faltando(["questão um", "tema dois", "exemplo três"]) == []
    [linha] = [json.loads(t) for t in livro_temporario.caminho.read_text(encoding="utf-8").splitlines()]
    assert (linha["etapa"], linha["n_textos"], linha["tokens_entrada"]) == ("embeddings", 3, 30)
    saida = capsys.readouterr().out
    assert "3 texto(s)" in saida and "3 a gerar" in saida


def test_embeddings_com_cache_completo_nao_chama_a_api(embeddings_de_teste, livro_temporario, capsys):
    cli.main(["embeddings"])
    capsys.readouterr()
    assert cli.main(["embeddings"]) == 0
    assert len(embeddings_de_teste.clientes) == 1
    assert len(livro_temporario.caminho.read_text(encoding="utf-8").splitlines()) == 1
    assert "0 a gerar" in capsys.readouterr().out


# --- custos ----------------------------------------------------------------------------------------------


def test_custos_imprime_e_grava_o_resumo_do_livro_em_uso(livro_temporario, capsys):
    livro_temporario.registrar(etapa="embeddings", provedor="openai", modelo="text-embedding-3-large", n_textos=100,
                               uso={"tokens_entrada": 1_000_000, "tokens_saida": 0})
    assert cli.main(["custos", "--painel", "openai=0,14"]) == 0
    texto = capsys.readouterr().out
    assert "Painel" in texto and "openai" in texto and "0.130000" in texto
    resumo = json.loads(livro_temporario.caminho.with_name("custos_resumo.json").read_text(encoding="utf-8"))
    assert resumo["painel"]["openai"]["informado"] == pytest.approx(0.14)


def test_custos_sem_livro_avisa(capsys):
    assert cli.main(["custos"]) == 0
    assert "nenhum registro" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("valor", ["openai", "=1", "openai=muito"])
def test_custos_recusa_painel_mal_formado(valor):
    with pytest.raises(SystemExit) as saida:
        cli.main(["custos", "--painel", valor])
    assert saida.value.code == 2


def test_ler_valor_painel():
    assert cli.ler_valor_painel(" openai = 0,70 ") == ("openai", 0.70)


def test_scripts_rodar_e_ponto_de_entrada_protegido_para_o_spawn():
    """Os processos da validação (spawn) reimportam o script principal como `__mp_main__`: ele não pode rodar o CLI
    de novo; chamado como script, roda o CLI com os argumentos dele."""
    import runpy
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "rodar.py"
    runpy.run_path(str(script), run_name="__mp_main__")  # sem a guarda, chamaria cli.main com os argumentos do pytest
    import os

    ambiente = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # a ajuda tem acentos; o console do Windows não é UTF-8
    resultado = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, encoding="utf-8",
                               env=ambiente)
    assert resultado.returncode == 0, resultado.stderr
    for subcomando in SUBCOMANDOS:
        assert subcomando in resultado.stdout
