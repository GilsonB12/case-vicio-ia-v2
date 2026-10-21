import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sistema import embeddings
from sistema.config import ChaveAusenteError
from sistema.custos import LivroRazao, calcular_custo
from sistema.embeddings import TAMANHO_LOTE, CacheEmbeddings, EmbeddingAusenteError

RAIZ = Path(__file__).resolve().parents[1]
MODELO = "text-embedding-3-large"


def linhas_do_livro(livro):
    if not livro.caminho.exists():
        return []
    return [json.loads(linha) for linha in livro.caminho.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def cache(tmp_path):
    return CacheEmbeddings(pasta=tmp_path / "embeddings")


@pytest.fixture
def livro(tmp_path):
    return LivroRazao(tmp_path / "custos" / "chamadas.jsonl")


# --- leitura só do cache ---------------------------------------------------------------------------------


def test_padroes_do_cache():
    padrao = CacheEmbeddings()
    assert padrao.modelo == MODELO
    assert padrao.pasta == RAIZ / "dados" / "embeddings" / MODELO
    assert TAMANHO_LOTE == 100


def test_vetor_e_matriz_leem_do_cache(cache, cliente_falso, vetor_esperado, livro):
    cache.gerar(["a", "b"], cliente=cliente_falso(), livro=livro)
    leitor = CacheEmbeddings(pasta=cache.pasta.parent)
    np.testing.assert_allclose(leitor.vetor("a"), vetor_esperado("a"), rtol=1e-6)
    matriz = leitor.matriz(["b", "a", "b"])
    assert matriz.shape == (3, 8)
    np.testing.assert_allclose(matriz[0], vetor_esperado("b"), rtol=1e-6)
    np.testing.assert_allclose(matriz[1], vetor_esperado("a"), rtol=1e-6)
    np.testing.assert_allclose(matriz[2], matriz[0])


def test_arquivo_por_hash_sha256_do_texto_com_o_vetor(cache, cliente_falso, vetor_esperado, livro):
    cache.gerar(["Tribunal do Júri"], cliente=cliente_falso(), livro=livro)
    chave = hashlib.sha256("Tribunal do Júri".encode("utf-8")).hexdigest()
    arquivo = cache.pasta / f"{chave}.json"
    assert cache.arquivo("Tribunal do Júri") == arquivo
    assert json.loads(arquivo.read_text(encoding="utf-8")) == pytest.approx(vetor_esperado("Tribunal do Júri"))


def test_texto_ausente_levanta_erro_com_a_lista(cache, cliente_falso, livro):
    cache.gerar(["a"], cliente=cliente_falso(), livro=livro)
    with pytest.raises(EmbeddingAusenteError) as erro:
        cache.matriz(["a", "x", "y", "x"])
    assert erro.value.textos_faltando == ["x", "y"]
    mensagem = str(erro.value)
    assert "2 texto(s)" in mensagem and "sistema embeddings" in mensagem
    assert "x" in mensagem and "y" in mensagem
    with pytest.raises(EmbeddingAusenteError):
        cache.vetor("z")


def test_mensagem_lista_os_10_primeiros_e_a_contagem(cache):
    textos = [f"texto {numero:02d}" for numero in range(15)]
    with pytest.raises(EmbeddingAusenteError) as erro:
        cache.matriz(textos)
    assert erro.value.textos_faltando == textos
    mensagem = str(erro.value)
    assert "15 texto(s)" in mensagem
    assert "texto 09" in mensagem and "texto 10" not in mensagem
    assert "mais 5" in mensagem


def test_mensagem_encurta_textos_longos(cache):
    longo = "palavra " * 200
    with pytest.raises(EmbeddingAusenteError) as erro:
        cache.vetor(longo)
    assert erro.value.textos_faltando == [longo]
    assert len(str(erro.value)) < 400


def test_leitura_nao_importa_cliente_nenhum(cache, cliente_falso, monkeypatch, livro):
    cache.gerar(["a"], cliente=cliente_falso(), livro=livro)
    monkeypatch.setitem(sys.modules, "openai", None)  # qualquer `import openai` levantaria ImportError
    leitor = CacheEmbeddings(pasta=cache.pasta.parent)
    assert leitor.matriz(["a"]).shape == (1, 8)
    with pytest.raises(EmbeddingAusenteError):
        leitor.matriz(["b"])
    assert leitor.faltando(["a", "b"]) == ["b"]


def test_modulos_do_sistema_nao_carregam_o_sdk_da_openai(tmp_path):
    programa = (
        "import sys\n"
        "import sistema.config, sistema.dados, sistema.arvore, sistema.custos, sistema.embeddings, sistema.cli\n"
        "from sistema.embeddings import CacheEmbeddings, EmbeddingAusenteError\n"
        f"cache = CacheEmbeddings(pasta={str(tmp_path)!r})\n"
        "try:\n"
        "    cache.matriz(['x'])\n"
        "except EmbeddingAusenteError:\n"
        "    pass\n"
        "print(sorted(nome for nome in sys.modules if nome == 'openai' or nome.startswith('openai.')))\n"
    )
    resultado = subprocess.run([sys.executable, "-c", programa], capture_output=True, text=True, cwd=RAIZ,
                               env={**os.environ, "PYTHONPATH": str(RAIZ / "src")}, check=True)
    assert resultado.stdout.strip() == "[]"


def test_cache_corrompido_conta_como_ausente_e_gerar_regrava(cache, cliente_falso, vetor_esperado, livro):
    cache.gerar(["x"], cliente=cliente_falso(), livro=livro)
    cache.arquivo("x").write_text("{isso nao e json", encoding="utf-8")
    leitor = CacheEmbeddings(pasta=cache.pasta.parent)
    with pytest.raises(EmbeddingAusenteError):
        leitor.vetor("x")
    cliente = cliente_falso()
    assert leitor.gerar(["x"], cliente=cliente, livro=livro) == 1
    assert len(cliente.lotes) == 1
    np.testing.assert_allclose(leitor.vetor("x"), vetor_esperado("x"), rtol=1e-6)


def test_outro_modelo_nao_enxerga_o_cache(cache, cliente_falso, livro):
    cache.gerar(["a"], cliente=cliente_falso(), livro=livro)
    outro = CacheEmbeddings(pasta=cache.pasta.parent, modelo="text-embedding-3-small")
    assert outro.faltando(["a"]) == ["a"]


def test_matriz_vazia(cache):
    assert cache.matriz([]).shape[0] == 0


# --- geração (única parte que fala com a API) ------------------------------------------------------------


def test_gerar_so_chama_a_api_para_os_ausentes_em_lotes_de_100(cache, cliente_falso, livro):
    cache.gerar(["t000", "t001", "t002"], cliente=cliente_falso(), livro=livro)
    textos = [f"t{numero:03d}" for numero in range(253)]
    cliente = cliente_falso()
    assert cache.gerar(textos + ["t010", "t250"], cliente=cliente, livro=livro) == 250
    assert [len(lote) for _, lote in cliente.lotes] == [100, 100, 50]
    assert {modelo for modelo, _ in cliente.lotes} == {MODELO}
    enviados = [texto for _, lote in cliente.lotes for texto in lote]
    assert enviados == textos[3:]
    assert cache.faltando(textos) == []


def test_gerar_registra_uma_linha_por_lote_no_livro(cache, cliente_falso, livro):
    cache.gerar([f"t{numero:03d}" for numero in range(150)], cliente=cliente_falso(), livro=livro)
    primeira, segunda = linhas_do_livro(livro)
    assert (primeira["n_textos"], segunda["n_textos"]) == (100, 50)
    for linha, n in ((primeira, 100), (segunda, 50)):
        assert (linha["etapa"], linha["provedor"], linha["modelo"]) == ("embeddings", "openai", MODELO)
        assert linha["tokens_entrada"] == 10 * n
        assert linha["tokens_saida"] == 0
        assert linha["custo_usd"] == pytest.approx(calcular_custo(MODELO, {"tokens_entrada": 10 * n}))
        assert (linha["sucesso"], linha["erro"]) == (True, None)
        assert linha["latencia_ms"] >= 0


def test_gerar_usa_o_livro_global_por_padrao(cache, cliente_falso, livro_temporario):
    cache.gerar(["a"], cliente=cliente_falso())
    [linha] = linhas_do_livro(livro_temporario)
    assert linha["n_textos"] == 1


def test_gerar_sem_ausentes_nao_chama_nada_nem_pede_chave(cache, cliente_falso, livro):
    cache.gerar(["a", "b"], cliente=cliente_falso(), livro=livro)
    antes = len(linhas_do_livro(livro))
    assert cache.gerar(["b", "a"], livro=livro) == 0
    assert len(linhas_do_livro(livro)) == antes


def test_gerar_sem_chave_levanta_chave_ausente_e_nao_grava(cache, livro):
    with pytest.raises(ChaveAusenteError, match="OPENAI_API_KEY"):
        cache.gerar(["x"], livro=livro)
    assert cache.faltando(["x"]) == ["x"]
    assert linhas_do_livro(livro) == []


def test_gerar_com_chave_cria_o_cliente_do_sdk_da_openai(cache, cliente_falso, monkeypatch, livro):
    criados = []

    def openai_falso(api_key):
        criados.append(api_key)
        return cliente_falso()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai_falso))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    assert cache.gerar(["a", "b"], livro=livro) == 2
    assert criados == ["sk-teste"]
    assert cache.faltando(["a", "b"]) == []


def test_falha_do_provedor_registra_linha_sem_sucesso_e_sobe(cache, cliente_falso, livro):
    textos = [f"t{numero:03d}" for numero in range(150)]
    with pytest.raises(RuntimeError, match="503"):
        cache.gerar(textos, cliente=cliente_falso(falhar_na_chamada=2), livro=livro)
    primeira, segunda = linhas_do_livro(livro)
    assert primeira["sucesso"] is True
    assert (segunda["sucesso"], segunda["erro"], segunda["n_textos"]) == (False, "RuntimeError: 503 indisponível", 50)
    assert segunda["tokens_entrada"] is None and segunda["custo_usd"] is None
    assert cache.faltando(textos) == textos[100:]


def test_resposta_com_menos_vetores_levanta_erro_e_nao_grava(cache, cliente_falso, livro):
    with pytest.raises(RuntimeError, match=r"1 vetor\(es\) para 2 texto\(s\)"):
        cache.gerar(["a", "b"], cliente=cliente_falso(cortar=True), livro=livro)
    assert cache.faltando(["a", "b"]) == ["a", "b"]
    [linha] = linhas_do_livro(livro)
    assert linha["sucesso"] is False
    assert linha["tokens_entrada"] == 20  # a chamada foi cobrada


def test_resposta_fora_de_ordem_e_reordenada_pelo_indice(cache, vetor_esperado, livro):
    class ClienteInvertido:
        def __init__(self):
            self.embeddings = self

        def create(self, *, model, input):
            itens = [SimpleNamespace(embedding=vetor_esperado(t), index=i) for i, t in enumerate(input)]
            return SimpleNamespace(data=itens[::-1], usage=SimpleNamespace(total_tokens=5))

    cache.gerar(["a", "b", "c"], cliente=ClienteInvertido(), livro=livro)
    for texto in "abc":
        np.testing.assert_allclose(cache.vetor(texto), vetor_esperado(texto), rtol=1e-6)


# --- textos que o sistema lê do cache --------------------------------------------------------------------


def test_textos_do_sistema_cobrem_questoes_temas_exemplos_e_corpus():
    from sistema import dados
    from sistema.arvore import ArvoreTemas

    textos = embeddings.textos_do_sistema()
    assert len(textos) == len(set(textos))
    conjunto = set(textos)
    arvore = ArvoreTemas.carregar()
    enriquecidos = dados.carregar_temas_enriquecidos()
    assert set(dados.carregar_questoes()["texto"]) <= conjunto
    assert {arvore.texto_tema(t, enriquecidos) for t in arvore.candidatos_possiveis()} <= conjunto
    assert {exemplo["texto"] for exemplo in dados.carregar_exemplos()} <= conjunto
    assert {item["texto"] for item in dados.carregar_corpus_v1()} <= conjunto
