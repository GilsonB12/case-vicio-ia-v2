import json

import pytest

from classificador import embeddings
from classificador.config import ChaveAusenteError
from classificador.custos import LivroRazao, calcular_custo, contexto
from classificador.embeddings import EmbeddingsComCache
from tests.fakes import EmbeddingsFalsas


def test_segunda_chamada_vem_do_cache(tmp_path):
    base = EmbeddingsFalsas()
    emb = EmbeddingsComCache(base, "falso", tmp_path)
    primeira = emb.embed_documents(["a", "b"])
    assert len(base.chamadas) == 2
    segunda = emb.embed_documents(["a", "b"])
    assert len(base.chamadas) == 2
    assert primeira == segunda
    assert emb.acertos_cache == 2


def test_mistura_de_acertos_e_faltas_preserva_ordem(tmp_path):
    base = EmbeddingsFalsas()
    emb = EmbeddingsComCache(base, "falso", tmp_path)
    emb.embed_documents(["b"])
    vetores = emb.embed_documents(["a", "b", "c"])
    assert vetores[0] == base.embed_documents(["a"])[0]
    assert vetores[1] == base.embed_documents(["b"])[0]
    assert vetores[2] == base.embed_documents(["c"])[0]


def test_sem_base_e_sem_cache_levanta_erro_claro(tmp_path):
    emb = EmbeddingsComCache(None, "falso", tmp_path)
    with pytest.raises(ChaveAusenteError, match="OPENAI_API_KEY"):
        emb.embed_query("x")


def test_sem_base_com_cache_funciona(tmp_path):
    EmbeddingsComCache(EmbeddingsFalsas(), "falso", tmp_path).embed_query("x")
    emb = EmbeddingsComCache(None, "falso", tmp_path)
    assert len(emb.embed_query("x")) == 16


def test_cache_corrompido_e_regravado(tmp_path):
    base = EmbeddingsFalsas()
    emb = EmbeddingsComCache(base, "falso", tmp_path)
    emb.embed_query("x")
    arquivo = next((tmp_path / "embeddings" / "falso").glob("*.json"))
    arquivo.write_text("{isso nao e json", encoding="utf-8")
    vetor = emb.embed_query("x")
    assert len(base.chamadas) == 2
    assert json.loads(arquivo.read_text(encoding="utf-8")) == vetor


class BaseQueDevolveMenos(EmbeddingsFalsas):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(texts)[:1]


def test_base_que_devolve_menos_vetores_levanta_erro(tmp_path):
    emb = EmbeddingsComCache(BaseQueDevolveMenos(), "falso", tmp_path)
    with pytest.raises(RuntimeError, match=r"1 vetor\(es\) para 2 texto\(s\)"):
        emb.embed_documents(["a", "b"])
    assert not list((tmp_path / "embeddings").rglob("*.json"))


# --- livro-razão de custos -------------------------------------------------------------------------------


def linhas_do_livro(livro):
    return [json.loads(linha) for linha in livro.caminho.read_text(encoding="utf-8").splitlines()]


def test_lote_real_registra_uma_linha_e_acerto_de_cache_nao_registra(tmp_path):
    livro = LivroRazao(tmp_path / "custos" / "chamadas.jsonl")
    emb = EmbeddingsComCache(EmbeddingsFalsas(), "text-embedding-3-large", tmp_path, livro=livro)
    emb.embed_documents(["Tribunal do Júri", "furto qualificado"])
    emb.embed_documents(["Tribunal do Júri", "furto qualificado"])
    emb.embed_documents(["Tribunal do Júri", "roubo majorado"])
    primeira, segunda = linhas_do_livro(livro)
    assert (primeira["n_textos"], segunda["n_textos"]) == (2, 1)
    assert primeira["tokens_entrada"] > 0
    assert primeira["tokens_saida"] == 0
    assert (primeira["etapa"], primeira["provedor"], primeira["modelo"], primeira["origem"]) == (
        "embeddings", "openai", "text-embedding-3-large", "api")
    assert primeira["custo_usd"] == pytest.approx(
        calcular_custo("text-embedding-3-large", {"tokens_entrada": primeira["tokens_entrada"], "tokens_saida": 0}))
    assert primeira["latencia_ms"] >= 0
    assert (primeira["sucesso"], primeira["erro"], primeira["n_candidatos"]) == (True, None, None)


def test_lote_real_usa_a_etapa_do_contexto(tmp_path):
    livro = LivroRazao(tmp_path / "chamadas.jsonl")
    emb = EmbeddingsComCache(EmbeddingsFalsas(), "text-embedding-3-large", tmp_path, livro=livro)
    with contexto(etapa="reparo", alvo="questoes", id_questao="Q0007"):
        emb.embed_query("porte ilegal de arma de fogo")
    [linha] = linhas_do_livro(livro)
    assert (linha["etapa"], linha["alvo"], linha["id_questao"]) == ("reparo", "questoes", "Q0007")


class BaseQueFalha(EmbeddingsFalsas):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("503 indisponível")


def test_falha_do_provedor_registra_linha_sem_sucesso_e_sobe(tmp_path):
    livro = LivroRazao(tmp_path / "chamadas.jsonl")
    emb = EmbeddingsComCache(BaseQueFalha(), "text-embedding-3-large", tmp_path, livro=livro)
    with pytest.raises(RuntimeError, match="503"):
        emb.embed_documents(["a", "b"])
    [linha] = linhas_do_livro(livro)
    assert (linha["sucesso"], linha["erro"], linha["n_textos"]) == (False, "RuntimeError: 503 indisponível", 2)
    assert linha["tokens_entrada"] is None and linha["custo_usd"] is None


def test_tokenizador_indisponivel_deixa_o_uso_desconhecido(tmp_path, monkeypatch):
    def sem_tokenizador():
        raise OSError("sem rede para baixar o cl100k_base")

    monkeypatch.setattr(embeddings, "_codificador", sem_tokenizador)
    livro = LivroRazao(tmp_path / "chamadas.jsonl")
    emb = EmbeddingsComCache(EmbeddingsFalsas(), "text-embedding-3-large", tmp_path, livro=livro)
    assert len(emb.embed_query("x")) == 16
    [linha] = linhas_do_livro(livro)
    assert (linha["sucesso"], linha["tokens_entrada"], linha["custo_usd"]) == (True, None, None)


def test_contar_tokens_usa_o_tokenizador_dos_modelos_text_embedding_3():
    assert embeddings.contar_tokens(["Tribunal do Júri", "furto"]) == sum(
        len(embeddings._codificador().encode(t)) for t in ["Tribunal do Júri", "furto"])
    assert embeddings._codificador().name == "cl100k_base"
