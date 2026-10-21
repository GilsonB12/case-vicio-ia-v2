import dataclasses
import json

import numpy as np
import pytest

from sistema import contexto as modulo_contexto
from sistema.config import CAMINHOS
from sistema.contexto import PESO_CORPUS, carregar_contexto
from sistema.dados import carregar_corpus_v1, carregar_exemplos, carregar_questoes
from sistema.embeddings import CacheEmbeddings, EmbeddingAusenteError

QUESTOES = [
    {"id": "Q0001", "enunciado_html": "Tício subtraiu, para si, coisa alheia móvel.<br>Art. 155 do CP.",
     "comando": "Assinale a correta.", "alternativas": {"A": "furto", "B": "roubo"},
     "disciplina_origem": "Direito Penal", "assunto_origem": "Furto"},
    {"id": "Q0002", "enunciado_html": "A Lei 11.340/2006 protege a mulher.", "comando": "",
     "alternativas": {"A": "certo", "B": "errado"}, "disciplina_origem": "Direito Penal - Legislação Especial",
     "assunto_origem": ""},
    {"id": "Q0003", "enunciado_html": "Crime militar de deserção.", "comando": "", "alternativas": {},
     "disciplina_origem": "Direito Penal Militar", "assunto_origem": ""},
]
FOLHA = (
    "id,tema_central_id,tema_central_caminho,extras_ids,fora_da_taxonomia\n"
    "Q0001,5159,Penal > Dos Crimes Contra o Patrimônio > Furto,5157,0\n"
    "Q0003,,,,1\n"
)
EXEMPLOS = [
    {"id": "S0001", "tema_id": 5160, "fora": False, "texto": "Exemplo sintético de roubo.", "fonte": "teste"},
    {"id": "S0002", "tema_id": None, "fora": True, "texto": "Exemplo de direito tributário.", "fonte": "teste"},
]
CORPUS = [{"id": "C0001", "tema_id": 5166, "enunciado": "Estelionato com fraude bancária.", "comando": "",
           "alternativas": {"A": "sim"}}]
AJUSTES = {"_formato": "teste", "lei:11340": {"nome": "Lei Maria da Penha", "temas": {"4704": 1.0}, "fora": False}}


def escrever_jsonl(caminho, linhas):
    caminho.write_text("".join(json.dumps(linha, ensure_ascii=False) + "\n" for linha in linhas), encoding="utf-8")


@pytest.fixture
def caminhos(tmp_path, monkeypatch):
    """Dados mínimos numa pasta temporária; a árvore é a real (anexos/temas.csv, offline)."""
    anexos, dados = tmp_path / "anexos", tmp_path / "dados"
    (dados / "fontes").mkdir(parents=True)
    anexos.mkdir()
    novos = dataclasses.replace(
        CAMINHOS,
        raiz=tmp_path, anexos=anexos, questoes=anexos / "questoes.jsonl", folha=anexos / "folha_verdade.csv",
        dados=dados, temas_enriquecidos=dados / "temas_enriquecidos.jsonl", exemplos=dados / "exemplos.jsonl",
        dicionario_leis=dados / "dicionario_leis.json", dicionario_ajustes=dados / "dicionario_leis_ajustes.json",
        termos_fora=dados / "termos_fora.json", corpus_v1=dados / "fontes" / "corpus_v1.jsonl",
        embeddings=dados / "embeddings", intermediarios=dados / "intermediarios", custos=dados / "custos",
        publicacao=tmp_path / "publicacao.csv", previsao=tmp_path / "previsao.txt",
    )
    escrever_jsonl(novos.questoes, QUESTOES)
    novos.folha.write_text(FOLHA, encoding="utf-8")
    escrever_jsonl(novos.exemplos, EXEMPLOS)
    escrever_jsonl(novos.corpus_v1, CORPUS)
    novos.dicionario_ajustes.write_text(json.dumps(AJUSTES), encoding="utf-8")
    novos.termos_fora.write_text(json.dumps({"_nota": "teste", "termos": ["militar", "direito tributário"]}),
                                 encoding="utf-8")
    monkeypatch.setattr(modulo_contexto, "CAMINHOS", novos)
    return novos


def textos_esperados(caminhos, com_corpus=False):
    """Textos que o contexto precisa do cache: temas, questões, exemplos e, com corpus, o corpus."""
    from sistema.arvore import ArvoreTemas

    arvore = ArvoreTemas.carregar()
    temas = [arvore.texto_tema(t, {}) for t in arvore.candidatos_possiveis()]
    itens = list(carregar_questoes(caminhos.questoes)["texto"])
    itens += [e["texto"] for e in carregar_exemplos(caminhos.exemplos)]
    corpus = [c["texto"] for c in carregar_corpus_v1(caminhos.corpus_v1)]
    return temas, itens, corpus


def preencher_cache(caminhos, textos, cliente_falso, livro):
    CacheEmbeddings(pasta=caminhos.embeddings).gerar(textos, cliente=cliente_falso(), livro=livro)


def test_cache_vazio_levanta_erro_com_todos_os_textos_que_faltam(caminhos):
    temas, itens, _ = textos_esperados(caminhos)
    with pytest.raises(EmbeddingAusenteError) as erro:
        carregar_contexto()
    faltando = erro.value.textos_faltando
    assert len(temas) == 371
    assert set(faltando) == set(temas) | set(itens)
    assert "sistema embeddings" in str(erro.value)


def test_corpus_so_e_exigido_quando_pedido(caminhos, cliente_falso, livro_temporario):
    temas, itens, corpus = textos_esperados(caminhos)
    preencher_cache(caminhos, temas + itens, cliente_falso, livro_temporario)
    assert carregar_contexto().corpus == []
    with pytest.raises(EmbeddingAusenteError) as erro:
        carregar_contexto(com_corpus=True)
    assert erro.value.textos_faltando == corpus


def test_contexto_carregado_do_cache(caminhos, cliente_falso, livro_temporario, vetor_esperado):
    temas, itens, corpus = textos_esperados(caminhos)
    preencher_cache(caminhos, temas + itens + corpus, cliente_falso, livro_temporario)
    ctx = carregar_contexto(com_corpus=True)
    assert ctx.temas_ids == ctx.arvore.candidatos_possiveis()
    assert ctx.textos_temas == temas
    assert ctx.matriz_temas.shape == (371, 8)
    np.testing.assert_allclose(ctx.matriz_temas[0], vetor_esperado(temas[0]), rtol=1e-6)
    np.testing.assert_allclose(ctx.vetor("Q0002"), vetor_esperado(ctx.texto("Q0002")), rtol=1e-6)
    np.testing.assert_allclose(ctx.vetor("C0001"), vetor_esperado(corpus[0]), rtol=1e-6)
    assert ctx.indice_bm25.ids == ctx.temas_ids and len(ctx.indice_temas) == 371
    assert len(ctx.questoes) == 3 and len(ctx.folha) == 2 and ctx.ids_folha == ["Q0001", "Q0003"]
    assert ctx.termos_fora == ["militar", "direito tributário"]
    # sem dicionario_leis.json, o dicionário sai dos ajustes (e das chaves fora por padrão)
    assert ctx.dicionario["lei:11340"]["temas"] == {4704: 1.0}
    assert ctx.dicionario["diploma:cpm"]["fora"] is True
    assert ctx.chaves("Q0001")["art:155:cp"] == 1
    assert ctx.disciplina_origem("Q0001") == "Direito Penal" and ctx.disciplina_origem("S0001") == ""
    assert ctx.item("C0001")["origem"] == "corpus" and ctx.item("C0001")["tema_id"] == 5166
    assert ctx.cache is not None and ctx.com_corpus


def test_dicionario_lido_do_arquivo_quando_existe(caminhos, cliente_falso, livro_temporario):
    temas, itens, _ = textos_esperados(caminhos)
    preencher_cache(caminhos, temas + itens, cliente_falso, livro_temporario)
    caminhos.dicionario_leis.write_text(json.dumps({
        "_fonte": {"teste": True},
        "art:155:cp": {"nome": "CP, art. 155", "temas": {"5159": 1.0}, "fora": False},
    }), encoding="utf-8")
    assert carregar_contexto().dicionario == {"art:155:cp": {"nome": "CP, art. 155", "temas": {5159: 1.0},
                                                             "fora": False}}


def test_indice_rotulado(caminhos, cliente_falso, livro_temporario):
    temas, itens, corpus = textos_esperados(caminhos)
    preencher_cache(caminhos, temas + itens + corpus, cliente_falso, livro_temporario)
    ctx = carregar_contexto(com_corpus=True)

    indice, pesos = ctx.indice_rotulado({"Q0001", "Q0003"}, com_corpus=False)
    assert indice.ids == ["Q0001", "Q0003", "S0001", "S0002"]
    assert indice.rotulos == [5159, None, 5160, None]
    assert pesos == {"Q0001": 1.0, "Q0003": 1.0, "S0001": 1.0, "S0002": 1.0}

    indice, pesos = ctx.indice_rotulado({"Q0003"}, com_corpus=True)
    assert indice.ids == ["Q0003", "S0001", "S0002", "C0001"]
    assert indice.rotulos == [None, 5160, None, 5166]
    assert pesos["C0001"] == PESO_CORPUS == 0.5
    # a questão de teste não entra no índice da rodada
    assert "Q0001" not in indice.ids
    vizinho = indice.buscar(ctx.vetor("C0001"), 1)[0]
    assert vizinho[0] == "C0001" and vizinho[2] == pytest.approx(1.0)

    with pytest.raises(ValueError):
        ctx.indice_rotulado({"Q0002"})  # fora da folha
    sem_corpus = carregar_contexto(com_corpus=False)
    with pytest.raises(ValueError):
        sem_corpus.indice_rotulado({"Q0001"}, com_corpus=True)


def test_modulos_do_caminho_de_tudo_nao_importam_cliente_de_modelo():
    import subprocess
    import sys

    codigo = (
        "import sys\n"
        "import sistema.contexto, sistema.candidatos, sistema.caracteristicas, sistema.indice_bm25, sistema.vizinhos\n"
        "proibidos = [m for m in sys.modules if m.split('.')[0] in ('openai', 'anthropic', 'tiktoken')]\n"
        "print(proibidos)\n"
        "sys.exit(1 if proibidos else 0)\n"
    )
    resultado = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
