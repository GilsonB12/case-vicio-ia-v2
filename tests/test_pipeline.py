import dataclasses
import json
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from apoio_contexto import DIMENSAO, EIXOS, dados_maiores
from sistema import config, pipeline
from sistema.arvore import ArvoreTemas
from sistema.config import CAMINHOS
from sistema.dados import carregar_corpus_v1, carregar_exemplos, carregar_questoes
from sistema.embeddings import CacheEmbeddings
from sistema.publicar import COLUNAS_PUBLICACAO, ler_publicacao, validar_publicacao

DIMENSAO_TOTAL = DIMENSAO + 6  # os outros temas da árvore vivem em 6 eixos próprios


class ClienteMapeado:
    """Dublê do cliente de embeddings que devolve o vetor escolhido para cada texto."""

    def __init__(self, vetores):
        self.vetores = vetores
        self.embeddings = self

    def create(self, *, model, input):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(x) for x in self.vetores[texto]]) for texto in input],
            usage=SimpleNamespace(total_tokens=len(input)),
        )


def escrever_jsonl(caminho, linhas):
    caminho.write_text("".join(json.dumps(linha, ensure_ascii=False) + "\n" for linha in linhas), encoding="utf-8")


def _completar(vetor):
    return np.concatenate([np.asarray(vetor, dtype=np.float32), np.zeros(DIMENSAO_TOTAL - DIMENSAO, np.float32)])


@pytest.fixture
def caminhos(tmp_path, monkeypatch, livro_temporario):
    """Anexos e dados pequenos numa pasta temporária, com o cache de embeddings preenchido por um dublê."""
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
    base = dados_maiores()
    escrever_jsonl(novos.questoes, [
        {"id": q["id"], "enunciado_html": q["texto"], "comando": "", "alternativas": {"A": "certo", "B": "errado"},
         "disciplina_origem": q["disciplina_origem"], "assunto_origem": ""}
        for q in base["questoes"]
    ])
    linhas_folha = ["id,tema_central_id,tema_central_caminho,extras_ids,fora_da_taxonomia"]
    for f in base["folha"]:
        central = "" if f["fora"] else str(f["tema_central_id"])
        linhas_folha.append(f"{f['id']},{central},,{';'.join(map(str, f['extras_ids']))},{int(f['fora'])}")
    novos.folha.write_text("\n".join(linhas_folha) + "\n", encoding="utf-8")
    escrever_jsonl(novos.exemplos, [{k: v for k, v in e.items() if k != "vetor"} for e in base["exemplos"]])
    escrever_jsonl(novos.corpus_v1, [{"id": c["id"], "tema_id": c["tema_id"], "enunciado": c["texto"], "comando": "",
                                      "alternativas": {}} for c in base["corpus"]])
    novos.termos_fora.write_text(json.dumps({"termos": ["militar", "código penal militar"]}), encoding="utf-8")

    # vetores por texto, do jeito que o sistema monta cada texto
    vetores = {}
    arvore = ArvoreTemas.carregar()
    gerador = np.random.default_rng(11)
    for tema_id in arvore.candidatos_possiveis():
        texto = arvore.texto_tema(tema_id, {})
        if tema_id in EIXOS:
            vetores[texto] = _completar(np.eye(DIMENSAO)[EIXOS[tema_id]])
        else:
            outro = np.zeros(DIMENSAO_TOTAL, np.float32)
            outro[DIMENSAO:] = gerador.normal(size=DIMENSAO_TOTAL - DIMENSAO)
            outro[:DIMENSAO] = 0.1 * gerador.normal(size=DIMENSAO)
            vetores[texto] = outro
    questoes = carregar_questoes(novos.questoes)
    for texto, q in zip(questoes["texto"], base["questoes"]):
        vetores[texto] = _completar(q["vetor"])
    for e, bruto in zip(carregar_exemplos(novos.exemplos), base["exemplos"]):
        vetores[e["texto"]] = _completar(bruto["vetor"])
    for c, bruto in zip(carregar_corpus_v1(novos.corpus_v1), base["corpus"]):
        vetores[c["texto"]] = _completar(bruto["vetor"])
    CacheEmbeddings(pasta=novos.embeddings).gerar(list(vetores), cliente=ClienteMapeado(vetores),
                                                  livro=livro_temporario)

    monkeypatch.setattr(config, "CAMINHOS", novos)
    return novos


@pytest.fixture
def sem_rede(monkeypatch):
    def proibido(*args, **kwargs):
        raise AssertionError("tentou usar a rede")

    monkeypatch.setattr(socket.socket, "connect", proibido)
    monkeypatch.setattr(CacheEmbeddings, "gerar", proibido)


def test_tudo_ponta_a_ponta_com_comparacao_de_fontes(caminhos, sem_rede, capsys):
    resumo = pipeline.tudo()
    ids = list(carregar_questoes(caminhos.questoes)["id"])

    publicacao = ler_publicacao(caminhos.publicacao)
    assert list(publicacao.columns) == COLUNAS_PUBLICACAO
    assert list(publicacao["id"]) == ids
    assert validar_publicacao(publicacao, ids, ArvoreTemas.carregar()) == []

    linhas = caminhos.previsao.read_text(encoding="utf-8").splitlines()
    assert [linha.split(":")[0] for linha in linhas[:4]] == [
        "precisao_prevista", "precisao_intervalo", "cobertura_prevista", "cobertura_intervalo"]
    assert linhas[0] == f"precisao_prevista: {resumo['precisao']:.2f}"
    assert linhas[2] == f"cobertura_prevista: {resumo['cobertura']:.2f}"

    pasta = caminhos.intermediarios
    for nome in ("candidatos.jsonl", "caracteristicas.csv", "probabilidades.jsonl", "avaliacao_folha.csv",
                 "grade.csv", "limiares.json", "comparacao_fontes.json", "pesos_modelo.json", "resumo.json"):
        assert (pasta / nome).exists(), nome
    comparacao = json.loads((pasta / "comparacao_fontes.json").read_text(encoding="utf-8"))
    assert {"sem_corpus", "com_corpus", "corpus_aprovado", "motivo"} <= set(comparacao)
    assert resumo["com_corpus"] is pipeline.USAR_CORPUS_NA_ENTREGA  # a fonte da entrega é decisão à mão
    assert comparacao["decisao_final_com_corpus"] is pipeline.USAR_CORPUS_NA_ENTREGA
    limiares = json.loads((pasta / "limiares.json").read_text(encoding="utf-8"))
    assert limiares["publicar"] == resumo["limiares"]["publicar"]
    pesos = json.loads((pasta / "pesos_modelo.json").read_text(encoding="utf-8"))
    assert set(pesos) == {"pares", "fora", "confianca"} and "cos_tema" in pesos["pares"]["pesos"]
    assert "votos" in pesos["pares"]["pesos"] and set(pesos["confianca"]["pesos"]) >= {"votos", "votos_ramo", "p"}
    texto = (pasta / "probabilidades.jsonl").read_text(encoding="utf-8")
    probabilidades = [json.loads(linha) for linha in texto.splitlines()]
    assert all(0 <= o["confianca"] <= 1 for linha in probabilidades for o in linha["opcoes"])
    assert resumo["n_efetivo"] == pytest.approx(resumo["publicadas_dobras"] / 3)
    gravado = json.loads((pasta / "resumo.json").read_text(encoding="utf-8"))
    assert gravado["contagens"] == resumo["contagens"] and gravado["tempo_segundos"] >= 0

    # a cobertura prevista é a observada nas questões fora da folha
    folha = {f["id"] for f in dados_maiores()["folha"]}
    avaliadas = publicacao[~publicacao["id"].isin(folha)]
    assert resumo["n_avaliadas"] == len(avaliadas) == 14
    assert resumo["cobertura"] == pytest.approx((avaliadas["decisao"] == "publicar").mean())
    assert sum(resumo["contagens"].values()) == len(ids)

    saida = capsys.readouterr().out
    assert "Tempo total" in saida and "degrau" in saida


def test_tudo_com_a_fonte_escolhida_na_linha_de_comando(caminhos, sem_rede):
    resumo = pipeline.tudo(com_corpus=False, imprimir=False)
    assert resumo["com_corpus"] is False and resumo["comparacao"] is None
    comparacao = json.loads((caminhos.intermediarios / "comparacao_fontes.json").read_text(encoding="utf-8"))
    assert comparacao["executada"] is False
    texto = caminhos.previsao.read_text(encoding="utf-8")
    assert "não rodou" in texto


def test_tudo_e_deterministico(caminhos, sem_rede):
    pipeline.tudo(com_corpus=True, imprimir=False)
    primeira = caminhos.publicacao.read_text(encoding="utf-8")
    pipeline.tudo(com_corpus=True, imprimir=False)
    assert caminhos.publicacao.read_text(encoding="utf-8") == primeira


def test_avaliar_folha_e_comparar_imprimem(caminhos, sem_rede, capsys):
    validacao = pipeline.avaliar_folha(com_corpus=False)
    saida = capsys.readouterr().out
    assert "Grade no limiar de fora" in saida and "irmão" in saida and "fora publicada" in saida
    assert len(validacao["dobras"]) == 15
    comparacao = pipeline.comparar()
    saida = capsys.readouterr().out
    assert "A sem corpus" in saida and "B com corpus" in saida and "Corpus" in saida
    assert isinstance(comparacao["corpus_aprovado"], bool)


def test_modulos_de_tudo_nao_importam_cliente_de_modelo():
    codigo = (
        "import sys\n"
        "import sistema.pipeline, sistema.modelo, sistema.calibracao, sistema.decisao, sistema.publicar\n"
        "proibidos = [m for m in sys.modules if m.split('.')[0] in ('openai', 'anthropic', 'tiktoken')]\n"
        "print(proibidos)\n"
        "sys.exit(1 if proibidos else 0)\n"
    )
    resultado = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
