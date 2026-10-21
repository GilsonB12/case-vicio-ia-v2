from dataclasses import fields
from pathlib import Path

from sistema import config

RAIZ = Path(__file__).resolve().parents[1]

ESPERADOS = {
    "raiz": RAIZ,
    "anexos": RAIZ / "anexos",
    "questoes": RAIZ / "anexos" / "questoes.jsonl",
    "temas": RAIZ / "anexos" / "temas.csv",
    "folha": RAIZ / "anexos" / "folha_verdade.csv",
    "dados": RAIZ / "dados",
    "temas_enriquecidos": RAIZ / "dados" / "temas_enriquecidos.jsonl",
    "exemplos": RAIZ / "dados" / "exemplos.jsonl",
    "dicionario_leis": RAIZ / "dados" / "dicionario_leis.json",
    "dicionario_ajustes": RAIZ / "dados" / "dicionario_leis_ajustes.json",
    "termos_fora": RAIZ / "dados" / "termos_fora.json",
    "corpus_v1": RAIZ / "dados" / "fontes" / "corpus_v1.jsonl",
    "embeddings": RAIZ / "dados" / "embeddings",
    "intermediarios": RAIZ / "dados" / "intermediarios",
    "custos": RAIZ / "dados" / "custos",
    "publicacao": RAIZ / "publicacao.csv",
    "previsao": RAIZ / "previsao.txt",
}


def test_caminhos_tem_exatamente_os_campos_da_interface():
    assert [campo.name for campo in fields(config.CAMINHOS)] == list(ESPERADOS)


def test_caminhos_apontam_para_anexos_dados_e_raiz():
    for nome, esperado in ESPERADOS.items():
        assert getattr(config.CAMINHOS, nome) == esperado, nome


def test_anexos_existem_onde_a_configuracao_aponta():
    for nome in ("questoes", "temas", "folha", "corpus_v1"):
        assert getattr(config.CAMINHOS, nome).is_file(), nome


def test_semente_fixa():
    assert config.SEMENTE == 42


def test_ler_env_devolve_padrao_e_vazio_conta_como_ausente(monkeypatch):
    monkeypatch.delenv("VARIAVEL_INEXISTENTE", raising=False)
    assert config.ler_env("VARIAVEL_INEXISTENTE") is None
    assert config.ler_env("VARIAVEL_INEXISTENTE", "x") == "x"
    monkeypatch.setenv("VARIAVEL_INEXISTENTE", "   ")
    assert config.ler_env("VARIAVEL_INEXISTENTE", "x") == "x"
    monkeypatch.setenv("VARIAVEL_INEXISTENTE", " y ")
    assert config.ler_env("VARIAVEL_INEXISTENTE", "x") == "y"


def test_chave_ausente_e_erro_de_execucao():
    assert issubclass(config.ChaveAusenteError, RuntimeError)
