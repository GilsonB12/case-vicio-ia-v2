"""Configuração comum dos testes."""
import pytest

from classificador import custos


@pytest.fixture(autouse=True)
def livro_temporario(monkeypatch, tmp_path_factory):
    """O livro-razão global grava numa pasta temporária: nenhum teste escreve em data/custos."""
    monkeypatch.setattr(custos.LIVRO, "caminho", tmp_path_factory.mktemp("custos") / "chamadas.jsonl")
    return custos.LIVRO
