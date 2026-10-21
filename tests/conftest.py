"""Configuração comum dos testes: nenhum teste escreve em dados/custos nem chega à rede."""
import hashlib
from types import SimpleNamespace

import pytest

from sistema import custos


@pytest.fixture(autouse=True)
def livro_temporario(monkeypatch, tmp_path_factory):
    """O livro-razão global grava numa pasta temporária: nenhum teste escreve em dados/custos."""
    monkeypatch.setattr(custos.LIVRO, "caminho", tmp_path_factory.mktemp("custos") / "chamadas.jsonl")
    return custos.LIVRO


@pytest.fixture(autouse=True)
def sem_chave_de_api(monkeypatch):
    """Sem chave, uma chamada esquecida à API falha com ChaveAusenteError em vez de ir à rede."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def vetor_falso(texto: str, dimensao: int = 8) -> list[float]:
    """Vetor determinístico por texto, tirado do hash."""
    return [byte / 255 for byte in hashlib.sha256(texto.encode("utf-8")).digest()[:dimensao]]


class ClienteEmbeddingsFalso:
    """Dublê de `openai.OpenAI` no que o cache usa: `cliente.embeddings.create(model=..., input=[...])`.

    Devolve `.data[i].embedding` e `.usage.total_tokens` (10 por texto). `falhar_na_chamada` (1, 2, …) levanta
    erro naquela chamada; `cortar` devolve um vetor a menos que os textos.
    """

    TOKENS_POR_TEXTO = 10

    def __init__(self, falhar_na_chamada: int | None = None, cortar: bool = False) -> None:
        self.lotes: list[tuple[str, list[str]]] = []
        self.falhar_na_chamada = falhar_na_chamada
        self.cortar = cortar
        self.embeddings = self

    def create(self, *, model: str, input: list[str]):
        self.lotes.append((model, list(input)))
        if self.falhar_na_chamada == len(self.lotes):
            raise RuntimeError("503 indisponível")
        textos = list(input)[:-1] if self.cortar else list(input)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=vetor_falso(texto)) for texto in textos],
            usage=SimpleNamespace(total_tokens=self.TOKENS_POR_TEXTO * len(input)),
        )


@pytest.fixture
def cliente_falso():
    """Fábrica do dublê do cliente de embeddings."""
    return ClienteEmbeddingsFalso


@pytest.fixture
def vetor_esperado():
    return vetor_falso
