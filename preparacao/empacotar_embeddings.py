"""Empacota os embeddings usados pelo sistema num único arquivo versionável.

O cache gerado por `uv run sistema embeddings` tem um JSON por texto (cerca de 60 KB cada, 166 MB no total, inclusive
vetores de versões antigas dos textos dos temas). Para o repositório, só os vetores dos textos que o sistema usa hoje
vão para `dados/embeddings/vetores.npz`: uma matriz float32 e a lista dos hashes sha256 dos textos, na mesma ordem.
`CacheEmbeddings` lê o pacote primeiro e a pasta de JSON depois, então regenerar o resultado não precisa de rede nem
da pasta de JSON.

Uso: uv run python preparacao/empacotar_embeddings.py
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from sistema.config import CAMINHOS  # noqa: E402
from sistema.embeddings import ARQUIVO_PACOTE, CacheEmbeddings, textos_do_sistema  # noqa: E402


def main() -> int:
    textos = textos_do_sistema()
    cache = CacheEmbeddings()
    faltando = cache.faltando(textos)
    if faltando:
        print(f"{len(faltando)} texto(s) sem vetor no cache; rode `uv run sistema embeddings` antes.")
        return 1
    matriz = cache.matriz(textos).astype(np.float32)
    hashes = np.array([hashlib.sha256(t.encode("utf-8")).hexdigest() for t in textos])
    destino = CAMINHOS.embeddings / ARQUIVO_PACOTE
    np.savez_compressed(destino, hashes=hashes, matriz=matriz)
    print(f"{len(textos)} vetores de dimensão {matriz.shape[1]} gravados em {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
