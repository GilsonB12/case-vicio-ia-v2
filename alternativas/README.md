# Alternativa: a versão sem o corpus da etapa 1

A entrega oficial é a da raiz (`publicacao.csv` e `previsao.txt`), que usa o corpus de 332 questões reais da etapa 1 como vizinho rotulado. Esta pasta guarda a versão mais conservadora, sem o corpus, para quem quiser comparar.

| Nas 559 questões avaliadas | Oficial (com corpus) | Alternativa (sem corpus) |
|---|---|---|
| Publicadas | 213 (38%) | 79 (14%) |
| Para revisão | 318 | 462 |
| Fora | 28 | 18 |
| Precisão prevista | 0,95 (0,85 a 1,00) | 0,96 (0,83 a 1,00) |
| Acertos na folha-verdade, nas dobras | 85 de 88 | 54 de 55 |

As duas têm chance parecida de passar da régua de 95% (cerca de 60% e 65%, pela mesma folha); a oficial publica quase o triplo. A escolha e o motivo estão no relatório, parte a), e em `docs/projeto/notes/2026-10-21-decisoes-depois-da-spec.md`.

Para regenerar esta versão: `uv run sistema tudo --sem-corpus` (grava na raiz; copie para cá).
