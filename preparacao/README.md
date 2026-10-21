# Preparação do conhecimento (a única fase que usa modelos de linguagem)

Tudo o que está em `dados/` e foi produzido com modelo de linguagem passou por aqui, antes de qualquer decisão, como o enunciado permite: enriquecer a descrição dos temas, propor termos e exemplos que são revisados e guardados como dado, e gerar embeddings em cache. Nenhum modelo escolheu, votou ou validou o tema de uma questão avaliada. O comando de regeneração (`uv run sistema tudo`) não chama nada disto.

## Rodadas

| Artefato | Como foi feito | Modelo | Data | Custo |
|---|---|---|---|---|
| `dados/temas_enriquecidos.jsonl` | 18 lotes de 20 temas com o prompt `prompts/enriquecer_temas.md`; lotes 1 a 9 pelo Codex CLI (gpt-6-astra, esforço médio, sandbox só de leitura, prompt pela entrada padrão); lotes 10 a 18 por Claude Opus 5.5 via Claude Code (agentes que leram o mesmo prompt e gravaram o JSONL); consolidação e validação por `consolidar_enriquecimento.py`; temas sem linha válida refeitos numa rodada de reparo | Codex gpt-6-astra; Claude Opus 5.5 | 20/10/2026 | assinatura, sem API |
| `dados/exemplos.jsonl` (sintéticos por tema) | os 3 exemplos de enunciado de cada tema enriquecido, extraídos pelo mesmo script | idem | 20/10/2026 | idem |
| `dados/exemplos.jsonl` (fora da taxonomia) | 60 questões de outras disciplinas (constitucional, administrativo, civil, tributário, trabalho, penal e processo penal militar, consumidor, empresarial) geradas por um agente com o prompt `prompts/exemplos_fora.md` | Claude Fable 5.1 | 20/10/2026 | idem |
| `dados/dicionario_leis.json` | `uv run sistema dicionario`: derivado dos temas enriquecidos (peso 1/n por tema que cita a chave) mais `dados/dicionario_leis_ajustes.json`, escrito à mão a partir do que a folha-verdade ensinou | sem modelo | | |
| `dados/termos_fora.json` | lista escrita à mão, sem termos de temas que existem na árvore | sem modelo | | |
| `dados/embeddings/` | `uv run sistema embeddings`: um vetor por texto (659 questões, 371 temas enriquecidos, exemplos, corpus opcional), modelo `text-embedding-3-large`, cada lote registrado em `dados/custos/chamadas.jsonl` | OpenAI | | ver `dados/custos/custos_resumo.json` |

## Revisão

O enriquecimento é dado gerado por modelo e foi tratado como tal: validação automática de formato (JSON, 7 chaves, 3 exemplos, pelo menos 8 termos), checagem automática de consistência (leis citadas em cada tema confrontadas com as leis extraídas das questões que a folha atribui ao tema; divergências listadas em `dados/intermediarios/revisao_enriquecimento.json`) e leitura por amostra de 20 temas, registrada abaixo. Os ajustes feitos à mão estão em `dados/dicionario_leis_ajustes.json` e explicados no README principal.

(Registro da amostra lida: preenchido ao final da preparação.)

## Por que modelos por assinatura

A regra do case permite modelo antes da decisão e pede documentação e versionamento, não um fornecedor específico. Os modelos por assinatura custaram zero de API e deixaram o orçamento de API só para os embeddings. Os prompts estão em `prompts/`, as saídas brutas de cada lote não ficam no repositório (são intermediárias), e os artefatos consolidados em `dados/` são o que o sistema lê.
