# Preparação do conhecimento (a única fase que usa modelos de linguagem)

Tudo o que está em `dados/` e foi produzido com modelo de linguagem passou por aqui, antes de qualquer decisão, como o enunciado permite: enriquecer a descrição dos temas, propor termos e exemplos que são revisados e guardados como dado, e gerar embeddings em cache. Nenhum modelo escolheu, votou ou validou o tema de uma questão avaliada. O comando de regeneração (`uv run sistema tudo`) não chama nada disto.

## Rodadas

| Artefato | Como foi feito | Modelo | Data | Custo |
|---|---|---|---|---|
| `dados/temas_enriquecidos.jsonl` | 18 lotes de 20 temas com o prompt `prompts/enriquecer_temas.md` (o último com 11) e um lote de reparo; lotes 1 a 9 pelo Codex CLI (gpt-6-astra, esforço médio, sandbox só de leitura, prompt pela entrada padrão); lotes 10 a 19 por Claude Opus 5.5 via Claude Code (agentes que leram o mesmo prompt e gravaram o JSONL); consolidação e validação por `consolidar_enriquecimento.py` | Codex gpt-6-astra; Claude Opus 5.5 | 20/10/2026 | assinatura, sem API |
| `dados/exemplos.jsonl` (sintéticos por tema) | os 3 exemplos de enunciado de cada tema enriquecido, extraídos pelo mesmo script | idem | 20/10/2026 | idem |
| `dados/exemplos.jsonl` (fora da taxonomia) | 60 questões de outras disciplinas (constitucional, administrativo, civil, tributário, trabalho, penal e processo penal militar, consumidor, empresarial) geradas por um agente com o prompt `prompts/exemplos_fora.md` | Claude Fable 5.1 | 20/10/2026 | idem |
| `dados/dicionario_leis.json` | `uv run sistema dicionario`: derivado dos temas enriquecidos (peso 1/n por tema que cita a chave) mais `dados/dicionario_leis_ajustes.json`, escrito à mão a partir do que a folha-verdade ensinou | sem modelo | | |
| `dados/termos_fora.json` | lista escrita à mão, sem termos de temas que existem na árvore | sem modelo | | |
| `dados/embeddings/vetores.npz` | `uv run sistema embeddings` gera um vetor por texto (659 questões, 371 textos de tema, 1.113 exemplos, 332 do corpus opcional), modelo `text-embedding-3-large`, cada lote registrado em `dados/custos/chamadas.jsonl`; `preparacao/empacotar_embeddings.py` junta os 2.475 vetores usados num arquivo de 18 MB | OpenAI (embeddings, não modelo de linguagem) | 20/10/2026 | US$ 0,08 no total, ver `dados/custos/custos_resumo.json` |

## Revisão

O enriquecimento é dado gerado por modelo e foi tratado como tal, em três camadas:

1. **Formato** (`consolidar_enriquecimento.py`): JSON válido, as 7 chaves, pelo menos 2 exemplos e 8 termos. Os 351 temas passaram; um tema do lote 1 saiu inválido e foi refeito num lote de reparo (lote 19), e o lote 5 veio com cada objeto quebrado em duas linhas, que o leitor passou a aceitar.
2. **Consistência com a folha-verdade, sem modelo** (`checar_enriquecimento.py`): das 14 questões da folha que citam uma lei pelo número, em 11 o tema que a equipe atribuiu (central, extra ou o pai deles) cita a mesma lei no enriquecimento. As 3 divergências são questões que mencionam leis de passagem (um conflito aparente de normas que cita três leis; uma questão de procedimento ordinário que cita a Lei 9.099; uma de prova testemunhal em lei de drogas), não erros do enriquecimento. Saída em `dados/intermediarios/revisao_enriquecimento.json`.
3. **Leitura por amostra**: 12 temas sorteados (semente 7), de Codex e de Opus, lidos inteiros (leis, artigos, termos, exemplos). 11 corretos. No tema Tortura (4707), o enriquecimento cita uma "Lei 15.410/2026" que não foi possível confirmar; nenhuma das 659 questões cita essa lei, e a entrada foi neutralizada em `dados/dicionario_leis_ajustes.json`.

## Por que modelos por assinatura

A regra do case permite modelo antes da decisão e pede documentação e versionamento, não um fornecedor específico. Os modelos por assinatura custaram zero de API e deixaram o orçamento de API só para os embeddings. Os prompts estão em `prompts/`, as saídas brutas de cada lote não ficam no repositório (são intermediárias), e os artefatos consolidados em `dados/` são o que o sistema lê.
