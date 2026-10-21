# Classificação de tópicos sem modelo de linguagem na decisão: case Vício, etapa 2

Sistema que classifica 659 questões de Penal, Processo Penal, Legislação Penal Especial e Criminologia nos temas da plataforma da Vício e decide, para cada uma, entre `publicar`, `revisar` e `fora`. **Na hora de classificar, nada chama modelo de linguagem**: a decisão é de um sistema próprio (buscas lexical, vetorial e por dicionário de leis e termos, vizinhos rotulados, regressão logística e um modelo de confiança calibrado na folha-verdade). Modelos de linguagem entraram só antes, para organizar o conhecimento, como o enunciado permite. O enunciado está em `docs/case/`; a primeira etapa do case, com dois modelos votando, está em https://github.com/GilsonB12/case-vicio-ia.

## Resultado

| Entrega | Onde | Resumo |
|---|---|---|
| Decisões | `publicacao.csv` | as 659 questões; nas 559 fora da folha-verdade: 213 publicadas (38%), 318 para revisão, 28 fora |
| Autoavaliação | `previsao.txt` | precisão prevista 0,95 (intervalo 0,85 a 1,00); cobertura 0,38 (0,37 a 0,39) |
| Alternativa | `alternativas/` | a versão sem o corpus da etapa 1: 79 publicadas (14%), precisão prevista 0,96 |
| Relatório | [`relatorio.md`](relatorio.md) | sistema peça por peça, os 100 rótulos, diagnóstico, produção e memo para o professor |
| Código | `src/sistema/` | o sistema; `preparacao/` registra o que foi feito com modelos antes da decisão |

## Como rodar

Pré-requisitos: Git e [uv](https://docs.astral.sh/uv/) (instala o Python 3.12 sozinho). Nenhuma chave de API é necessária para regenerar o resultado.

```bash
git clone https://github.com/GilsonB12/case-vicio-ia-v2.git && cd case-vicio-ia-v2
uv sync
uv run sistema tudo          # regenera publicacao.csv e previsao.txt a partir dos anexos, sem rede
uv run pytest -q             # testes, sem rede
```

`uv run sistema tudo` lê os anexos e os artefatos versionados em `dados/`, valida o sistema na folha-verdade por validação cruzada, escolhe os limiares, treina o modelo final com os 100 rótulos, decide as 659 questões e grava `publicacao.csv`, `previsao.txt` e os intermediários em `dados/intermediarios/`. Tempo: cerca de 2 minutos num notebook comum, sem GPU (a maior parte é a validação cruzada; decidir as 659 questões leva segundos). A variável `SISTEMA_TAREFAS` limita os processos paralelos.

Outros comandos:

| Comando | O que faz | Rede |
|---|---|---|
| `uv run sistema avaliar-folha` | só a validação cruzada na folha, com a grade de limiares e a matriz de erros | não |
| `uv run sistema tudo --sem-corpus` | a versão alternativa, sem o corpus da etapa 1 | não |
| `uv run sistema comparar-fontes` | a mesma validação sem e com o corpus da etapa 1, e o veredito da regra dos 3 pontos | não |
| `uv run sistema dicionario` | reconstrói `dados/dicionario_leis.json` a partir dos temas enriquecidos e dos ajustes | não |
| `uv run sistema embeddings` | gera os vetores que faltarem no cache (OpenAI `text-embedding-3-large`); a única etapa que chama API, e só é preciso se um texto mudar | sim, `OPENAI_API_KEY` no `.env` |
| `uv run sistema custos` | resume o livro-razão de chamadas de API | não |

## De onde veio cada artefato

| Artefato | Origem | Modelo de linguagem? |
|---|---|---|
| `anexos/` | enviados pela Vício, intocados | não |
| `dados/temas_enriquecidos.jsonl` | para cada um dos 351 temas originais: resumo, leis, artigos, termos técnicos, sinônimos e 3 exemplos de enunciado; gerado em lotes, consolidado por `preparacao/consolidar_enriquecimento.py` e revisado em três camadas (`preparacao/README.md`) | sim, antes da decisão: Codex CLI com gpt-6-astra (lotes 1 a 9) e Claude Opus 5.5 (lotes 10 a 19), por assinatura |
| `dados/exemplos.jsonl` | os 3 exemplos de cada tema (1.053) mais 60 questões de outras disciplinas para o detector de fora | sim, antes da decisão (os de outras disciplinas por Claude Fable 5.1) |
| `dados/dicionario_leis.json` | derivado dos temas enriquecidos por `uv run sistema dicionario` | não |
| `dados/dicionario_leis_ajustes.json`, `dados/termos_fora.json` | escritos à mão; os ajustes registram convenções que a folha-verdade mostrou (seção "Decisões tomadas à mão") | não |
| `dados/embeddings/vetores.npz` | 2.475 vetores `text-embedding-3-large` de questões, temas e exemplos (18 MB), gerados por `uv run sistema embeddings` e empacotados por `preparacao/empacotar_embeddings.py`; custo de US$ 0,08 no livro-razão `dados/custos/chamadas.jsonl` | não é modelo de linguagem; é a única chamada de API |
| `dados/fontes/corpus_v1.jsonl` | 332 questões reais com tema conhecido, coletadas na etapa 1; entra como vizinho rotulado com metade do peso | coletadas na etapa 1 por agentes de pesquisa |

## Decisões tomadas à mão

- **Duplicados publicáveis.** Os 20 temas duplicados (6716 a 6738) podem ser publicados, porque a folha-verdade os usa; para o texto do tema, o duplicado herda o enriquecimento do original.
- **Ajustes do dicionário de leis** (`dados/dicionario_leis_ajustes.json`): Lei 9.099 aponta para Juizado Especial Criminal antes de Procedimento Sumaríssimo, e a Lei de Execução Penal também para o duplicado em Processo Penal, como a folha mostrou; Código Penal e CPP genéricos não apontam tema; códigos militar, tributário e trabalhista marcam fora; uma lei citada no enriquecimento sem confirmação foi neutralizada.
- **Limiar.** Escolhido pela escada, que nunca aceita, para publicar, precisão prevista abaixo de 0,95; ficou em confiança 0,95 (degrau 3: a régua, com pelo menos 10 publicações efetivas).
- **Corpus da etapa 1 dentro, contra a regra fixada antes de medir.** A regra (só entra se aumentar a cobertura em 3 pontos sem baixar precisão nem limite inferior) o rejeitava porque a precisão nas dobras caía de 0,957 para 0,951, menos de uma questão efetiva, enquanto a cobertura subia de 18% para 29% e o limite inferior de 0,834 para 0,849, sem nenhuma questão quase igual a uma do corpus. Decidi usar o corpus (`pipeline.USAR_CORPUS_NA_ENTREGA`); `uv run sistema tudo --sem-corpus` gera a alternativa, guardada em `alternativas/`.
- **Dicionário de termos por tema descartado** depois de testado: subiu a cobertura, mas nenhum limiar chegou a 95%.
- Detalhes, com os números de cada decisão: `docs/projeto/notes/2026-10-21-decisoes-depois-da-spec.md`.

## Documentos

- Relatório e memo: [`relatorio.md`](relatorio.md)
- Contexto e diferenças para a etapa 1: `docs/projeto/contexto.md`
- Spec: `docs/projeto/specs/2026-10-20-sistema-v2-design.md`; decisões tomadas depois dela, com números: `docs/projeto/notes/2026-10-21-decisoes-depois-da-spec.md`
- Preparação com modelos de linguagem: `preparacao/README.md`
