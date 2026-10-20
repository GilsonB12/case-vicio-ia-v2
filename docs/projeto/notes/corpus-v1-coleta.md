# Protocolo de coleta do corpus de referência

O corpus de referência é um conjunto de questões reais de concurso, com tema conhecido, usado em duas coisas: o voto por vizinhança do pipeline e a estimativa de precisão (seção 10 da spec). Ele não é coletado por código do repositório: agentes de pesquisa baseados em LLM, com acesso a busca e leitura de páginas, recebem lotes de temas e devolvem questões em JSONL. Este documento registra o protocolo, o prompt e os critérios, porque a coleta é uma decisão manual do projeto.

## Dimensionamento e orçamento

A primeira passada de candidatos apontou 341 temas alvo (`data/intermediarios/temas_alvo_corpus.json`), quase a árvore inteira. O plano original (1.460 questões, 2 a 6 por tema conforme o peso) foi reduzido por orçamento de API: medida numa questão real, uma chamada ao modelo principal custa cerca de US$ 0,008, e o orçamento disponível era de US$ 10, que precisa cobrir as 359 questões e o corpus. Decisão: corpus de cerca de 400 questões, concentrado nos 100 temas de maior peso, isto é, os que mais aparecem entre os candidatos das 359 (eles respondem por 64% da massa de candidatos), com 4 questões por tema, em 10 lotes (`data/coleta/lotes-v2.json`). A calibração é por célula (regra × faixa de confiança), não por tema, então o corpus não precisa cobrir todos os temas; precisa parecer com as 359 em dificuldade e distribuição.

## O que o piloto ensinou

Um lote piloto com 5 temas de Criminologia (`data/coleta/lote-piloto.json`) encontrou 16 das 22 questões pedidas em 25 minutos e mudou o protocolo:

- A leitura de página por ferramenta de resumo não serve para copiar texto: ela parafraseia e corta. O agente precisa baixar a página com `curl` e converter HTML ou PDF em texto (`pdftotext`), e só então copiar.
- Fontes que funcionam: Estude Grátis (público, mostra banca, ano, cargo e gabarito; listagens por matéria, banca e ano com 10 questões por página), banco público do Estratégia em `cj.estrategia.com/public/questoes` (descartar simulados), PDFs oficiais do Cebraspe em `cdn.cebraspe.org.br` e cadernos em portais `.gov.br`, PDFs de provas comentadas do Estratégia.
- Fontes bloqueadas para automação: QConcursos e Gran Questões (anti-bot), TecConcursos (login), blog e portal do Estratégia (anti-bot), PCI Concursos (captcha). Nenhum bloqueio foi contornado.
- Temas pai ficam sem questões se a regra é "sempre o mais específico": a regra passou a ser a abaixo.
- Temas raros na história das provas (por exemplo a fase pré-científica da Criminologia) não rendem; a meta não é forçada.

## Regra para temas pai e filho

Para um tema de nível 1, aceitam-se questões sobre o assunto em geral ou que cobrem vários de seus subtemas. Uma questão claramente sobre um subtema é rotulada com o subtema quando ele está no mesmo lote; caso contrário, com o pai. A régua do cliente conta pai e filho como acerto, então essa rotulagem é coerente com a avaliação.

## Níveis de confiança da fonte

| Nível | Fontes |
|---|---|
| alta | PDFs oficiais de banca ou de órgão (`cdn.cebraspe.org.br`, portais `.gov.br`, FGV, FCC, Vunesp, IBFC, AOCP) e entradas do Estude Grátis com banca e ano identificados |
| media | Banco público do Estratégia (provas, não simulados), PDFs comentados do Estratégia, e qualquer questão sem banca ou ano identificável |
| baixa | Blogs, fóruns e agregadores |

## Formato de saída

Um objeto JSON por linha, em `data/coleta/lote-NN.jsonl`:

```json
{"tema_id": 5159, "enunciado": "...", "comando": "...", "alternativas": {"A": "...", "B": "..."}, "url": "https://...", "site": "dominio.com.br", "banca_ano": "FGV 2023", "confianca_fonte": "alta", "gabarito": "B"}
```

Regras: `enunciado` e `alternativas` copiados da página, sem comentários nem gabarito comentado; questões Certo/Errado usam `{"C": "Certo", "E": "Errado"}`; `comando` pode ficar vazio; `gabarito` é a letra quando a página traz, senão vazio; `tema_id` é um dos temas do lote; uma questão aparece uma vez só. Descartam-se simulados, questões anuladas, questões sem alternativas, questões de outras matérias e questões cujo texto integral não foi lido.

## Prompt dos agentes

> Você é um pesquisador que monta um corpus de questões reais de concurso público brasileiro para calibrar um classificador de temas de Direito Penal, Processo Penal, Legislação Penal Especial e Criminologia. Para cada tema do seu lote, encontre em páginas públicas questões reais de concurso que sejam claramente sobre aquele tema, na quantidade pedida. Busque na web com o título do tema e termos como "questão", "concurso", "prova", "banca", restringindo às fontes que funcionam. Baixe a página com curl e converta em texto antes de copiar; copie o enunciado e as alternativas exatamente como estão, sem comentários nem gabarito comentado. Registre URL, domínio, banca e ano quando aparecerem, e o nível de confiança pelos critérios dados. Não invente, não parafraseie, não contorne bloqueios nem use sites que exigem login. Se a questão serve para mais de um tema do lote, use o mais específico, uma vez só; para temas pai, siga a regra de pai e filho. Limite de esforço por tema: 5 buscas e 6 páginas, contando uma página de listagem como uma. Se não alcançar a meta, entregue o que achou e diga quantas faltaram. Grave em JSONL no caminho indicado, valide cada linha e devolva um resumo por tema com contagens, fontes e consultas que funcionaram.

## Pós-processamento

1. Validação de formato e de `tema_id` (publicável e do lote).
2. Deduplicação interna por texto normalizado e por similaridade de embedding acima de 0,95.
3. Deduplicação contra as 359 por similaridade de embedding acima de 0,95, para nenhuma questão do case vazar para o corpus.
4. Remoção de questões sem alternativas ou com enunciado menor que 40 caracteres.
5. Gravação em `data/corpus_referencia.jsonl` com `id` sequencial `C0001...`, mantendo `url`, `site`, `banca_ano`, `gabarito` e `confianca_fonte` como proveniência. O piloto entra no corpus.

## Limitações conhecidas

Os rótulos do corpus vêm da classificação do site de origem ou da leitura do agente, não da equipe da Vício. Questões achadas por busca pelo nome do tema tendem a ser as mais fáceis daquele tema. O corpus cobre só os 100 temas de maior peso. As três coisas puxam a precisão medida no corpus para cima ou a deixam menos representativa; a reponderação por célula e a âncora manual mitigam, e o relatório declara.

## Checagem independente dos rótulos (17/10/2026)

Para medir quanto confiar nos rótulos do corpus, um rotulador independente (Codex CLI com o modelo gpt-6-astra (esforço de raciocínio alto, por assinatura), distinto dos modelos do pipeline e dos agentes de coleta; prompt em `docs/projeto/prompts/checagem-corpus.md`) classificou às cegas, com a árvore inteira e sem ver o rótulo do agente, uma amostra aleatória de 60 questões do corpus (semente 42). Resultado em `data/corpus_amostra_rotulador_independente.jsonl`: 36 idênticas (60%), 53 compatíveis pela régua do cliente, igual, pai ou filho (88%), 7 divergentes. Das 7, quatro são pares de nós que cobrem o mesmo assunto em ramos diferentes da árvore (por exemplo "Sentença" dentro de Procedimento Ordinário e "Sentença Penal" como tema próprio; "Conceitos e Espécies de Prisão" e "Prisões em Espécie"), duas parecem erro do rótulo do corpus (uma questão sobre exame de corpo de delito rotulada como Confissão; uma sobre o recurso cabível contra a pronúncia rotulada como Apelação) e uma é diferença de subtema em Criminologia.

Leitura: o rótulo do corpus concorda com um leitor independente em 88% pela régua, e o resto se concentra na ambiguidade da própria árvore. A precisão medida no corpus (98% nas células publicadas) é mais alta que isso porque as publicadas são as questões em que dois modelos concordaram com confiança alta, isto é, as claras; as ambíguas vão para revisão. Ainda assim, uma parte do que a régua da Vício contaria como erro depende da convenção da equipe para esses nós sobrepostos, e o memo do professor pede essa convenção.
