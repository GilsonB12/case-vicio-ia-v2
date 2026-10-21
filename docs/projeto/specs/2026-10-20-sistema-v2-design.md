# Sistema de classificação sem modelo de linguagem na decisão: spec

Etapa 2 do case Vício. Esta spec é a referência de desenho; o que divergir dela durante a implementação é registrado em `docs/projeto/notes/`.

> As seções 7 a 9 foram superadas pelos dados: a decisão usa um segundo estágio de confiança treinado fora da amostra, o intervalo usa o número efetivo de questões, a escada nunca aceita precisão abaixo da régua, e o dicionário de termos foi testado e descartado. Ver `docs/projeto/notes/2026-10-21-decisoes-depois-da-spec.md`.

## 1. Regras do case que moldam tudo

- 659 questões (Q0001 a Q0359 do banco da Vício; Q0360 a Q0659 de concursos recentes), a mesma árvore de 377 temas, e `folha_verdade.csv` com 100 questões rotuladas pela equipe, todas entre as 359.
- Régua: nas publicadas, pelo menos 95% certas. Conta como certo publicar no tema central, num extra, ou num nó do mesmo ramo (pai ou filho) de um deles. Publicar uma questão que a equipe considera fora da taxonomia é erro. Avaliação nas 559 questões fora da folha.
- Na hora de classificar, **nenhuma chamada a modelo de linguagem**. O comando de regeneração produz `publicacao.csv` a partir dos anexos e dos artefatos versionados, sem rede. Modelo de linguagem só antes, documentado e versionado: embeddings em cache, limpeza, extração de lei e artigo, enriquecimento dos temas, termos e exemplos propostos e revisados. Nunca para escolher, votar ou validar o tema de uma questão avaliada.
- A decisão é do sistema e devolve uma probabilidade calibrável.

## 2. Arquitetura em duas fases

```
PREPARAÇÃO (usa modelos; roda à parte; cada saída versionada em dados/, com data, modelo e custo)
  temas.csv ──► temas_enriquecidos.jsonl ──► dicionario_leis.json
             └► exemplos.jsonl (sintéticos por tema + fora da taxonomia)
  questoes.jsonl + temas enriquecidos + exemplos ──► embeddings/ (cache OpenAI, text-embedding-3-large)

SISTEMA (sem modelo; `uv run sistema tudo`; CPU; menos de 2 minutos)
  limpar texto ─► extrair leis e artigos ─► candidatos (dicionário ∪ BM25 ∪ embedding-tema ∪ kNN)
  ─► características por par (questão, tema) ─► regressão logística ─► probabilidade por tema
  ─► detector de fora ─► validação cruzada na folha ─► limiares ─► decisão ─► publicacao.csv, previsao.txt
```

## 3. Dados e classes

- **Temas publicáveis:** ativos e fora das raízes: 371 (inclui os 20 duplicados 6716 a 6738, porque a equipe os usa: a folha tem um central e um extra em duplicado). A canonização da etapa 1 vira só uma relação `original_de(tema)` usada para copiar o conteúdo enriquecido do original para o duplicado; a decisão pode publicar qualquer um dos 371.
- **Texto da questão:** enunciado limpo de HTML, comando e alternativas concatenados (como na etapa 1). `disciplina_origem` e `assunto_origem` nunca entram no texto; `disciplina_origem` entra como característica fraca (seção 7).
- **Folha-verdade:** `id`, `tema_central_id` (vazio quando fora), `extras_ids`, `fora_da_taxonomia`. Os alvos aceitos de uma questão são: central, extras, e pai ou filho direto de cada um. É a única fonte de avaliação e de calibração.
- **Fontes de conhecimento** (todas em `dados/`, com campo `fonte` dizendo modelo, data e revisão):
  - `temas_enriquecidos.jsonl`: por tema, `resumo`, `leis` (número e nome), `artigos` (com diploma), `termos`, `sinonimos`, `exemplos_enunciado`, `descricao_original`. Texto do tema para BM25 e embedding = caminho + resumo + leis + artigos + termos + sinônimos.
  - `exemplos.jsonl`: exemplos sintéticos por tema (meta: 3 por tema publicável) e exemplos fora da taxonomia (meta: 60, de outras disciplinas: constitucional, civil, tributário, administrativo, trabalhista, militar, consumidor). Campos: `id` (S0001…), `tema_id` (nulo quando fora), `fora`, `texto`, `fonte`.
  - `dicionario_leis.json`: chave normalizada (`lei:11340`, `art:157:cp`, `sumula:v:11`, `diploma:cpm`) → `{nome, temas: {tema_id: peso}, fora: bool}`. Construído por script a partir dos temas enriquecidos (peso = 1 / número de temas que citam) e de `dicionario_leis_ajustes.json`, revisado à mão, que sobrescreve ou acrescenta entradas.
  - `fontes/corpus_v1.jsonl` (opcional, interruptor `--com-corpus`, padrão desligado): 332 questões reais coletadas na etapa 1, rotuladas por busca dirigida por tema e checadas por amostra (88% compatíveis). Só entra como vizinho rotulado, com peso 0,5. Nunca toca a folha nem a calibração.
  - `embeddings/`: cache por hash do texto, modelo `text-embedding-3-large`, gerado por `uv run sistema embeddings` (único comando que chama a API). `tudo` lê o cache e falha com a lista do que falta se algum texto não estiver nele; nunca chama a API.

## 4. Extração de leis e artigos (`leis.py`)

Expressões regulares sobre o texto limpo:

- Lei: `Lei (Complementar )?(n[º°o.]?\s*)?\d{1,2}\.?\d{3}(/\d{2,4})?` → `lei:11340` (sem pontos; o ano não entra na chave). Decreto-Lei idem → `dl:2848`.
- Diplomas por nome e sigla: Código Penal, CP, Código de Processo Penal, CPP, Código Penal Militar, CPM, CPPM, Constituição, CF, CTN, CLT, CDC, ECA, LEP, Lei de Drogas, Lei das Contravenções, Estatuto do Desarmamento, Lei Maria da Penha, Lei de Execução Penal → chaves `diploma:cp`, `diploma:cpp`, … com uma tabela de apelidos em `leis.py` (os apelidos conhecidos de leis especiais apontam também para a chave `lei:` correspondente).
- Artigo: `art(igo)?s?\.?\s*\d+(-[A-Z])?(º|o)?` com o diploma decidido pelo contexto: a menção de diploma mais próxima nas 80 letras seguintes ("do CP", "do CPP", "da Lei 11.340", "da CF") ou, sem menção, o diploma dominante do texto. Chave `art:157:cp`, `art:5:lei:11340`.
- Súmula: `Súmula( Vinculante)? n?º?\s*\d+` → `sumula:stf:11` ou `sumula:v:11`; sem tribunal explícito, `sumula:11`.
- Saída: lista de chaves com contagem. Testes com frases reais das questões.

## 5. Candidatos (`candidatos.py`)

União, por questão, de quatro buscas, mais o pai de todo candidato de nível 2 e os filhos do pai quando o pai veio pelo dicionário:

1. **Dicionário:** temas apontados pelas chaves extraídas, com peso somado.
2. **BM25** (`rank-bm25`, tokens normalizados sem acento, sem stopwords) sobre os textos dos 371 temas: 10 primeiros.
3. **Embedding questão × tema:** cosseno com os 371 textos de tema: 10 primeiros.
4. **kNN rotulado:** cosseno com os exemplos rotulados disponíveis na rodada (folha de treino, sintéticos, corpus se ligado): 15 vizinhos, cada um traz o seu tema.

Teto de 30 candidatos por questão, ordenados por quantas buscas os trouxeram e pela maior pontuação.

## 6. Características por par (questão, tema) (`caracteristicas.py`)

| Nome | Definição |
|---|---|
| `bm25` | pontuação BM25 do tema dividida pela maior pontuação da questão (0 a 1) |
| `cos_tema` | cosseno entre o embedding da questão e o do texto do tema |
| `knn_max`, `knn_media3`, `knn_n10` | maior similaridade a um exemplo rotulado do tema; média das 3 maiores; quantos dos 10 vizinhos mais próximos são do tema |
| `knn_folha_max` | igual a `knn_max`, só com vizinhos da folha de treino |
| `dic_lei`, `dic_art` | soma dos pesos das chaves de lei e de artigo da questão que apontam para o tema |
| `nivel2` | 1 se o tema é subtema |
| `pai_de_candidato` | para tema de nível 1: maior `cos_tema` entre os filhos candidatos; 0 se não tem filho candidato |
| `n_buscas` | quantas das 4 buscas trouxeram o tema (0 a 4) |
| `disciplina_bate` | 1 se `disciplina_origem` normalizada é a disciplina do tema |
| `prior` | log do número de questões do tema na fonte (campo da descrição original), normalizado |

Todas em escala 0 a 1 ou padronizadas no treino; o modelo é linear e a tabela de pesos vai para o relatório.

## 7. Modelo (`modelo.py`)

- **Pares:** regressão logística (scikit-learn, `class_weight="balanced"`, regularização L2 escolhida por validação cruzada interna). Rótulo positivo = o tema é o central ou um extra da questão (não pai nem filho: o pai é tratado na decisão). Exemplos de treino: pares das questões da folha de treino (peso 1) e dos exemplos sintéticos e do corpus, se ligado (peso 0,5).
- **Probabilidade por tema:** a saída do par é normalizada entre os candidatos da questão (`p_t = s_t / Σ s`), para somar 1 e ser comparável entre questões.
- **Agregação pelo pai:** `p_ramo(pai) = p_pai + Σ p_filhos`. Serve para publicar o pai quando os filhos dividem a probabilidade, o que a régua aceita.
- **Detector de fora** (regressão logística separada): características da questão: maior `p_t`, maior `cos_tema`, maior `knn_max`, soma de pesos de chaves `fora` (CPM, CTN, CLT, CF…), contagem de termos de outras disciplinas (lista em `dados/termos_fora.json`), `disciplina_origem` fora das quatro. Treino: folha de treino (8 fora em 100) e exemplos fora da taxonomia (sintéticos, peso 0,5). Saída `p_fora`, calibrada com Platt.
- Sementes fixas; `tudo` é determinístico.

## 8. Validação cruzada e calibração (`calibracao.py`)

- 5 dobras estratificadas por (fora, disciplina do central), repetidas 3 vezes com sementes diferentes: 15 avaliações, cada uma com 80 rótulos de treino e 20 de teste.
- Em cada dobra: índice kNN com folha de treino + sintéticos (+ corpus se ligado); treina o modelo de pares e o detector; nas 20 de teste, para cada limiar `τ` numa grade de 0,30 a 0,95 (passo 0,01) e cada `τ_fora` numa grade: decisão (seção 9), acerto pela régua completa, cobertura.
- Agrupa as 15 dobras: para cada par de limiares, acertos e publicadas somados; precisão suavizada de Jeffreys e intervalo de 95% pela Beta; cobertura média.
- **Escada de seleção** (a mesma da etapa 1): degrau 0, precisão prevista ≥ 0,97 e limite inferior ≥ 0,95, maximizando cobertura; degrau 1, só o limite inferior ≥ 0,95; degrau 2, só a média ≥ 0,97; degrau 3, maior limite inferior. O `previsao.txt` diz o degrau.
- **Comparação de fontes:** a validação roda duas vezes, sem e com o corpus. Regra fixada antes de ver os números: o corpus fica se a cobertura prevista subir pelo menos 3 pontos com precisão prevista e limite inferior não menores que os da versão sem corpus. Resultado em `dados/intermediarios/comparacao_fontes.json` e no relatório.
- Modelo final: treinado com os 100 rótulos (e sintéticos, e corpus se aprovado), com os limiares da validação.
- `previsao.txt`: precisão prevista = precisão agrupada das dobras no limiar escolhido, com o intervalo; cobertura prevista = fração publicada observada nas 559 (intervalo ±0,01), com a cobertura das dobras como conferência.

## 9. Decisão (`decisao.py`) e saídas (`publicar.py`)

Para cada questão, com `p_t` dos candidatos, `p_ramo` e `p_fora`:

1. `p_fora ≥ τ_fora` → `fora`, motivo com as chaves ou termos que dispararam.
2. Tema de maior `p_t` com `p_t ≥ τ` → `publicar` nele.
3. Senão, pai de nível 1 com `p_ramo ≥ τ` → `publicar` no pai (motivo em branco; extras recebem os filhos).
4. Senão → `revisar` com o tema de maior `p_t`, motivo "probabilidade X abaixo do limiar Y" ou "divergência entre ramos".
5. Extras: até 3 outros candidatos com `p_t ≥ τ_extra` (0,15, fixo), excluindo o central e seus ancestrais e descendentes diretos; `confianca` = `p_t` do central (ou `p_ramo` no pai).

`publicacao.csv` com as 659 linhas no formato do LEIA-ME, validado contra ele; `previsao.txt` conforme a seção 8. As 100 da folha recebem decisão como as outras (não entram na avaliação).

## 10. Comando de regeneração e rastreabilidade

- `uv run sistema tudo [--com-corpus]`: limpa, extrai, monta índices, treina, valida, calibra, decide, publica. Sem rede. Grava intermediários em `dados/intermediarios/` (candidatos, características, probabilidades, dobras, limiares, comparação de fontes) e imprime tempo total.
- `uv run sistema embeddings`: a única etapa que chama a API (OpenAI); grava no cache e no livro-razão `dados/custos/chamadas.jsonl` (módulo `custos.py` da etapa 1).
- `uv run sistema dicionario`: reconstrói `dicionario_leis.json` a partir dos temas enriquecidos e dos ajustes.
- `uv run sistema avaliar-folha`: imprime a validação cruzada e a tabela de confusão por tipo de erro (irmão, ramo, fora).
- Os passos de preparação que usam modelos de linguagem (enriquecer temas, gerar exemplos) são procedimentos documentados em `preparacao/README.md`, com os prompts em `preparacao/prompts/`, o modelo e a data de cada rodada e a revisão feita; as saídas ficam em `dados/`. Não há código no caminho de `tudo` que importe cliente de modelo.

## 11. Testes

Sem rede. Extração de leis com frases reais; dicionário (peso, ajustes, chaves fora); candidatos (união, pai, teto); características (escala, par sem vizinho); modelo (treino em dados pequenos, normalização soma 1, agregação pelo pai); detector; validação cruzada com folha sintética (estratificação, nenhuma questão de teste no índice da própria dobra, escada); decisão (as cinco regras); formato do CSV; `tudo` em dados de exemplo; guarda do cache de embeddings (texto ausente levanta erro, sem chamar nada).

## 12. Entregáveis e prazo

`publicacao.csv`, `previsao.txt`, `relatorio.md` (até 4 páginas: sistema peça por peça; os 100 rótulos; diagnóstico; produção com custo, correção do professor como treino e cadência de retreino; memo do professor), `README.md` (como rodar, tempo, proveniência de cada artefato, decisões à mão), código e artefatos versionados. Prazo: quinta 22/10 às 14h; pedido de prazo até segunda 26/10 se na quarta à noite faltar parte.

## 13. Decisões registradas

- Duplicados publicáveis, por evidência na folha.
- `disciplina_origem` como característica fraca, nunca como regra, por evidência da etapa 1 (7 de 8 "fora do escopo" eram penais).
- Exemplos e enriquecimento gerados por modelos por assinatura (sem custo de API), revisados por amostra; os prompts e as revisões ficam versionados.
- Regressão logística em vez de modelos maiores: poucos rótulos, pesos legíveis, determinismo, e a literatura comparativa mostra que baselines bem calibrados competem com métodos hierárquicos elaborados.
- Corpus da etapa 1 desligado por padrão e decidido pela regra dos 3 pontos.
