# Plano de implementação da etapa 2

Spec: `docs/projeto/specs/2026-10-20-sistema-v2-design.md`. Tarefas autocontidas; cada uma termina com testes verdes e um commit. Português em código, comentários, testes e commits; identificadores sem acento; `git add` por caminho explícito; sem trailers. Nenhum teste chama rede. Nenhum módulo no caminho de `sistema tudo` importa cliente de modelo de linguagem.

## Interfaces compartilhadas

- `sistema.config.CAMINHOS`: `raiz`, `anexos`, `questoes`, `temas`, `folha`, `dados`, `temas_enriquecidos`, `exemplos`, `dicionario_leis`, `dicionario_ajustes`, `termos_fora`, `corpus_v1`, `embeddings` (pasta), `intermediarios` (pasta), `custos` (pasta), `publicacao`, `previsao`. `SEMENTE = 42`. `ler_env`, `ChaveAusenteError`.
- `sistema.dados`: `carregar_questoes()` (659 linhas: `id, texto, enunciado_html, comando, alternativas, disciplina_origem, assunto_origem`); `carregar_folha()` (`id, tema_central_id` inteiro ou nulo, `extras_ids` lista de inteiros, `fora` booleano); `alvos_aceitos(linha, arvore) -> set[int]` (central, extras e pai ou filho direto de cada um); `acerto_regua(tema_id, linha, arvore) -> bool` (falso quando `fora`); `carregar_temas_enriquecidos()` (dict por `tema_id`); `carregar_exemplos()` (`id, tema_id` ou nulo, `fora`, `texto`, `fonte`); `carregar_corpus_v1()` (`id, tema_id, texto`); `ler_jsonl`, `escrever_jsonl`.
- `sistema.arvore.ArvoreTemas`: `carregar()`, `existe`, `ativo`, `eh_raiz`, `nivel`, `pai`, `filhos`, `disciplina`, `caminho`, `titulo`, `publicavel` (ativo e não raiz; duplicados são publicáveis), `original_de(tema_id)` (o original de um duplicado, ou ele mesmo), `candidatos_possiveis()` (371), `acerto(decidido, verdadeiro)`, `mesmo_ramo`, `ancestral_nivel1`, `texto_tema(tema_id, enriquecidos)` (caminho + resumo + leis + artigos + termos + sinônimos; para duplicado, o conteúdo do original com o caminho do duplicado; sem enriquecido, caminho + descrição original).
- `sistema.embeddings.CacheEmbeddings`: `vetor(texto) -> np.ndarray` e `matriz(textos) -> np.ndarray`, só do cache; levantam `EmbeddingAusenteError(textos_faltando)` sem tocar na rede. `gerar(textos, livro)` chama a OpenAI, grava no cache e no livro-razão; usada só por `sistema embeddings`.
- `sistema.leis`: `extrair_chaves(texto) -> Counter[str]` (chaves `lei:`, `dl:`, `diploma:`, `art:`, `sumula:`), `DIPLOMAS` (apelidos → chave), `normalizar_lei`.
- `sistema.dicionario`: `construir(enriquecidos, ajustes) -> dict`, `carregar()`, `pontuar(chaves, dicionario) -> (dict[tema_id, float], peso_fora: float)`.
- `sistema.indice_bm25.IndiceBM25(ids, textos)`: `.buscar(texto, k) -> list[(id, pontuacao)]`; `tokenizar(texto)`.
- `sistema.vizinhos.IndiceVetorial(ids, matriz, rotulos)`: `.buscar(vetor, k) -> list[(id, rotulo, similaridade)]`.
- `sistema.candidatos.gerar(questao, contexto) -> list[Candidato(tema_id, origens: set[str], pontuacao)]` com `contexto` carregando árvore, dicionário, índices e cache.
- `sistema.caracteristicas.tabela_pares(questoes, candidatos_por_id, contexto, folha_treino_ids) -> DataFrame` com as colunas da spec seção 6 mais `id`, `tema_id`.
- `sistema.modelo`: `ModeloPares.treinar(tabela, rotulos, pesos)`, `.probabilidades(tabela) -> DataFrame[id, tema_id, p]` normalizadas por questão; `agregar_ramo(prob, arvore)`; `DetectorFora.treinar(...)`, `.p_fora(...)`.
- `sistema.calibracao.validar(contexto, com_corpus) -> Resultado` (dobras, grade de limiares, escolha pela escada, intervalos); `comparar_fontes(contexto)`.
- `sistema.decisao.decidir(id, prob, p_ramo, p_fora, limiares, arvore) -> Decisao(id, decisao, tema_central_id, extras_ids, confianca, motivo)`.
- `sistema.publicar.escrever_publicacao(decisoes)`, `escrever_previsao(resumo)`, `validar_publicacao(df)`.
- `sistema.cli.main`: subcomandos `tudo [--com-corpus]`, `embeddings`, `dicionario`, `avaliar-folha [--com-corpus]`, `comparar-fontes`, `custos`.

## Tarefas

### T1. Base: config, dados, árvore, embeddings, custos, CLI
Adaptar os módulos copiados da etapa 1 (`src/sistema/{config,dados,arvore,embeddings,custos}.py`, que ainda importam `classificador.*` e apontam para caminhos antigos) às interfaces acima; criar `cli.py` com os subcomandos (os que dependem de T3 e T4 levantam `NotImplementedError` com mensagem); adaptar os testes copiados (`tests/test_{arvore,dados,embeddings,custos}.py`, `conftest.py`) e acrescentar: 659 questões carregam; folha com 100 linhas, 8 `fora`, `alvos_aceitos` inclui pai e filho e extras; `acerto_regua`; `publicavel` conta 371 e inclui 6734; `original_de(6734)`; `texto_tema` com e sem enriquecido; cache de embeddings levanta `EmbeddingAusenteError` com a lista e não importa cliente nenhum fora de `gerar`; `sistema embeddings` grava cache e livro (com dublê de cliente). `uv run pytest -q` verde.

### T2. Leis e dicionário
`src/sistema/leis.py`, `src/sistema/dicionario.py`, `dados/termos_fora.json` (lista inicial: militar, CPM, CPPM, tributário, ICMS, ITCD, IPTU, constitucional, administrativo, licitação, trabalhista, CLT, civil, contrato, consumidor, previdenciário, eleitoral, ambiental? — cuidado: crimes ambientais e eleitorais estão na árvore; não incluir termos que pertencem a temas publicáveis), `dados/dicionario_leis_ajustes.json` (vazio inicial, com formato documentado), subcomando `sistema dicionario`. Testes com frases reais tiradas de `anexos/questoes.jsonl` (ler o arquivo nos testes é permitido): "Lei nº 11.340/06", "Lei 8.072/90", "art. 157, § 2º, do CP", "arts. 240 a 245 do CPP", "Súmula Vinculante 11", "Código Penal Militar", "ECA"; artigo sem diploma usa o dominante do texto; chaves `fora` para CPM, CPPM, CTN, CLT, CF (CF só quando é o único diploma). Dicionário: peso 1/n, ajustes sobrescrevem, `pontuar` soma pesos por tema e devolve `peso_fora`.

### T3. Texto, índices, candidatos e características
Depende de T1 e T2. `indice_bm25.py`, `vizinhos.py` (NumPy, cosseno), `candidatos.py` (as 4 buscas, pai, filhos do pai vindo do dicionário, teto 30), `caracteristicas.py` (colunas da spec), `contexto.py` (carrega tudo uma vez: árvore, enriquecidos, dicionário, índices, cache, exemplos, folha, corpus opcional). Testes: tokenização sem acento e stopwords; BM25 devolve o tema certo para um texto igual ao do tema; vizinhos por cosseno; união e teto dos candidatos; cada característica num caso construído à mão; par sem vizinho tem `knn_*` zero.

### T4. Modelo, validação, decisão, publicação e `tudo`
Depende de T3. `modelo.py`, `calibracao.py`, `decisao.py`, `publicar.py`, e os subcomandos `tudo`, `avaliar-folha`, `comparar-fontes`. Testes com dados pequenos construídos: normalização soma 1; agregação pelo pai; detector; estratificação e vazamento zero nas dobras (nenhuma questão de teste entra no índice kNN nem no treino da própria dobra); escada; regra dos 3 pontos; as 5 regras de decisão; CSV válido pelo LEIA-ME (659 linhas, motivo obrigatório em revisar e fora, central vazio em fora); `previsao.txt` com as 4 linhas. `tudo` roda em dados de exemplo pequenos sem rede.

### T5. Preparação do conhecimento (fora do código)
Prompts em `preparacao/prompts/`; rodadas com modelos por assinatura; saídas em `dados/temas_enriquecidos.jsonl` e `dados/exemplos.jsonl`; revisão por amostra registrada em `preparacao/README.md`; `uv run sistema dicionario`; `uv run sistema embeddings` (API, livro-razão).

### T6. Documentos
`README.md` (como rodar, tempo, proveniência de cada artefato, decisões à mão), `relatorio.md` (5 partes, até 4 páginas), notas de decisões.

## Ordem e paralelismo
T1 ∥ T2 ∥ T5 → T3 → T4 → rodada completa → T6.
