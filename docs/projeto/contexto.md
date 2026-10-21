# Contexto do projeto (para quem chega agora)

Este repositório é a **segunda etapa** do case de classificação de tópicos da Vício (Método VDE). A primeira etapa está em https://github.com/GilsonB12/case-vicio-ia e usou dois modelos de linguagem votando por questão, com calibração num corpus de referência; ela foi avaliada e a Vício abriu esta etapa complementar com uma regra nova: **na hora de classificar, o sistema não pode chamar nenhum modelo de linguagem**. O enunciado está em `docs/case/`.

## O que mudou da etapa 1 para a 2

| | Etapa 1 | Etapa 2 |
|---|---|---|
| Questões | 359 | 659 (as 359 mais 300 de concursos recentes) |
| Gabarito | nenhum | `folha_verdade.csv`, 100 questões rotuladas pela equipe, todas entre as 359 |
| Régua | central ou pai/filho | central, **extras**, ou pai/filho de um deles; publicar uma questão fora da taxonomia é erro |
| Avaliação | nas 359 | nas 559 fora da folha |
| Modelo de linguagem | vota na decisão | só antes da decisão: embeddings em cache, limpeza, extração de lei, enriquecimento dos temas, termos e exemplos revisados |
| Decisão | regras sobre dois votos e vizinhança | sistema próprio com probabilidade calibrada |
| Relatório | até 5 páginas, 4 partes | até 4 páginas, 5 partes (sistema, os 100 rótulos, diagnóstico, produção, memo) |

## O que a folha-verdade revelou (usado como regra de dado)

- 8 das 100 são fora da taxonomia (a etapa 1 achou 1 em 359).
- 39 das 100 têm extras, e extras contam como acerto.
- 53% dos temas centrais são de nível 1.
- A equipe usa o Juizado Especial Criminal (4701, Legislação Especial), não o Procedimento Sumaríssimo, para questões da Lei 9.099.
- A equipe usa temas duplicados (um central em 6734, Lei de Execução Penal em Processo Penal), então os duplicados são publicáveis aqui.
- A etapa 1, medida contra a folha pela régua completa: 41 certas em 42 publicadas (98%); nenhuma das 8 fora foi publicada.

## O que veio da etapa 1 por cópia

Módulos de base (`arvore`, `dados`, `embeddings`, `custos`), o corpus de 332 questões reais coletadas na internet com tema conhecido (`dados/fontes/corpus_v1.jsonl`, protocolo em `docs/projeto/notes/corpus-v1-coleta.md`), a árvore legível (`docs/projeto/arvore-temas.md`). Os votos de modelos, a âncora e as auditorias da etapa 1 **não** são usados aqui, porque seriam modelos escolhendo o tema das questões avaliadas.

## Documentos de referência

- Spec: `docs/projeto/specs/2026-10-20-sistema-v2-design.md`
- Plano de implementação: `docs/projeto/notes/2026-10-20-plano-de-implementacao.md`
- Preparação com modelos (prompts, rodadas, revisão): `preparacao/`
