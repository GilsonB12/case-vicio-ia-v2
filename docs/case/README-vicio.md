# Case: classificação de tópicos

Processo seletivo da área de IA da **Vício (Método VDE)**.

## O problema

Na nossa plataforma, o aluno estuda por **tópico**: escolhe "Lei Maria da Penha" ou "Juiz das Garantias" e resolve as questões ligadas a ele. Se uma questão está no tópico errado, ele estuda a matéria errada.

Boa parte das questões foi classificada automaticamente, e isso criou tópicos genéricos que atraíram centenas de questões que não têm nada a ver com eles. E toda semana chegam questões novas, de concursos novos. Neste case você recebe **659 questões reais** de Penal, Processo Penal, Legislação Penal Especial e Criminologia, a árvore de temas em que elas podem ser classificadas e uma **folha-verdade com 100 questões classificadas pela nossa equipe**.

Para cada questão, você decide:

| Decisão | Significado |
|---|---|
| **publicar** | Vai direto para o aluno, no tema que você escolheu |
| **revisar** | Vai para a fila de um especialista, com a sua sugestão |
| **fora** | Nenhum tema da árvore serve |

**A régua:** das questões publicadas, pelo menos **95% precisam estar certas**. Dentro disso, quanto mais você publicar, melhor.

## O que queremos ver

Pôr um modelo de linguagem para classificar e outro para conferir, nós já sabemos fazer. Queremos ver **o sistema de classificação que você constrói**: os dados organizados (árvore, leis, termos, os 100 exemplos), o modelo e a regra de decisão.

- **A decisão é do seu sistema.** Embeddings, busca, regras, dicionários, classificador treinado, kNN: o que você quiser, desde que devolva uma probabilidade que você consiga calibrar.
- **Modelo de linguagem só antes da decisão:** gerar embeddings, limpar texto, extrair a lei citada, enriquecer os temas, propor termos ou exemplos que você revisa. Nunca para escolher, votar ou validar o tema.
- **Não precisa saber Direito** nem classificar questões à mão: a folha-verdade existe para isso.

## Downloads

- 📄 **[Baixar o enunciado completo (PDF)](https://github.com/viciod/case-vicio-ia/raw/main/case-ia-classificacao-de-topicos.pdf)**: leia antes de começar. Tem a régua, o formato de entrega e os critérios de avaliação.
- 📦 **[Baixar os anexos (ZIP)](https://github.com/viciod/case-vicio-ia/raw/main/anexos-case-ia.zip)**: `questoes.jsonl`, `temas.csv`, `folha_verdade.csv`, `LEIA-ME.md` (formatos) e `entrega_modelo.csv`.

## Entrega

- **Prazo:** a combinar.
- **Formato:** um **repositório público no GitHub** com `publicacao.csv` (as 659 questões), `previsao.txt`, o relatório (PDF ou Markdown) e o código. Envie o link para **carla@viciodeumaestudante.com.br**.
- **Código:** um comando que regenera o `publicacao.csv` a partir dos anexos **sem chamar modelo de linguagem**, com os artefatos que você construiu (dicionários, regras, embeddings em cache, modelo treinado) versionados. Inclua um README dizendo como rodar, quanto tempo leva e de onde veio cada artefato.

Dúvidas: **carla@viciodeumaestudante.com.br**.
