# Relatório: um sistema de classificação sem modelo de linguagem na decisão

Das 559 questões fora da folha-verdade, o sistema publica **213 (38%)**, manda **318** para revisão e marca **28** como fora da árvore. Previsão, feita na folha-verdade antes de olhar as 559: **precisão de 0,95** (intervalo de 0,85 a 1,00) e **cobertura de 0,38** (0,37 a 0,39). Nenhuma chamada a modelo de linguagem acontece na decisão: `uv run sistema tudo` regenera tudo dos anexos e dos artefatos versionados em cerca de 2 minutos, num notebook comum, sem rede e sem GPU. Uma versão mais conservadora, sem o corpus da etapa 1, está em `alternativas/`.

## a) O sistema, peça por peça

A primeira etapa deste case usou dois modelos de linguagem votando em cada questão. Aqui o modelo só pode organizar o conhecimento antes; a decisão é do sistema. O trabalho tem duas fases.

**Fase 1: conhecimento organizado, com modelos de linguagem, antes da decisão** (`preparacao/`, versionado em `dados/`). A árvore da Vício tem, para a maioria dos temas, só o caminho e uma descrição genérica. Para cada um dos 351 temas, um modelo produziu um resumo do que o tema cobre e do que o distingue dos vizinhos, as leis e os artigos centrais, 15 a 25 termos técnicos, sinônimos e três exemplos de enunciado; mais 60 questões de outras disciplinas. O material foi validado em três camadas: formato, consistência com as leis citadas nas questões da folha (11 de 14 coerentes; as 3 divergências são questões que citam leis de passagem) e leitura por amostra (11 de 12 temas corretos; uma lei não confirmada foi neutralizada). Dele saem, sem modelo, um **dicionário lei-artigo-tema** com 1.530 entradas e os **vetores** (embeddings) de questões, temas e exemplos, guardados num arquivo de 18 MB. Custo de API do projeto inteiro: US$ 0,08, só de embeddings.

**Fase 2: o sistema, sem modelo de linguagem** (`src/sistema/`).

1. **Candidatos por quatro buscas independentes.** As leis e os artigos citados na questão (extraídos por expressões regulares: "art. 157 do CP", "Lei 11.340/06") consultados no dicionário; busca por palavras (BM25) no texto dos temas; semelhança de vetores com os temas; e as questões rotuladas mais parecidas (folha, exemplos e as 332 questões reais com tema conhecido coletadas na etapa 1, estas com metade do peso). A união, mais o pai de cada subtema, dá até 30 candidatos. O tema certo está entre eles em 99% das questões da folha.
2. **Modelo de pares.** Uma regressão logística avalia cada par questão-tema com 19 características simples: a força de cada busca, quantas buscas o apontaram como primeiro ("votos"), a semelhança com o vizinho rotulado mais próximo do mesmo tema, o nível na árvore. Os pesos são legíveis (`dados/intermediarios/pesos_modelo.json`): BM25 e número de buscas pesam mais.
3. **Modelo de confiança.** O achado central: a probabilidade do modelo de pares ordena bem, mas não diz se ele está certo; o que diz é a **concordância entre as buscas independentes**, o mesmo princípio dos dois votantes da etapa 1, agora sem modelo de linguagem. Um segundo modelo, treinado só com previsões feitas fora da amostra, transforma concordância e força do primeiro colocado na chance de acerto, para duas opções: publicar no tema ou no tema pai, que a régua aceita.
4. **Detector de fora**, com sinais de outra disciplina (código penal militar, CTN, CLT), semelhança baixa com todos os temas e vizinhos fora.
5. **Decisão.** Fora, se o detector dispara; publicar na opção de maior confiança, se ela passa do limiar; senão, revisar com a melhor sugestão e o motivo. Extras: os candidatos seguintes com probabilidade relevante, porque publicar num extra conta como acerto.

**O que testei e descartei.**
- *A probabilidade normalizada como confiança:* acertos e erros tinham a mesma distribuição; com ela a cobertura não passava de 5%.
- *Um dicionário de termos técnicos por tema*, com peso para expressões exatas como "erro de tipo". Subiu a cobertura para 51% nas dobras, mas criou erros com confiança máxima, e nenhum limiar chegou a 95%. Desfeito (está no histórico do git).
- *Ajustar um encoder jurídico* (RoBERTaLexPT, LegalBert-pt) ou modelar a hierarquia com grafos: com 100 rótulos para 371 temas não há o que ajustar, e a literatura comparativa mostra ganho pequeno dos métodos hierárquicos sobre um classificador plano bem calibrado.

**A decisão sobre o corpus, tomada à mão.** O sistema roda com e sem as 332 questões da etapa 1 como vizinhos. Antes de medir, fixei uma regra: o corpus só entraria se aumentasse a cobertura em 3 pontos sem baixar a precisão nem o limite inferior. Medido, ele quase dobra a cobertura nas dobras (de 18% para 29%; nas 559, de 14% para 38%), sobe o limite inferior (de 0,834 para 0,849) e baixa a precisão de 0,957 para 0,951. Pela letra da regra, ficaria de fora. Conferi que nenhuma das 659 questões tem semelhança de 0,90 ou mais com alguma questão do corpus, então o ganho não vem de cópia; a diferença de precisão é menor que uma questão efetiva; e as duas versões têm chance parecida de passar da régua (cerca de 60% com corpus e 65% sem). Decidi usar o corpus e registro isso como decisão à mão, contra a regra que eu mesmo tinha fixado. A versão sem corpus está em `alternativas/`.

## b) Os 100 rótulos

**Como usei.** A folha é a única fonte da convenção da equipe, e faz três papéis sem se contaminar: **teste** (é dela que sai a previsão), **calibração** (é nela que se escolhe o limiar) e **treino pequeno** (os pesos das regressões e os vizinhos rotulados). Tudo por validação cruzada: 5 dobras estratificadas por disciplina e por "fora", repetidas 3 vezes; em cada dobra, 80 questões treinam e 20 medem, e nenhuma questão é avaliada por um modelo que a viu. O modelo de confiança é treinado dentro de cada dobra com uma segunda validação cruzada, para nunca aprender com previsões feitas dentro da amostra. Como as repetições reusam as mesmas questões, a precisão e o intervalo usam o número efetivo de questões.

**A regra do limiar.** Com 100 rótulos, nenhum limiar consegue um limite inferior de 95%: isso exigiria umas 60 publicações sem erro. A escada tenta primeiro precisão de 0,97 com folga e depois a régua, 0,95, com pelo menos 10 publicações efetivas, sempre com a maior cobertura; nunca aceita um limiar abaixo da régua só porque o intervalo é mais estreito. No limiar escolhido, confiança de 0,95, a folha deu 85 acertos em 88 publicações nas dobras (29 efetivas). Os três erros são duas questões, ambas da mesma convenção: um crime narrado para cobrar erro de tipo.

**O que mostraram sobre o sistema.** Fora da amostra, o primeiro colocado está certo em 78% das questões; o certo está entre os 3 primeiros em 93% e entre os 5 primeiros em 97%. Quando três ou quatro buscas concordam com o primeiro colocado, o acerto é de 93% (14 de 15); quando só uma concorda, 66%. É por isso que, abaixo do limiar, o sistema ainda é útil: as sugestões da fila estão certas na maioria das vezes, e o professor confirma rápido.

**O que mostraram sobre a convenção da equipe.**
- 8% das questões são fora da árvore, e **não** são de outras disciplinas: são questões penais (fixação da pena, aplicação da lei penal, tráfico de drogas) que a equipe não encaixa em nenhum tema. Com 8 exemplos heterogêneos, o padrão não é aprendível; o sistema só marca fora com segurança outra disciplina.
- 39% têm extras, e extras contam como acerto.
- 53% dos temas centrais são de nível 1: a equipe prefere o tema ao subtema quando a questão se espalha. O sistema publica no nível 1 em 152 das 213.
- A equipe usa temas duplicados (Lei de Execução Penal em Processo Penal), então eles são publicáveis aqui.
- Lei 9.099 vai para Juizado Especial Criminal, na Legislação Especial, e não para Procedimento Sumaríssimo.
- Quando uma questão narra um crime para cobrar um instituto da parte geral, a equipe rotula o instituto (erro de tipo, coautoria), não o crime; execução da pena vai para Lei de Execução Penal; regras gerais de prova, para Teoria Geral das Provas. São os erros que sobram com confiança alta.

As quatro primeiras viraram regra de dado (detector, extras, opção pelo pai, duplicados publicáveis); a quinta, um ajuste escrito à mão no dicionário de leis, o que deixa a validação um pouco otimista nesse caso, como está declarado no `previsao.txt`; a última fica para o professor.

## c) Diagnóstico

**O que atrapalha.** Os defeitos da etapa 1 continuam: 20 temas duplicados, títulos com erro, nós que cobrem o mesmo assunto em ramos diferentes, enunciados genéricos com o assunto só nas alternativas. As 300 questões novas trazem 10 de direito penal militar e processual penal militar, quase todas mandadas para fora; o sistema publicou 126 das novas, contra 87 das 259 antigas fora da folha, porque as novas tendem a citar leis e usar o vocabulário dos temas.

**O que o sistema ainda erra, e o que resolveria.**
- *Convenções de fronteira:* crime narrado para cobrar a parte geral; execução contra pena; prova em geral contra prova em espécie. Resolve a equipe fixar a convenção e o sistema aprender com as correções do professor.
- *Questões mistas*, com alternativas de leis diferentes: o sistema manda para revisão; a resposta certa é central mais extras, e qual é o central é convenção.
- *Questões sem sinal*, curtas ou genéricas, em que nenhuma busca tem força: vão para revisão (163 das 318 têm só uma busca apontando o tema sugerido), ou, quando nada parece com nenhum tema, para fora. Das 28 marcadas como fora, 7 são de outra disciplina e 21 são penais com sinal fraco, que o professor deve olhar.
- *Temas raros*, sem rótulo e com pouco vocabulário próprio: ficam em revisão até ganharem rótulos.

**Por que a cobertura não é maior.** A ordenação é boa (78% de top-1), mas a régua exige saber quando se está certo, e 100 rótulos só certificam o topo. Com mais rótulos, o mesmo sistema publica mais com a mesma segurança: nas dobras, o limiar de 0,90 acerta 90% com 51% de cobertura, e o de 0,85, 87% com 58%; é a distância entre esses números e 95% que os rótulos do professor fecham.

## d) Produção

**Custo e tempo para 50 mil questões por semana.** O vetor de uma questão custa cerca de US$ 0,00005 (350 tokens a US$ 0,13 por milhão): **cerca de US$ 2,50 por semana** para 50 mil. O resto roda em CPU: buscas, características e decisão das 659 questões levam poucos segundos; 50 mil cabem em menos de 10 minutos numa máquina comum. Modelos de linguagem voltam só quando a árvore muda ou um tema acumula correções: reenriquecer um tema custa centavos e passa por revisão.

**Como a correção do professor vira dado.** Cada decisão na fila (confirmar, trocar, marcar fora) é gravada como rótulo, no formato da folha-verdade. Três velocidades:
1. *Na hora:* o rótulo entra como vizinho rotulado; a próxima questão parecida já ganha um voto.
2. *Toda semana:* as regressões (pares, confiança, fora) são retreinadas com todos os rótulos, em minutos.
3. *Todo mês:* o limiar é recalibrado por validação cruzada, com o mesmo código de hoje; os temas mais corrigidos voltam para o enriquecimento.

Com algumas centenas de rótulos por semana, o intervalo da previsão estreita depressa e o limiar desce com segurança. É a mesma lógica da etapa 1, invertida: hoje o sistema usa muitas peças porque não tem gabarito; com os rótulos do professor, dá para medir cada peça e ficar só com as que pagam o que custam.

**Qualidade contínua.** Uma amostra semanal das publicadas vai para o professor sem a decisão do sistema; a precisão medida ali é a única que a régua respeita. Reportes de alunos por tema disparam revisão das questões do tema.

## e) Memo para o professor

**O que encontrei.** A árvore tem temas duplicados (a mesma Lei Maria da Penha em Penal e em Legislação Especial), sete títulos com erro de digitação ("Estaturo de Idoso") e pares de temas que cobrem o mesmo assunto em ramos diferentes (Sentença em Procedimento Ordinário e Sentença Penal; Abolitio Criminis em Punibilidade e em Lei Penal no Tempo). Nos 100 exemplos da equipe, três convenções decidem a maioria dos erros que sobram: questão que narra um crime para cobrar erro de tipo ou coautoria vai para o instituto; execução da pena vai para a Lei de Execução Penal; regras gerais de prova vão para Teoria Geral das Provas. E 8 dos 100 exemplos são questões penais marcadas como fora da árvore (sobre fixação da pena, aplicação da lei penal, tráfico de drogas), sem um critério que eu consiga reproduzir.

**O que recomendo mudar nos temas.** Fundir os duplicados e corrigir os títulos. Para cada par de temas sobrepostos, decidir qual recebe as questões. Escrever as convenções acima numa linha cada, junto da árvore, para que pessoa e máquina classifiquem igual. Definir o que é "fora da árvore" para uma questão penal: se é questão mista, desatualizada ou de tema que falta, porque isso muda o que o sistema deve fazer com ela.

**Do que preciso do senhor.**
- **A fila deste lote:** 318 questões, agrupadas por tema, com a sugestão e o motivo. Nas que têm três ou mais buscas concordando, a sugestão costuma estar certa e basta confirmar.
- **Uma olhada nas 28 marcadas como fora:** 7 são de outra disciplina (penal militar e tributário); as outras 21 são penais em que nada se parece com nenhum tema, e só o senhor sabe se falta um tema ou se a questão serve em algum.
- **Uma amostra mensal de 50 publicadas** para auditoria. É ela que confirma os 95%, e cada correção do senhor faz o sistema publicar mais na semana seguinte.
