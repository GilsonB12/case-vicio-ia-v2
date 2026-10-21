# Decisões tomadas depois da spec (20 e 21/10/2026)

A spec (`docs/projeto/specs/2026-10-20-sistema-v2-design.md`) descreve o desenho inicial. Os dados reais mudaram quatro coisas. Cada uma está num commit próprio, e os números abaixo são fora da amostra na folha-verdade (5 dobras estratificadas, 3 repetições, contagens efetivas por questão).

## 1. A confiança passou a ser um segundo estágio (commit 1a85f47)

Na primeira rodada real, a probabilidade normalizada do modelo de pares (`p`) ordenava bem (top-1 de 73%, top-3 de 83%), mas tinha a mesma distribuição em acertos e erros, e a escada só achou cobertura de 5%. O sinal que separava era a concordância entre as buscas independentes (BM25, cosseno com o tema, dicionário de leis): 0 buscas concordando com o top-1, 57% de acerto; 2, 90%; 3, 100%. Daí:

- os votos de cada busca (top-1 de cada uma) viraram características do par;
- um `ModeloConfianca` (regressão logística) estima a chance de acerto de duas opções, o tema e o tema pai, treinado só com previsões feitas fora da amostra (validação interna de 5 dobras dentro de cada dobra externa);
- a decisão escolhe a opção de maior confiança e publica se ela passa do limiar.

## 2. O intervalo passou a usar o número efetivo de questões (commit 1a85f47)

As 3 repetições reaproveitam as mesmas 100 questões; somar as 300 avaliações estreitava o intervalo artificialmente. Precisão (Jeffreys) e intervalo (Beta) usam acertos e publicadas divididos pelo número de repetições.

## 3. O enriquecimento completo dos temas (20/10, noite)

Com 199 dos 351 temas enriquecidos, o top-1 era 73%; com os 351, 78%, o top-3 foi de 86% para 93% e o top-5 de 90% para 97%.

## 4. O dicionário de termos por tema foi testado e desfeito (commits ade3606 e fe36e51)

Ideia: dar peso a expressões técnicas exatas do enriquecimento ("erro de tipo", "coautoria"), com peso pela raridade, para resolver os erros de convenção vistos na folha. Resultado: a cobertura subiu muito (até 51% nas dobras), mas o sistema passou a errar com confiança máxima (a Q0319 saía com confiança 1,00, errada), e nenhum limiar chegou a 95% de precisão. As cinco questões-alvo continuaram erradas, porque as listas de termos não tinham as expressões que elas usam. Desfeito.

## 5. A característica de vizinho rotulado ficou (episódio do commit ade3606)

`knn_folha_max` (semelhança com a questão rotulada mais próxima do mesmo tema) parecia não transferir: média 0,28 nas questões da folha contra 0,17 nas outras 559. A comparação estava errada: os valores da folha vinham do modelo final, treinado nelas. Medido fora da amostra (`scripts/diagnostico_transferencia.py`), vale 0,21 na folha e 0,19 nas 559, com 71% e 74% de zeros. Ela transfere e voltou para a confiança; tirá-la foi o que derrubou a precisão no topo.

## 6. A escada de limiares não aceita mais precisão abaixo da régua (commit b86aa89)

O último degrau da escada original era "maior limite inferior". Com 100 rótulos, o limite inferior nunca chega a 0,95, e maximizá-lo empurrou a escolha para um limiar com precisão prevista de 0,91, abaixo da régua. A escada agora é: degrau 0 (média ≥ 0,97 e limite inferior ≥ 0,95), 1 (limite inferior ≥ 0,95), 2 (média ≥ 0,97 com pelo menos 10 publicações efetivas), 3 (média ≥ 0,95, a régua, com pelo menos 10 publicações efetivas), 4 (maior precisão, declarando que a régua não foi atingida). A execução entregue ficou no degrau 3.

## 7. Corpus da etapa 1 dentro, por decisão à mão contra a regra fixada antes

Com o corpus, o limiar 0,95 dá precisão de 0,951 (85 acertos em 88 publicações nas dobras) e cobertura de 29% (38% nas 559); sem ele, o limiar 0,97 dá 0,957 (54 em 55) e 18% (14% nas 559). A regra combinada antes de medir (corpus entra se a cobertura sobe 3 pontos sem perder precisão nem limite inferior) o deixava de fora, porque a precisão cai 0,006. Antes de contrariar a regra, conferi que nenhuma das 659 questões tem semelhança de 0,90 ou mais com uma questão do corpus (a maior está abaixo disso; a mediana é 0,73, igual entre folha, antigas e novas), então o ganho não vem de cópia. Pela mesma folha, a chance de a precisão observada nas publicadas passar de 95% é de cerca de 60% com corpus e 65% sem. Decisão: usar o corpus, porque a diferença de risco é pequena e a cobertura quase triplica. A constante `USAR_CORPUS_NA_ENTREGA` em `pipeline.py` registra a decisão; o `previsao.txt` traz o veredito da regra e a decisão final; a versão sem corpus está em `alternativas/`.
