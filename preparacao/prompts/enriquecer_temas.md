# Prompt de enriquecimento dos temas

Usado em lotes de 20 temas, por modelo de linguagem por assinatura, antes de qualquer decisão. Saída versionada em `dados/temas_enriquecidos.jsonl`.

```text
Você é um professor de Direito Penal, Processo Penal, Legislação Penal Especial e Criminologia, especialista em concursos públicos brasileiros. Vou te dar temas de uma árvore de temas de uma plataforma de estudos (caminho na árvore e descrição original, que às vezes é pobre). Para cada tema, produza o conhecimento que um sistema de busca lexical vai usar para reconhecer questões de concurso desse tema. Capriche na cobertura de leis, artigos e vocabulário técnico.

Para cada tema, devolva UMA linha JSON com exatamente estas chaves:
{"tema_id": 1234,
 "resumo": "2 a 3 frases dizendo o que o tema cobre e o que o distingue dos temas vizinhos",
 "leis": [{"numero": "11.340/2006", "nome": "Lei Maria da Penha"}],      // leis e decretos-lei que o tema cobre; número com ano; Código Penal = "2.848/1940", CPP = "3.689/1941", LEP = "7.210/1984"; lista vazia se não há lei específica
 "artigos": [{"diploma": "CP", "artigo": "157"}],                         // artigos centrais do tema; diploma em sigla (CP, CPP, CF, LEP, CPM) ou "Lei 11.340"; até 15
 "termos": ["roubo", "grave ameaça", "violência à pessoa"],              // 15 a 25 termos e expressões técnicas que aparecem em questões do tema, em minúsculas
 "sinonimos": ["subtração violenta"],                                    // outras formas de dizer o assunto; pode ser vazia
 "exemplos": ["texto de uma questão curta e realista sobre o tema, com enunciado e 2 a 4 alternativas resumidas", "...", "..."]   // exatamente 3, 60 a 120 palavras cada, variando banca e estilo (múltipla escolha, certo/errado, caso hipotético)
}

Regras: só JSON, uma linha por tema, na ordem dada, sem comentários, sem cercas de código, sem texto antes ou depois. Os exemplos devem ser inéditos (não copie questões reais), em português do Brasil, e devem ser inequivocamente do tema, não de um tema vizinho. Em Criminologia não há leis; preencha "leis" e "artigos" vazios e capriche em termos e autores.

TEMAS:
```
