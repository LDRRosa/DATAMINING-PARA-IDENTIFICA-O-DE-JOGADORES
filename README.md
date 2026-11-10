# Data mining para identificação de jogadores subvalorizados

Código do trabalho de conclusão de curso **"Desenvolvimento de um método de data mining para identificação de jogadores subvalorizados"**, do curso de Sistemas de Informação do Instituto Federal Goiano, Campus Urutaí.

**Autores:** Leandro Rosa da Silva e Cristian Martins Fernandes.

## O que o projeto faz

Estima o valor de mercado de jogadores de futebol (Transfermarkt) a partir de estatísticas de desempenho (FBref, temporada 2024/25), usando *Random Forest*, *XGBoost* e *LightGBM* treinados separadamente por posição (goleiros, defensores, meio-campistas e atacantes). Cada jogador é classificado como **subvalorizado**, **preço justo** ou **supervalorizado** (margem de 20% entre o valor estimado e o valor de mercado). As classificações são avaliadas em uma **validação prospectiva**: comparam-se com a variação do valor de mercado após 31/07/2025.

## Estrutura

```
.
├── scripts/
│   ├── cenario1_previsao_bruta.R             # Cenário 1
│   ├── cenario2_filtro_idade.R               # Cenário 2
│   ├── cenario3_depreciacao_idade.R          # Cenário 3
│   └── grafico_variaveis_selecionadas.R      # gráficos das variáveis selecionadas
├── dados/
│   └── README.md                             # onde obter os dados (não versionados)
├── .gitignore
└── README.md
```

## Cenários

| Script | Descrição | Arquivos gerados |
|---|---|---|
| `cenario1_previsao_bruta.R` | Classificação direta com a previsão dos modelos | `tcc_metricas_comparativas.csv`, `tcc_jogadores_analise_completa.csv`, `tcc_validacao_prospectiva.csv`, `tabela_final.csv` |
| `cenario2_filtro_idade.R` | Mesmas previsões; a taxa de acerto dos subvalorizados considera apenas jogadores com até 22 anos | os três primeiros, com sufixo `_cenario2` |
| `cenario3_depreciacao_idade.R` | Previsão multiplicada por um fator de depreciação por faixa etária, sem filtro de idade na validação | os três primeiros, com sufixo `_cenario3` |

## Como executar

1. Instale o R e os pacotes:
   ```r
   install.packages(c("dplyr", "readr", "caret", "ranger", "xgboost", "lightgbm",
                      "lubridate", "stringdist", "doParallel",
                      "tidyr", "tibble", "ggplot2", "forcats"))
   ```
2. Baixe os dados (ver [`dados/README.md`](dados/README.md)).
3. Em cada script, ajuste `DIR_BASE` para a pasta que contém os arquivos CSV (os resultados também são gravados nela).
4. Execute o script do cenário desejado, por exemplo `source("scripts/cenario1_previsao_bruta.R")`. Os scripts usam paralelismo (todos os núcleos menos um).
5. Para os gráficos, execute `scripts/grafico_variaveis_selecionadas.R` na mesma sessão, depois do Cenário 1.

## Decisões metodológicas

- Jogadores com menos de 5 jogos completos (`90s` < 5) são descartados.
- Os dados do FBref e do Transfermarkt são integrados por *Fuzzy Join* (distância Jaro-Winkler menor que 0,15).
- A variável-alvo é o valor de mercado na data de corte, na escala `log1p`.
- A liga (`Comp`) não é usada como preditor. Idade (`Age`) e minutagem (`90s`) são preditores em todas as posições.
- Validação cruzada com 5 partições, estratificada pelo valor de mercado. As métricas e a classificação usam **previsões fora da partição** (*out-of-fold*), de modo que cada jogador é estimado por um modelo que não o utilizou no treinamento.

## Limitações conhecidas

- O *Fuzzy Join* considera apenas a similaridade dos nomes, e há jogadores duplicados na base integrada. O tratamento está previsto como trabalho futuro.
- Os hiperparâmetros do *Random Forest* e do *XGBoost* são escolhidos nas mesmas previsões fora da partição usadas para reportar o desempenho (sem validação cruzada aninhada).
