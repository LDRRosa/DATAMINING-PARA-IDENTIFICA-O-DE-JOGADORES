# =============================================================================
# TCC — Estimativa de valor de mercado de jogadores de futebol
# -----------------------------------------------------------------------------
# Objetivo: identificar jogadores SUBVALORIZADOS / SUPERVALORIZADOS comparando
# o valor de mercado real (Transfermarkt) com o valor que um modelo de ML
# estima a partir do desempenho esportivo (FBref, temporada 2024/25).
#
# Fluxo geral do script:
#   0. Bibliotecas, configurações e paralelismo
#   1. Dados financeiros (Transfermarkt): valor de mercado na data de corte
#   2. Dados esportivos (FBref): estatísticas + variáveis por posição
#   3. Cruzamento das duas bases por similaridade de nome (fuzzy join)
#   4. Modelagem: Random Forest, XGBoost e LightGBM, por posição
#   5. Exportação dos resultados
#   6. Validação prospectiva: o rótulo dado pelo modelo se confirmou depois?
# =============================================================================

# --- 0. Bibliotecas ----------------------------------------------------------
library(dplyr)        # manipulação de dados (filter, mutate, joins...)
library(readr)        # leitura/escrita de CSV
library(caret)        # framework de treino/validação (usado no Random Forest e nas métricas)
library(ranger)       # implementação rápida de Random Forest (chamada pelo caret)
library(xgboost)      # gradient boosting (XGBoost)
library(lightgbm)     # gradient boosting (LightGBM)
library(lubridate)    # manipulação de datas
library(stringdist)   # distância entre textos (fuzzy join dos nomes)
library(doParallel)   # backend de paralelismo reconhecido pelo caret
library(parallel)     # detectCores() / makeCluster()

# --- Configurações Globais ---------------------------------------------------
SEMENTE            <- 42                      # semente fixa para reprodutibilidade
DATA_CORTE_TREINO  <- as.Date("2025-07-31")   # "foto" do mercado: só valores até esta data entram no treino
N_FOLDS            <- 5                       # nº de folds da validação cruzada
DIR_BASE           <- "C:/Users/leand/Downloads/TCC Leandro/TCC"  # pasta com os CSVs de entrada e saída

# --- Configuração do Paralelismo ---------------------------------------------
# Usa todos os núcleos menos um (deixa 1 livre para o sistema não travar).
# O cluster registrado aqui é usado automaticamente pelo caret (Random Forest);
# XGBoost e LightGBM paralelizam por conta própria via nthread/num_threads.
N_CORES <- max(1, detectCores() - 1)
cl <- makeCluster(N_CORES)
registerDoParallel(cl)
message("Paralelismo ativado com ", N_CORES, " núcleos.")

# --- 1. Dados Financeiros (Transfermarkt) ------------------------------------
players_tm    <- read_csv(file.path(DIR_BASE, "players.csv"), show_col_types = FALSE)
valuations_tm <- read_csv(file.path(DIR_BASE, "player_valuations.csv"), show_col_types = FALSE) %>%
  mutate(date = as.Date(date))   # garante que a coluna de data é do tipo Date

# Data mais recente disponível no histórico de valorizações.
# Será o "ponto de chegada" da validação prospectiva (etapa 6).
DATA_VALIDACAO <- max(valuations_tm$date, na.rm = TRUE)

# Para cada jogador, pega a avaliação de mercado MAIS RECENTE até a data de corte.
# Este é o "Valor_Mercado" real que o modelo tentará prever.
valores_corte <- valuations_tm %>%
  filter(date <= DATA_CORTE_TREINO) %>%
  group_by(player_id) %>%
  slice_max(order_by = date, n = 1) %>%   # última avaliação de cada jogador
  ungroup() %>%
  select(player_id, Valor_Mercado = market_value_in_eur)

# Junta o valor de mercado ao cadastro de jogadores (só quem tem valor na data de corte)
# e cria um nome "limpo" (minúsculo e sem acentos) para facilitar o cruzamento por nome.
base_valores <- players_tm %>%
  inner_join(valores_corte, by = "player_id") %>%
  select(player_id, Nome_TM = name, Valor_Mercado) %>%
  mutate(Nome_TM_Limpo = tolower(iconv(Nome_TM, from = "UTF-8", to = "ASCII//TRANSLIT")))

# --- 2. Estatísticas Esportivas (FBref) --------------------------------------
base_stats <- read_csv(file.path(DIR_BASE, "players_data-2024_2025 (1).csv"), show_col_types = FALSE) %>%
  mutate(
    Pos_Principal = substr(Pos, 1, 2),   # posição principal (ex.: "FW,MF" -> "FW")
    Nome_FB_Limpo = tolower(iconv(Player, from = "UTF-8", to = "ASCII//TRANSLIT"))  # mesmo tratamento de nome do Transfermarkt
  ) %>%
  filter(`90s` >= 5.0)   # descarta quem jogou menos de 5 "jogos completos" (amostra pequena = estatística instável)

# Variáveis preditoras usadas em cada posição (cada posição tem um perfil de desempenho diferente).
features_gk <- c("Age", "90s", "Saves", "Save%", "CS", "CS%", "PKA", "PKsv", "GA90", "PSxG+/-", "Stp%", "#OPA/90", "AvgDist", "Cmp%_stats_keeper_adv")   # goleiros
features_df <- c("Age", "90s", "Tkl", "TklW", "Int", "Clr", "Err", "Recov", "Blocks_stats_defense", "Tkl+Int", "Cmp%", "PrgP", "PrgC", "Won%")         # defensores
features_mf <- c("Age", "90s", "PrgP", "PrgC", "KP", "Cmp%", "xA", "PPA", "Touches", "Carries", "Recov", "Tkl+Int", "SCA90", "GCA90")                  # meio-campistas
features_fw <- c("Age", "90s", "Gls", "Ast", "xG", "xAG", "npxG", "PrgC", "Touches", "KP", "SCA90", "GCA90", "Sh/90", "SoT%", "Succ%")                 # atacantes

# --- 3. Fuzzy Join (cruzamento por similaridade de nome) ---------------------
# Os dois sites grafam nomes de forma diferente, então não dá para usar um join exato.
# Calcula a distância Jaro-Winkler (0 = idêntico, 1 = totalmente diferente) entre
# TODOS os nomes do FBref (linhas) e TODOS os do Transfermarkt (colunas).
matriz_distancia <- stringdistmatrix(base_stats$Nome_FB_Limpo, base_valores$Nome_TM_Limpo, method = "jw", nthread = N_CORES)

# Para cada jogador do FBref, acha o índice do nome mais parecido no Transfermarkt
# e a distância desse melhor par.
melhor_match_idx <- apply(matriz_distancia, 1, which.min)
distancia_minima <- apply(matriz_distancia, 1, min)

# Anexa o player_id e o valor de mercado do melhor match e descarta os matches ruins
# (distância >= 0.15 = nomes parecidos demais só por acaso, provável jogador diferente).
dados_completos <- base_stats %>%
  mutate(
    player_id     = base_valores$player_id[melhor_match_idx],
    Valor_Mercado = base_valores$Valor_Mercado[melhor_match_idx],
    Dist_Nome     = distancia_minima
  ) %>%
  filter(Dist_Nome < 0.15) %>%
  select(-Dist_Nome, -Nome_FB_Limpo)

# --- 4. Modelagem ------------------------------------------------------------

# Grade de hiperparâmetros do XGBoost (2 x 2 x 2 = 8 combinações testadas por posição):
#   nrounds   = nº de árvores; max_depth = profundidade máxima; eta = taxa de aprendizado.
# Os demais parâmetros ficam fixos. Profundidades baixas ajudam a evitar overfitting
# porque há poucos jogadores por posição.
grade_xgb <- expand.grid(
  nrounds = c(100, 300), max_depth = c(2, 4), eta = c(0.05, 0.1),
  gamma = 0, colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8
)

# Parâmetros fixos do LightGBM. num_threads = N_CORES usa todos os núcleos configurados.
# min_data_in_leaf e num_leaves pequenos são adequados para bases pequenas.
params_lgbm <- list(
  objective = "regression", metric = "rmse", learning_rate = 0.05, num_leaves = 15,
  min_data_in_leaf = 8, feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 1,
  deterministic = FALSE, force_row_wise = TRUE, num_threads = N_CORES, verbosity = -1
)

# Cria os folds da validação cruzada UMA vez por posição e reaproveita nos 3 modelos,
# garantindo que todos sejam comparados exatamente nas mesmas divisões dos dados.
# returnTrain = TRUE: cada elemento da lista traz os índices de TREINO daquele fold.
gerar_folds <- function(y, k = N_FOLDS, seed = SEMENTE) {
  set.seed(seed)
  caret::createFolds(y, k = k, list = TRUE, returnTrain = TRUE)
}

# --- Modelo 1: Random Forest (via caret + ranger) ---
# O caret reconhece automaticamente o cluster do doParallel e treina o Random Forest simultaneamente.
# Devolve uma lista padronizada com:
#   $oof          -> previsões "out-of-fold" (cada jogador previsto por um modelo que NÃO o viu no treino)
#                    É ESTA a previsão usada na classificação sub/super/justo e nas métricas.
#   $predict_full -> função que aplica o modelo final (treinado com todos os jogadores) a novos dados.
#                    NÃO é usada na classificação; fica disponível para aplicar a jogadores novos.  
treinar_caret <- function(dados_treino, method, folds, tune_grid = NULL, seed = SEMENTE, ...) {
  # index = folds: usa os MESMOS folds dos outros modelos; savePredictions = "final" guarda o OOF do melhor ajuste
  controle <- trainControl(method = "cv", index = folds, savePredictions = "final", allowParallel = TRUE)
  set.seed(seed)
  args <- list(form = Valor_Mercado_log ~ ., data = dados_treino, method = method, trControl = controle)
  if (!is.null(tune_grid)) args$tuneGrid <- tune_grid
  args <- c(args, list(...))   # repassa argumentos extras (ex.: importance = "impurity")
  modelo <- do.call(train, args)
  
  list(
    oof = modelo$pred %>% arrange(rowIndex) %>% select(rowIndex, pred, obs),  # ordena pela linha original
    predict_full = function(novos_dados) predict(modelo, novos_dados)
  )
}

# --- Modelo 2: XGBoost (validação cruzada + busca em grade feita "à mão") ---
treinar_xgboost <- function(dados_treino, folds, grade = grade_xgb, seed = SEMENTE) {
  y <- dados_treino$Valor_Mercado_log                                # alvo (valor em log)
  X <- as.matrix(dados_treino %>% select(-Valor_Mercado_log))        # matriz de preditores
  n <- nrow(dados_treino)
  
  # Converte uma linha da grade em lista de parâmetros do xgboost.
  # XGBoost configurado para paralelizar a construção das árvores internamente (nthread).
  parametros_xgb <- function(combo) {
    list(objective = "reg:squarederror", max_depth = combo$max_depth, eta = combo$eta,
         gamma = combo$gamma, colsample_bytree = combo$colsample_bytree,
         min_child_weight = combo$min_child_weight, subsample = combo$subsample,
         nthread = N_CORES)
  }
  
  # Para UMA combinação de hiperparâmetros: treina em cada fold e prevê o fold de teste,
  # montando o vetor de previsões out-of-fold (OOF) de todos os jogadores.
  oof_da_combo <- function(combo) {
    oof <- rep(NA_real_, n)
    for (idx_treino in folds) {
      idx_teste <- setdiff(seq_len(n), idx_treino)   # o que não é treino é teste
      bst <- xgb.train(params = parametros_xgb(combo),
                       data = xgb.DMatrix(X[idx_treino, , drop = FALSE], label = y[idx_treino]),
                       nrounds = combo$nrounds, verbose = 0)
      oof[idx_teste] <- predict(bst, xgb.DMatrix(X[idx_teste, , drop = FALSE]))
    }
    oof
  }
  
  # Roda todas as combinações da grade e escolhe a de menor RMSE nas previsões OOF.
  set.seed(seed)
  oof_por_combo <- lapply(seq_len(nrow(grade)), function(i) oof_da_combo(grade[i, ]))
  i_melhor <- which.min(vapply(oof_por_combo, function(p) RMSE(p, y), numeric(1)))
  
  # Reajusta o modelo final com a melhor combinação, usando TODOS os dados.
  # (Não é usado na classificação; fica disponível para jogadores novos.)
  modelo_final <- xgb.train(params = parametros_xgb(grade[i_melhor, ]),
                            data = xgb.DMatrix(X, label = y),
                            nrounds = grade[i_melhor, ]$nrounds, verbose = 0)
  
  list(
    oof = tibble(rowIndex = seq_len(n), pred = oof_por_combo[[i_melhor]], obs = y),
    # select(all_of(colnames(X))) garante mesmas colunas e mesma ordem do treino
    predict_full = function(novos_dados) {
      predict(modelo_final, xgb.DMatrix(as.matrix(novos_dados %>% select(all_of(colnames(X))))))
    }
  )
}

# --- Modelo 3: LightGBM (validação cruzada, hiperparâmetros fixos) ---
treinar_lightgbm <- function(dados_treino, folds, params = params_lgbm, nrounds = 300, seed = SEMENTE) {
  y <- dados_treino$Valor_Mercado_log
  X <- as.matrix(dados_treino %>% select(-Valor_Mercado_log))
  n <- nrow(dados_treino)
  
  # Validação cruzada manual: treina em cada fold e prevê o fold de teste (OOF)
  oof_pred <- rep(NA_real_, n)
  for (idx_treino in folds) {
    idx_teste <- setdiff(seq_len(n), idx_treino)
    modelo_fold <- lgb.train(params = c(params, list(seed = seed)),
                             data = lgb.Dataset(X[idx_treino, , drop = FALSE], label = y[idx_treino]),
                             nrounds = nrounds, verbose = -1)
    oof_pred[idx_teste] <- predict(modelo_fold, X[idx_teste, , drop = FALSE])
  }
  
  # Modelo final com todos os dados (não é usado na classificação; fica disponível para jogadores novos)
  modelo_final <- lgb.train(params = c(params, list(seed = seed)),
                            data = lgb.Dataset(X, label = y), nrounds = nrounds, verbose = -1)
  
  list(
    oof = tibble(rowIndex = seq_len(n), pred = oof_pred, obs = y),
    predict_full = function(novos_dados) predict(modelo_final, as.matrix(novos_dados %>% select(all_of(colnames(X)))))
  )
}

# Lista de modelos a comparar. Cada item tem um nome e uma função que recebe
# (dados de treino, folds) e devolve a lista padronizada (oof + predict_full).
# Para incluir um novo modelo, basta acrescentar um item aqui.
especificacoes_modelos <- list(
  list(nome = "Random Forest", treinar = function(d, f) treinar_caret(d, method = "ranger", folds = f, importance = "impurity")),
  list(nome = "XGBoost",       treinar = function(d, f) treinar_xgboost(d, folds = f)),
  list(nome = "LightGBM",      treinar = function(d, f) treinar_lightgbm(d, folds = f))
)

# Métricas de qualidade a partir das previsões OOF (ou seja, desempenho em dados "não vistos").
# Calculadas na escala log (a que o modelo aprende) e em euros (expm1 desfaz o log1p).
calcular_metricas_oof <- function(oof) {
  tibble(
    RMSE_log = RMSE(oof$pred, oof$obs),                                   # erro quadrático médio (log)
    MAE_log  = MAE(oof$pred, oof$obs),                                    # erro absoluto médio (log)
    R2_log   = caret::R2(oof$pred, oof$obs, formula = "corr"),            # R² (correlação ao quadrado)
    RMSE_eur = RMSE(expm1(oof$pred), expm1(oof$obs)),                     # erro em euros
    MAE_eur  = MAE(expm1(oof$pred), expm1(oof$obs))
  )
}

# Função central: para UMA posição, prepara os dados, treina os 3 modelos,
# calcula as métricas e classifica cada jogador (sub/super/justo) por modelo.
avaliar_posicao_multi_modelo <- function(dados, posicao_alvo, variaveis_preditoras) {
  # Mantém só as variáveis preditoras que realmente existem na base (evita erro por coluna ausente)
  colunas_existentes <- intersect(variaveis_preditoras, names(dados))
  
  # Filtra a posição, seleciona colunas, troca NA por 0 nas numéricas
  # e cria o alvo em log (log1p) — reduz a assimetria forte dos valores de mercado.
  dados_posicao <- dados %>%
    filter(Pos_Principal == posicao_alvo) %>%
    select(Player, Pos_Principal, Squad, Comp, player_id, Valor_Mercado, all_of(colunas_existentes)) %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), 0, .)),
           Valor_Mercado_log = log1p(Valor_Mercado))
  
  if (nrow(dados_posicao) == 0) stop("Nenhum dado sobrou após filtros.")
  
  # Base de treino: só preditores + alvo (remove colunas de identificação)
  dados_treino <- dados_posicao %>% select(-Player, -Pos_Principal, -Squad, -Comp, -player_id, -Valor_Mercado)
  folds <- gerar_folds(dados_treino$Valor_Mercado_log)   # mesmos folds para todos os modelos
  
  resultados <- list()
  metricas   <- list()
  
  for (spec in especificacoes_modelos) {
    res_mod <- spec$treinar(dados_treino, folds)
    
    # Métricas OOF deste modelo nesta posição
    metricas[[spec$nome]] <- calcular_metricas_oof(res_mod$oof) %>%
      mutate(Modelo = spec$nome, Posicao = posicao_alvo, .before = 1)
    
    # Garante exatamente 1 previsão OOF por jogador, na mesma ordem de dados_posicao,
    # e sem valores ausentes (cada jogador cai em um fold de teste exatamente uma vez).  
    stopifnot(nrow(res_mod$oof) == nrow(dados_posicao), !anyNA(res_mod$oof$pred))        
    
    # Valor estimado por jogador e classificação.
    # O valor estimado vem das previsões OUT-OF-FOLD: cada jogador é estimado por um modelo
    # que NÃO o viu no treino, de modo que a estimativa não é contaminada pelo próprio
    # valor de mercado dele.
    #   estimado > real + 20%  -> Subvalorizado (o mercado paga menos do que o desempenho sugere)
    #   estimado < real - 20%  -> Supervalorizado
    #   caso contrário         -> Preço Justo
    resultados[[spec$nome]] <- dados_posicao %>%
      mutate(
        Valor_Estimado = expm1(res_mod$oof$pred),   
        Diferenca = Valor_Estimado - Valor_Mercado,
        Status = case_when(
          Diferenca > (0.2 * Valor_Mercado)  ~ "Subvalorizado",
          Diferenca < -(0.2 * Valor_Mercado) ~ "Supervalorizado",
          TRUE ~ "Preço Justo"
        ),
        Modelo = spec$nome
      )
  }
  list(resultados = bind_rows(resultados), metricas = bind_rows(metricas))
}

# Executa o pipeline completo para cada posição (um modelo separado por posição)
saida_fw <- avaliar_posicao_multi_modelo(dados_completos, "FW", features_fw)
saida_mf <- avaliar_posicao_multi_modelo(dados_completos, "MF", features_mf)
saida_df <- avaliar_posicao_multi_modelo(dados_completos, "DF", features_df)
saida_gk <- avaliar_posicao_multi_modelo(dados_completos, "GK", features_gk)

# --- 5. Exportação de Resultados ---------------------------------------------
# (a) Tabela comparativa de métricas: uma linha por modelo x posição
tabela_metricas <- bind_rows(saida_fw$metricas, saida_mf$metricas, saida_df$metricas, saida_gk$metricas) %>% arrange(Posicao, Modelo)
write_csv(tabela_metricas, file.path(DIR_BASE, "tcc_metricas_comparativas.csv"))

# (b) Tabela final por jogador e modelo, com colunas renomeadas para leitura no texto do TCC,
# valores arredondados e ordenada da maior "subvalorização" para a menor dentro de cada modelo.
tabela_final <- bind_rows(saida_fw$resultados, saida_mf$resultados, saida_df$resultados, saida_gk$resultados) %>%
  select(Jogador = Player, Posição = Pos_Principal, Clube = Squad, Liga = Comp,
         ID_Transfermarkt = player_id, Modelo, `Valor Real (€)` = Valor_Mercado,
         `Valor Estimado (€)` = Valor_Estimado, `Diferença (€)` = Diferenca, Status) %>%
  mutate(across(c(`Valor Estimado (€)`, `Diferença (€)`), ~round(., 2))) %>%
  arrange(Modelo, desc(`Diferença (€)`))

write_csv(tabela_final, file.path(DIR_BASE, "tcc_jogadores_analise_completa.csv"))

# --- 6. Validação Prospectiva ------------------------------------------------
# Pergunta: os jogadores que o modelo chamou de "subvalorizados" de fato valorizaram
# MAIS depois da data de corte do que os "supervalorizados"?

# Valorização real de cada jogador: última avaliação disponível (até DATA_VALIDACAO)
# comparada com o valor na data de corte. Só entram avaliações posteriores ao corte.
valorizacao_real <- valuations_tm %>%
  filter(date <= DATA_VALIDACAO) %>%
  group_by(player_id) %>%
  slice_max(order_by = date, n = 1) %>%
  ungroup() %>%
  inner_join(valores_corte, by = "player_id", suffix = c("_Futuro", "_Corte")) %>%
  filter(date > DATA_CORTE_TREINO, Valor_Mercado > 0) %>%
  mutate(Valorizacao_Pct = (market_value_in_eur / Valor_Mercado - 1) * 100) %>%   # variação % do valor
  select(player_id, Valorizacao_Pct)

# Avalia UM modelo com três indicadores:
#   - Mediana de valorização de cada grupo (sub vs. super);
#   - Teste de Kruskal-Wallis: os grupos diferem estatisticamente em valorização?
#   - Taxa de acerto direcional: % de subvalorizados que valorizaram ACIMA da mediana geral
#     e % de supervalorizados que valorizaram ABAIXO dela (média das duas taxas).
validar_modelo <- function(nome_modelo, tab_final, val_real) {
  # Junta as classificações do modelo com a valorização real observada
  base_val <- tab_final %>% filter(Modelo == nome_modelo) %>% inner_join(val_real, by = c("ID_Transfermarkt" = "player_id"))
  # Referência: valorização mediana de TODOS os jogadores (o "mercado em geral")
  med_geral <- median(val_real$Valorizacao_Pct, na.rm = TRUE)
  
  resumo <- base_val %>% group_by(Status) %>% summarise(Mediana = median(Valorizacao_Pct, na.rm = TRUE), .groups = "drop")
  
  med_sub <- resumo$Mediana[resumo$Status == "Subvalorizado"]
  med_sup <- resumo$Mediana[resumo$Status == "Supervalorizado"]
  
  # Kruskal-Wallis só roda se houver pelo menos 2 grupos de Status com dados
  p_kruskal <- if (n_distinct(base_val$Status[!is.na(base_val$Valorizacao_Pct)]) >= 2) {
    kruskal.test(Valorizacao_Pct ~ Status, data = base_val)$p.value
  } else NA_real_
  
  # Taxas de acerto direcional (em %)
  tx_sub <- mean(base_val$Valorizacao_Pct[base_val$Status == "Subvalorizado"] > med_geral, na.rm = TRUE) * 100
  tx_sup <- mean(base_val$Valorizacao_Pct[base_val$Status == "Supervalorizado"] < med_geral, na.rm = TRUE) * 100
  
  tibble(
    Modelo = nome_modelo, N_Validacao = nrow(base_val),
    Mediana_Subvalorizado = if(length(med_sub)>0) med_sub else NA_real_,
    Mediana_Supervalorizado = if(length(med_sup)>0) med_sup else NA_real_,
    p_valor_Kruskal = p_kruskal,
    Taxa_Acerto_Subvalorizado   = tx_sub,   # % dos "subvalorizados" que valorizaram acima da mediana geral
    Taxa_Acerto_Supervalorizado = tx_sup,   # % dos "supervalorizados" que valorizaram abaixo da mediana geral
    Taxa_Acerto_Direcional = mean(c(tx_sub, tx_sup), na.rm = TRUE)   # média das duas taxas
  )
}

# Roda a validação para cada modelo e ordena do mais para o menos acertivo
tabela_validacao <- bind_rows(lapply(unique(tabela_final$Modelo), validar_modelo, tab_final = tabela_final, val_real = valorizacao_real)) %>%
  arrange(desc(Taxa_Acerto_Direcional))

write_csv(tabela_validacao, file.path(DIR_BASE, "tcc_validacao_prospectiva.csv"))
write_csv(tabela_final, file.path(DIR_BASE, "tabela_final.csv"))   # cópia da tabela final (nome alternativo)

# Encerra o cluster paralelo e libera os núcleos
stopCluster(cl)
