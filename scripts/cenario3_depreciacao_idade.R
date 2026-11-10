# =============================================================================
# TCC — Estimativa de valor de mercado de jogadores de futebol
# CENÁRIO 3: previsões fora da partição (OOF) + depreciação por idade
# (SEM a variável Comp e SEM corte de idade na validação)
# -----------------------------------------------------------------------------
# Objetivo: identificar jogadores SUBVALORIZADOS / SUPERVALORIZADOS comparando
# o valor de mercado real (Transfermarkt) com o valor que um modelo de ML
# estima a partir do desempenho esportivo (FBref, temporada 2024/25).
# =============================================================================

# --- 0. Bibliotecas ----------------------------------------------------------
library(dplyr)        # manipulação de dados
library(readr)        # leitura/escrita de CSV
library(caret)        # framework de treino e validação
library(ranger)       # Random Forest
library(xgboost)      # XGBoost
library(lightgbm)     # LightGBM
library(lubridate)    # manipulação de datas
library(stringdist)   # distância Jaro-Winkler
library(doParallel)   # paralelismo
library(parallel)

# --- Configurações Globais ---------------------------------------------------
SEMENTE                 <- 42
DATA_CORTE_TREINO       <- as.Date("2025-07-31")
N_FOLDS                 <- 5
LIMIAR_MARGEM           <- 0.20  # Margem de 20% para classificar Sub/Supervalorizado
DIR_BASE                <- "C:/Users/leand/Downloads/TCC Leandro/TCC"
SUFIXO_SAIDA            <- "_cenario3"   #evita sobrescrever os CSVs dos outros cenários

# --- Curva de Depreciação por Idade (multiplicador sobre o valor estimado) ---
# A redução é LINEAR, em pontos percentuais por ano dentro de cada faixa:
#   <= 24 anos: 1,00
#   25-27: 1,00 - 0,07 x (idade - 24)  -> 0,93 | 0,86 | 0,79
#   28-30: 0,79 - 0,12 x (idade - 27)  -> 0,67 | 0,55 | 0,43
#   31-33: 0,43 - 0,09 x (idade - 30)  -> 0,34 | 0,25 | 0,16
#   34+  : 0,16 - 0,03 x (idade - 33), com piso de 0,05 -> 0,13 | 0,10 | 0,07 | 0,05 (a partir dos 37)   # <<< ALTERADO (comentário)
calcular_multiplicador_idade <- function(idade) {
  case_when(
    idade <= 24 ~ 1.00,
    idade <= 27 ~ 1.00 - (idade - 24) * 0.07,
    idade <= 30 ~ 0.79 - (idade - 27) * 0.12,
    idade <= 33 ~ 0.43 - (idade - 30) * 0.09,
    TRUE        ~ pmax(0.05, 0.16 - (idade - 33) * 0.03)
  )
}

# --- Configuração do Paralelismo ---------------------------------------------
N_CORES <- max(1, detectCores() - 1)
cl <- makeCluster(N_CORES)
registerDoParallel(cl)
message("Paralelismo ativado com ", N_CORES, " núcleos.")

# --- 1. Dados Financeiros (Transfermarkt) ------------------------------------
players_tm    <- read_csv(file.path(DIR_BASE, "players.csv"), show_col_types = FALSE)
valuations_tm <- read_csv(file.path(DIR_BASE, "player_valuations.csv"), show_col_types = FALSE) %>%
  mutate(date = as.Date(date))

DATA_VALIDACAO <- max(valuations_tm$date, na.rm = TRUE)

valores_corte <- valuations_tm %>%
  filter(date <= DATA_CORTE_TREINO) %>%
  group_by(player_id) %>%
  slice_max(order_by = date, n = 1) %>%
  ungroup() %>%
  select(player_id, Valor_Mercado = market_value_in_eur)

base_valores <- players_tm %>%
  inner_join(valores_corte, by = "player_id") %>%
  select(player_id, Nome_TM = name, Valor_Mercado) %>%
  mutate(Nome_TM_Limpo = tolower(iconv(Nome_TM, from = "UTF-8", to = "ASCII//TRANSLIT")))

# --- 2. Estatísticas Esportivas (FBref) --------------------------------------
base_stats <- read_csv(file.path(DIR_BASE, "players_data-2024_2025 (1).csv"), show_col_types = FALSE) %>%
  mutate(
    Pos_Principal = substr(Pos, 1, 2),
    Nome_FB_Limpo = tolower(iconv(Player, from = "UTF-8", to = "ASCII//TRANSLIT"))
  ) %>%
  filter(`90s` >= 5.0)

# Variáveis preditoras por posição. 'Comp' (liga) NÃO é preditor.   # <<< ALTERADO
features_gk <- c("Age", "90s", "Saves", "Save%", "CS", "CS%", "PKA", "PKsv", "GA90", "PSxG+/-", "Stp%", "#OPA/90", "AvgDist", "Cmp%_stats_keeper_adv")   # <<< ALTERADO
features_df <- c("Age", "90s", "Tkl", "TklW", "Int", "Clr", "Err", "Recov", "Blocks_stats_defense", "Tkl+Int", "Cmp%", "PrgP", "PrgC", "Won%")         # <<< ALTERADO
features_mf <- c("Age", "90s", "PrgP", "PrgC", "KP", "Cmp%", "xA", "PPA", "Touches", "Carries", "Recov", "Tkl+Int", "SCA90", "GCA90")                  # <<< ALTERADO
features_fw <- c("Age", "90s", "Gls", "Ast", "xG", "xAG", "npxG", "PrgC", "Touches", "KP", "SCA90", "GCA90", "Sh/90", "SoT%", "Succ%")                 # <<< ALTERADO

# --- 3. Fuzzy Join (cruzamento por similaridade de nome) ---------------------
matriz_distancia <- stringdistmatrix(base_stats$Nome_FB_Limpo, base_valores$Nome_TM_Limpo, method = "jw", nthread = N_CORES)

melhor_match_idx <- apply(matriz_distancia, 1, which.min)
distancia_minima <- apply(matriz_distancia, 1, min)

dados_completos <- base_stats %>%
  mutate(
    player_id     = base_valores$player_id[melhor_match_idx],
    Valor_Mercado = base_valores$Valor_Mercado[melhor_match_idx],
    Dist_Nome     = distancia_minima
  ) %>%
  filter(Dist_Nome < 0.15) %>%
  select(-Dist_Nome, -Nome_FB_Limpo)

# --- 4. Modelagem e Funções de Treino ----------------------------------------

grade_xgb <- expand.grid(
  nrounds = c(100, 300), max_depth = c(2, 4), eta = c(0.05, 0.1),
  gamma = 0, colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8
)

params_lgbm <- list(
  objective = "regression", metric = "rmse", learning_rate = 0.05, num_leaves = 15,
  min_data_in_leaf = 8, feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 1,
  deterministic = FALSE, force_row_wise = TRUE, num_threads = N_CORES, verbosity = -1
)

gerar_folds <- function(y, k = N_FOLDS, seed = SEMENTE) {
  set.seed(seed)
  caret::createFolds(y, k = k, list = TRUE, returnTrain = TRUE)
}

# $oof          -> previsões out-of-fold (usadas nas métricas E na classificação)
# $predict_full -> modelo final com todos os jogadores; NÃO é usado na classificação
#                  (fica disponível para jogadores novos)
treinar_caret <- function(dados_treino, method, folds, tune_grid = NULL, seed = SEMENTE, ...) {
  controle <- trainControl(method = "cv", index = folds, savePredictions = "final", allowParallel = TRUE)
  set.seed(seed)
  args <- list(form = Valor_Mercado_log ~ ., data = dados_treino, method = method, trControl = controle)
  if (!is.null(tune_grid)) args$tuneGrid <- tune_grid
  args <- c(args, list(...))
  modelo <- do.call(train, args)
  
  list(
    oof = modelo$pred %>% arrange(rowIndex) %>% select(rowIndex, pred, obs),
    predict_full = function(novos_dados) predict(modelo, novos_dados)
  )
}

treinar_xgboost <- function(dados_treino, folds, grade = grade_xgb, seed = SEMENTE) {
  y <- dados_treino$Valor_Mercado_log
  X <- as.matrix(dados_treino %>% select(-Valor_Mercado_log))
  n <- nrow(dados_treino)
  
  parametros_xgb <- function(combo) {
    list(objective = "reg:squarederror", max_depth = combo$max_depth, eta = combo$eta,
         gamma = combo$gamma, colsample_bytree = combo$colsample_bytree,
         min_child_weight = combo$min_child_weight, subsample = combo$subsample,
         nthread = N_CORES)
  }
  
  oof_da_combo <- function(combo) {
    oof <- rep(NA_real_, n)
    for (idx_treino in folds) {
      idx_teste <- setdiff(seq_len(n), idx_treino)
      bst <- xgb.train(params = parametros_xgb(combo),
                       data = xgb.DMatrix(X[idx_treino, , drop = FALSE], label = y[idx_treino]),
                       nrounds = combo$nrounds, verbose = 0)
      oof[idx_teste] <- predict(bst, xgb.DMatrix(X[idx_teste, , drop = FALSE]))
    }
    oof
  }
  
  set.seed(seed)
  oof_por_combo <- lapply(seq_len(nrow(grade)), function(i) oof_da_combo(grade[i, ]))
  i_melhor <- which.min(vapply(oof_por_combo, function(p) RMSE(p, y), numeric(1)))
  
  modelo_final <- xgb.train(params = parametros_xgb(grade[i_melhor, ]),
                            data = xgb.DMatrix(X, label = y),
                            nrounds = grade[i_melhor, ]$nrounds, verbose = 0)
  
  list(
    oof = tibble(rowIndex = seq_len(n), pred = oof_por_combo[[i_melhor]], obs = y),
    predict_full = function(novos_dados) {
      predict(modelo_final, xgb.DMatrix(as.matrix(novos_dados %>% select(all_of(colnames(X))))))
    }
  )
}

treinar_lightgbm <- function(dados_treino, folds, params = params_lgbm, nrounds = 300, seed = SEMENTE) {
  y <- dados_treino$Valor_Mercado_log
  X <- as.matrix(dados_treino %>% select(-Valor_Mercado_log))
  n <- nrow(dados_treino)
  
  oof_pred <- rep(NA_real_, n)
  for (idx_treino in folds) {
    idx_teste <- setdiff(seq_len(n), idx_treino)
    modelo_fold <- lgb.train(params = c(params, list(seed = seed)),
                             data = lgb.Dataset(X[idx_treino, , drop = FALSE], label = y[idx_treino]),
                             nrounds = nrounds, verbose = -1)
    oof_pred[idx_teste] <- predict(modelo_fold, X[idx_teste, , drop = FALSE])
  }
  
  modelo_final <- lgb.train(params = c(params, list(seed = seed)),
                            data = lgb.Dataset(X, label = y), nrounds = nrounds, verbose = -1)
  
  list(
    oof = tibble(rowIndex = seq_len(n), pred = oof_pred, obs = y),
    predict_full = function(novos_dados) predict(modelo_final, as.matrix(novos_dados %>% select(all_of(colnames(X)))))
  )
}

especificacoes_modelos <- list(
  list(nome = "Random Forest", treinar = function(d, f) treinar_caret(d, method = "ranger", folds = f, importance = "impurity")),
  list(nome = "XGBoost",       treinar = function(d, f) treinar_xgboost(d, folds = f)),
  list(nome = "LightGBM",      treinar = function(d, f) treinar_lightgbm(d, folds = f))
)

# As métricas são calculadas sobre a previsão BRUTA (antes da depreciação por idade),
# portanto coincidem com as do Cenário 1.   # <<< ALTERADO (comentário)
calcular_metricas_oof <- function(oof) {
  tibble(
    RMSE_log = RMSE(oof$pred, oof$obs),
    MAE_log  = MAE(oof$pred, oof$obs),
    R2_log   = caret::R2(oof$pred, oof$obs, formula = "corr"),
    RMSE_eur = RMSE(expm1(oof$pred), expm1(oof$obs)),
    MAE_eur  = MAE(expm1(oof$pred), expm1(oof$obs))
  )
}

# --- Função Principal por Posição --------------------------------------------
avaliar_posicao_multi_modelo <- function(dados, posicao_alvo, variaveis_preditoras) {
  colunas_existentes <- intersect(variaveis_preditoras, names(dados))
  
  # 'Comp' é mantida aqui apenas para identificar o jogador na tabela de saída (Liga);
  # ela NÃO entra no treino.   # <<< ALTERADO (comentário)
  dados_posicao <- dados %>%
    filter(Pos_Principal == posicao_alvo) %>%
    select(Player, Pos_Principal, Squad, Comp, Age, player_id, Valor_Mercado, all_of(colunas_existentes)) %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), 0, .)),
           Valor_Mercado_log = log1p(Valor_Mercado))
  
  if (nrow(dados_posicao) == 0) stop("Nenhum dado sobrou após filtros.")
  
  # Base de treino: somente os preditores numéricos + alvo, como nos Cenários 1 e 2
  # (removido o passo com dummyVars, que só era necessário por causa da variável Comp).   # <<< ALTERADO
  dados_treino <- dados_posicao %>% select(all_of(colunas_existentes), Valor_Mercado_log)   # <<< ALTERADO
  
  folds <- gerar_folds(dados_treino$Valor_Mercado_log)
  
  resultados <- list()
  metricas   <- list()
  
  for (spec in especificacoes_modelos) {
    res_mod <- spec$treinar(dados_treino, folds)   # <<< ALTERADO: antes usava dados_treino_numeric
    
    metricas[[spec$nome]] <- calcular_metricas_oof(res_mod$oof) %>%
      mutate(Modelo = spec$nome, Posicao = posicao_alvo, .before = 1)
    
    # Garante 1 previsão OOF por jogador, na mesma ordem de dados_posicao, sem NA   # <<< ALTERADO
    stopifnot(nrow(res_mod$oof) == nrow(dados_posicao), !anyNA(res_mod$oof$pred))   # <<< ALTERADO
    
    # Previsão OUT-OF-FOLD + depreciação por idade + classificação
    resultados[[spec$nome]] <- dados_posicao %>%
      mutate(
        Valor_Estimado_Bruto = expm1(res_mod$oof$pred),   # <<< ALTERADO: antes era expm1(res_mod$predict_full(dados_treino_numeric))
        Fator_Idade          = calcular_multiplicador_idade(Age),
        Valor_Estimado       = Valor_Estimado_Bruto * Fator_Idade,
        Diferenca            = Valor_Estimado - Valor_Mercado,
        Status = case_when(
          Diferenca > (LIMIAR_MARGEM * Valor_Mercado)  ~ "Subvalorizado",
          Diferenca < -(LIMIAR_MARGEM * Valor_Mercado) ~ "Supervalorizado",
          TRUE ~ "Preço Justo"
        ),
        Modelo = spec$nome
      )
  }
  list(resultados = bind_rows(resultados), metricas = bind_rows(metricas))
}

# Execução por Posição
saida_fw <- avaliar_posicao_multi_modelo(dados_completos, "FW", features_fw)
saida_mf <- avaliar_posicao_multi_modelo(dados_completos, "MF", features_mf)
saida_df <- avaliar_posicao_multi_modelo(dados_completos, "DF", features_df)
saida_gk <- avaliar_posicao_multi_modelo(dados_completos, "GK", features_gk)

# --- 5. Exportação de Resultados ---------------------------------------------
tabela_metricas <- bind_rows(saida_fw$metricas, saida_mf$metricas, saida_df$metricas, saida_gk$metricas) %>% arrange(Posicao, Modelo)
write_csv(tabela_metricas, file.path(DIR_BASE, paste0("tcc_metricas_comparativas", SUFIXO_SAIDA, ".csv")))   # <<< ALTERADO

tabela_final <- bind_rows(saida_fw$resultados, saida_mf$resultados, saida_df$resultados, saida_gk$resultados) %>%
  select(Jogador = Player, Posição = Pos_Principal, Idade = Age, Clube = Squad, Liga = Comp,
         ID_Transfermarkt = player_id, Modelo, `Valor Real (€)` = Valor_Mercado,
         `Valor Estimado Bruto (€)` = Valor_Estimado_Bruto, `Fator Idade` = Fator_Idade,
         `Valor Estimado (€)` = Valor_Estimado, `Diferença (€)` = Diferenca, Status) %>%
  mutate(across(c(`Valor Estimado Bruto (€)`, `Valor Estimado (€)`, `Diferença (€)`), ~round(., 2)),
         `Fator Idade` = round(`Fator Idade`, 3)) %>%
  arrange(Modelo, desc(`Diferença (€)`))

write_csv(tabela_final, file.path(DIR_BASE, paste0("tcc_jogadores_analise_completa", SUFIXO_SAIDA, ".csv")))   # <<< ALTERADO

# --- 6. Validação Prospectiva (SEM FILTRO DE IDADE) --------------------------
valorizacao_real <- valuations_tm %>%
  filter(date <= DATA_VALIDACAO) %>%
  group_by(player_id) %>%
  slice_max(order_by = date, n = 1) %>%
  ungroup() %>%
  inner_join(valores_corte, by = "player_id", suffix = c("_Futuro", "_Corte")) %>%
  filter(date > DATA_CORTE_TREINO, Valor_Mercado > 0) %>%
  mutate(Valorizacao_Pct = (market_value_in_eur / Valor_Mercado - 1) * 100) %>%
  select(player_id, Valorizacao_Pct)

validar_modelo <- function(nome_modelo, tab_final, val_real) {
  base_val <- tab_final %>% 
    filter(Modelo == nome_modelo) %>% 
    inner_join(val_real, by = c("ID_Transfermarkt" = "player_id"))
  
  med_geral <- median(val_real$Valorizacao_Pct, na.rm = TRUE)
  
  resumo <- base_val %>% group_by(Status) %>% summarise(Mediana = median(Valorizacao_Pct, na.rm = TRUE), .groups = "drop")
  
  med_sub  <- resumo$Mediana[resumo$Status == "Subvalorizado"]
  med_sup  <- resumo$Mediana[resumo$Status == "Supervalorizado"]
  med_just <- resumo$Mediana[resumo$Status == "Preço Justo"]   
  
  p_kruskal <- if (n_distinct(base_val$Status[!is.na(base_val$Valorizacao_Pct)]) >= 2) {
    kruskal.test(Valorizacao_Pct ~ Status, data = base_val)$p.value
  } else NA_real_
  
  # Avaliação de todos os subvalorizados e supervalorizados (sem filtro de idade)
  sub_todos <- base_val %>% filter(Status == "Subvalorizado")
  sup_todos <- base_val %>% filter(Status == "Supervalorizado")
  
  tx_sub <- mean(sub_todos$Valorizacao_Pct > med_geral, na.rm = TRUE) * 100
  tx_sup <- mean(sup_todos$Valorizacao_Pct < med_geral, na.rm = TRUE) * 100
  
  tibble(
    Modelo = nome_modelo, 
    N_Validacao = nrow(base_val),
    N_Subvalorizados = nrow(sub_todos),
    N_Supervalorizados = nrow(sup_todos),                          
    N_PrecoJusto = sum(base_val$Status == "Preço Justo"),          
    Mediana_Subvalorizado = if(length(med_sub)>0) med_sub else NA_real_,
    Mediana_Supervalorizado = if(length(med_sup)>0) med_sup else NA_real_,
    Mediana_PrecoJusto = if(length(med_just)>0) med_just else NA_real_,   
    p_valor_Kruskal = p_kruskal,
    Taxa_Acerto_Subvalorizado = tx_sub,
    Taxa_Acerto_Supervalorizado = tx_sup,
    Taxa_Acerto_Direcional      = mean(c(tx_sub, tx_sup), na.rm = TRUE),
    Pct_Acima_Mediana_Geral = mean(base_val$Valorizacao_Pct > med_geral, na.rm = TRUE) * 100   #referência para as taxas de acerto
  )
}

tabela_validacao <- bind_rows(lapply(unique(tabela_final$Modelo), validar_modelo, tab_final = tabela_final, val_real = valorizacao_real)) %>%
  arrange(desc(Taxa_Acerto_Direcional))

write_csv(tabela_validacao, file.path(DIR_BASE, paste0("tcc_validacao_prospectiva", SUFIXO_SAIDA, ".csv")))
   
stopCluster(cl)