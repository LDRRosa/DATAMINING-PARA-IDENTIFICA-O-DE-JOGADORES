# =============================================================================
# Gráficos das variáveis preditoras selecionadas e excluídas por posição
# -----------------------------------------------------------------------------
# ATENÇÃO: este script usa o objeto `dados_completos`, criado em
# scripts/cenario1_previsao_bruta.R. Execute o Cenário 1 antes, na mesma sessão
# do R, ou carregue o objeto de outra forma.
# =============================================================================


library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(forcats)

# ============================================================
# 1. Variáveis selecionadas no script
# ============================================================

features_gk <- c(
  "Age", "90s", "Saves", "Save%", "CS", "CS%", "PKA",
  "PKsv", "GA90", "PSxG+/-", "Stp%", "#OPA/90",
  "AvgDist", "Cmp%_stats_keeper_adv"
)

features_df <- c(
  "Age", "90s", "Tkl", "TklW", "Int", "Clr", "Err",
  "Recov", "Blocks_stats_defense", "Tkl+Int", "Cmp%",
  "PrgP", "PrgC", "Won%"
)

features_mf <- c(
  "Age", "90s", "PrgP", "PrgC", "KP", "Cmp%", "xA",
  "PPA", "Touches", "Carries", "Recov", "Tkl+Int",
  "SCA90", "GCA90"
)

features_fw <- c(
  "Age", "90s", "Gls", "Ast", "xG", "xAG", "npxG",
  "PrgC", "Touches", "KP", "SCA90", "GCA90",
  "Sh/90", "SoT%", "Succ%"
)

selecionadas <- list(
  GK = features_gk,
  DF = features_df,
  MF = features_mf,
  FW = features_fw
)

# ============================================================
# 2. Variáveis estatísticas disponíveis na base
#    (requer o objeto dados_completos já criado)
# ============================================================

colunas_nao_estatisticas <- c(
  "Player", "Pos", "Pos_Principal", "Squad", "Comp",
  "player_id", "Valor_Mercado", "Valor_Mercado_log",
  "Nome_FB_Limpo", "Nome_TM", "Nome_TM_Limpo", "Dist_Nome"
)

variaveis_disponiveis <- setdiff(names(dados_completos), colunas_nao_estatisticas)

selecionadas <- lapply(selecionadas, function(x) intersect(x, variaveis_disponiveis))

# ============================================================
# 3. Base longa: todas as variáveis x todas as posições
# ============================================================

rotulos_pos <- c(
  GK = "Goleiros (GK)",
  DF = "Defensores (DF)",
  MF = "Meio-campistas (MF)",
  FW = "Atacantes (FW)"
)

base_long <- expand_grid(
  Posicao  = names(selecionadas),
  Variavel = variaveis_disponiveis
) %>%
  mutate(
    Selecionada = unname(mapply(
      function(p, v) v %in% selecionadas[[p]],
      Posicao, Variavel
    ))
  ) %>%
  group_by(Variavel) %>%
  mutate(n_posicoes = sum(Selecionada)) %>%
  ungroup() %>%
  mutate(
    Status  = factor(if_else(Selecionada, "Selecionada", "Excluída"),
                     levels = c("Selecionada", "Excluída")),
    Posicao = factor(Posicao, levels = names(rotulos_pos), labels = rotulos_pos)
  )

cores <- c("Selecionada" = "#2E8B57", "Excluída" = "#D9D9D9")

# ============================================================
# 4. Gráfico 1: resumo (quantas variáveis entraram e saíram)
# ============================================================

resumo <- base_long %>%
  count(Posicao, Status) %>%
  group_by(Posicao) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup()

g_resumo <- ggplot(resumo, aes(x = Posicao, y = n, fill = Status)) +
  geom_col(width = 0.65, color = "white") +
  geom_text(
    aes(label = paste0(n, "\n(", round(pct * 100), "%)")),
    position = position_stack(vjust = 0.5),
    size = 3.4,
    color = "black"
  ) +
  scale_fill_manual(values = cores, name = NULL) +
  labs(
    title = "Variáveis selecionadas e excluídas por posição",
    subtitle = paste0("Total de ", length(variaveis_disponiveis),
                      " variáveis estatísticas candidatas na base"),
    x = NULL,
    y = "Número de variáveis"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "bottom",
    panel.grid.major.x = element_blank()
  )

print(g_resumo)

# ============================================================
# 5. Gráfico 2: mapa de quais variáveis cada posição usa
#    (só variáveis usadas por pelo menos uma posição)
# ============================================================

dados_mapa <- base_long %>% filter(n_posicoes > 0)

ordem_vars <- dados_mapa %>%
  distinct(Variavel, n_posicoes) %>%
  arrange(n_posicoes, desc(Variavel)) %>%   # mais compartilhadas no topo
  pull(Variavel)

dados_mapa <- dados_mapa %>%
  mutate(Variavel = factor(Variavel, levels = ordem_vars))

g_mapa <- ggplot(dados_mapa, aes(x = Posicao, y = Variavel, fill = Status)) +
  geom_tile(color = "white", linewidth = 0.6) +
  scale_fill_manual(values = cores, name = NULL) +
  scale_x_discrete(position = "top") +
  labs(
    title = "Mapa de variáveis preditoras por posição",
    subtitle = "Variáveis usadas por ao menos uma posição, ordenadas pelo número de posições que as utilizam",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 9),
    axis.text.x = element_text(face = "bold"),
    axis.text.y = element_text(size = 9),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )

print(g_mapa)

# ============================================================
# 6. Salvar (ajuste a pasta se quiser)
# ============================================================

 ggsave("grafico_resumo_variaveis.png", g_resumo, width = 8, height = 5, dpi = 300)
 ggsave("grafico_mapa_variaveis.png",   g_mapa,   width = 8, height = 8, dpi = 300)
