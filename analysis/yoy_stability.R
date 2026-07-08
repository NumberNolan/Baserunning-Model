library(dplyr)
library(readr)
library(tidyr)

bsr <- read_csv("bsr_results.csv", show_col_types = FALSE) %>%
  mutate(season = as.integer(season))

yoy_corr <- function(bsr, min_bip = 0) {
  wide <- bsr %>%
    filter(n_bip >= min_bip) %>%
    select(batter, season, Total_BsR) %>%
    arrange(batter, season)

  paired <- wide %>%
    group_by(batter) %>%
    arrange(season, .by_group = TRUE) %>%
    mutate(next_season = lead(season), next_bsr = lead(Total_BsR)) %>%
    ungroup() %>%
    filter(!is.na(next_bsr), next_season == season + 1)

  tibble(min_bip = min_bip,
         n_pairs = nrow(paired),
         r = suppressWarnings(cor(paired$Total_BsR, paired$next_bsr)))
}

thresholds <- c(0, 20, 50, 100, 150, 200, 300)
results <- bind_rows(lapply(thresholds, yoy_corr, bsr = bsr))
print(results)
