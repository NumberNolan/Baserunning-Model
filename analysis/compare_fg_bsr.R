library(baseballr)
library(dplyr)
library(readr)
library(purrr)

bsr <- read_csv("bsr_results.csv", show_col_types = FALSE) %>%
  mutate(season = as.integer(season))

## pull FanGraphs BsR (their own baserunning value column) for the same seasons
## NOTE: column is `BaseRunning` in the returned tibble, not `BsR`.
## fg_batter_leaders() already returns xMLBAMID, so no Chadwick round-trip needed.
pull_fg_bsr <- function(years) {
  map_dfr(years, function(yr) {
    baseballr::fg_batter_leaders(startseason = yr, endseason = yr, qual = 0) %>%
      transmute(batter = xMLBAMID, season = yr, fg_BsR = BaseRunning)
  })
}

years <- sort(unique(bsr$season))
fg_bsr <- pull_fg_bsr(years) %>% filter(!is.na(batter))

compare <- bsr %>%
  select(batter, season, Total_BsR) %>%
  inner_join(fg_bsr, by = c("batter","season"))

cat("n matched player-seasons:", nrow(compare), "\n")

fit <- lm(Total_BsR ~ fg_BsR, data = compare)
cat("R^2:", summary(fit)$r.squared, "\n")
cat("correlation r:", cor(compare$Total_BsR, compare$fg_BsR), "\n")

# optional: quick look at biggest divergences
compare %>%
  mutate(diff = Total_BsR - fg_BsR) %>%
  arrange(desc(abs(diff))) %>%
  head(15) %>%
  print()
