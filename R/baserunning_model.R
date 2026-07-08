## ============================================================
## Baserunning Value Model (BsR)
## Components:
##   1. Batter-Advancement-Effect (BAE)  -- own baserunning on batted ball
##   2. Runner-Advancement-Effect (RAE)  -- existing runner advancement on BIP
##   3. SB/CS value
## Sum(1,2,3) = Total BsR per player
## ============================================================

library(baseballr)
library(dplyr)
library(purrr)
library(xgboost)
library(tidyr)

## ------------------------------------------------------------
## 0. PULL DATA
## statcast_search is capped ~ a few days per call, so chunk by week.
## Pull as much as possible: default 2021-01-01 through most recent complete season.
## ------------------------------------------------------------

pull_statcast_range <- function(start_date, end_date) {
  dates <- seq(as.Date(start_date), as.Date(end_date), by = "4 days")
  out <- map_dfr(seq_along(dates), function(i) {
    d1 <- dates[i]
    d2 <- min(d1 + 3, as.Date(end_date))
    message(sprintf("Pulling %s to %s", d1, d2))
    tryCatch(
      baseballr::statcast_search(start_date = as.character(d1),
                                  end_date   = as.character(d2)),
      error = function(e) { message("  failed: ", e$message); NULL }
    )
  })
  out
}

## Example (uncomment / edit years as needed):
# raw_2021 <- pull_statcast_range("2021-04-01", "2021-10-03")
# raw_2022 <- pull_statcast_range("2022-04-07", "2022-10-05")
# raw_2023 <- pull_statcast_range("2023-03-30", "2023-10-01")
# raw_2024 <- pull_statcast_range("2024-03-28", "2024-09-29")
# raw_2025 <- pull_statcast_range("2025-03-27", "2025-09-28")
# raw_all <- bind_rows(raw_2021, raw_2022, raw_2023, raw_2024, raw_2025)
# saveRDS(raw_all, "statcast_raw_all.rds")

## ------------------------------------------------------------
## 1. SHARED PREP: spray angle + batted-ball-in-play filter
## ------------------------------------------------------------

add_spray_angle <- function(df) {
  df %>%
    mutate(
      spray_angle = atan((hc_x - 125.42) / (198.27 - hc_y)) * 180 / pi,
      # flip sign for RHH so + = pull, - = oppo consistently across handedness
      spray_angle = if_else(stand == "R", -spray_angle, spray_angle)
    )
}

## outcome bucket used by both batted-ball models
classify_bip_outcome <- function(events) {
  case_when(
    events %in% c("single") ~ "1B",
    events %in% c("double") ~ "2B",
    events %in% c("triple") ~ "3B",
    events %in% c("home_run") ~ "HR",
    events %in% c(
      "field_out","force_out","grounded_into_double_play","double_play",
      "fielders_choice_out","fielders_choice","sac_fly","sac_bunt",
      "triple_play","sac_fly_double_play","sac_bunt_double_play"
    ) ~ "OUT",
    TRUE ~ NA_character_
  )
}

bip_base <- function(raw) {
  raw %>%
    add_spray_angle() %>%
    mutate(bip_outcome = classify_bip_outcome(events),
           season = as.integer(substr(as.character(game_date), 1, 4))) %>%
    filter(
      type == "X",                     # ball in play
      !is.na(bip_outcome), bip_outcome != "HR",
      !is.na(launch_speed), !is.na(launch_angle), !is.na(spray_angle)
    )
}

RUN_VAL_BATTER <- c(OUT = -0.28, `1B` = 0.46, `2B` = 0.77, `3B` = 1.03)

## ------------------------------------------------------------
## MODEL 1: Batter's own baserunning-adjusted outcome value (BAE)
## ------------------------------------------------------------

## xgboost 2.x renamed watchlist->evals and dropped predict(reshape=); handle both.
safe_predict_probs <- function(model, X, levels) {
  raw_pred <- predict(model, newdata = X)
  if (is.matrix(raw_pred)) {
    probs <- raw_pred
  } else {
    probs <- matrix(raw_pred, ncol = length(levels), byrow = TRUE)
  }
  colnames(probs) <- levels
  probs
}

## ------------------------------------------------------------
## Per-class Platt scaling, fit on the held-out val split that's already
## carved out at training time. Corrects XGBoost's tendency to push
## multiclass softprob toward overconfident 0/1, which otherwise inflates
## the actual-minus-expected residuals used everywhere in this pipeline.
## ------------------------------------------------------------

fit_platt_calibrators <- function(raw_probs_val, y_val_int, levels) {
  eps <- 1e-6
  calibrators <- lapply(seq_along(levels), function(i) {
    p <- pmin(pmax(raw_probs_val[, i], eps), 1 - eps)
    y_bin <- as.integer(y_val_int == (i - 1L))
    glm(y_bin ~ qlogis(p), family = binomial())
  })
  names(calibrators) <- levels
  calibrators
}

apply_platt_calibrators <- function(raw_probs, calibrators, levels) {
  eps <- 1e-6
  calibrated <- sapply(seq_along(levels), function(i) {
    p <- pmin(pmax(raw_probs[, i], eps), 1 - eps)
    predict(calibrators[[i]], newdata = data.frame(p = p), type = "response")
  })
  colnames(calibrated) <- levels
  calibrated / rowSums(calibrated)   # renormalize so rows sum to 1
}

MODEL1_LEVELS <- c("OUT","1B","2B","3B")   # index 0..3 for xgboost labels

## fixed team-code set so train/score dummy columns always align, even if a
## given season subset doesn't include every park
PARK_LEVELS <- c("ARI","ATL","BAL","BOS","CHC","CWS","CIN","CLE","COL","DET",
                  "HOU","KC","LAA","LAD","MIA","MIL","MIN","NYM","NYY","OAK",
                  "PHI","PIT","SD","SEA","SF","STL","TB","TEX","TOR","WSH")

## manual one-hot (never drops rows, unlike model.matrix() which silently
## drops rows with NA in the factor)
one_hot <- function(x, levels, prefix) {
  x <- as.character(x)
  m <- sapply(levels, function(lv) as.integer(!is.na(x) & x == lv))
  colnames(m) <- paste0(prefix, levels)
  m
}

model1_matrix <- function(df) {
  park_dummies <- one_hot(df$home_team, PARK_LEVELS, "park_")
  cbind(as.matrix(df[, c("launch_speed","launch_angle","spray_angle")]), park_dummies)
}

fit_model1 <- function(bip, nrounds = 300, eta = 0.05, max_depth = 4,
                        early_stopping_rounds = 20, val_frac = 0.15) {
  bip$bip_outcome <- factor(bip$bip_outcome, levels = MODEL1_LEVELS)
  X <- model1_matrix(bip)
  y <- as.integer(bip$bip_outcome) - 1L   # 0-indexed

  set.seed(1)
  n <- nrow(X)
  val_idx <- sample(n, size = floor(n * val_frac))
  dtrain <- xgb.DMatrix(X[-val_idx, , drop = FALSE], label = y[-val_idx])
  dval   <- xgb.DMatrix(X[val_idx, , drop = FALSE],  label = y[val_idx])

  booster <- xgb.train(
    params = list(objective = "multi:softprob", num_class = length(MODEL1_LEVELS),
                  eta = eta, max_depth = max_depth, eval_metric = "mlogloss"),
    data = dtrain, nrounds = nrounds,
    evals = list(train = dtrain, val = dval),
    early_stopping_rounds = early_stopping_rounds, verbose = 0
  )

  raw_val_probs <- safe_predict_probs(booster, X[val_idx, , drop = FALSE], MODEL1_LEVELS)
  calibrators <- fit_platt_calibrators(raw_val_probs, y[val_idx], MODEL1_LEVELS)

  list(booster = booster, calibrators = calibrators)
}

score_model1 <- function(bip, model) {
  X <- model1_matrix(bip)
  raw_probs <- safe_predict_probs(model$booster, X, MODEL1_LEVELS)
  probs <- apply_platt_calibrators(raw_probs, model$calibrators, MODEL1_LEVELS)
  expected_rv <- as.numeric(probs %*% RUN_VAL_BATTER[MODEL1_LEVELS])
  actual_rv   <- RUN_VAL_BATTER[as.character(bip$bip_outcome)]
  bip %>%
    mutate(expected_rv = expected_rv,
           actual_rv   = as.numeric(actual_rv),
           bae_play    = actual_rv - expected_rv)
}

batter_bae_totals <- function(scored) {
  scored %>%
    group_by(batter, season) %>%
    summarise(BAE_AE = sum(bae_play), n_bip = n(), .groups = "drop")
}

## ------------------------------------------------------------
## MODEL 2: Existing-runner advancement effect (RAE)
## Needs: starting base, post-play base state, whether the runner scored/out.
## Statcast fields: on_1b, on_2b, on_3b (pre-pitch runner ids),
##                   post_on_1b/2b/3b if present (baseballr sometimes lacks these
##                   -- if absent they must be derived from des/events; the
##                   fallback below uses the most common available approach).
## ------------------------------------------------------------

RUN_VAL_RUNNER <- c(OUT = -0.4, STAY = -0.1, ADV1 = 0.2, ADV2 = 0.4, ADV3 = 0.7)

## baseballr::statcast_search does NOT return post_on_1b/2b/3b. Reconstruct the
## post-play base/out state from the next pitch-row in the same game (Statcast
## is pitch-level, so the first pitch of the next plate appearance carries the
## resulting on_1b/2b/3b + score state). If the next row starts a new half-inning,
## the on-base fields reset to empty and can't be used -- fall back to score diff.
add_post_state <- function(raw) {
  raw %>%
    arrange(game_pk, at_bat_number, pitch_number) %>%
    group_by(game_pk) %>%
    mutate(
      next_on_1b     = lead(on_1b),
      next_on_2b     = lead(on_2b),
      next_on_3b     = lead(on_3b),
      next_inning    = lead(inning),
      next_topbot    = lead(inning_topbot),
      next_bat_score = lead(bat_score),
      next_fld_score = lead(fld_score),
      same_half      = !is.na(next_inning) &
                        next_inning == inning & next_topbot == inning_topbot
    ) %>%
    ungroup()
}

## Determine each pre-existing runner's fate for one row (one plate appearance/BIP)
## start_base: 1, 2, or 3
## returns outcome in {OUT, STAY, ADV1, ADV2, ADV3} or NA if indeterminate
resolve_runner_fate <- function(runner_id, start_base,
                                 next_on_1b, next_on_2b, next_on_3b, same_half,
                                 bat_score_before, next_bat_score,
                                 fld_score_before, next_fld_score,
                                 is_home_half) {
  if (is.na(runner_id)) return(NA_character_)

  # scored? offense's score increased between this row and the next row.
  runs_scored <- if (is_home_half) (next_fld_score - fld_score_before) else (next_bat_score - bat_score_before)
  if (!is.na(runs_scored) && runs_scored > 0) {
    # ambiguous if >1 run scored on the play and multiple runners present --
    # resolved at the caller by checking landed-base first when same_half is TRUE.
  }

  landed <- NA_real_
  if (isTRUE(same_half)) {
    landed <- case_when(
      !is.na(next_on_1b) && next_on_1b == runner_id ~ 1,
      !is.na(next_on_2b) && next_on_2b == runner_id ~ 2,
      !is.na(next_on_3b) && next_on_3b == runner_id ~ 3,
      TRUE ~ NA_real_
    )
  }

  if (!is.na(landed)) {
    bases_moved <- landed - start_base
    if (bases_moved <= 0) return("STAY")
    return(paste0("ADV", bases_moved))
  }

  # not found on base post-play: either scored or was put out.
  if (!is.na(runs_scored) && runs_scored > 0) {
    bases_moved <- 4 - start_base
    return(paste0("ADV", min(bases_moved, 3)))
  }
  if (!isTRUE(same_half)) {
    # half-inning ended right after this play and runner never scored -> out
    # (stranded runners can't occur here since the half literally ended, and a
    # non-force-out third out still counts this runner as failing to advance/out
    # for our per-play purposes only when they're also off the bases; if they're
    # simply stranded this branch is rare because outs_when_up will have hit 3
    # only via an out somewhere on the bases or the batter -- treat as OUT is the
    # conservative default here since we cannot confirm STAY without post-state).
  }
  return("OUT")
}

build_runner_events <- function(raw_with_spray) {
  d <- raw_with_spray %>%
    add_post_state() %>%
    mutate(bip_outcome = classify_bip_outcome(events),
           season = as.integer(substr(as.character(game_date), 1, 4))) %>%
    filter(type == "X", !is.na(bip_outcome), bip_outcome != "HR",
           !is.na(launch_speed), !is.na(launch_angle), !is.na(spray_angle))

  d <- d %>% mutate(is_home_half = inning_topbot == "Bot")

  runner1 <- d %>% filter(!is.na(on_1b)) %>%
    mutate(start_base = 1, runner_id = on_1b)
  runner2 <- d %>% filter(!is.na(on_2b)) %>%
    mutate(start_base = 2, runner_id = on_2b)
  runner3 <- d %>% filter(!is.na(on_3b)) %>%
    mutate(start_base = 3, runner_id = on_3b)

  runners <- bind_rows(runner1, runner2, runner3) %>%
    rowwise() %>%
    mutate(fate = resolve_runner_fate(
      runner_id, start_base,
      next_on_1b, next_on_2b, next_on_3b, same_half,
      bat_score, next_bat_score, fld_score, next_fld_score, is_home_half
    )) %>%
    ungroup() %>%
    filter(!is.na(fate)) %>%
    mutate(fate = factor(fate, levels = c("OUT","STAY","ADV1","ADV2","ADV3")))

  runners
}

MODEL2_LEVELS <- c("OUT","STAY","ADV1","ADV2","ADV3")

## start_base + park (home_team) one-hot, built consistently for train/score
## so column sets always match
model2_matrix <- function(df) {
  base_dummies <- one_hot(df$start_base, c(1,2,3), "base")
  park_dummies <- one_hot(df$home_team, PARK_LEVELS, "park_")
  cbind(as.matrix(df[, c("launch_speed","launch_angle","spray_angle")]),
        base_dummies, park_dummies)
}

fit_model2 <- function(runner_events, nrounds = 300, eta = 0.05, max_depth = 4,
                        early_stopping_rounds = 20, val_frac = 0.15) {
  runner_events$fate <- factor(runner_events$fate, levels = MODEL2_LEVELS)
  X <- model2_matrix(runner_events)
  y <- as.integer(runner_events$fate) - 1L

  set.seed(1)
  n <- nrow(X)
  val_idx <- sample(n, size = floor(n * val_frac))
  dtrain <- xgb.DMatrix(X[-val_idx, , drop = FALSE], label = y[-val_idx])
  dval   <- xgb.DMatrix(X[val_idx, , drop = FALSE],  label = y[val_idx])

  booster <- xgb.train(
    params = list(objective = "multi:softprob", num_class = length(MODEL2_LEVELS),
                  eta = eta, max_depth = max_depth, eval_metric = "mlogloss"),
    data = dtrain, nrounds = nrounds,
    evals = list(train = dtrain, val = dval),
    early_stopping_rounds = early_stopping_rounds, verbose = 0
  )

  raw_val_probs <- safe_predict_probs(booster, X[val_idx, , drop = FALSE], MODEL2_LEVELS)
  calibrators <- fit_platt_calibrators(raw_val_probs, y[val_idx], MODEL2_LEVELS)

  list(booster = booster, calibrators = calibrators)
}

score_model2 <- function(runner_events, model) {
  X <- model2_matrix(runner_events)
  raw_probs <- safe_predict_probs(model$booster, X, MODEL2_LEVELS)
  probs <- apply_platt_calibrators(raw_probs, model$calibrators, MODEL2_LEVELS)
  rv <- RUN_VAL_RUNNER[MODEL2_LEVELS]
  expected_rv <- as.numeric(probs %*% rv)
  actual_rv   <- RUN_VAL_RUNNER[as.character(runner_events$fate)]
  runner_events %>%
    mutate(expected_rv = expected_rv,
           actual_rv   = as.numeric(actual_rv),
           rae_play    = actual_rv - expected_rv)
}

runner_rae_totals <- function(scored2) {
  scored2 %>%
    group_by(runner_id, season) %>%
    summarise(RAE_AE = sum(rae_play), n_baserunning_events = n(), .groups = "drop") %>%
    rename(batter = runner_id)   # runner_id is the player's MLBAM id, same key space as `batter`
}

## ------------------------------------------------------------
## MODEL 3: SB / CS value
## ------------------------------------------------------------

sbcs_totals_from_statcast <- function(raw) {
  raw %>%
    filter(events %in% c(
      "stolen_base_2b","stolen_base_3b","stolen_base_home",
      "caught_stealing_2b","caught_stealing_3b","caught_stealing_home",
      "pickoff_caught_stealing_2b","pickoff_caught_stealing_3b","pickoff_caught_stealing_home"
    )) %>%
    mutate(
      is_sb = grepl("^stolen_base", events),
      runner_id = case_when(
        grepl("2b", events) ~ on_1b,   # runner who *was* on 1B attempts 2B, etc.
        grepl("3b", events) ~ on_2b,
        grepl("home", events) ~ on_3b
      ),
      rv = if_else(is_sb, 0.2, -0.45)
    ) %>%
    filter(!is.na(runner_id)) %>%
    group_by(runner_id) %>%
    summarise(SBCS_value = sum(rv), n_sb = sum(is_sb), n_cs = sum(!is_sb), .groups = "drop") %>%
    rename(batter = runner_id)
}

## Model 3 via FanGraphs season leaderboards (matches how N already pulls
## SB/CS elsewhere). fg_batter_leaders() rows carry a FanGraphs playerid, not
## the MLBAM id used everywhere else in this pipeline (`batter`), so map via
## baseballr's Chadwick register before joining.
sbcs_totals_from_fangraphs <- function(fg_leaders) {
  id_map <- baseballr::chadwick_player_lu() %>%
    filter(!is.na(key_fangraphs), !is.na(key_mlbam)) %>%
    distinct(key_fangraphs, key_mlbam)

  fg_leaders %>%
    transmute(key_fangraphs = as.character(playerid),
              season = Season, SB = SB, CS = CS) %>%
    mutate(SBCS_value = SB * 0.2 + CS * -0.45) %>%
    left_join(id_map %>% mutate(key_fangraphs = as.character(key_fangraphs)),
              by = "key_fangraphs") %>%
    filter(!is.na(key_mlbam)) %>%
    group_by(batter = key_mlbam, season) %>%
    summarise(SBCS_value = sum(SBCS_value), n_sb = sum(SB), n_cs = sum(CS), .groups = "drop")
}

## pull SB/CS for one or more seasons, e.g. pull_fg_sbcs(2021:2025)
pull_fg_sbcs <- function(years) {
  map_dfr(years, function(yr) {
    baseballr::fg_batter_leaders(startseason = yr, endseason = yr, qual = 0) %>%
      mutate(Season = yr)
  })
}

## ------------------------------------------------------------
## 4. FULL PIPELINE
## ------------------------------------------------------------

run_baserunning_pipeline <- function(raw, fg_sbcs_years) {
  raw_sp <- add_spray_angle(raw)

  ## Model 1
  bip1 <- bip_base(raw)
  m1   <- fit_model1(bip1)
  s1   <- score_model1(bip1, m1)
  t1   <- batter_bae_totals(s1)

  ## Model 2
  re2  <- build_runner_events(raw_sp)
  m2   <- fit_model2(re2)
  s2   <- score_model2(re2, m2)
  t2   <- runner_rae_totals(s2)

  ## Model 3 -- FanGraphs SB/CS leaderboards
  fg_leaders <- pull_fg_sbcs(fg_sbcs_years)
  t3   <- sbcs_totals_from_fangraphs(fg_leaders)

  full_join(t1, t2, by = c("batter","season")) %>%
    full_join(t3, by = c("batter","season")) %>%
    mutate(across(c(BAE_AE, RAE_AE, SBCS_value), ~ replace_na(., 0))) %>%
    mutate(Total_BsR = BAE_AE + RAE_AE + SBCS_value) %>%
    arrange(season, desc(Total_BsR))
}

## Usage:
# raw_all <- readRDS("statcast_raw_all.rds")
# bsr <- run_baserunning_pipeline(raw_all, fg_sbcs_years = 2021:2025)
# write.csv(bsr, "bsr_results.csv", row.names = FALSE)
