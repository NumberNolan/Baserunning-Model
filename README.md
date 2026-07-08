# Baserunning Value (BsR) Model

A three-component model estimating player baserunning value above expectation,
built on Statcast batted-ball data (exit velocity, launch angle, spray angle).

## Components

1. **BAE — Batter's own baserunning-adjusted outcome value**
   For each ball in play (HR excluded), a multiclass XGBoost model predicts
   `P(out, 1B, 2B, 3B)` from exit velocity, launch angle, spray angle, and
   park (`home_team`, one-hot). Expected run value is `P · run_value`, where
   `run_value = {out: -0.28, 1B: 0.46, 2B: 0.77, 3B: 1.03}`. Actual − expected,
   summed per player-season.

2. **RAE — Existing-runner advancement effect**
   For each runner on base when a ball is put in play (HR excluded), a
   multiclass XGBoost model predicts `P(out, stay, +1, +2, +3 bases)` from
   the same batted-ball features plus `start_base` and park, one-hot encoded.
   Run values: `{out: -0.4, stay: -0.1, +1: 0.2, +2: 0.4, +3: 0.7}`.
   Actual − expected, summed per player-season.

3. **SBCS — Stolen base / caught stealing value**
   Pulled from FanGraphs season leaderboards (`fg_batter_leaders()`), mapped
   to MLBAM player IDs via `chadwick_player_lu()`. `SB × 0.2 + CS × -0.45`.

`Total_BsR = BAE_AE + RAE_AE + SBCS_value`, aggregated by `(batter, season)`.
Player names are attached in the same pipeline (via the same
`chadwick_player_lu()` lookup used for the FanGraphs ID mapping).

## Pipeline

```r
raw_all <- pull_statcast_range("2021-04-01", "2025-09-28")  # or load cached
bsr <- run_baserunning_pipeline(raw_all, fg_sbcs_years = 2021:2025)
write.csv(bsr, "analysis/bsr_results.csv", row.names = FALSE)
```

See `R/baserunning_model.R` for all pipeline functions.

## Modeling notes

- **XGBoost (`multi:softprob`)**
- Both classifiers use a 15% holdout for early stopping, then fit **per-class
  Platt scaling** on that same holdout to correct softmax overconfidence
  before computing expected run values.
- Park effects included as a fixed 30-team one-hot block (`home_team`)

## Known limitations

- **Flat run values ignore base-out state.** A single that scores a runner
  from second is worth more than a single with the bases empty; this model
  uses the same `0.46` either way. This is the most likely driver of the
  issues below, and the natural next step is swapping in a full 24-state run
  expectancy matrix.
- **Extreme value magnitudes.** Total_BsR ranges roughly ±25-29 per
  player-season, wider than published metrics like FanGraphs' BsR (typically
  ±10-12). Per-class Platt calibration was applied and did **not** meaningfully
  shrink this range — full-sample player-seasons (400+ BIP) still hit the
  extremes, ruling out small-sample noise or raw softmax overconfidence as
  the primary cause. Points back to the flat run-value assumption above.
- **Weak agreement with FanGraphs' own BsR.** R² ≈ 0.17, r ≈ 0.41 across 7,384
  matched player-seasons (see `analysis/compare_fg_bsr.R`). Partly expected —
  FanGraphs' BsR includes UBR components this model doesn't capture (tag-ups,
  wild pitch/passed ball advances, double-play avoidance) — but the flat
  run-value issue likely also contributes.
- **Low year-over-year correlation** (r ≈ 0.25-0.33 depending on BIP
  threshold; see `analysis/yoy_stability.R`). Treated as expected for a
  descriptive (not predictive) stat, but flagged here since it's a natural
  question for anyone using this to project future value.
- **Half-inning boundary fallback is imprecise.** When a play ends the half
  inning, the runner-fate resolver defaults ambiguous cases to `OUT` rather
  than distinguishing "stranded" from "put out," since post-play state can't
  be confirmed without pulling `outs_when_up`/`des`.

## Repo structure

```
R/
  baserunning_model.R    # core pipeline: data pull, both XGBoost models, SBCS,
                          # name join, and final (batter, season) aggregation
analysis/
  compare_fg_bsr.R       # R² / correlation vs FanGraphs' own BsR
  yoy_stability.R        # year-over-year correlation by BIP threshold
  bsr_results.csv        # latest pipeline output (2021-2025, named)
```

## Requirements

R packages: `baseballr`, `dplyr`, `purrr`, `xgboost`, `tidyr`, `readr`.

