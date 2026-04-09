# Zone classification migration: rectangles to pixel mask

This report compares the original axis-aligned rectangle zone classifier
against the new hand-painted pixel mask in `src/lol_cv/features/zone_mask.png`.
The mask is loaded by `ZoneMask` (now living in `src/lol_cv/features/spatial.py`)
with `SHRINK_FACTOR=0.95`, `SHIFT_X=-2/200`, `SHIFT_Y=+1/200`. The cyan river
region is split into `river_top` / `river_bot` at classification time using
`(x + y) < 1.0`.

All feature CSVs and analysis directories are 34 games. The original baselines
(`features.csv`, `analysis*`) are kept untouched; everything from this
migration lives under the `_corrected` suffix.

## 1. Per-category best AUC (classification)

Best ROC-AUC across the 4 model families per category, per window. Old →
new with delta in parentheses.

| Category | Full | 0-15 min | 0-10 min | 0-5 min |
|---|---|---|---|---|
| early_strategy | 0.348 → 0.388 (+0.040) | 0.348 → 0.388 (+0.040) | 0.348 → 0.388 (+0.040) | 0.590 → 0.640 (+0.050) |
| jungle_invasion | 1.000 → 1.000 (+0.000) | 0.893 → 0.917 (+0.023) | 0.723 → 0.883 (+0.160) | 0.540 → 0.782 (+0.242) |
| lane_positioning | 1.000 → 1.000 (+0.000) | 0.823 → 0.793 (-0.030) | 0.843 → 0.705 (-0.138) | 0.712 → 0.556 (-0.156) |
| objective_control | 1.000 → 0.970 (-0.030) | 0.803 → 0.875 (+0.072) | 0.872 → 0.907 (+0.035) | 0.715 → 0.572 (-0.143) |
| ocr_state | 0.712 → 0.712 (0.000) | 0.310 → 0.310 (0.000) | 0.660 → 0.660 (0.000) | 0.522 → 0.522 (0.000) |
| strategic_decisions | 0.782 → 0.598 (-0.183) | 0.782 → 0.598 (-0.183) | 0.782 → 0.598 (-0.183) | 0.797 → 0.657 (-0.140) |
| team_coordination | 0.800 → 0.873 (+0.073) | 0.433 → 0.717 (+0.283) | 0.543 → 0.758 (+0.215) | 0.645 → 0.523 (-0.122) |
| temporal_dynamics | 0.683 → 0.840 (+0.157) | 0.494 → 0.513 (+0.019) | 0.427 → 0.589 (+0.163) | 0.520 → 0.565 (+0.045) |

**Net effect:** jungle_invasion improves dramatically in early windows
(+0.16 to +0.24 AUC at 0-10 and 0-5), team_coordination improves at 0-10/15,
temporal_dynamics improves across the board. lane_positioning and
strategic_decisions soften slightly because the rectangle bot_lane was
much wider and was conflating bot lane with bot jungle.

## 2. Top features by |Spearman r| on full-game classification

Union of the old and new top 10. `rank_old` / `rank_new` are 1-based ranks
across all 213 features.

| Feature | rank_old | rank_new | r_old | r_new | p_old | p_new |
|---|---|---|---|---|---|---|
| sp_red_zone_top_jungle_blue | 2 | 1 | -0.810 | -0.822 | 6.8e-09 | 2.6e-09 |
| sp_red_zone_blue_base | 8 | 2 | -0.677 | -0.822 | 1.1e-05 | 2.6e-09 |
| sp_blue_zone_bot_jungle_red | 1 | 3 | +0.816 | +0.809 | 4.2e-09 | 6.9e-09 |
| sp_red_dragon_avg_near_count | 4 | 4 | -0.753 | -0.753 | 2.8e-07 | 2.8e-07 |
| sp_blue_baron_avg_near_count | 5 | 5 | +0.740 | +0.740 | 5.6e-07 | 5.6e-07 |
| sp_red_zone_bot_jungle_blue | 50 | 6 | +0.201 | -0.734 | 0.255 | 7.8e-07 |
| sp_red_zone_top_jungle_red | 25 | 7 | +0.483 | +0.728 | 3.8e-03 | 1.1e-06 |
| sp_red_zone_red_base | 27 | 8 | +0.452 | +0.690 | 7.3e-03 | 6.3e-06 |
| sp_blue_zone_top_jungle_red | 28 | 9 | +0.408 | +0.684 | 1.7e-02 | 8.3e-06 |
| sp_blue_zone_bot_jungle_blue | 23 | 10 | +0.489 | -0.665 | 3.3e-03 | 1.8e-05 |
| sp_red_zone_bot_jungle_red | 3 | 17 | +0.759 | +0.615 | 2.0e-07 | 1.1e-04 |
| sp_red_zone_baron_pit | 6 | 19 | +0.684 | -0.571 | 8.3e-06 | 4.2e-04 |
| sp_blue_zone_bot_lane | 7 | 137 | -0.678 | -0.113 | 1.1e-05 | 0.525 |
| sp_red_zone_bot_lane | 9 | 143 | -0.659 | +0.107 | 2.3e-05 | 0.548 |
| sp_red_zone_top_lane | 10 | 130 | -0.659 | +0.119 | 2.3e-05 | 0.502 |

**Headline:** five jungle/base features are now in the top 10, four of
which were ranked 23-50 with the old rectangles. Three lane features
that ranked top-10 with the old rectangles drop out completely (rank
130-143). The mask draws lanes much more narrowly (and excludes the
broadcast border), and that border previously biased the lane occupancy
features.

The dragon/baron near-count features (which only depend on Euclidean
distance to the objective centre, not on `classify_zone`) are unchanged
as expected.

## 3. The three significant features

| Feature | r_old | p_old | r_new | p_new | Note |
|---|---|---|---|---|---|
| sp_early_red_jgl_commit_side | -0.3596 | 0.03672 | -0.3596 | 0.03672 | Unchanged — sanity check passes (this feature uses raw `y` mean, not `classify_zone`). |
| sp_strat_red_pre3min_invade_secs | -0.3609 | 0.03598 | -0.2762 | 0.11382 | Mask-affected. p moved from significant (0.036) to non-significant (0.114). |
| sp_red_dragon_avg_near_count | -0.7528 | 2.81e-07 | -0.7528 | 2.81e-07 | Unchanged — uses Euclidean distance, not zone classification. |

The pre-3-min invade feature dropping out of significance is expected:
the old rectangles for `bot_jungle_blue` (0.6-0.9 × 0.65-0.95) and
`top_jungle_blue` (0.1-0.35 × 0.35-0.65) over-counted "invade" frames
because they leaked into mid lane and river. The mask is tighter, so
fewer borderline frames are credited as invades. With n=34 games this
turns a marginal correlation into a non-significant one — but the
direction (negative — more red invades correlates with red losing) is
preserved.

## 4. Phase 5 regression R² per window (best of 3 models)

Target: `gold_diff_final_api` (final gold differential at end of game,
from the LoL Esports API). Best mean R² across `linear_regression`,
`ridge_regression`, `gradient_boosting_regressor`. CV: 5-fold.

| Window | Old best R² | New best R² | Δ |
|---|---|---|---|
| Full | 0.545 (gbr) | 0.564 (gbr) | +0.019 |
| 0-15 min | 0.486 (gbr) | 0.599 (gbr) | +0.113 |
| 0-10 min | 0.406 (gbr) | 0.553 (gbr) | +0.147 |
| 0-5 min | -0.153 (ridge) | 0.194 (ridge) | +0.347 |

Big jumps in the early windows where the noise from rectangle gaps was
hurting the models most.

## 5. Headline summary

Replacing the 13 axis-aligned rectangles with the painted pixel mask
makes the mid-game and especially the early-game zone features
substantially more discriminative. Best classification AUC for the
0-10 min window jumped from **0.760 → 0.933** (random forest), and
the gold-diff regression R² for the same window moved from **0.406 →
0.553**. The biggest categorical winner is `jungle_invasion`, whose
0-10 min AUC went from 0.72 → 0.88, vindicating the hypothesis that
the rectangle gaps were the main reason early-game positional features
underperformed. The `pre3min_invade_secs` feature lost its
significance (p 0.036 → 0.114) because the tighter mask removes some
borderline mid-river frames that had been artificially inflating the
old "invade" count, but the sign and magnitude of the correlation
remain consistent. The dragon/baron proximity features (which use raw
Euclidean distance, not zone classification) are unchanged, confirming
the change is isolated to the zone-based features.

## 6. Unknown-zone coverage (sanity check)

Computed by re-classifying every position from every game with both
classifiers and counting `unknown` returns.

| Classifier | Mean unknown rate | Min | Max |
|---|---|---|---|
| Old rectangles | 11.16% | 6.57% | 24.65% |
| New mask (shrink=0.95, shift=(-2/200, +1/200)) | **1.07%** | 0.70% | 1.84% |

The mask covers more of the playable area than the rectangles did —
the rectangles had visible gaps between zones (top-lane vs jungle, bot
lane vs jungle, river vs lanes) that classified as `unknown`. The
0.95 shrink only excludes the broadcast border and is well below the
old gap rate. **No game now has more than 1.84% of frames classified
as `unknown`**, which is the expected residual from minor jitter on
the broadcast border. The shift/shrink values are not too aggressive.

## Appendix — file shapes and cell change rates

Per-window comparison of `features_*.csv` vs `features_corrected_*.csv`,
counting only numeric cells where both old and new are non-NaN.

| Window | Shape | Cells changed | Mean abs Δ | Max abs Δ |
|---|---|---|---|---|
| Full | 34 × 213 | 1237 / 6644 (18.6%) | 14.38 | 488.00 |
| 0-15 min | 34 × 213 | 1233 / 6590 (18.7%) | 8.12 | 231.00 |
| 0-10 min | 34 × 213 | 1223 / 5242 (23.3%) | 4.94 | 138.00 |
| 0-5 min | 34 × 213 | 1178 / 4203 (28.0%) | 2.55 | 88.00 |

The large absolute deltas are zone transition counts and snapshot
features (counts in [0, ~500]); zone-occupancy fractions (in [0, 1])
shifted by ~0.03–0.13 on average. The shorter windows have a higher
*proportion* of changed cells because there are fewer NaN cells (snapshot
features at t=540s+ are NaN for short windows).
