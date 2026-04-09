# Feature Categorisation v2 — Strategic Taxonomy Re-ranking

Corrected First Stand 2026 dataset, 34 games, 213 features.
Classification target: `blue_wins` (binary), 5-fold stratified CV,
RF / Gradient Boosting / SVM-RBF (same setup as
`scripts/run_analysis.py::category_ablation`).

## Why the old taxonomy was misleading

The v1 categorisation in `scripts/run_analysis.py` bucketed features purely
by **feature-name prefix**. That approach conflated three different things:
tactical concept (what a coach trains), rare-vs-aggregate event rarity, and
aggregation type (full-game vs time-window snapshot). Two concrete failures:

1. `sp_early_red_jgl_commit_side` (early jungle commit direction) and
   `sp_strat_red_pre3min_invade_secs` (time spent in enemy jungle pre-3:00)
   were placed in the catch-all buckets `early_strategy` and
   `strategic_decisions`, even though they describe the same coachable
   concept as `sp_blue_zone_top_jungle_red` (jungle quadrant occupancy) which
   was in the separate bucket `jungle_invasion`. The category-level AUC
   ranking therefore double-counted the jungle-pathing signal across three
   categories.
2. `sp_snap_t420_red_centroid_x` (team centroid at 7:00) was bucketed under
   `temporal_dynamics`, alongside totally unrelated tp\_ kill tempo features,
   even though conceptually it is a **map-control / territorial** snapshot.
   Snapshots and aggregates of the same concept landed in different
   categories solely because one had an `sp_snap_` prefix.

The consequence was that the strongest reported v1 category,
`jungle_invasion` (AUC 1.00, n=8), was only strong because it happened to
exclude the rare-event jungle features that weakened the signal when merged.
Meanwhile the weakest v1 category, `early_strategy` (AUC 0.39), was dragged
down by containing roam/scuttle/lvl1-invade features alongside mid-lane roam
targets — two totally different coaching concepts averaged together.

## The new taxonomy (v2)

Eight categories, designed so each one is something a head coach could
assign as a focused VOD-review topic. Feature counts are from
`data/processed/features_corrected.csv` (Full window); earlier windows have
slightly fewer features because snapshot times beyond the window are empty.

| # | Category | Definition | n features |
|---|---|---|---|
| 1 | **jungle_pathing_invasion** | Early jungle commit direction, level-1 and pre-3:00 enemy-jungle invade, scuttle arrival/contest, jungle-quadrant occupancy, and map-asymmetry (mirror-pathing) index | 26 |
| 2 | **objective_contestation** | Dragon/baron/herald: grouping-near, convergence speed, pit occupancy, dragon/baron quadrant snapshot counts, and `tp_first_<obj>_time` / `tp_*_<obj>_count` phase buckets | 56 |
| 3 | **lane_priority_pressure** | Top/mid/bot lane occupancy, mid-lane roam timing/target, bot-lane zoning depth, first-tower / first-inhibitor timings, tower and inhibitor counts | 21 |
| 4 | **map_control_territory** | Team centroid (x, y), enemy-half count, base occupancy, total/avg zone transitions, avg rotations, synced recalls — where the team physically lives on the map | 56 |
| 5 | **team_coordination_grouping** | Full-game `grouped_pct` / `avg_grouping_dist` plus per-snapshot `grouping_dist` and `spread` (pairwise distances, not tied to an objective) | 32 |
| 6 | **tempo_early_mid_transition** | Kill tempo: first-kill timing, kill counts per phase, avg / max / variance of tempo | 7 |
| 7 | **river_vision_presence** | River-top and river-bot zone occupancy — scuttle-lane and vision-denial presence | 4 |
| 8 | **game_state_ocr** | Outcome-adjacent OCR state (final kills / gold diff / max lead / mean gold diff / duration) plus `tp_gold_diff_slope_*` derivatives | 11 |

Total: 213 features.

Key moves from v1: `sp_early_*` / `sp_strat_*` jungle-pathing features merged
with full-game jungle-quadrant features (category 1). All snapshot
`centroid_x/y` and `enemy_half_count` features moved from `temporal_dynamics`
into category 4. Snapshot `dragon_quadrant_count` / `baron_quadrant_count`
moved into objective_contestation (2). `tp_first_dragon_time` /
`tp_first_baron_time` / dragon-herald-baron phase counts moved out of
`temporal_dynamics` into objective_contestation (2) — they are direct
objective control signals. Kill-tempo `tp_*_kill_count` and `tp_avg_tempo`
stayed grouped as a (thin) tempo category (6). OCR + gold-diff slopes were
kept together in (8) to make the outcome-leakage pathway visible.

## Per-window best ROC-AUC per category

Best-of-three-models AUC for each (category, window) pair, sorted by average
AUC across windows descending. **Bold** marks the strongest category per
window. Data source: `data/processed/analysis_corrected_v2/best_auc_wide.csv`.

| Category | 0–5 min | 0–10 min | 0–15 min | Full | Avg |
|---|---|---|---|---|---|
| map_control_territory       | **0.813** | 0.892     | **0.975** | **1.000** | 0.920 |
| jungle_pathing_invasion     | 0.682     | 0.850     | 0.917     | 0.967     | 0.854 |
| objective_contestation      | 0.572     | **0.907** | 0.883     | 0.980     | 0.835 |
| river_vision_presence       | 0.548     | 0.807     | 0.825     | 0.915     | 0.774 |
| team_coordination_grouping  | 0.548     | 0.742     | 0.795     | 0.775     | 0.715 |
| lane_priority_pressure      | 0.613     | 0.715     | 0.683     | 0.742     | 0.688 |
| tempo_early_mid_transition  | 0.637     | 0.487     | 0.628     | 0.573     | 0.581 |
| game_state_ocr              | 0.363     | 0.592     | 0.425     | 0.645     | 0.506 |

Best CV model is chosen independently per cell; see
`per_window_category_best.csv` for the winning model, accuracy, and f1
alongside each AUC.

## Top 3 categories per window (coach-facing narrative)

**0–5 min (first-clear window).** Top three: `map_control_territory` (0.81),
`jungle_pathing_invasion` (0.68), `tempo_early_mid_transition` (0.64). In
the first 5 minutes the only signal strong enough to separate winners from
losers is **where on the map the five champions physically are at t=180 and
t=300** — centroid, enemy-half count, early rotation count. A coach should
focus first-clear prep on early team positioning rather than kill trades:
the kill-tempo category is a weak third. Jungle pathing is recognisable but
the small feature count and small window hurt power.

**0–10 min (pre-mid-game transition).** Top three: `objective_contestation`
(0.91), `map_control_territory` (0.89), `jungle_pathing_invasion` (0.85). By
10 minutes the first dragon and first herald are usually decided, so the
objective-contestation bucket lights up for the first time — team-group
density around dragon/baron quadrants, convergence speed, and `tp_first_*`
objective timings all become predictive. Coaches should treat this window
as the "first dragon dance" prep window: setups around the dragon pit are
the most informative training target.

**0–15 min (late laning / mid-game transition).** Top three:
`map_control_territory` (0.975), `jungle_pathing_invasion` (0.92),
`objective_contestation` (0.88). Map control is now nearly saturating AUC:
15-minute centroid positions + enemy-half occupancy alone effectively
separate winners. Coaches should focus mid-game prep on macro shape
(whether the team centroid is in the enemy half at 12–15:00), which is
exactly the feature that the old taxonomy buried under `temporal_dynamics`.

**Full game.** Top three: `map_control_territory` (1.00),
`objective_contestation` (0.98), `jungle_pathing_invasion` (0.97). All three
macro categories fully saturate, and the ranking matches the pre-mid-game
story: territory → objectives → jungle paths. Lane priority and tempo come
out clearly second-tier (0.74 and 0.57 respectively), which is consistent
with First Stand's high-execution meta where losing lane but winning map is
a common winning pattern.

## What moved between v1 and v2 — representative migrations

Ten illustrative moves (see `data/processed/feature_categories_v2.csv` for
the full mapping):

| Feature | v1 category | v2 category |
|---|---|---|
| `sp_early_red_jgl_commit_side`         | early_strategy      | jungle_pathing_invasion |
| `sp_early_blue_lvl1_invade_frames`     | early_strategy      | jungle_pathing_invasion |
| `sp_strat_red_pre3min_invade_secs`     | strategic_decisions | jungle_pathing_invasion |
| `sp_strat_map_asymmetry_index_mean`    | strategic_decisions | jungle_pathing_invasion |
| `sp_snap_t420_red_centroid_x`          | temporal_dynamics   | map_control_territory |
| `sp_snap_t300_blue_enemy_half_count`   | temporal_dynamics   | map_control_territory |
| `sp_snap_t540_blue_dragon_quadrant_count` | temporal_dynamics | objective_contestation |
| `tp_first_dragon_time`                 | temporal_dynamics   | objective_contestation |
| `tp_early_herald_count`                | temporal_dynamics   | objective_contestation |
| `tp_first_tower_time`                  | temporal_dynamics   | lane_priority_pressure |

The net effect: the v2 ranking no longer double-counts jungle-pathing
signal, and the snapshot-vs-aggregate split across `temporal_dynamics` is
eliminated. `map_control_territory` replaces `jungle_invasion` as the top
category because it now correctly contains all centroid / enemy-half /
territory features regardless of whether they are aggregates or snapshots.

## Artifacts

- Mapping CSV: `data/processed/feature_categories_v2.csv`
- Long-form AUC table: `data/processed/analysis_corrected_v2/per_window_category_models.csv`
- Best-model-per-(window, category): `data/processed/analysis_corrected_v2/per_window_category_best.csv`
- Wide table used above: `data/processed/analysis_corrected_v2/best_auc_wide.csv`
- Builder script: `scripts/build_categories_v2.py`
- Ablation runner: `scripts/rerun_categories_v2.py`
