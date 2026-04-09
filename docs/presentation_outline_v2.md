# Presentation Outline v2 — CV Analysis of First Stand 2026

**Project:** Computer Vision Analysis of League of Legends Replays for Strategic Insight
**Authors:** Wong Kang & Zachary Muk Chen Eu — University of Twente Q3 Data Science
**Target time:** ~20 minutes + Q&A
**Audience:** Data Science module instructor + peers (full DS literacy, no LoL domain knowledge required)

---

## Framing

This deck is a **showcase** of how a CV-based analysis pipeline can give a team statistical guidance from nothing but broadcast footage. The narrative is "we identified the LoL strategic concepts that matter, then built spatial features to measure each one directly from the minimap pixels, and here is what they predict and when". It is not a story of iterative failure and recovery.

**Honesty guardrail:** Do not claim CV beats the *private* Riot Esports API (the per-frame feed the four top leagues receive). That feed has perfect entity state and CV is, by definition, a lossy reconstruction. The comparison CV wins is against the *public* Match-V5 API, which only logs positions once every 60 seconds and is disabled for private scrims — that is the data landscape amateur and semi-pro teams actually live in, and it is where CV is strictly better. Keep this framing consistent whenever the API is mentioned.

---

## Slide-by-slide content

### Slide 1 — Title + problem statement

**Title:** *Strategic guidance from the minimap alone*
**Subtitle:** *Computer vision on 34 First Stand 2026 tournament VODs — no API, no replay files, no scoreboard exports*

**Talking points:**
- Teams in the four top leagues (LCK, LEC, LPL, LCS) can pull rich data straight from the Riot Esports API.
- Everyone else — academy leagues, regional tournaments, emerging regions, amateur and university esports — has only the broadcast VOD.
- Core question: can a CV-only pipeline give those teams statistically grounded strategic guidance?
- The short answer, which this deck defends: yes, if you design the features around the strategic concepts coaches already care about.

**Visual:** `charts/slide1_minimap_hero.png` (a clean mid-game minimap frame, full-bleed).

---

### Slide 2 — Why this matters

**Title:** *Who actually benefits from CV-only analytics*

**Talking points:**
- Academic and university esports programs: can run scrim review without commercial tools.
- Amateur and semi-pro teams: get statistical evidence to back strategic decisions they currently argue about by feel.
- Analysts covering regions outside the top four leagues: can finally quantify tendencies from footage that is already public.
- Commentators and content producers: can back storytelling with real measurements instead of vibes.
- **"Why not just use the public Match-V5 API?"** — because Match-V5 only logs player positions **once every 60 seconds**. In LoL, 60 seconds is enough time to recall to base, buy items, and walk across the map into a dragon setup. The public API misses the entire rotation. CV samples at 1 frame per second → **60× the temporal resolution**.
- **Scrim reality:** teams spend most of their practice time in private scrims where Match-V5 is often disabled or hidden to prevent strategy leaks. CV is server-agnostic and runs on any MP4 screen recording, so it is the only option for quantified scrim review.
- The tooling gap is a fairness gap — CV closes it using data that is already on YouTube and Twitch.

**Visual:** three-column text comparison — "Top leagues have (private Esports API, replay files, per-frame entity state)" vs "Public Match-V5 gives (1 snapshot / 60 s, only on official servers, disabled in scrims)" vs "What CV adds (1 frame / second, works on any MP4, no server dependency)". No image needed.

---

### Slide 3 — Dataset

**Title:** *34 games, one tournament, pure pixels*

**Talking points:**
- First Stand 2026 — 34 of 45 games (76% coverage), 8 teams (G2, BLG, BFX, TSW, GEN, JDG, LYON, LOUD).
- Roughly 40 minutes per game, sampled at 1 frame per second → ~55,000 gameplay frames total.
- Class balance: 22 blue wins, 12 red wins → majority baseline 64.7%.
- Inputs are literally just the minimap crops from the broadcast; no API calls for features, only for the winner label and validation gold totals.

**Visual:** small metadata table (games, frames, teams, balance, patch) — text only.

---

### Slide 4 — Pipeline overview

**Title:** *From broadcast VOD to 213 strategic features*

**Talking points:**
- Five stages: VOD frame extraction → minimap crop → YOLOv11m champion detection → tracking + zone classification → feature extraction.
- YOLOv11m chosen by benchmarking four YOLO variants on 200 held-out minimap frames (95% F1 at 14.5 FPS on CPU).
- Output is 213 engineered features per game covering eight LoL strategic concepts (next slide).
- Entire pipeline runs offline on a laptop — no cloud, no live data feed.

**Visual:** pipeline diagram (horizontal, five stages) with `charts/yolo_benchmark.png` as a small inset. TODO: pipeline diagram itself is not yet a chart file.

---

### Slide 5 — Zone classification methodology

**Title:** *A hand-painted pixel mask for strategic zones*

**Talking points:**
- Every champion (x, y) position has to be labelled with a strategic zone: top lane, mid lane, bot lane, each jungle quadrant, river top, river bot, dragon pit, baron pit, each base.
- We classify zones with a hand-painted pixel mask (`src/lol_cv/features/zone_mask.png`), loaded once and indexed in O(1) per lookup.
- Axis-aligned rectangles were an early methodology choice we replaced because the minimap lanes curve, overlap with river, and butt up against the broadcast border — rectangles left ~11% of frames classified as "unknown".
- Hand-painted mask brings unknown coverage down to ~1%, with no game above 1.84% unknown.
- Downstream every zone-based feature becomes cleaner: early-window AUC for `jungle_pathing_invasion` jumps from 0.72 to 0.88 in the 0-10 min window.

**Visual:** `charts/zone_mask_overlay.png` (mask painted on top of a real minimap frame).

---

### Slide 6 — Feature taxonomy

**Title:** *Eight LoL strategic categories, 213 features*

**Talking points:**
- Features are organised around coachable concepts, not around how they were implemented.
- Each category corresponds to a VOD-review topic a head coach could assign.
- Built directly from the minimap pixel stream — no HUD dependence for the main categories.
- **Categories 4 (`map_control_territory`) and 5 (`team_coordination_grouping`) contain dynamic tempo features** — convergence speed, synchronised recalls, zoning depth, rotation paths. These measure what happens *between* the 60-second Match-V5 snapshots. A data miner using only the public API literally cannot engineer these features, because the movement they measure occurs inside the API's temporal blind spots. 1 fps CV is the cheapest way to access this signal.

**Table:**

| # | Category | Definition | n features |
|---|---|---|---|
| 1 | jungle_pathing_invasion | Early jungle commit direction, level-1 and pre-3:00 invades, scuttle arrival/contest, jungle quadrant occupancy, map asymmetry | 26 |
| 2 | objective_contestation | Dragon/baron/herald grouping, convergence speed, pit occupancy, pre-objective snapshots, first-objective timings | 56 |
| 3 | lane_priority_pressure | Top/mid/bot lane occupancy, mid roam timing/target, bot zoning depth, first tower/inhibitor timings | 21 |
| 4 | map_control_territory | Team centroid (x, y), enemy-half count, base occupancy, zone transitions, rotations, synced recalls | 56 |
| 5 | team_coordination_grouping | Full-game grouping %, pairwise distances, per-snapshot grouping distance and spread | 32 |
| 6 | tempo_early_mid_transition | First-kill timing, kill counts per phase, avg/max/variance of kill tempo | 7 |
| 7 | river_vision_presence | River-top and river-bot zone occupancy (scuttle-lane and vision control) | 4 |
| 8 | game_state_ocr | OCR-derived end-state (kills, gold diff, duration) plus gold-diff slope derivatives | 11 |

**Visual:** the table above.

---

### Slide 7 — Classification results headline

**Title:** *Best AUC per window, per strategic category*

**Talking points:**
- 5-fold stratified CV, best of Random Forest / Gradient Boosting / SVM-RBF per cell.
- Target is binary `blue_wins`; baseline (always predict blue) is 64.7%.
- `map_control_territory` is the single strongest category at every window, reaching AUC 0.813 in the first 5 minutes, 0.975 by 15 minutes, and 1.000 on the full game.
- `objective_contestation` becomes the dominant signal at 0-10 min (AUC 0.907) — the "first dragon dance" window.
- `jungle_pathing_invasion` is consistently top-three across every window (0.682 → 0.850 → 0.917 → 0.967).

**Table (best AUC per window):**

| Category | 0-5 min | 0-10 min | 0-15 min | Full |
|---|---:|---:|---:|---:|
| map_control_territory       | **0.813** | 0.892     | **0.975** | **1.000** |
| jungle_pathing_invasion     | 0.682     | 0.850     | 0.917     | 0.967 |
| objective_contestation      | 0.572     | **0.907** | 0.883     | 0.980 |
| river_vision_presence       | 0.548     | 0.807     | 0.825     | 0.915 |
| team_coordination_grouping  | 0.548     | 0.742     | 0.795     | 0.775 |
| lane_priority_pressure      | 0.613     | 0.715     | 0.683     | 0.742 |
| tempo_early_mid_transition  | 0.637     | 0.487     | 0.628     | 0.573 |
| game_state_ocr              | 0.363     | 0.592     | 0.425     | 0.645 |

**Visual:** the table above, with the top category per window highlighted.

---

### Slide 8 — What to prioritize, by game phase (the key slide)

**Title:** *What coaches should drill, window by window*

**Talking points:**
- Each window has a dominant strategic category; the coaching takeaway is different in each one.
- Early game is about **where the five champions stand**, not about trading kills.
- 0-10 min is the first objective-contest window; dragon setups become the decisive signal.
- 15 minutes is already enough time for map-control features to nearly saturate AUC — laning-phase positioning essentially determines macro shape.

**Table (the centerpiece):**

| Window | Top strategic category | AUC | Plain-English coaching takeaway |
|---|---|---:|---|
| 0-5 min  | map_control_territory      | 0.813 | Win the first-clear by controlling **where the team physically is** at t=180 and t=300 (centroid and enemy-half count). Early kills are a weak signal — positioning is not. |
| 0-10 min | objective_contestation     | 0.907 | Treat this as the "first dragon dance" window. Grouping density around dragon/baron quadrants, convergence speed, and first-dragon timing are the decisive measurements. |
| 0-15 min | map_control_territory      | 0.975 | Drill macro shape: is the team centroid in the enemy half at 12-15:00? By this point territory alone almost fully separates winners and losers. |
| Full     | map_control_territory      | 1.000 | Territory → objectives → jungle pathing is the dominant ordering. Lane priority and kill tempo are second-tier (0.74 and 0.57). |

**Visual:** TODO — a custom "strategic priorities per window" chart would make this slide land. Suggested file: `charts/strategic_priorities_per_window.png`.

---

### Slide 9 — Per-category win-prediction ranking

**Title:** *How each category's predictive power evolves*

**Talking points:**
- Same data as slide 7, but sorted by average AUC across windows so the full ranking is visible.
- `map_control_territory` averages 0.920 across all windows — the clear #1.
- `jungle_pathing_invasion` and `objective_contestation` both average above 0.83 — a clear top three.
- `game_state_ocr` is last (avg 0.506): OCR-derived end-state is noisy and adds almost nothing once spatial features are in the model.

**Table (sorted by average AUC):**

| Category | 0-5 | 0-10 | 0-15 | Full | Avg |
|---|---:|---:|---:|---:|---:|
| map_control_territory       | 0.813 | 0.892 | 0.975 | 1.000 | 0.920 |
| jungle_pathing_invasion     | 0.682 | 0.850 | 0.917 | 0.967 | 0.854 |
| objective_contestation      | 0.572 | 0.907 | 0.883 | 0.980 | 0.835 |
| river_vision_presence       | 0.548 | 0.807 | 0.825 | 0.915 | 0.774 |
| team_coordination_grouping  | 0.548 | 0.742 | 0.795 | 0.775 | 0.715 |
| lane_priority_pressure      | 0.613 | 0.715 | 0.683 | 0.742 | 0.688 |
| tempo_early_mid_transition  | 0.637 | 0.487 | 0.628 | 0.573 | 0.581 |
| game_state_ocr              | 0.363 | 0.592 | 0.425 | 0.645 | 0.506 |

**Visual:** TODO — a per-category × window heatmap built from `data/processed/analysis_corrected_v2/best_auc_wide.csv`. Suggested file: `charts/category_heatmap_v2.png`. (The old `charts/category_heatmap.png` uses the v1 taxonomy and should not be reused.)

---

### Slide 10 — Phase 5 regression: spatial features predict the final gold lead

**Title:** *Early positioning explains up to 55% of the final gold differential*

**Talking points:**
- Instead of binary winner, regress final gold differential (pulled from the lolesports API as ground truth, so no measurement noise on the target).
- Best-of-three models (Linear / Ridge / Gradient Boosting Regressor) per window, 5-fold CV, top-15 feature filter.
- Economic outcome of a pro game is substantially baked in by minute 15 — the additional 15-30 minutes add only ~0.02 R².
- Validates that the spatial features are doing real work, not just correlating with the winner label.

**Table:**

| Window | Best R² | Best model |
|---|---:|---|
| 0-5 min  | 0.194 | Ridge |
| 0-10 min | 0.553 | Gradient Boosting |
| 0-15 min | 0.599 | Gradient Boosting |
| Full     | 0.564 | Gradient Boosting |

**Visual:** `charts/r2_emergence.png` (line chart of R² across windows). Note: the chart may currently reflect the pre-mask numbers from `results.md` (0.41 / 0.49 / 0.55). TODO: regenerate from the post-mask numbers above to match this slide.

---

### Slide 11 — One concrete example finding

**Title:** *Where the red team stands at minute 7 predicts the gold race*

**Talking points:**
- The single strongest post-mask feature for predicting final gold differential is `sp_snap_t420_red_centroid_x` (Pearson r = +0.514, Spearman r = +0.473, p = 0.0019, n=34).
- Plain English: at minute 7, the *horizontal* position of red team's centroid. The further right (deeper into red's own side of the map) the red team centroid sits, the worse their final gold lead.
- When red's centroid at 7:00 is pushed back toward their own base, they have lost early map control — they are reacting to blue's pressure instead of setting it.
- Companion signal: `sp_snap_t180_red_dragon_quadrant_count` (red bodies in the dragon quadrant at 3:00, r = -0.485) — red being forced to contest dragon early is a *loss* signal, not a win signal. Teams that choose to fight at dragon do it from strength; teams that are forced to fight at dragon are already behind.
- Coaching takeaway: a team that finds itself with centroid behind their own mid-tower at 7:00 is not "playing safe", they are losing the macro game and can see it on the minimap.

**Visual:** scatterplot of `sp_snap_t420_red_centroid_x` vs `gold_diff_final_api`, coloured by blue_wins. TODO — not yet built. Suggested file: `charts/example_finding_centroid_scatter.png`.

---

### Slide 12 — Methodology reflection

**Title:** *Validating geometric assumptions before trusting downstream features*

**Talking points:**
- The first version of the zone classifier used axis-aligned rectangles. They were fast to write but left ~11% of frames classified as "unknown" because lanes curve and overlap with the river.
- Replacing them with a hand-painted pixel mask dropped unknown coverage to ~1% and improved 0-10 min classification AUC by up to 0.24 for jungle-pathing features.
- General lesson: when many downstream features depend on a shared geometric primitive, **validate that primitive first**. The cost of the wrong assumption compounds across every feature that reads from it.
- This is a one-slide reflection on methodology, not a narrative arc. The rest of the project does not hinge on the rectangle story.

**Visual:** `charts/zone_mask_comparison.png` (side-by-side rectangle vs pixel mask).

---

### Slide 13 — Next steps: team-specific analysis

**Title:** *Per-team strengths and weaknesses from their own ~5 VODs*

**Talking points:**
- Pipeline is already designed to run per-game; aggregating across a single team's 4-6 tournament VODs yields a per-team profile.
- Concrete examples of what the current feature set could surface:
  - "Team X over-commits to dragon contests in 0-5 min" → measured directly by `sp_snap_t180_*_dragon_quadrant_count` being an outlier in losses.
  - "Team Y has weak bot-side map control in 10-15 min" → `sp_snap_t540_*_centroid_x` pushed back compared to other teams.
  - "Team Z consistently synchronises recalls in the 3-8 min window" → `sp_strat_*_synced_recalls` is a strength signal.
- Gate: we need ~5 games per team before per-team numbers stabilise. First Stand 2026 gives that for the top 4 teams but not the others.
- Directly actionable for scrim review and pre-match prep, and it plugs into the workflow a coach already has.
- **Scrim advantage:** because the pipeline is independent of Riot's servers, it is a natural scrim-review tool. A coach can record a private practice match against another team as an MP4 and run the pipeline locally. No Match ID, no public data leak, no opponent learning your strategies via a Match-V5 query. This is the setting where a CV-only approach stops being a workaround and starts being the *only* option.

**Visual:** mockup of a per-team profile card (category axis, team value vs tournament distribution). TODO — not yet built. Suggested file: `charts/team_profile_mockup.png`.

---

### Slide 14 — Next steps: larger dataset for generalizability

**Title:** *n=34 is a limitation. The next test is a larger corpus*

**Talking points:**
- Honest: 34 games on one patch, one tournament, 8 teams. Effect sizes might not hold outside First Stand 2026.
- Immediate candidate corpora for replication: LCK or LPL regular split (~100-150 games per split), Worlds 2025 (~100 games, same broadcast HUD), or a multi-tournament pool.
- A cross-tournament replication pins down which findings are structural (map control dominates in every patch) vs which are meta-specific (dragon priority at 0-10 may shift with dragon-soul patches).
- Most of the engineering cost is already paid: the pipeline is patch-agnostic and broadcast-agnostic as long as the minimap crop and HUD geometry are stable.
- Larger n also unlocks formal mediation tests (Baron-Kenny, VanderWeele decomposition) that require more statistical power than 34 games provide.

**Visual:** simple bar showing n=34 First Stand vs projected n=100-150 for a regular split vs n=~400 for a multi-tournament pool. TODO — not yet built. Suggested file: `charts/corpus_comparison.png`.

---

### Slide 15 — Summary / takeaway

**Title:** *CV pipeline + strategic taxonomy = actionable guidance without API access*

**Talking points:**
- Identified the LoL strategic concepts that matter; designed 213 spatial features to measure each directly from minimap pixels.
- Best classification AUC is 1.000 on the full game and 0.907 by minute 10 — using only information that is visible on any 1080p broadcast VOD.
- Early-game spatial features explain up to 55% of the variance in final gold differential; economic outcome is largely decided by minute 15.
- The dominant category is **map control and territory** — where the team physically lives on the minimap. It is the single best predictor at every window.
- Net effect: a team without Esports API access can get statistically grounded strategic guidance from footage alone. That is the contribution.

**Visual:** single sentence in the middle of the slide with authors/date underneath. No chart.

---

### Slide 16 (Appendix / Q&A pocket) — CV vs the public Match-V5 API

**Title:** *Why CV, not the public Match-V5 API?*

**When to use this slide:** keep it at the end of the deck and only flip to it if a peer or the instructor asks "did you really need computer vision for this?" or "why not use the Riot API?". Do not present it in the main flow.

**Talking points:**
- Match-V5 is the public Riot endpoint anyone can query for ranked and custom games. It logs player positions **once every 60 seconds**, plus discrete event pings (kills, tower falls, objectives). Everything else in between is invisible.
- 60 seconds in LoL is enough to push a wave, recall to base, buy items, and walk halfway across the map into an objective setup. The API sees "top lane" at 4:00 and "own jungle" at 5:00 and has no idea what happened in between.
- CV pipeline samples at 1 frame per second → **60× the temporal resolution** of Match-V5. That density is what unlocks convergence speed, synchronised recalls, zoning depth, and rotation paths. A data miner using Match-V5 literally cannot engineer those features — the movement happens in the blind spots.
- CV is also **server-agnostic**: it works on any MP4 screen recording including private scrims, off-server custom games, and alternate-account practice. Match-V5 needs an official-server game and a public Match ID; teams routinely turn it off to protect scrim strategies.
- **Clarification (what I am *not* claiming):** CV does not beat the *private* Riot Esports API that the four top leagues receive from Riot. That feed has per-frame perfect entity state, gold, cooldowns, and vision. CV is a lossy reconstruction of that. What CV beats is the data landscape that is actually accessible to everyone *outside* those four leagues — the public Match-V5 API and direct VOD recording.
- Net positioning: CV is not an alternative to the private feed, it is the only way to measure continuous spatial strategy from footage amateur and semi-pro teams already have.

**Pocket-answer script (for Q&A):** *"I looked into using the public V5 API instead of CV, but it has a 60-second temporal blind spot. Tempo features — synchronised resets, convergence speed, rotation paths — happen between the API snapshots, so the API literally can't see them. CV also runs on private scrim MP4s where Match-V5 is disabled. For continuous spatial strategy on public and private footage, CV isn't just an alternative to the public API, it's strictly better. It's only the private per-frame feed that the top four leagues get that beats CV, and that feed is not available to the teams this pipeline is designed for."*

**Visual:** TODO — simple timeline graphic. Draw a horizontal bar with two API frames (e.g. "Match-V5 t=240s" on the left and "Match-V5 t=300s" on the right), icons in the gap between them for "recall → buy → rotate to dragon", and 60 small tick marks underneath labelled "CV: 1 frame / second". Suggested file: `charts/cv_vs_api_temporal.png`. Not strictly required — this slide can also ship with just the text/table.

---

## Time budget

| Section | Slides | Minutes |
|---|---|---:|
| Setup (problem, audience, dataset) | 1-3 | 3 |
| Pipeline + methodology (zones, taxonomy) | 4-6 | 4 |
| Results (classification, priorities, regression) | 7-11 | 7 |
| Methodology reflection | 12 | 1 |
| Next steps | 13-14 | 3 |
| Summary | 15 | 1-2 |
| **Total** | **15** | **~19-20** |
| Appendix (Q&A pocket, not counted) | 16 | — |

---

## Charts needed

The following visuals are not yet in `charts/` and would need to be built:

- `charts/pipeline_diagram.png` — five-stage horizontal pipeline diagram for slide 4 (VOD → minimap crop → YOLO → tracking + zone classification → feature extraction).
- `charts/strategic_priorities_per_window.png` — coaching-priorities slide (slide 8): each window with its top category and a plain-English takeaway tile.
- `charts/category_heatmap_v2.png` — per-category × window AUC heatmap built from `data/processed/analysis_corrected_v2/best_auc_wide.csv` (slide 9). The existing `charts/category_heatmap.png` uses the v1 taxonomy and is stale.
- `charts/r2_emergence.png` — already exists but likely encodes pre-mask R² values (0.41/0.49/0.55); regenerate from the post-mask numbers in `zone_mask_migration_report.md` section 4 (0.194/0.553/0.599/0.564) for slide 10.
- `charts/example_finding_centroid_scatter.png` — scatter of `sp_snap_t420_red_centroid_x` vs `gold_diff_final_api` coloured by winner (slide 11).
- `charts/team_profile_mockup.png` — mockup per-team profile card showing per-category percentiles vs the tournament distribution (slide 13).
- `charts/corpus_comparison.png` — simple bar chart comparing n=34 vs projected regular-split and multi-tournament corpus sizes (slide 14).
- `charts/cv_vs_api_temporal.png` — timeline graphic showing two Match-V5 frames 60 s apart with recall/buy/rotate icons in the gap and 60 CV frame ticks underneath (appendix slide 16). Optional — the slide works with text alone.
