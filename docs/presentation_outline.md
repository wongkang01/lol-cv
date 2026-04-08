# Presentation Outline — CV Analysis of First Stand 2026

**Project:** Computer Vision Analysis of League of Legends Replays for Performance Insight
**Authors:** Wong Kang & Zachary Muk Chen Eu — University of Twente Q3 Data Science
**Target time:** ~14-16 minutes + 5 min Q&A
**Audience:** Data Science module instructor + peers (full DS literacy, no LoL domain knowledge required)
**Rubric:** `docs/references/Data driven Storytelling-Rubric-1.pdf` — 50 % storytelling, 50 % visualisation

---

## Design system (apply to every slide)

| Element | Value |
|---|---|
| Accent colour | `#1f77b4` (cool blue) |
| Accent dark (emphasis) | `#0d3b66` |
| Warning colour | `#d62728` (red — only for "below baseline" / "failed") |
| Neutral | `#cccccc` light grey, `#555555` dark grey |
| Font | DejaVu Sans (matches the matplotlib charts) |
| Slide aspect | 16:9 |
| Headline rule | One key message per slide, in the title — never a noun-only label |
| Annotation rule | Annotations live on the chart with arrows, never as side legends |
| White space rule | At least 30 % of every slide is empty |
| Jargon rule | No raw feature names (`sp_blue_zone_bot_jungle_red` → "Time blue spent inside red's bot jungle"). DS terminology (R², AUC, CV, regression) is fine — the audience knows it. LoL terminology is the actual jargon barrier. |

---

## Asset inventory

### Charts already built (in `charts/`)

| File | Slide | Purpose |
|---|---|---|
| `charts/yolo_benchmark.png` | 3 | YOLOv11 size variants speed/accuracy scatter |
| `charts/cv_results_table.png` | 5 | First-attempt 4-model CV results |
| `charts/top_features.png` | 6 | Top-5 Spearman correlations with plain English labels |
| `charts/window_collapse.png` | 7 | GBM accuracy collapse across early windows |
| `charts/category_heatmap.png` | 9 | Per-category × window AUC heatmap |
| `charts/r2_emergence.png` | 10 | Spatial → gold R² across windows |

### Images still needed (need extraction or creation)

| Slide | Image | Source |
|---|---|---|
| 1 | One clean minimap frame from a real First Stand game (no champion icons highlighted) | `data/raw/<match_id>/minimap/frame_*.png` — pick a mid-game frame around minute 10-15 with a balanced spatial spread |
| 8 | Pipeline diagram showing the 5-stage pipeline | Build in slide tool — see slide 4 layout sketch below |
| 8 | Feature taxonomy table | Build in slide tool — see slide 8 content below |
| 11 | Four minimap frame examples (one per downstream advantage) | `data/raw/<match_id>/minimap/frame_*.png` — see slide 11 for the four scenes to extract |
| 12 | Two-column comparison "what pros get vs what everyone else gets" | Build in slide tool — text only, no image needed |

---

## Slide-by-slide content

### Section 1 — Setup (slides 1-3)

---

### Slide 1 — Hook

**Title:** *Can the minimap alone tell us who will win?*

**Subtitle:** *What can a computer learn about pro esports from broadcast footage alone — no API, no scoreboard, no replay files?*

**Visual:** one clean minimap frame, full-bleed, no champion icons highlighted. Recommended source: `data/raw/fs_G2_vs_BLG_finals_2026-03-22_g2/minimap/frame_*.png` (picking a mid-game frame around minute 12-15 will show a balanced spatial spread without giving away the eventual winner).

**Layout:**
```
+----------------------------------------------+
| TITLE (top-left, plain sans-serif)           |
|                                              |
|  [          MINIMAP IMAGE 80% width        ] |
|  [                                          ] |
|  [                                          ] |
|                                              |
| Subtitle (top-left under title, smaller)     |
|                                              |
| Authors / date (bottom-right, small)         |
+----------------------------------------------+
```

**Speaker notes:**
> "Pro coaches and VOD analysts often have footage but no privileged data access. Riot doesn't expose its esports API to amateur leagues, regional broadcasts, or anyone outside the four major regions. We tested how much strategic insight can actually be recovered from the broadcast video alone — no API calls, no replay file, no scoreboard. Just the pixels."

**Rubric mapping:** Context (target audience hook), Choosing effective visuals (the actual data, not an abstract diagram).

---

### Slide 2 — Why this matters

**Title:** *Most analysts have footage. Few have data.*

**Visual:** two-column comparison, text only, no images.

**Layout sketch:**
```
+------------------------------+------------------------------+
|  WHAT PROS GET               |  WHAT EVERYONE ELSE GETS     |
|  ──────────────────          |  ──────────────────          |
|  • Riot esports API          |  • Broadcast VODs            |
|  • Replay (.rofl) files      |  • Screen recordings         |
|  • Scoreboard exports        |  • Twitch / YouTube uploads  |
|  • Per-frame ping tracking   |                              |
|  • Item / build timelines    |  And that's it.              |
|                              |                              |
|  (Available to ~4 leagues)   |  (Available everywhere)      |
+------------------------------+------------------------------+
```

**Speaker notes:**
> "There's a tooling asymmetry in pro esports analytics. The top four leagues — LCK, LEC, LPL, LCS — have access to the Riot esports API, replay files, and rich scoreboard data. Everyone else — academy leagues, regional tournaments, university esports, amateur scrims — has just the broadcast video. If we can do strategic analysis from public broadcasts alone, every coach in every region gets the same toolset."

**Rubric mapping:** Convincing story (motivation framed around a real audience problem), Removing clutter (text-only with strong contrast).

---

### Slide 3 — The pipeline

**Title:** *34 games. 55,000 frames. One pipeline.*

**Visual:**
- Top half: a horizontal pipeline diagram (build in slide tool) showing five stages.
- Bottom-right inset: `charts/yolo_benchmark.png` as a small embedded chart at maybe 35 % width.

**Layout sketch:**
```
+-----------------------------------------------------------+
| TITLE                                                     |
|                                                           |
| [VOD] -> [Minimap crop] -> [YOLO] -> [(x,y)] -> [Features]|
|  1080p     512x512        v11m       per sec    213 cols  |
|  1 fps                    95% F1                          |
|                                                           |
|                              +-----------------------+    |
|                              | yolo_benchmark.png    |    |
|                              | (inset, 35% width)    |    |
|                              +-----------------------+    |
+-----------------------------------------------------------+
```

**Speaker notes:**
> "Here's the pipeline. We sample tournament VODs at 1 frame per second, crop the minimap from the bottom-right corner, and run YOLOv11m to detect every champion icon on the minimap. That gives us a stream of (champion, x, y) coordinates per second per game. From there, 213 engineered features go into the cross-validated models. The chart in the corner is why we picked YOLOv11m specifically — we benchmarked four sizes; the smallest loses 14 % of detections, the largest is twice as slow with negligible accuracy gain. m-size is the sweet spot."

**Source data:** `data/processed/analysis/benchmark.csv`

**Rubric mapping:** Methodology rigour (the pipeline visible at a glance), Choosing effective visuals (scatter chart for two-axis tradeoff is the right viz for size vs speed).

---

### Section 2 — First analysis and the methodological turn (slides 4-7)

---

### Slide 4 — How we got the 88.6 %

**Title:** *First attempt: classify the winner from full-game spatial features*

**Visual:** 4-row methodology recipe table, built in slide tool.

**Layout sketch:**
```
+---------------------------------------------------------------+
| TITLE                                                         |
|                                                               |
| +-----------------+-----------------------------------------+ |
| | Setup item      | Value                                   | |
| +-----------------+-----------------------------------------+ |
| | Task            | Binary classification: predict the     | |
| |                 | winning side, one feature vector / game | |
| | Features        | 137 spatial features per game,         | |
| |                 | aggregated over the FULL game           | |
| | Models          | Random Forest, Gradient Boosting,      | |
| |                 | MLP, RBF SVM (4 families, no tuning)    | |
| | Validation      | 5-fold stratified CV, 22 blue / 12 red | |
| |                 | wins. Baseline = 64.7 % (always-blue)   | |
| +-----------------+-----------------------------------------+ |
+---------------------------------------------------------------+
```

**Speaker notes:**
> "Here's the recipe behind the headline number you'll see on the next slide. Four model families. Five-fold stratified cross-validation, with the minority class — red wins — at 12 games. Baseline of 64.7 % from always guessing blue side. The features at this point were aggregated over the entire game — average grouping distance, percentage of time in each zone, things like that. No early/late split, no time windowing, no bespoke features yet. This is the naive approach a first-time student would build."

**Rubric mapping:** Convincing story (the methodology is shown before the result, so the result is interpretable), Removing clutter (a 4-row table beats a paragraph of prose).

---

### Slide 5 — The first result

**Title:** *We hit 88.6 %. That's +24 points over baseline.*

**Visual:** `charts/cv_results_table.png` (full slide).

**Layout:** chart fills ~85 % of the slide width, centred. Title above, no subtitle (the chart has its own).

**Speaker notes:**
> "Four models, five-fold stratified cross-validation, full feature set. Random Forest hit 93.8 % but with 137 features and only 34 games it's almost certainly overfit, so we report Gradient Boosting at 88.6 % as the credible headline. The +24 percentage points over the 64.7 % always-blue baseline are real and reproducible. We thought we'd cracked it. But the next slide is the experiment we ran when we asked **what** the model was actually using."

**Source data:** `data/processed/analysis/cv_results.csv`

**Rubric mapping:** Convincing story (a result table beats a single big number for a DS audience — it shows the comparison), Choosing effective visuals (table is correct for cross-model comparison), Focus on attention (the chart has the GBM row pre-highlighted).

---

### Slide 6 — What was the model actually looking at?

**Title:** *Top 5 features by correlation with winning*

**Visual:** `charts/top_features.png` (full slide).

**Layout:** chart fills ~90 % of the slide. The on-chart annotation in the right margin already carries the punchline.

**Speaker notes:**
> "This is the slide that made us suspect the result. We pulled the top five features by Spearman correlation with the winner label. Every single one of them measures **enemy-territory occupation**. Time blue spent inside red's jungle. Time red spent inside blue's jungle. Red's defensive presence in its own jungle when it was being invaded. Red being forced near its dragon. Blue camping the baron pit. Of course winning teams are in the loser's territory — they've already won. We weren't predicting the future, we were describing the past. So we ran an experiment to confirm it."

**Source data:** `data/processed/analysis/feature_correlations.csv`

**Rubric mapping:** Convincing story (this is the analytical reasoning that motivates the next experiment — DS instructors reward this kind of "we noticed → we tested" sequence), Pre-attentive attention (mixed positive/negative bars in two colours read instantly).

---

### Slide 7 — The twist: that result is mostly hindsight

**Title:** *Restricting features to the early game collapses the model*

**Visual:** `charts/window_collapse.png` (full slide).

**Layout:** chart fills ~90 % of the slide. The brace annotation already carries the "−30 percentage points lost to hindsight" punchline.

**Speaker notes:**
> "Same gradient boosting model, same features, same five-fold cross-validation — but the features are now computed only over the first 5, 10, or 15 minutes of each game, before anyone has clearly won. Accuracy collapses from 88.6 % to roughly the always-blue baseline. The first 5 minutes are actually **below** baseline, meaning the model is worse than guessing. Our 'almost perfect' classifier needed almost the entire game to work. The 88.6 % was hindsight. The early-game features carried no signal."

**Source data:** `data/processed/analysis/window_comparison.csv`

**Rubric mapping:** Convincing story (the experimental confirmation of the diagnosis from slide 6), Reliability (we surfaced our own failure), Pre-attentive (red bars below the baseline line read as "failed" instantly).

---

### Section 3 — The methodological response (slides 8-10)

---

### Slide 8 — What we built to fix it

**Title:** *We added 125 new features specifically for the early game*

**Visual:** feature taxonomy table, built in slide tool.

**Layout sketch:**
```
+-----------------------------------------------------------------+
| TITLE                                                           |
|                                                                 |
| +----------+-------+----------------------+--------------------+ |
| | Phase    | Count | What they capture    | Example            | |
| +----------+-------+----------------------+--------------------+ |
| | Phase 3  |   17  | Rare-event early-    | Direction the      | |
| | sp_early |       | game decisions       | jungler clears     | |
| |          |       |                      | first              | |
| +----------+-------+----------------------+--------------------+ |
| | Phase 4  |   10  | Tempo and team       | Synchronised team  | |
| | sp_strat |       | coordination signals | recall events      | |
| +----------+-------+----------------------+--------------------+ |
| | Phase 5  |   98  | Pre-objective        | Team centroid 30 s | |
| | sp_snap  |       | positional snapshots | before first dragon| |
| +----------+-------+----------------------+--------------------+ |
|                                                                 |
| Total: 213 features × 34 games. Smart NaN handling for          |
| rare events (parallel _missing indicator, not silent zero-fill).|
+-----------------------------------------------------------------+
```

**Speaker notes:**
> "125 new features built specifically to capture strategic decisions in the first ten minutes. Phase 3 added rare-event signals — the jungler's first clear direction, mid laner's first roam, level-1 invade detection. Phase 4 added tempo and coordination — bot lane zoning depth, synchronised team recalls, map asymmetry. Phase 5 added the biggest set: 98 snapshot features that freeze the team formation at fixed times before each objective spawns. Each captures a positional **state**, not an aggregate. And we handled missing values carefully — for rare events, NaN means 'event didn't happen', not 'event happened at time zero', so we added parallel missingness indicators instead of silently filling zeros."

**Rubric mapping:** Methodology depth (this is the engineering work that distinguishes the project from a naive ML pipeline), Choosing effective visuals (a feature taxonomy table is the right shape for this content).

---

### Slide 9 — What survives the early-window test

**Title:** *Only one feature category stays predictive in the first 5 minutes*

**Visual:** `charts/category_heatmap.png` (full slide).

**Layout:** chart fills ~85 % of the slide. The accent border around the "Strategic decisions" row already carries the punchline; do not duplicate it in the slide title.

**Speaker notes:**
> "This is the chart that took us months. Eight feature categories, four time windows, best AUC across three model families per cell. Almost every category does great on the full game and falls apart when restricted to early windows — that's the hindsight collapse we just diagnosed. **One row stays roughly flat across every window.** That's the strategic-decision features we built in Phase 4 — bot zoning, synchronised recalls, map asymmetry, pre-3-minute counter-jungle. They sit at AUC 0.78 from the first 5 minutes through the full game. They're the only category that genuinely predicts rather than recaps. That validated the entire feature engineering effort."

**Source data:** `data/processed/analysis/category_comparison.csv`

**Rubric mapping:** Convincing story (strongest analytical artifact in the project — this is the slide that justifies the work), Choosing effective visuals (heatmap is the right shape for a category × window comparison), Removing clutter (no legend, single colormap, annotations on the cells).

---

### Slide 10 — From positioning to gold to winning

**Title:** *Early-game positioning shapes the gold lead — and the gold lead decides the winner*

**Visual:** `charts/r2_emergence.png` (top 75 % of slide).

**Below the chart (bottom 25 % of slide):** a small causal-chain diagram, text only:

```
[Early-game spatial decisions]  --R² = 0.41-->  [End-game gold lead]  --AUC = 1.00-->  [Winner]
   features from minutes 0-10                       from lolesports API
```

**Speaker notes:**
> "We changed the question. Instead of predicting **who** wins — which is binary and noisy at n=34 — we asked **by how much**. The end-of-game gold differential is a continuous variable with a lot more information than the binary winner label, and we get it directly from the lolesports API, so there's no measurement noise. From the same first-10-minutes spatial features, gradient-boosted regression explains 41 % of the variance in the eventual gold differential. By minute 15 it's 49 %. The remaining 15 minutes of game time only add 6 more points. **The economic outcome of a pro game is essentially baked in by minute 15.** This was validated against three independent end-state targets — raw gold lead, gold-per-minute (length-controlled), and final kill differential — all in the 0.4-0.5 range. The signal is real."

**Source data:** `data/processed/analysis_0_{300,600,900}/regression_gold_diff_final_api/cv_results.csv`, `data/processed/analysis/regression_gold_diff_final_api/cv_results.csv`

**Rubric mapping:** Convincing story (the Phase 5 punchline + the methodology shift from classification to regression), Choosing effective visuals (line chart for trend over windows is exactly right), Focus on attention (single line, two annotations, shaded lock-in band).

---

### Section 4 — The reframe (slide 11)

---

### Slide 11 — Which behaviours actually matter

**Title:** *Lane priority creates four measurable advantages — and we can see all four on the minimap*

**Visual:** four-quadrant grid. Each quadrant has a small minimap frame on the left and a feature label on the right. **All four minimap frames need to be extracted from `data/raw/<match_id>/minimap/`.**

**Layout sketch:**
```
+------------------------------------+------------------------------------+
| Q1: ENEMY JUNGLE PRESSURE          | Q2: OBJECTIVE PRIORITY             |
| [minimap frame: blue avatars       | [minimap frame: red team scrambled |
|  deep in red's bot jungle around   |  to defend dragon area at minute 3,|
|  minute 12-15]                     |  blue avatars converging]          |
| Caption: "Blue jungler farming     | Caption: "Red FORCED to defend     |
| red's camps unopposed"             | dragon — not choosing to"          |
| Feature: sp_blue_zone_bot_jungle_red| Feature: sp_snap_t180_red_dragon_*|
+------------------------------------+------------------------------------+
| Q3: COORDINATED TEMPO              | Q4: MAP ASYMMETRY                  |
| [minimap frame: 4+ blue avatars    | [side-by-side: blue spread across  |
|  at the blue base simultaneously,  |  all 4 quadrants vs red clustered  |
|  around minute 5-7]                |  in 1-2 quadrants]                 |
| Caption: "Multiple teammates       | Caption: "One team using more of   |
| recalling in the same window"      | the map than the other"            |
| Feature: sp_strat_synced_recalls   | Feature: sp_strat_map_asymmetry    |
+------------------------------------+------------------------------------+
```

**Frame extraction guide** (so the user can grab specific frames):

For each quadrant, pick a candidate game and approximate in-game time, then find the closest minimap PNG:

| Quadrant | Game suggestion | Approx in-game time | What to look for |
|---|---|---|---|
| Enemy jungle pressure | Any game where blue won decisively (e.g. `fs_G2_vs_BLG_finals_2026-03-22_g2`) | ~14:00 | 2-3 blue dots inside the bottom-right (red bot) jungle area |
| Objective priority | Any game with an early dragon contest (e.g. `fs_BFX_vs_BLG_groups_2026-03-16_g1`) | ~3:00 | Red team rotated to dragon area, blue team converging |
| Coordinated tempo | Any game with a clean recall window (e.g. `fs_LYON_vs_LOUD_groups_2026-03-17_g1`) | ~6:00 | 3+ blue dots clustered at blue base (bottom-left corner) |
| Map asymmetry | Any game where the spatial spread is visually asymmetric | ~8:00 | One side spread across all four quadrants, other side bunched |

To extract a specific frame, the absolute frame index in `data/raw/<game>/minimap/frame_XXXXXX.png` corresponds to the `timestamp` column in `data/processed/<game>/positions.csv` BEFORE per-game normalisation. Read the first few rows of that game's positions.csv to find the per-game start frame, then add `(target_seconds × 1)` to get the right frame number.

**Speaker notes:**
> "What our spatial features actually measure isn't lane play itself — it's what teams **do with** their lane priority. Lane priority gives you four things you can see on the minimap: your jungler can farm the enemy jungle, your support can roam to ward objectives, you can choose which neutral objective to contest first, and you can reset your gold and items in sync. Every one of our top features measures one of these four downstream effects. Coaches already know lane priority matters. What they don't have is a way to **quantify** whether their team converts lane priority into these four advantages. That's the gap we fill."

**Rubric mapping:** Convincing story (the reframe is the slide that distinguishes this project from generic ML — it translates analytics into something a coach can act on), Choosing effective visuals (real minimap frames > stylised diagrams), Removing clutter (4-quadrant grid is a clean information density).

---

### Section 5 — Honest evaluation (slide 12)

---

### Slide 12 — How reliable is this?

**Title:** *Five things to be careful about*

**Visual:** numbered list, plain text, no images.

**Layout sketch:**
```
+--------------------------------------------------------------------+
| TITLE                                                              |
|                                                                    |
| 1. n = 34 games, single tournament, single patch                   |
|    -> Per-category survival and R² consistency hold across windows |
|                                                                    |
| 2. Multiple comparisons (~210 features tested)                     |
|    -> Headline claims use CV scores, not per-feature p-values.     |
|       One feature does survive Bonferroni (p = 9 × 10⁻⁵).         |
|                                                                    |
| 3. OCR digit hallucination on broadcast HUD                        |
|    -> Phase 5 uses lolesports API gold directly, not OCR           |
|                                                                    |
| 4. Xin Zhao detection blind spot affects 9 games                   |
|    -> YOLO weight limitation; the other 25 games are unaffected    |
|                                                                    |
| 5. Correlational mediation, not formal causal test                 |
|    -> 50 % feature overlap is the SIGNATURE of mediation, not      |
|       a proof. Formal Baron-Kenny test needs n > 50.               |
+--------------------------------------------------------------------+
```

**Speaker notes:**
> "Five things to be transparent about. The sample is 34 games from one tournament on one patch — meaningful but not yet conclusive. We tested ~210 features so multiple comparisons matter, though only one feature survives Bonferroni. The HUD OCR was unreliable enough that Phase 5 entirely sidesteps it by pulling gold from the lolesports API instead. The YOLO detector has a blind spot for one specific champion, Xin Zhao, which affects 9 of our 34 games. And the mediation analysis on slide 10 is correlational, not a formal causal test — that needs more games. The main finding holds because the per-category survival and the temporal R² emergence both replicate the same conclusion through independent statistical lenses."

**Rubric mapping:** Reliability (explicit, numbered, with mitigations — the rubric explicitly grades this), Convincing story (acknowledging limitations strengthens credibility for a DS audience).

---

### Section 6 — Value and close (slides 13-14)

---

### Slide 13 — Value for coaches and analysts

**Title:** *Four things this enables that weren't possible before*

**Visual:** four-row layout, icon + text per row, no images required.

**Layout sketch:**
```
+--------------------------------------------------------------------+
| TITLE                                                              |
|                                                                    |
| 📊 QUANTIFY "TEMPO" OBJECTIVELY                                   |
|    Compute synchronised recalls and map asymmetry from any VOD.    |
|    Tempo stops being a slippery commentator word and becomes a     |
|    number coaches can compare across games.                        |
|                                                                    |
| 🎯 DETECT EARLY-GAME OVER-COMMITMENT                              |
|    Watch the minute-3 dragon-quadrant count for either team.       |
|    Catches a tempo trap BEFORE it shows up in the kill score —     |
|    actionable in scrim review.                                     |
|                                                                    |
| 🔍 COMPARE TWO TEAMS' EARLY MACRO SHAPE                           |
|    Run the spatial features across both teams' recent VODs.        |
|    Pre-match prep based on tendencies, not just pick/ban stats.    |
|                                                                    |
| 🌍 COVER REGIONS WITHOUT API ACCESS                               |
|    The whole pipeline runs on any 1080p VOD — amateur leagues,     |
|    regional broadcasts, university tournaments, scrims.            |
+--------------------------------------------------------------------+
```

**Speaker notes:**
> "Coaches already have most of the tools they need for the four major leagues. What they don't have is a way to quantify pre-objective positioning, tempo recalls, and map asymmetry from the only data source they ever actually have access to — the VOD. And they certainly don't have it for amateur tournaments or regions outside the top four leagues. That's the practical contribution: not 'we built another winner predictor', but 'we built a measurable spatial substrate that anyone with footage can use'."

**Rubric mapping:** **Actionable outcomes** (the rubric explicitly grades this), Context (concrete use cases for the named target audience).

---

### Slide 14 — One-line takeaway

**Title:** none — let the sentence carry the slide.

**Visual:** centred sentence in large type, plenty of white space.

**Layout sketch:**
```
+--------------------------------------------------------------------+
|                                                                    |
|                                                                    |
|                                                                    |
|     "Broadcast footage alone cannot predict the game at            |
|      5 minutes — but it can quantify how much enemy-jungle         |
|      pressure, objective priority, and tempo a team is             |
|      converting their early advantages into. And those             |
|      convert into the gold lead that decides the match."           |
|                                                                    |
|                                                                    |
|                                                                    |
|     Wong Kang & Zachary Muk Chen Eu                                |
|     Data Science Q3 — University of Twente — 2026                  |
|                                                                    |
+--------------------------------------------------------------------+
```

**Speaker notes:**
> "That's the project. Thank you. Happy to take questions."

**Rubric mapping:** Convincing story (closing summary), Removing clutter (single sentence, maximum white space).

---

## Time budget

| Section | Slides | Minutes |
|---|---|---|
| Setup | 1-3 | 2.5 |
| First analysis & methodological turn | 4-7 | 4 |
| Methodological response | 8-10 | 4 |
| Reframe (downstream advantages) | 11 | 1.5 |
| Honest evaluation | 12 | 1.5 |
| Value & close | 13-14 | 1.5 |
| **Total** | **14** | **~15** |

Add ~5 min Q&A buffer.

---

## How this maps to the rubric

| Rubric component | Where it lives in this deck |
|---|---|
| **Context (20 %)** | Slides 1-3 — research question, real-world analyst pain point, pipeline overview |
| **Convincing story (20 %)** | Slides 4-10 — methodology recipe → result → diagnostic → twist → engineering response → category survival → mediation evidence. Every beat is anchored in an analytical decision. |
| **Reliability (in convincing story)** | Slide 12 (numbered limitations with mitigations); slide 7 also implicitly ("we caught our own bias"). Two slides on reliability is rubric-favourable. |
| **Actionable outcomes (in convincing story)** | Slide 13 (four concrete use cases for coaches and analysts) |
| **Creative storytelling (10 %)** | Story arc with the "we caught hindsight ourselves" turn (slides 6→7); the running visual anchor (minimap frames recur on slides 1 and 11); the four-quadrant reframe on slide 11 reorganises a familiar idea ("laning matters") into new structure |
| **Choosing effective visuals (20 %)** | Each chart matches its data shape — heatmap for category × window, line chart for R² over time, bar chart for window collapse, signed bars for top features, scatter for size/speed tradeoff, table for cross-model comparison, four-quadrant grid for the four advantages |
| **Removing clutter (10 %)** | One message per slide, white space ≥30 %, no gridlines, no legends (annotations on charts instead), no logos, no extra colours beyond the accent + warning |
| **Focus on attention (20 %)** | Single accent colour (`#1f77b4`) throughout. Strategic_decisions row pre-highlighted on slide 9. GBM row pre-highlighted on slide 5. Failure bars in red on slide 7. Same accent applied across charts so the eye learns "this colour = the thing to look at". |

---

## What's still needed before the deck is complete

1. **Extract one minimap frame for slide 1.** Pick a neutral mid-game frame from `data/raw/fs_G2_vs_BLG_finals_2026-03-22_g2/minimap/` (or any visually clean game). Save as `charts/slide1_minimap_hero.png` for consistency.

2. **Extract four minimap frames for slide 11.** See the frame extraction guide in slide 11 above. Save as `charts/slide11_q1_jungle_pressure.png`, `charts/slide11_q2_objective_priority.png`, `charts/slide11_q3_synced_tempo.png`, `charts/slide11_q4_map_asymmetry.png`.

3. **Build the pipeline diagram in slide 3.** No image needed — use the slide tool's shape primitives. The chart inset (`charts/yolo_benchmark.png`) embeds in the bottom-right corner.

4. **Build the feature taxonomy table in slide 8.** No image needed — use a slide table primitive.

5. **Build the methodology recipe table in slide 4.** Same as slide 8 — slide table primitive.

---

## Reproducing the charts

Both chart-building scripts live alongside the analysis pipeline. To regenerate any chart:

```bash
PYTHONPATH=src uv run python scripts/build_critical_charts.py     # heatmap, R² emergence, window collapse
PYTHONPATH=src uv run python scripts/build_supporting_charts.py   # top features, YOLO benchmark, CV results
```

Both write to `charts/` and overwrite existing files. Style constants live at the top of each script — change `ACCENT`, `WARNING`, or font there to recolour the entire deck consistently.
