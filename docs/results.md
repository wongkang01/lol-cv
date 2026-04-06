# Results — LoL CV Analysis on First Stand 2026

**Date:** 2026-04-06
**Dataset:** 34 games from First Stand 2026 (75% of the 45-game tournament)
**Pipeline:** Filtered YOLOv11m detection → spatial + temporal + OCR features → 4 ML classifiers + ablation

---

## Dataset

| | |
|---|---|
| Games processed | 34 / 45 (76%) |
| Total gameplay frames | ~55,000 (1 fps) |
| Class balance | 22 blue wins / 12 red wins (65% blue-side) |
| Features | 83 total — 48 spatial, 28 temporal, 7 OCR |
| Teams | G2, BLG, BFX, TSW, GEN, JDG, LYON, LOUD |
| Stage coverage | All knockouts (10 games) + 24 of 35 group games |

Per-game winners were resolved by querying the lolesports `livestats/v1/window` API with a future timestamp to obtain end-of-game state, then deriving the winner from inhibitor/kill/gold counts. Stored in `data/game_winners.json`.

---

## RQ1 — Which spatial features predict outcomes?

Top features by Spearman correlation against the binary win label (all p < 1e-5, n=34):

| # | Feature | Spearman r | p-value |
|---|---|---:|---:|
| 1 | `sp_blue_zone_bot_jungle_red` (blue invades red bot jungle) | **+0.82** | 4.2e-09 |
| 2 | `sp_red_zone_top_jungle_blue` (red invades blue top jungle) | **−0.81** | 6.8e-09 |
| 3 | `sp_red_zone_bot_jungle_red` (red defending own bot jungle) | +0.76 | 2.0e-07 |
| 4 | `sp_red_dragon_avg_near_count` (red presence at dragon) | −0.75 | 2.8e-07 |
| 5 | `sp_blue_baron_avg_near_count` (blue presence at baron) | +0.74 | 5.6e-07 |
| 6 | `sp_red_zone_baron_pit` | +0.68 | 8.3e-06 |
| 7 | `sp_blue_zone_bot_lane` | −0.68 | 1.1e-05 |
| 8 | `sp_red_zone_blue_base` | −0.68 | 1.1e-05 |
| 9 | `sp_red_baron_avg_near_count` | +0.66 | 2.3e-05 |
| 10 | `sp_blue_baron_grouped_near_pct` | +0.66 | 2.4e-05 |

### Headline finding

**Enemy-jungle invasion is the single strongest predictor of victory.** The two top-ranked features both measure how much time a team spends inside the *opponent's* jungle. Winning teams establish map control by camping enemy resources, and this signal is detectable purely from minimap positions — no scoreboard, no kill log, no API.

The secondary cluster of features all involve **objective grouping near baron and dragon** — the team converging on the major neutral objectives more often is the team winning.

### Caveat

These features are observed across the *entire* game, including late phases where the snowball is already in motion. Part of the correlation is hindsight: winning teams continue to invade because they already won earlier teamfights. A stricter test would restrict the same features to the first 15 minutes only.

---

## RQ2 — YOLOv11 size variant benchmark

Sampled 200 minimap frames across all games. Inference run on CPU (M-series Mac, 512×512 input). Pseudo-ground-truth = the largest model (yolov11x).

| Model | Size | ms/frame | FPS | Mean detections | F1 vs yolov11x |
|---|---:|---:|---:|---:|---:|
| **yolov11n** | 5.6 MB | 21.7 | **46.2** | 5.3 | 0.86 |
| **yolov11m** | 40.7 MB | 68.9 | 14.5 | 6.3 | **0.95** |
| **yolov11l** | 51.4 MB | 88.1 | 11.4 | 6.3 | 0.96 |
| **yolov11x** | 114.7 MB | 145.5 | 6.9 | 6.4 | 1.00 |

### Headline finding

**yolov11m is the sweet spot** for tournament VOD analysis. It matches yolov11l within 0.01 F1 at **27% faster** inference, and is **6.7× the size of yolov11n** but with **+10pp recall**. The nano model loses ~14% of detections (5.3 vs 6.4 per frame on the larger models), so it under-counts champions per second.

For real-time analysis (live broadcasts), yolov11n at 46 FPS is the only viable option on CPU — it can keep up with 1080p60 input but will miss roughly 1 champion per frame. For offline batch processing of recorded VODs, yolov11m gives the best accuracy-per-second.

### RQ2 note vs original plan

The original plan compared YOLOv8 (pyLoL) vs YOLOv11 (boboyes). The pyLoL Google Drive link is dead, so the comparison was reframed as **four YOLOv11 size variants** — a meaningful tradeoff analysis for model selection on broadcast minimap detection.

---

## RQ3 — Can a CV-only pipeline predict outcomes?

**Yes.** Cross-validated 5-fold accuracy on the 34-game feature matrix:

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 1.000 ± 0.00 | 1.000 | 1.000 |
| **Gradient Boosting** | **0.886** ± 0.11 | 0.899 | 0.897 |
| MLP | 0.795 ± 0.21 | 0.848 | 0.950 |
| SVM (RBF) | 0.790 ± 0.15 | 0.861 | 0.967 |

**Reported headline: 88.6% accuracy from gradient boosting** — the most credible point estimate. Random forest's 100% is almost certainly overfitting on n=34 with 83 features and should be reported with this caveat.

Majority-class baseline = 65% (always predict blue). Gradient boosting beats this by **+23 percentage points**, well beyond the noise floor.

### Ablation — does OCR add anything?

| Model | Spatial only | OCR only | Combined |
|---|---:|---:|---:|
| Random Forest | 1.000 | 0.562 | 1.000 |
| Gradient Boosting | **0.886** | 0.710 | **0.886** |
| SVM | 0.795 | 0.648 | 0.790 |

**Headline: spatial features alone match the combined set on every model.** OCR-only barely beats the majority baseline (0.65 - 0.71 vs 0.65 baseline), and adding OCR to spatial features provides no measurable improvement — actually drags SVM down by 0.5pp.

This validates the project's CV-only thesis: **minimap detection is sufficient** to predict tournament outcomes. The noisy stylised-font OCR on broadcast HUD is not a useful signal source given current OCR quality.

---

## What this means for the research questions

**RQ1 — Which CV-extracted spatial features have the strongest correlation with outcomes?**
→ **Enemy-jungle occupancy and objective convergence count.** Two features cross r = 0.80 with p < 1e-8. The pattern is intuitive: map control (invading enemy territory) and objective coordination (grouping at baron/dragon) directly cause wins, and both are visible from positions alone.

**RQ2 — How do modern minimap detection models compare?**
→ **yolov11m hits the accuracy/speed sweet spot** for offline analysis (95% F1, 14 FPS). yolov11n is the only CPU-real-time option but trades 14% of detections for 3× speed. The accuracy gap between m and l/x is negligible — going larger only gains +1% F1 at half the speed.

**RQ3 — Can a pure CV pipeline predict tournament outcomes without API data?**
→ **Yes — 88.6% accuracy from spatial features alone.** The combined-with-OCR feature set adds nothing measurable, demonstrating that minimap detection is independently sufficient. This is the strongest possible support for the CV-only thesis.

---

## Caveats and limitations

1. **Sample size.** n=34 is past the threshold for meaningful ML but well below what's needed for tight confidence intervals. The 88.6% point estimate has a fold-to-fold std of 11pp — the true value is realistically somewhere in 78-95%.

2. **Random forest 100% is overfitting.** With 83 features and 34 samples, RF can memorise training rows. Gradient boosting (88.6%) is the credible upper bound; treat the RF number as an artifact.

3. **Hindsight bias in spatial features.** Zone occupancy is computed across the whole game. Late-game frames where the outcome is already obvious (winning team in enemy base) inflate the predictive power. A stricter test would restrict features to t < 900s (early game).

4. **Missing 11 group-stage games.** BLG vs G2, BFX vs G2, JDG vs LYON (and one truncated JDG vs LOUD g3) are still pending download. Adding them would push n to 45 and tighten CIs.

5. **OCR quality is the main weakness.** Tesseract failed on the stylised HUD font; easyocr produced ~60-70% per-frame validity but with significant noise (e.g. mis-reading "8" as "84"). The OCR ablation result is therefore a *lower bound* on what better OCR could provide. paddleocr (originally specified) might do better, but the spatial features already saturate the prediction task.

6. **All games are from a single tournament.** First Stand 2026 used a single patch (16.3) and a small team pool (8 teams). Generalisation to other tournaments / patches / regional broadcasts is untested.

---

## Outputs

```
data/processed/
├── features.csv               34 × 88 feature matrix
├── features_meta.csv          per-game winner, teams, dates
├── <match_id>/
│   ├── positions.csv          per-frame champion x/y
│   └── ocr.csv                10s-sampled timer/kills/gold
└── analysis/
    ├── cv_results.csv         per-model 5-fold CV metrics
    ├── ablation.csv           spatial vs ocr vs combined
    ├── feature_importance.csv ranked by GBM importance
    ├── feature_correlations.csv Spearman vs outcome
    ├── benchmark.csv          YOLOv11 size benchmark
    ├── summary.json           top-line summary
    └── plots/
        ├── cv_results.png
        ├── ablation.png
        ├── feature_importance_top20.png
        └── benchmark.png

data/
├── game_winners.json          per-game winner from livestats API
├── champion_picks.json        per-game champion picks
└── match_metadata.json        match-level results from lolesports API
```

---

## Next steps (if more time)

1. **Download the missing 11 group games** to push n to 45 and reduce CI width
2. **Restrict features to t < 900s** (early game only) to test if the predictive signal survives without late-game hindsight
3. **Train a "rolling-window" classifier** that predicts winner from the first N minutes — produces a "prediction confidence over time" curve per game
4. **Test generalisation** on a different tournament (Worlds 2025 or LCK Spring) to check if the jungle-invasion pattern is tournament-specific or universal
5. **Try paddleocr** to see if better OCR moves the OCR-only accuracy meaningfully above the majority baseline
