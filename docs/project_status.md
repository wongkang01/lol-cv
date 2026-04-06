# Project Status — LoL CV Analysis

**Last updated**: April 6, 2026
**Stage**: Phase 1 (Data Collection) in progress, Phase 2 (Detection) validated

---

## Overview

We are building a CV-only pipeline that extracts structured performance data from League of Legends tournament broadcast VODs — specifically the 2026 First Stand international tournament (Knockouts through Grand Final). The core research question is whether minimap-based spatial features (zone transitions, team grouping, objective convergence) can predict match outcomes.

---

## What's Changed vs research_landscape.md

| Topic | research_landscape.md says | Actual status |
|-------|---------------------------|---------------|
| **Detection model** | "Benchmark pyLoL YOLOv8 vs boboyes YOLOv11" | pyLoL YOLOv8 weights unavailable (Google Drive blocked). Using **boboyes YOLOv11m** (39MB, 170 classes, works well). YOLOv11n (5.3MB) also available for speed comparison. |
| **OCR for HUD** | "OCR the spectator HUD for gold, kill score, timers" | **Blocked**: the 7 downloaded games use 3 different regional broadcast HUD layouts (LCK, CBLOL, LLA) with different scoreboard positions. OCR regions only work for one layout. International stage VODs (Semifinals + Finals) use the standard LoL Esports layout — these are the priority for re-download. |
| **Data source** | "Tournament VODs from YouTube" | VOD discovery is fully automated via lolesports API. YouTube download is blocked by bot detection after ~6 videos. **12 YouTube URLs need manual download** (see below). |
| **Champion identification** | Not discussed | **Solved**: lolesports livestats API provides exact champion picks per game (blue/red side, roles, player names). `detect_frame_filtered()` uses this to narrow YOLO from 170 classes to the exact 10 per game — eliminates false positives. |
| **Riot API** | "Descoped — not part of core pipeline" | The lolesports API (public, no auth) is used for VOD discovery and champion metadata, but this is broadcast metadata, not Riot match data. The Riot Games API for match timelines remains descoped. |
| **VLM analysis** | "Gemini Flash tactical analysis" | Not yet started. Planned for Phase 5 after spatial features are validated. |
| **Embedding & clustering** | "Gemini Embedding 2" | Not yet started. Planned for Phase 5. |

---

## Current Architecture

```
                     lolesports API
                    /              \
           VodDiscovery      MatchMetadataFetcher
           (VOD URLs +       (champion picks,
            timestamps)       roles, sides)
                |                    |
                v                    v
          VodProcessor         champion_picks.json
          (download,                 |
           extract frames,           |
           phase filter,             |
           crop minimap + HUD)       |
                |                    |
                v                    v
          MinimapTracker -----> detect_frame_filtered()
          (YOLOv11m)           (only 10 valid champions)
                |
                v
          SpatialFeatures
          (zone transitions,
           team grouping,
           objective convergence)
                |
                v
          WinPredictor
          (RF, SVM, GBM, MLP)
```

---

## Implemented Modules (all tested, 132 tests passing)

| Module | File | Status |
|--------|------|--------|
| **Minimap detection** | `extraction/minimap.py` | Done — `detect_frame()` + `detect_frame_filtered()` |
| **VOD discovery** | `extraction/vod_discovery.py` | Done — 39 games discovered with URLs + timestamps |
| **VOD processing** | `extraction/vod_processor.py` | Done — download, frame extraction, phase filtering, HUD cropping |
| **Match metadata** | `extraction/match_metadata.py` | Done — champion picks for all 39 games from livestats API |
| **Detection benchmark** | `extraction/benchmark.py` | Done — multi-model comparison framework |
| **OCR extraction** | `extraction/ocr.py` | Done (code) — but HUD regions only calibrated for international broadcast layout |
| **VLM analysis** | `extraction/vlm.py` | Done (code) — not yet run on data |
| **Spatial features** | `features/spatial.py` | Done — zone transitions, grouping, convergence, heatmaps, 48 features |
| **Temporal features** | `features/temporal.py` | Done — phase detection, event tempo, gold slope, lane stability |
| **Win prediction** | `analysis/classifiers.py` | Done — RF/SVM/GBM/MLP with ablation study |
| **Clustering** | `analysis/clustering.py` | Done — K-means/DBSCAN with outcome correlation |
| **Visualization** | `visualization/plots.py` | Done — heatmaps, trajectories, PR curves, feature importance |
| **Pipeline orchestrator** | `pipeline.py` | Done — chains all stages |

---

## Data Status

**Scope correction** (Apr 6): Previous ingestion pulled games from regional finals (LCK, CBLOL, LLA) that happen to share block names ("Finals", "Knockouts") with First Stand. These have been deleted. The target is now strictly the **First Stand 2026 international tournament**: Mar 16 (Groups) through Mar 22 (Grand Final), with the 8 attending teams only.

### Target Dataset: First Stand 2026

- **13 matches** (10 Groups + 2 Semifinals + 1 Final)
- **45 games** (all best-of-5 or best-of-3 bo5)
- **~25 hours of gameplay**
- **8 teams**: G2, BLG, BFX, TSW, GEN, JDG, LYON, LOUD
- **Dates**: 2026-03-16 to 2026-03-22

### Downloaded: 10 games (knockout stage fully processed)

### Needs Manual Download: 10 unique YouTube VODs (35 games total)

Save each video as MP4 into `data/raw/<match_id>/video.mp4`. Each URL covers **all games of that match** — one download per match.

#### Group Stage (Mar 16-20) — 10 matches, 35 games

| # | Match | URL | Games | Gameplay window |
|---|-------|-----|-------|-----------------|
| 1 | BFX vs BLG | https://www.youtube.com/watch?v=daMNjuX7_os | G1-5 | 7:44 – 192:32 |
| 2 | TSW vs G2 | https://www.youtube.com/watch?v=KJ_Y9Y2Kc1o | G1-3 | 5:10 – 126:18 |
| 3 | JDG vs GEN | https://www.youtube.com/watch?v=GjSUFWnAVWg | G1-3 | 9:23 – 104:23 |
| 4 | LYON vs LOUD | https://www.youtube.com/watch?v=NTUNkcig4ug | G1-5 | 3:41 – 219:07 |
| 5 | BLG vs G2 | https://www.youtube.com/watch?v=4UfsF6PKjOA | G1-3 | 42:31 – 164:58 |
| 6 | TSW vs BFX | https://www.youtube.com/watch?v=B5Hmb67-pRI | G1-3 | 10:06 – 126:58 |
| 7 | LYON vs GEN | https://www.youtube.com/watch?v=XuKYbGN1pEk | G1-3 | 42:56 – 152:04 |
| 8 | JDG vs LOUD | https://www.youtube.com/watch?v=GYFJBSyvT98 | G1-3 | 10:20 – 116:45 |
| 9 | BFX vs G2 | https://www.youtube.com/watch?v=oryyKR0Em6Q | G1-3 | 0:07 – 143:00 |
| 10 | JDG vs LYON | https://www.youtube.com/watch?v=As88gGER5CY | G1-4 | 0:10 – 191:00 |

#### Knockout Stage (Mar 21-22) — 3 matches, 10 games — **✓ DOWNLOADED & PROCESSED**

| # | Match | URL | Games | Status |
|---|-------|-----|-------|--------|
| 11 | G2 vs GEN (Semi) | https://www.youtube.com/watch?v=rdzerQ9TYpo | G1-3 | ✓ Processed |
| 12 | JDG vs BLG (Semi) | https://www.youtube.com/watch?v=05WupJLWfSI | G1-3 | ✓ Processed |
| 13 | G2 vs BLG (Final) | https://www.youtube.com/watch?v=I4uLe1RegxQ | G1-4 | ✓ Processed |

---

## Next Steps

### Immediate (this week)
1. **Download priority VODs** (#1-3 above) — these are the international stage with standard HUD
2. **Re-run pipeline with filtered detection** on all available games
3. **Validate OCR** on international broadcast HUD (G2 vs GEN, JDG vs BLG, G2 vs BLG)

### After data collection
4. **Detection benchmark** (RQ2) — compare YOLOv11n vs YOLOv11m on tournament frames
5. **Feature engineering** — compute full spatial + temporal features with filtered detection
6. **ML prediction** (RQ1, RQ3) — train classifiers, feature importance ranking, ablation study
7. **VLM analysis** — Gemini Flash on key moments (gold swings, teamfights)
8. **Visualization & report** — heatmaps, trajectories, feature importance plots

---

## Research Questions (unchanged)

1. *"Which CV-extracted spatial features (zone transitions, team grouping near objectives) have the strongest correlation with match outcomes in professional LoL matches?"*
2. *"How do modern minimap detection models (YOLOv8 vs YOLOv11) compare in accuracy and inference speed for champion tracking on tournament broadcast footage?"*
3. *"Can a purely CV-based pipeline extract sufficient game-state information from tournament VODs — where no API data exists — to predict match outcomes?"*

**Note on RQ2**: The original plan compared YOLOv8 (pyLoL) vs YOLOv11 (boboyes). The pyLoL weights are no longer available (Google Drive link dead). We will instead compare **four YOLOv11 size variants** (nano 5.3MB, medium 39MB, large 49MB, xlarge 109MB) — a meaningful analysis of model size vs accuracy/speed tradeoffs for minimap champion detection on tournament broadcast footage.
