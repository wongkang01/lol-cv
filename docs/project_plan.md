# Project Plan: CV-Based Performance Analysis from League of Legends Tournament VODs

**Authors**: Wong Kang, Zachary Muk Chen Eu
**Supervisor**: Dr. G.W.J. (Guido) Bruinsma — University of Twente
**Course**: Data Science Q3 (2025-2A) | Primary topic: CV&IC | Secondary topic: Data Mining
**Date**: April 2, 2026 (revised after supervisor/colleague feedback)

---

## 1. Motivation & Research Gap

### 1.1 The "Why" — Answering Max's Core Question

Professional League of Legends matches played on the **tournament realm** (e.g., the 2026 First Stand tournament) have **no Riot API access**. The tournament realm is a private server — match data, player statistics, and event timelines that are normally available through the Riot Games API simply do not exist for these matches. The only record is the **spectator-mode broadcast** (VOD) uploaded to YouTube/Twitch.

This creates a real, practical problem:

- **For teams and coaches**: They cannot review structured performance data from their own tournament matches unless they manually re-watch footage. There is no automated way to extract positional patterns, objective timing, or gold trajectories from tournament-realm games.
- **For analysts and researchers**: Any quantitative analysis of professional tournament play requires manual annotation of video footage — a time-intensive process that doesn't scale.
- **For the esports data ecosystem**: Platforms like Oracle's Elixir, Games of Legends, and Leaguepedia rely on the Riot API for structured data. Tournament-realm matches are a blind spot.

**Our project fills this gap**: we build a purely CV-based pipeline that extracts structured, quantitative performance data directly from tournament broadcast footage — the only data source available. This is not a redundant alternative to the API; it is the **only automated method** for this class of matches.

### 1.2 What CV Adds Beyond Watching the Replay

A human watching a replay can observe individual moments, but cannot:
- Track 10 champions' zone transitions across a 30-minute match and compute rotation frequency per phase
- Compute team grouping distances relative to objective spawn timers across dozens of matches
- Extract gold difference trajectories at 1-second resolution and correlate them with spatial features
- Systematically compare positional patterns between winning and losing teams across a tournament

CV automates the extraction of spatiotemporal data at a scale and granularity that manual observation cannot match. The downstream ML analysis then identifies which patterns are most predictive of outcomes — insights that are non-obvious even to expert analysts.

### 1.3 Academic Context

This project extends Vafi (2021), who applied process mining to LoL using Riot API data (supervised by Dr. Bruinsma). Our contribution is demonstrating that **equivalent or complementary analysis is possible using only computer vision on video footage**, removing the dependency on API access entirely. This is relevant to the broader esports analytics community where API access is not guaranteed (tournaments, older patches, other games without public APIs).

---

## 2. Research Questions

1. **RQ1** (CV&IC + Data Mining): *"Which CV-extracted spatial features (zone transitions, team grouping near objectives) have the strongest correlation with match outcomes in professional LoL matches?"*

2. **RQ2** (CV&IC): *"How do modern minimap detection models (YOLOv8 vs YOLOv11) compare in accuracy and inference speed for champion tracking on tournament broadcast footage?"*

3. **RQ3** (CV&IC + Data Mining): *"Can a purely CV-based pipeline extract sufficient game-state information from tournament VODs — where no API data exists — to predict match outcomes?"*

These questions are **SMART**:
- **Specific**: Each targets a concrete, measurable outcome (feature correlation, model comparison, prediction accuracy).
- **Measurable**: RQ1 uses correlation coefficients and feature importance scores; RQ2 uses mAP, precision, recall, and FPS; RQ3 uses classification accuracy, F1, and AUC-ROC.
- **Achievable**: The tools exist (YOLO, PaddleOCR, Gemini Flash); the data source is publicly available (YouTube VODs).
- **Relevant**: Directly addresses the CV&IC primary topic and Data Mining secondary topic.
- **Time-bound**: Scoped to be completable within the Q3 project period.

---

## 3. Data Sources

### 3.1 Primary: 2026 First Stand Tournament VODs
- **Source**: YouTube/Twitch spectator-mode broadcasts
- **Format**: 1080p video (MP4/MKV)
- **Properties**: Consistent spectator HUD layout, full-vision minimap (no fog of war), standardized camera control
- **Estimated size**: 15-25 matches, each 25-45 minutes
- **Why this tournament**: No API data exists (tournament realm), consistent broadcast quality, recent patch (all 170 champions possible)

### 3.2 Detection Model Training/Benchmarking
| Dataset | Purpose | Size |
|---------|---------|------|
| boboyes/leagueoflegends-minimap (HuggingFace) | YOLOv11 benchmark — pre-trained models available (nano to xlarge) | CC-BY-NC-4.0 |
| pyLoL's YOLOv8 weights (`yolov12.pt`) | YOLOv8 benchmark — 167 champions, 227 MB | mAP 92.2% |
| Roboflow LoL Minimap | Additional validation set | 4,468 images |
| DeepLeague labeled images | Historical comparison | 100k+ images |

### 3.3 Data Quality Considerations
- **Video compression**: YouTube re-encodes at variable bitrate; 1080p should preserve minimap icon detail but will be validated
- **Champion icon variability**: Skins, in-game transformations (e.g., Kayn forms), and level-up indicators slightly alter icon appearance
- **Known tracking limitations** (per maknee's blog):
  - Clone champions (LeBlanc, Wukong, Neeko, Shaco) create duplicate minimap icons — flag/exclude these matches
  - Yuumi's icon disappears when attached to an ally — tracking gap is a known limitation
  - No single confidence threshold perfectly balances precision/recall — will document threshold tuning via PR curves

---

## 4. Technical Pipeline

### 4.1 Architecture Overview

```
Tournament VOD (YouTube)
         |
         v
  [Frame Extraction] ---- 1 FPS sampling
         |
    +----+----+
    |         |
    v         v
[Minimap     [HUD Region
 Cropping]    Cropping]
    |         |
    v         v
[YOLO        [PaddleOCR]
 Detection]       |
    |         v
    v    Gold, KDA, Timers,
Champion     Items
Positions
    |         |
    +----+----+
         |
         v
  [Feature Engineering]
  - Zone transitions
  - Team grouping
  - Objective context
  - Gold/kill trajectories
         |
         v
  [ML Classification]
  - Random Forest, SVM, 
    Gradient Boosting, MLP
  - Feature importance ranking
         |
         v
  [VLM Tactical Analysis] (Gemini Flash)
  - Key moment annotation
  - Tactical pattern description
         |
         v
  [Embedding & Clustering] (Gemini Embedding 2)
  - Game-state similarity search
  - Unsupervised pattern discovery
```

### 4.2 Stage 1 — Minimap Champion Detection (RQ2)

**Goal**: Track all 10 champion positions on the minimap at 1-second intervals throughout each match.

**Approach**: Benchmark two pre-trained models on tournament VOD frames:

| Model | Source | Architecture | Training Data |
|-------|--------|-------------|---------------|
| pyLoL `yolov12.pt` | pyLoL repo | YOLOv8 (v12 is internal version) | Synthetic composite images |
| boboyes models | HuggingFace | YOLOv11 (5 variants: n/s/m/l/x) | Community-labeled dataset |

**Evaluation protocol**:
1. Sample 200-500 minimap frames from tournament VODs, spanning early/mid/late game
2. Manually annotate ground truth bounding boxes for all visible champion icons
3. Run both models at multiple confidence thresholds (0.2 to 0.8, step 0.05)
4. Compare: mAP@50, mAP@50:95, precision, recall, F1 at each threshold
5. Generate precision-recall curves to determine optimal threshold per model
6. Measure inference time (FPS) on available hardware

**Output**: Per-frame champion positions as `(champion_id, x, y, confidence, timestamp)` tuples.

### 4.3 Stage 2 — OCR HUD Extraction

**Goal**: Extract structured numerical data from the spectator HUD overlay.

**Targets**:
| HUD Element | Location (spectator mode) | Extraction |
|-------------|--------------------------|------------|
| Game timer | Top center | PaddleOCR on cropped region |
| Kill score (blue/red) | Top center | PaddleOCR |
| Gold difference | Top center bar | PaddleOCR |
| Champion levels | Scoreboard panels | PaddleOCR |
| Item builds | Scoreboard panels | Template matching / PaddleOCR |
| Objective timers | Right side | PaddleOCR (when visible) |

**Validation**: Cross-check OCR output against manually recorded values for a subset of frames. Report OCR accuracy (character-level and field-level).

### 4.4 Stage 3 — Feature Engineering

**Spatial features** (from minimap detections):

| Feature | Definition | Game Relevance |
|---------|-----------|----------------|
| Zone transition frequency | Number of zone changes per minute (16x16 grid) | Measures rotation/roaming activity |
| Unique zones visited | Distinct grid cells visited per game phase | Map control breadth |
| Team grouping distance | Average pairwise distance between teammates | Measures team coordination |
| Objective proximity | Distance of each team to dragon/baron pit at spawn time | Objective prioritization |
| Convergence speed | Time for 3+ teammates to move within grouping threshold of objective | Reaction coordination |
| Lane assignment stability | % of time each player spends in their assigned lane per phase | Role adherence vs. flexible rotations |

**Temporal features** (from OCR + detections):

| Feature | Definition | Game Relevance |
|---------|-----------|----------------|
| Gold diff at phase transitions | Gold difference at 15 min, 25 min | Economic state at key moments |
| Gold diff trajectory slope | Linear regression slope of gold diff over time | Momentum direction |
| Kill tempo | Kills per minute per phase | Aggression level |
| Objective sequence | Order and timing of dragon/baron/tower takes | Strategic prioritization |
| First objective timing | Timestamp of first dragon, herald, tower | Early game execution |

**Game phases**:
- Early game: 0-15 minutes (laning phase)
- Mid game: 15-25 minutes (rotations and objective fights)
- Late game: 25+ minutes (teamfights and siege)

**Features explicitly excluded** (with rationale):
- ~~Movement speed~~: Minimap icons move in discrete steps at 1 FPS sampling; per-second displacement is unreliable as a speed proxy (acknowledged per Joschka's feedback)
- ~~Ability usage / spell tracking~~: Not visible on minimap; would require full-screen analysis which is out of scope
- ~~Clone/Yuumi tracking~~: Known limitation of minimap-based detection (per maknee's blog)

### 4.5 Stage 4 — ML Classification (RQ1, RQ3)

**Task**: Binary classification — predict match winner from CV-extracted features.

**Models** (aligning with Data Mining secondary topic):
1. Random Forest — interpretable feature importance via Gini impurity
2. Support Vector Machine — effective on small-N high-dimensional data
3. Gradient Boosting (XGBoost) — strong performance on tabular data
4. Multi-Layer Perceptron — non-linear feature interactions

**Evaluation**:
- 5-fold stratified cross-validation (due to small dataset: ~15-25 matches = 30-50 team-match samples)
- Metrics: Accuracy, F1 (macro), AUC-ROC, precision, recall
- Ablation study: train models on subsets of features (spatial-only, OCR-only, combined) to measure each category's predictive contribution
- Feature importance ranking (permutation importance) to answer RQ1

**Addressing small sample size**:
- Report confidence intervals on all metrics
- Use leave-one-out cross-validation as a sensitivity check
- Discuss statistical power limitations transparently
- Frame results as "indicative correlations" rather than definitive claims

### 4.6 Stage 5 — VLM Tactical Analysis

**Goal**: Use Gemini 3 Flash to generate natural-language tactical analysis of key game moments.

**Approach**:
1. Identify key moments programmatically: large gold swings (>1000g in 60s), multi-kills (3+), objective takes
2. Extract 10-15 second video clips around each key moment
3. Prompt Gemini Flash with the clip + OCR context (gold state, kill score, game timer)
4. Generate structured analysis: what happened, team positioning, tactical quality

**Purpose**: Adds qualitative depth to the quantitative feature analysis. Demonstrates VLM applicability to esports analytics as a supplementary tool (not core to RQ answering, but adds technical depth per course requirement).

**Budget**: ~$0.30/M tokens. At ~50 key moments across all matches, estimated cost < $5.

### 4.7 Stage 6 — Embedding & Clustering

**Goal**: Discover recurring tactical patterns across matches using multimodal embeddings.

**Approach**:
1. Embed minimap frames at key moments using Gemini Embedding 2 (3072-dim vectors)
2. Optionally embed OCR-extracted state alongside the visual frame
3. Cluster embeddings (K-means, DBSCAN) to find groups of similar game states
4. Analyze cluster composition: do certain clusters correlate with winning/losing?
5. Visualize via t-SNE/UMAP dimensionality reduction

**Purpose**: Unsupervised pattern discovery — complements the supervised classification approach and adds technical depth beyond the primary topic.

---

## 5. Existing Work & Our Contribution

### 5.1 Relationship to Prior Work

| Prior Work | What They Did | What We Add |
|-----------|---------------|-------------|
| **pyLoL** (GitHub) | Automated minimap detection pipeline for replay files | Apply to tournament VODs (not replays); benchmark vs. newer models |
| **boboyes** (HuggingFace) | YOLOv11 minimap detection models | Systematic comparison with pyLoL's YOLOv8 on tournament footage |
| **maknee** (blog, 2021) | Faster R-CNN minimap detection; documented clone/Yuumi limitations | Acknowledge these limitations; use YOLO instead of R-CNN; add downstream analysis |
| **DeepLeague** | 100k labeled minimap images, YOLO detection | Our pipeline goes beyond detection to feature engineering and outcome prediction |
| **Vafi (2021)** | Process mining on LoL using Riot API data | We achieve similar analysis using CV only — no API dependency |
| **"Deep Learning for Game State" (2025)** | 97% win prediction from API data | We test whether CV-extracted features achieve comparable prediction |
| **Valoscribe (2026)** | Full VOD-to-structured-data for Valorant | We do the equivalent for LoL, targeting tournament-realm broadcasts |

### 5.2 Our Specific Contributions

1. **First systematic benchmark** of YOLOv8 vs. YOLOv11 for LoL minimap detection on tournament broadcast footage (as opposed to replay screenshots or synthetic data)
2. **End-to-end CV-only pipeline** from tournament VOD to match outcome prediction — demonstrating API-free analysis is viable
3. **Feature importance analysis** identifying which CV-extractable spatial/temporal features best predict professional match outcomes
4. **Multimodal embedding clustering** of game states using Gemini Embedding 2 — a novel application of multimodal embeddings to esports analytics

---

## 6. Implementation Timeline

### Phase 1: Data Collection & Detection Setup (Week 1-2)
- [ ] Download 15-25 First Stand tournament VODs from YouTube
- [ ] Set up frame extraction pipeline (FFmpeg, 1 FPS)
- [ ] Implement minimap cropping from spectator-mode frames
- [ ] Download and load pyLoL YOLOv8 weights and boboyes YOLOv11 weights
- [ ] Run initial detection on sample frames, verify output format
- **Deliverable**: Raw frames extracted, both detection models running

### Phase 2: Detection Benchmark & OCR (Week 2-3)
- [ ] Manually annotate 200-500 ground truth frames for minimap champions
- [ ] Run both models at multiple thresholds, compute mAP/precision/recall
- [ ] Generate precision-recall curves, select optimal thresholds
- [ ] Measure inference speed (FPS) for each model
- [ ] Implement PaddleOCR extraction for HUD elements
- [ ] Validate OCR accuracy on manually-checked subset
- **Deliverable**: Model comparison results (RQ2 answered), validated OCR pipeline

### Phase 3: Feature Engineering (Week 3-4)
- [ ] Process all matches through the best-performing detector
- [ ] Compute spatial features: zone transitions, team grouping, objective proximity
- [ ] Compute temporal features: gold trajectories, kill tempo, objective sequences
- [ ] Define game phases and compute per-phase features
- [ ] Build feature matrix (rows = team-match instances, columns = features)
- **Deliverable**: Complete feature dataset ready for ML

### Phase 4: ML Analysis & Prediction (Week 4-5)
- [ ] Train Random Forest, SVM, XGBoost, MLP classifiers
- [ ] Run 5-fold stratified CV, compute accuracy/F1/AUC-ROC
- [ ] Ablation study: spatial-only vs. OCR-only vs. combined features
- [ ] Permutation importance ranking for all features
- [ ] Statistical significance testing (where sample size permits)
- **Deliverable**: Classification results, feature importance rankings (RQ1, RQ3 answered)

### Phase 5: VLM & Embedding Analysis (Week 5-6)
- [ ] Identify key moments from feature data (gold swings, multi-kills)
- [ ] Run Gemini Flash tactical analysis on key moment clips
- [ ] Embed game-state frames via Gemini Embedding 2
- [ ] Cluster embeddings, visualize with t-SNE/UMAP
- [ ] Analyze cluster-outcome correlations
- **Deliverable**: VLM analysis results, embedding clusters

### Phase 6: Visualization, Report & Presentation (Week 6-7)
- [ ] Generate heatmaps (champion positioning per team/match)
- [ ] Create trajectory plots, gold diff charts, feature importance bar plots
- [ ] Write project reflection report (per Ma'at template)
- [ ] Prepare 10-minute DSDV presentation (non-technical, for "upper management")
- [ ] Final code cleanup and documentation
- **Deliverable**: Report submitted, presentation ready

---

## 7. Evaluation Metrics Summary

| Component | Metrics |
|-----------|---------|
| Minimap detection (RQ2) | mAP@50, mAP@50:95, precision, recall, F1, FPS |
| OCR extraction | Character-level accuracy, field-level accuracy |
| Feature correlation (RQ1) | Permutation importance, Spearman/Pearson correlation, p-values |
| Match prediction (RQ3) | Accuracy, F1 (macro), AUC-ROC, precision, recall (5-fold CV) |
| Embedding clustering | Silhouette score, cluster purity (win/loss composition) |

---

## 8. Technology Stack

| Component | Tool | Justification |
|-----------|------|---------------|
| Object detection | Ultralytics YOLO (v8 + v11) | Industry standard real-time detection; both models available pre-trained for LoL minimap |
| OCR | PaddleOCR | Lightweight, no external API dependency, strong on printed text/numbers |
| VLM analysis | Gemini 3 Flash | Best price-performance ratio ($0.30/M tokens), 86.9% Video-MMMU, native video support |
| Embeddings | Gemini Embedding 2 | Only natively multimodal embedding model; 3072-dim, video support |
| ML classifiers | scikit-learn, XGBoost | Standard ML toolkit, well-suited for tabular feature data |
| Visualization | matplotlib, seaborn, plotly | Standard Python visualization stack |
| Video processing | FFmpeg, OpenCV | Frame extraction, image cropping/resizing |
| Package management | uv | Fast Python package manager |

---

## 9. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Small sample size (15-25 matches) | Weak statistical power for ML | Use CV with confidence intervals; frame as indicative; consider supplementing with ranked VODs if needed |
| Detection accuracy insufficient on tournament VODs | Noisy position data | Benchmark both models; apply post-processing (temporal smoothing, trajectory interpolation) |
| OCR fails on compressed YouTube video | Missing gold/KDA data | Test early; fall back to VLM-based extraction for critical fields |
| Clone champions in matches | False positive positions | Flag matches with LeBlanc/Wukong/Neeko/Shaco; report results with and without these matches |
| Gemini API costs exceed budget | Cannot run VLM/embedding stages | Use Flash-Lite ($0.075/M tokens) for bulk work; limit VLM to key moments only |
| Tournament VODs removed from YouTube | No data source | Download and archive all VODs locally before starting analysis |

---

## 10. Mapping to Report Template

| Report Section | Content Source |
|----------------|---------------|
| **Motivation** | Section 1 of this plan (the "why" — tournament realm has no API) |
| **Research questions** | Section 2 (SMART-formatted RQs) |
| **Source data** | Section 3 (VODs, datasets, quality considerations) |
| **Method** | Sections 4.1-4.5 (pipeline, models, evaluation) |
| **CV&IC: Image representation** | Section 4.2 (minimap frame preprocessing, YOLO input format) |
| **CV&IC: ML comparison** | Section 4.5 (RF vs. SVM vs. XGBoost vs. MLP with 5-fold CV) |
| **Results** | Sections 4.2-4.7 outputs (detection benchmark, feature importance, prediction accuracy, clusters) |
| **Technical depth** | Sections 4.6-4.7 (VLM analysis, multimodal embeddings — beyond primary topic) |
| **ML issues** | Clone champions, class imbalance (win/loss may not be 50/50), small sample size |
| **Generalisation** | Cross-validation results, discussion of applicability to other tournaments/patches |
| **Conclusions** | Which features matter most (RQ1), which model is better (RQ2), is CV-only viable (RQ3) |

---

## 11. Descoped Elements

The following were considered but intentionally excluded to keep the project focused:

| Element | Reason for Exclusion |
|---------|---------------------|
| Riot API integration | Tournament realm has no API data; CV-only is a stronger research angle |
| Movement speed features | Minimap icons move in discrete steps; unreliable at 1 FPS (per Joschka's feedback) |
| Full-screen gameplay analysis | Scope explosion; minimap + HUD provides sufficient structured data |
| SAM 3 segmentation | 1-2 FPS on consumer GPU; designed for natural images, not 15px minimap icons |
| Real-time analysis | Not needed for post-match tournament analysis; real-time adds complexity without research value |
| Clone/Yuumi resolution | Known hard problem; better to acknowledge as limitation than attempt partial solutions |
