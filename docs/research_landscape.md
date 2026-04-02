# Research Landscape: Computer Vision Analysis of League of Legends VODs

## 1. Project Context

**Course**: Data Science Q3, University of Twente (CV&IC primary topic)
**Goal**: Determine if it is possible to gain insight into performance through game key events & tactics, using computer vision on LoL replay/VOD footage.
**Key Resource**: The `league-of-legends-replay-extractor/pyLoL` GitHub repo for extracting spatiotemporal data from LoL videos.
**Reference Paper**: Vafi (2021) - Process mining on LoL using API data extraction (supervised by Dr. Bruinsma). This used the Riot API; your project extends this with **computer vision** on actual game footage.

---

## 2. The pyLoL Replay Extractor (Starting Point)

The `pyLoL` project from the `league-of-legends-replay-extractor` GitHub organization automates extraction of positional and analytics data from LoL replay videos. Key capabilities:

- Frame-by-frame analysis of replay footage
- Player position tracking (every second)
- Ward detection and tracking
- Reported performance: mAP 92.2%, Precision 91.3%, Recall 90.2%
- Self-augmented dataset generation for training
- Automatic replay file handling

This is your **data pipeline foundation**. Your project can build on this by adding deeper analysis layers.

---

## 3. Existing LoL CV Research & Tools

### 3.1 Minimap Analysis (Most Mature Area)

| Project | What It Does | Tech Stack |
|---------|-------------|------------|
| **DeepLeague** | Champion detection on minimap; 100k+ labeled images | YOLO, PyTorch, Deep Learning |
| **LeagueMinimapDetectionOpenCV** | Champion detection using classical CV | OpenCV, template matching |
| **Minimap_Detector** | Detects 169 champion classes | Multi-object detection |
| **League-Minimap-Scanner** | CNN-based enemy champion identification | CNN, HoughCircles |
| **MinimapTracker-MobileNet** | Lightweight tracking | OpenCV, PyTorch, MobileNet |

**Key Paper**: "Real-Time Player Tracking Framework on MOBA Game Video" (IEEE, 2024) - Extracts real-time player trajectories from MOBA video through object detection on the minimap with synthetic image generation for training.

### 3.2 OCR & HUD Data Extraction

| Project | What It Does | Tech |
|---------|-------------|------|
| **LeagueOCR** | OCR-based spectator mode data collection | OCR engines |
| **League-OCR-HUD** | Google Cloud Vision API for spectator data | Google Vision API |
| **league-of-legends-ocr** | General LoL text extraction | Tesseract OCR |
| **ScoreSight** | Open-source OCR for gaming scoreboards | OCR, OBS integration |

### 3.3 Game State Recognition

| Project | What It Does | Tech |
|---------|-------------|------|
| **LeagueAI** | Game state detection from screenshots | OpenCV, PyTorch, YOLOv3 |
| **MOBA-AI-Gamer** | Bot AI using screen analysis, 20-object detection | YOLOv5 |
| **LeaguePyBot** | Python CV bot for LoL | Computer Vision |

### 3.4 Key Academic Papers on LoL/Esports CV

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| "Autohighlight: Highlight Detection in LoL" | 2022 | Positive-unlabeled learning for detecting highlight moments in broadcasts |
| "Deep Learning for Game State and KPI in LoL" | 2025 | 97% accuracy win prediction; identifies KPIs for performance |
| "Real-Time Player Tracking on MOBA Game Video" | 2024 | Synthetic training data, minimap tracking, occlusion handling |
| "Commentary Generation from Multimodal Game Data" | 2025 | VLMs generating commentary from screenshots + JSON data |
| "Round Outcome Prediction in VALORANT" | 2024 | TimeSformer on minimap tactical features; 81% accuracy |
| "Valoscribe: Turning Broadcasts into Structured Data" | 2026 | Full VOD-to-structured-data pipeline using YOLO + OCR |
| "Learning to Automatically Spectate Games" | 2023 | Mask R-CNN for automated observing in esports |

---

## 4. Modern Vision Techniques Applicable to Your Project

### 4.1 Object Detection (Champion/Objective Detection)

**YOLOv8-v11 & RT-DETR**: State-of-the-art real-time detectors.
- YOLOv10 is 15% faster than v9, 25% faster than v8
- RT-DETR: First real-time end-to-end Transformer detector (NMS-free), 108 FPS on T4 GPU
- **Application**: Detect champions, abilities, objectives, minion waves, wards frame-by-frame

### 4.2 Segment Anything Model (SAM)

Meta's promptable segmentation model (SAM 3 supports text prompts).
- **Application**: Segment individual champions in cluttered teamfights, isolate ability effects, separate HUD from gameplay area
- Zero-shot capability means it works without game-specific training

### 4.3 Action Recognition (Event Detection)

**VideoMAE**: Self-supervised video pre-training with masked autoencoders.
**TimeSformer**: Transformer-based, decomposes video into spatial+temporal patches.
- **Application**: Detect teamfights, ganks, objective takes, lane rotations, and phase transitions from raw video

### 4.4 OCR for HUD Extraction

**PaddleOCR**: Lightweight, supports 100+ languages, no external dependencies.
**TrOCR**: Transformer-based (ViT encoder + BERT decoder), strong on printed text.
- **Application**: Extract gold counts, KDA, champion levels, item builds, cooldown timers, objective timers

### 4.5 Vision-Language Models (VLMs)

**GPT-4V/o, Claude Vision, Gemini, LLaVA, Qwen-VL**: Combine visual understanding with language reasoning.
- **Application**: Analyze screenshots for tactical insights, generate natural-language commentary about game state, identify strategic patterns
- **Limitation**: High latency (seconds per inference) - best for post-game analysis, not real-time

### 4.6 Video Understanding Models

**Video-ChatGPT, VideoLLaMA**: Process video sequences with LLM reasoning.
- **Application**: Summarize teamfights, describe rotations, generate play-by-play analysis
- Trained on 100k+ video-instruction pairs

---

## 5. Chosen Project Direction (Revised April 2)

### Approach: CV-Only Performance Analysis from Tournament VODs

**Data source**: 2026 First Stand tournament VODs (spectator mode broadcasts from YouTube/Twitch). Tournament realm matches have **no Riot API data**, making CV the only way to extract structured performance data — this is the core research justification.

**Why not the Riot API hybrid approach?**
- Tournament realm matches are not accessible via the API
- Spectator-mode broadcasts guarantee consistent HUD layout and full minimap visibility
- Focusing on CV-only keeps the project tightly scoped around the primary topic (CV&IC)
- The Riot API client remains in the codebase as an optional utility but is not part of the core pipeline

### Pipeline Overview

1. **Detect** — Track champion positions on the minimap using YOLO (benchmark pyLoL YOLOv8 vs boboyes YOLOv11)
2. **Extract** — OCR the spectator HUD for gold difference, kill score, objective timers, item levels
3. **Analyse** — VLM tactical analysis of key moments via Gemini Flash
4. **Engineer** — Compute spatial features (zone transitions, team grouping near objectives, positional heatmaps) and correlate with OCR-extracted game state (gold diff, item levels, objective timers)
5. **Predict** — ML classifiers to identify which CV-extracted features have the highest correlation with match outcome
6. **Embed & Cluster** — Gemini Embedding on game-state frames for unsupervised pattern discovery

### Core Analysis: Feature Correlation with Win Rate

The central experiment: which CV-extracted features best predict match outcomes?

| Feature Category | Specific Features | Source |
|-----------------|-------------------|--------|
| **Zone transitions** | Rotation frequency, unique zones visited per phase, lane swap timing | Minimap YOLO |
| **Team grouping** | Grouping distance near objectives, grouping timing relative to objective spawns | Minimap YOLO |
| **Objective context** | Team proximity to dragon/baron at spawn time, convergence speed | Minimap YOLO + OCR timers |
| **Gold state** | Gold difference at key events, gold diff trajectory over time | OCR |
| **Item progression** | Item completion timing, item level at key moments | OCR |
| **Kill differential** | Kill score at phase transitions, kill tempo | OCR |

**Not included** (with rationale):
- ~~Movement speed~~: Minimap icons move in discrete steps; per-second sampling yields unreliable speed values (acknowledged per Joschka's feedback)
- ~~Clone/Yuumi detection~~: Champions like LeBlanc, Wukong, Neeko, Shaco create duplicate minimap icons; Yuumi's icon disappears when attached. These are **known limitations** of minimap-based tracking (per maknee's blog)

### Research Questions (Revised)

1. *"Which CV-extracted spatial features (zone transitions, team grouping near objectives) have the strongest correlation with match outcomes in professional LoL matches?"*
2. *"How do modern minimap detection models (YOLOv8 vs YOLOv11) compare in accuracy and inference speed for champion tracking on tournament broadcast footage?"*
3. *"Can a purely CV-based pipeline extract sufficient game-state information from tournament VODs — where no API data exists — to predict match outcomes?"*

### Known Limitations

1. **Clone champions**: LeBlanc, Wukong, Neeko, and Shaco create duplicate minimap icons that the detector cannot distinguish from real champions. These matches may need to be flagged or excluded.
2. **Yuumi**: Her icon disappears from the minimap when attached to an ally. The detector cannot track what is not visible.
3. **Fog of war**: In spectator mode with fog of war enabled, champion icons may be hidden. Tournament broadcasts typically use full-vision spectator mode, mitigating this.
4. **OCR accuracy**: HUD text extraction depends on video resolution and compression quality. Tournament broadcasts at 1080p should be reliable.
5. **Detection thresholds**: No single confidence threshold perfectly balances true detections vs false positives (per maknee's work). Threshold tuning via precision-recall curves will be documented.

---

## 6. Relevant Datasets & Benchmarks

| Dataset | Domain | Contents |
|---------|--------|----------|
| DeepLeague labeled images | LoL minimap | 100k+ labeled champion positions |
| Roboflow LoL Minimap | LoL minimap | Community-labeled detection dataset (4,468 images) |
| boboyes/leagueoflegends-minimap | LoL minimap | HuggingFace dataset for YOLOv11 training (Feb 2026) |
| League Minimap Dataset (Kaggle) | LoL minimap | Champion minimap icon dataset |
| TLoL | LoL replays | ~2,488 early-game replays at 4 frames/sec |
| ESTA | CS:GO | 1,558 pro matches with trajectories/actions |
| CS-lol | LoL broadcasts | Pro matches with scene annotations |
| 2026 First Stand VODs | LoL tournament | **Primary data source** — spectator-mode broadcasts |

---

## 7. Key References to Cite

### LoL-Specific
1. DeepLeague - Minimap champion detection with 100k+ labeled images
2. pyLoL - Automated spatiotemporal data extraction from LoL videos (mAP 92.2%)
3. VisuaLeague (Afonso et al., 2019) - Player performance analysis using spatial-temporal data
4. Kho et al. (2020) - Logic mining in LoL, sequence of objective events
5. Charleer et al. (2018) - Esports dashboards for LoL and CS:GO
6. "Deep Learning for Game State and KPI in LoL" (2025) - 97% win prediction accuracy

### Esports General
7. "Real-Time Player Tracking Framework on MOBA Game Video" (IEEE, 2024)
8. "Autohighlight: Highlight Detection in LoL" (2022)
9. "Commentary Generation from Multimodal Game Data" (ACL, 2025)
10. Valoscribe (2026) - Full VOD-to-structured-data for Valorant
11. "Round Outcome Prediction in VALORANT" (IEEE, 2024) - TimeSformer on minimap
12. PandaSkill (2025) - Player performance evaluation

### Minimap Detection (Added April 2)
13. boboyes/leagueoflegends-minimap-detection (HuggingFace, Feb 2026) - YOLOv11, 5 model variants, CC-BY-NC-4.0
14. jparedesDS/lol-map-tracking-object-detection (HuggingFace) - YOLOv11m, mAP@50 99.3%, precision 97.8%
15. Henry Zhu / maknee (2021) - "ML with LoL Minimap Detection" blog series: synthetic data generation, Faster R-CNN, discussion of Yuumi/clone edge cases
16. PandaScore (2018) - Two-stage approach: champion-agnostic detection + classifier

### Methodology
17. YOLOv11 (Ultralytics) - Current real-time object detection
18. RT-DETR (CVPR 2024) - Transformer-based real-time detection
19. PaddleOCR (Baidu) - Lightweight OCR toolkit
20. Gemini Flash / Gemini Embedding (Google) - VLM analysis and multimodal embeddings

---

## 8. Technology Deep Dives (Added March 23)

### 8.1 Gemini Embedding 2 — Multimodal Embeddings

**What it is**: Google's first natively multimodal embedding model (announced March 10, 2026). Embeds text, images, video, audio, and PDFs into a single 3072-dimensional vector space using Matryoshka Representation Learning (flexible dimensionality).

**Key specs**:
- Video input: up to 120 seconds per request (MP4, MOV)
- Images: up to 6 per request
- Text: 8,192 tokens
- Pricing: $0.25/M tokens (text/image/video)
- #1 on MTEB English benchmark (68.32, +5.09 above second place)
- Video retrieval benchmarks: 68.8 (vs Amazon Nova 2: 60.3, Voyage: 55.2)

**Applicability to LoL project**: MEDIUM-HIGH potential, NOT too complex.
- You could embed minimap frames, OCR-extracted text stats, and even commentary audio into the same vector space
- Use case: "find game moments that are similar to this teamfight frame" — cross-modal similarity search
- Build a retrieval system: embed frames from winning games, then query with frames from a new game to find similar tactical situations and their outcomes
- Much simpler API than training your own model — it's just an embedding endpoint
- **Limitation**: 120-second video cap means you'd need to chunk games into segments
- **Compared to CLIP**: Gemini Embedding 2 handles video natively (CLIP only does text+image), and uses a single shared transformer rather than separate encoders

**Verdict**: This is a realistic and impressive addition to your project. You don't need to train anything — just call the API and use the embeddings for similarity search or clustering.

---

### 8.2 SAM 3 vs pyLoL's YOLOv12 for Minimap Tracking

**SAM 3 (Segment Anything Model 3)** — Released November 2025, introduces Promptable Concept Segmentation (text-prompted segmentation of all instances of a concept).

**SAM 3 specs**:
- Video: memory-based tracking across frames with occlusion handling
- Speed: ~44 FPS on A100 GPU, but **1-2 FPS on consumer GPU (RTX 3070 Ti)**
- Precision: pixel-level segmentation masks
- Requires initial prompts (point, box, mask, or text) for each object to track

**pyLoL's YOLOv12 specs**:
- Trained specifically on LoL champion minimap icons
- 167 champion classes supported
- mAP: 92.2%, Precision: 91.3%, Recall: 90.2%
- 512x512 input, outputs bounding boxes
- Runs 60+ FPS on consumer hardware

**Head-to-head for minimap champion tracking**:

| Factor | SAM 3 | YOLOv12 (pyLoL) |
|--------|-------|-----------------|
| Speed (consumer GPU) | 1-2 FPS | 60+ FPS |
| Object size handling | Struggles with <15px objects | Trained on 15x15 minimap icons |
| Setup | Needs prompts per champion | Zero-shot after training |
| Output | Pixel masks | Bounding boxes |
| Domain fit | General-purpose | LoL-specific |
| Model size | Very large | 227MB |

**Verdict: Stick with YOLOv12 for minimap detection.** SAM 3 is the wrong tool here. It's designed for general-purpose segmentation of natural images and video, not for detecting 15x15 pixel stylized icons on a game minimap. The speed penalty alone (30-60x slower) makes it impractical. SAM 3 would be interesting if you needed to segment ability effects or champions in the main gameplay view (not minimap), but that's a much harder problem and not what pyLoL is designed for. Your best bet is to use pyLoL's purpose-built YOLOv12 which was literally trained for this exact task.

**Where SAM 3 COULD add value**: If you expand beyond minimap analysis to full-screen gameplay analysis — segmenting ability particle effects, detecting champion models in teamfights, or isolating HUD elements — SAM 3's zero-shot capability could be useful as a supplementary tool.

---

### 8.3 Vision-Language Model Benchmarks — Post-November 2025 Models Only

**Only models released after November 2025 are included.** Benchmarks sourced from official announcements, Artificial Analysis, and pricepertoken.com leaderboards as of March 2026.

#### Frontier Models (API-only)

| Model | Release | MMMU | MMMU-Pro | Video-MMMU | Video Support | Context | Cost (input/M tok) |
|-------|---------|------|----------|------------|---------------|---------|-------------------|
| **Claude Opus 4.6** | Mar 2026 | — | **85.1%** | — | Frame sampling | 200K | ~$5.00 |
| **GPT-5.4** | Mar 2026 | — | ~66% | — | Native video | 1M | $2.50+ |
| **Gemini 3.1 Pro** | Feb 2026 | — | — | — | Native (1M+) | 2M | ~$1.25 |
| **Gemini 3 Flash** | Jan 2026 | ~79% | **81.2%** | **86.9%** | Native video | 200K | ~$0.30 |
| **Gemini 3.1 Flash-Lite** | Mar 2026 | — | **76.8%** | **84.8%** | Native video | 200K | ~$0.075 |
| **GPT-5** | Dec 2025 | 79.1% | 66% | — | Native video | 128K | $2.50 |
| **Claude Opus 4.5** | Feb 2025 → refresh Dec 2025 | 80.7% | — | — | Frame sampling | 200K | $5.00 |
| **Claude Sonnet 4.5** | late 2025 | 77.8% | — | — | Frame sampling | 200K | $3.00 |
| **Gemini 2.5 Pro** | Dec 2025 | 77.6% | — | 83.6% | Native (1M+) | 2M | $1.25 |
| **o4 Mini High** | early 2026 | 79.2% | — | — | Frame sampling | 128K | varies |

#### Open-Weight / Open-Source Models

| Model | Release | MMMU | MMMU-Pro | Video-MMMU | Video Support | Params (active) | License |
|-------|---------|------|----------|------------|---------------|----------------|---------|
| **Qwen3-VL-235B** (MoE) | Jan 2026 | Rivals frontier | — | — | Native, 256K context | 22B active | Apache 2.0 |
| **Qwen3-VL-32B** | Jan 2026 | Top open-source | — | — | Native | 32B | Apache 2.0 |
| **InternVL3-78B** | Dec 2025 | 72.2% | — | — | Limited | 78B | Open |
| **DeepSeek-VL2** (MoE) | Dec 2025 | — | — | — | Up to 1hr video | 4.5B active | Open |
| **Kimi-VL-A3B** | Jan 2026 | 64.0% | 46.3% | — | Native | 3B active | Open |

#### Key Specialized Benchmarks (Post-Nov 2025 Models)

**OCR Performance (OCRBench)**:
1. DeepSeek-VL2: **834** (Dec 2025, open weights)
2. Qwen3-VL: high (Jan 2026, open weights)
3. GPT-4o: 736 (baseline comparison)

**Video Understanding (Video-MMMU)**:
1. Gemini 3 Flash: **86.9%** (Jan 2026)
2. Gemini 3.1 Flash-Lite: **84.8%** (Mar 2026)
3. Gemini 2.5 Pro: **83.6%** (Dec 2025)

**Spatial Reasoning (SpatialBench)**:
1. Qwen3-VL: **13.5** (Jan 2026)
2. Gemini 3.0 Pro: 9.6
3. GPT-5.1: 7.5

**Visual Reasoning (MMMU-Pro)**:
1. Claude Opus 4.6: **85.1%** (Mar 2026)
2. Gemini 3 Flash: **81.2%** (Jan 2026)
3. Gemini 3.1 Flash-Lite: **76.8%** (Mar 2026)
4. GPT-5/5.4: ~66%

**Abstract Reasoning (GPQA Diamond)**:
1. Gemini 3.1 Pro: **94.3%**
2. GPT-5.4: 92.8%
3. Claude Opus 4.6: 91.3%
4. Gemini 3 Flash: 90.4%

---

#### Practical Recommendation for LoL Project (Updated March 2026)

**Best overall pick: Gemini 3 Flash** — It dominates the price-to-performance ratio for your use case. At $0.30/M tokens it's 17x cheaper than Claude Opus 4.6, yet scores 81.2% on MMMU-Pro and **86.9% on Video-MMMU** (the best video understanding score of any model). It natively processes video without frame sampling, which matters for analyzing game footage. For a student budget, this is the clear winner.

**For OCR (gold/KDA extraction)**: DeepSeek-VL2 (834 OCRBench, open weights = free on university GPUs) or Gemini 3 Flash as a simpler API alternative.

**For the highest-quality tactical analysis on key moments**: Claude Opus 4.6 (85.1% MMMU-Pro, best visual reasoning) — use sparingly on 5-10 critical game moments per match to control costs.

**For running locally with zero API cost**: Qwen3-VL-32B or Qwen3-VL-235B — top open-weight models, Apache 2.0 licensed, native video support. If your university has GPU cluster access, this is the way to go.

**For ultra-cheap bulk processing**: Gemini 3.1 Flash-Lite at $0.075/M tokens — scores 84.8% on Video-MMMU, perfect for processing hundreds of game frames across many matches.

**For multimodal embeddings (similarity search / clustering)**: Gemini Embedding 2 ($0.25/M tokens) — embeds video, images, and text into the same 3072-dim vector space. No other model offers this.

---

## 9. Research Questions (Final — April 2)

1. *"Which CV-extracted spatial features (zone transitions, team grouping near objectives) have the strongest correlation with match outcomes in professional LoL matches?"*

2. *"How do modern minimap detection models (YOLOv8 vs YOLOv11) compare in accuracy and inference speed for champion tracking on tournament broadcast footage?"*

3. *"Can a purely CV-based pipeline extract sufficient game-state information from tournament VODs — where no API data exists — to predict match outcomes?"*

---

## 10. Replay File Access & Data Acquisition (Added March 28)

### 10.1 How LoL Replays Work

`.rofl` (Replay OF LoL) files are **not video files** — they contain encrypted game state data that the LoL client reconstructs visually. They are stored at `C:\Users\<user>\Documents\League of Legends\Replays\`.

**Critical constraint**: Replays are **patch-locked** — a replay from patch 15.1 cannot play on patch 15.2. Patches cycle every ~2 weeks, so replays have a short usable window.

### 10.2 Sources of Replay Data

| Source | Type | Availability |
|--------|------|-------------|
| **Your own games** | .rofl files | Save from client within current patch |
| **replays.xyz** | .rofl files | Indexes Challenger matches, archives old client versions |
| **League of Graphs** | .rofl files | Browse by champion/rank/region |
| **YouTube VODs** | Video | Pro/ranked spectator footage, no expiry |
| **DeepLeague dataset** | Labeled images | 100k+ minimap images with bounding boxes |
| **Roboflow Universe** | Labeled images | 4,468 images (jasperan/league-of-legends-detection) |
| **TLoL dataset** | Structured data | ~2,488 early-game replays at 4 frames/sec |
| **Pro match replays** | ❌ Not available | Played on private Tournament Realm |

### 10.3 Practical Data Strategy

1. **Immediate (no replay needed)**: Use YouTube spectator VODs as video input for OCR and VLM analysis. Use DeepLeague/Roboflow datasets for training/testing minimap detection.
2. **Short-term**: Play/spectate ranked games, save replays within the current patch, extract via pyLoL before patch expires.
3. **Alternative**: Screen-record spectator mode directly — avoids .rofl format entirely.

pyLoL's pipeline: launches LoL client → plays .rofl → captures minimap frames via screen capture → runs YOLO detection. Requires **Windows + LoL client installed**.

---

## 11. pyLoL Model Details (Added March 28)

### 11.1 Architecture Clarification

The model is branded as "YOLOv12" in the pyLoL README, but the repo pins `ultralytics==8.0.124`, which only supports up to **YOLOv8**. The "v12" is a project-internal version number, not the YOLO architecture version. The actual model is **YOLOv8**.

### 11.2 Weights & Training

- **Weights file**: `yolov12.pt` (227 MB), hosted on Google Drive
- **Download**: `https://drive.google.com/uc?export=download&id=1ymd7Thcz1XdejEW94LjSFDl3zDqYH0qq`
- **Also on Roboflow**: project `lolpago-multi-tracking-service` (version 18)
- **Input size**: 512×512 PNG
- **Champions supported**: 167 / 170
- **Metrics**: mAP 92.2%, Precision 91.3%, Recall 90.2%

### 11.3 Self-Augmented Training Pipeline

pyLoL describes a synthetic data generation approach:
1. Download champion portrait images from Riot's DataDragon CDN
2. Composite portraits onto minimap backgrounds with augmentation
3. Include "ping" and "turret" icons to handle occlusion
4. No manual labeling required

**Caveat**: The actual generation code is not fully open-sourced in the repo — the described pipeline appears partially implemented or lives elsewhere.

### 11.4 Loading the Model

```python
from ultralytics import YOLO
model = YOLO("data/models/yolov12.pt")
results = model.predict("minimap_frame.png", conf=0.4)
```

---

## 12. Riot API — Descoped to Optional Utility (Updated April 2)

The Riot API client remains in the codebase (`extraction/api.py`) as an optional utility for future work or validation against ranked solo queue matches. However, it is **not part of the core pipeline** because:

1. **Tournament realm matches have no API data** — the 2026 First Stand VODs are our primary data source, and the tournament realm is a completely separate server with no public API access.
2. **CV-only is a stronger research angle** — it demonstrates that structured performance data can be extracted purely from video, which is the core CV&IC contribution.
3. **Simpler project scope** — removes the need for API key management, rate limiting, and data fusion complexity.

The API could be revisited as a bonus for validating CV accuracy against ranked matches where both data sources are available.
