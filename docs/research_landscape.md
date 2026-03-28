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

## 5. Potential Project Avenues (Ranked by Ambition & Feasibility)

### Avenue A: Enhanced Minimap Tracking + Macro Strategy Analysis
**Feasibility**: HIGH | **Impressiveness**: MEDIUM-HIGH

Build on pyLoL's position tracking to create a macro strategy analyzer:
1. Use YOLOv10/v11 to track all 10 champions on the minimap every second
2. Construct movement trajectories and heatmaps over time
3. Detect macro events: rotations, split-push setups, objective groupings
4. Compare winning vs. losing macro patterns
5. Supplement with Riot API data for gold/XP context

**Why it works**: Minimap detection is the most mature area with existing labeled data (DeepLeague's 100k+ images). You can focus your novelty on the *analysis layer* rather than reinventing detection.

### Avenue B: Multi-Modal Game State Extraction Pipeline
**Feasibility**: MEDIUM-HIGH | **Impressiveness**: HIGH

Build a comprehensive pipeline that combines multiple CV techniques:
1. **Minimap tracking** (YOLO) for positions
2. **OCR** (PaddleOCR/TrOCR) for gold, KDA, items, timers
3. **Object detection** for on-screen events (ability effects, champion deaths)
4. Fuse all extracted data into a structured timeline
5. Apply data mining/process mining to find performance patterns

**Why it's impressive**: Demonstrates breadth of CV techniques + data fusion, directly echoing the reference paper's process mining approach but from *video* instead of API data.

### Avenue C: Teamfight Detection & Analysis
**Feasibility**: MEDIUM | **Impressiveness**: HIGH

Focus specifically on teamfight events:
1. Use action recognition (VideoMAE/TimeSformer) to detect when teamfights occur
2. Extract participants, outcomes, ability usage from fight footage
3. Analyze what distinguishes winning vs. losing teamfights
4. Identify key moments (first death, ultimate usage, positioning)

**Why it's interesting**: Teamfights are the highest-impact events in LoL and are poorly captured by API data alone.

### Avenue D: VLM-Powered Game Analysis
**Feasibility**: MEDIUM | **Impressiveness**: VERY HIGH

Use modern multimodal AI to generate tactical insights:
1. Sample key frames from replays at regular intervals
2. Feed to a VLM (Claude Vision / GPT-4V) with structured prompts
3. Extract high-level tactical assessments: positioning quality, vision control, objective prioritization
4. Compare VLM assessments against match outcomes
5. Validate whether VLM-generated insights correlate with winning

**Why it's cutting-edge**: This is genuinely novel research territory. No existing papers do this comprehensively for LoL.

### Avenue E: Automated Event Timeline from VODs
**Feasibility**: MEDIUM-HIGH | **Impressiveness**: HIGH

Create a system that watches a full game VOD and produces a structured event timeline:
1. Detect key events: kills, objectives, tower falls, teamfights
2. Timestamp each event with game time
3. Extract context: which champions involved, gold state, map position
4. Produce a narrative summary of the game
5. Compare with Riot API event data for validation

**Why it connects**: Directly extends the reference paper's sequence-of-events analysis but extracts from video rather than API, capturing events the API might miss.

### Avenue F: Hybrid CV + API Approach (Recommended)
**Feasibility**: HIGH | **Impressiveness**: VERY HIGH

Combine the strengths of both data sources:
1. Use pyLoL/CV for spatial data (positions, movements, visual events)
2. Use Riot API for structured data (gold, XP, items, match outcomes)
3. Fuse both into a rich event timeline
4. Apply ML to identify performance-predictive patterns
5. Show that CV-derived features add predictive value beyond API data alone

**Why this is the winner**: It demonstrates you understand both data sources, shows technical depth in CV, and produces a compelling comparison story. The report can directly argue that CV provides insights the API cannot.

---

## 6. Relevant Datasets & Benchmarks

| Dataset | Domain | Contents |
|---------|--------|----------|
| DeepLeague labeled images | LoL minimap | 100k+ labeled champion positions |
| Roboflow LoL Minimap | LoL minimap | Community-labeled detection dataset |
| ESTA | CS:GO | 1,558 pro matches with trajectories/actions |
| CEPAV | CS:GO | 3,000+ matches with physiological data |
| CS-lol | LoL broadcasts | Pro matches with scene annotations |
| SC2EGSet | StarCraft II | Replays with game-state annotations |

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

### Methodology
13. YOLOv10 (Tsinghua University) - Latest real-time object detection
14. RT-DETR (CVPR 2024) - Transformer-based real-time detection
15. Segment Anything Model (Meta) - Zero-shot segmentation
16. VideoMAE - Self-supervised video pre-training
17. TimeSformer (Meta) - Video understanding transformer
18. PaddleOCR (Baidu) - Lightweight OCR toolkit

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

## 9. Suggested SMART Research Questions

Based on the landscape, here are potential research questions for your project:

1. "To what extent can computer vision techniques extract accurate game state information (champion positions, objectives, kills) from League of Legends replay footage, compared to Riot API data?"

2. "Can minimap tracking data derived from computer vision predict match outcomes with comparable accuracy to API-derived features?"

3. "What additional performance insights can be gained from CV-extracted spatial data (movement patterns, positioning) that are not available through the Riot API?"

4. "How effectively can modern object detection models (YOLOv10) detect and classify champion positions on the LoL minimap across different game phases?"

5. "Can a multimodal pipeline combining minimap tracking, OCR, and event detection produce a more comprehensive game analysis than any single technique alone?"
