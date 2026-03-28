# LoL CV Analysis

Computer Vision Analysis of League of Legends Replays for Performance Insight.

**University of Twente — Data Science Q3 (2025-2A)**
Primary topic: CV&IC | Secondary topic: Data Mining

## Overview

This project investigates whether computer vision techniques can extract meaningful performance insights from League of Legends replay footage. Using the [pyLoL replay extractor](https://github.com/league-of-legends-replay-extractor/pyLoL) as a data pipeline foundation, we build an analysis layer that combines minimap champion tracking (YOLOv12), OCR-based HUD extraction, and modern vision-language models to derive spatiotemporal features — then apply machine learning to discover patterns that predict match outcomes.

## Project Structure

```
├── src/lol_cv/             # Main Python package
│   ├── extraction/         # Data extraction (minimap, OCR, API, VLM)
│   ├── features/           # Feature engineering (spatial, temporal)
│   ├── analysis/           # ML models (classifiers, clustering)
│   ├── visualization/      # Charts, heatmaps, dashboards
│   └── utils/              # Shared utilities
├── notebooks/              # Jupyter notebooks for exploration & reporting
├── data/                   # Data directory (raw, processed, models, embeddings)
├── configs/                # YAML configuration files
├── docs/                   # Documentation and reference materials
│   └── references/         # Course materials, rubric, reference paper
├── tests/                  # Unit tests
├── report.md               # Project reflection report
└── pyproject.toml          # Project config & dependencies (managed by uv)
```

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone <your-repo-url>
cd project

# Install dependencies (including pyLoL from GitHub)
uv sync

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your Riot API key and Gemini API key
```

## Pipeline

1. **Extract** — Run pyLoL to capture minimap videos from replays, then detect champion positions with YOLOv12
2. **Enrich** — Extract HUD data (gold, KDA, items) via OCR; supplement with Riot API data
3. **Engineer** — Compute spatial features (movement, grouping, zone control) and temporal features (event sequences, phase patterns)
4. **Analyze** — Train and compare ML classifiers; cluster game states using multimodal embeddings
5. **Visualize** — Generate heatmaps, trajectory plots, and performance dashboards

## Authors

- Wong Kang
- Zachary Muk Chen Eu

Supervised by Dr. G.W.J. (Guido) Bruinsma — University of Twente
