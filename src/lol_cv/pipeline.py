"""
End-to-end pipeline orchestrator for the LoL CV analysis project.

Chains all six stages into a single entry point:
    1. Detect  -- Track champion positions on the minimap (YOLO)
    2. Extract -- OCR the spectator HUD for gold, kills, timers
    3. Analyse -- VLM tactical analysis of key moments (Gemini)
    4. Engineer -- Compute spatial + temporal features
    5. Predict -- ML classifiers to predict match outcomes
    6. Embed & Cluster -- Game-state embeddings for pattern discovery

Each ``run_*`` method is independently callable, and ``run_full`` chains
them all together for batch processing of multiple matches.

Intermediate results are persisted to ``data/processed/{match_id}/``
as CSV and JSON so individual stages can be re-run without repeating
expensive upstream work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lol_cv.utils import load_config, setup_logger, get_data_dir, ensure_dir

logger = setup_logger("lol_cv.pipeline")


class Pipeline:
    """Orchestrate all stages of the LoL CV analysis pipeline.

    Components are initialised lazily on first use to avoid importing
    heavy dependencies (ultralytics, paddleocr, google-genai) until
    they are actually needed.
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Args:
            config_path: Path to a YAML configuration file.
                         Defaults to ``configs/default.yaml``.
        """
        self.config = load_config(config_path)
        self._tracker: "MinimapTracker | None" = None
        self._ocr: "HudExtractor | None" = None
        self._vlm: "VlmAnalyzer | None" = None
        self._spatial: "SpatialFeatures | None" = None
        self._temporal: "TemporalFeatures | None" = None
        self._predictor: "WinPredictor | None" = None
        self._clustering: "GameStateClustering | None" = None

    # ── Lazy component accessors ────────────────────────────────────

    @property
    def tracker(self) -> "MinimapTracker":
        """Lazy-initialise the minimap tracker from config."""
        if self._tracker is None:
            from lol_cv.extraction.minimap import MinimapTracker

            cfg = self.config.get("extraction", {}).get("minimap", {})
            self._tracker = MinimapTracker(
                confidence=cfg.get("confidence", 0.4),
                overlap=cfg.get("overlap", 0.3),
            )
        return self._tracker

    @property
    def ocr(self) -> "HudExtractor":
        """Lazy-initialise the HUD OCR extractor from config."""
        if self._ocr is None:
            from lol_cv.extraction.ocr import HudExtractor

            cfg = self.config.get("extraction", {}).get("ocr", {})
            self._ocr = HudExtractor(engine=cfg.get("engine", "paddleocr"))
        return self._ocr

    @property
    def vlm(self) -> "VlmAnalyzer":
        """Lazy-initialise the VLM analyser from config."""
        if self._vlm is None:
            from lol_cv.extraction.vlm import VlmAnalyzer

            vlm_cfg = self.config.get("extraction", {}).get("vlm", {})
            embed_cfg = self.config.get("embedding", {})
            self._vlm = VlmAnalyzer(
                model=vlm_cfg.get("model", "gemini-3-flash"),
                embed_model=embed_cfg.get("model", "gemini-embedding-002"),
            )
        return self._vlm

    @property
    def spatial(self) -> "SpatialFeatures":
        """Lazy-initialise the spatial feature engineer."""
        if self._spatial is None:
            from lol_cv.features.spatial import SpatialFeatures

            cfg = self.config.get("features", {}).get("spatial", {})
            threshold = cfg.get("grouping_threshold", 200)
            # Config value is in pixels (512-grid); convert to normalised [0,1]
            self._spatial = SpatialFeatures(
                grouping_threshold=threshold / 512 if threshold > 1 else threshold,
            )
        return self._spatial

    @property
    def temporal(self) -> "TemporalFeatures":
        """Lazy-initialise the temporal feature engineer."""
        if self._temporal is None:
            from lol_cv.features.temporal import TemporalFeatures

            cfg = self.config.get("features", {}).get("temporal", {})
            self._temporal = TemporalFeatures(
                early_game_end=cfg.get("early_game_end", 900),
                mid_game_end=cfg.get("mid_game_end", 1500),
                event_types=cfg.get("event_types"),
            )
        return self._temporal

    @property
    def predictor(self) -> "WinPredictor":
        """Lazy-initialise the ML win-predictor."""
        if self._predictor is None:
            from lol_cv.analysis.classifiers import WinPredictor

            cfg = self.config.get("analysis", {})
            self._predictor = WinPredictor(
                models=cfg.get("models"),
                cv_folds=cfg.get("cv_folds", 5),
                test_size=cfg.get("test_size", 0.2),
                random_state=cfg.get("random_state", 42),
            )
        return self._predictor

    @property
    def clustering(self) -> "GameStateClustering":
        """Lazy-initialise the game-state clustering module."""
        if self._clustering is None:
            from lol_cv.analysis.clustering import GameStateClustering

            cfg = self.config.get("analysis", {})
            self._clustering = GameStateClustering(
                random_state=cfg.get("random_state", 42),
            )
        return self._clustering

    # ── Persistence helpers ─────────────────────────────────────────

    def _match_dir(self, match_id: str) -> Path:
        """Return (and create) ``data/processed/{match_id}/``."""
        return ensure_dir(get_data_dir("processed") / match_id)

    def _save_csv(self, df: pd.DataFrame, match_id: str, name: str) -> Path:
        """Persist a DataFrame as CSV under the match directory."""
        path = self._match_dir(match_id) / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved %s (%d rows) -> %s", name, len(df), path)
        return path

    def _save_json(self, data: dict | list, match_id: str, name: str) -> Path:
        """Persist a dict/list as JSON under the match directory."""
        path = self._match_dir(match_id) / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved %s -> %s", name, path)
        return path

    # ── Stage 1: Detection ──────────────────────────────────────────

    def run_detection(
        self,
        video_path: str,
        minimap_region: tuple[int, int, int, int] | None = None,
        match_id: str | None = None,
    ) -> pd.DataFrame:
        """Stage 1: Run minimap champion detection on a video.

        When *minimap_region* is provided the full-frame video is cropped
        to that region before detection.  Otherwise the video is assumed
        to already contain only the minimap.

        Args:
            video_path: Path to the game video file.
            minimap_region: ``(x1, y1, x2, y2)`` pixel coordinates of
                the minimap in the full frame.  ``None`` if the video is
                already a minimap-only recording.
            match_id: Optional identifier used to persist the result.

        Returns:
            DataFrame with columns ``[timestamp, champion, x, y, confidence]``.
        """
        cfg = self.config.get("extraction", {}).get("minimap", {})
        fps = cfg.get("fps", 1)

        logger.info("Stage 1 [Detect] — %s (region=%s)", video_path, minimap_region)

        if minimap_region is not None:
            positions = self.tracker.extract_from_full_frame(
                video_path, minimap_region=minimap_region, fps=fps,
            )
        else:
            positions = self.tracker.extract_positions(video_path, fps=fps)

        df = self.tracker.positions_to_dataframe(positions)

        if match_id:
            self._save_csv(df, match_id, "positions")

        return df

    # ── Stage 2: OCR ────────────────────────────────────────────────

    def run_ocr(
        self,
        video_path: str,
        match_id: str | None = None,
    ) -> pd.DataFrame:
        """Stage 2: Extract HUD data from a spectator-mode video.

        Runs PaddleOCR on sampled frames to extract game timer, kill
        score, and gold values.

        Args:
            video_path: Path to the spectator-mode video.
            match_id: Optional identifier used to persist the result.

        Returns:
            DataFrame with columns derived from
            ``HudExtractor.extract_all`` — one row per sampled frame.
        """
        cfg = self.config.get("extraction", {}).get("minimap", {})
        fps = cfg.get("fps", 1)

        logger.info("Stage 2 [OCR] — %s", video_path)

        raw_results = self.ocr.extract_from_video(video_path, fps=fps)

        # Flatten the nested dicts into a tabular format.
        rows = []
        for entry in raw_results:
            row = {
                "video_time": entry.get("video_time"),
                "game_time": entry.get("game_time"),
                "game_time_seconds": entry.get("game_time_seconds"),
            }
            scoreboard = entry.get("scoreboard", {})
            row["blue_kills"] = scoreboard.get("blue_kills")
            row["red_kills"] = scoreboard.get("red_kills")
            row["blue_gold"] = scoreboard.get("blue_gold")
            row["red_gold"] = scoreboard.get("red_gold")
            rows.append(row)

        df = pd.DataFrame(rows)

        if match_id:
            self._save_csv(df, match_id, "ocr")

        return df

    # ── Stage 3: VLM Analysis ──────────────────────────────────────

    def run_vlm_analysis(
        self,
        frame_paths: list[str],
        ocr_context: dict | None = None,
        match_id: str | None = None,
    ) -> list[dict]:
        """Stage 3: VLM tactical analysis on key-moment frames.

        If *ocr_context* is supplied, the analysis prompt is enriched
        with the OCR-extracted scoreboard data to improve accuracy.

        Args:
            frame_paths: Paths to screenshot images of key moments.
            ocr_context: Optional dict of OCR data to include in the
                prompt (e.g. ``{"gold_diff": 1200, "kills": "5-3"}``).
            match_id: Optional identifier used to persist the result.

        Returns:
            List of analysis dicts (one per frame).  Each element is the
            raw text response from the VLM — typically JSON-formatted.
        """
        logger.info("Stage 3 [VLM] — analysing %d frames", len(frame_paths))

        prompt = None
        if ocr_context:
            context_str = json.dumps(ocr_context, indent=2)
            prompt = (
                f"Additional context from OCR extraction:\n{context_str}\n\n"
                f"{self.vlm.__class__.__dict__.get('_default_prompt', '')}"
            )

        responses = self.vlm.analyze_frames_batch(frame_paths, prompt=prompt)

        results = []
        for path, response in zip(frame_paths, responses):
            results.append({
                "frame_path": path,
                "analysis": response,
            })

        if match_id:
            self._save_json(results, match_id, "vlm_analysis")

        return results

    # ── Stage 4: Feature Engineering ────────────────────────────────

    def engineer_features(
        self,
        positions_df: pd.DataFrame,
        ocr_df: pd.DataFrame,
        blue_team: list[str],
        red_team: list[str],
        match_id: str | None = None,
    ) -> dict:
        """Stage 4: Compute spatial and temporal features for one match.

        Combines position-based spatial features (zone transitions,
        grouping, objective proximity) with OCR-derived temporal
        features (gold trends, kill bursts).

        Args:
            positions_df: Output from ``run_detection``.
            ocr_df: Output from ``run_ocr``.
            blue_team: List of 5 blue-side champion names.
            red_team: List of 5 red-side champion names.
            match_id: Optional identifier used to persist the result.

        Returns:
            Flat dict of feature name -> value, suitable as one row in
            an ML feature matrix.
        """
        logger.info("Stage 4 [Features] — computing spatial + temporal features")

        features: dict = {}

        # Spatial features from minimap positions
        if not positions_df.empty:
            spatial_feats = self.spatial.compute_all(positions_df, blue_team, red_team)
            features.update(spatial_feats)

        # Temporal features from OCR data
        # Build a lightweight events DataFrame from the OCR scoreboard to
        # feed into TemporalFeatures.  Detected kill-count changes become
        # "kill" events; gold swings become a continuous feature.
        events_df = self._ocr_to_events(ocr_df)
        if not events_df.empty:
            temporal_feats = self.temporal.compute_all(
                events_df,
                positions=positions_df if not positions_df.empty else None,
                blue_team=blue_team,
                red_team=red_team,
            )
            features.update(temporal_feats)

        # OCR-derived aggregate features
        ocr_feats = self._ocr_aggregate_features(ocr_df)
        features.update(ocr_feats)

        if match_id:
            self._save_json(features, match_id, "features")

        logger.info("Computed %d total features", len(features))
        return features

    def _ocr_to_events(self, ocr_df: pd.DataFrame) -> pd.DataFrame:
        """Convert OCR scoreboard data into an event-style DataFrame.

        Detects frame-to-frame kill-count increases and emits one ``kill``
        event per increment at the corresponding timestamp.

        Returns:
            DataFrame with columns ``[timestamp, event_type, team]``.
        """
        if ocr_df.empty or "game_time_seconds" not in ocr_df.columns:
            return pd.DataFrame(columns=["timestamp", "event_type", "team"])

        df = ocr_df.dropna(subset=["game_time_seconds"]).sort_values("game_time_seconds").copy()
        events: list[dict] = []

        prev_blue_kills = 0
        prev_red_kills = 0

        for _, row in df.iterrows():
            ts = row["game_time_seconds"]
            bk = row.get("blue_kills")
            rk = row.get("red_kills")

            if bk is not None and bk > prev_blue_kills:
                for _ in range(int(bk - prev_blue_kills)):
                    events.append({"timestamp": ts, "event_type": "kill", "team": "blue"})
                prev_blue_kills = int(bk)

            if rk is not None and rk > prev_red_kills:
                for _ in range(int(rk - prev_red_kills)):
                    events.append({"timestamp": ts, "event_type": "kill", "team": "red"})
                prev_red_kills = int(rk)

        return pd.DataFrame(events)

    def _ocr_aggregate_features(self, ocr_df: pd.DataFrame) -> dict:
        """Derive aggregate features from OCR scoreboard time-series.

        Captures gold difference trajectory, total kills, and game
        duration.
        """
        features: dict = {}

        if ocr_df.empty:
            return features

        # Game duration (max observed game time)
        if "game_time_seconds" in ocr_df.columns:
            valid_times = ocr_df["game_time_seconds"].dropna()
            if not valid_times.empty:
                features["game_duration_seconds"] = float(valid_times.max())

        # Final kill score
        for col in ("blue_kills", "red_kills"):
            if col in ocr_df.columns:
                valid = ocr_df[col].dropna()
                if not valid.empty:
                    features[f"final_{col}"] = float(valid.iloc[-1])

        # Gold difference trajectory
        if "blue_gold" in ocr_df.columns and "red_gold" in ocr_df.columns:
            gold_df = ocr_df[["blue_gold", "red_gold"]].dropna()
            if not gold_df.empty:
                gold_diff = gold_df["blue_gold"] - gold_df["red_gold"]
                features["final_gold_diff"] = float(gold_diff.iloc[-1])
                features["max_gold_lead_blue"] = float(gold_diff.max())
                features["max_gold_lead_red"] = float((-gold_diff).max())
                features["mean_gold_diff"] = float(gold_diff.mean())

        return features

    # ── Key Moment Detection ────────────────────────────────────────

    def identify_key_moments(self, ocr_df: pd.DataFrame) -> list[dict]:
        """Identify key moments from OCR scoreboard data.

        Detects:
            - **Gold swings**: >1000 g change within a 60-second window.
            - **Kill bursts**: 3+ kills (either team) within 30 seconds.
            - **Objective events**: Large, sudden gold jumps that likely
              correspond to dragon / baron / tower takes.

        Args:
            ocr_df: DataFrame from ``run_ocr`` with columns
                ``[game_time_seconds, blue_kills, red_kills,
                blue_gold, red_gold]``.

        Returns:
            Sorted list of ``{timestamp, type, description}`` dicts.
        """
        moments: list[dict] = []

        if ocr_df.empty or "game_time_seconds" not in ocr_df.columns:
            return moments

        df = ocr_df.dropna(subset=["game_time_seconds"]).sort_values(
            "game_time_seconds"
        ).reset_index(drop=True)

        # --- Gold swings (>1000 g change in 60 s) ---
        if "blue_gold" in df.columns and "red_gold" in df.columns:
            gold_df = df[["game_time_seconds", "blue_gold", "red_gold"]].dropna()
            if len(gold_df) >= 2:
                gold_df = gold_df.reset_index(drop=True)
                gold_diff = gold_df["blue_gold"] - gold_df["red_gold"]

                for i in range(len(gold_df)):
                    t_i = gold_df["game_time_seconds"].iloc[i]
                    window = gold_df[
                        (gold_df["game_time_seconds"] >= t_i)
                        & (gold_df["game_time_seconds"] <= t_i + 60)
                    ]
                    if len(window) < 2:
                        continue
                    idx_start = window.index[0]
                    idx_end = window.index[-1]
                    swing = abs(gold_diff.loc[idx_end] - gold_diff.loc[idx_start])
                    if swing > 1000:
                        t_end = gold_df["game_time_seconds"].loc[idx_end]
                        leader = "blue" if (gold_diff.loc[idx_end] - gold_diff.loc[idx_start]) > 0 else "red"
                        moments.append({
                            "timestamp": float(t_i),
                            "type": "gold_swing",
                            "description": (
                                f"Gold swing of {swing:.0f}g in favour of "
                                f"{leader} between {t_i:.0f}s and {t_end:.0f}s"
                            ),
                        })

        # --- Kill bursts (3+ kills in 30 s) ---
        for side in ("blue", "red"):
            col = f"{side}_kills"
            if col not in df.columns:
                continue
            kill_df = df[["game_time_seconds", col]].dropna().reset_index(drop=True)
            if len(kill_df) < 2:
                continue

            for i in range(len(kill_df)):
                t_i = kill_df["game_time_seconds"].iloc[i]
                window = kill_df[
                    (kill_df["game_time_seconds"] >= t_i)
                    & (kill_df["game_time_seconds"] <= t_i + 30)
                ]
                if len(window) < 2:
                    continue
                kills_start = window[col].iloc[0]
                kills_end = window[col].iloc[-1]
                if kills_start is None or kills_end is None:
                    continue
                kill_burst = kills_end - kills_start
                if kill_burst >= 3:
                    moments.append({
                        "timestamp": float(t_i),
                        "type": "kill_burst",
                        "description": (
                            f"{side.title()} team scored {int(kill_burst)} kills "
                            f"in 30 s starting at {t_i:.0f}s"
                        ),
                    })

        # Deduplicate moments that overlap in time (keep earliest per type).
        moments = self._deduplicate_moments(moments)
        moments.sort(key=lambda m: m["timestamp"])

        logger.info("Identified %d key moments", len(moments))
        return moments

    @staticmethod
    def _deduplicate_moments(moments: list[dict], merge_window: float = 30.0) -> list[dict]:
        """Merge moments of the same type that fall within *merge_window* seconds."""
        if not moments:
            return moments

        grouped: dict[str, list[dict]] = {}
        for m in moments:
            grouped.setdefault(m["type"], []).append(m)

        deduped: list[dict] = []
        for _type, items in grouped.items():
            items.sort(key=lambda m: m["timestamp"])
            last_ts = -float("inf")
            for item in items:
                if item["timestamp"] - last_ts > merge_window:
                    deduped.append(item)
                    last_ts = item["timestamp"]

        return deduped

    # ── Stage 5: Prediction ─────────────────────────────────────────

    def build_feature_matrix(
        self,
        match_features: list[dict],
        outcomes: list[int],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Combine per-match feature dicts into an ML-ready dataset.

        Missing features are filled with ``NaN`` (sklearn imputers or
        tree-based models handle this gracefully).

        Args:
            match_features: List of feature dicts, one per match.
            outcomes: Parallel list of binary outcomes
                (1 = blue win, 0 = red win).

        Returns:
            ``(X, y)`` tuple where *X* is a features DataFrame and *y*
            is a Series of outcomes.
        """
        X = pd.DataFrame(match_features)
        y = pd.Series(outcomes, name="outcome")

        # Drop columns that are entirely NaN
        X = X.dropna(axis=1, how="all")
        # Fill remaining NaN with 0 (safe default for count / ratio features)
        X = X.fillna(0.0)

        logger.info(
            "Built feature matrix: %d matches x %d features", X.shape[0], X.shape[1]
        )
        return X, y

    def run_prediction(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """Stage 5: Train classifiers and evaluate via cross-validation.

        Args:
            X: Feature matrix (n_matches, n_features).
            y: Binary target (1 = blue win, 0 = red win).

        Returns:
            Dict of ``{model_name: {accuracy, precision, recall, f1, roc_auc}}``.
        """
        logger.info("Stage 5 [Predict] — training %d models on %d matches",
                     len(self.predictor.models), len(y))

        results = self.predictor.train_and_evaluate(X, y)

        logger.info("Best model: %s (AUC=%.3f)",
                     max(results, key=lambda k: results[k]["roc_auc"]),
                     max(r["roc_auc"] for r in results.values()))
        return results

    # ── Stage 6: Clustering ─────────────────────────────────────────

    def run_clustering(
        self,
        embeddings: np.ndarray,
        outcomes: np.ndarray | None = None,
    ) -> dict:
        """Stage 6: Cluster game-state embeddings and analyse results.

        Args:
            embeddings: Array of shape ``(n_samples, dim)`` — typically
                3072-dim Gemini embeddings from ``VlmAnalyzer.embed_frame``.
            outcomes: Optional binary outcome per sample for correlation
                analysis.

        Returns:
            Dict with keys ``labels``, ``n_clusters``, and optionally
            ``outcome_correlation`` if *outcomes* is provided.
        """
        embed_cfg = self.config.get("embedding", {}).get("clustering", {})
        method = embed_cfg.get("method", "kmeans")
        n_clusters = embed_cfg.get("n_clusters", 8)

        logger.info("Stage 6 [Cluster] — %s with k=%d on %d samples",
                     method, n_clusters, len(embeddings))

        labels = self.clustering.cluster_embeddings(
            embeddings, method=method, n_clusters=n_clusters,
        )

        result: dict = {
            "labels": labels.tolist(),
            "n_clusters": len(set(labels) - {-1}),
        }

        if outcomes is not None:
            correlation = self.clustering.cluster_outcome_correlation(labels, outcomes)
            result["outcome_correlation"] = {
                "chi2": correlation["chi2"],
                "p_value": correlation["p_value"],
                "significant": correlation["significant"],
                "per_cluster_win_rate": correlation["per_cluster_win_rate"],
            }

        return result

    # ── Full Pipeline ───────────────────────────────────────────────

    def run_full(self, matches: list[dict]) -> dict:
        """Run the complete pipeline on a list of matches.

        Processes each match through stages 1-4 independently, then
        aggregates results for stages 5-6.  Errors in individual
        matches are logged and skipped so the remaining matches can
        still be processed.

        Args:
            matches: List of match descriptors, each containing::

                {
                    "video_path": str,
                    "match_id": str,
                    "blue_team": ["Ahri", "LeeSin", ...],
                    "red_team": ["Jinx", "Thresh", ...],
                    "winner": "blue" or "red",
                    "minimap_region": (x1, y1, x2, y2),  # optional
                }

        Returns:
            Dict with keys ``per_match`` (list of per-match results),
            ``prediction`` (Stage 5 results or ``None``),
            ``clustering`` (Stage 6 results or ``None``).
        """
        logger.info("Starting full pipeline for %d matches", len(matches))

        all_features: list[dict] = []
        all_outcomes: list[int] = []
        per_match_results: list[dict] = []

        for idx, match in enumerate(matches):
            match_id = match.get("match_id", f"match_{idx}")
            video_path = match["video_path"]
            blue_team = match["blue_team"]
            red_team = match["red_team"]
            winner = match.get("winner")
            minimap_region = match.get("minimap_region")

            logger.info(
                "Processing match %d/%d [%s]", idx + 1, len(matches), match_id
            )

            result: dict = {"match_id": match_id, "error": None}

            try:
                # Stage 1: Detection
                positions_df = self.run_detection(
                    video_path,
                    minimap_region=minimap_region,
                    match_id=match_id,
                )
                result["n_positions"] = len(positions_df)

                # Stage 2: OCR
                ocr_df = self.run_ocr(video_path, match_id=match_id)
                result["n_ocr_frames"] = len(ocr_df)

                # Key moments
                key_moments = self.identify_key_moments(ocr_df)
                result["key_moments"] = key_moments
                self._save_json(key_moments, match_id, "key_moments")

                # Stage 4: Feature engineering
                features = self.engineer_features(
                    positions_df, ocr_df, blue_team, red_team,
                    match_id=match_id,
                )
                result["n_features"] = len(features)
                all_features.append(features)

                if winner is not None:
                    outcome = 1 if winner == "blue" else 0
                    all_outcomes.append(outcome)

            except Exception as exc:
                logger.error("Match %s failed: %s", match_id, exc, exc_info=True)
                result["error"] = str(exc)

            per_match_results.append(result)

        # --- Aggregate stages ---
        pipeline_result: dict = {
            "per_match": per_match_results,
            "prediction": None,
            "clustering": None,
        }

        # Stage 5: Prediction (requires at least 10 matches with labels)
        if len(all_features) >= 10 and len(all_outcomes) == len(all_features):
            try:
                X, y = self.build_feature_matrix(all_features, all_outcomes)
                pipeline_result["prediction"] = self.run_prediction(X, y)
            except Exception as exc:
                logger.error("Prediction stage failed: %s", exc, exc_info=True)
        else:
            logger.warning(
                "Skipping prediction — need >= 10 labelled matches, got %d features / %d labels",
                len(all_features), len(all_outcomes),
            )

        logger.info("Pipeline complete — %d/%d matches succeeded",
                     sum(1 for r in per_match_results if r["error"] is None),
                     len(matches))

        return pipeline_result
