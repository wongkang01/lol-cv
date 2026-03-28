"""
Unsupervised clustering on multimodal embeddings.

Uses Gemini Embedding vectors to discover recurring game state patterns
without labeled data. Supports K-Means, DBSCAN, and Agglomerative clustering.

Workflow:
    1. Generate embeddings for game frames via VlmAnalyzer.embed_frame()
    2. Optionally reduce dimensionality (PCA / UMAP)
    3. Cluster embeddings
    4. Correlate clusters with match outcomes (chi-squared test)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency

from lol_cv.utils import setup_logger

logger = setup_logger("lol_cv.analysis.clustering")


class GameStateClustering:
    """Cluster game states using multimodal embeddings."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.labels_ = None
        self.model_ = None
        self.reduced_embeddings_ = None

    def reduce_dimensions(
        self, embeddings: np.ndarray, n_components: int = 50
    ) -> np.ndarray:
        """Reduce embedding dimensionality with PCA before clustering.

        Gemini embeddings are 3072-dim, which is expensive for clustering.
        PCA to 50-100 dimensions preserves most variance.

        Args:
            embeddings: Array of shape (n_samples, 3072).
            n_components: Target dimensionality.

        Returns:
            Reduced array of shape (n_samples, n_components).
        """
        n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_components, random_state=self.random_state)
        reduced = pca.fit_transform(embeddings)
        explained = sum(pca.explained_variance_ratio_)
        logger.info(
            "PCA: %d → %d dimensions (%.1f%% variance retained)",
            embeddings.shape[1], n_components, explained * 100,
        )
        self.reduced_embeddings_ = reduced
        return reduced

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        method: str = "kmeans",
        n_clusters: int = 8,
        reduce_first: bool = True,
        n_components: int = 50,
        **kwargs,
    ) -> np.ndarray:
        """Cluster game state embeddings.

        Args:
            embeddings: Array of shape (n_samples, dim).
            method: 'kmeans', 'dbscan', or 'hierarchical'.
            n_clusters: Number of clusters (ignored for DBSCAN).
            reduce_first: Whether to apply PCA before clustering.
            n_components: PCA target dimensions.

        Returns:
            Cluster labels array of shape (n_samples,).
        """
        data = self.reduce_dimensions(embeddings, n_components) if reduce_first else embeddings

        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10, **kwargs)
        elif method == "dbscan":
            eps = kwargs.pop("eps", 0.5)
            min_samples = kwargs.pop("min_samples", 5)
            model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        elif method == "hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: kmeans, dbscan, hierarchical")

        self.labels_ = model.fit_predict(data)
        self.model_ = model

        n_found = len(set(self.labels_) - {-1})
        logger.info("Clustering (%s): found %d clusters from %d samples", method, n_found, len(data))

        # Compute silhouette score if more than 1 cluster
        if n_found > 1:
            sil = silhouette_score(data, self.labels_)
            logger.info("Silhouette score: %.3f", sil)

        return self.labels_

    def cluster_outcome_correlation(
        self, clusters: np.ndarray, outcomes: np.ndarray
    ) -> dict:
        """Test if clusters correlate with win/loss outcomes.

        Uses chi-squared test on the contingency table of
        cluster labels vs. match outcomes.

        Args:
            clusters: Cluster label per sample.
            outcomes: Binary outcome per sample (1=win, 0=loss).

        Returns:
            Dict with chi2 statistic, p_value, contingency table,
            and per-cluster win rates.
        """
        ct = pd.crosstab(
            pd.Series(clusters, name="cluster"),
            pd.Series(outcomes, name="outcome"),
        )
        chi2, p_value, dof, expected = chi2_contingency(ct)

        # Per-cluster win rate
        win_rates = {}
        for cluster_id in sorted(set(clusters)):
            mask = clusters == cluster_id
            win_rates[int(cluster_id)] = float(outcomes[mask].mean())

        logger.info(
            "Chi-squared test: χ²=%.2f, p=%.4f, dof=%d", chi2, p_value, dof
        )

        return {
            "chi2": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof),
            "contingency_table": ct,
            "per_cluster_win_rate": win_rates,
            "significant": p_value < 0.05,
        }

    def find_optimal_k(
        self, embeddings: np.ndarray, k_range: range = range(2, 16),
        reduce_first: bool = True, n_components: int = 50,
    ) -> dict:
        """Find optimal number of clusters using silhouette score.

        Args:
            embeddings: Embedding array.
            k_range: Range of k values to try.

        Returns:
            Dict with best_k, scores per k, and inertias per k.
        """
        data = self.reduce_dimensions(embeddings, n_components) if reduce_first else embeddings

        scores = {}
        inertias = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(data)
            scores[k] = silhouette_score(data, labels)
            inertias[k] = km.inertia_
            logger.info("k=%d: silhouette=%.3f, inertia=%.0f", k, scores[k], inertias[k])

        best_k = max(scores, key=scores.get)
        logger.info("Best k=%d (silhouette=%.3f)", best_k, scores[best_k])

        return {
            "best_k": best_k,
            "silhouette_scores": scores,
            "inertias": inertias,
        }
