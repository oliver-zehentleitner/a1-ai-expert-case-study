"""Semantic clustering of SMS messages using embeddings + K-Means."""

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """All outputs of a clustering run."""

    labels: np.ndarray            # cluster index per message
    embeddings: np.ndarray        # raw embedding vectors  (n_texts × dims)
    embeddings_2d: np.ndarray     # UMAP-reduced for plotting  (n_texts × 2)
    n_clusters: int
    silhouette: float
    inertias: list[float] = field(default_factory=list)   # for elbow plot
    cluster_sizes: dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        sizes = ", ".join(f"C{k}:{v}" for k, v in sorted(self.cluster_sizes.items()))
        return f"{self.n_clusters} clusters | silhouette={self.silhouette:.3f} | {sizes}"


class SemanticClusterer:
    """
    Embed SMS texts and group them by semantic similarity.

    The clustering logic is provider-agnostic: it only calls
    :meth:`.EmbeddingProvider.embed` — swap Ollama, OpenAI, or Azure
    by passing a different provider instance.

    K-means is used for clustering.  If ``n_clusters`` is not given,
    the optimal K is selected automatically via silhouette score over
    a configurable range.

    Example::

        from sms_analysis.clustering import SemanticClusterer
        from sms_analysis.clustering.providers import OllamaProvider

        provider = OllamaProvider()
        clusterer = SemanticClusterer(provider)
        result = clusterer.fit(dataset.texts[:500])
        print(result.summary())
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        n_clusters: int | None = None,
    ) -> None:
        self.provider = provider
        self.n_clusters = n_clusters

    def fit(
        self,
        texts: list[str],
        k_range: range = range(3, 16),
    ) -> ClusteringResult:
        """
        Embed ``texts`` and cluster them.

        Args:
            texts: SMS texts to cluster.
            k_range: Range of K values to evaluate when ``n_clusters`` is None.

        Returns:
            :class:`ClusteringResult` with labels, embeddings, and metrics.
        """
        logger.info(
            "Starting semantic clustering of %d texts (provider=%s)",
            len(texts), self.provider.__class__.__name__,
        )
        logger.info("Embedding %d messages — this may take a minute …", len(texts))

        embeddings = np.array(self.provider.embed(texts))
        logger.info("Embeddings done. Running K-Means and UMAP …")

        if self.n_clusters is not None:
            best_k = self.n_clusters
            inertias: list[float] = []
        else:
            safe_k_range = range(k_range.start, min(k_range.stop, len(embeddings)), k_range.step)
            best_k, inertias = self._select_k(embeddings, safe_k_range)

        labels = self._kmeans(embeddings, best_k)
        score = silhouette_score(
            embeddings, labels, sample_size=min(1_000, len(texts)), random_state=42
        )
        embeddings_2d = self._reduce_2d(embeddings)

        result = ClusteringResult(
            labels=labels,
            embeddings=embeddings,
            embeddings_2d=embeddings_2d,
            n_clusters=best_k,
            silhouette=float(score),
            inertias=inertias,
            cluster_sizes={k: int((labels == k).sum()) for k in range(best_k)},
        )
        logger.info("Clustering complete: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _kmeans(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return km.fit_predict(embeddings)

    def _select_k(
        self,
        embeddings: np.ndarray,
        k_range: range,
    ) -> tuple[int, list[float]]:
        """Select K by maximising the silhouette score."""
        inertias: list[float] = []
        silhouettes: list[float] = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            inertias.append(float(km.inertia_))
            score = silhouette_score(
                embeddings, labels,
                sample_size=min(500, len(embeddings)),
                random_state=42,
            )
            silhouettes.append(float(score))
            logger.debug("k=%d  silhouette=%.3f  inertia=%.0f", k, score, km.inertia_)

        best_idx = int(np.argmax(silhouettes))
        best_k = list(k_range)[best_idx]
        logger.info(
            "Auto-selected k=%d (silhouette=%.3f)", best_k, silhouettes[best_idx]
        )
        return best_k, inertias

    def _reduce_2d(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce to 2D for plotting.  Falls back to PCA if umap-learn is absent."""
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1, verbose=False)
            return reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not installed — falling back to PCA for 2D reduction")
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=42).fit_transform(embeddings)
