"""
drift_monitor.py — Query Embedding Drift Detection using UMAP + HDBSCAN.

Detects semantic drift in incoming user queries by clustering embeddings
and comparing against a baseline. Generates alerts for new/emerging topics.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

try:
    import hdbscan
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    hdbscan = None
    umap = None

from schemas import DriftAlert, DriftCluster, DriftReport, DriftSeverity

logger = logging.getLogger("wealthchronicle.drift")

DEFAULT_DRIFT_DB = "drift_baseline.pkl"
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 3
DEFAULT_UMAP_N_NEIGHBORS = 10
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 5
DEFAULT_HDBSCAN_MIN_SAMPLES = 3
DRIFT_COSINE_THRESHOLD = 0.35  # Max cosine distance to consider same cluster
NEW_CLUSTER_MIN_SIZE = 3  # Min queries to flag as new cluster


class DriftMonitor:
    """Monitors query embeddings for semantic drift using UMAP + HDBSCAN."""

    def __init__(
        self,
        dense_embedding_model: Any,
        baseline_path: str = DEFAULT_DRIFT_DB,
        min_cluster_size: int = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
        umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
        umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
    ):
        if not UMAP_AVAILABLE:
            raise RuntimeError("UMAP/HDBSCAN not installed. Install with: pip install umap-learn hdbscan")

        self.dense_embedding_model = dense_embedding_model
        self.baseline_path = Path(baseline_path)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist

        # State
        self.baseline_clusters: list[DriftCluster] = []
        self.baseline_umap: umap.UMAP | None = None
        self.baseline_hdbscan: hdbscan.HDBSCAN | None = None
        self.alerts: list[DriftAlert] = []

    def _embed_queries(self, queries: list[str]) -> np.ndarray:
        """Generate dense embeddings for a list of queries."""
        embeddings = list(self.dense_embedding_model.embed(queries))
        return np.array([e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings])

    def _reduce_dimensions(self, embeddings: np.ndarray, fit: bool = False) -> np.ndarray:
        """Reduce embeddings to 2D using UMAP."""
        if fit or self.baseline_umap is None:
            self.baseline_umap = umap.UMAP(
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                n_components=2,
                metric="cosine",
                random_state=42,
            )
            return self.baseline_umap.fit_transform(embeddings)
        return self.baseline_umap.transform(embeddings)

    def _cluster_embeddings(self, embeddings_2d: np.ndarray, fit: bool = False) -> np.ndarray:
        """Cluster 2D embeddings using HDBSCAN."""
        if fit or self.baseline_hdbscan is None:
            self.baseline_hdbscan = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            return self.baseline_hdbscan.fit_predict(embeddings_2d)
        return self.baseline_hdbscan.fit_predict(embeddings_2d)

    def _compute_centroid(self, embeddings: np.ndarray, labels: np.ndarray, cluster_id: int) -> np.ndarray:
        """Compute centroid embedding for a cluster."""
        mask = labels == cluster_id
        if not np.any(mask):
            return np.zeros(embeddings.shape[1])
        return embeddings[mask].mean(axis=0)

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two vectors."""
        return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def build_baseline(self, queries: list[str]) -> DriftReport:
        """Build baseline clusters from historical queries."""
        if len(queries) < self.min_cluster_size:
            logger.warning(f"Need at least {self.min_cluster_size} queries for baseline, got {len(queries)}")
            return DriftReport(
                total_queries_analyzed=len(queries),
                num_clusters=0,
                num_noise_queries=len(queries),
                severity=DriftSeverity.NONE,
                recommendation="Insufficient queries for baseline. Collect more query history.",
            )

        logger.info(f"Building drift baseline from {len(queries)} queries...")

        # Embed queries
        embeddings = self._embed_queries(queries)

        # Reduce dimensions
        embeddings_2d = self._reduce_dimensions(embeddings, fit=True)

        # Cluster
        labels = self._cluster_embeddings(embeddings_2d, fit=True)

        # Build cluster objects
        self.baseline_clusters = []
        unique_labels = set(labels) - {-1}  # Exclude noise (-1)

        for cluster_id in sorted(unique_labels):
            mask = labels == cluster_id
            cluster_embeddings = embeddings[mask]
            cluster_queries = [queries[i] for i in np.where(mask)[0]]

            centroid = self._compute_centroid(embeddings, labels, cluster_id)
            distances = [self._cosine_distance(centroid, e) for e in cluster_embeddings]

            cluster = DriftCluster(
                cluster_id=int(cluster_id),
                centroid_embedding=centroid.tolist(),
                query_count=int(mask.sum()),
                sample_queries=cluster_queries[:5],
                avg_distance_to_centroid=float(np.mean(distances)),
                first_seen=datetime.now(timezone.utc),
                last_seen=datetime.now(timezone.utc),
                is_new=False,
            )
            self.baseline_clusters.append(cluster)

        noise_count = int(np.sum(labels == -1))

        # Save baseline
        self.save_baseline()

        report = DriftReport(
            total_queries_analyzed=len(queries),
            num_clusters=len(self.baseline_clusters),
            num_noise_queries=noise_count,
            new_clusters=[],
            severity=DriftSeverity.NONE,
            recommendation="Baseline established successfully.",
            baseline_coverage=1.0 - (noise_count / len(queries)),
        )

        logger.info(f"Baseline built: {len(self.baseline_clusters)} clusters, {noise_count} noise queries")
        return report

    def detect_drift(self, new_queries: list[str]) -> DriftReport:
        """Detect drift in new queries against baseline."""
        if not self.baseline_clusters:
            return DriftReport(
                total_queries_analyzed=len(new_queries),
                num_clusters=0,
                num_noise_queries=len(new_queries),
                severity=DriftSeverity.NONE,
                recommendation="No baseline available. Call build_baseline() first.",
            )

        if not new_queries:
            return DriftReport(
                total_queries_analyzed=0,
                num_clusters=0,
                num_noise_queries=0,
                severity=DriftSeverity.NONE,
                recommendation="No new queries to analyze.",
            )

        logger.info(f"Analyzing {len(new_queries)} new queries for drift...")

        # Embed new queries
        new_embeddings = self._embed_queries(new_queries)

        # Reduce dimensions using fitted UMAP
        new_embeddings_2d = self._reduce_dimensions(new_embeddings, fit=False)

        # Cluster using fitted HDBSCAN (predict mode)
        new_labels = self._cluster_embeddings(new_embeddings_2d, fit=False)

        # Analyze clusters
        new_clusters: list[DriftCluster] = []
        unique_labels = set(new_labels) - {-1}

        for cluster_id in sorted(unique_labels):
            mask = new_labels == cluster_id
            cluster_embeddings = new_embeddings[mask]
            cluster_queries = [new_queries[i] for i in np.where(mask)[0]]

            centroid = self._compute_centroid(new_embeddings, new_labels, cluster_id)

            # Check if this cluster matches any baseline cluster
            min_dist = float("inf")
            for baseline in self.baseline_clusters:
                dist = self._cosine_distance(centroid, np.array(baseline.centroid_embedding))
                if dist < min_dist:
                    min_dist = dist

            is_new = min_dist > DRIFT_COSINE_THRESHOLD
            query_count = int(mask.sum())

            # Only flag as new cluster if it has enough queries
            if is_new and query_count >= NEW_CLUSTER_MIN_SIZE:
                cluster = DriftCluster(
                    cluster_id=int(cluster_id),
                    centroid_embedding=centroid.tolist(),
                    query_count=query_count,
                    sample_queries=cluster_queries[:5],
                    avg_distance_to_centroid=0.0,  # Will compute below
                    first_seen=datetime.now(timezone.utc),
                    last_seen=datetime.now(timezone.utc),
                    is_new=True,
                )
                # Compute avg distance to own centroid
                distances = [self._cosine_distance(centroid, e) for e in cluster_embeddings]
                cluster.avg_distance_to_centroid = float(np.mean(distances))
                new_clusters.append(cluster)

                # Generate alert
                self._generate_alert(cluster, min_dist, cluster_queries[:3])

        noise_count = int(np.sum(new_labels == -1))

        # Determine overall severity
        severity = self._compute_severity(new_clusters)

        # Generate recommendation
        recommendation = self._generate_recommendation(new_clusters, len(new_queries), noise_count)

        report = DriftReport(
            total_queries_analyzed=len(new_queries),
            num_clusters=len(unique_labels),
            num_noise_queries=noise_count,
            new_clusters=new_clusters,
            severity=severity,
            recommendation=recommendation,
            baseline_coverage=1.0 - (noise_count / len(new_queries)),
        )

        logger.info(f"Drift analysis: {len(new_clusters)} new clusters, severity={severity.value}")
        return report

    def _compute_severity(self, new_clusters: list[DriftCluster]) -> DriftSeverity:
        """Compute overall drift severity."""
        if not new_clusters:
            return DriftSeverity.NONE

        total_new_queries = sum(c.query_count for c in new_clusters)
        max_cluster_size = max(c.query_count for c in new_clusters)

        if max_cluster_size >= 20 or total_new_queries >= 50:
            return DriftSeverity.CRITICAL
        elif max_cluster_size >= 10 or total_new_queries >= 20:
            return DriftSeverity.HIGH
        elif max_cluster_size >= 5 or total_new_queries >= 10:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW

    def _generate_recommendation(
        self,
        new_clusters: list[DriftCluster],
        total_queries: int,
        noise_count: int,
    ) -> str:
        """Generate actionable recommendation based on drift findings."""
        if not new_clusters:
            return "No significant drift detected. Query distribution matches baseline."

        total_new = sum(c.query_count for c in new_clusters)
        pct_new = (total_new / total_queries) * 100 if total_queries > 0 else 0

        if len(new_clusters) >= 3:
            return (
                f"Multiple new topics detected ({len(new_clusters)} clusters, {total_new} queries). "
                f"Consider updating golden eval set and retraining retrieval models."
            )
        elif pct_new > 30:
            return (
                f"Significant topic shift ({pct_new:.1f}% of queries are new topics). "
                f"Review new clusters and consider corpus expansion."
            )
        else:
            return (
                f"Emerging topics detected ({len(new_clusters)} new cluster(s), {total_new} queries). "
                f"Monitor for persistence and add to eval set if sustained."
            )

    def _generate_alert(self, cluster: DriftCluster, distance: float, samples: list[str]) -> None:
        """Generate a drift alert for a new cluster."""
        if cluster.query_count >= 10:
            severity = DriftSeverity.HIGH
        elif cluster.query_count >= 5:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW

        alert = DriftAlert(
            severity=severity,
            message=(
                f"New query cluster detected: {cluster.query_count} queries "
                f"(cosine distance to nearest baseline: {distance:.3f}). "
                f"Sample: {'; '.join(samples)}"
            ),
            affected_clusters=[cluster.cluster_id],
            new_query_samples=samples,
        )
        self.alerts.append(alert)
        logger.warning(f"DRIFT ALERT [{severity.value.upper()}]: {alert.message}")

    def get_unacknowledged_alerts(self) -> list[DriftAlert]:
        """Get all unacknowledged alerts."""
        return [a for a in self.alerts if not a.acknowledged]

    def acknowledge_alert(self, alert_id: UUID, acknowledged_by: str) -> bool:
        """Acknowledge a drift alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now(timezone.utc)
                return True
        return False

    def save_baseline(self, path: str | None = None) -> None:
        """Save baseline to disk."""
        save_path = Path(path or self.baseline_path)
        data = {
            "clusters": [c.model_dump() for c in self.baseline_clusters],
            "umap_params": {
                "n_neighbors": self.umap_n_neighbors,
                "min_dist": self.umap_min_dist,
            },
            "hdbscan_params": {
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
            },
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Baseline saved to {save_path}")

    def load_baseline(self, path: str | None = None) -> bool:
        """Load baseline from disk."""
        load_path = Path(path or self.baseline_path)
        if not load_path.exists():
            logger.warning(f"No baseline found at {load_path}")
            return False

        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)

            self.baseline_clusters = [DriftCluster(**c) for c in data.get("clusters", [])]
            logger.info(f"Loaded baseline with {len(self.baseline_clusters)} clusters from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            return False


def run_drift_detection(
    queries: list[str],
    dense_embedding_model: Any,
    baseline_path: str = DEFAULT_DRIFT_DB,
    baseline_queries: list[str] | None = None,
) -> DriftReport:
    """Convenience function to run drift detection end-to-end."""
    monitor = DriftMonitor(dense_embedding_model, baseline_path=baseline_path)

    if baseline_queries:
        monitor.build_baseline(baseline_queries)
    else:
        monitor.load_baseline()

    return monitor.detect_drift(queries)