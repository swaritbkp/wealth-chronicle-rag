"""
tests/test_telemetry.py — Unit tests for Dual-Mode Qdrant Storage Fallback and SQLite Telemetry.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from engine import (
    get_qdrant_client,
    get_storage_mode,
)
from telemetry import (
    fetch_recent_audit_logs,
    get_audit_summary_stats,
    init_telemetry_db,
    log_query_audit,
)


class TestDualModeQdrantStorageFallback:
    """Test automatic fallback from unreachable cloud cluster to local disk storage."""

    def test_fallback_when_cloud_url_unreachable(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            local_path = str(Path(tmpdir) / "qdrant_test_storage")

            # Attempt connection to an unresolvable hostname
            client, mode = get_qdrant_client(
                url="https://unreachable-cluster-id.region.aws.cloud.qdrant.io:6333",
                api_key="mock_key",
                local_path=local_path,
                timeout=1.0,
            )

            assert mode == "LOCAL_DISK"
            assert get_storage_mode() == "LOCAL_DISK"
            assert client is not None
            assert Path(local_path).exists()
            if hasattr(client, "close"):
                client.close()

    def test_fallback_when_url_is_none_or_placeholder(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            local_path = str(Path(tmpdir) / "qdrant_placeholder_storage")

            client, mode = get_qdrant_client(
                url="https://your-cluster-id.region.aws.cloud.qdrant.io:6333",
                api_key=None,
                local_path=local_path,
            )

            assert mode == "LOCAL_DISK"
            assert get_storage_mode() == "LOCAL_DISK"
            assert client is not None
            if hasattr(client, "close"):
                client.close()

    def test_cloud_connection_successful(self) -> None:
        with patch("engine.QdrantClient") as mock_qdrant_cls:
            mock_instance = MagicMock()
            mock_instance.get_collections.return_value = MagicMock()
            mock_qdrant_cls.return_value = mock_instance

            client, mode = get_qdrant_client(
                url="https://live-valid-cluster.qdrant.io:6333",
                api_key="valid_key",
                timeout=5.0,
            )

            assert mode == "CLOUD"
            assert get_storage_mode() == "CLOUD"
            assert client == mock_instance


class TestTelemetrySQLiteAudit:
    """Test SQLite database initialization, audit logging, and summary retrieval."""

    def test_init_and_logging_query_audit(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "test_telemetry.db")

            init_telemetry_db(db_path)
            assert Path(db_path).exists()

            # Log a passed query
            log_query_audit(
                query_text="What are 80C limits for FY 2025-26?",
                storage_mode="CLOUD",
                top_score=0.88,
                gate_status="PASSED",
                latency_ms=145.2,
                chunks_retrieved_count=4,
                db_path=db_path,
            )

            # Log a refused query
            log_query_audit(
                query_text="What is the weather in Delhi?",
                storage_mode="LOCAL_DISK",
                top_score=0.08,
                gate_status="REFUSED",
                latency_ms=42.1,
                chunks_retrieved_count=0,
                db_path=db_path,
            )

            logs = fetch_recent_audit_logs(limit=10, db_path=db_path)
            assert len(logs) == 2

            # Most recent first
            assert logs[0]["query_text"] == "What is the weather in Delhi?"
            assert logs[0]["gate_status"] == "REFUSED"
            assert logs[0]["storage_mode"] == "LOCAL_DISK"

            assert logs[1]["query_text"] == "What are 80C limits for FY 2025-26?"
            assert logs[1]["gate_status"] == "PASSED"
            assert logs[1]["storage_mode"] == "CLOUD"

            stats = get_audit_summary_stats(db_path=db_path)
            assert stats["total_queries"] == 2
            assert stats["passed"] == 1
            assert stats["refused"] == 1
            assert stats["avg_latency_ms"] > 0

    def test_non_blocking_error_handling(self) -> None:
        with patch("sqlite3.connect", side_effect=Exception("Database lock/IO error")):
            # Safe non-blocking execution should not raise exception
            log_query_audit(
                query_text="Test safe fail",
                storage_mode="LOCAL_DISK",
                top_score=0.5,
                gate_status="PASSED",
                latency_ms=10.0,
                chunks_retrieved_count=1,
                db_path="invalid_db_path.db",
            )

            logs = fetch_recent_audit_logs(db_path="invalid_db_path.db")
            assert logs == []

            stats = get_audit_summary_stats(db_path="invalid_db_path.db")
            assert stats["total_queries"] == 0
