"""
telemetry.py — SQLite-backed Local Retrieval Audit Trail & Telemetry Logging.
Provides persistent audit logging for query latency, refusal gating, matching scores, and storage mode.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("wealthchronicle.telemetry")
DEFAULT_DB_PATH = "telemetry.db"


def init_telemetry_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite telemetry database and create table if not existing."""
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS query_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    query_text TEXT,
                    storage_mode TEXT,
                    top_score REAL,
                    gate_status TEXT,
                    latency_ms REAL,
                    chunks_retrieved_count INTEGER
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON query_telemetry (timestamp);
                """
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to initialize telemetry database at {db_path}: {e}")


def log_query_audit(
    query_text: str,
    storage_mode: str,
    top_score: float,
    gate_status: str,
    latency_ms: float,
    chunks_retrieved_count: int,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Safely log a query execution transaction to SQLite audit database."""
    try:
        init_telemetry_db(db_path)
        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_telemetry (
                    timestamp, query_text, storage_mode, top_score, gate_status, latency_ms, chunks_retrieved_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    utc_now,
                    query_text,
                    storage_mode,
                    float(top_score),
                    gate_status,
                    float(latency_ms),
                    int(chunks_retrieved_count),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Non-blocking telemetry log failure: {e}")


def fetch_recent_audit_logs(
    limit: int = 50,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Retrieve the most recent audit records from SQLite."""
    if not Path(db_path).exists():
        return []

    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, query_text, storage_mode, top_score, gate_status, latency_ms, chunks_retrieved_count
                FROM query_telemetry
                ORDER BY id DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.warning(f"Failed to fetch telemetry audit logs: {e}")
        return []


def get_audit_summary_stats(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Compute summary metrics from query telemetry."""
    if not Path(db_path).exists():
        return {"total_queries": 0, "passed": 0, "refused": 0, "avg_latency_ms": 0.0}

    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN gate_status = 'PASSED' THEN 1 ELSE 0 END) as passed_cnt,
                    SUM(CASE WHEN gate_status = 'REFUSED' THEN 1 ELSE 0 END) as refused_cnt,
                    AVG(latency_ms) as avg_latency
                FROM query_telemetry;
                """
            )
            row = cursor.fetchone()
            if row and row[0] > 0:
                return {
                    "total_queries": row[0] or 0,
                    "passed": row[1] or 0,
                    "refused": row[2] or 0,
                    "avg_latency_ms": round(row[3] or 0.0, 1),
                }
    except Exception as e:
        logger.warning(f"Failed to compute audit summary stats: {e}")

    return {"total_queries": 0, "passed": 0, "refused": 0, "avg_latency_ms": 0.0}
