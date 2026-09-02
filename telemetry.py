"""
telemetry.py — SQLite-backed Local Retrieval Audit Trail & Telemetry Logging.
Provides persistent audit logging for query latency, TTFT, refusal gating, matching scores, and storage mode.
Also handles user feedback for active learning loop.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger("wealthchronicle.telemetry")
DEFAULT_DB_PATH = "telemetry.db"


def init_telemetry_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite telemetry database and create tables if not existing."""
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
                    chunks_retrieved_count INTEGER,
                    ttft_ms REAL DEFAULT 0.0
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON query_telemetry (timestamp);
                """
            )
            # Feedback table for active learning
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS query_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    query_text TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    label TEXT NOT NULL,
                    corrected_answer TEXT,
                    citations_json TEXT,
                    trace_id TEXT,
                    user_id TEXT,
                    metadata_json TEXT
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON query_feedback (timestamp);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_label ON query_feedback (label);
                """
            )
            # Golden set candidates table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS golden_set_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT UNIQUE NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    question TEXT NOT NULL,
                    ground_truth TEXT NOT NULL,
                    source_edition_dates_json TEXT,
                    source_pages_json TEXT,
                    category TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    derived_from_feedback_id TEXT NOT NULL,
                    reviewer_notes TEXT,
                    status TEXT DEFAULT 'pending'
                );
                """
            )
            # Ensure ttft_ms column exists for schema evolution
            try:
                cursor.execute("ALTER TABLE query_telemetry ADD COLUMN ttft_ms REAL DEFAULT 0.0;")
            except Exception:
                pass
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
    ttft_ms: float = 0.0,
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
                    timestamp, query_text, storage_mode, top_score, gate_status, latency_ms, chunks_retrieved_count, ttft_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    utc_now,
                    query_text,
                    storage_mode,
                    float(top_score),
                    gate_status,
                    float(latency_ms),
                    int(chunks_retrieved_count),
                    float(ttft_ms),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Non-blocking telemetry log failure: {e}")


def log_query_feedback(
    feedback_id: UUID,
    query_text: str,
    answer_text: str,
    label: str,
    corrected_answer: str | None = None,
    citations: list[dict] | None = None,
    trace_id: str | None = None,
    user_id: str | None = None,
    metadata: dict | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Log user feedback for active learning loop."""
    try:
        init_telemetry_db(db_path)
        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_feedback (
                    feedback_id, timestamp, query_text, answer_text, label, corrected_answer,
                    citations_json, trace_id, user_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(feedback_id),
                    utc_now,
                    query_text,
                    answer_text,
                    label,
                    corrected_answer,
                    json.dumps(citations) if citations else None,
                    trace_id,
                    user_id,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Non-blocking feedback log failure: {e}")


def fetch_feedback_logs(
    limit: int = 100,
    label_filter: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Retrieve recent feedback logs, optionally filtered by label."""
    if not Path(db_path).exists():
        return []

    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if label_filter:
                cursor.execute(
                    """
                    SELECT feedback_id, timestamp, query_text, answer_text, label, corrected_answer,
                           citations_json, trace_id, user_id, metadata_json
                    FROM query_feedback
                    WHERE label = ?
                    ORDER BY timestamp DESC
                    LIMIT ?;
                    """,
                    (label_filter, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT feedback_id, timestamp, query_text, answer_text, label, corrected_answer,
                           citations_json, trace_id, user_id, metadata_json
                    FROM query_feedback
                    ORDER BY timestamp DESC
                    LIMIT ?;
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if d.get("citations_json"):
                    d["citations"] = json.loads(d["citations_json"])
                if d.get("metadata_json"):
                    d["metadata"] = json.loads(d["metadata_json"])
                result.append(d)
            return result
    except Exception as e:
        logger.warning(f"Failed to fetch feedback logs: {e}")
        return []


def get_feedback_summary_stats(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Compute summary metrics from feedback logs."""
    if not Path(db_path).exists():
        return {"total": 0, "positive": 0, "negative": 0, "corrected": 0}

    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN label = 'positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN label = 'negative' THEN 1 ELSE 0 END) as negative,
                    SUM(CASE WHEN label = 'corrected' THEN 1 ELSE 0 END) as corrected
                FROM query_feedback;
                """
            )
            row = cursor.fetchone()
            if row and row[0] > 0:
                return {
                    "total": row[0] or 0,
                    "positive": row[1] or 0,
                    "negative": row[2] or 0,
                    "corrected": row[3] or 0,
                }
    except Exception as e:
        logger.warning(f"Failed to compute feedback summary stats: {e}")

    return {"total": 0, "positive": 0, "negative": 0, "corrected": 0}


def log_golden_set_candidate(
    candidate_id: str,
    question: str,
    ground_truth: str,
    source_edition_dates: list[str],
    source_pages: list[int],
    category: str,
    difficulty: str,
    derived_from_feedback_id: UUID,
    reviewer_notes: str | None = None,
    status: str = "pending",
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Log a golden set candidate derived from user feedback."""
    try:
        init_telemetry_db(db_path)
        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO golden_set_candidates (
                    candidate_id, timestamp, question, ground_truth, source_edition_dates_json,
                    source_pages_json, category, difficulty, derived_from_feedback_id,
                    reviewer_notes, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    candidate_id,
                    utc_now,
                    question,
                    ground_truth,
                    json.dumps(source_edition_dates),
                    json.dumps(source_pages),
                    category,
                    difficulty,
                    str(derived_from_feedback_id),
                    reviewer_notes,
                    status,
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Non-blocking golden set candidate log failure: {e}")


def fetch_golden_set_candidates(
    status_filter: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Retrieve golden set candidates, optionally filtered by status."""
    if not Path(db_path).exists():
        return []

    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if status_filter:
                cursor.execute(
                    """
                    SELECT candidate_id, timestamp, question, ground_truth, source_edition_dates_json,
                           source_pages_json, category, difficulty, derived_from_feedback_id,
                           reviewer_notes, status
                    FROM golden_set_candidates
                    WHERE status = ?
                    ORDER BY timestamp DESC;
                    """,
                    (status_filter,),
                )
            else:
                cursor.execute(
                    """
                    SELECT candidate_id, timestamp, question, ground_truth, source_edition_dates_json,
                           source_pages_json, category, difficulty, derived_from_feedback_id,
                           reviewer_notes, status
                    FROM golden_set_candidates
                    ORDER BY timestamp DESC;
                    """
                )
            rows = cursor.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if d.get("source_edition_dates_json"):
                    d["source_edition_dates"] = json.loads(d["source_edition_dates_json"])
                if d.get("source_pages_json"):
                    d["source_pages"] = json.loads(d["source_pages_json"])
                result.append(d)
            return result
    except Exception as e:
        logger.warning(f"Failed to fetch golden set candidates: {e}")
        return []


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
                SELECT id, timestamp, query_text, storage_mode, top_score, gate_status, latency_ms, ttft_ms, chunks_retrieved_count
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
        return {"total_queries": 0, "passed": 0, "refused": 0, "avg_latency_ms": 0.0, "avg_ttft_ms": 0.0}

    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN gate_status = 'PASSED' THEN 1 ELSE 0 END) as passed_cnt,
                    SUM(CASE WHEN gate_status = 'REFUSED' THEN 1 ELSE 0 END) as refused_cnt,
                    AVG(latency_ms) as avg_latency,
                    AVG(ttft_ms) as avg_ttft
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
                    "avg_ttft_ms": round(row[4] or 0.0, 1),
                }
    except Exception as e:
        logger.warning(f"Failed to compute audit summary stats: {e}")

    return {"total_queries": 0, "passed": 0, "refused": 0, "avg_latency_ms": 0.0, "avg_ttft_ms": 0.0}
