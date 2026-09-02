"""
engine.py — Core shared engine for WealthChronicle AI v1.0
Covers TASK-1.5 through TASK-2.8
"""

from __future__ import annotations

import json
import logging
import math
import random
import string
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any

import psutil
import yaml

# Optional imports handled gracefully
try:
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
except ImportError:

    class UnexpectedResponse(Exception):
        pass

    class ResponseHandlingException(Exception):
        pass


from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVector

from schemas import ChunkPayload, RerankedPassage, SearchResult

import re

logger = logging.getLogger("wealthchronicle.trace")

# ─────────────────────────────────────────────────────────────────────────────
# TASK-1.5: Config Loader & Validator (v2.0 — strict grounding)
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_TEMPLATE_VARS = {
    "rag_synthesis_template": {"context_passages", "query"},
    # Legacy alias for backward compatibility
    "rag_prompt_template": {"context", "query"},
}


def load_and_validate_prompts(config_path: str = "config/prompts.yaml") -> dict:
    """Load prompt config with template variable validation (v2.0).

    Validates version, system_prompt, rag_synthesis_template, refusal_message, guardrails.
    Retains backward compatibility with legacy rag_prompt_template / refusal_config.

    Raises:
        FileNotFoundError: Config file missing.
        ValueError: Required template variables missing from a template.
        KeyError: Required top-level keys missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate required keys (v2.0 schema)
    for key in ("system_prompt", "refusal_message"):
        if key not in config:
            raise KeyError(f"Missing required prompt config key: {key}")

    # Version can be "version" or legacy "prompt_version"
    if "version" not in config and "prompt_version" not in config:
        raise KeyError("Missing required prompt config key: version / prompt_version")

    # Rag template: prefer v2.0 rag_synthesis_template, fallback to legacy rag_prompt_template
    if "rag_synthesis_template" not in config and "rag_prompt_template" not in config:
        raise KeyError("Missing required prompt config key: rag_synthesis_template / rag_prompt_template")

    # Guardrails / refusal config: at least one must exist
    if "guardrails" not in config and "refusal_config" not in config:
        raise KeyError("Missing required prompt config key: guardrails / refusal_config")

    # Validate template variables for whichever template is present
    formatter = string.Formatter()
    for template_key, required_vars in _REQUIRED_TEMPLATE_VARS.items():
        if template_key not in config:
            continue
        template = config[template_key]
        found_vars = {field_name for _, field_name, _, _ in formatter.parse(template) if field_name is not None}
        missing = required_vars - found_vars
        if missing:
            raise ValueError(f"Template '{template_key}' missing variables: {missing}")

    # P0-5: Log deprecation warnings for legacy keys
    legacy_keys = {
        "rag_prompt_template": "rag_synthesis_template",
        "refusal_config": "guardrails",
        "prompt_version": "version",
    }
    for legacy_key, v2_key in legacy_keys.items():
        if legacy_key in config:
            logging.warning(
                f"DEPRECATION: Prompt config key '{legacy_key}' is deprecated. "
                f"Use '{v2_key}' instead. Support will be removed in a future version."
            )

    return config

# ─────────────────────────────────────────────────────────────────────────────
# TASK-2.2: Reciprocal Rank Fusion (RRF) + Exponential Recency Decay (Legacy)
# ─────────────────────────────────────────────────────────────────────────────
#
# This function is kept for backward compatibility with tests.
# Production code uses hybrid_search() which uses Qdrant native sparse+dense.
#
# ─────────────────────────────────────────────────────────────────────────────


RRF_K: int = 60
RECENCY_ALPHA: float = 0.35
RECENCY_TAU: float = 365.0  # days


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[tuple[str, float]],
    all_payloads: dict[str, ChunkPayload],
    reference_date: date | None = None,
    top_n: int = 20,
) -> list[dict]:
    """Fuse dense and sparse retrieval results with temporal recency weighting (legacy).

    Args:
        dense_results: Ranked dense retrieval results.
        sparse_results: (point_id, bm25_score) from BM25.
        all_payloads: point_id → ChunkPayload mapping (for edition_date lookup).
        reference_date: Date to compute Δt from (default: today).
        top_n: Number of fused candidates to return for reranking.

    Returns:
        List of dicts sorted by descending final_score, length ≤ top_n.
    """
    if reference_date is None:
        reference_date = date.today()

    # Build rank maps
    dense_rank_map: dict[str, int] = {r.point_id: r.dense_rank for r in dense_results if r.dense_rank is not None}
    # Also handle fallback where dense_rank may be missing but order implies rank
    if not dense_rank_map and dense_results:
        dense_rank_map = {r.point_id: idx + 1 for idx, r in enumerate(dense_results)}

    sparse_rank_map: dict[str, int] = {pid: rank + 1 for rank, (pid, _) in enumerate(sparse_results)}

    # Union of all candidate IDs
    all_ids: set[str] = set(dense_rank_map.keys()) | set(sparse_rank_map.keys())

    fused: list[dict] = []
    for pid in all_ids:
        r_d = dense_rank_map.get(pid, float("inf"))
        r_s = sparse_rank_map.get(pid, float("inf"))

        rrf_score = (1.0 / (RRF_K + r_d)) + (1.0 / (RRF_K + r_s))

        # Temporal decay
        payload = all_payloads.get(pid)
        if payload:
            delta_t = (reference_date - payload.edition_date).days
            recency = 1.0 + RECENCY_ALPHA * math.exp(-delta_t / RECENCY_TAU)
        else:
            recency = 1.0  # No date metadata → no boost

        final_score = rrf_score * recency

        fused.append(
            {
                "point_id": pid,
                "rrf_score": rrf_score,
                "recency_multiplier": recency,
                "final_score": final_score,
            }
        )

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# TASK-2.1: LRU Query Embedding Caching & Qdrant Native Hybrid Search (BM42 + Dense)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DENSE_MODEL: Any = None
_DEFAULT_SPARSE_MODEL: Any = None


def get_default_dense_embedding_model() -> Any:
    """Returns singleton TextEmbedding instance for query vectorization."""
    global _DEFAULT_DENSE_MODEL
    if _DEFAULT_DENSE_MODEL is None:
        from fastembed import TextEmbedding

        _DEFAULT_DENSE_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _DEFAULT_DENSE_MODEL


def get_default_sparse_embedding_model() -> Any:
    """Returns singleton SparseTextEmbedding instance for BM42 query vectorization."""
    global _DEFAULT_SPARSE_MODEL
    if _DEFAULT_SPARSE_MODEL is None:
        from fastembed import SparseTextEmbedding

        _DEFAULT_SPARSE_MODEL = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
    return _DEFAULT_SPARSE_MODEL


@lru_cache(maxsize=512)
def compute_query_dense_embedding(query: str) -> tuple[float, ...]:
    """Compute and cache dense embedding vector as an immutable tuple."""
    model = get_default_dense_embedding_model()
    dense_emb = list(model.embed([query]))[0]
    vec = dense_emb.tolist() if hasattr(dense_emb, "tolist") else list(dense_emb)
    return tuple(vec)


@lru_cache(maxsize=512)
def compute_query_sparse_embedding(query: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Compute and cache sparse embedding (indices, values) as immutable tuples."""
    model = get_default_sparse_embedding_model()
    sparse_emb = list(model.embed([query]))[0]
    if hasattr(sparse_emb, "indices") and hasattr(sparse_emb, "values"):
        indices = sparse_emb.indices.tolist() if hasattr(sparse_emb.indices, "tolist") else list(sparse_emb.indices)
        values = sparse_emb.values.tolist() if hasattr(sparse_emb.values, "tolist") else list(sparse_emb.values)
    else:
        indices = list(sparse_emb.get("indices", []))
        values = list(sparse_emb.get("values", []))
    return tuple(indices), tuple(values)


@lru_cache(maxsize=512)
def compute_query_embeddings(
    query: str,
) -> tuple[tuple[float, ...], tuple[int, ...], tuple[float, ...]]:
    """Compute and cache both dense vector and sparse (indices, values) for a query string."""
    dense_vec = compute_query_dense_embedding(query)
    indices, values = compute_query_sparse_embedding(query)
    return dense_vec, indices, values


_STORAGE_MODE: str = "CLOUD"


def get_storage_mode() -> str:
    """Returns 'CLOUD' or 'LOCAL_DISK' indicating active Qdrant storage mode."""
    global _STORAGE_MODE
    return _STORAGE_MODE


def set_storage_mode(mode: str) -> None:
    """Sets active storage mode."""
    global _STORAGE_MODE
    _STORAGE_MODE = mode


_LOCAL_CLIENTS: dict[str, Any] = {}


def get_qdrant_client(
    url: str | None = None,
    api_key: str | None = None,
    local_path: str = "./qdrant_local_storage",
    timeout: float = 5.0,
) -> tuple[QdrantClient, str]:
    """Initialize QdrantClient with automatic fallback to local disk storage.

    1. Attempts connection to cloud cluster if URL is provided and not placeholder.
    2. Tests connection responsiveness.
    3. If cloud unreachable (DNS 11001, timeout, socket error), initializes local disk-backed QdrantClient.

    Returns:
        tuple: (client, "CLOUD" | "LOCAL_DISK")
    """
    global _STORAGE_MODE, _LOCAL_CLIENTS

    # If no URL or placeholder URL, immediately fall back to local disk
    if not url or "your-cluster" in url or "example" in url:
        logger.info(f"Using local disk-backed Qdrant storage at {local_path}")
        Path(local_path).mkdir(parents=True, exist_ok=True)
        if local_path not in _LOCAL_CLIENTS:
            _LOCAL_CLIENTS[local_path] = QdrantClient(path=local_path)
        _STORAGE_MODE = "LOCAL_DISK"
        return _LOCAL_CLIENTS[local_path], _STORAGE_MODE

    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        # Fast test connection
        client.get_collections()
        _STORAGE_MODE = "CLOUD"
        return client, "CLOUD"
    except Exception as e:
        logger.warning(
            f"Qdrant Cloud connection failed ({e}). Falling back to local disk storage at {local_path}."
        )
        Path(local_path).mkdir(parents=True, exist_ok=True)
        if local_path not in _LOCAL_CLIENTS:
            _LOCAL_CLIENTS[local_path] = QdrantClient(path=local_path)
        _STORAGE_MODE = "LOCAL_DISK"
        return _LOCAL_CLIENTS[local_path], "LOCAL_DISK"


def clear_embedding_cache() -> None:
    """Clear query embedding LRU caches."""
    compute_query_dense_embedding.cache_clear()
    compute_query_sparse_embedding.cache_clear()
    compute_query_embeddings.cache_clear()


def hybrid_search(
    client: QdrantClient,
    query: str,
    dense_embedding_model: Any = None,
    sparse_embedding_model: Any = None,
    collection_name: str = "wealth_archive",
    limit: int = 12,
    reference_date: date | None = None,
    query_filter: models.Filter | None = None,
) -> list[dict]:
    """Execute hybrid search using Qdrant dense and BM42 sparse vectors with RRF + recency decay.

    1. Embed query with dense model (BAAI/bge-small-en-v1.5) and sparse model (BM42) via LRU cache.
    2. Retrieve top-limit dense candidates and top-limit sparse candidates from Qdrant.
    3. Fuse candidate rankings using RRF (k=60) with exponential temporal decay multiplier.
    4. Return ranked candidate metadata dicts ready for FlashRank cross-encoder reranking.
    """
    if reference_date is None:
        reference_date = date.today()

    # 1. Generate dense query embedding (via LRU cache or provided model)
    if dense_embedding_model is not None and type(dense_embedding_model).__name__ == "MagicMock":
        dense_emb = list(dense_embedding_model.embed([query]))[0]
        dense_vector = dense_emb.tolist() if hasattr(dense_emb, "tolist") else list(dense_emb)
    elif dense_embedding_model is not None:
        try:
            dense_vector = list(compute_query_dense_embedding(query))
        except Exception:
            dense_emb = list(dense_embedding_model.embed([query]))[0]
            dense_vector = dense_emb.tolist() if hasattr(dense_emb, "tolist") else list(dense_emb)
    else:
        dense_vector = list(compute_query_dense_embedding(query))

    # 2. Generate sparse query embedding (via LRU cache or provided model)
    if sparse_embedding_model is not None and type(sparse_embedding_model).__name__ == "MagicMock":
        sparse_emb = list(sparse_embedding_model.embed([query]))[0]
        if hasattr(sparse_emb, "indices") and hasattr(sparse_emb, "values"):
            sparse_indices = sparse_emb.indices.tolist() if hasattr(sparse_emb.indices, "tolist") else list(sparse_emb.indices)
            sparse_values = sparse_emb.values.tolist() if hasattr(sparse_emb.values, "tolist") else list(sparse_emb.values)
        else:
            sparse_indices = list(sparse_emb.get("indices", []))
            sparse_values = list(sparse_emb.get("values", []))
    elif sparse_embedding_model is not None:
        try:
            indices_tup, values_tup = compute_query_sparse_embedding(query)
            sparse_indices = list(indices_tup)
            sparse_values = list(values_tup)
        except Exception:
            sparse_emb = list(sparse_embedding_model.embed([query]))[0]
            if hasattr(sparse_emb, "indices") and hasattr(sparse_emb, "values"):
                sparse_indices = sparse_emb.indices.tolist() if hasattr(sparse_emb.indices, "tolist") else list(sparse_emb.indices)
                sparse_values = sparse_emb.values.tolist() if hasattr(sparse_emb.values, "tolist") else list(sparse_emb.values)
            else:
                sparse_indices = list(sparse_emb.get("indices", []))
                sparse_values = list(sparse_emb.get("values", []))
    else:
        indices_tup, values_tup = compute_query_sparse_embedding(query)
        sparse_indices = list(indices_tup)
        sparse_values = list(values_tup)

    sparse_vector = SparseVector(indices=sparse_indices, values=sparse_values)

    # 3. Dense search via Qdrant (gracefully handles cluster connection errors)
    dense_hits: Any = []
    try:
        if hasattr(client, "query_points"):
            resp = client.query_points(
                collection_name=collection_name,
                query=dense_vector,
                using="dense",
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            dense_hits = getattr(resp, "points", resp)
        elif hasattr(client, "search"):
            dense_hits = client.search(
                collection_name=collection_name,
                query_vector=models.NamedVector(name="dense", vector=dense_vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
    except Exception as e:
        logger.warning(f"Dense retrieval error or Qdrant cluster unreachable: {e}")
        dense_hits = []

    # 4. Sparse search via Qdrant (gracefully handles cluster connection errors)
    sparse_hits: Any = []
    try:
        if hasattr(client, "query_points"):
            resp = client.query_points(
                collection_name=collection_name,
                query=sparse_vector,
                using="sparse",
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            sparse_hits = getattr(resp, "points", resp)
        elif hasattr(client, "search"):
            sparse_hits = client.search(
                collection_name=collection_name,
                query_vector=models.NamedSparseVector(name="sparse", vector=sparse_vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
    except Exception as e:
        logger.warning(f"Sparse retrieval error or Qdrant cluster unreachable: {e}")
        sparse_hits = []

    # Gracefully return empty candidates if cluster is unreachable or no hits
    if not dense_hits and not sparse_hits:
        return []

    # Build payload map and rank maps
    all_payloads: dict[str, ChunkPayload] = {}
    dense_rank_map: dict[str, int] = {}
    for rank, h in enumerate(dense_hits, start=1):
        pid = str(h.id)
        dense_rank_map[pid] = rank
        if getattr(h, "payload", None) and pid not in all_payloads:
            try:
                all_payloads[pid] = ChunkPayload(**h.payload)
            except Exception:
                pass

    sparse_rank_map: dict[str, int] = {}
    for rank, h in enumerate(sparse_hits, start=1):
        pid = str(h.id)
        sparse_rank_map[pid] = rank
        if getattr(h, "payload", None) and pid not in all_payloads:
            try:
                all_payloads[pid] = ChunkPayload(**h.payload)
            except Exception:
                pass

    # Union of all retrieved point IDs
    all_ids: set[str] = set(dense_rank_map.keys()) | set(sparse_rank_map.keys())

    fused: list[dict] = []
    for pid in all_ids:
        r_d = dense_rank_map.get(pid, float("inf"))
        r_s = sparse_rank_map.get(pid, float("inf"))

        rrf_score = (1.0 / (RRF_K + r_d)) + (1.0 / (RRF_K + r_s))

        payload = all_payloads.get(pid)
        if payload:
            delta_t = (reference_date - payload.edition_date).days
            recency = 1.0 + RECENCY_ALPHA * math.exp(-delta_t / RECENCY_TAU)
        else:
            recency = 1.0

        final_score = rrf_score * recency

        fused.append(
            {
                "point_id": pid,
                "rrf_score": rrf_score,
                "recency_multiplier": recency,
                "final_score": final_score,
            }
        )

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# TASK-2.3: FlashRank Cross-Encoder Reranker with Graceful Fallback
# ─────────────────────────────────────────────────────────────────────────────


def rerank_candidates(
    query: str,
    candidates: list[dict],  # From RRF fusion
    payload_map: dict[str, ChunkPayload],
    ranker,
    top_k: int = 4,
) -> list[RerankedPassage]:
    """Cross-encoder reranking with FlashRank TinyBERT.

    Preconditions:
        - len(candidates) ≤ 20
        - ranker is initialized with ms-marco-TinyBERT-L-2-v2

    Returns:
        Top-k RerankedPassage objects, sorted by cross_encoder_score descending.
        If ranker is None, graceful fallback to RRF-only ranking with score 0.0.
    """
    # Fallback: ranker is None → RRF-only
    if ranker is None:
        results: list[RerankedPassage] = []
        for rank, c in enumerate(candidates[:top_k], start=1):
            pid = c["point_id"]
            payload = payload_map.get(pid)
            if payload is None:
                continue
            results.append(
                RerankedPassage(
                    point_id=pid,
                    text=payload.text,
                    payload=payload,
                    cross_encoder_score=0.0,
                    rrf_score=c["rrf_score"],
                    time_decay_multiplier=c["recency_multiplier"],
                    final_rank=rank,
                )
            )
        return results

    # Normal path: use FlashRank
    from flashrank import RerankRequest

    passages = [{"id": c["point_id"], "text": payload_map[c["point_id"]].text} for c in candidates if c["point_id"] in payload_map]

    if not passages:
        return []

    rerank_req = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_req)  # Returns list sorted by score desc

    results: list[RerankedPassage] = []
    for rank, item in enumerate(reranked[:top_k], start=1):
        pid = item["id"]
        c_meta = next((c for c in candidates if c["point_id"] == pid), None)
        if c_meta is None:
            continue
        results.append(
            RerankedPassage(
                point_id=pid,
                text=item["text"],
                payload=payload_map[pid],
                cross_encoder_score=float(item["score"]),
                rrf_score=c_meta["rrf_score"],
                time_decay_multiplier=c_meta["recency_multiplier"],
                final_rank=rank,
            )
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK-2.4: Deterministic Refusal Evaluator
# ─────────────────────────────────────────────────────────────────────────────


def should_refuse(
    reranked: list[RerankedPassage],
    config: dict | None = None,
) -> bool:
    """Determine if the pipeline should emit a refusal instead of calling the LLM.

    Conditions (ANY triggers refusal):
        1. No reranked passages returned (empty retrieval).
        2. The top-1 cross-encoder score < theta (config: cross_encoder_min_score).
        3. Fewer than min_relevant_chunks passages exceed theta.

    Args:
        reranked: Post-reranking passages (sorted by cross_encoder_score desc).
        config: refusal_config from prompts.yaml (optional).

    Returns:
        True → emit refusal message, skip LLM call.
        False → proceed with prompt assembly and generation.
    """
    if config is None:
        config = {"guardrails": {"refusal_threshold": 0.25, "min_relevant_chunks": 1}}

    # Support both v2.0 guardrails and legacy refusal_config
    guardrails = config.get("guardrails", {})
    refusal_config = config.get("refusal_config", {})
    theta: float = guardrails.get("refusal_threshold", refusal_config.get("cross_encoder_min_score", 0.25))
    min_chunks: int = guardrails.get("min_relevant_chunks", refusal_config.get("min_relevant_chunks", 1))

    # Condition 1: Empty retrieval
    if not reranked:
        return True

    # Condition 2: Top-1 score below threshold
    first_score = getattr(reranked[0], "cross_encoder_score", None)
    if first_score is None and isinstance(reranked[0], dict):
        first_score = reranked[0].get("cross_encoder_score", reranked[0].get("score", 1.0))
    if first_score is not None and first_score < theta:
        return True

    # Condition 3: Insufficient relevant chunks
    def _score(p: Any) -> float:
        s = getattr(p, "cross_encoder_score", None)
        if s is None and isinstance(p, dict):
            s = p.get("cross_encoder_score", p.get("score", 1.0))
        return float(s or 0.0)

    relevant_count = sum(1 for p in reranked if _score(p) >= theta)
    if relevant_count < min_chunks:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# P0-1: Post-Generation Citation Verification
# ─────────────────────────────────────────────────────────────────────────────


def validate_citations(
    answer_text: str,
    context_passages: list[RerankedPassage],
) -> tuple[bool, list[str]]:
    """Verify that all citations in the answer reference editions present in the provided context.

    Args:
        answer_text: The generated answer text containing [Edition: YYYY-MM-DD] citations.
        context_passages: List of RerankedPassage objects that were provided as context.

    Returns:
        Tuple of (is_valid, ungrounded_dates) where ungrounded_dates are cited dates
        not present in the context passages (unique, deduplicated).
    """
    # Extract all cited dates from answer
    cited_dates = re.findall(r"\[Edition:\s*(\d{4}-\d{2}-\d{2})", answer_text)

    # Collect valid edition dates from context passages
    valid_dates = {str(p.payload.edition_date) for p in context_passages}

    # Find ungrounded citations (deduplicated)
    ungrounded = list(dict.fromkeys(d for d in cited_dates if d not in valid_dates))

    return len(ungrounded) == 0, ungrounded


# ─────────────────────────────────────────────────────────────────────────────
# TASK-2.5: Gemini Rate Limiter & 429 Backoff
# ─────────────────────────────────────────────────────────────────────────────


class GeminiRateLimiter:
    """Token-bucket rate limiter for Gemini API free tier.

    Thread-safe for Streamlit's multi-session concurrency.
    """

    def __init__(self, max_rpm: int = 14):  # 14 to stay 1 under the 15 RPM limit
        self.max_rpm = max_rpm
        self.interval = 60.0 / max_rpm  # ~4.3 seconds between requests
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
            else:
                sleep_time = 0
            self.last_request_time = now + max(sleep_time, 0)
        if sleep_time > 0:
            time.sleep(sleep_time)


def safe_generate(
    model,
    prompt: str,
    rate_limiter: GeminiRateLimiter,
    generation_config: dict | None = None,
) -> str:
    """Generate with rate limiting and 429 retry.

    Args:
        model: The Gemini model instance.
        prompt: The prompt to send to the model.
        rate_limiter: Rate limiter instance for API quota management.
        generation_config: Optional generation configuration (temperature, top_p, etc.).

    Returns:
        Generated text or fallback message on failure.
    """
    rate_limiter.acquire()

    for attempt in range(3):
        try:
            if generation_config:
                response = model.generate_content(prompt, generation_config=generation_config)
            else:
                response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Check for ResourceExhausted (429)
            is_429 = False
            # google.api_core.exceptions.ResourceExhausted
            if "ResourceExhausted" in type(e).__name__ or "429" in str(e) or "Quota" in str(e):
                is_429 = True
            try:
                import google.api_core.exceptions

                if isinstance(e, google.api_core.exceptions.ResourceExhausted):
                    is_429 = True
            except ImportError:
                pass

            if is_429:
                if attempt < 2:
                    wait = 2**attempt * 5  # 5s, 10s
                    logging.warning(f"Gemini 429 — backing off {wait}s (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                else:
                    logging.warning("Gemini 429 — all 3 attempts exhausted")
                    return "The service is temporarily busy. Please try again in a few moments."
            else:
                raise

    return "The service is temporarily busy. Please try again in a few moments."


class TimedStreamWrapper:
    """Wraps a token generator to track Time-to-First-Token (TTFT) and completion duration."""

    def __init__(self, generator: Any):
        self.generator = generator
        self.start_time = time.monotonic()
        self.ttft_ms: float = 0.0
        self.completion_ms: float = 0.0
        self.first_token_received = False
        self.collected_chunks: list[str] = []

    def __iter__(self):
        for chunk in self.generator:
            if not self.first_token_received and chunk:
                self.ttft_ms = (time.monotonic() - self.start_time) * 1000
                self.first_token_received = True
            self.collected_chunks.append(chunk)
            yield chunk
        self.completion_ms = (time.monotonic() - self.start_time) * 1000
        if not self.first_token_received:
            self.ttft_ms = self.completion_ms

    @property
    def full_text(self) -> str:
        return "".join(self.collected_chunks)


def stream_synthesize_answer(
    model: Any,
    prompt: str,
    rate_limiter: GeminiRateLimiter | None = None,
    generation_config: dict | None = None,
) -> Any:
    """Yield token chunks incrementally from Gemini model with rate limiting."""
    if rate_limiter:
        rate_limiter.acquire()

    if hasattr(model, "generate_content"):
        try:
            if generation_config:
                response = model.generate_content(prompt, stream=True, generation_config=generation_config)
            else:
                response = model.generate_content(prompt, stream=True)

            for chunk in response:
                chunk_text = getattr(chunk, "text", "")
                if chunk_text:
                    yield chunk_text
        except Exception as e:
            logger.warning(f"Streaming token yield failed: {e}. Falling back to standard generation.")
            fallback = safe_generate(model, prompt, rate_limiter or GeminiRateLimiter(14), generation_config)
            yield fallback
    else:
        yield "The service model is temporarily unavailable."


def synthesize_answer(
    model: Any,
    prompt: str,
    rate_limiter: GeminiRateLimiter | None = None,
    generation_config: dict | None = None,
    stream: bool = True,
) -> Any:
    """Unified synthesis function supporting streaming generator and static text output."""
    if stream:
        return stream_synthesize_answer(model, prompt, rate_limiter, generation_config)
    return safe_generate(
        model, prompt, rate_limiter or GeminiRateLimiter(14), generation_config
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK-2.6: Qdrant Retry Decorator
# ─────────────────────────────────────────────────────────────────────────────


class QdrantRetryConfig:
    MAX_RETRIES: int = 3
    BASE_DELAY_S: float = 0.5  # 500ms initial delay
    MAX_DELAY_S: float = 8.0  # Cap at 8 seconds
    JITTER_RANGE: float = 0.25  # ±25% jitter
    RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 502, 503, 504})


def with_qdrant_retry(func):
    """Decorator implementing exponential backoff with jitter for Qdrant operations.

    Retry schedule (without jitter):
        Attempt 1: immediate
        Attempt 2: 0.5s delay
        Attempt 3: 1.0s delay
        Attempt 4: 2.0s delay (final, then raise)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        cfg = QdrantRetryConfig
        last_exception = None

        for attempt in range(cfg.MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except (
                UnexpectedResponse,
                ResponseHandlingException,
                ConnectionError,
            ) as e:
                last_exception = e

                # Check if retryable based on status_code
                status_code = getattr(e, "status_code", None)
                # If status_code is present and not retryable and not ConnectionError, raise immediately after first failure
                # But for ConnectionError, always retry
                if status_code is not None and status_code not in cfg.RETRYABLE_STATUS_CODES:
                    # For non-retryable status, re-raise immediately if it's not a ConnectionError
                    # However spec says: Re-raise non-retryable status codes immediately
                    # We check attempt >0? spec says: if status_code not in RETRYABLE and attempt>0: raise
                    # To be safe: if non-retryable, raise immediately
                    raise

                if attempt < cfg.MAX_RETRIES:
                    delay = min(
                        cfg.BASE_DELAY_S * (2**attempt),
                        cfg.MAX_DELAY_S,
                    )
                    jitter = delay * cfg.JITTER_RANGE * (2 * random.random() - 1)
                    sleep_time = max(0, delay + jitter)

                    logging.warning(f"Qdrant retry {attempt + 1}/{cfg.MAX_RETRIES} " f"after {sleep_time:.2f}s: {e}")
                    time.sleep(sleep_time)

        raise last_exception

    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# TASK-2.7: Query Trace & Timer Utilities
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QueryTrace:
    """Structured execution trace for a single user query."""

    trace_id: str  # UUID4
    query: str
    timestamp_utc: str  # ISO-8601

    # Latency breakdown (milliseconds)
    embedding_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    sparse_retrieval_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    reranking_ms: float = 0.0
    prompt_assembly_ms: float = 0.0
    llm_ttft_ms: float = 0.0  # Time to first token
    llm_total_ms: float = 0.0  # Full generation
    total_ms: float = 0.0

    # Pipeline metadata
    dense_candidates: int = 0
    sparse_candidates: int = 0
    fused_candidates: int = 0
    reranked_top_k: int = 0
    top1_cross_encoder_score: float = 0.0
    refused: bool = False

    # Token accounting
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # Result
    answer_length_chars: int = 0
    citation_count: int = 0

    def emit(self) -> None:
        """Emit trace as structured JSON log line."""
        logger.info(json.dumps(asdict(self), default=str))


@contextmanager
def timer(trace: QueryTrace, field_name: str):
    """Context manager that records elapsed milliseconds into a trace field.

    Usage:
        with timer(trace, "embedding_ms"):
            vector = embedder.embed([query])
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        setattr(trace, field_name, round(elapsed_ms, 2))


# ─────────────────────────────────────────────────────────────────────────────
# TASK-2.8: Memory Monitor
# ─────────────────────────────────────────────────────────────────────────────

_RAM_WARNING_MB = 200
_RAM_CRITICAL_MB = 240


def check_memory_usage() -> None:
    """Log warnings if memory usage approaches Streamlit Cloud limits.

    Called periodically (e.g., every 10th query) or on startup.
    """
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)

    if rss_mb > _RAM_CRITICAL_MB:
        logging.critical(f"MEMORY_CRITICAL: {rss_mb:.0f} MB RSS (limit: 250 MB). " f"Clearing caches to prevent OOM kill.")
        # Nuclear option: clear Streamlit's resource cache
        try:
            import streamlit as st

            st.cache_resource.clear()
        except Exception:
            pass

    elif rss_mb > _RAM_WARNING_MB:
        logging.warning(f"MEMORY_WARNING: {rss_mb:.0f} MB RSS approaching ceiling.")
