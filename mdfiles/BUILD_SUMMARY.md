# WealthChronicle AI — Engineering Build & Systems Summary

**System Version:** WealthChronicle AI v2.0  
**Repository State:** Commit `296cd70` (Dual-Plane Streaming & Evaluation Architecture)  
**Total Test Suite:** 148 tests passing (`pytest tests/ -v`)  
**Static Verification:** Clean MyPy (7 source files), 0 Ruff lint errors  
**Classification:** Evaluated Dual-Plane Financial Intelligence RAG Engine  

---

## 1. System Inventory (7 Core Source Modules + Hardening Suites)

| Module | Location / Port | Primary Engineering Function | Lines |
|--------|-----------------|------------------------------|-------|
| `app.py` | Port 8501 (Public Terminal) | Read-only financial research UI, token streaming via `st.write_stream(TimedStreamWrapper)`, 5-metric telemetry ribbon, and `@st.dialog` inspection expanders. | ~745 |
| `admin_ingest_app.py` | Port 8502 (Admin Cockpit) | Dedicated document staging, batch vectorization (`batch_size=32`), live status telemetry, payload index management, and collection purging modal (`@st.dialog`). | ~495 |
| `engine.py` | Core Search Engine | Dual-mode Qdrant connection manager (`CLOUD` vs `LOCAL_DISK`), `@lru_cache(maxsize=512)` query embedding memoization, parallel hybrid search (`BM42` + `bge-small-en-v1.5`), RRF fusion ($k=60$), temporal recency decay ($\alpha=0.35, \tau=365$), FlashRank TinyBERT reranker, rate limiting, and `TimedStreamWrapper`. | ~970 |
| `ingest.py` | Ingestion Engine | PyMuPDF4LLM layout extraction, table isolation & Markdown normalization, watermark stripping, sliding window chunking ($S=600, O=100$), deterministic UUIDv5/MD5 IDs, and server-side payload index initialization. | ~460 |
| `schemas.py` | Data Contracts | Pydantic v2 domain models: `ChunkPayload` (with `has_table: bool`), `SearchResult`, `RerankedPassage`, `EvaluationItem`, and `CitationMetadata`. | ~175 |
| `telemetry.py` | Telemetry & Audit | SQLite-backed persistent audit logger (`telemetry.db`) tracking query text, latency, TTFT ms, gate status, top cross-encoder score, and active storage mode. | ~180 |
| `eval_runner.py` | IR Evaluation Runner | Standalone CLI evaluation benchmark runner computing Hit Rate @ 3, Hit Rate @ 5, MRR @ 5, and Refusal Precision against `tests/golden_eval_set_2026.json`. | ~270 |

---

## 2. Core Architectural Pillars Implemented

### 2.1 Dual-Plane Network & Process Separation
- **Public Terminal (`app.py`, Port 8501):**
  - Read-only queries using `QDRANT_READ_KEY` (or `LOCAL_DISK` fallback).
  - Streamlit UI with 5-metric telemetry readout: Total Latency, TTFT, Top Score, Time Decay, and Gate Status.
  - Streaming token delivery using `TimedStreamWrapper` and `st.write_stream()`.
- **Admin Ingestion Cockpit (`admin_ingest_app.py`, Port 8502):**
  - Staging area in `data/` with duplicate filename collision checks.
  - One-click batch extraction with SIMD multi-threading (`batch_size=32`).
  - Indexing matrix displaying payload indexes (`edition_date`, `has_table`, `page_number`, `source`).
  - Safe 2-step confirmation modal (`@st.dialog`) for collection purging.

### 2.2 Dual-Mode Storage Backend Fallback
- `get_qdrant_client()` tests connection against `QDRANT_URL`.
- If cloud connection fails (DNS Error 11001, connect timeout, or missing/placeholder cloud keys), gracefully initialises and returns cached local disk storage (`QdrantClient(path="./qdrant_local_storage")`).
- Prevents application failure behind corporate firewalls or offline networks.

### 2.3 Query Vector Memoization (`@lru_cache`)
- Query embedding functions in `engine.py` are decorated with `@lru_cache(maxsize=512)`:
  - `compute_query_dense_embedding(query_text)`
  - `compute_query_sparse_embedding(query_text)`
  - `compute_query_embeddings(query_text)`
- Reduces vectorization latency on repeated queries from ~20 ms down to < 0.1 ms.

### 2.4 Token Streaming Synthesis & TTFT Tracking
- `synthesize_answer(gemini_model, prompt, rate_limiter, stream=True)` yields incremental token chunks.
- `TimedStreamWrapper` measures:
  - **Time-to-First-Token (`ttft_ms`)**: Monotonic duration until first non-empty token is received.
  - **Total Completion Time (`completion_ms`)**: End-to-end token generation duration.
- Persisted directly to `telemetry.db`.

### 2.5 Standalone IR Benchmark Runner (`eval_runner.py`)
- Evaluates hybrid retrieval + cross-encoder reranking against `tests/golden_eval_set_2026.json`.
- Evaluates 4 key IR metrics:
  - **Hit Rate @ 3:** Target $\ge 85.0\%$
  - **Hit Rate @ 5:** Target $\ge 90.0\%$
  - **MRR @ 5:** Target $\ge 0.80$
  - **Refusal Precision:** Target $\ge 95.0\%$
- Emits formatted institutional ASCII summary reports to stdout and optional JSON output for CI/CD tracking.

---

## 3. Mathematical Formulations

### 3.1 Server-Side Reciprocal Rank Fusion (RRF)
$$RRF(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{60 + rank_m(d)}$$

### 3.2 Temporal Recency Decay
$$Score(d) = RRF(d) \times \left(1.0 + 0.35 \cdot e^{-\Delta t / 365}\right)$$

### 3.3 Deterministic Refusal Gate
$$\text{Gate Status} = \begin{cases} \text{REFUSED}, & \text{if } \max(S_{\text{rerank}}) < 0.25 \lor \text{chunks} = 0 \\ \text{PASSED}, & \text{otherwise} \end{cases}$$

---

## 4. Test Harness & Quality Metrics (148 Passing Tests)

### 4.1 Test Breakdown

| Test Suite | Test File | Count | Focus Areas |
|------------|-----------|-------|-------------|
| **Engine Extended** | `tests/test_engine_extended.py` | 36 | RRF math, temporal decay boundaries, FlashRank fallback, refusal thresholds, rate limiting concurrency, retry jitter. |
| **Ingest Mocked** | `tests/test_ingest_mocked.py` | 55 | Layout extraction, table-aware chunking, watermark stripping, deterministic UUIDv5/MD5 IDs, payload indexing. |
| **App Isolated** | `tests/test_app_isolated.py` | 18 | Session cap, prompt ordering, citation verification, telemetry ribbon formatting. |
| **Telemetry Suite** | `tests/test_telemetry.py` | 5 | Dual-mode storage fallback, SQLite logging, schema column migrations, error handling. |
| **Streaming Suite** | `tests/test_streaming.py` | 5 | Token chunk generation, `TimedStreamWrapper` TTFT measurement, empty generator handling. |
| **Eval Runner Suite** | `tests/test_eval_runner.py` | 7 | Hit Rate math, MRR math, Refusal Precision math, query item evaluation, ASCII report formatting. |
| **RAGAS Suite** | `tests/test_ragas_eval.py` | 1 | Faithfulness, Answer Relevancy, and Context Precision verification. |
| **Legacy Engine** | `tests/test_engine.py` | 21 | Foundational embedding and search tests. |
| **Total** | `tests/` | **148** | **100% test pass rate across all suites** |

### 4.2 Quality Verification Output

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
# ======================= 148 passed, 1 warning in 21.27s =======================

.\.venv\Scripts\python.exe -m ruff check . --fix
# All checks passed!

.\.venv\Scripts\python.exe -m mypy schemas.py engine.py ingest.py app.py admin_ingest_app.py telemetry.py eval_runner.py --ignore-missing-imports
# Success: no issues found in 7 source files

.\.venv\Scripts\python.exe eval_runner.py
# ======================================================================
#   WEALTHCHRONICLE AI — RETRIEVAL BENCHMARK EVALUATION REPORT
# ======================================================================
#   Storage Backend:          LOCAL_DISK
#   Total Queries Evaluated:  29
#   In-Domain Test Queries:   25
#   Out-of-Domain Queries:    4
#   Total Duration:           1.00 seconds
# ----------------------------------------------------------------------
#   Metric                       | Score      | Target     | Status    
# ----------------------------------------------------------------------
#   Hit Rate @ 3                 |   100.00% |    85.00% | [PASS]
#   Hit Rate @ 5                 |   100.00% |    90.00% | [PASS]
#   MRR @ 5                      |     1.0000 |     0.8000 | [PASS]
#   Refusal Precision            |   100.00% |    95.00% | [PASS]
# ======================================================================
#   OVERALL BENCHMARK VERDICT:  PASSED
# ======================================================================
```
