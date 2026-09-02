# WealthChronicle AI — Evaluated Financial Intelligence RAG Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Test Suite](https://img.shields.io/badge/tests-148%20passing-brightgreen.svg)](tests/)
[![Memory Footprint](https://img.shields.io/badge/memory-~135%20MB-blue.svg)](mdfiles/BUILD_SUMMARY.md)
[![Type Checking](https://img.shields.io/badge/mypy-clean-success.svg)](mdfiles/BUILD_SUMMARY.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## 1. System Overview

WealthChronicle AI is an evaluated Retrieval-Augmented Generation (RAG) system for querying archived weekly financial dossiers. The system uses a dual-plane architecture:
- **Public Query Terminal (`app.py`, Port 8501):** Read-only user interface featuring real-time token streaming, Time-to-First-Token (TTFT) metrics, post-generation citation verification, and inspection modals.
- **Admin Ingestion Cockpit (`admin_ingest_app.py`, Port 8502):** Corpus management console for PDF upload, layout-aware extraction, multi-threaded batch vectorization (`batch_size=32`), payload index maintenance, and cluster purging.

### Core Architecture & Execution Flow

```text
[ ADMIN COCKPIT — Port 8502 ]                    [ QDRANT DUAL-MODE STORAGE ]
   PDF Upload / Staging                                │
     ├── PyMuPDF4LLM (Layout & Tables)                 ├── Qdrant Cloud Cluster ('wealth_archive')
     ├── Sliding Window Chunker (600/100)              └── Local Disk Storage ('./qdrant_local_storage')
     ├── Batch Vectorization (batch_size=32)               Named Vectors: 'dense' (384d) + 'sparse' (BM42)
     └── Deterministic UUIDv5 / MD5 Upsert                 Payload Indexes: edition_date, has_table, page_number
                                                        ▲
[ PUBLIC TERMINAL — Port 8501 ]                         │
   User Query + Optional Filters                        │
     ├── LRU Cache Lookup (@lru_cache maxsize=512)      │
     ├── Parallel Dense (384d) + Sparse (BM42) Search ──┘
     ├── Server-Side RRF (k=60) + Temporal Decay (α=0.35, τ=365)
     ├── FlashRank TinyBERT Cross-Encoder Reranking (top 4)
     ├── Deterministic Refusal Gate (θ = 0.25)
     │     ├── Triggered (< 0.25) ──► Deterministic Refusal Notice (0 LLM Tokens)
     │     └── Passed (>= 0.25)   ──► Structured Prompt ──► Gemini 2.5 Flash API
     ├── Token Streaming (st.write_stream) with TTFT Measurement
     ├── Post-Generation Citation Grounding Verification
     └── SQLite Audit Logging (telemetry.db)
```

---

## 2. Mathematical Formulations

- **Reciprocal Rank Fusion (RRF):**
  $$RRF(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{60 + rank_m(d)}$$

- **Temporal Recency Decay:**
  $$Score(d) = RRF(d) \times \left(1.0 + 0.35 \cdot e^{-\Delta t / 365}\right)$$

- **Deterministic Refusal Gate:**
  $$\text{Gate Decision} = \begin{cases} \text{REFUSED}, & \text{if } \max(S_{\text{rerank}}) < 0.25 \lor \text{chunks} = 0 \\ \text{PASSED}, & \text{otherwise} \end{cases}$$

---

## 3. Quickstart Guide

### 3.1 Setup Environment

```powershell
# Clone repository
git clone https://github.com/swaritbkp/wealth-chronicle-rag.git
cd wealth-chronicle-rag

# Create and activate Python 3.11 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install locked dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest pytest-cov ruff mypy
```

### 3.2 Configure Secrets

Create `.streamlit/secrets.toml` (for Streamlit apps) or set environment variables:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-gemini-api-key"
QDRANT_URL = "https://your-cluster-id.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_READ_KEY = "your-read-only-key"
QDRANT_ADMIN_KEY = "your-admin-key"
```

> **Note:** If `QDRANT_URL` is omitted or unreachable, the system automatically falls back to local disk-backed storage (`./qdrant_local_storage`).

---

## 4. Running the Applications

### 4.1 Launch Public Query Terminal (Port 8501)

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```
Access the public terminal at `http://localhost:8501`.

### 4.2 Launch Admin Ingestion Cockpit (Port 8502)

```powershell
.\.venv\Scripts\python.exe -m streamlit run admin_ingest_app.py --server.port 8502
```
Access the admin management cockpit at `http://localhost:8502`.

### 4.3 Ingest via CLI

```powershell
# Auto-detect edition date from PDF masthead
.\.venv\Scripts\python.exe ingest.py data/wealth_edition.pdf

# Ingest with explicit edition date
.\.venv\Scripts\python.exe ingest.py data/wealth_edition.pdf 2026-08-24
```

---

## 5. Verification & Benchmark Testing

### 5.1 Run Full Test Suite (148 Tests)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### 5.2 Run IR Benchmark Evaluation Runner

```powershell
.\.venv\Scripts\python.exe eval_runner.py --eval-set tests/golden_eval_set_2026.json
```

**Target Thresholds vs Results:**
- **Hit Rate @ 3:** Target $\ge 85.0\%$
- **Hit Rate @ 5:** Target $\ge 90.0\%$
- **MRR @ 5:** Target $\ge 0.80$
- **Refusal Precision:** Target $\ge 95.0\%$

### 5.3 Static Code Quality & Type Checking

```powershell
# Linting
.\.venv\Scripts\python.exe -m ruff check . --fix

# Strict Static Typing across all 7 source files
.\.venv\Scripts\python.exe -m mypy schemas.py engine.py ingest.py app.py admin_ingest_app.py telemetry.py eval_runner.py --ignore-missing-imports
```

---

## 6. Repository Layout

```text
wealth-chronicle-rag/
├── app.py                     # Public Query Terminal (Port 8501)
├── admin_ingest_app.py        # Admin Ingestion Cockpit (Port 8502)
├── engine.py                  # Search engine, LRU caching, fallback storage, reranking
├── ingest.py                  # Layout parsing, chunking, SIMD batch vectorization
├── schemas.py                 # Pydantic v2 domain schemas (ChunkPayload, SearchResult)
├── telemetry.py               # SQLite persistent audit logger (telemetry.db)
├── eval_runner.py             # CLI IR benchmark evaluation runner
├── config/
│   └── prompts.yaml           # System prompt, synthesis template, guardrails config
├── mdfiles/
│   ├── TECH_SPEC.md           # Full architecture and sequence specifications
│   ├── BUILD_SUMMARY.md       # Implementation and build summary
│   ├── PLAN.md                # Task tracking and execution checklist
│   └── PRD.md                 # Product requirements document
└── tests/
    ├── golden_eval_set_2026.json # 25 curated reference test cases
    ├── test_engine.py
    ├── test_engine_extended.py
    ├── test_ingest_mocked.py
    ├── test_app_isolated.py
    ├── test_telemetry.py
    ├── test_streaming.py
    ├── test_eval_runner.py
    └── test_ragas_eval.py
```
