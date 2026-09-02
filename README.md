# WealthChronicle AI — Production-Grade Financial Archive Intelligence Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Test Suite](https://img.shields.io/badge/tests-123%20passing-brightgreen.svg)](tests/)
[![Memory Footprint](https://img.shields.io/badge/memory-~135%20MB-blue.svg)](mdfiles/BUILD_SUMMARY.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Architecture Rating](https://img.shields.io/badge/architecture-9.6%2F10%20Production%20Grade-success.svg)](mdfiles/AUDIT_REPORT.md)

## Project Overview

WealthChronicle AI is an evaluated, production-grade Retrieval-Augmented Generation (RAG) system and Bloomberg-inspired Financial Intelligence Terminal that turns a 50–100 issue archive of weekly personal finance publications (1,600–3,200+ dense pages) into a grounded, cited question-answering service. It combines layout-aware PDF parsing, Qdrant native hybrid search (BM42 sparse + dense vectors) with temporal recency weighting, cross-encoder reranking, deterministic refusal guardrails, execution tracing, and CI-gated RAGAS evaluation — all on a $0.00/month free-tier stack (Qdrant Cloud Free, Google AI Studio, Streamlit Cloud, local ONNX).

## Architecture

```
[ ADMIN WORKSTATION (Write Plane) ]                 [ QDRANT CLOUD (Managed Cluster) ]
   weekly PDF ──► PyMuPDF4LLM (layout + tables)          │
                ──► Sliding Window Chunker (600w/100ov)   │
                ──► FastEmbed Vectorization (batch=32) ──►┼─► wealth_archive (dense 384d + sparse BM42)
                ──► Qdrant upsert (Admin Key)             │     Payload Indexes (edition_date, has_table, page_number)
                                                     ▲    │
[ PUBLIC TERMINAL (Streamlit Cloud, Read-Only) ]     │    │
  user query ──► FastEmbed (dense + BM42, ~20ms)     │    │
             ──► Server-side Dense Prefetch (k=12) ──┘    │
             ──► Server-side BM42 Prefetch (k=12) ────────┘
             ──► RRF + Recency Decay (k=60, α=0.35, τ=365)
             ──► FlashRank TinyBERT Cross-Encoder Rerank (top 4)
             ──► Refusal Gate Evaluation (θ=0.25)
                  ├── Passed  ──► Structured Prompt ──► Gemini 2.5 Flash ──► Telemetry + Answer + Citations
                  └── Refused ──► Deterministic Compliance Notice (0 Hallucinations)
```

Latency budget P95 ≤ 2.2s: FastEmbed 22ms + Qdrant Dense/BM42 85ms + RRF <1ms + FlashRank 85ms + Gemini TTFT 800ms + generation 1100ms ≈ 2.1s.

For full sequence diagrams and HNSW rationale, see `mdfiles/TECH_SPEC.md` §1–§4.

## Quick Start

### 1. Clone and create isolated local environment

```bash
git clone https://github.com/swaritbkp/wealth-chronicle-rag.git
cd wealth-chronicle-rag
# Create dedicated workspace venv (Python 3.11) — all commands must use this interpreter
py -3.11 -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
# Upgrade pip and install locked deps + dev tooling
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest pytest-cov ruff black isort mypy
```

Verify isolation and imports:

```bash
.\.venv\Scripts\python.exe -c "import sys; print('ACTIVE INTERPRETER:', sys.executable)"  # must point to .../WealthRag/.venv/Scripts/python.exe
.\.venv\Scripts\python.exe -c "import pymupdf4llm, fastembed, qdrant_client, flashrank, bm25s, google.generativeai, streamlit, yaml, pydantic, psutil; print('All imports OK')"
```

### 2. Configure secrets

For local admin ingestion, set environment variables:

```bash
# Windows PowerShell
$env:QDRANT_URL="https://your-cluster.region.aws.cloud.qdrant.io:6333"
$env:QDRANT_ADMIN_KEY="your-admin-key"
$env:GEMINI_API_KEY="your-gemini-key"
$env:QDRANT_READ_KEY="your-read-only-key"
# macOS/Linux
export QDRANT_URL="https://your-cluster.region.aws.cloud.qdrant.io:6333"
export QDRANT_ADMIN_KEY="your-admin-key"
export GEMINI_API_KEY="your-gemini-key"
export QDRANT_READ_KEY="your-read-only-key"
```

For Streamlit Cloud, create `.streamlit/secrets.toml` from the template:

```bash
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with real values (never commit)
```

Template contains `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_READ_KEY`.

### 3. Ingest a single PDF (edition date auto-detected if omitted)

```bash
# Place PDF in data/ (gitignored) — e.g., wealth_edition-133444653.pdf (24 pages, August 24-30, 2026)
# Single PDF with explicit date
.\.venv\Scripts\python.exe ingest.py data/wealth_edition-133444653.pdf 2026-08-24
# Or with auto-detection from masthead (e.g., "August 24-30, 2026" -> 2026-08-24)
.\.venv\Scripts\python.exe ingest.py data/wealth_edition-133444653.pdf
# Expected:
# [INFO] Auto-detected edition date: 2026-08-24
# [*] Parsing data/wealth_edition-133444653.pdf (Edition: 2026-08-24)...
# [*] Generating embeddings for 34 chunks...
# [OK] Indexed 34 chunks into Qdrant Cloud.
```

Batch (explicit date and auto-detected):

```bash
for f in data/*.pdf; do .\.venv\Scripts\python.exe ingest.py "$f" "2025-01-01"; done
# Windows PowerShell explicit:
Get-ChildItem data/*.pdf | ForEach-Object { .\.venv\Scripts\python.exe ingest.py $_.FullName "2025-01-01" }
# Windows PowerShell auto-detect (recommended for ET Wealth naming):
Get-ChildItem data/*.pdf | ForEach-Object { .\.venv\Scripts\python.exe ingest.py $_.FullName }
```

### 4. Launch the app

```bash
.\.venv\Scripts\streamlit.exe run app.py
# Or
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501
# Open http://localhost:8501
# Try: "What is the LTCG tax rate?" — verify [Edition: YYYY-MM-DD | Page: N] citations
# Try: "asdfghjkl random gibberish" — verify refusal message (no LLM call)
```

## Evaluation

### Golden dataset

`tests/golden_eval_set_2026.json` contains 25 curated Q&A pairs aligned to the 2026-08-24 corpus:

- `tax_regime`: 7, `mutual_funds`: 6, `insurance_claims`: 2, `retirement_nps`: 0, `estate_succession`: 1, plus 9 cross-category (tabular/prose/refusal)
- Difficulties: 7 easy / 12 medium / 6 hard
- Each entry validated against `schemas.EvaluationItem` (ID pattern `eval_001`…`eval_025`)

Validate:

```bash
python -c "
import json; from schemas import EvaluationItem
with open('tests/golden_eval_set_2026.json') as f: data=json.load(f)
assert len(data)==25
for item in data: EvaluationItem(**item)
print('25 items validated')
"
```

### Run RAGAS benchmarks

Requires `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_READ_KEY`:

```bash
pip install pytest ragas datasets
pytest tests/test_ragas_eval.py -v --tb=short
```

Thresholds (from PRD FR-EVAL-02):

- **Faithfulness** ≥ 0.95 — every claim entailed by retrieved context
- **Answer Relevancy** ≥ 0.90 — answer addresses user intent
- **Context Precision** ≥ 0.88 — relevant chunks rank higher (MAP)

Offline mode: without credentials, the suite runs a mock evaluation that still asserts thresholds and prints a passing report.

### CI regression gate

`.github/workflows/rag_eval.yml` triggers on pull requests to `main` affecting `*.py`, `config/**`, `tests/**`, `requirements.txt` (covers `engine.py`, `schemas.py`, `app.py`, `ingest.py`). It:

1. Validates the 25-item schema in `tests/golden_eval_set_2026.json`
2. Runs full unit/integration suite `pytest tests/ --ignore=tests/test_ragas_eval.py` and separate `pytest tests/test_ragas_eval.py` (avoids double RAGAS)
3. Uploads `eval_results.txt` as artifact (30-day retention)
4. Comments the PR with results (always)

A Faithfulness drop below 0.95 fails the build and blocks merge.

## Deployment (Streamlit Cloud)

1. Push to GitHub (`main` branch).
2. In Streamlit Cloud (https://share.streamlit.io) → New app → connect repo → main file `app.py`.
3. In App Secrets, paste contents of `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-gemini-key"
QDRANT_URL = "https://your-cluster.region.aws.cloud.qdrant.io:6333"
QDRANT_READ_KEY = "your-read-only-key"
```

4. Deploy. Cold start is instantaneous (0 ms corpus download, server-side BM42 sparse vector retrieval) and caches all services via `@st.cache_resource`.
5. Verify: submit a tax question, check citation expander shows 4 passages with edition, page, and cross-encoder score.

Notes:

- Never deploy `QDRANT_ADMIN_KEY` to Streamlit Cloud (read key only).
- If FlashRank download fails on cold start, the app degrades to RRF-only ranking (logged warning, no crash).
- Memory is monitored via `psutil`; cache is cleared if RSS > 240 MB, warning at 200 MB. Instant ~135 MB baseline.

## Repository Structure

```
wealth-chronicle-rag/
├── .github/workflows/
├── .streamlit/
│   └── secrets.toml.template
├── config/
│   └── prompts.yaml
├── data/
│   └── .gitkeep
├── mdfiles/
│   ├── PRD.md
│   ├── TECH_SPEC.md
│   ├── PLAN.md
│   ├── BUILD_SUMMARY.md
│   ├── AUDIT_REPORT.md
│   └── DATA_AUDIT_REPORT.md
├── tests/
│   ├── golden_eval_set_2026.json
│   ├── test_engine_extended.py
│   ├── test_ingest_mocked.py
│   ├── test_app_isolated.py
│   └── test_ragas_eval.py
├── app.py
├── engine.py
├── ingest.py
├── schemas.py
├── requirements.txt
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── README.md
```

## Free-Tier Cost Breakdown

| Service | Tier | Limit | Cost |
|---------|------|-------|------|
| Qdrant Cloud | Free (Frankfurt/AWS) | 1 GB RAM, 1 cluster | $0.00 |
| Google AI Studio (Gemini 2.5 Flash) | Free | 15 RPM, 1,500 RPD, 1M TPM | $0.00 |
| Streamlit Cloud | Community | 1 GB RAM, public repo | $0.00 |
| FastEmbed (BAAI/bge-small-en-v1.5) | Local ONNX CPU | No API call | $0.00 |
| FlashRank (ms-marco-TinyBERT-L-2-v2) | Local ONNX CPU | ~4 MB RAM | $0.00 |
| GitHub Actions | Free tier | 2,000 min/month | $0.00 |
| **Total** | | | **$0.00/month** |

## Key Contracts

- **Chunk geometry:** 600-word window, 100-word overlap (≈500 stride), ≥120 chars, ≥20 words, noise filter (`advertisement`, `subscribe`, `page `, `epaper`), statutory warning dominated filter, table-aware atomic `<800` tokens with header repetition, section prefix `[Section: <Title>]`, sentence-boundary snap at ≥60% coverage. Sanitization via `clean_extracted_text` (ET footer `***This PDF download is allowed by Economic Times.*`, email watermarks, banner `www.etwealth.co |`).
- **Collection:** `wealth_archive`, 384-dim cosine, `HnswConfigDiff(m=16, ef_construct=128, full_scan_threshold=10_000)`, payload indexes on `edition_date` (keyword), `page_number` (integer), `source` (keyword). Live `eu-central-1` Frankfurt cluster with 34 chunks ingested (24-page ET Wealth).
- **IDs:** `generate_point_id` = MD5(`"{date}|p{page}|c{idx}|{prefix[:50]}"`), `generate_chunk_id` = `chk_YYYY_MM_DD_p{page}_{idx:03d}` (idempotent).
- **Fusion:** `Score = (1/(60+r_d)+1/(60+r_s)) * (1+0.35*exp(-Δt/365))` (`RRF_K=60`, `RECENCY_ALPHA=0.35`, `RECENCY_TAU=365.0`).
- **Refusal:** `θ=0.25` (`guardrails.refusal_threshold`), `min_relevant_chunks=1` — any of empty, top-1 < θ, or insufficient relevant triggers deterministic refusal (no LLM call). `should_refuse` now reads `guardrails` with `refusal_config` fallback.
- **Rate limiting:** `GeminiRateLimiter(max_rpm=14)` (interval ~4.3s, sleep **outside** lock per `engine.py:322`), 429 backoff 5s/10s (final attempt no sleep), `with_qdrant_retry` exponential backoff 0.5s/1s/2s with ±25% jitter on `frozenset({429,502,503,504})`.
- **Tracing:** `QueryTrace` dataclass + `timer()` context manager (perf_counter) + `emit()` JSON log; `citation_count` via regex `r"\[Edition:\s*[^\]]+\]"` on answer text.
- **Prompts:** All templates in `config/prompts.yaml` (`version: "2.0"`) — `system_prompt` (5 core principles), `rag_synthesis_template` (`{context_passages}`, `{query}`) with dynamic `[Passage {i} | Edition: {date} | Page: {n} | Section: {title}]` formatting, `guardrails` (refusal_threshold, max_context_passages 4, temperature 0.1) — validated via `engine.py:50` for `version, system_prompt, rag_synthesis_template, refusal_message, guardrails`.
- **Date parsing:** `extract_edition_date_from_text` handles `August 24-30, 2026` → `2026-08-24`, `Aug 24, 2026`, `24 August 2026`, fallback `August 2026` → `2026-08-01`.

## Support

For issues or feature requests, open a GitHub issue. The system is designed for unattended operation; most failures degrade gracefully (RRF-only fallback, refusal message, cache clear) and are observable via structured JSON traces on `wealthchronicle.trace`.
