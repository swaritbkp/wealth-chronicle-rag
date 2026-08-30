# Next Session Handoff — WealthChronicle AI v1.0

**Date:** 2026-08-30  
**Branch:** `main` (up to date with `origin/main` via `f3cc5f7..df63a4a` + `bea0797`/`0658e02`)  
**Status:** 108 tests passing, ruff/mypy clean, live Frankfurt cluster with 34 chunks, prompts v2.0 active

---

## Active Environment

- **Interpreter:** `.\.venv\Scripts\python.exe` (`C:\Users\S\OneDrive\Desktop\WealthRag\.venv\Scripts\python.exe`, Python 3.11.15)
- **Activation:**
  ```powershell
  # Windows PowerShell
  .\.venv\Scripts\Activate.ps1
  # Or directly
  .\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
  # Must report: ...\WealthRag\.venv\Scripts\python.exe
  ```
- **Pytest:** `.\.venv\Scripts\pytest.exe tests/ -v`  or  `.\.venv\Scripts\python.exe -m pytest tests/ -v`
- **Linters:** `.\.venv\Scripts\python.exe -m ruff check . --fix` (`All checks passed!`), `.\.venv\Scripts\python.exe -m mypy schemas.py engine.py ingest.py app.py --ignore-missing-imports` (`Success`)
- **Dependencies:** `.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest pytest-cov ruff black isort mypy` already installed (pymupdf4llm 1.28.2, fastembed 0.8.0, qdrant-client 1.19.0, flashrank 0.2.10, bm25s 0.3.11, etc.)

---

## Live Qdrant Collection

- **Cluster Endpoint:** `https://955ef1b4-3a7d-4a9a-9aee-1fe9a2e17491.eu-central-1-0.aws.cloud.qdrant.io:6333` (AWS Frankfurt `eu-central-1`)
- **Collection:** `wealth_archive`
  - **Vectors:** 384-d, `Distance.COSINE`
  - **HNSW:** `m=16, ef_construct=128, full_scan_threshold=10_000`
  - **Optimizer:** `indexing_threshold=20_000, memmap_threshold=50_000`
  - **Payload Indexes:** `edition_date` (keyword), `page_number` (integer), `source` (keyword)
  - **Current Points:** 34 chunks from `data/wealth_edition-133444653.pdf` (24 pages, Edition `2026-08-24`, avg 79.8 words / 446.5 chars)
  - **Status:** `green` (verified via `QdrantClient(url, api_key=ADMIN).get_collection("wealth_archive")`)
- **Credentials:** Stored in `.streamlit/secrets.toml` (`QDRANT_URL`, `QDRANT_READ_KEY` len 176) and `.env` (`QDRANT_URL`, `QDRANT_ADMIN_KEY` len 176 `m`, `QDRANT_READ_KEY` len 176 `r`, `GEMINI_API_KEY`) — both gitignored (`_check-ignore` verified)
- **Verification:**
  ```powershell
  .\.venv\Scripts\python.exe -c "from qdrant_client import QdrantClient; import os; [exec(open('.env').read())]; client=QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_READ_KEY']); print(client.get_collection('wealth_archive').points_count)"
  # Expected: 34
  ```

---

## Current Test Count

- **108 passing tests** across `tests/` (`--cov` 45% overall, engine 74%, schemas 96%):
  - `tests/test_engine_extended.py` — 36 tests (RRF & recency math, FlashRank fallback, refusal boundaries θ=0.25, concurrency 20 threads @14/120 RPM, retry jitter frozenset)
  - `tests/test_ingest_mocked.py` — 54 tests (chunker edge, ID determinism 500 zero collisions, watermark sanitization `coolbilota@gmail.com`/`***This PDF download...`, table-aware atomic/<800 & oversized row-split, date parser `August 24-30, 2026` → `2026-08-24`, word_count ≥20)
  - `tests/test_app_isolated.py` — 18 tests (session cap 20, prompt v2.0 `WealthChronicle`/`RETRIEVED CONTEXT`, citation `:.4f`, memory guard every 10)
  - `tests/test_ragas_eval.py` — 1 test (mock Faithfulness 0.97 Relevancy 0.93 Precision 0.90)
- **Static checks:** `ruff` `All checks passed!` (select E,F, line-length 250), `black` `All done!`, `mypy` `Success: no issues found in 4 source files`
- **Benchmark:** `scripts/benchmark_latency.py` 300×384-d, 100 queries → P50 5.0ms P95 5.5ms dense 0.7ms sparse 0.4ms RRF 0.03ms, peak RSS 135.7 MB (<200 ✅, headroom 295ms)

---

## Immediate Next Tasks

1. **Add real Gemini API key to `.streamlit/secrets.toml`**
   - Current `GEMINI_API_KEY = "your-gemini-api-key-here"` (placeholder, 24 chars) — retrieval works, generation will fail until replaced
   - Get key from https://aistudio.google.com/apikey
   - Update both `.streamlit/secrets.toml` and `.env`:
     ```powershell
     # Edit .streamlit/secrets.toml:
     GEMINI_API_KEY = "AIza...your_real_key"
     # And .env:
     GEMINI_API_KEY=AIza...your_real_key
     ```
   - Verify via `.\.venv\Scripts\python.exe -c "import google.generativeai as genai; genai.configure(api_key=open('.streamlit/secrets.toml').read().split('\"')[1]); print('Gemini OK')"`

2. **Batch ingest historical magazine issues from `data/`**
   - Place additional PDFs in `data/` (gitignored, currently `wealth_edition-133444653.pdf` 24 pages + `test_issue.pdf` 2 pages)
   - Single: `.\.venv\Scripts\python.exe ingest.py data/wealth_edition-133444653.pdf 2026-08-24` or auto-detect `.\.venv\Scripts\python.exe ingest.py data/wealth_edition-133444653.pdf`
   - Batch: `Get-ChildItem data/*.pdf | ForEach-Object { .\.venv\Scripts\python.exe ingest.py $_.FullName }` (auto-detects `August 24-30, 2026` etc.)
   - Verify: `.\.venv\Scripts\python.exe -c "from qdrant_client import QdrantClient; import os; c=QdrantClient(url=open('.env').read().split('QDRANT_URL=')[1].split()[0], api_key=open('.env').read().split('QDRANT_READ_KEY=')[1].split()[0]); print(c.get_collection('wealth_archive').points_count)"`

3. **Start Streamlit UI and perform end-to-end user queries**
   ```powershell
   .\.venv\Scripts\streamlit.exe run app.py
   # Or headless
   .\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501
   # Open http://localhost:8501
   # Test queries:
   # - "What is the FAST-DS foreign asset disclosure scheme limit and tax rate?" → should cite [Edition: 2026-08-24 | Page: 16] with FAST-DS table, Top-1 Score ~0.98
   # - "Why is a 5 to 10 lakh health insurance cover no longer adequate?" → should cite [Edition: 2026-08-24 | Page: 2] Cover Story, Score ~0.87
   # Verify: refusal gate `should_refuse` correctly blocks hallucination, citation_count via regex `r"\[Edition:\s*[^\]]+\]"`, session cap 20, memory guard every 10
   ```

---

## Git State (for fresh session)

- **Branch:** `main` up to date with `origin/main` (last push `0658e02..df63a4a` plus `bea0797`/`0658e02` + audit `ecddc48`)
- **Last commits:**
  ```
  df63a4a feat(prompts): upgrade RAG prompt architecture to v2.0
  ecddc48 fix(audit): remediate 11 findings F-01 to F-11
  0658e02 fix(ingest): refine chunking for real 24-page PDF
  bea0797 feat(ingest): upgrade pipeline with watermark sanitization and table-aware chunking
  ```
- **Working tree:** clean (`git status` → `nothing to commit`) except ignored `.venv/`, `data/*.pdf`, `.env`, `.streamlit/secrets.toml`, `.mypy_cache`, `htmlcov`
- **Remote:** `origin https://github.com/swaritbkp/wealth-chronicle-rag.git (fetch/push)` — public, workflow `.github/workflows/rag_eval.yml` (8 steps, `pytest tests/ --ignore=tests/test_ragas_eval.py` + RAGAS) visible

---

## Quick Resume Commands

```powershell
# 1. Activate and verify
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\pytest.exe tests/ -v  # expect 108 passed

# 2. Check live collection
.\.venv\Scripts\python.exe -c "from qdrant_client import QdrantClient; c=QdrantClient(url='https://955ef1b4-3a7d-4a9a-9aee-1fe9a2e17491.eu-central-1-0.aws.cloud.qdrant.io:6333', api_key=open('.env').read().split('QDRANT_ADMIN_KEY=')[1].split()[0]); print(c.get_collection('wealth_archive').points_count)"

# 3. Ingest new issue (auto-detect date)
.\.venv\Scripts\python.exe ingest.py data/new_issue.pdf

# 4. Launch UI
.\.venv\Scripts\streamlit.exe run app.py
```

*End of handoff — ready for fresh session with isolated .venv, live Frankfurt cluster (34 chunks), and 108-test green suite.*
