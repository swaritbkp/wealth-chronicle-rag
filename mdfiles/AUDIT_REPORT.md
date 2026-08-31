# AUDIT REPORT — WealthChronicle AI v1.0

**Audit Type:** Architectural, Compliance & Security  
**Auditor Role:** Principal AI Architect & Lead Code Auditor  
**Audit Date:** 2026-08-30  
**Codebase Snapshot:** Current `main` branch  
**Specification Baseline:** [PRD.md](file:///c:/Users/S/OneDrive/Desktop/WealthRag/PRD.md) · [TECH_SPEC.md](file:///c:/Users/S/OneDrive/Desktop/WealthRag/TECH_SPEC.md) · [PLAN.md](file:///c:/Users/S/OneDrive/Desktop/WealthRag/PLAN.md) · [BUILD_SUMMARY.md](file:///c:/Users/S/OneDrive/Desktop/WealthRag/BUILD_SUMMARY.md)

---

## 1. Executive Verdict

### 🟡 CHANGES REQUIRED — Conditionally Production-Ready

The codebase demonstrates strong engineering fundamentals: all core mathematical invariants (RRF, recency decay, refusal thresholds) are correctly implemented, the test suite provides exhaustive coverage of edge cases, and the security posture is solid. However, **11 findings** across 4 severity levels require attention before unrestricted production deployment.

| Severity | Count | Summary |
|----------|-------|---------|
| 🔴 **Critical** | 2 | Deprecated API usage (`datetime.utcnow`), rate limiter `time.sleep` blocks under GIL contention |
| 🟠 **High** | 3 | FlashRank `/tmp/models` cache path on Windows, CI double-runs RAGAS, `safe_generate` swallows non-429 errors on 3rd attempt |
| 🟡 **Medium** | 4 | `_dense_search` closure re-defined every query, missing `engine.py`/`schemas.py` path triggers in CI, `QdrantRetryConfig.RETRYABLE_STATUS_CODES` mutable class-level `set`, `article_title` extraction never wired in `ingest_pdf` |
| 🟢 **Low** | 2 | `confloat`/`constr` deprecation warnings in Pydantic v3+, `citation_count` tracks passage count not actual citation format count |

---

## 2. Contract Compliance Matrix

| Phase | Target | Files | Spec Compliance | Test Coverage | Verdict |
|-------|--------|-------|:---:|:---:|:---:|
| **Phase 1** — Foundation & Schemas | `schemas.py`, `requirements.txt`, `config/prompts.yaml`, `.gitignore`, `.streamlit/secrets.toml.template` | ✅ All present | ✅ 7/7 Pydantic models match TECH_SPEC §2.2 | ✅ Validated via `test_ingest_mocked.py` & `test_engine_extended.py` | ✅ **PASS** |
| **Phase 2** — Core Engine | `engine.py` | ✅ All 8 tasks implemented | ✅ RRF k=60, α=0.35, τ=365, θ=0.25 all correct | ✅ 36 tests across 5 test classes | ✅ **PASS** |
| **Phase 3** — Ingestion | `ingest.py` | ✅ All 7 tasks implemented + enhanced with watermark/table-aware upgrades | ✅ HNSW m=16/ef_construct=128, 384-dim cosine | ✅ 35+ tests covering chunker, IDs, sanitization, tables | ✅ **PASS** |
| **Phase 4** — Streamlit App | `app.py` | ✅ All 5 tasks implemented | ✅ Hybrid retrieval, refusal gate, citation expander, session pruning | 🟡 `test_app_isolated.py` exists but limited to unit mocks | 🟡 **PASS** (with notes) |
| **Phase 5** — Eval & CI | `tests/golden_eval_set.json`, `test_ragas_eval.py`, `.github/workflows/rag_eval.yml`, `README.md` | ✅ All present | ✅ 50-item golden set, correct category/difficulty distribution | 🟡 CI workflow has redundant RAGAS step | 🟡 **PASS** (with notes) |

---

## 3. Detailed Audit Findings

---

### 🔴 FINDING F-01: `datetime.utcnow()` Deprecated Since Python 3.12

> [!CAUTION]
> `datetime.utcnow()` is deprecated in Python 3.12+ and will be removed in a future release. It returns a naive datetime that is ambiguous about its timezone.

**Locations:**
- [schemas.py:58](file:///c:/Users/S/OneDrive/Desktop/WealthRag/schemas.py#L58) — `default_factory=datetime.utcnow` in `ChunkPayload.ingested_at`
- [app.py:218](file:///c:/Users/S/OneDrive/Desktop/WealthRag/app.py#L218) — `datetime.utcnow().isoformat() + "Z"` in trace timestamp
- [app.py:220](file:///c:/Users/S/OneDrive/Desktop/WealthRag/app.py#L220) — `datetime.utcnow()` for total_ms calculation

**Impact:** DeprecationWarning in Python 3.12+; will error in Python 3.14+. Naive UTC timestamps are ambiguous when serialized to JSON.

**Recommendation:**
```python
# Replace all occurrences of:
datetime.utcnow()
# With:
from datetime import datetime, timezone
datetime.now(timezone.utc)
```

---

### 🔴 FINDING F-02: `GeminiRateLimiter.acquire()` Calls `time.sleep()` Inside `threading.Lock()`

**Location:** [engine.py:323-330](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L323-L330)

```python
def acquire(self) -> None:
    with self._lock:           # Lock acquired
        ...
        if elapsed < self.interval:
            time.sleep(sleep_time)  # Sleeping while holding lock!
        self.last_request_time = time.monotonic()
```

**Impact:** Under concurrent Streamlit sessions (multiple threads), all threads contend for the same lock. If Thread A sleeps for 4.3s while holding the lock, Threads B, C, D are **blocked from even entering `acquire()`** — they can't even check timing. With 3 concurrent sessions at 14 RPM, effective throughput degrades to serialized single-thread behavior. Under GIL + lock contention, this creates **priority inversion**: a thread that could proceed immediately waits for another thread's sleep to complete.

**Recommendation:** Release the lock before sleeping:
```python
def acquire(self) -> None:
    with self._lock:
        now = time.monotonic()
        elapsed = now - self.last_request_time
        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
        else:
            sleep_time = 0
        self.last_request_time = now + max(sleep_time, 0)
    # Sleep OUTSIDE the lock
    if sleep_time > 0:
        time.sleep(sleep_time)
```

---

### 🟠 FINDING F-03: FlashRank Cache Dir `/tmp/models` Incompatible with Windows

**Location:** [app.py:73](file:///c:/Users/S/OneDrive/Desktop/WealthRag/app.py#L73)

```python
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/models")
```

**Impact:** `/tmp/models` is a Unix-only path. On Windows (the documented user OS), this either creates `C:\tmp\models\` (unusual, potentially permission-blocked) or fails silently with FlashRank's internal fallback. The fallback path is caught by the `try/except`, so the app gracefully degrades to RRF-only — but this means **FlashRank reranking is silently disabled on Windows deployments**, degrading answer quality.

**Recommendation:**
```python
import tempfile, os
cache_dir = os.path.join(tempfile.gettempdir(), "flashrank_models")
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=cache_dir)
```

---

### 🟠 FINDING F-04: CI Workflow Runs RAGAS Tests Twice

**Location:** [rag_eval.yml:52-68](file:///c:/Users/S/OneDrive/Desktop/WealthRag/.github/workflows/rag_eval.yml#L52-L68)

```yaml
- name: Run Full Test Suite (Expanded Harness)    # Runs ALL tests/ including test_ragas_eval.py
  run: pytest tests/ -v ...

- name: Run RAGAS Faithfulness Evaluation Suite    # Runs test_ragas_eval.py AGAIN
  run: pytest tests/test_ragas_eval.py -v ...
```

**Impact:** `test_ragas_eval.py` is executed twice — once as part of the full `tests/` glob and once explicitly. Each run makes 50 Gemini API calls (100 total per CI run), doubling token consumption and risking RPD quota exhaustion on the free tier (1,500 RPD limit).

**Recommendation:** Either exclude RAGAS from the full suite run:
```yaml
- name: Run Unit Tests
  run: pytest tests/ -v --ignore=tests/test_ragas_eval.py
- name: Run RAGAS Suite
  run: pytest tests/test_ragas_eval.py -v
```
Or remove the second explicit step entirely.

---

### 🟠 FINDING F-05: `safe_generate` Third-Attempt 429 Handler Sleeps Then Returns Fallback — But Also Enters a 4th Iteration

**Location:** [engine.py:337-369](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L337-L369)

```python
for attempt in range(3):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        ...
        if is_429 and attempt < 2:     # Attempts 0, 1 → retry
            ...
            continue
        elif is_429:                    # Attempt 2 → sleep THEN return fallback
            wait = 2**attempt * 5      # 2^2 * 5 = 20s — correct
            time.sleep(wait)           # Sleeps 20s...
            return "The service is temporarily busy..."  # ...then returns
        else:
            raise                      # Non-429 errors are raised
```

**Issues:**
1. On the 3rd attempt (attempt=2), the code sleeps for 20s **before returning the fallback message** — a useless delay since no retry follows. The user waits 20s for a "try again" message.
2. The `for` loop's natural exhaustion path (line 369) returns the fallback string redundantly, but this is only reachable if `continue` skips it — which it does only for 429 errors with `attempt < 2`. This path is dead code for 429 cases but actually **important**: if a non-429 exception is swallowed (which shouldn't happen due to `raise`), the fallback fires silently.

**Recommendation:** On the final 429 attempt, return immediately without sleeping:
```python
elif is_429:
    logging.warning(f"Gemini 429 — all 3 attempts exhausted")
    return "The service is temporarily busy. Please try again in a few moments."
```

---

### 🟡 FINDING F-06: `_dense_search()` Closure Re-Defined on Every Query

**Location:** [app.py:232-238](file:///c:/Users/S/OneDrive/Desktop/WealthRag/app.py#L232-L238)

```python
@with_qdrant_retry
def _dense_search():
    return qdrant_client.search(...)
```

This `@with_qdrant_retry`-decorated function is **defined inside the query handler** (within the `if user_query:` block). Every user query re-defines and re-decorates the closure. While functionally correct, this is:
1. **Inefficient** — unnecessary object creation per query.
2. **Semantically misleading** — `@with_qdrant_retry` wraps a fresh function each time, losing any cached decorator state.

**Recommendation:** Extract to module-level helper or use the decorator inline:
```python
# Module level
@with_qdrant_retry
def _dense_search(client, vector, collection, limit):
    return client.search(collection_name=collection, query_vector=vector, limit=limit)
```

---

### 🟡 FINDING F-07: CI Path Triggers Miss `engine.py` and `schemas.py`

**Location:** [rag_eval.yml:7-12](file:///c:/Users/S/OneDrive/Desktop/WealthRag/.github/workflows/rag_eval.yml#L7-L12)

```yaml
paths:
  - 'app.py'
  - 'ingest.py'
  - 'config/**'
  - 'tests/**'
  - 'requirements.txt'
```

**Missing:** Changes to `engine.py` or `schemas.py` — which contain the core math (RRF, refusal), all Pydantic contracts, and the rate limiter — **will not trigger the CI regression gate**. A developer could break `should_refuse()` or the RRF formula and merge without CI catching it.

**Recommendation:**
```yaml
paths:
  - '*.py'           # Catches engine.py, schemas.py, app.py, ingest.py
  - 'config/**'
  - 'tests/**'
  - 'requirements.txt'
```

---

### 🟡 FINDING F-08: `QdrantRetryConfig.RETRYABLE_STATUS_CODES` Is a Mutable Class-Level `set`

**Location:** [engine.py:382](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L382)

```python
class QdrantRetryConfig:
    RETRYABLE_STATUS_CODES: set = {429, 502, 503, 504}
```

**Impact:** Mutable class attribute. Any code calling `QdrantRetryConfig.RETRYABLE_STATUS_CODES.add(500)` would globally mutate the retry behavior for all decorated functions. Low probability in current codebase but violates defensive coding principles.

**Recommendation:** Use `frozenset`:
```python
RETRYABLE_STATUS_CODES: frozenset = frozenset({429, 502, 503, 504})
```

---

### 🟡 FINDING F-09: `article_title` Extraction Never Wired in `ingest_pdf`

**Location:** [ingest.py:822-829](file:///c:/Users/S/OneDrive/Desktop/WealthRag/ingest.py#L822-L829)

```python
payload = ChunkPayload(
    chunk_id=chunk_id,
    edition_date=edition_date,
    page_number=page_number,
    text=chunk_text,
    char_count=char_count,
    word_count=word_count,
    # article_title is never set!
)
```

**Impact:** `article_title` in `ChunkPayload` defaults to `None` for **every chunk**, even though `_extract_section_title()` already exists in `ingest.py` and is called during chunking. The section title is added as a `[Section: ...]` prefix in the chunk text but never populated as structured metadata.

**Consequence:** The UI citation expander in `app.py` displays `article_title=None` for every passage, losing structured metadata that could improve the UI. The data is present in the text but not in the payload field.

**Recommendation:**
```python
section_title = _extract_section_title(text)  # already called within sliding_window_chunk
# Pass to ChunkPayload:
payload = ChunkPayload(
    ...,
    article_title=section_title,
)
```

---

### 🟢 FINDING F-10: `confloat` / `constr` Deprecation in Pydantic v3+

**Location:** [schemas.py:6](file:///c:/Users/S/OneDrive/Desktop/WealthRag/schemas.py#L6)

```python
from pydantic import BaseModel, Field, confloat, constr, field_validator
```

**Impact:** `confloat` and `constr` are deprecated in Pydantic v2 and will be removed in v3. Current `pydantic>=2.7.0` constraint means this works today but will break on future upgrades.

**Recommendation:** Replace with `Annotated` types:
```python
from typing import Annotated
from pydantic import Field
# Instead of confloat(ge=0.0, le=1.0):
Annotated[float, Field(ge=0.0, le=1.0)]
```

---

### 🟢 FINDING F-11: `citation_count` Tracks Passage Count, Not Actual Citation Markers

**Location:** [app.py:351](file:///c:/Users/S/OneDrive/Desktop/WealthRag/app.py#L351)

```python
trace.citation_count = len(reranked_sorted)
```

This sets `citation_count` to the number of **reranked passages** (always 4), not the number of actual `[Edition: ...]` citation markers in the generated answer text. For observability accuracy, this should count regex matches in the answer.

**Impact:** Misleading trace data. Low severity since it's an observability metric, not a functional issue.

---

## 4. Audit Section A: Mathematical Invariants & Retrieval Integrity

### A.1 — RRF & Recency Decay ✅ VERIFIED

| Invariant | Spec Value | Code Value | Location | Status |
|-----------|-----------|------------|----------|--------|
| RRF constant $k$ | 60 | `RRF_K = 60` | [engine.py:119](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L119) | ✅ |
| Recency $\alpha$ | 0.35 | `RECENCY_ALPHA = 0.35` | [engine.py:120](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L120) | ✅ |
| Recency $\tau$ | 365.0 days | `RECENCY_TAU = 365.0` | [engine.py:121](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L121) | ✅ |
| RRF formula | $\frac{1}{k+r_d} + \frac{1}{k+r_s}$ | Lines 162 | [engine.py:162](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L162) | ✅ |
| Recency formula | $1.0 + \alpha \cdot e^{-\Delta t / \tau}$ | Line 168 | [engine.py:168](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L168) | ✅ |
| Missing rank → `float("inf")` | Yes | Lines 159-160 | [engine.py:159-160](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L159-L160) | ✅ |
| Missing payload → recency=1.0 | Yes | Lines 169-170 | [engine.py:169-170](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L169-L170) | ✅ |
| Sort descending by final_score | Yes | Line 183 | [engine.py:183](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L183) | ✅ |
| Top-N capping | Yes | Line 184 | [engine.py:184](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L184) | ✅ |

**Test Coverage:** `TestRRFRecencyMath` — 11 tests covering Δt=0 (recency=1.35 ✅), Δt=365 (recency≈1.129 ✅), Δt=1000 (recency≈1.022 ✅), negative Δt (future dates ✅), missing payloads, dense-only/sparse-only/both scenarios, tie-breaking, and top-N capping.

### A.2 — Deterministic Refusal ($\theta = 0.25$) ✅ VERIFIED

| Condition | Spec | Code | Status |
|-----------|------|------|--------|
| Empty reranked list → refuse | Yes | [engine.py:290-291](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L290-L291) | ✅ |
| Top-1 score < θ → refuse | Yes | [engine.py:293-294](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L293-L294) | ✅ |
| Relevant count < min_chunks → refuse | Yes | [engine.py:297-299](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L297-L299) | ✅ |
| Default θ from config | 0.25 | [engine.py:286](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L286) | ✅ |
| Refusal skips LLM call | Yes | [app.py:330-336](file:///c:/Users/S/OneDrive/Desktop/WealthRag/app.py#L330-L336) | ✅ |

**Test Coverage:** `TestRefusalThresholdBoundaries` — 10 boundary tests including exact-at-threshold (0.25 → proceed), just-below (0.2499 → refuse), zero score, empty list, custom min_chunks=2, and missing config defaults.

### A.3 — FlashRank Resilience ✅ VERIFIED

FlashRank initialization is wrapped in `try/except` at [app.py:70-77](file:///c:/Users/S/OneDrive/Desktop/WealthRag/app.py#L70-L77). On failure:
- `ranker` is set to `None`, `ranker_available` to `False`
- `rerank_candidates()` checks `ranker is None` at [engine.py:210](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L210) and falls back to RRF ordering with `cross_encoder_score=0.0`
- App continues normally — no crash

**Test Coverage:** `TestFlashRankFallback` — 5 tests confirming fallback order preservation, missing payload handling, exception propagation, and empty candidates.

---

## 5. Audit Section B: Ingestion Plane & Layout Robustness

### B.1 — Table Integrity ✅ VERIFIED

**Implementation:** [ingest.py:246-372](file:///c:/Users/S/OneDrive/Desktop/WealthRag/ingest.py#L246-L372)

- `_split_table_blocks()` isolates markdown tables from prose using pipe-delimiter detection
- `_chunk_table_atomic()` keeps tables < 800 tokens as single atomic chunks
- Oversized tables (≥ 800 tokens) are split row-by-row **with header repetition** per TECH_SPEC §3.2

**Test Coverage:** `TestTableAwareChunking` — 4 tests:
- Small table atomic preservation ✅
- Table not split across boundaries ✅
- Oversized table split with header repetition ✅
- Section prefix added to table chunks ✅

### B.2 — Noise & Watermark Elimination ✅ VERIFIED

**Implementation:** [ingest.py:93-218](file:///c:/Users/S/OneDrive/Desktop/WealthRag/ingest.py#L93-L218)

| Pattern | Regex | Strips |
|---------|-------|--------|
| ET footer | `***This PDF download is allowed by Economic Times.*` | ✅ Full line removal |
| Email watermarks | `[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+` | ✅ Both inline and line-level |
| Banner headers | `www\.etwealth\.co\s*\|.*` | ✅ Case-insensitive |
| Statutory warnings | "Mutual Fund investments are subject to market risks" | ✅ Dominated-chunk filter |
| Noise prefixes | `advertisement`, `subscribe`, `page `, `epaper` | ✅ Chunk-level filter |

**Test Coverage:** `TestWatermarkSanitization` — 11 tests covering footer removal, case-insensitivity, multiple email formats, banner stripping, multiple simultaneous watermarks, editorial preservation, statutory warning filtering, and empty string handling.

### B.3 — Idempotency ✅ VERIFIED

- `generate_point_id()` uses deterministic MD5 hashing of `"{date}|p{page}|c{chunk}|{text[:50]}"` — [ingest.py:558-559](file:///c:/Users/S/OneDrive/Desktop/WealthRag/ingest.py#L558-L559)
- Re-ingesting produces identical point IDs → Qdrant `upsert()` overwrites, no duplicates
- **Test Coverage:** `TestDeterministicIDInvariants` — 9 tests including determinism, format validation, collision testing across 500 inputs, prefix truncation at 50 chars, and cross-call idempotency verification.

---

## 6. Audit Section C: Security, Secrets & Concurrency

### C.1 — Secrets Isolation ✅ VERIFIED

| Secret | Gitignored | In Code | Exposure Risk |
|--------|:---:|---------|:---:|
| `.streamlit/secrets.toml` | ✅ [.gitignore:21](file:///c:/Users/S/OneDrive/Desktop/WealthRag/.gitignore#L21) | Read via `st.secrets[]` only | **None** |
| `.env` | ✅ [.gitignore:22](file:///c:/Users/S/OneDrive/Desktop/WealthRag/.gitignore#L22) | Read via `os.environ.get()` only | **None** |
| `.env.local` | ✅ [.gitignore:23](file:///c:/Users/S/OneDrive/Desktop/WealthRag/.gitignore#L23) | Not referenced | **None** |
| `QDRANT_ADMIN_KEY` | N/A (env var only) | Only in `ingest.py` (admin plane) | **None** |
| CI secrets | Via `${{ secrets.* }}` | Never echoed/logged | **None** |

**Template file:** `.streamlit/secrets.toml.template` is committed (placeholder values only) ✅

**Trace logging check:** `QueryTrace.emit()` serializes only query text and timing metadata — no secrets, no API keys, no payloads. ✅

### C.2 — Rate Limiting & Thread Safety 🟡 PARTIALLY VERIFIED

- Thread-safe lock exists: `self._lock = threading.Lock()` ✅
- RPM enforcement: `60.0 / 14 ≈ 4.29s` interval between requests ✅
- **Issue:** Sleep occurs **inside** the lock (see F-02), causing lock contention under concurrent sessions

**Test Coverage:** `TestGeminiRateLimiterConcurrency` — 4 tests:
- Throughput capping under 5 concurrent threads ✅
- 14 RPM interval computation ✅
- No deadlock under 20 threads at 120 RPM ✅
- Lock attribute verification ✅

### C.3 — Memory & Leak Guards ✅ VERIFIED

- RSS monitoring: `psutil.Process().memory_info().rss` at [engine.py:512-513](file:///c:/Users/S/OneDrive/Desktop/WealthRag/engine.py#L512-L513)
- Warning at 200 MB, critical cache-clear at 240 MB ✅
- Periodic check every 10 queries: [app.py:203](file:///c:/Users/S/OneDrive/Desktop/WealthRag/app.py#L203) ✅
- Session pruning at 20 messages: [app.py:195-198](file:///c:/Users/S/OneDrive/Desktop/WealthRag/app.py#L195-L198) ✅
- `st.cache_resource.clear()` guarded by `try/except` in non-Streamlit context ✅

---

## 7. Test Infrastructure Assessment

### Coverage Summary

| Test File | Tests | Lines | What It Covers | Mock vs Live |
|-----------|:---:|:---:|----------------|:---:|
| [test_engine_extended.py](file:///c:/Users/S/OneDrive/Desktop/WealthRag/tests/test_engine_extended.py) | 36 | 655 | RRF math, recency decay, FlashRank fallback, refusal boundaries, rate limiter concurrency, retry jitter | All mocked (FlashRank model optional) |
| [test_ingest_mocked.py](file:///c:/Users/S/OneDrive/Desktop/WealthRag/tests/test_ingest_mocked.py) | 35+ | 444 | Chunker edge cases, ID determinism, watermark sanitization, table-aware chunking, date extraction | All mocked |
| [test_app_isolated.py](file:///c:/Users/S/OneDrive/Desktop/WealthRag/tests/test_app_isolated.py) | ~10 | 254 | App service init mocks, session state, query flow stubs | All mocked |
| [test_ragas_eval.py](file:///c:/Users/S/OneDrive/Desktop/WealthRag/tests/test_ragas_eval.py) | 1 | 200 | Full RAG pipeline evaluation against golden set | **Live** (Gemini + Qdrant) |

### Golden Evaluation Set ✅

```
Total: 50 items
Categories: tax_regime=15, mutual_funds=12, insurance_claims=10, retirement_nps=8, estate_succession=5
Difficulties: easy=15, medium=25, hard=10
```

All counts match TECH_SPEC §6.1.2 exactly. ✅

---

## 8. Actionable Recommendations

### Priority 1 (Fix Before Deployment)

| ID | Finding | Fix | Effort |
|----|---------|-----|--------|
| F-02 | Rate limiter sleep inside lock | Move `time.sleep()` outside `with self._lock:` block | 10 min |
| F-07 | CI missing `engine.py`/`schemas.py` triggers | Change path filter to `'*.py'` | 2 min |

### Priority 2 (Fix Within Sprint)

| ID | Finding | Fix | Effort |
|----|---------|-----|--------|
| F-01 | `datetime.utcnow()` deprecated | Replace with `datetime.now(timezone.utc)` (3 locations) | 15 min |
| F-03 | `/tmp/models` on Windows | Use `tempfile.gettempdir()` | 5 min |
| F-04 | CI double RAGAS run | Add `--ignore=tests/test_ragas_eval.py` to full suite step | 2 min |
| F-05 | `safe_generate` 20s sleep before fallback | Remove sleep on final attempt | 5 min |

### Priority 3 (Improve Quality)

| ID | Finding | Fix | Effort |
|----|---------|-----|--------|
| F-06 | `_dense_search` re-defined per query | Extract to module-level function | 10 min |
| F-08 | Mutable `set` on config class | Change to `frozenset` | 2 min |
| F-09 | `article_title` never populated | Wire `_extract_section_title()` into `ingest_pdf()` | 20 min |
| F-10 | `confloat`/`constr` deprecation | Migrate to `Annotated[float, Field(...)]` | 30 min |
| F-11 | `citation_count` inaccurate | Count `[Edition:` regex matches in answer | 10 min |

---

## 9. Architecture Strengths

1. **Clean separation of concerns** — `schemas.py` (data contracts), `engine.py` (business logic), `ingest.py` (admin plane), `app.py` (public plane). No circular imports.
2. **Defensive fallback chains** — FlashRank → RRF-only, Qdrant timeout → exponential backoff, Gemini 429 → rate limit + retry, empty retrieval → deterministic refusal. Each failure mode degrades gracefully.
3. **Deterministic idempotency** — MD5-based point IDs ensure re-ingestion is safe. Qdrant `upsert` semantics prevent duplicates.
4. **Table-aware chunking** — Goes beyond the original spec with atomic table preservation and header repetition for oversized tables. This is a significant quality improvement.
5. **Comprehensive watermark stripping** — Multiple regex patterns catch ET Wealth-specific boilerplate, emails, and statutory warnings. Editorial content is preserved.
6. **Observability by design** — `QueryTrace` dataclass with `timer()` context manager instruments every pipeline stage. JSON-structured logs are Langfuse-compatible.
7. **Exhaustive test suite** — 80+ tests covering mathematical boundary conditions, concurrency, idempotency, and sanitization edge cases. No test uses `pytest.skip` as a way to avoid testing.

---

*End of Audit Report — WealthChronicle AI v1.0*
