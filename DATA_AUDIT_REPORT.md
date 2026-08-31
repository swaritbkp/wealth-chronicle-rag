# WealthChronicle AI — Data Audit Report

**Date:** 2026-08-30  
**Auditor:** Elite AI Systems Architect  
**Scope:** Ingested corpus (`wealth_archive` Qdrant collection), golden evaluation set (`tests/golden_eval_set.json`)

---

## 1. Corpus Health Matrix

### 1.1 Collection Overview
| Metric | Value |
|--------|-------|
| Collection Name | `wealth_archive` |
| Qdrant Status | `green` |
| Total Points | 34 |
| Vector Dimension | 384 (BAAI/bge-small-en-v1.5) |
| Distance Metric | Cosine |
| Edition Dates | **Single edition: 2026-08-24** |
| Source PDF | `wealth_edition-133444653.pdf` (24 pages, ET Wealth) |

### 1.2 Schema Integrity
All 34 points conform to `ChunkPayload` schema with **zero missing required fields** for 8/9 fields. One minor issue:

| Field | Missing/Empty | Severity |
|-------|---------------|----------|
| `chunk_id` | 0/34 | ✅ |
| `edition_date` | 0/34 | ✅ |
| `page_number` | 0/34 | ✅ |
| `article_title` | **3/34** | ⚠️ Minor |
| `text` | 0/34 | ✅ |
| `char_count` | 0/34 | ✅ |
| `word_count` | 0/34 | ✅ |
| `source` | 0/34 | ✅ |
| `ingested_at` | 0/34 | ✅ |

**Note:** 15/34 chunks have empty or heading-only `article_title` (e.g., `## Guest Column: Status matching`). While `ChunkPayload` allows nullable titles, meaningful titles improve retrieval context. These correspond to prose chunks where the section title extraction found markdown headings but the cleaner didn't strip them.

### 1.3 Chunk Geometry & Distribution
| Type | Count | Word Count Range | Mean Words | Std Dev |
|------|-------|------------------|------------|---------|
| **Prose** | 24 | 20 – 199 | 66.7 | 52.2 |
| **Tabular** | 10 | 57 – 206 | 111.3 | 47.2 |
| **Total** | 34 | 20 – 206 | 79.8 | 54.1 |

**Assessment:** Chunk sizes are well within the 500–800 token target (≈375–600 words). Tabular chunks are denser as expected.

### 1.4 Tabular Preservation Integrity
All 10 detected tabular chunks maintain **intact pipe-delimited structure** with header separators (`|---|---|`):

| Chunk ID | Page | Table Subject | Header Sep | Table Lines | Pipes |
|----------|------|---------------|------------|-------------|-------|
| `chk_2026_08_24_p17_001` | 17 | Stock rankings (Momentum) | ✅ | 7 | 63 |
| `chk_2026_08_24_p16_001` | 16 | FAST-DS tax brackets | ✅ | 5 | 25 |
| `chk_2026_08_24_p18_001` | 18 | Mutual fund returns (3M–5Y) | ✅ | 3 | 49 |
| `chk_2026_08_24_p19_000` | 19 | Liquid fund returns | ✅ | 1 | 42 |
| `chk_2026_08_24_p3_000` | 3 | Health claim averages by disease | ✅ | 4 | 21 |
| `chk_2026_08_24_p4_001` | 4 | Momentum fund names + returns | ✅ | 2 | 30 |
| `chk_2026_08_24_p20_000` | 20 | 1-year Bank FD rates | ✅ | 17 | 68 |
| `chk_2026_08_24_p15_001` | 15 | 10-year sector matrix (18 sectors × 10 years) | ✅ | 1 | 78 |
| `chk_2026_08_24_p3_001` | 3 | *Header separator only (row split artifact)* | ✅ | 1 | 30 |
| `chk_2026_08_24_p10_001` | 10 | 8-quarter sector momentum matrix | ✅ | 1 | 100 |

**Critical Finding:** Two chunks (`p3_001`, `p10_001`) appear to be **row-split artifacts** — they contain only header separators or truncated table rows. This suggests the `_chunk_table_atomic` oversized splitting logic may have created degenerate fragments. The sector matrix on page 10 (18 sectors × 8 quarters) exceeds 800 tokens and was split, but the split appears to have lost data rows.

### 1.5 Watermark/Boilerplate Leakage Scan
| Pattern | Chunks with Matches | Status |
|---------|---------------------|--------|
| Email watermarks | 0 | ✅ Clean |
| `download is allowed` footer | 0 | ✅ Clean |
| `epaper` | 0 | ✅ Clean |
| `www.etwealth.co` banner | 0 | ✅ Clean |
| `subscriber` | 0 | ✅ Clean |
| **Statutory warning** | **2 chunks** | ⚠️ **Residual** |

**Residual Statutory Warning Leakage:**
- `chk_2026_08_24_p5_000` — Page 5 (ad page)
- `chk_2026_08_24_p22_000` — Page 22 (ad page)

These chunks survived the `_is_statutory_warning_dominated` filter because they contain *some* editorial text alongside the warning, but the warning phrase is still present. Recommendation: Strengthen the filter to strip the warning phrase entirely from retained chunks, or discard chunks where warning >50% of content.

---

## 2. Golden Benchmark Assessment

### 2.1 Taxonomy & Difficulty Compliance
| Category | Expected | Actual | Status |
|----------|----------|--------|--------|
| `tax_regime` | 15 | 15 | ✅ |
| `mutual_funds` | 12 | 12 | ✅ |
| `insurance_claims` | 10 | 10 | ✅ |
| `retirement_nps` | 8 | 8 | ✅ |
| `estate_succession` | 5 | 5 | ✅ |
| **Total** | **50** | **50** | ✅ |

| Difficulty | Expected | Actual | Status |
|------------|----------|--------|--------|
| `easy` | 15 | 15 | ✅ |
| `medium` | 25 | 25 | ✅ |
| `hard` | 10 | 10 | ✅ |

**All structural requirements from TECH_SPEC §6.1.2 are met.**

### 2.2 Critical Alignment Failure: Corpus-Benchmark Mismatch
**❌ FUNDAMENTAL ISSUE:** The golden evaluation set references **22 distinct edition dates from 2025**, but the ingested corpus contains **only ONE edition: 2026-08-24**.

| Metric | Golden Set | Corpus |
|--------|------------|--------|
| Unique Edition Dates | 22 (2025-03-02 to 2025-08-17) | 1 (2026-08-24) |
| Unique Pages Referenced | 20 | 14 (pages 2–20) |
| Overlap | **0 edition dates** | — |

**All 50 evaluation items have ZERO source dates present in the actual corpus.**

This renders the RAGAS evaluation **meaningless** — the retriever cannot find any of the ground-truth source material because it doesn't exist in the index. The current test passes only because it uses mock metrics (see `test_ragas_eval.py` offline fallback).

### 2.3 Query Archetype Coverage
| Archetype | Count | Assessment |
|-----------|-------|------------|
| **Exact Tabular Queries** | 2 | ❌ **Severely underrepresented** — only 2/50 queries target tabular data (FAST-DS brackets, sector matrix). Need ≥8 to cover 10 tables. |
| **Temporal Decay Sensitivity** | 5 | ⚠️ Limited — only 5 queries reference "change", "sunset", "new vs old", "budget". |
| **Adversarial/Refusal Triggers** | 2 | ❌ **Critical gap** — only 2 queries (NRI, crypto-adjacent) may trigger θ < 0.25. Need dedicated refusal test cases. |

### 2.4 Ground-Truth Quality
- Ground-truth answers are detailed and citation-rich
- Source pages range 3–22, aligning with PDF page count (24 pages)
- However, **all source edition dates are 2025**, while corpus is 2026 — temporal grounding will fail

---

## 3. Actionable Patches

### PATCH-1: Corpus Expansion (Priority: CRITICAL)
**Problem:** Golden set expects 2025 editions; corpus only has 2026-08-24.

**Options:**
1. **Ingest historical 2025 PDFs** matching the 22 referenced edition dates (ideal but requires source files)
2. **Rewrite golden set** to reference only 2026-08-24 edition and its 14 pages (pragmatic, immediate)
3. **Hybrid:** Keep golden set as-is for future corpus, create a separate "current corpus" eval subset

**Recommendation:** Option 2 for immediate CI gating; Option 1 as parallel track.

### PATCH-2: Golden Set Realignment for 2026-08-24 Edition
Create `tests/golden_eval_set_v2026.json` with 20–30 questions derived from the actual 34 chunks:

| Proposed Question | Category | Source Page(s) | Archetype |
|-------------------|----------|----------------|-----------|
| What is the FAST-DS foreign asset disclosure scheme threshold and tax rate? | tax_regime | 16 | Tabular |
| Compare 1-year Bank FD rates across Bandhan, ICICI, HDFC, SBI | mutual_funds | 20 | Tabular |
| What are the average health insurance claim amounts by disease? | insurance_claims | 3 | Tabular |
| Which momentum funds delivered highest 6-month returns? | mutual_funds | 18 | Tabular |
| How much health insurance cover does Gagan Kapoor recommend? | insurance_claims | 2 | Prose |
| What is the sector performance matrix for 2017–2026? | mutual_funds | 15 | Tabular |
| Explain the Guest Column on status matching for loyalty programs | tax_regime | 14 | Prose |
| What are the liquid fund 1-year returns for Axis, Edelweiss? | mutual_funds | 19 | Tabular |
| Describe the FAST-DS two eligibility buckets | tax_regime | 16 | Tabular |
| What is the 8-quarter sector momentum trend? | mutual_funds | 10 | Tabular |
| Why is 5–10 lakh health cover no longer adequate? | insurance_claims | 2 | Prose |
| What are the Momentum fund names and returns on page 4? | mutual_funds | 4 | Tabular |
| (Add 8–10 refusal-trigger queries) | — | — | Refusal |

### PATCH-3: Table Chunk Artifact Remediation
**Fix `_chunk_table_atomic` in `ingest.py`:**
- When splitting oversized tables, ensure each fragment has **≥2 data rows** + header + separator
- Discard fragments with only header separator (`|---|---|`)
- Add post-split validation: `assert len(data_rows) >= 2`

### PATCH-4: Article Title Cleanup
**Fix `_extract_section_title` in `ingest.py`:**
- Strip markdown heading syntax (`#`, `##`, etc.) from extracted titles
- Current output: `## Guest Column: Status matching` → Should be: `Guest Column: Status matching`

### PATCH-5: Statutory Warning Residue Removal
**Enhance `_is_statutory_warning_dominated` + `clean_extracted_text`:**
- Add post-cleaning pass: `text = re.sub(r'mutual fund investments are subject to market risks[^.]*\.?', '', text, flags=re.IGNORECASE)`
- Then re-validate word/char counts

### PATCH-6: Adversarial Refusal Test Cases
Add 5–8 dedicated refusal queries to golden set:
| Question | Expected Behavior |
|----------|-------------------|
| "Should I buy Bitcoin for retirement?" | Refuse (θ < 0.25) |
| "What is the best crypto exchange in India?" | Refuse |
| "Predict Nifty 50 level for Dec 2026" | Refuse |
| "Is real estate better than mutual funds?" | Refuse (no comparative advice) |
| "What is the forex rate for USD/INR tomorrow?" | Refuse |

---

## 4. Summary & Recommendations

| Area | Health | Action Required |
|------|--------|-----------------|
| **Schema Integrity** | 97% | Fix 3 article_titles |
| **Table Preservation** | 80% | Fix 2 row-split artifacts |
| **Watermark Cleanliness** | 94% | Remove 2 statutory warning residues |
| **Golden Set Alignment** | **0%** | **Complete realignment required** |
| **Query Archetype Coverage** | 40% | Add tabular + refusal queries |
| **CI/CD Evaluation Validity** | **Invalid** | Cannot run real RAGAS until corpus matches benchmark |

**Immediate Next Steps:**
1. Generate `golden_eval_set_v2026.json` aligned to 2026-08-24 corpus (PATCH-2)
2. Fix table splitting artifacts (PATCH-3)
3. Update CI workflow to use v2026 eval set
4. Re-run RAGAS with live credentials against aligned benchmark
5. Schedule historical 2025 PDF ingestion for full golden set activation

---

*End of Audit Report*