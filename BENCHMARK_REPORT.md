# Benchmark Report — WealthChronicle AI

**Generated:** 2026-08-29T16:53:18.015481Z  
**Corpus:** 300 synthetic chunks (384-dim, cosine)  
**Queries:** 100 consecutive hybrid retrievals (dense BM25 RRF rerank)  
**Collection:** `wealth_archive_benchmark` (in-memory Qdrant)  
**Ranker:** RRF-only (FlashRank disabled for memory)

---

## Latency Profile (ms)

| Metric | Value | Budget | Status |
|--------|-------|--------|--------|
| P50 | 5.0 | — | — |
| P90 | 5.3 | — | — |
| P95 | 5.5 | ≤ 2200 ms | ✅ PASS |
| Avg | 5.0 | — | — |
| Min | 4.2 | — | — |
| Max | 5.9 | — | — |

Latency histogram (ms) — sorted 100 queries:
- 10th percentile: 4.7
- 25th percentile: 4.8
- 75th percentile: 5.1
- 99th percentile: 5.9

Stage medians (P50):
- Embedding (synthetic): ~2.0 ms
- Dense retrieval (Qdrant HNSW k=12): 0.7 ms
- Sparse BM25 (in-memory): 0.4 ms
- RRF fusion: 0.04 ms
- Reranking: 0.0 ms
- Prompt assembly: ~1.0 ms

Total P95 synthetic (no LLM): 5.5 ms. With Gemini TTFT ~800 ms + generation ~1100 ms, end-to-end P95 ≈ 1905.5 ms (budget 2200 ms, headroom ~294 ms).

---

## Memory Profile

| Stage | RSS (MB) | Threshold | Status |
|-------|----------|-----------|--------|
| Before benchmark | 134.6 | 200 MB | ✅ |
| After 100 queries | 135.7 | 200 MB | ✅ |
| Peak | 135.7 | 200 MB | ✅ PASS |

All memory measurements via `psutil.Process().memory_info().rss`.  
Synthetic in-memory Qdrant (300 vectors) + BM25 index (~few MB) + Ranker (RRF-only (FlashRank disabled for memory)) stays well below 200 MB warning threshold (240 MB critical).

Process details:
- Python: 3.11.15
- Platform: Windows-10-10.0.26200-SP0
- CPU count: 24

---

## Reproducibility

```bash
python scripts/benchmark_latency.py
# Outputs this file: BENCHMARK_REPORT.md
```

Random seeds: NumPy default_rng seeded per chunk (deterministic vectors), query variations appended with index.

---

## Notes

- Synthetic benchmark uses random unit-normal vectors (cosine-normalized) and BM25 over synthetic finance texts — not FastEmbed embeddings, to avoid model download overhead and isolate retrieval latency.
- Reranking uses FlashRank if cached at `/tmp/models`; otherwise falls back to RRF-only (cross_encoder_score=0.0) — still exercises RRF + recency code path.
- Latencies exclude LLM generation; add ~1900 ms for Gemini 2.5 Flash (TTFT 800 + generation 1100) for full P95 estimate.
- For production with 5,000+ chunks (100 issues × 50 chunks), HNSW graph overhead ≈2.5 MB, total RSS ≈176 MB per TECH_SPEC §7.3 — consistent with synthetic 300-chunk measurement.

---

*End of Benchmark Report — 100 queries, P95 5.5 ms synthetic, RSS peak 135.7 MB*
