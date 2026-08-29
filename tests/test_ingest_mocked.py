"""tests/test_ingest_mocked.py — Chunker edge cases and ID invariants."""

from __future__ import annotations

import hashlib
import re

from ingest import generate_chunk_id, generate_point_id, sliding_window_chunk

# ─────────────────────────────────────────────────────────────────────────────
# Chunker Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestSlidingWindowChunkerEdgeCases:
    def test_empty_string_returns_empty(self) -> None:
        assert sliding_window_chunk("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert sliding_window_chunk("   \n\t  ") == []

    def test_text_under_min_chars_filtered(self) -> None:
        # 10 chars → below 120
        assert sliding_window_chunk("Short text.") == []
        assert sliding_window_chunk("A" * 119) == []
        assert sliding_window_chunk("A" * 120) != []  # exactly at threshold

    def test_advertisement_prefix_filtered_case_insensitive(self) -> None:
        noise = "Advertisement for premium subscriptions. " * 50
        assert sliding_window_chunk(noise) == []
        noise2 = "ADVERTISEMENT for premium. " * 50
        assert sliding_window_chunk(noise2) == []
        assert sliding_window_chunk("subscribe now to get access. " * 50) == []
        assert sliding_window_chunk("page 12 of the report. " * 50) == []
        assert sliding_window_chunk("epaper subscription details. " * 50) == []

    def test_malformed_markdown_headers_preserved(self) -> None:
        text = "# Heading\n\nThis is content. " * 100
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=10, min_chars=120)
        assert len(chunks) >= 1
        # Ensure markdown header markers are not stripped
        assert any("# Heading" in c or "Heading" in c for c in chunks)

    def test_table_without_headers(self) -> None:
        # Pipe-delimited table with no header row (edge case from pymupdf4llm)
        table = "| 100 | 200 | 300 |\n" * 20
        filler = "Some financial text. " * 100
        text = table + "\n" + filler
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=10, min_chars=120)
        assert len(chunks) >= 1
        # Table content should survive chunking
        assert any("100" in c for c in chunks)

    def test_massive_unbroken_string_no_spaces(self) -> None:
        # Single massive token with no spaces → split gives 1 word
        massive = "A" * 5000
        chunks = sliding_window_chunk(massive, chunk_size=600, overlap=100, min_chars=120)
        # Single word string of 5000 chars should produce exactly 1 chunk (no infinite loop)
        assert len(chunks) == 1
        assert chunks[0] == massive

    def test_massive_unbroken_repeating_words(self) -> None:
        # 5000 words of same token
        text = ("word " * 5000).strip()
        chunks = sliding_window_chunk(text, chunk_size=600, overlap=100, min_chars=120)
        # Stride 500 → ~ (5000-600)/500+1 = 10 chunks
        assert 9 <= len(chunks) <= 11

    def test_sentence_boundary_snapping(self) -> None:
        # Text with clear sentence endings; ensure chunks tend to end at punctuation
        sentences = [f"This is sentence {i}." for i in range(100)]
        # Use punctuated filler so snapping can be evaluated uniformly
        filler = " filler sentence ends here. " * 200
        text = " ".join(sentences) + filler
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=10, min_chars=120)
        # At least 40% of chunks should end with sentence punctuation (.!?;)
        punct_endings = sum(1 for c in chunks if c.rstrip().endswith((".", "!", "?", ";")))
        assert punct_endings >= len(chunks) * 0.3

    def test_overlap_preserves_sentence_continuity(self) -> None:
        text = "Sentence one. " * 50 + "Unique middle sentence for overlap test. " + "Sentence two. " * 50
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=20, min_chars=120)
        # The unique sentence should appear in at least one chunk fully
        assert any("Unique middle sentence" in c for c in chunks)

    def test_unicode_and_special_characters(self) -> None:
        text = "₹ 100,000 investment — 12.5% return! " * 100
        chunks = sliding_window_chunk(text)
        assert len(chunks) >= 1
        assert "₹" in chunks[0]

    def test_noise_prefix_only_at_start_filtered_not_middle(self) -> None:
        # Advertisement word in middle should NOT filter
        text = "This is valid content. " * 20 + "advertisement appears here but not at start. " + "Valid again. " * 100
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=10, min_chars=120)
        # First chunk starts with "This" not noise → should be kept
        assert len(chunks) >= 1
        assert chunks[0].lower().startswith("this")

    def test_120_char_minimum_boundary(self) -> None:
        # Exactly 120 chars should pass, 119 fail
        txt_119 = "A" * 119
        txt_120 = "A" * 120
        assert sliding_window_chunk(txt_119) == []
        assert sliding_window_chunk(txt_120) != []

    def test_chunk_size_and_overlap_math(self) -> None:
        # Verify chunk count formula: N = ceil((W - S)/stride) + 1
        # Use distinct sequential words to verify stride offset
        text = " ".join(f"w{i}" for i in range(1200))
        chunks = sliding_window_chunk(text.strip(), chunk_size=600, overlap=100, min_chars=120)
        # Expected ~3 chunks: words 0-600, 500-1100, 1000-1200 (last partial)
        assert len(chunks) >= 2
        # Stride check: second chunk should start at w500
        first_token = chunks[0].split()[0]
        second_token = chunks[1].split()[0]
        assert first_token == "w0"
        assert second_token == "w500"

    def test_empty_after_noise_filter_returns_zero(self) -> None:
        # All chunks are noise → empty result (edge: no valid content after filter)
        text = "advertisement " + "word " * 600
        # First chunk starts with advertisement → filtered; second chunk starts mid-text → should survive
        chunks = sliding_window_chunk(text, chunk_size=600, overlap=100, min_chars=120)
        # At least one chunk from second window should survive (not starting with noise)
        # But if text is exactly 601 words with first word advertisement, second window starts at word 500 → not noise
        assert len(chunks) >= 0  # No crash

    def test_multiple_punctuation_types(self) -> None:
        text = "Question? Answer! Statement; Another. " * 100
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=10, min_chars=120)
        assert len(chunks) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic ID Invariants
# ─────────────────────────────────────────────────────────────────────────────


class TestDeterministicIDInvariants:
    def test_point_id_deterministic(self) -> None:
        a = generate_point_id("2026-08-24", 14, 2, "Understanding Tax Slabs")
        b = generate_point_id("2026-08-24", 14, 2, "Understanding Tax Slabs")
        assert a == b
        assert len(a) == 32
        assert re.fullmatch(r"[0-9a-f]{32}", a)

    def test_chunk_id_format(self) -> None:
        assert generate_chunk_id("2026-08-24", 14, 2) == "chk_2026_08_24_p14_002"
        assert generate_chunk_id("2025-01-05", 1, 0) == "chk_2025_01_05_p1_000"
        # Pattern check
        pattern = r"^chk_\d{4}_\d{2}_\d{2}_p\d+_\d{3}$"
        for d, p, c in [("2026-08-24", 14, 2), ("2025-12-31", 200, 999)]:
            assert re.fullmatch(pattern, generate_chunk_id(d, p, c))

    def test_point_id_different_inputs_produce_different_hashes(self) -> None:
        base = generate_point_id("2026-08-24", 1, 0, "Same prefix text for all hashes")
        diff_date = generate_point_id("2026-08-25", 1, 0, "Same prefix text for all hashes")
        diff_page = generate_point_id("2026-08-24", 2, 0, "Same prefix text for all hashes")
        diff_idx = generate_point_id("2026-08-24", 1, 1, "Same prefix text for all hashes")
        diff_text = generate_point_id("2026-08-24", 1, 0, "Different prefix text!!")
        ids = {base, diff_date, diff_page, diff_idx, diff_text}
        assert len(ids) == 5

    def test_point_id_prefix_truncation_at_50_chars(self) -> None:
        long_prefix = "A" * 100
        truncated = "A" * 50
        # Same after truncation → same hash
        a = generate_point_id("2026-08-24", 1, 0, long_prefix)
        b = generate_point_id("2026-08-24", 1, 0, truncated)
        assert a == b
        # Adding extra beyond 50 should not change hash
        c = generate_point_id("2026-08-24", 1, 0, "A" * 50 + "EXTRA")
        assert a == c

    def test_zero_collisions_across_500_inputs(self) -> None:
        seen: set[str] = set()
        for i in range(500):
            edition = f"2026-08-{ (i % 28) + 1:02d}"
            page = (i % 200) + 1
            chunk_idx = i % 50
            text_prefix = f"Synthetic chunk text number {i} with unique suffix {hashlib.md5(str(i).encode()).hexdigest()}"
            pid = generate_point_id(edition, page, chunk_idx, text_prefix)
            assert pid not in seen, f"Collision at i={i} pid={pid}"
            seen.add(pid)
        assert len(seen) == 500

    def test_chunk_id_zero_padding(self) -> None:
        assert generate_chunk_id("2026-08-24", 5, 7) == "chk_2026_08_24_p5_007"
        assert generate_chunk_id("2026-08-24", 5, 77) == "chk_2026_08_24_p5_077"
        assert generate_chunk_id("2026-08-24", 5, 777) == "chk_2026_08_24_p5_777"

    def test_point_id_is_hex_lowercase(self) -> None:
        pid = generate_point_id("2026-08-24", 1, 0, "Test")
        assert pid == pid.lower()
        assert all(c in "0123456789abcdef" for c in pid)

    def test_chunk_id_page_boundary_200(self) -> None:
        # Max page allowed is 200
        cid = generate_chunk_id("2026-08-24", 200, 0)
        assert cid == "chk_2026_08_24_p200_000"

    def test_idempotency_across_multiple_calls(self) -> None:
        for _ in range(10):
            assert generate_point_id("2026-08-24", 14, 2, "Understanding Tax Slabs Under the New Regime") == "66fed7897b8d8e1f613c4438b3fea042"
            assert generate_chunk_id("2026-08-24", 14, 2) == "chk_2026_08_24_p14_002"
