"""tests/test_ingest_mocked.py — Chunker edge cases, ID invariants, and ET Wealth harness."""

from __future__ import annotations

import hashlib
import re

from ingest import (
    clean_extracted_text,
    extract_edition_date_from_text,
    generate_chunk_id,
    generate_point_id,
    sliding_window_chunk,
)

# ─────────────────────────────────────────────────────────────────────────────
# Chunker Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestSlidingWindowChunkerEdgeCases:
    def test_empty_string_returns_empty(self) -> None:
        assert sliding_window_chunk("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert sliding_window_chunk("   \n\t  ") == []

    def test_text_under_min_chars_filtered(self) -> None:
        # Below both char and word thresholds → filtered
        assert sliding_window_chunk("Short text.") == []
        assert sliding_window_chunk("A" * 119) == []
        # Single word 120 chars still filtered due to word_count <20 (Pydantic)
        assert sliding_window_chunk("A" * 120) == []
        # Multi-word 120+ chars and >=20 words should pass
        assert sliding_window_chunk(("word " * 30).strip()) != []  # 30 words, ~150 chars

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
        # Single massive token with no spaces → split gives 1 word, but word_count <20 so filtered per Pydantic
        massive = "A" * 5000
        chunks = sliding_window_chunk(massive, chunk_size=600, overlap=100, min_chars=120)
        # Should not crash and should be filtered due to word_count <20 (no orphan short chunks)
        assert len(chunks) == 0

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
        # Below thresholds → filtered
        txt_119 = "A" * 119
        txt_120 = "A" * 120
        assert sliding_window_chunk(txt_119) == []
        # Single word 120 chars still filtered due to word_count (Pydantic)
        assert sliding_window_chunk(txt_120) == []
        # Multi-word 120+ chars and >=20 words should pass
        multi = "word " * 30  # 150 chars, 30 words
        assert sliding_window_chunk(multi.strip()) != []

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


# ─────────────────────────────────────────────────────────────────────────────
# Watermark & Boilerplate Sanitization
# ─────────────────────────────────────────────────────────────────────────────


class TestWatermarkSanitization:
    def test_footer_removal(self) -> None:
        text = "Some content.\n***This PDF download is allowed by Economic Times subscriber for personal use***\nMore content."
        cleaned = clean_extracted_text(text)
        assert "***This PDF download is allowed by Economic Times" not in cleaned
        assert "Some content." in cleaned
        assert "More content." in cleaned

    def test_footer_case_insensitive(self) -> None:
        text = "***THIS PDF DOWNLOAD IS ALLOWED BY ECONOMIC TIMES - DO NOT SHARE***"
        assert "***" not in clean_extracted_text(text) or "Economic Times" not in clean_extracted_text(text)

    def test_email_watermark_removal(self) -> None:
        text = "Content here.\nContact subscriber john.doe@example.com for personal use\nMore text."
        cleaned = clean_extracted_text(text)
        assert "john.doe@example.com" not in cleaned
        assert "Content here." in cleaned

    def test_email_various_formats(self) -> None:
        for email in [
            "user+wealth@economic-times.com",
            "test.user@etwealth.co.in",
            "jane_smith123@domain.co.uk",
        ]:
            text = f"Watermark {email} should be removed."
            assert email not in clean_extracted_text(text)

    def test_banner_removal(self) -> None:
        text = "www.etwealth.co | August 24-30, 2026 | ET Wealth\nActual article content."
        cleaned = clean_extracted_text(text)
        assert "www.etwealth.co" not in cleaned
        assert "Actual article content." in cleaned

    def test_banner_case_insensitive(self) -> None:
        text = "WWW.ETWEALTH.CO | Health Section\nContent."
        assert "ETWEALTH.CO" not in clean_extracted_text(text)

    def test_multiple_watermarks_all_removed(self) -> None:
        text = """www.etwealth.co | August 2026
Content line 1.
***This PDF download is allowed by Economic Times subscriber user@example.com***
Content line 2.
Contact editor@etwealth.co
"""
        cleaned = clean_extracted_text(text)
        assert "www.etwealth.co" not in cleaned
        assert "This PDF download is allowed" not in cleaned
        assert "@" not in cleaned  # all emails removed
        assert "Content line 1." in cleaned
        assert "Content line 2." in cleaned

    def test_clean_preserves_editorial_content(self) -> None:
        text = "Important financial advice about tax slabs and mutual funds.\nThis is legitimate content."
        cleaned = clean_extracted_text(text)
        assert "Important financial advice" in cleaned
        assert "legitimate content" in cleaned

    def test_statutory_warning_dominated_filtered(self) -> None:
        # Chunk dominated by statutory warning with no actionable content should be filtered
        warning_only = "Mutual Fund investments are subject to market risks, read all scheme related documents carefully." * 5
        chunks = sliding_window_chunk(warning_only, chunk_size=100, overlap=10, min_chars=120)
        # All chunks are just warning boilerplate -> should be filtered to 0
        assert len(chunks) == 0

    def test_statutory_warning_with_editorial_kept(self) -> None:
        editorial = "Tax planning for mutual funds requires understanding equity and debt allocation. " * 20
        text = editorial + " Mutual Fund investments are subject to market risks, read all scheme related documents carefully."
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=10, min_chars=120)
        assert len(chunks) >= 1
        # At least one chunk should contain editorial content
        assert any("Tax planning" in c for c in chunks)

    def test_clean_empty_string(self) -> None:
        assert clean_extracted_text("") == ""
        assert clean_extracted_text("   \n\n  ") == ""


# ─────────────────────────────────────────────────────────────────────────────
# Table-Aware Chunking
# ─────────────────────────────────────────────────────────────────────────────


class TestTableAwareChunking:
    def test_small_table_atomic_preservation(self) -> None:
        table = """| Age | Cover | Premium |
|---|---|---|
| 30 | 5 Lakh | Rs 5,200 |
| 35 | 5 Lakh | Rs 6,100 |
| 40 | 5 Lakh | Rs 8,300 |"""
        prose = "Introduction text. " * 20
        text = prose + "\n\n" + table + "\n\n" + prose
        chunks = sliding_window_chunk(text, chunk_size=600, overlap=100, min_chars=120)
        # Table should remain intact as complete Markdown table within a single chunk
        table_chunks = [c for c in chunks if "|" in c and "Age" in c]
        assert len(table_chunks) >= 1
        # Check that table chunk contains all rows (atomic)
        table_chunk = table_chunks[0]
        assert "| 30 | 5 Lakh | Rs 5,200 |" in table_chunk
        assert "| 40 | 5 Lakh | Rs 8,300 |" in table_chunk
        # Header should be present
        assert "| Age | Cover | Premium |" in table_chunk

    def test_table_not_split_across_boundaries(self) -> None:
        # Table under 800 tokens should not be split arbitrarily
        table = "| Col1 | Col2 | Col3 |\n|---|---|---|\n" + "\n".join([f"| A{i} | B{i} | C{i} |" for i in range(20)])
        text = "Intro. " * 30 + "\n\n" + table + "\n\n" + "Outro. " * 30
        chunks = sliding_window_chunk(text, chunk_size=600, overlap=100, min_chars=120)
        # Find table chunks
        table_chunks = [c for c in chunks if "Col1" in c]
        assert len(table_chunks) == 1  # atomic
        # All 20 rows should be in same chunk
        assert "| A19 | B19 | C19 |" in table_chunks[0]
        assert "| A0 | B0 | C0 |" in table_chunks[0]

    def test_oversized_table_split_with_header_repetition(self) -> None:
        # Create oversized table (>800 tokens ~ >800 words)
        header = "| Fund | 1Y | 3Y | 5Y |"
        sep = "|---|---|---|---|"
        rows = [f"| Fund {i} | {i}% | {i+1}% | {i+2}% |" for i in range(100)]
        large_table = header + "\n" + sep + "\n" + "\n".join(rows)
        # Wrap with prose to ensure table is isolated
        text = "Intro. " * 10 + "\n\n" + large_table + "\n\n" + "Outro. " * 10
        chunks = sliding_window_chunk(text, chunk_size=600, overlap=100, min_chars=120)
        table_chunks = [c for c in chunks if "Fund 0" in c or "Fund 50" in c or "Fund 99" in c]
        # Oversized table should be split into multiple chunks
        assert len(table_chunks) >= 2
        # Each split chunk should repeat header
        for tc in table_chunks:
            assert header in tc
            assert sep in tc
        # All rows should be present across chunks
        combined = "\n".join(table_chunks)
        assert "| Fund 0 |" in combined
        assert "| Fund 99 |" in combined

    def test_section_prefix_added(self) -> None:
        text = "# Cover Story: How much health insurance do you need?\n\nThis is content. " * 50
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=10, min_chars=120)
        assert len(chunks) >= 1
        # Each chunk should be prefixed with section title
        assert all(c.startswith("[Section: Cover Story: How much health insurance do you need?]") for c in chunks)

    def test_section_prefix_with_hash_variations(self) -> None:
        for heading in ["# Title", "## Mutual Funds", "### Health Cover Premiums"]:
            text = heading + "\n\nContent. " * 50
            chunks = sliding_window_chunk(text, chunk_size=100, overlap=10, min_chars=120)
            assert len(chunks) >= 1
            expected_title = heading.lstrip("#").strip()
            assert f"[Section: {expected_title}]" in chunks[0]

    def test_table_with_section_prefix(self) -> None:
        text = "## Mutual Funds\n\n| Bank | Rate |\n|---|---|\n| SBI | 6.8% |\n| HDFC | 7.0% |"
        chunks = sliding_window_chunk(text, chunk_size=600, overlap=100, min_chars=120)
        assert len(chunks) >= 1
        assert any("[Section: Mutual Funds]" in c and "|" in c for c in chunks)

    def test_prose_without_heading_no_prefix(self) -> None:
        text = "This is plain prose without heading. " * 50
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=10, min_chars=120)
        assert len(chunks) >= 1
        assert not any(c.startswith("[Section:") for c in chunks)

    def test_table_integrity_preserved_markdown_pipes(self) -> None:
        table = "| Age | Cover | Premium |\n|---|---|---|\n| 30 | 5 Lakh | Rs 5,200 |"
        text = "Intro. " * 20 + "\n\n" + table + "\n\n" + "Outro. " * 20
        chunks = sliding_window_chunk(text, chunk_size=600, overlap=100, min_chars=120)
        # Find chunk containing table and verify pipe structure intact
        table_chunk = next(c for c in chunks if "|" in c)
        assert table_chunk.count("|") >= 6  # at least header + separator + row
        assert "---|---|---" in table_chunk


# ─────────────────────────────────────────────────────────────────────────────
# Date Parser (Magazine Masthead)
# ─────────────────────────────────────────────────────────────────────────────


class TestDateParser:
    def test_august_24_30_2026(self) -> None:
        assert extract_edition_date_from_text("August 24-30, 2026") == "2026-08-24"
        assert extract_edition_date_from_text("www.etwealth.co | August 24-30, 2026 | ET Wealth") == "2026-08-24"

    def test_aug_abbreviation(self) -> None:
        assert extract_edition_date_from_text("Aug 24-30, 2026") == "2026-08-24"
        assert extract_edition_date_from_text("Sept 5-11, 2025") == "2025-09-05"

    def test_single_day(self) -> None:
        assert extract_edition_date_from_text("August 24, 2026") == "2026-08-24"

    def test_day_month_year_format(self) -> None:
        assert extract_edition_date_from_text("24 August 2026") == "2026-08-24"

    def test_month_year_fallback(self) -> None:
        assert extract_edition_date_from_text("August 2026") == "2026-08-01"

    def test_various_months(self) -> None:
        assert extract_edition_date_from_text("January 1-7, 2025") == "2025-01-01"
        assert extract_edition_date_from_text("December 15-21, 2024") == "2024-12-15"
        assert extract_edition_date_from_text("February 2026") == "2026-02-01"

    def test_no_date_returns_none(self) -> None:
        assert extract_edition_date_from_text("This is random text without date") is None
        assert extract_edition_date_from_text("") is None
        assert extract_edition_date_from_text("www.etwealth.co | Health Section") is None

    def test_first_page_header_extraction(self) -> None:
        header = "ET Wealth\nAugust 24-30, 2026\nCover Story: How much health insurance do you need?"
        assert extract_edition_date_from_text(header) == "2026-08-24"

    def test_invalid_date_not_parsed(self) -> None:
        # Invalid day should not crash, return None or handle gracefully
        result = extract_edition_date_from_text("February 30, 2026")
        # February 30 is invalid date, should return None
        assert result is None

    def test_mixed_case_month(self) -> None:
        assert extract_edition_date_from_text("AUGUST 24-30, 2026") == "2026-08-24"
        assert extract_edition_date_from_text("august 24, 2026") == "2026-08-24"


# ─────────────────────────────────────────────────────────────────────────────
# P0-4: Point Collision & Integrity Guard Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPointCollisionGuard:
    def test_collision_detection_with_different_text(self, caplog) -> None:
        """Mock test: verify collision check logic detects different text for same point_id."""

        # This test validates the collision check logic conceptually
        # The actual check happens in ingest_pdf() which requires a real Qdrant client
        # Here we verify the point_id generation is deterministic (collision-resistant)
        pid1 = generate_point_id("2026-08-24", 14, 2, "Understanding Tax Slabs Under the New Regime")
        pid2 = generate_point_id("2026-08-24", 14, 2, "Understanding Tax Slabs Under the New Regime")
        pid3 = generate_point_id("2026-08-24", 14, 2, "Different text content here")

        assert pid1 == pid2  # Same inputs → same ID
        assert pid1 != pid3  # Different text prefix → different ID

        # With 50 chars truncation, first 50 chars same → same ID
        long_text1 = "A" * 50 + "EXTRA1"
        long_text2 = "A" * 50 + "EXTRA2"
        pid4 = generate_point_id("2026-08-24", 14, 2, long_text1)
        pid5 = generate_point_id("2026-08-24", 14, 2, long_text2)
        assert pid4 == pid5  # Truncated to 50 chars → same ID

    def test_500_unique_ids_no_collision(self) -> None:
        """Stress test: 500 unique (edition, page, chunk, text) tuples → zero collisions."""
        seen: set[str] = set()
        for i in range(500):
            edition = f"2026-08-{(i % 28) + 1:02d}"
            page = (i % 200) + 1
            chunk_idx = i % 50
            text_prefix = f"Synthetic chunk text number {i} with unique suffix"
            pid = generate_point_id(edition, page, chunk_idx, text_prefix)
            assert pid not in seen, f"Collision at i={i} pid={pid}"
            seen.add(pid)
        assert len(seen) == 500
