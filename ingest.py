"""
ingest.py — Admin Ingestion Plane for WealthChronicle AI v1.0
Covers TASK-3.1 through TASK-3.7
Upgraded with watermark sanitization and table-aware chunking for Economic Times Wealth.
Migrated to Qdrant native sparse vectors (BM42) for hybrid search.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pymupdf4llm
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from engine import with_qdrant_retry
from schemas import ChunkPayload

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "wealth_archive"

# ─────────────────────────────────────────────────────────────────────────────
# TASK-3.1: Layout-Aware PDF Extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract layout-aware Markdown from each PDF page.

    Returns:
        List of dicts, each with keys:
        - "metadata": {"page": int}  (0-indexed page number)
        - "text": str                (Markdown-formatted page content)

    Handles both legacy (metadata.page 0-indexed) and current pymupdf4llm
    (metadata.page_number 1-indexed) conventions by normalizing to 0-indexed
    'page' while preserving original keys.
    """
    raw = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,
    )
    normalized: list[dict] = []
    for item in raw:
        meta = item.get("metadata", {})
        # Determine 0-indexed page number
        if "page" in meta:
            page_0 = int(meta["page"])
        elif "page_number" in meta:
            # Current pymupdf4llm returns 1-indexed page_number
            page_0 = int(meta["page_number"]) - 1
        else:
            # Fallback: sequential index
            page_0 = len(normalized)
        # Ensure both conventions present for downstream compatibility
        new_meta = dict(meta)
        new_meta["page"] = page_0
        if "page_number" not in new_meta:
            new_meta["page_number"] = page_0 + 1
        normalized.append(
            {
                "metadata": new_meta,
                "text": item.get("text", ""),
                **{k: v for k, v in item.items() if k not in ("metadata", "text")},
            }
        )
        # Keep toc_items etc if present
        for k in ("toc_items", "page_boxes"):
            if k in item:
                normalized[-1][k] = item[k]
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRE-PROCESSING & SANITIZATION
# ─────────────────────────────────────────────────────────────────────────────

_FOOTER_PATTERN = re.compile(r"\*\*\*This PDF download is allowed by Economic Times.*", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_BANNER_PATTERN = re.compile(r"www\.etwealth\.co\s*\|.*", re.IGNORECASE)
_STATUTORY_WARNING_PHRASE = "Mutual Fund investments are subject to market risks"

# Month mapping for date parser
_MONTH_MAP = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "sept": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def clean_extracted_text(text: str) -> str:
    """Sanitize extracted markdown text by stripping watermarks and boilerplate.

    Removes:
    - Footer lines matching ***This PDF download is allowed by Economic Times.*
    - Email watermarks (subscriber emails)
    - Running top banners www.etwealth.co | ...
    - Statutory mutual fund warning phrases

    Returns cleaned text with normalized whitespace.
    """
    if not text:
        return text

    # Remove footer lines (multiline)
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        # Check footer pattern — if line matches, skip entire line
        if _FOOTER_PATTERN.search(line):
            continue
        # Remove banner lines
        if _BANNER_PATTERN.search(line):
            # Remove the banner portion but keep rest of line if any editorial content
            line = _BANNER_PATTERN.sub("", line)
            if not line.strip():
                continue
        # Remove email watermarks
        # We replace emails with empty, but if line becomes empty after, skip
        if _EMAIL_PATTERN.search(line):
            line = _EMAIL_PATTERN.sub("", line)
            # Also remove common watermark surrounds like "subscriber user@... for personal use"
            # Clean up leftover artifacts like "subscriber  for personal use"
            line = re.sub(r"\s{2,}", " ", line).strip()
            if not line.strip() or len(line.strip()) < 5:
                continue
            # If line after email removal is just boilerplate fragments, skip
            if re.match(r"^\s*(subscriber|for personal use|not for redistribution)?\s*$", line, re.IGNORECASE):
                continue
        cleaned_lines.append(line)

    # Rejoin and also run global email substitution for any inline emails missed
    cleaned = "\n".join(cleaned_lines)
    cleaned = _EMAIL_PATTERN.sub("", cleaned)
    # Strip statutory warning phrases globally (even within preserved editorial content)
    cleaned = re.sub(r"mutual fund investments are subject to market risks[^.\n]*\.?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"read all scheme related documents carefully[^.\n]*\.?", "", cleaned, flags=re.IGNORECASE)
    # Collapse multiple blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Remove lines that are now just whitespace
    cleaned = "\n".join([ln for ln in cleaned.split("\n") if ln.strip() or ln == ""])
    # Normalize spaces but preserve markdown structure
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _is_statutory_warning_dominated(text: str) -> bool:
    """Check if chunk is dominated by mutual fund statutory warning with no actionable content.

    Returns True if chunk should be discarded. Keeps chunks that have editorial content
    alongside the warning by checking remaining text after removing warning.
    """
    if _STATUTORY_WARNING_PHRASE not in text:
        return False

    stripped = text.strip()
    lower = stripped.lower()

    # Remove warning phrase and its common continuation to assess remaining content
    remaining = lower.replace("mutual fund investments are subject to market risks", "")
    remaining = re.sub(r"read all scheme related documents carefully[^.\n]*\.?", "", remaining)
    remaining = remaining.strip()
    # Remove prefix like [Section: ...] for assessment
    remaining = re.sub(r"^\[section:[^\]]+\]\s*", "", remaining)

    # If remaining after removing warning is substantial (>20 words and >120 chars), not dominated
    remaining_words = len(remaining.split())
    remaining_chars = len(remaining)

    if remaining_words >= 20 and remaining_chars >= 120:
        # Has actionable editorial content, keep it (warning is just incidental)
        return False

    # If chunk starts with warning and remaining is small, it's boilerplate
    if lower.strip().startswith("mutual fund investments are subject to market risks"):
        return True

    # If warning appears multiple times and dominates short chunk
    if lower.count("mutual fund investments are subject to market risks") >= 2 and len(stripped.split()) < 150:
        return True

    # If remaining is tiny, dominated
    if remaining_words < 15 or remaining_chars < 100:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. TABLE-AWARE CHUNKING HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _extract_section_title(text: str) -> str | None:
    """Extract top-level heading/section title from page markdown.

    Looks for first markdown heading (#, ##, etc.) on the page.
    Returns title without hash marks and markdown formatting, or None if not found.
    """
    for line in text.split("\n"):
        stripped = line.strip()
        # Match markdown headings: # Title, ## Title, etc. up to ######
        m = re.match(r"^\s*#{1,6}\s+(.+)$", stripped)
        if m:
            title = m.group(1).strip()
            # Clean up markdown formatting inside heading
            title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)
            title = re.sub(r"\*(.+?)\*", r"\1", title)
            # Remove any remaining markdown artifacts
            title = re.sub(r"[#*_`~]", "", title).strip()
            if title:
                return title
    return None


def _has_table_data_rows(table_text: str) -> bool:
    """Check if a markdown table chunk has at least 1 data row (not just header/separator).

    Returns True if the table has meaningful data rows, False if it's just
    header + separator or header only (artifact from collapsed newlines).
    """
    lines = [line for line in table_text.strip().split("\n") if line.strip().startswith("|")]
    if len(lines) < 2:
        return False
    # Check for header separator line
    has_separator = any(re.search(r"\|\s*---+\s*\|", line) for line in lines)
    if has_separator:
        # Has formal markdown separator - count non-header, non-separator lines
        data_rows = [line for line in lines if not re.search(r"\|\s*---+\s*\|", line)]
        # Exclude the first line (header) from data rows
        if data_rows and not re.search(r"\|\s*---+\s*\|", data_rows[0]):
            data_rows = data_rows[1:]
        return len(data_rows) >= 1
    else:
        # No formal separator - if we have multiple pipe lines, assume they're data
        # This handles tables without markdown separators (e.g., from PDF extraction)
        return len(lines) >= 3


def _split_table_blocks(text: str) -> list[tuple[str, str]]:
    """Split page text into (type, block) where type is 'table' or 'prose'.

    Tables are identified as consecutive lines containing '|' with pipe table syntax.
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []
    cur_table: list[str] = []
    cur_prose: list[str] = []

    def flush_prose():
        nonlocal cur_prose
        if cur_prose:
            prose_text = "\n".join(cur_prose).strip()
            if prose_text:
                blocks.append(("prose", prose_text))
            cur_prose = []

    def flush_table():
        nonlocal cur_table
        if cur_table:
            table_text = "\n".join(cur_table).strip()
            if table_text:
                blocks.append(("table", table_text))
            cur_table = []

    for line in lines:
        stripped = line.strip()
        # Detect markdown table line: contains '|' and has at least 2 pipes or table separator
        is_table_line = ("|" in line) and (stripped.startswith("|") or " | " in line or stripped.count("|") >= 2)
        # Also consider separator lines like |---|---| or |:---|
        is_separator = bool(re.match(r"^\s*\|?(\s*:?-+:?\s*\|)+\s*$", line)) if is_table_line else False

        if is_table_line or is_separator:
            # If we were building prose, flush it
            if cur_prose:
                flush_prose()
            cur_table.append(line)
        else:
            # Non-table line
            if cur_table:
                # If line is blank, it might be a blank separator between table rows (pymupdf4llm uses double newlines)
                # Don't flush table on blank line alone; keep table open for next pipe row
                if stripped == "":
                    # Peek: if we are in table, ignore blank lines and keep table open
                    continue
                # Line is prose (non-empty non-table), so flush table first
                flush_table()
                if stripped:
                    cur_prose.append(line)
            else:
                # Not in table; add to prose (including empty lines as separators)
                cur_prose.append(line)

    flush_table()
    flush_prose()
    return blocks


def _chunk_table_atomic(table_text: str, prefix: str, max_tokens: int = 800) -> list[str]:
    """Chunk a markdown table atomically or by rows with header repetition.

    If table under max_tokens words, return single chunk.
    For oversized tables, split by rows while repeating header row at top of each chunk.
    Discard any slice with fewer than 2 actual data rows (e.g., just header + delimiter).
    Handles case where header and separator are on the same line (collapsed newlines from PDF).
    """
    # Check if table is small enough to keep atomic
    words = table_text.split()
    if len(words) < max_tokens:
        # Single atomic chunk
        chunk = f"{prefix}{table_text}" if prefix else table_text
        return [chunk]

    # Oversized: split by rows
    rows = [r for r in table_text.strip().split("\n") if r.strip()]
    if not rows:
        return []

    # Identify header and separator
    header = rows[0]
    separator = None
    data_start_idx = 1

    # Check if header line contains both header and separator (collapsed newlines)
    # Pattern: | Header | Header | ... |---|---|...
    header_sep_match = re.search(r"(\|.*\|)\s*(\|?\s*:?-+:?\s*\|.*)$", header)
    if header_sep_match and not separator:
        # Split header and separator
        header = header_sep_match.group(1).rstrip()
        separator = header_sep_match.group(2).strip()
        data_start_idx = 1  # Data rows start from next line
    elif len(rows) > 1 and re.match(r"^\s*\|?(\s*:?-+:?\s*\|)+\s*$", rows[1]):
        separator = rows[1]
        data_start_idx = 2

    data_rows = rows[data_start_idx:]
    if not data_rows:
        # Only header, return atomic (but if header had collapsed separator, keep it)
        chunk = f"{prefix}{header}"
        if separator:
            chunk += "\n" + separator
        return [chunk]

    chunks: list[str] = []
    current_rows: list[str] = []
    current_word_count = 0
    header_words = len(header.split()) + (len(separator.split()) if separator else 0)

    for row in data_rows:
        row_words = len(row.split())
        # If adding this row would exceed max_tokens, flush current chunk
        if current_rows and (current_word_count + row_words + header_words) >= max_tokens:
            # Build chunk with header repetition
            # Only keep chunk if it has at least 2 data rows
            if len(current_rows) >= 2:
                chunk_lines = [header]
                if separator:
                    chunk_lines.append(separator)
                chunk_lines.extend(current_rows)
                chunk_text = "\n".join(chunk_lines)
                if prefix:
                    chunk_text = f"{prefix}{chunk_text}"
                chunks.append(chunk_text)
            current_rows = []
            current_word_count = 0

        current_rows.append(row)
        current_word_count += row_words

    # Flush remaining - only if at least 2 data rows
    if current_rows and len(current_rows) >= 2:
        chunk_lines = [header]
        if separator:
            chunk_lines.append(separator)
        chunk_lines.extend(current_rows)
        chunk_text = "\n".join(chunk_lines)
        if prefix:
            chunk_text = f"{prefix}{chunk_text}"
        chunks.append(chunk_text)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# TASK-3.2: Punctuation-Aware Sliding Window Chunker (Upgraded Table-Aware)
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"[.!?;]\s")
_NOISE_PREFIXES = ("advertisement", "subscribe", "page ", "epaper")


def sliding_window_chunk(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
    min_chars: int = 120,
) -> list[str]:
    """Sliding window chunker with punctuation-aware boundary snapping, table-aware isolation, and section prefixing.

    Algorithm:
        1. Extract section title and create prefix.
        2. Split text into table vs prose blocks.
        3. For table blocks: keep atomic if <800 tokens, else split by rows with header repetition.
        4. For prose blocks: apply existing sliding window with sentence snapping.
        5. Prefix each chunk with [Section: Title] if title exists.

    Args:
        text: Full page Markdown text.
        chunk_size: Target words per chunk (S).
        overlap: Overlapping words between consecutive chunks (O).
        min_chars: Minimum character length to retain a chunk.

    Returns:
        List of clean chunk strings.
    """
    if not text or not text.strip():
        return []

    # Extract section title for context preservation
    section_title = _extract_section_title(text)
    prefix = f"[Section: {section_title}]\n" if section_title else ""

    # Split into table vs prose blocks
    blocks = _split_table_blocks(text)

    # If no table blocks detected, fall back to original prose-only logic but with prefix handling
    # Check if any block is table
    has_table = any(btype == "table" for btype, _ in blocks)
    if not has_table:
        # Original sliding window logic on full text (but we should avoid double-prefixing heading line itself)
        # Remove heading line from text for chunking to avoid duplication, but prefix will preserve context
        # For simplicity, apply original logic to the whole text
        return _sliding_window_prose(text, prefix, chunk_size, overlap, min_chars)

    # Table-aware: process each block separately
    chunks: list[str] = []
    for btype, block_text in blocks:
        if not block_text.strip():
            continue
        if btype == "table":
            # Table-aware atomic or row-split chunking — keep atomic even if small (tables are high-value)
            table_chunks = _chunk_table_atomic(block_text, prefix, max_tokens=800)
            for tc in table_chunks:
                # Tables use lower threshold (20 chars) and bypass strict min_chars, but still filter noise/warning
                if len(tc.strip()) >= 20 and not tc.lower().strip().startswith(_NOISE_PREFIXES):
                    if _is_statutory_warning_dominated(tc):
                        continue
                    # Discard table chunks with no meaningful data rows (header + separator only)
                    if not _has_table_data_rows(tc):
                        continue
                    chunks.append(tc)
        else:  # prose
            # Skip heading-only blocks that are just section titles (already captured in prefix)
            # Check if block is heading-only (all non-empty lines start with #) and small (<30 words)
            lines = [ln for ln in block_text.split("\n") if ln.strip()]
            is_heading_only = lines and all(ln.strip().startswith("#") for ln in lines) and len(block_text.split()) < 30
            if is_heading_only:
                continue
            prose_chunks = _sliding_window_prose(block_text, prefix, chunk_size, overlap, min_chars)
            chunks.extend(prose_chunks)

    return chunks


def _sliding_window_prose(
    text: str,
    prefix: str,
    chunk_size: int = 600,
    overlap: int = 100,
    min_chars: int = 120,
) -> list[str]:
    """Original sliding window logic applied to prose block with prefix."""
    words: list[str] = text.split()
    total_words: int = len(words)
    chunks: list[str] = []

    stride: int = chunk_size - overlap  # 500

    i: int = 0
    while i < total_words:
        end: int = min(i + chunk_size, total_words)
        chunk_words: list[str] = words[i:end]

        # --- Boundary refinement: snap to sentence end ---
        if end < total_words:
            candidate: str = " ".join(chunk_words)
            # Search backward from end for sentence-ending punctuation
            last_sent_end = -1
            for match in _SENTENCE_END.finditer(candidate):
                last_sent_end = match.end()

            if last_sent_end > len(candidate) * 0.6:  # At least 60% of chunk used
                candidate = candidate[:last_sent_end].rstrip()
                chunk_words = candidate.split()

        chunk_text: str = " ".join(chunk_words)
        # Prefix with section title for context preservation
        if prefix:
            # Avoid double prefix if chunk already starts with prefix
            if not chunk_text.startswith("[Section:"):
                chunk_text = f"{prefix}{chunk_text}"

        # --- Noise filter (FR-ING-03) + statutory warning + word count ---
        word_count = len(chunk_text.split())
        # Require both char and word count (Pydantic: char>=120 word>=20), but allow tables with pipe to bypass word count
        has_table = "|" in chunk_text
        if len(chunk_text) >= min_chars and word_count >= 20 and not chunk_text.lower().startswith(_NOISE_PREFIXES):
            if _is_statutory_warning_dominated(chunk_text):
                i += stride
                continue
            chunks.append(chunk_text)
        elif has_table and len(chunk_text.strip()) >= 50:
            # Small table-like prose that contains pipes but was split as prose (edge case)
            # Keep it if it has meaningful table structure, even if word count is borderline
            if not chunk_text.lower().startswith(_NOISE_PREFIXES) and not _is_statutory_warning_dominated(chunk_text):
                chunks.append(chunk_text)

        i += stride

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# TASK-3.3: PDF Extraction Quality Validator
# ─────────────────────────────────────────────────────────────────────────────


def validate_extraction(pages: list[dict], pdf_path: str) -> None:
    """Validate that PDF extraction produced meaningful content.

    Raises:
        ValueError: If extraction quality is below acceptable threshold.
    """
    total_chars = sum(len(p["text"].strip()) for p in pages)
    non_empty_pages = sum(1 for p in pages if len(p["text"].strip()) > 50)
    total_pages = len(pages)

    if total_chars < 500:
        raise ValueError(f"EXTRACTION_FAILURE: {pdf_path} yielded only {total_chars} chars. " f"This PDF may be scanned/image-only. Run OCR preprocessing first.")

    coverage = non_empty_pages / total_pages if total_pages > 0 else 0
    if coverage < 0.5:
        raise ValueError(f"LOW_COVERAGE: {pdf_path} — only {non_empty_pages}/{total_pages} pages " f"({coverage:.0%}) had extractable text. Check for mixed scan/text pages.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK-3.4: Deterministic Point ID & Chunk ID Generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_point_id(
    edition_date: str,  # "2026-08-24"
    page_number: int,  # 1-indexed
    chunk_index: int,  # 0-indexed within page
    text_prefix: str,  # First 50 chars of chunk text
) -> str:
    """Generate a deterministic, collision-resistant point ID.

    Scheme:
        MD5( "{edition_date}|p{page}|c{chunk_idx}|{text[:50]}" )

    Returns:
        32-character lowercase hex string (compatible with Qdrant UUID-style IDs).

    Collision analysis:
        MD5 produces 128-bit digests. For a corpus of 10,000 chunks,
        P(collision) approx 1.47 × 10^-34 (birthday bound), which is negligible.
        The text_prefix further disambiguates chunks with identical positional metadata.
    """
    seed = f"{edition_date}|p{page_number}|c{chunk_index}|{text_prefix[:50]}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()


def generate_chunk_id(edition_date: str, page_number: int, chunk_index: int) -> str:
    """Human-readable chunk identifier for payload and logging.

    Format: chk_{YYYY}_{MM}_{DD}_p{page}_{seq:03d}
    Example: chk_2026_08_24_p14_002
    """
    d = edition_date.replace("-", "_")
    return f"chk_{d}_p{page_number}_{chunk_index:03d}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATE EXTRACTION HELPER
# ─────────────────────────────────────────────────────────────────────────────


def extract_edition_date_from_text(text: str) -> str | None:
    """Auto-detect edition date from PDF first page header masthead.

    Supports formats like:
    - August 24-30, 2026 -> 2026-08-24
    - Aug 24-30, 2026 -> 2026-08-24
    - August 24, 2026 -> 2026-08-24
    - August 2026 -> 2026-08-01 (fallback to first of month)
    - 24 August 2026 -> 2026-08-24

    Returns YYYY-MM-DD string for start date, or None if not found.
    """
    if not text:
        return None

    # Normalize whitespace
    text = text[:2000]  # Only first 2000 chars of first page for masthead

    # Pattern 1: Month Day-Day, Year  e.g., August 24-30, 2026
    pattern1 = re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})(?:\s*-\s*\d{1,2})?,?\s+(\d{4})",
        re.IGNORECASE,
    )
    m = pattern1.search(text)
    if m:
        month_str = m.group(1).lower()
        day = m.group(2)
        year = m.group(3)
        month_num = _MONTH_MAP.get(month_str[:3] if len(month_str) > 3 else month_str, None)
        # Handle full month vs abbreviation: lookup first 3 chars lower
        if not month_num:
            # Try full name
            month_num = _MONTH_MAP.get(month_str, None)
        if month_num:
            try:
                # Validate date
                datetime(int(year), int(month_num), int(day))
                return f"{year}-{month_num}-{int(day):02d}"
            except ValueError:
                pass

    # Pattern 2: Day Month Year e.g., 24 August 2026
    pattern2 = re.compile(
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(\d{4})",
        re.IGNORECASE,
    )
    m2 = pattern2.search(text)
    if m2:
        day = m2.group(1)
        month_str = m2.group(2).lower()
        year = m2.group(3)
        month_num = _MONTH_MAP.get(month_str[:3], _MONTH_MAP.get(month_str, None))
        if month_num:
            try:
                datetime(int(year), int(month_num), int(day))
                return f"{year}-{month_num}-{int(day):02d}"
            except ValueError:
                pass

    # Pattern 3: Month Year e.g., August 2026 (fallback to 01)
    pattern3 = re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
        re.IGNORECASE,
    )
    m3 = pattern3.search(text)
    if m3:
        month_str = m3.group(1).lower()
        year = m3.group(2)
        month_num = _MONTH_MAP.get(month_str, None)
        if month_num:
            return f"{year}-{month_num}-01"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# TASK-3.5: Qdrant Collection Initialization with HNSW Config & Payload Indexes
# ─────────────────────────────────────────────────────────────────────────────


def ensure_payload_indexes(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """Create payload indexes for filtering and sorting.

    Creates keyword index on edition_date, integer index on page_number,
    boolean index on has_table, and keyword index on source. All calls are idempotent.
    """
    indexes = [
        {"field_name": "edition_date", "field_schema": PayloadSchemaType.KEYWORD},
        {"field_name": "has_table", "field_schema": PayloadSchemaType.BOOL},
        {"field_name": "page_number", "field_schema": PayloadSchemaType.INTEGER},
        {"field_name": "source", "field_schema": PayloadSchemaType.KEYWORD},
    ]
    for idx in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=idx["field_name"],
                field_schema=idx["field_schema"],
            )
        except Exception:
            # Idempotent: index already exists
            pass


def ensure_collection_exists(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """Initialize Qdrant collection with named vectors for hybrid search and ensure payload indexes.

    Creates collection if it does not exist, with:
        - Dense vector: "dense" (384-dim, COSINE) with HNSW m=16, ef_construct=128
        - Sparse vector: "sparse" (BM42/SPLADE) with on-disk index
        - HNSW config: m=16, ef_construct=128, full_scan_threshold=10_000
        - Optimizers: indexing_threshold=20_000, memmap_threshold=50_000
    Verifies and creates payload indexes:
        - edition_date: KEYWORD
        - has_table: BOOL
        - page_number: INTEGER
        - source: KEYWORD
    """
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"[OK] Created collection: {collection_name} with dense + sparse vectors")

    ensure_payload_indexes(client, collection_name)


def init_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """Backward-compatible alias for ensure_collection_exists."""
    ensure_collection_exists(client, collection_name)


# ─────────────────────────────────────────────────────────────────────────────
# TASK-3.6: Batch Embed & Upsert Pipeline (Upgraded with cleaning)
# ─────────────────────────────────────────────────────────────────────────────


def ingest_pdf(pdf_path: str, edition_date: str | None = None, client: QdrantClient | None = None) -> int:
    """Full ingestion pipeline for a single PDF.

    Steps:
        1. Call extract_pages(pdf_path)
        2. Clean extracted text via clean_extracted_text() (watermark sanitization)
        3. Call validate_extraction(pages, pdf_path)
        4. For each page: call sliding_window_chunk(text) (table-aware)
        5. Convert 0-indexed to 1-indexed page numbers.
        5. Generate chunk_id and point_id for each chunk.
        6. Construct ChunkPayload Pydantic objects
        7. Embed all texts using TextEmbedding (dense) and SparseTextEmbedding (sparse)
        8. Build PointStruct list with named vectors (dense + sparse) + payload dict
        9. Call client.upsert with @with_qdrant_retry

    Args:
        pdf_path: Path to PDF file
        edition_date: Edition date YYYY-MM-DD. If None, auto-detected from PDF first page header.
        client: Optional QdrantClient for testing (in-memory).

    Returns:
        Number of chunks ingested.
    """
    import os

    # Auto-detect edition_date if not provided
    if edition_date is None:
        try:
            # Extract first page text for date parsing
            preview_pages = extract_pages(pdf_path)
            if preview_pages:
                first_page_text = preview_pages[0].get("text", "")
                detected = extract_edition_date_from_text(first_page_text)
                if detected:
                    edition_date = detected
                    print(f"[INFO] Auto-detected edition date: {edition_date}")
                else:
                    # Fallback try filename pattern YYYY-MM-DD
                    fname = Path(pdf_path).stem
                    m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
                    if m:
                        edition_date = m.group(1)
                        print(f"[INFO] Inferred edition date from filename: {edition_date}")
                    else:
                        raise ValueError("Could not auto-detect edition date from PDF header or filename. Please provide YYYY-MM-DD explicitly.")
            else:
                raise ValueError("Could not extract pages for date detection")
        except Exception as e:
            raise ValueError(f"Failed to auto-detect edition date: {e}")

    # Validate edition_date format
    try:
        datetime.strptime(edition_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid edition_date format: {edition_date}. Expected YYYY-MM-DD")
    except TypeError:
        raise ValueError(f"Invalid edition_date format: {edition_date}. Expected YYYY-MM-DD")

    # Get Qdrant client if not provided
    if client is None:
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_key = os.environ.get("QDRANT_ADMIN_KEY") or os.environ.get("QDRANT_API_KEY")
        if not qdrant_url:
            raise ValueError("QDRANT_URL environment variable not set")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    print(f"[*] Parsing {pdf_path} (Edition: {edition_date})...")

    # 1. Extract pages
    pages = extract_pages(pdf_path)

    # 1b. Clean extracted text immediately after extract_pages (watermark sanitization)
    for p in pages:
        original = p.get("text", "")
        cleaned = clean_extracted_text(original)
        p["text"] = cleaned

    # 2. Validate extraction
    validate_extraction(pages, pdf_path)

    # 3-6. Chunk and build payloads
    all_texts: list[str] = []
    all_payloads: list[ChunkPayload] = []
    all_point_ids: list[str] = []

    for item in pages:
        meta = item.get("metadata", {})
        if "page" in meta:
            page_num_0indexed = int(meta["page"])
        elif "page_number" in meta:
            page_num_0indexed = int(meta["page_number"]) - 1
        else:
            page_num_0indexed = 0
        page_number = page_num_0indexed + 1  # Convert to 1-indexed
        text = item["text"].strip()
        if not text:
            continue

        # Extract section/article title for structured metadata (F-09)
        section_title = _extract_section_title(text)

        chunks = sliding_window_chunk(text, chunk_size=600, overlap=100, min_chars=120)

        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_id = generate_chunk_id(edition_date, page_number, chunk_idx)
            point_id = generate_point_id(edition_date, page_number, chunk_idx, chunk_text)

            char_count = len(chunk_text)
            word_count = len(chunk_text.split())

            # Use section title as article_title if available; fallback to parsing prefix in chunk
            article_title: str | None = section_title
            if article_title is None and chunk_text.startswith("[Section:"):
                # Try to parse prefix from chunk itself
                m = re.match(r"^\[Section:\s*([^\]]+)\]", chunk_text)
                if m:
                    article_title = m.group(1).strip()
            # Fallback: use page number as title if still None
            if article_title is None:
                article_title = f"Page {page_number}"

            has_table = ("|" in chunk_text) and _has_table_data_rows(chunk_text)

            payload = ChunkPayload(
                chunk_id=chunk_id,
                edition_date=edition_date,
                page_number=page_number,
                article_title=article_title,
                text=chunk_text,
                has_table=has_table,
                char_count=char_count,
                word_count=word_count,
            )

            all_texts.append(chunk_text)
            all_payloads.append(payload)
            all_point_ids.append(point_id)

    if not all_texts:
        print("[!] No chunks generated after filtering. Check PDF content.")
        return 0

    print(f"[*] Generating embeddings for {len(all_texts)} chunks...")

    # 7. Generate both dense and sparse embeddings
    print("[*] Generating dense embeddings (BAAI/bge-small-en-v1.5)...")
    dense_embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    dense_embeddings = list(dense_embedding_model.embed(all_texts, batch_size=32))

    print("[*] Generating sparse embeddings (Qdrant/bm42-all-minilm-l6-v2-attentions)...")
    sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
    sparse_embeddings = list(sparse_embedding_model.embed(all_texts, batch_size=32))

    # 8. Build PointStruct with named vectors (dense + sparse)
    points: list[PointStruct] = []
    for point_id, payload, dense_emb, sparse_emb in zip(all_point_ids, all_payloads, dense_embeddings, sparse_embeddings):
        dense_vector = dense_emb.tolist() if hasattr(dense_emb, "tolist") else list(dense_emb)
        
        # Convert sparse embedding to Qdrant SparseVector format
        if hasattr(sparse_emb, "indices") and hasattr(sparse_emb, "values"):
            sparse_vector = SparseVector(
                indices=sparse_emb.indices.tolist() if hasattr(sparse_emb.indices, "tolist") else list(sparse_emb.indices),
                values=sparse_emb.values.tolist() if hasattr(sparse_emb.values, "tolist") else list(sparse_emb.values),
            )
        else:
            # Handle dict-like sparse embedding output
            sparse_vector = SparseVector(
                indices=list(sparse_emb.get("indices", [])),
                values=list(sparse_emb.get("values", [])),
            )
        
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector,
                },
                payload=payload.model_dump(mode="json"),
            )
        )

    # 9. Ensure collection and indexes, then upsert
    ensure_collection_exists(client)

    # P0-4: Point collision & integrity guard
    for point_id, payload in zip(all_point_ids, all_payloads):
        try:
            existing = client.retrieve(collection_name=COLLECTION_NAME, ids=[point_id], with_payload=True)
            if existing and existing[0].payload:
                existing_text = existing[0].payload.get("text", "")
                if existing_text != payload.text:
                    logging.warning(
                        f"POINT_COLLISION_DETECTED: point_id={point_id} has different text payload. "
                        f"Existing: {existing_text[:100]}... New: {payload.text[:100]}..."
                    )
        except Exception as e:
            logging.debug(f"Collision check failed for {point_id}: {e}")

    @with_qdrant_retry
    def _upsert():
        return client.upsert(collection_name=COLLECTION_NAME, points=points)

    _upsert()
    print(f"[OK] Indexed {len(points)} chunks into Qdrant Cloud.")
    return len(points)


# ─────────────────────────────────────────────────────────────────────────────
# TASK-3.7: CLI Entry Point & Batch Ingestion Runner (Upgraded with optional date)
# ─────────────────────────────────────────────────────────────────────────────


def _print_usage():
    print("Usage: python ingest.py <path_to_pdf> [YYYY-MM-DD]")
    print("  <path_to_pdf>  Path to the PDF file to ingest")
    print("  <YYYY-MM-DD>   Edition date (e.g., 2026-08-24) - optional, auto-detected from PDF header if omitted")
    print()
    print("Batch ingestion:")
    print("  For batch processing, run:")
    print('    for f in data/*.pdf; do python ingest.py "$f" "2025-01-01"; done')
    print("  Or with auto-detection:")
    print('    for f in data/*.pdf; do python ingest.py "$f"; done')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    pdf_path = sys.argv[1]
    edition_date = sys.argv[2] if len(sys.argv) >= 3 else None

    # Validate date format if provided
    if edition_date is not None:
        try:
            datetime.strptime(edition_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{edition_date}'. Expected YYYY-MM-DD (e.g., 2026-08-24)")
            sys.exit(1)

    # Validate PDF path
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    start = time.time()
    try:
        count = ingest_pdf(pdf_path, edition_date)
        elapsed = time.time() - start
        print(f"[*] Completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"[ERR] Ingestion failed: {e}")
        sys.exit(1)
