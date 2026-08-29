"""
ingest.py — Admin Ingestion Plane for WealthChronicle AI v1.0
Covers TASK-3.1 through TASK-3.7
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pymupdf4llm
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
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
# TASK-3.2: Punctuation-Aware Sliding Window Chunker
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"[.!?;]\s")
_NOISE_PREFIXES = ("advertisement", "subscribe", "page ", "epaper")


def sliding_window_chunk(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
    min_chars: int = 120,
) -> list[str]:
    """Sliding window chunker with punctuation-aware boundary snapping.

    Algorithm:
        1. Split text into words.
        2. Advance by (chunk_size - overlap) words per step.
        3. For each chunk boundary, search backward up to 30 words
           for a sentence-ending punctuation mark and snap to it.
        4. Discard chunks below min_chars or matching noise patterns.

    Args:
        text: Full page Markdown text.
        chunk_size: Target words per chunk (S).
        overlap: Overlapping words between consecutive chunks (O).
        min_chars: Minimum character length to retain a chunk.

    Returns:
        List of clean chunk strings.
    """
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

        # --- Noise filter (FR-ING-03) ---
        if len(chunk_text) >= min_chars and not chunk_text.lower().startswith(_NOISE_PREFIXES):
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
# TASK-3.5: Qdrant Collection Initialization with HNSW Config & Payload Indexes
# ─────────────────────────────────────────────────────────────────────────────


def init_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """Initialize Qdrant collection with HNSW config.

    Creates collection if it does not exist, with:
        - VectorParams(size=384, distance=Distance.COSINE)
        - HnswConfigDiff(m=16, ef_construct=128, full_scan_threshold=10_000)
        - OptimizersConfigDiff(indexing_threshold=20_000, memmap_threshold=50_000)
    """
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=128,
            full_scan_threshold=10_000,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20_000,
            memmap_threshold=50_000,
        ),
    )
    print(f"[OK] Created collection: {collection_name}")


def ensure_payload_indexes(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """Create payload indexes for filtering and sorting.

    Creates keyword index on edition_date, integer index on page_number,
    and keyword index on source. All calls are idempotent.
    """
    indexes = [
        {"field_name": "edition_date", "field_schema": PayloadSchemaType.KEYWORD},
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


# ─────────────────────────────────────────────────────────────────────────────
# TASK-3.6: Batch Embed & Upsert Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def ingest_pdf(pdf_path: str, edition_date: str, client: QdrantClient | None = None) -> int:
    """Full ingestion pipeline for a single PDF.

    Steps:
        1. Call extract_pages(pdf_path)
        2. Call validate_extraction(pages, pdf_path)
        3. For each page: call sliding_window_chunk(text)
        4. Convert 0-indexed to 1-indexed page numbers.
        5. Generate chunk_id and point_id for each chunk.
        6. Construct ChunkPayload Pydantic objects
        7. Embed all texts using TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        8. Build PointStruct list with vector + payload dict
        9. Call client.upsert with @with_qdrant_retry

    Returns:
        Number of chunks ingested.
    """
    import os

    # Validate edition_date format
    try:
        datetime.strptime(edition_date, "%Y-%m-%d")
    except ValueError:
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

        chunks = sliding_window_chunk(text, chunk_size=600, overlap=100, min_chars=120)

        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_id = generate_chunk_id(edition_date, page_number, chunk_idx)
            point_id = generate_point_id(edition_date, page_number, chunk_idx, chunk_text)

            char_count = len(chunk_text)
            word_count = len(chunk_text.split())

            payload = ChunkPayload(
                chunk_id=chunk_id,
                edition_date=edition_date,
                page_number=page_number,
                text=chunk_text,
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

    # 7. Embed
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embeddings = list(embedding_model.embed(all_texts))

    # 8. Build PointStruct
    points: list[PointStruct] = []
    for point_id, payload, emb in zip(all_point_ids, all_payloads, embeddings):
        vector = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload.model_dump(mode="json"),
            )
        )

    # 9. Ensure collection and indexes, then upsert
    init_collection(client)
    ensure_payload_indexes(client)

    @with_qdrant_retry
    def _upsert():
        return client.upsert(collection_name=COLLECTION_NAME, points=points)

    _upsert()
    print(f"[OK] Indexed {len(points)} chunks into Qdrant Cloud.")
    return len(points)


# ─────────────────────────────────────────────────────────────────────────────
# TASK-3.7: CLI Entry Point & Batch Ingestion Runner
# ─────────────────────────────────────────────────────────────────────────────


def _print_usage():
    print("Usage: python ingest.py <path_to_pdf> <YYYY-MM-DD>")
    print("  <path_to_pdf>  Path to the PDF file to ingest")
    print("  <YYYY-MM-DD>   Edition date (e.g., 2026-08-24)")
    print()
    print("Batch ingestion:")
    print("  For batch processing, run:")
    print('    for f in data/*.pdf; do python ingest.py "$f" "2025-01-01"; done')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _print_usage()
        sys.exit(1)

    pdf_path = sys.argv[1]
    edition_date = sys.argv[2]

    # Validate date format
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
