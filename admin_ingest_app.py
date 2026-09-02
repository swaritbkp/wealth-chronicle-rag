"""
admin_ingest_app.py — Dedicated Admin Ingestion Console & Corpus Management Cockpit.
Independent administrative tool for WealthChronicle AI v1.0.
Provides document staging, batch vectorization, cluster diagnostics, and collection management.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import streamlit as st
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from ingest import (
    COLLECTION_NAME,
    clean_extracted_text,
    ensure_collection_exists,
    ensure_payload_indexes,
    extract_edition_date_from_text,
    extract_pages,
    generate_chunk_id,
    generate_point_id,
    sliding_window_chunk,
    validate_extraction,
)
from schemas import ChunkPayload

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration & Admin Theme Styling
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WealthChronicle Admin | Ingestion Cockpit",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger("wealthchronicle.admin")
logging.basicConfig(level=logging.INFO)

# Scoped CSS for Dark Slate Administration Cockpit
st.markdown(
    """
    <style>
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #2b313e !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }

    .admin-badge-indexed {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .admin-badge-unindexed {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .cluster-status-online {
        color: #10b981;
        font-weight: 700;
    }
    .cluster-status-offline {
        color: #ef4444;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Service Initialization & Client Setup
# ─────────────────────────────────────────────────────────────────────────────


def get_qdrant_admin_client() -> tuple[QdrantClient | None, str | None]:
    """Initialize Qdrant client using admin credentials from secrets or environment."""
    qdrant_url = (
        st.secrets.get("QDRANT_URL")
        or os.environ.get("QDRANT_URL")
        or "http://localhost:6333"
    )
    admin_key = (
        st.secrets.get("QDRANT_ADMIN_KEY")
        or st.secrets.get("QDRANT_API_KEY")
        or os.environ.get("QDRANT_ADMIN_KEY")
        or os.environ.get("QDRANT_API_KEY")
        or st.secrets.get("QDRANT_READ_KEY")
        or os.environ.get("QDRANT_READ_KEY")
    )

    try:
        client = QdrantClient(url=qdrant_url, api_key=admin_key)
        return client, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def init_embedding_models():
    """Cache dense and BM42 sparse embedding models for administrative vectorization."""
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
    return dense_model, sparse_model


client, client_err = get_qdrant_admin_client()
dense_model, sparse_model = init_embedding_models()


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics & Cluster Inspection Helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_cluster_diagnostics(client: QdrantClient | None) -> dict[str, Any]:
    """Retrieve collection points count, indexed status, and connectivity."""
    if client is None:
        return {"online": False, "points": 0, "editions": 0, "error": "Client not initialized"}

    try:
        if not client.collection_exists(COLLECTION_NAME):
            return {"online": True, "points": 0, "editions": 0, "error": "Collection does not exist"}

        info = client.get_collection(COLLECTION_NAME)
        points_count = info.points_count or 0

        # Discover unique editions via payload scroll
        unique_editions: set[str] = set()
        try:
            scroll_res = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=["edition_date"],
                with_vectors=False,
            )
            for record in scroll_res[0]:
                if record.payload and "edition_date" in record.payload:
                    unique_editions.add(str(record.payload["edition_date"]))
        except Exception:
            pass

        return {
            "online": True,
            "points": points_count,
            "editions": len(unique_editions),
            "error": None,
        }
    except Exception as e:
        return {"online": False, "points": 0, "editions": 0, "error": str(e)}


def scan_corpus_directory(client: QdrantClient | None, data_dir: str = "data") -> list[dict[str, Any]]:
    """Scan data/ for PDF files and query Qdrant index status for each document."""
    p_dir = Path(data_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(p_dir.glob("*.pdf"))
    items: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        detected_date: str | None = None

        # Attempt masthead date extraction
        try:
            pages = extract_pages(str(pdf_path))
            if pages:
                detected_date = extract_edition_date_from_text(pages[0].get("text", ""))
        except Exception:
            detected_date = None

        if not detected_date:
            m = re.search(r"(\d{4}-\d{2}-\d{2})", pdf_path.stem)
            if m:
                detected_date = m.group(1)

        # Query Qdrant for indexed count for this file / edition
        indexed_chunks = 0
        is_indexed = False

        if client and detected_date:
            try:
                if client.collection_exists(COLLECTION_NAME):
                    count_resp = client.count(
                        collection_name=COLLECTION_NAME,
                        count_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="edition_date",
                                    match=models.MatchValue(value=detected_date),
                                )
                            ]
                        ),
                    )
                    indexed_chunks = count_resp.count
                    is_indexed = indexed_chunks > 0
            except Exception:
                pass

        items.append(
            {
                "Filename": pdf_path.name,
                "Path": str(pdf_path),
                "Size (MB)": round(size_mb, 2),
                "Detected Date": detected_date or "Unknown",
                "Status": "Indexed" if is_indexed else "Unindexed",
                "Indexed Chunks": indexed_chunks,
            }
        )

    return items


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion Worker Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline_for_pdf(
    pdf_path: str,
    edition_date: str,
    client: QdrantClient,
    dense_model: Any,
    sparse_model: Any,
) -> int:
    """Execute full 5-stage ingestion for a single PDF issue."""
    # 1. Extraction & Sanitization
    pages = extract_pages(pdf_path)
    for p in pages:
        p["text"] = clean_extracted_text(p.get("text", ""))

    validate_extraction(pages, pdf_path)

    # 2. Layout-Aware Chunking
    all_texts: list[str] = []
    all_payloads: list[ChunkPayload] = []
    all_point_ids: list[str] = []

    for item in pages:
        meta = item.get("metadata", {})
        page_0 = meta.get("page", 0)
        page_num = page_0 + 1
        raw_text = item["text"].strip()
        if not raw_text:
            continue

        chunks = sliding_window_chunk(raw_text, chunk_size=600, overlap=100, min_chars=120)
        for c_idx, chunk_text in enumerate(chunks):
            pid = generate_point_id(edition_date, page_num, c_idx, chunk_text[:50])
            cid = generate_chunk_id(edition_date, page_num, c_idx)
            has_table = "|" in chunk_text and "\n|---" in chunk_text

            payload = ChunkPayload(
                chunk_id=cid,
                edition_date=edition_date,
                page_number=page_num,
                text=chunk_text,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
                source=Path(pdf_path).name,
                article_title=f"Page {page_num}",
                has_table=has_table,
            )
            all_texts.append(chunk_text)
            all_payloads.append(payload)
            all_point_ids.append(pid)

    if not all_texts:
        return 0

    # 3. Vectorization with batch_size=32
    dense_embs = list(dense_model.embed(all_texts, batch_size=32))
    sparse_embs = list(sparse_model.embed(all_texts, batch_size=32))

    # 4. Upsert Points
    points: list[models.PointStruct] = []
    for i in range(len(all_texts)):
        dense_vec = dense_embs[i].tolist() if hasattr(dense_embs[i], "tolist") else list(dense_embs[i])
        sp = sparse_embs[i]
        sparse_vec = models.SparseVector(
            indices=sp.indices.tolist() if hasattr(sp.indices, "tolist") else list(sp.indices),
            values=sp.values.tolist() if hasattr(sp.values, "tolist") else list(sp.values),
        )
        points.append(
            models.PointStruct(
                id=all_point_ids[i],
                vector={"dense": dense_vec, "sparse": sparse_vec},
                payload=all_payloads[i].model_dump(),
            )
        )

    ensure_collection_exists(client, COLLECTION_NAME)
    ensure_payload_indexes(client, COLLECTION_NAME)

    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    return len(points)


# ─────────────────────────────────────────────────────────────────────────────
# Purge / Recreate Collection Modal Dialog
# ─────────────────────────────────────────────────────────────────────────────

if hasattr(st, "dialog"):

    @st.dialog("Purge / Recreate Collection")
    def purge_collection_modal(client: QdrantClient) -> None:
        st.warning("⚠️ **DANGER ZONE:** This action will permanently delete all indexed points, vectors, and payload schemas in `wealth_archive`.")
        st.write("Type **CONFIRM PURGE** below to authorize collection recreation:")

        confirm_text = st.text_input("Confirmation Phrase", placeholder="CONFIRM PURGE")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚨 Recreate Collection", use_container_width=True, type="primary"):
                if confirm_text.strip() == "CONFIRM PURGE":
                    try:
                        if client.collection_exists(COLLECTION_NAME):
                            client.delete_collection(COLLECTION_NAME)
                        ensure_collection_exists(client, COLLECTION_NAME)
                        ensure_payload_indexes(client, COLLECTION_NAME)
                        st.success("✅ Collection `wealth_archive` successfully purged and recreated!")
                        time.sleep(1.0)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Purge failed: {e}")
                else:
                    st.error("Incorrect confirmation phrase. Action cancelled.")
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Admin Layout Header & Diagnostic Ribbon
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚙️ WealthChronicle Admin Cockpit")
st.markdown("##### *Corpus Ingestion, FastEmbed Vectorization & Cluster Diagnostics*")

diag = get_cluster_diagnostics(client)

with st.container(border=True):
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        status_html = '<span class="cluster-status-online">● ONLINE</span>' if diag["online"] else '<span class="cluster-status-offline">● OFFLINE</span>'
        st.markdown(f"**Qdrant Cloud:** {status_html}", unsafe_allow_html=True)
        if diag.get("error"):
            st.caption(f"Error: {diag['error']}")
        else:
            st.caption(f"Collection: `{COLLECTION_NAME}` | Host: AWS eu-central-1")
    with col2:
        st.metric("Total Points", f"{diag['points']:,}")
    with col3:
        st.metric("Indexed Editions", f"{diag['editions']}")
    with col4:
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024 * 1024)
        st.metric("Admin RAM RSS", f"{ram_mb:.1f} MB")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: File Upload & Staging
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 📥 1. Document Staging & File Uploader")
st.caption("Upload raw Economic Times Wealth publication issues into the local `data/` directory for parsing.")

uploaded_files = st.file_uploader(
    "Upload ET Wealth PDF Editions",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF files. They will be saved to data/ and sanitized.",
)

if uploaded_files:
    saved_count = 0
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        sanitized_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", uploaded_file.name)
        target_path = data_dir / sanitized_name

        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_count += 1

    st.success(f"✅ Successfully staged {saved_count} file(s) into `data/`.")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Corpus Directory & Index Status Inspector
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 📊 2. Corpus Directory & Index Status")
st.caption("Live comparison between local files in `data/` and indexed points in Qdrant Cloud.")

corpus_items = scan_corpus_directory(client, data_dir="data")

if not corpus_items:
    st.info("No PDF files found in `data/`. Upload files above to stage documents.")
else:
    # Display table of files
    display_rows = [
        {
            "Filename": item["Filename"],
            "Size (MB)": item["Size (MB)"],
            "Detected Date": item["Detected Date"],
            "Status": item["Status"],
            "Chunks": item["Indexed Chunks"],
        }
        for item in corpus_items
    ]
    st.dataframe(display_rows, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Batch Pipeline Ingestion
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 🚀 3. Batch Vector Ingestion Pipeline")
st.caption("Run PyMuPDF layout parsing, table isolation, SIMD dual-vectorization (`batch_size=32`), and idempotent upsert.")

if corpus_items and client:
    ingest_mode = st.radio(
        "Ingestion Scope",
        options=["Ingest Unindexed Only", "Ingest All Staged Files", "Select Specific Documents"],
        horizontal=True,
    )

    targets: list[dict[str, Any]] = []
    if ingest_mode == "Ingest Unindexed Only":
        targets = [it for it in corpus_items if it["Status"] == "Unindexed"]
    elif ingest_mode == "Ingest All Staged Files":
        targets = corpus_items
    else:
        selected_names = st.multiselect(
            "Choose documents to ingest",
            options=[it["Filename"] for it in corpus_items],
            default=[it["Filename"] for it in corpus_items if it["Status"] == "Unindexed"],
        )
        targets = [it for it in corpus_items if it["Filename"] in selected_names]

    st.write(f"**Target Documents Selected:** `{len(targets)}` issue(s)")

    if st.button("⚡ Start Vector Ingestion", type="primary", disabled=len(targets) == 0):
        progress_bar = st.progress(0.0)
        status_box = st.status(f"Starting batch ingestion for {len(targets)} documents...", expanded=True)

        total_indexed_all = 0
        start_time = time.time()

        for idx, doc in enumerate(targets):
            fname = doc["Filename"]
            fpath = doc["Path"]
            ed_date = doc["Detected Date"]

            if not ed_date or ed_date == "Unknown":
                ed_date = datetime.now().strftime("%Y-%m-%d")

            status_box.write(f"📄 Processing **{fname}** (Edition: `{ed_date}`)...")
            try:
                num_chunks = run_pipeline_for_pdf(
                    pdf_path=fpath,
                    edition_date=ed_date,
                    client=client,
                    dense_model=dense_model,
                    sparse_model=sparse_model,
                )
                total_indexed_all += num_chunks
                status_box.write(f"✅ **{fname}**: Indexed `{num_chunks}` chunks into Qdrant.")
            except Exception as e:
                status_box.write(f"❌ Error processing **{fname}**: {e}")

            progress_bar.progress((idx + 1) / len(targets))

        elapsed = time.time() - start_time
        status_box.update(
            label=f"Ingestion Completed! Indexed {total_indexed_all} total chunks across {len(targets)} files in {elapsed:.1f}s.",
            state="complete",
            expanded=False,
        )
        st.success(f"🎉 Batch Vector Ingestion Finished! {total_indexed_all} chunks indexed in {elapsed:.1f}s.")
        time.sleep(1.5)
        st.rerun()

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Collection Maintenance & Danger Zone
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 🛠️ 4. Cluster Maintenance & Schema Reset")
st.caption("Perform administrative schema creation or emergency point purging.")

maintenance_col1, maintenance_col2 = st.columns(2)

with maintenance_col1:
    with st.container(border=True):
        st.markdown("#### 🔄 Ensure Payload Indexes")
        st.caption("Verify and create `edition_date` (KEYWORD), `has_table` (BOOL), `page_number` (INTEGER), `source` (KEYWORD).")
        if st.button("Verify Payload Indexes", use_container_width=True):
            if client:
                try:
                    ensure_collection_exists(client, COLLECTION_NAME)
                    ensure_payload_indexes(client, COLLECTION_NAME)
                    st.success("✅ Payload indexes verified and active.")
                except Exception as e:
                    st.error(f"Failed to verify indexes: {e}")

with maintenance_col2:
    with st.container(border=True):
        st.markdown("#### 🚨 Purge / Recreate Collection")
        st.caption("Permanently delete collection `wealth_archive` and rebuild from scratch.")
        if st.button("Open Purge Dialog", use_container_width=True):
            if client and hasattr(st, "dialog"):
                purge_collection_modal(client)
