"""
app.py — Public Streamlit Application for WealthChronicle AI v1.0
Institutional-Grade Financial Intelligence Terminal.
Covers TASK-4.1 through TASK-4.5.
Uses Qdrant native hybrid search (BM42 sparse + dense vectors) and payload-indexed filtering.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from datetime import date, datetime, timezone

import google.generativeai as genai
import psutil
import streamlit as st
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from engine import (
    GeminiRateLimiter,
    QueryTrace,
    check_memory_usage,
    extract_citation_spans_from_text,
    get_qdrant_client,
    get_storage_mode,
    hybrid_search,
    load_and_validate_prompts,
    rerank_candidates,
    should_refuse,
    synthesize_with_streaming_and_validation,
    timer,
)
from schemas import ChunkPayload, CitationMetadata, RerankedPassage
from telemetry import log_query_audit

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration & Terminal Theme Styling
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WealthChronicle AI | Institutional Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger("wealthchronicle.trace")
logging.basicConfig(level=logging.INFO)

# Scoped CSS for Modern Slate Financial Intelligence Terminal
st.markdown(
    """
    <style>
    /* Scoped Typography & Base */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Bordered Container Muted Slate Accents */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #2b313e !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }

    /* Match Quality Badges */
    .badge-conf-high {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-conf-low {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Source Metadata Badges */
    .source-pill {
        display: inline-block;
        padding: 2px 6px;
        margin: 1px 3px;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 600;
        font-family: ui-monospace, monospace;
    }
    .pill-date {
        background: rgba(59, 130, 246, 0.12);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }
    .pill-page {
        background: rgba(168, 85, 247, 0.12);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.25);
    }
    .pill-score {
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }

    /* Refusal Compliance Card */
    .refusal-container {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(217, 119, 6, 0.02));
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 14px;
        margin-top: 8px;
    }
    .refusal-header {
        color: #f59e0b;
        font-weight: 700;
        font-size: 0.92rem;
        margin-bottom: 6px;
    }
    .refusal-text {
        color: #f1f5f9;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .refusal-note {
        color: #94a3b8;
        font-size: 0.74rem;
        margin-top: 8px;
        border-top: 1px dashed rgba(148, 163, 184, 0.2);
        padding-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# TASK-4.1: Service Initialization & Cache Setup
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def init_services():
    """Initialize all services with caching.

    Returns:
        tuple: (gemini_model, qdrant_client, dense_embedding_model, sparse_embedding_model, ranker, prompts, rate_limiter, ranker_available)
    """
    # Load and validate prompts
    prompts = load_and_validate_prompts("config/prompts.yaml")

    # Configure Gemini
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    # Qdrant client (automatic cloud connection with local disk fallback)
    qdrant_url = st.secrets.get("QDRANT_URL") or os.environ.get("QDRANT_URL")
    qdrant_key = st.secrets.get("QDRANT_READ_KEY") or os.environ.get("QDRANT_READ_KEY")
    qdrant_client, _ = get_qdrant_client(url=qdrant_url, api_key=qdrant_key)

    # FastEmbed - Dense
    dense_embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # FastEmbed - Sparse (BM42)
    sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

    # FlashRank with fallback
    ranker = None
    ranker_available = False
    try:
        from flashrank import Ranker

        cache_dir = os.path.join(tempfile.gettempdir(), "flashrank_models")
        ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=cache_dir)
        ranker_available = True
    except Exception as e:
        logging.warning(f"FlashRank init failed, degrading to RRF-only: {e}")
        ranker_available = False

    # Rate limiter
    rate_limiter = GeminiRateLimiter(max_rpm=14)

    return (
        gemini_model,
        qdrant_client,
        dense_embedding_model,
        sparse_embedding_model,
        ranker,
        prompts,
        rate_limiter,
        ranker_available,
    )


# Initialize services with error handling
try:
    (
        gemini_model,
        qdrant_client,
        dense_embedding_model,
        sparse_embedding_model,
        ranker,
        prompts,
        rate_limiter,
        ranker_available,
    ) = init_services()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Modal Dialog: Document Inspection Drilldown
# ─────────────────────────────────────────────────────────────────────────────

if hasattr(st, "dialog"):

    @st.dialog("Document Inspection")
    def inspect_document_modal(
        title: str,
        edition_date: str,
        page_number: int,
        text: str,
        score: float,
        has_table: bool,
    ) -> None:
        """Modal dialog for deep inspection of retrieved publication passages."""
        st.markdown(f"#### 📄 {title or f'Publication Edition {edition_date} — Page {page_number}'}")

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Edition Date", str(edition_date))
            with c2:
                st.metric("Page Number", str(page_number))
            with c3:
                st.metric("Match Score", f"{score:.4f}")

        if has_table or "|" in text:
            tab1, tab2 = st.tabs(["Formatted Table", "Raw Source Context"])
            with tab1:
                st.markdown(text)
            with tab2:
                st.code(text, language="markdown")
        else:
            st.markdown(text)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Component: Telemetry Ribbon Container
# ─────────────────────────────────────────────────────────────────────────────


def render_telemetry_ribbon(telem: dict) -> None:
    """Renders structured telemetry metrics in a 5-column bordered container."""
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([1.5, 1.2, 1, 1, 1])
        with c1:
            st.metric(
                "Total Latency",
                f"{telem.get('total_ms', 0):.0f} ms",
                help="End-to-end retrieval, reranking, and generation latency.",
            )
        with c2:
            ttft = telem.get("ttft_ms")
            ttft_str = f"{ttft:.0f} ms" if ttft and ttft > 0 else "N/A"
            st.metric(
                "TTFT",
                ttft_str,
                help="Time-to-First-Token generation latency.",
            )
        with c3:
            st.metric(
                "Top-1 Score",
                f"{telem.get('top_score', 0.0):.4f}",
                help="Cross-encoder relevance score of top candidate.",
            )
        with c4:
            st.metric(
                "Temporal Boost",
                f"{telem.get('time_decay', 1.0):.2f}x",
                help="Exponential recency decay multiplier based on edition date.",
            )
        with c5:
            is_refused = telem.get("refused", False)
            st.metric(
                "Safety Gate",
                "Refused" if is_refused else "Passed",
                help="Deterministic threshold evaluation (θ = 0.25).",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Helper Component: Source Inspection Expander
# ─────────────────────────────────────────────────────────────────────────────


def render_citation_expander(citations: list[dict], message_idx: int | str) -> None:
    """Renders verified source passages inside an expander with confidence badges and table tabs."""
    with st.expander("🔍 View Verified Source Passages"):
        for p_idx, c in enumerate(citations, start=1):
            is_high = c["cross_encoder_score"] >= 0.60
            conf_badge = (
                '<span class="badge-conf-high">Confidence: High</span>'
                if is_high
                else '<span class="badge-conf-low">Confidence: Low</span>'
            )
            st.markdown(
                f"""
                <div style="margin-bottom: 6px;">
                    <span style="font-weight: 700; color: #f1f5f9;">Passage {p_idx}</span> &nbsp;
                    {conf_badge} &nbsp;
                    <span class="source-pill pill-date">📅 {c['edition_date']}</span>
                    <span class="source-pill pill-page">📄 Page {c['page_number']}</span>
                    <span class="source-pill pill-score">⭐ Score {c['cross_encoder_score']:.4f}</span>
                    {f"<span style='color: #cbd5e1; font-weight: 500; margin-left: 6px;'>{c['article_title']}</span>" if c.get('article_title') else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

            passage_text = c["text"]
            if c.get("has_table") or "|" in passage_text:
                tab1, tab2 = st.tabs(["Formatted Table", "Raw Source Context"])
                with tab1:
                    st.markdown(passage_text)
                with tab2:
                    st.code(passage_text, language="markdown")
            else:
                st.caption(passage_text)

            if hasattr(st, "dialog"):
                btn_key = f"inspect_{message_idx}_{c['edition_date']}_{c['page_number']}_{p_idx}"
                if st.button(f"🔍 Document Inspection (Passage {p_idx})", key=btn_key):
                    inspect_document_modal(
                        title=c.get("article_title", ""),
                        edition_date=str(c["edition_date"]),
                        page_number=c["page_number"],
                        text=passage_text,
                        score=c["cross_encoder_score"],
                        has_table=bool(c.get("has_table", False)),
                    )

            st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Terminal Telemetry & Payload Metadata Filtering
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏛️ Terminal Telemetry & Filters")
    st.caption("Hardware-isolated read plane connected to Qdrant Cloud.")

    # Live telemetry cards
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RAM RSS", f"{rss_mb:.1f} MB")
        with col2:
            st.metric("Storage", get_storage_mode())

    st.divider()

    st.markdown("#### 🎯 Retrieval Payload Filters")
    st.caption("Server-side Qdrant payload indexes (`edition_date`, `has_table`, `page_number`).")

    # Date filter
    filter_date = st.text_input(
        "Filter by Edition Date (YYYY-MM-DD)",
        placeholder="e.g., 2026-08-24",
        help="Filters chunks strictly matching the specified publication edition date.",
    ).strip()

    # Table-only focus toggle
    filter_tables_only = st.checkbox(
        "📊 Tables & Structured Data Only",
        value=False,
        help="Restricts search to chunks flagged with has_table=True (payload boolean index).",
    )

    # Retrieval depth slider
    retrieval_k = st.slider(
        "Candidate Retrieval Depth (K)",
        min_value=5,
        max_value=20,
        value=12,
        step=1,
        help="Number of dense and sparse candidates fetched prior to RRF fusion and reranking.",
    )

    st.divider()

    # Session control
    if st.button("🔄 Reset Terminal Session", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["query_count"] = 0
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #94a3b8; line-height: 1.4;">
        <b>Engine Architecture:</b><br/>
        • Hybrid BM42 Sparse + BAAI Dense<br/>
        • Reciprocal Rank Fusion (k=60)<br/>
        • Temporal Decay (α=0.35, τ=365d)<br/>
        • FlashRank Cross-Encoder (θ=0.25)<br/>
        • Strict Citation Grounding
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Terminal UI & Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 WealthChronicle Search")
st.markdown("##### *Institutional Financial Intelligence & Archive Research Terminal*")
st.info("⚠️ **Disclaimer:** Educational research tool indexing archived publications. Does not constitute registered financial, legal, or tax advisory services.")

# Defensive session state initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "query_count" not in st.session_state:
    st.session_state["query_count"] = 0

if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None


# ─────────────────────────────────────────────────────────────────────────────
# Empty State UX: Suggested Research Inquiries (Bordered Container, 4 Columns)
# ─────────────────────────────────────────────────────────────────────────────

if len(st.session_state["messages"]) == 0:
    with st.container(border=True):
        st.markdown("#### 💡 Institutional Research Inquiries")
        st.caption("Select a prompt to execute hybrid dense + BM42 retrieval against verified financial archives:")

        c1, c2, c3, c4 = st.columns(4)
        sample_queries = [
            ("💼 Tax Slabs FY 2025-26", "New vs Old Tax Regime slabs for FY 2025-26"),
            ("📈 Capital Gains on Debt Funds", "Capital gains taxation changes on Arbitrage & Debt funds"),
            ("🏥 Senior Health Insurance", "Health insurance deductible limits for senior citizens"),
            ("🌐 FAST-DS Asset Reporting", "FAST-DS foreign asset reporting guidelines"),
        ]
        cols = [c1, c2, c3, c4]
        for idx, (label, query_val) in enumerate(sample_queries):
            if cols[idx].button(label, key=f"chip_{idx}", use_container_width=True):
                st.session_state["pending_query"] = query_val
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Render Chat History with Telemetry & Citations
# ─────────────────────────────────────────────────────────────────────────────

for msg_idx, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "telemetry" in msg:
            render_telemetry_ribbon(msg["telemetry"])

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "citations" in msg and msg["citations"]:
            render_citation_expander(msg["citations"], message_idx=msg_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Query Input & Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.get("pending_query"):
    user_query = st.session_state["pending_query"]
    st.session_state["pending_query"] = None
else:
    user_query = st.chat_input("Ask about tax slabs, capital gains, health claim rejections, NPS allocations...")

if user_query:
    # TASK-4.5: Session state pruning & memory guard — cap history
    MAX_MESSAGES = 20
    st.session_state["messages"].append({"role": "user", "content": user_query})
    if len(st.session_state["messages"]) > MAX_MESSAGES:
        st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]

    st.session_state["query_count"] += 1

    # Periodic memory guard every 10 queries
    if st.session_state["query_count"] % 10 == 0:
        try:
            check_memory_usage()
        except Exception as e:
            logging.warning(f"Memory check failed: {e}")

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Initialize execution trace
        trace = QueryTrace(
            trace_id=str(uuid.uuid4()),
            query=user_query,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        overall_start = datetime.now(timezone.utc)

        try:
            # ─── 1. Build Payload Filter ───
            filter_conditions: list[models.Condition] = []
            if filter_date:
                filter_conditions.append(
                    models.FieldCondition(
                        key="edition_date",
                        match=models.MatchValue(value=filter_date),
                    )
                )
            if filter_tables_only:
                filter_conditions.append(
                    models.FieldCondition(
                        key="has_table",
                        match=models.MatchValue(value=True),
                    )
                )

            query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

            # ─── Multi-Step Progress Tracking with st.status ───
            with st.status("Executing hybrid retrieval pipeline...", expanded=True) as status_box:
                # Hybrid Search
                status_box.write("⚡ Prefetching Hybrid Vectors (Dense BAAI 384d + Sparse BM42)...")
                with timer(trace, "hybrid_search_ms"):
                    fused_candidates = hybrid_search(
                        client=qdrant_client,
                        query=user_query,
                        dense_embedding_model=dense_embedding_model,
                        sparse_embedding_model=sparse_embedding_model,
                        collection_name="wealth_archive",
                        limit=retrieval_k,
                        reference_date=date.today(),
                        query_filter=query_filter,
                    )
                    trace.fused_candidates = len(fused_candidates)

                # Reranking
                status_box.write("🔄 Computing RRF & Temporal Decay Fusion...")
                status_box.write("🎯 FlashRank TinyBERT Cross-Encoder Reranking...")
                with timer(trace, "reranking_ms"):
                    fused_ids = [c["point_id"] for c in fused_candidates]
                    payload_map: dict[str, ChunkPayload] = {}
                    if fused_ids:
                        retrieved = qdrant_client.retrieve(
                            collection_name="wealth_archive",
                            ids=fused_ids,
                            with_payload=True,
                        )
                        for p in retrieved:
                            try:
                                if p.payload:
                                    payload = ChunkPayload(**p.payload)
                                    payload_map[str(p.id)] = payload
                            except Exception:
                                continue

                    reranked: list[RerankedPassage] = rerank_candidates(
                        query=user_query,
                        candidates=fused_candidates,
                        payload_map=payload_map,
                        ranker=ranker,
                        top_k=4,
                    )
                    trace.reranked_top_k = len(reranked)
                    trace.top1_cross_encoder_score = reranked[0].cross_encoder_score if reranked else 0.0

                # Deterministic Refusal Check
                status_box.write("🛡️ Evaluating Institutional Refusal Guardrails (θ = 0.25)...")
                refused = should_refuse(reranked, prompts)
                trace.refused = refused

                top_score = reranked[0].cross_encoder_score if reranked else 0.0
                time_decay = reranked[0].time_decay_multiplier if reranked else 1.0

                if not refused:
                    status_box.write("✨ Synthesizing grounded response with Gemini 2.5 Flash...")

                status_box.update(label="Retrieval & Ranking Pipeline Complete", state="complete", expanded=False)

            citations: list[dict] = []

            if refused:
                refusal_text = prompts.get(
                    "refusal_message",
                    "The indexed publication archives do not contain sufficient guidance to answer this question authoritatively.",
                )
                st.markdown(
                    f"""
                    <div class="refusal-container">
                        <div class="refusal-header">🛡️ Institutional Refusal Gate Triggered</div>
                        <div class="refusal-text">{refusal_text}</div>
                        <div class="refusal-note">
                            <b>Deterministic Guardrail Active:</b> Top cross-encoder relevance score ({top_score:.4f}) fell below institutional threshold (θ = 0.25). Synthesized hallucinations are suppressed.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                answer_text = refusal_text
                trace.answer_length_chars = len(answer_text)
                trace.citation_count = 0
            else:
                # ─── STREAMING GENERATION WITH LIVE TTFT + POST-STREAM VALIDATION ───
                with timer(trace, "prompt_assembly_ms"):
                    reranked_sorted = sorted(reranked, key=lambda x: x.payload.edition_date, reverse=True)
                    # Build streaming prompt (simpler, no JSON constraints)
                    context_passages_str = "\n\n---\n\n".join(
                        [
                            f"[Passage {i} | Edition: {p.payload.edition_date} | Page: {p.payload.page_number} | Section: {p.payload.article_title or 'Untitled'}]\n{p.text}"
                            for i, p in enumerate(reranked_sorted, start=1)
                        ]
                    )
                    full_prompt = f"{prompts['system_prompt']}\n\n" + prompts["rag_synthesis_template"].format(
                        context_passages=context_passages_str, query=user_query
                    ) if "rag_synthesis_template" in prompts else f"{prompts['system_prompt']}\n\n" + prompts["rag_prompt_template"].format(
                        context=context_passages_str, query=user_query
                    )

                guardrails = prompts.get("guardrails", {})
                generation_config = {
                    "temperature": guardrails.get("temperature", 0.1),
                    "top_p": guardrails.get("top_p", 0.95),
                    "top_k": 40,
                    "candidate_count": 1,
                }

                # Stream tokens live with TTFT tracking, then validate citations
                stream_gen = synthesize_with_streaming_and_validation(
                    model=gemini_model,
                    prompt=full_prompt,
                    rate_limiter=rate_limiter,
                    generation_config=generation_config,
                    context_passages=reranked_sorted,
                    system_prompt=prompts["system_prompt"],
                )
                
                # Stream to UI and capture citations from the special marker
                citation_metadata: list[CitationMetadata] = []
                answer_chunks = []
                
                for item in stream_gen:
                    if isinstance(item, dict) and "__citations__" in item:
                        citation_metadata = item["__citations__"]
                    else:
                        answer_chunks.append(item)
                
                answer_text = "".join(answer_chunks)
                
                # Fallback: extract citations from streamed text if validation failed
                if not citation_metadata:
                    citation_metadata = extract_citation_spans_from_text(answer_text, reranked_sorted)

                trace.llm_total_ms = 0

                # Convert citations to legacy format for UI
                citations = []
                for c in citation_metadata:
                    citations.append({
                        "edition_date": str(c.edition_date),
                        "page_number": c.page_number,
                        "cross_encoder_score": c.cross_encoder_score,
                        "text": c.excerpt_preview,
                        "article_title": c.article_title,
                        "has_table": False,
                    })

                # Display Telemetry Ribbon
                trace.total_ms = (datetime.now(timezone.utc) - overall_start).total_seconds() * 1000
                telemetry_data = {
                    "total_ms": trace.total_ms,
                    "ttft_ms": 0,
                    "top_score": top_score,
                    "time_decay": time_decay,
                    "refused": False,
                }
                render_telemetry_ribbon(telemetry_data)
                trace.answer_length_chars = len(answer_text)
                trace.citation_count = len(citations)

                # Render Citation Expander
                render_citation_expander(citations, message_idx=len(st.session_state["messages"]))

            # Finalize Trace & Persistent SQLite Telemetry Log
            trace.total_ms = (datetime.now(timezone.utc) - overall_start).total_seconds() * 1000
            trace.emit()
            log_query_audit(
                query_text=user_query,
                storage_mode=get_storage_mode(),
                top_score=top_score,
                gate_status="REFUSED" if refused else "PASSED",
                latency_ms=trace.total_ms,
                chunks_retrieved_count=len(citations),
                ttft_ms=trace.llm_ttft_ms,
            )

            # Append assistant message
            assistant_msg = {
                "role": "assistant",
                "content": answer_text,
                "telemetry": {
                    "total_ms": trace.total_ms,
                    "ttft_ms": trace.llm_ttft_ms,
                    "top_score": top_score,
                    "time_decay": time_decay,
                    "refused": refused,
                },
            }
            if citations:
                assistant_msg["citations"] = citations
            st.session_state["messages"].append(assistant_msg)

            if len(st.session_state["messages"]) > MAX_MESSAGES:
                st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]

        except Exception as e:
            logging.error(f"Query pipeline failed: {e}", exc_info=True)
            err_msg = f"An error occurred while processing your query: {e}"
            st.error(err_msg)
            trace.refused = True
            trace.emit()
            st.session_state["messages"].append({"role": "assistant", "content": err_msg})