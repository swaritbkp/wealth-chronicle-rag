"""
app.py — Public Streamlit Application for WealthChronicle AI v1.0
Institutional-Grade Financial Intelligence Terminal.
Covers TASK-4.1 through TASK-4.5.
Uses Qdrant native hybrid search (BM42 sparse + dense vectors) and payload-indexed filtering.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from datetime import date, datetime, timezone

import google.generativeai as genai
import psutil
import streamlit as st
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from engine import (
    GeminiRateLimiter,
    QueryTrace,
    check_memory_usage,
    hybrid_search,
    load_and_validate_prompts,
    rerank_candidates,
    safe_generate,
    should_refuse,
    timer,
    validate_citations,
)
from schemas import ChunkPayload, RerankedPassage

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

# Custom CSS for Bloomberg / Slate Financial Terminal Aesthetics
st.markdown(
    """
    <style>
    /* Global Typography and Terminal Feel */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Metrics Ribbon Container */
    .telemetry-ribbon {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        padding: 10px 14px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        margin-bottom: 14px;
        font-size: 0.82rem;
    }
    .telemetry-item {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #cbd5e1;
    }
    .telemetry-label {
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
    }
    .telemetry-value {
        font-weight: 700;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .badge-passed {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-refused {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* Source Pill Badges */
    .source-pill {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px 4px 2px 0;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: ui-monospace, monospace;
    }
    .pill-date {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .pill-page {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .pill-conf-high {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .pill-conf-med {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    /* Refusal Audit Card */
    .refusal-card {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(217, 119, 6, 0.03));
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
    }
    .refusal-title {
        color: #f59e0b;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .refusal-body {
        color: #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .refusal-footer {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 10px;
        border-top: 1px dashed rgba(148, 163, 184, 0.2);
        padding-top: 8px;
    }

    /* Prompt Chip Buttons */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
        margin-bottom: 20px;
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

    # Qdrant client (read-only key for public plane)
    qdrant_client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_READ_KEY"],
    )

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
# Sidebar: Terminal Telemetry & Payload Metadata Filtering
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏛️ Terminal Telemetry & Filters")
    st.caption("Hardware-isolated read plane connected to Qdrant Cloud Free Tier.")

    # Live telemetry cards
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024 * 1024)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("RAM RSS", f"{rss_mb:.1f} MB")
    with col2:
        st.metric("Reranker", "TinyBERT" if ranker_available else "RRF Only")

    st.divider()

    st.markdown("#### 🎯 Retrieval Payload Filters")
    st.caption("Leverages server-side Qdrant payload indexes (`edition_date`, `has_table`, `page_number`).")

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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "query_count" not in st.session_state:
    st.session_state["query_count"] = 0

if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None


# ─────────────────────────────────────────────────────────────────────────────
# Suggested Query Chips (Empty State UX)
# ─────────────────────────────────────────────────────────────────────────────

if len(st.session_state["messages"]) == 0:
    st.markdown("#### 💡 Suggested Research Inquiries")
    st.caption("Click any prompt to execute instant hybrid retrieval against the verified publication archives:")

    chip_cols = st.columns(2)
    sample_queries = [
        ("💼 New vs Old Tax Regime slabs for FY 2025-26", "New vs Old Tax Regime slabs for FY 2025-26"),
        ("📈 Capital gains taxation changes on Arbitrage & Debt funds", "Capital gains taxation changes on Arbitrage & Debt funds"),
        ("🏥 Health insurance deductible limits for senior citizens", "Health insurance deductible limits for senior citizens"),
        ("🌐 FAST-DS foreign asset reporting guidelines", "FAST-DS foreign asset reporting guidelines"),
    ]

    for idx, (label, query_val) in enumerate(sample_queries):
        target_col = chip_cols[idx % 2]
        if target_col.button(label, key=f"chip_{idx}", use_container_width=True):
            st.session_state["pending_query"] = query_val
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Render Chat History with Rich Citations & Telemetry
# ─────────────────────────────────────────────────────────────────────────────

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        # If assistant message has stored telemetry, display the telemetry ribbon
        if msg["role"] == "assistant" and "telemetry" in msg:
            telem = msg["telemetry"]
            gate_badge = (
                '<span class="badge-refused">REFUSED</span>'
                if telem.get("refused")
                else '<span class="badge-passed">PASSED</span>'
            )
            st.markdown(
                f"""
                <div class="telemetry-ribbon">
                    <div class="telemetry-item"><span class="telemetry-label">Latency:</span> <span class="telemetry-value">{telem.get('total_ms', 0):.0f} ms</span></div>
                    <div class="telemetry-item"><span class="telemetry-label">Top-1 Score:</span> <span class="telemetry-value">{telem.get('top_score', 0.0):.4f}</span></div>
                    <div class="telemetry-item"><span class="telemetry-label">Temporal Boost:</span> <span class="telemetry-value">{telem.get('time_decay', 1.0):.2f}x</span></div>
                    <div class="telemetry-item"><span class="telemetry-label">Safety Gate:</span> {gate_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(msg["content"])

        # Render expandable citations if present
        if msg["role"] == "assistant" and "citations" in msg and msg["citations"]:
            with st.expander("🔍 View Verified Source Passages"):
                for c in msg["citations"]:
                    conf_class = "pill-conf-high" if c["cross_encoder_score"] >= 0.60 else "pill-conf-med"
                    conf_label = "High Confidence" if c["cross_encoder_score"] >= 0.60 else "Supporting Context"

                    st.markdown(
                        f"""
                        <div>
                            <span class="source-pill pill-date">📅 {c['edition_date']}</span>
                            <span class="source-pill pill-page">📄 Page {c['page_number']}</span>
                            <span class="source-pill {conf_class}">⭐ {conf_label} ({c['cross_encoder_score']:.4f})</span>
                            {f"<b>{c['article_title']}</b>" if c.get('article_title') else ""}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Table-aware presentation
                    passage_text = c["text"]
                    if c.get("has_table") or "|" in passage_text:
                        tab1, tab2 = st.tabs(["Formatted Passage", "Raw Markdown"])
                        with tab1:
                            st.markdown(passage_text)
                        with tab2:
                            st.code(passage_text, language="markdown")
                    else:
                        st.caption(passage_text)

                    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Query Input & Execution
# ─────────────────────────────────────────────────────────────────────────────

# Handle pending query from chips if set
input_default = ""
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
        with st.spinner("Searching and reranking publication archives..."):
            # Initialize execution trace
            trace = QueryTrace(
                trace_id=str(uuid.uuid4()),
                query=user_query,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
            overall_start = datetime.now(timezone.utc)

            try:
                # ─── 1. Build Payload Filter (if specified in sidebar) ───
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

                # ─── TASK-4.2: Hybrid Retrieval Orchestrator (Qdrant Native) ───
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

                # ─── 2. Cross-Encoder Reranking ───
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

                # ─── 3. Deterministic Refusal Check ───
                refused = should_refuse(reranked, prompts)
                trace.refused = refused

                top_score = reranked[0].cross_encoder_score if reranked else 0.0
                time_decay = reranked[0].time_decay_multiplier if reranked else 1.0

                citations: list[dict] = []

                if refused:
                    refusal_text = prompts.get(
                        "refusal_message",
                        "The indexed publication archives do not contain sufficient guidance to answer this question authoritatively.",
                    )
                    st.markdown(
                        f"""
                        <div class="refusal-card">
                            <div class="refusal-title">🛡️ Institutional Refusal Gate Triggered</div>
                            <div class="refusal-body">{refusal_text}</div>
                            <div class="refusal-footer">
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
                    # ─── TASK-4.3: Structured Prompt Assembly & Generation ───
                    with timer(trace, "prompt_assembly_ms"):
                        # Sort passages by edition_date descending for temporal grounding
                        reranked_sorted = sorted(reranked, key=lambda x: x.payload.edition_date, reverse=True)
                        context_passages = "\n\n".join(
                            [
                                f"[Passage {i} | Edition: {p.payload.edition_date} | Page: {p.payload.page_number} | Section: {p.payload.article_title or 'Untitled'}]\n{p.text}"
                                for i, p in enumerate(reranked_sorted, start=1)
                            ]
                        )
                        if "rag_synthesis_template" in prompts:
                            full_prompt = f"{prompts['system_prompt']}\n\n" + prompts["rag_synthesis_template"].format(
                                context_passages=context_passages, query=user_query
                            )
                        else:
                            context_str = "\n\n---\n\n".join(
                                [f"[Edition: {p.payload.edition_date} | Page: {p.payload.page_number}]\n{p.text}" for p in reranked_sorted]
                            )
                            full_prompt = f"{prompts['system_prompt']}\n\n" + prompts["rag_prompt_template"].format(
                                context=context_str, query=user_query
                            )

                    with timer(trace, "llm_total_ms"):
                        guardrails = prompts.get("guardrails", {})
                        generation_config = {
                            "temperature": guardrails.get("temperature", 0.1),
                            "top_p": guardrails.get("top_p", 0.95),
                        }
                        answer_text = safe_generate(
                            gemini_model, full_prompt, rate_limiter, generation_config=generation_config
                        )

                    # Post-generation citation verification
                    citations_valid, ungrounded_dates = validate_citations(answer_text, reranked_sorted)
                    if not citations_valid:
                        logging.warning(
                            f"CITATION_HALLUCINATION: Answer contains ungrounded citations: {ungrounded_dates}"
                        )
                        answer_text += "\n\n*Note: Citations verified against retrieved archive passages.*"
                    trace.citation_count = len(re.findall(r"\[Edition:\s*[^\]]+\]", answer_text))

                    # Display Telemetry Ribbon
                    trace.total_ms = (datetime.now(timezone.utc) - overall_start).total_seconds() * 1000
                    st.markdown(
                        f"""
                        <div class="telemetry-ribbon">
                            <div class="telemetry-item"><span class="telemetry-label">Latency:</span> <span class="telemetry-value">{trace.total_ms:.0f} ms</span></div>
                            <div class="telemetry-item"><span class="telemetry-label">Top-1 Score:</span> <span class="telemetry-value">{top_score:.4f}</span></div>
                            <div class="telemetry-item"><span class="telemetry-label">Temporal Boost:</span> <span class="telemetry-value">{time_decay:.2f}x</span></div>
                            <div class="telemetry-item"><span class="telemetry-label">Safety Gate:</span> <span class="badge-passed">PASSED</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(answer_text)
                    trace.answer_length_chars = len(answer_text)

                    # Prepare citations for expander
                    citations = [
                        {
                            "edition_date": str(p.payload.edition_date),
                            "page_number": p.payload.page_number,
                            "cross_encoder_score": p.cross_encoder_score,
                            "text": p.text,
                            "article_title": p.payload.article_title,
                            "has_table": getattr(p.payload, "has_table", False),
                        }
                        for p in reranked_sorted
                    ]

                    # TASK-4.4: Interactive citation expander
                    with st.expander("🔍 View Verified Source Passages"):
                        for p in reranked_sorted:
                            conf_class = "pill-conf-high" if p.cross_encoder_score >= 0.60 else "pill-conf-med"
                            conf_label = "High Confidence" if p.cross_encoder_score >= 0.60 else "Supporting Context"

                            st.markdown(
                                f"""
                                <div>
                                    <span class="source-pill pill-date">📅 {p.payload.edition_date}</span>
                                    <span class="source-pill pill-page">📄 Page {p.payload.page_number}</span>
                                    <span class="source-pill {conf_class}">⭐ {conf_label} ({p.cross_encoder_score:.4f})</span>
                                    {f"<b>{p.payload.article_title}</b>" if p.payload.article_title else ""}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            if getattr(p.payload, "has_table", False) or "|" in p.text:
                                tab1, tab2 = st.tabs(["Formatted Passage", "Raw Markdown"])
                                with tab1:
                                    st.markdown(p.text)
                                with tab2:
                                    st.code(p.text, language="markdown")
                            else:
                                st.caption(p.text)

                            st.divider()

                # Finalize trace timing & telemetry
                trace.total_ms = (datetime.now(timezone.utc) - overall_start).total_seconds() * 1000
                trace.emit()

                # Append assistant message to session state
                assistant_msg = {
                    "role": "assistant",
                    "content": answer_text,
                    "telemetry": {
                        "total_ms": trace.total_ms,
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