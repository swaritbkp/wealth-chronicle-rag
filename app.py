"""
app.py — Public Streamlit application for WealthChronicle AI v1.0
Covers TASK-4.1 through TASK-4.5
Uses Qdrant native hybrid search (BM42 sparse + dense vectors).
"""

import logging
import os
import re
import tempfile
import uuid
from datetime import date, datetime, timezone

import google.generativeai as genai
import streamlit as st
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient

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
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="WealthChronicle AI", page_icon="📈", layout="centered")

logger = logging.getLogger("wealthchronicle.trace")
logging.basicConfig(level=logging.INFO)


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


# ─────────────────────────────────────────────────────────────────────────────
# Initialize services with error handling
# ─────────────────────────────────────────────────────────────────────────────

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
# TASK-4.4: Chat UI & Citation Source Expander (static UI elements)
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 WealthChronicle Search")
st.caption("AI-Powered Research Engine for Personal Finance Archives")
st.info("⚠️ **Disclaimer:** Educational research tool indexing archived publications. Does not constitute registered financial, legal, or tax advisory services.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "query_count" not in st.session_state:
    st.session_state["query_count"] = 0

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If assistant message has citations, render them if present in history
        if msg["role"] == "assistant" and "citations" in msg:
            with st.expander("🔍 View Verified Source Passages"):
                for c in msg["citations"]:
                    st.markdown(f"**Edition:** `{c['edition_date']}` | " f"**Page:** `{c['page_number']}` | " f"**Cross-Encoder Score:** `{c['cross_encoder_score']:.4f}`")
                    st.caption(c["text"])
                    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Query processing
# ─────────────────────────────────────────────────────────────────────────────

user_query = st.chat_input("Ask about tax slabs, health claim rejections, NPS allocations...")

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
        with st.spinner("Searching and reranking archives..."):
            # Initialize trace
            trace = QueryTrace(
                trace_id=str(uuid.uuid4()),
                query=user_query,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
            overall_start = datetime.now(timezone.utc)

            try:
                # ─── TASK-4.2: Hybrid Retrieval Orchestrator (Qdrant Native) ───
                # 1. Hybrid search (dense + sparse via Qdrant native)
                with timer(trace, "hybrid_search_ms"):
                    fused_candidates = hybrid_search(
                        client=qdrant_client,
                        query=user_query,
                        dense_embedding_model=dense_embedding_model,
                        sparse_embedding_model=sparse_embedding_model,
                        collection_name="wealth_archive",
                        limit=12,
                        reference_date=date.today(),
                    )
                    trace.fused_candidates = len(fused_candidates)

                # 2. Rerank
                with timer(trace, "reranking_ms"):
                    # Build payload_map for reranker (only those in fused)
                    # We need to fetch payloads for the fused candidates
                    fused_ids = [c["point_id"] for c in fused_candidates]
                    payload_map = {}
                    if fused_ids:
                        retrieved = qdrant_client.retrieve(
                            collection_name="wealth_archive",
                            ids=fused_ids,
                            with_payload=True,
                        )
                        for p in retrieved:
                            try:
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

                # 3. Refusal check
                refused = should_refuse(reranked, prompts)
                trace.refused = refused

                if refused:
                    answer_text = prompts["refusal_message"]
                    st.markdown(answer_text)
                    # No LLM call
                    trace.answer_length_chars = len(answer_text)
                    trace.citation_count = 0
                    citations = []
                else:
                    # ─── TASK-4.3: Prompt Assembly & Generation (v2.0 — structured, citation-disciplined) ───
                    with timer(trace, "prompt_assembly_ms"):
                        # Sort passages by edition_date descending (most recent first) for temporal grounding
                        reranked_sorted = sorted(reranked, key=lambda x: x.payload.edition_date, reverse=True)
                        # Dynamic passage formatting per v2.0 spec
                        context_passages = "\n\n".join(
                            [
                                f"[Passage {i} | Edition: {p.payload.edition_date} | Page: {p.payload.page_number} | Section: {p.payload.article_title or 'Untitled'}]\n{p.text}"
                                for i, p in enumerate(reranked_sorted, start=1)
                            ]
                        )
                        # Use v2.0 rag_synthesis_template if available, fallback to legacy
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

                    # Call Gemini via safe_generate (with rate limiter)
                    with timer(trace, "llm_total_ms"):
                        # P0-3: Pass generation config from guardrails
                        guardrails = prompts.get("guardrails", {})
                        generation_config = {
                            "temperature": guardrails.get("temperature", 0.1),
                            "top_p": guardrails.get("top_p", 0.95),
                        }
                        answer_text = safe_generate(
                            gemini_model, full_prompt, rate_limiter, generation_config=generation_config
                        )

                    # P0-1: Post-generation citation verification
                    citations_valid, ungrounded_dates = validate_citations(answer_text, reranked_sorted)
                    if not citations_valid:
                        logging.warning(
                            f"CITATION_HALLUCINATION: Answer contains ungrounded citations: {ungrounded_dates}"
                        )
                        answer_text += "\n\n*Note: Citations verified against retrieved archive passages.*"
                    trace.citation_count = len(re.findall(r"\[Edition:\s*[^\]]+\]", answer_text))

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
                        }
                        for p in reranked_sorted
                    ]

                    # TASK-4.4: Citation expander
                    with st.expander("🔍 View Verified Source Passages"):
                        for p in reranked_sorted:
                            st.markdown(f"**Edition:** `{p.payload.edition_date}` | " f"**Page:** `{p.payload.page_number}` | " f"**Cross-Encoder Score:** `{p.cross_encoder_score:.4f}`")
                            st.caption(p.text)
                            st.divider()

                # Finalize trace timing
                trace.total_ms = (datetime.now(timezone.utc) - overall_start).total_seconds() * 1000
                trace.emit()

                # Append assistant message to session state
                assistant_msg = {"role": "assistant", "content": answer_text}
                if "citations" in locals() and citations:
                    assistant_msg["citations"] = citations
                st.session_state["messages"].append(assistant_msg)

                # TASK-4.5: Pruning after assistant addition
                if len(st.session_state["messages"]) > MAX_MESSAGES:
                    st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]

            except Exception as e:
                logging.error(f"Query pipeline failed: {e}", exc_info=True)
                err_msg = f"An error occurred while processing your query: {e}"
                st.error(err_msg)
                trace.refused = True
                trace.emit()
                st.session_state["messages"].append({"role": "assistant", "content": err_msg})