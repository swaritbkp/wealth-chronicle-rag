"""
app.py — Public Streamlit application for WealthChronicle AI v1.0
Covers TASK-4.1 through TASK-4.5
"""

import logging
import uuid
from datetime import date, datetime

import google.generativeai as genai
import streamlit as st
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from engine import (BM25Index, GeminiRateLimiter, QueryTrace,
                    check_memory_usage, load_and_validate_prompts,
                    reciprocal_rank_fusion, rerank_candidates, safe_generate,
                    should_refuse, timer, with_qdrant_retry)
from schemas import (ChunkPayload, RerankedPassage, RetrievalSource,
                     SearchResult)

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
        tuple: (gemini_model, qdrant_client, embedding_model, ranker, prompts, bm25_index, rate_limiter, ranker_available)
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

    # FastEmbed
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # FlashRank with fallback
    ranker = None
    ranker_available = False
    try:
        from flashrank import Ranker

        ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/models")
        ranker_available = True
    except Exception as e:
        logging.warning(f"FlashRank init failed, degrading to RRF-only: {e}")
        ranker_available = False

    # Rate limiter
    rate_limiter = GeminiRateLimiter(max_rpm=14)

    # BM25 index: scroll all documents from Qdrant
    bm25_index = None
    try:
        # Use scroll to fetch all points (payload only)
        all_texts: list[str] = []
        all_ids: list[str] = []
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name="wealth_archive",
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for p in points:
                try:
                    # Validate payload via ChunkPayload if possible
                    payload = p.payload
                    if payload and "text" in payload:
                        all_texts.append(payload["text"])
                        all_ids.append(str(p.id))
                except Exception:
                    continue
            if offset is None:
                break

        if all_texts:
            bm25_index = BM25Index(corpus_texts=all_texts, corpus_ids=all_ids)
            logging.info(f"BM25 index built with {len(all_texts)} documents")
        else:
            # Empty corpus: create index with dummy data to avoid crash
            bm25_index = BM25Index(
                corpus_texts=["dummy placeholder text for initialization"],
                corpus_ids=["dummy_id"],
            )
            logging.warning("BM25 index built with dummy data (empty corpus)")
    except Exception as e:
        logging.warning(f"BM25 index build failed: {e}")
        # Fallback dummy index
        try:
            bm25_index = BM25Index(
                corpus_texts=["dummy placeholder"], corpus_ids=["dummy_id"]
            )
        except Exception:
            bm25_index = None

    return (
        gemini_model,
        qdrant_client,
        embedding_model,
        ranker,
        prompts,
        bm25_index,
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
        embedding_model,
        ranker,
        prompts,
        bm25_index,
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
st.info(
    "⚠️ **Disclaimer:** Educational research tool indexing archived publications. Does not constitute registered financial, legal, or tax advisory services."
)

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
                    st.markdown(
                        f"**Edition:** `{c['edition_date']}` | "
                        f"**Page:** `{c['page_number']}` | "
                        f"**Cross-Encoder Score:** `{c['cross_encoder_score']:.4f}`"
                    )
                    st.caption(c["text"])
                    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Query processing
# ─────────────────────────────────────────────────────────────────────────────

user_query = st.chat_input(
    "Ask about tax slabs, health claim rejections, NPS allocations..."
)

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
                timestamp_utc=datetime.utcnow().isoformat() + "Z",
            )
            overall_start = datetime.utcnow()

            try:
                # ─── TASK-4.2: Hybrid Retrieval Orchestrator ───
                # 1. Embed query
                with timer(trace, "embedding_ms"):
                    q_vector = list(embedding_model.embed([user_query]))[0].tolist()

                # 2. Dense retrieval
                dense_results: list[SearchResult] = []
                with timer(trace, "dense_retrieval_ms"):

                    @with_qdrant_retry
                    def _dense_search():
                        return qdrant_client.search(
                            collection_name="wealth_archive",
                            query_vector=q_vector,
                            limit=12,
                        )

                    try:
                        hits = _dense_search()
                    except Exception as e:
                        logging.error(f"Dense retrieval failed: {e}")
                        hits = []

                    trace.dense_candidates = len(hits)
                    # Build SearchResult objects and payload map
                    dense_results = []
                    all_payloads: dict[str, ChunkPayload] = {}
                    for rank, h in enumerate(hits):
                        try:
                            payload = ChunkPayload(**h.payload)
                            all_payloads[str(h.id)] = payload
                            dense_results.append(
                                SearchResult(
                                    point_id=str(h.id),
                                    text=h.payload.get("text", ""),
                                    payload=payload,
                                    score=(
                                        float(h.score) if hasattr(h, "score") else 0.0
                                    ),
                                    source=RetrievalSource.DENSE,
                                    dense_rank=rank + 1,
                                )
                            )
                        except Exception as e:
                            logging.warning(f"Skipping invalid payload for {h.id}: {e}")
                            continue

                # 3. Sparse retrieval
                sparse_results: list[tuple[str, float]] = []
                with timer(trace, "sparse_retrieval_ms"):
                    try:
                        if bm25_index is not None:
                            sparse_results = bm25_index.search(user_query, limit=12)
                            trace.sparse_candidates = len(sparse_results)
                            # Ensure BM25 payloads are in all_payloads
                            # Need to fetch payloads for sparse-only IDs that weren't in dense
                            sparse_only_ids = [
                                pid
                                for pid, _ in sparse_results
                                if pid not in all_payloads
                            ]
                            if sparse_only_ids:
                                try:
                                    # Retrieve sparse-only payloads via Retrieve
                                    retrieved = qdrant_client.retrieve(
                                        collection_name="wealth_archive",
                                        ids=sparse_only_ids,
                                        with_payload=True,
                                    )
                                    for p in retrieved:
                                        try:
                                            payload = ChunkPayload(**p.payload)
                                            all_payloads[str(p.id)] = payload
                                        except Exception:
                                            continue
                                except Exception as e:
                                    logging.warning(
                                        f"Failed to fetch sparse payloads: {e}"
                                    )
                        else:
                            trace.sparse_candidates = 0
                    except Exception as e:
                        logging.warning(f"Sparse retrieval failed: {e}")
                        sparse_results = []
                        trace.sparse_candidates = 0

                # 4. Fuse via RRF
                with timer(trace, "rrf_fusion_ms"):
                    fused_candidates = reciprocal_rank_fusion(
                        dense_results=dense_results,
                        sparse_results=sparse_results,
                        all_payloads=all_payloads,
                        reference_date=date.today(),
                        top_n=20,
                    )
                    trace.fused_candidates = len(fused_candidates)

                # 5. Rerank
                with timer(trace, "reranking_ms"):
                    # Build payload_map for reranker (only those in fused)
                    payload_map = {
                        pid: all_payloads[pid]
                        for pid in [c["point_id"] for c in fused_candidates]
                        if pid in all_payloads
                    }
                    reranked: list[RerankedPassage] = rerank_candidates(
                        query=user_query,
                        candidates=fused_candidates,
                        payload_map=payload_map,
                        ranker=ranker,
                        top_k=4,
                    )
                    trace.reranked_top_k = len(reranked)
                    trace.top1_cross_encoder_score = (
                        reranked[0].cross_encoder_score if reranked else 0.0
                    )

                # 6. Refusal check
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
                    # ─── TASK-4.3: Prompt Assembly & Generation ───
                    with timer(trace, "prompt_assembly_ms"):
                        # Sort passages by edition_date descending (most recent first)
                        reranked_sorted = sorted(
                            reranked, key=lambda x: x.payload.edition_date, reverse=True
                        )
                        context_str = "\n\n---\n\n".join(
                            [
                                f"[Edition: {p.payload.edition_date} | Page: {p.payload.page_number}]\n{p.text}"
                                for p in reranked_sorted
                            ]
                        )
                        full_prompt = (
                            f"{prompts['system_prompt']}\n\n"
                            f"{prompts['rag_prompt_template'].format(context=context_str, query=user_query)}"
                        )

                    # Call Gemini via safe_generate (with rate limiter)
                    with timer(trace, "llm_total_ms"):
                        answer_text = safe_generate(
                            gemini_model, full_prompt, rate_limiter
                        )

                    st.markdown(answer_text)
                    trace.answer_length_chars = len(answer_text)
                    trace.citation_count = len(reranked_sorted)

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
                            st.markdown(
                                f"**Edition:** `{p.payload.edition_date}` | "
                                f"**Page:** `{p.payload.page_number}` | "
                                f"**Cross-Encoder Score:** `{p.cross_encoder_score:.4f}`"
                            )
                            st.caption(p.text)
                            st.divider()

                # Finalize trace timing
                trace.total_ms = (
                    datetime.utcnow() - overall_start
                ).total_seconds() * 1000
                trace.emit()

                # Append assistant message to session state
                assistant_msg = {"role": "assistant", "content": answer_text}
                if "citations" in locals() and citations:
                    assistant_msg["citations"] = citations
                st.session_state["messages"].append(assistant_msg)

                # TASK-4.5: Pruning after assistant addition
                if len(st.session_state["messages"]) > MAX_MESSAGES:
                    st.session_state["messages"] = st.session_state["messages"][
                        -MAX_MESSAGES:
                    ]

            except Exception as e:
                logging.error(f"Query pipeline failed: {e}", exc_info=True)
                err_msg = f"An error occurred while processing your query: {e}"
                st.error(err_msg)
                trace.refused = True
                trace.emit()
                st.session_state["messages"].append(
                    {"role": "assistant", "content": err_msg}
                )
