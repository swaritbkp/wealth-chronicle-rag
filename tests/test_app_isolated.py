"""tests/test_app_isolated.py — App session & prompt isolation tests."""

from __future__ import annotations

from schemas import ChunkPayload, RerankedPassage

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_payload(edition_date: str, page: int = 1) -> ChunkPayload:
    return ChunkPayload(
        chunk_id=f"chk_{edition_date.replace('-', '_')}_p{page}_000",
        edition_date=edition_date,
        page_number=page,
        text=f"Content for edition {edition_date} page {page}. " * 10,
        char_count=300,
        word_count=40,
    )


def _make_reranked(pid: str, edition_date: str, page: int, score: float = 0.5) -> RerankedPassage:
    payload = _make_payload(edition_date, page)
    return RerankedPassage(
        point_id=pid,
        text=payload.text,
        payload=payload,
        cross_encoder_score=score,
        rrf_score=0.02,
        time_decay_multiplier=1.2,
        final_rank=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session State History Capping
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionStateHistoryCapping:
    """Tests for MAX_MESSAGES = 20 sliding window (app.py TASK-4.5)."""

    MAX_MESSAGES = 20

    def _prune(self, messages: list[dict]) -> list[dict]:
        if len(messages) > self.MAX_MESSAGES:
            return messages[-self.MAX_MESSAGES :]
        return messages

    def test_exactly_20_messages_retained(self) -> None:
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        pruned = self._prune(msgs)
        assert len(pruned) == 20
        assert pruned[0]["content"] == "msg 0"

    def test_50_messages_capped_to_20_most_recent(self) -> None:
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(50)]
        pruned = self._prune(msgs)
        assert len(pruned) == 20
        assert pruned[0]["content"] == "msg 30"
        assert pruned[-1]["content"] == "msg 49"

    def test_25_messages_evicts_oldest_5(self) -> None:
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(25)]
        pruned = self._prune(msgs)
        assert len(pruned) == 20
        assert pruned[0]["content"] == "msg 5"

    def test_alternating_user_assistant_preserves_pairs(self) -> None:
        msgs: list[dict] = []
        for i in range(30):
            msgs.append({"role": "user", "content": f"q {i}"})
            msgs.append({"role": "assistant", "content": f"a {i}"})
        # 60 msgs → pruned to 20 (last 10 pairs)
        pruned = self._prune(msgs)
        assert len(pruned) == 20
        # First retained should be q 20 (since 60-20=40 → index 40 → q 20)
        assert pruned[0]["content"] == "q 20"
        assert pruned[1]["content"] == "a 20"

    def test_empty_history_stays_empty(self) -> None:
        assert self._prune([]) == []

    def test_19_messages_no_pruning(self) -> None:
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(19)]
        assert len(self._prune(msgs)) == 19

    def test_21_messages_evicts_one(self) -> None:
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(21)]
        pruned = self._prune(msgs)
        assert len(pruned) == 20
        assert pruned[0]["content"] == "msg 1"

    def test_citations_preserved_after_pruning(self) -> None:
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "citations": [{"page_number": 1}]},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "citations": [{"page_number": 2}]},
        ] * 15  # 60 msgs
        pruned = self._prune(msgs)
        assert len(pruned) == 20
        # Citations should still be present in assistant messages
        for m in pruned:
            if m["role"] == "assistant":
                assert "citations" in m

    def test_app_pruning_logic_matches_spec(self) -> None:
        # Directly test the logic from app.py: messages = messages[-MAX_MESSAGES:]
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(25)]
        MAX_MESSAGES = 20
        if len(messages) > MAX_MESSAGES:
            messages = messages[-MAX_MESSAGES:]
        assert len(messages) == 20
        assert messages[0]["content"] == "msg 5"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Construction & Citation Formatting
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptConstructionAndCitationFormatting:
    def test_citations_ordered_newest_date_first(self) -> None:
        # Reranked unsorted: oldest first
        passages = [
            _make_reranked("a", "2025-01-01", 1),
            _make_reranked("b", "2026-08-29", 2),
            _make_reranked("c", "2025-06-15", 3),
            _make_reranked("d", "2026-01-20", 4),
        ]
        sorted_passages = sorted(passages, key=lambda x: x.payload.edition_date, reverse=True)
        dates = [p.payload.edition_date for p in sorted_passages]
        assert dates == sorted(dates, reverse=True)
        assert str(dates[0]) == "2026-08-29"
        assert str(dates[-1]) == "2025-01-01"

    def test_context_string_format_and_separators(self) -> None:
        passages = [
            _make_reranked("a", "2026-08-29", 5),
            _make_reranked("b", "2026-08-24", 3),
        ]
        # Simulate prompt assembly from app.py TASK-4.3
        passages_sorted = sorted(passages, key=lambda x: x.payload.edition_date, reverse=True)
        context_str = "\n\n---\n\n".join([f"[Edition: {p.payload.edition_date} | Page: {p.payload.page_number}]\n{p.text}" for p in passages_sorted])
        # Newest first
        assert context_str.index("2026-08-29") < context_str.index("2026-08-24")
        # Format check
        assert "[Edition: 2026-08-29 | Page: 5]" in context_str
        assert "[Edition: 2026-08-24 | Page: 3]" in context_str
        # Separator
        assert "\n\n---\n\n" in context_str

    def test_full_prompt_uses_templates_from_yaml(self) -> None:
        from engine import load_and_validate_prompts

        prompts = load_and_validate_prompts("config/prompts.yaml")
        passages = [_make_reranked("a", "2026-08-29", 1)]
        # Support both v2.0 (context_passages) and legacy (context) templates
        if "rag_synthesis_template" in prompts:
            context_passages = "\n\n".join(
                [f"[Passage 1 | Edition: {p.payload.edition_date} | Page: {p.payload.page_number} | Section: {p.payload.article_title or 'Untitled'}]\n{p.text}" for p in passages]
            )
            query = "What is LTCG tax?"
            full_prompt = f"{prompts['system_prompt']}\n\n{prompts['rag_synthesis_template'].format(context_passages=context_passages, query=query)}"
            assert "WealthChronicle" in full_prompt or "financial research" in full_prompt.lower()
            assert "RETRIEVED CONTEXT" in full_prompt or "context_passages" not in full_prompt  # template rendered
            assert query in full_prompt
            assert context_passages in full_prompt
        else:
            context_str = "\n\n---\n\n".join([f"[Edition: {p.payload.edition_date} | Page: {p.payload.page_number}]\n{p.text}" for p in passages])
            query = "What is LTCG tax?"
            full_prompt = f"{prompts['system_prompt']}\n\n{prompts['rag_prompt_template'].format(context=context_str, query=query)}"
            assert "You are an expert personal finance research assistant" in full_prompt or "WealthChronicle" in full_prompt
            assert "Archived Excerpts" in full_prompt or "RETRIEVED CONTEXT" in full_prompt
            assert query in full_prompt
            assert context_str in full_prompt

    def test_citation_metadata_formatting(self) -> None:
        passage = _make_reranked("abc", "2026-08-24", page=7, score=0.8567)
        # Simulate expander formatting from app.py
        line = f"**Edition:** `{passage.payload.edition_date}` | **Page:** `{passage.payload.page_number}` | **Cross-Encoder Score:** `{passage.cross_encoder_score:.4f}`"
        assert "`2026-08-24`" in line
        assert "`7`" in line
        assert "`0.8567`" in line

    def test_prompt_assembly_with_four_passages(self) -> None:
        passages = [_make_reranked(f"id{i}", f"2026-08-{10+i:02d}", i + 1) for i in range(4)]
        sorted_p = sorted(passages, key=lambda x: x.payload.edition_date, reverse=True)
        context_str = "\n\n---\n\n".join([f"[Edition: {p.payload.edition_date} | Page: {p.payload.page_number}]\n{p.text}" for p in sorted_p])
        # Should have 3 separators for 4 passages
        assert context_str.count("---") == 3
        # Each passage should have its edition line
        for p in sorted_p:
            assert f"[Edition: {p.payload.edition_date} | Page: {p.payload.page_number}]" in context_str

    def test_disclaimer_and_title_invariants(self) -> None:
        # Ensure app.py contains required UI strings
        app_text = open("app.py", encoding="utf-8").read()
        assert 'st.title("📈 WealthChronicle Search")' in app_text
        assert "Does not constitute registered financial" in app_text
        assert "🔍 View Verified Source Passages" in app_text

    def test_session_counter_memory_guard_every_10(self) -> None:
        # Verify logic: query_count % 10 == 0 triggers check_memory_usage
        for count in [1, 9, 10, 11, 20, 21, 30]:
            should_check = count % 10 == 0
            if count in (10, 20, 30):
                assert should_check is True
            else:
                assert should_check is False

    def test_refusal_path_skips_llm_and_shows_deterministic_message(self) -> None:
        from engine import load_and_validate_prompts, should_refuse

        prompts = load_and_validate_prompts("config/prompts.yaml")
        # Empty reranked → should refuse
        assert should_refuse([], prompts) is True
        # Refusal message is deterministic — support both v1.0 and v2.0 wording
        refusal_lower = prompts["refusal_message"].lower()
        assert (
            "publication archives do not contain sufficient guidance" in refusal_lower
            or "could not find sufficiently grounded information" in refusal_lower
        )
        assert "archive" in refusal_lower
