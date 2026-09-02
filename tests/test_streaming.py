"""
tests/test_streaming.py — Unit tests for Token Streaming Synthesis and TTFT Telemetry.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from engine import (
    GeminiRateLimiter,
    TimedStreamWrapper,
    stream_synthesize_answer,
    synthesize_answer,
)


class TestTokenStreamingSynthesis:
    """Test streaming chunk yields and rate limiting in synthesize_answer."""

    def test_stream_synthesize_answer_yields_chunks(self) -> None:
        mock_model = MagicMock()
        chunk1 = MagicMock()
        chunk1.text = "Under FAST-DS, "
        chunk2 = MagicMock()
        chunk2.text = "foreign assets up to Rs 1 crore "
        chunk3 = MagicMock()
        chunk3.text = "are taxed at 10%."
        mock_model.generate_content.return_value = [chunk1, chunk2, chunk3]

        rate_limiter = GeminiRateLimiter(max_rpm=60)
        chunks = list(stream_synthesize_answer(mock_model, "Test prompt", rate_limiter=rate_limiter))

        assert len(chunks) == 3
        assert chunks == ["Under FAST-DS, ", "foreign assets up to Rs 1 crore ", "are taxed at 10%."]

    def test_synthesize_answer_streaming_mode(self) -> None:
        mock_model = MagicMock()
        chunk1 = MagicMock()
        chunk1.text = "Token 1 "
        chunk2 = MagicMock()
        chunk2.text = "Token 2"
        mock_model.generate_content.return_value = [chunk1, chunk2]

        gen = synthesize_answer(mock_model, "Test", stream=True)
        tokens = list(gen)
        assert "".join(tokens) == "Token 1 Token 2"

    def test_synthesize_answer_non_streaming_mode(self) -> None:
        mock_model = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Complete generated string."
        mock_model.generate_content.return_value = mock_resp

        result = synthesize_answer(mock_model, "Test", stream=False)
        assert result == "Complete generated string."


class TestTimedStreamWrapper:
    """Test Time-to-First-Token (TTFT) and completion duration measurement."""

    def test_ttft_and_completion_timing(self) -> None:
        def mock_generator():
            time.sleep(0.02)  # 20ms before first token
            yield "Hello "
            time.sleep(0.03)  # 30ms before second token
            yield "World!"

        wrapper = TimedStreamWrapper(mock_generator())
        received = []
        for chunk in wrapper:
            received.append(chunk)

        assert "".join(received) == "Hello World!"
        assert wrapper.full_text == "Hello World!"
        # TTFT should be >= 10ms (allowing for timing precision) and <= completion_ms
        assert wrapper.ttft_ms >= 10.0
        assert wrapper.completion_ms >= wrapper.ttft_ms
        assert wrapper.first_token_received is True

    def test_empty_generator_handling(self) -> None:
        def empty_generator():
            return
            yield

        wrapper = TimedStreamWrapper(empty_generator())
        received = list(wrapper)
        assert received == []
        assert wrapper.first_token_received is False
        assert wrapper.ttft_ms >= 0.0
