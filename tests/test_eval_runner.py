"""
tests/test_eval_runner.py — Unit tests for IR Benchmark Evaluation Runner metrics and logic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from eval_runner import (
    compute_hit_rate,
    compute_mrr,
    compute_refusal_precision,
    evaluate_query_item,
    format_benchmark_report,
    run_benchmark,
)


class TestEvaluationMetricsMath:
    """Validate mathematical precision of IR evaluation formulas."""

    def test_compute_hit_rate(self) -> None:
        assert compute_hit_rate([]) == 0.0
        assert compute_hit_rate([True, True, True, True]) == 1.0
        assert compute_hit_rate([True, False, True, False]) == 0.5
        assert compute_hit_rate([False, False, False]) == 0.0

    def test_compute_mrr(self) -> None:
        assert compute_mrr([]) == 0.0
        # All rank 1 -> MRR = 1.0
        assert compute_mrr([1, 1, 1]) == 1.0
        # Rank 1 (1.0), Rank 2 (0.5), Rank 4 (0.25), None (0.0) -> (1 + 0.5 + 0.25 + 0) / 4 = 0.4375
        assert compute_mrr([1, 2, 4, None]) == 0.4375
        # All missed -> MRR = 0.0
        assert compute_mrr([None, None]) == 0.0

    def test_compute_refusal_precision(self) -> None:
        assert compute_refusal_precision([], []) == 1.0
        # 2 predicted refusals, both true OOD -> Precision 1.0
        pred = [True, True, False]
        true_ood = [True, True, False]
        assert compute_refusal_precision(pred, true_ood) == 1.0

        # 2 predicted refusals, but 1 was an in-domain false refusal -> Precision 0.5
        pred = [True, True]
        true_ood = [True, False]
        assert compute_refusal_precision(pred, true_ood) == 0.5


class TestQueryEvaluationItem:
    """Test single item retrieval and hit determination."""

    def test_evaluate_in_domain_hit(self) -> None:
        from schemas import ChunkPayload

        mock_client = MagicMock()
        mock_dense = MagicMock()
        mock_sparse = MagicMock()

        text_content = (
            "Under FAST-DS (Foreign Asset Small Taxpayers Disclosure Scheme), foreign assets up to "
            "Rs 1 crore are subject to a 10 percent tax on undisclosed foreign income when filed in Schedule FA."
        )
        payload = ChunkPayload(
            chunk_id="chk_2026_08_24_p16_001",
            edition_date="2026-08-24",
            page_number=16,
            text=text_content,
            char_count=len(text_content),
            word_count=len(text_content.split()),
        )

        cand1 = {
            "id": "chk_2026_08_24_p16_001",
            "point_id": "chk_2026_08_24_p16_001",
            "payload": payload,
        }

        with patch("eval_runner.hybrid_search", return_value=[cand1]):
            item = {
                "id": "eval_001",
                "question": "What is FAST-DS?",
                "source_edition_dates": ["2026-08-24"],
                "source_pages": [16],
            }

            res = evaluate_query_item(item, mock_client, mock_dense, mock_sparse, ranker=None)
            assert res["hit_at_3"] is True
            assert res["hit_at_5"] is True
            assert res["first_match_rank"] == 1
            assert res["is_ood"] is False

    def test_evaluate_out_of_domain_refusal(self) -> None:
        mock_client = MagicMock()
        mock_dense = MagicMock()
        mock_sparse = MagicMock()

        with patch("eval_runner.hybrid_search", return_value=[]):
            item = {
                "id": "ood_001",
                "category": "out_of_domain",
                "question": "How to make pasta?",
                "source_edition_dates": [],
                "source_pages": [],
            }

            res = evaluate_query_item(item, mock_client, mock_dense, mock_sparse)
            assert res["is_ood"] is True
            assert res["refused"] is True


class TestBenchmarkReportAndRunner:
    """Test ASCII summary report and overall benchmark runner."""

    def test_format_benchmark_report(self) -> None:
        metrics = {
            "storage_mode": "CLOUD",
            "total_queries": 25,
            "in_domain_queries": 21,
            "ood_queries": 4,
            "hit_rate_at_3": 0.90,
            "hit_rate_at_5": 0.95,
            "mrr_at_5": 0.88,
            "refusal_precision": 1.0,
            "duration_s": 1.25,
        }
        report = format_benchmark_report(metrics)
        assert "WEALTHCHRONICLE AI — RETRIEVAL BENCHMARK EVALUATION REPORT" in report
        assert "90.00%" in report
        assert "95.00%" in report
        assert "OVERALL BENCHMARK VERDICT:  PASSED" in report

    def test_run_benchmark_json_output(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            out_json = Path(tmpdir) / "eval_out.json"
            metrics = run_benchmark(
                eval_set_path="tests/golden_eval_set_2026.json",
                output_json=str(out_json),
            )
            assert metrics["total_queries"] > 0
            assert out_json.exists()
