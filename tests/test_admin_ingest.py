"""
tests/test_admin_ingest.py — Unit tests for Admin Ingestion Cockpit and Engine LRU Embedding Caching.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from admin_ingest_app import (
    get_cluster_diagnostics,
    scan_corpus_directory,
)
from engine import (
    clear_embedding_cache,
    compute_query_dense_embedding,
    compute_query_embeddings,
    compute_query_sparse_embedding,
    hybrid_search,
)


class TestPDFStagingAndUpload:
    """Test PDF upload saving and filename sanitization."""

    def test_pdf_saving_to_staged_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            sample_content = b"%PDF-1.4 Mock PDF Content"
            filename = "sample edition (1).pdf"

            # Sanitization logic matching admin_ingest_app
            import re

            sanitized_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", filename)
            target_path = data_dir / sanitized_name

            with open(target_path, "wb") as f:
                f.write(sample_content)

            assert target_path.exists()
            assert target_path.name == "sample_edition__1_.pdf"
            assert target_path.read_bytes() == sample_content


class TestCorpusInspectionLogic:
    """Test corpus status scanning and Qdrant index count checks."""

    def test_scan_corpus_directory_with_mocked_qdrant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            pdf1 = data_dir / "wealth_2026-08-24.pdf"
            pdf1.write_bytes(b"%PDF-1.4 Mock 1")

            pdf2 = data_dir / "wealth_2026-08-31.pdf"
            pdf2.write_bytes(b"%PDF-1.4 Mock 2")

            # Mock Qdrant client
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True

            # Return count=27 for 2026-08-24, count=0 for 2026-08-31
            def mock_count(collection_name, count_filter=None):
                mock_res = MagicMock()
                # Check filter value
                if count_filter and hasattr(count_filter, "must") and count_filter.must:
                    val = count_filter.must[0].match.value
                    if val == "2026-08-24":
                        mock_res.count = 27
                        return mock_res
                mock_res.count = 0
                return mock_res

            mock_client.count.side_effect = mock_count

            items = scan_corpus_directory(mock_client, data_dir=str(data_dir))
            assert len(items) == 2

            indexed_item = next(it for it in items if it["Detected Date"] == "2026-08-24")
            assert indexed_item["Status"] == "Indexed"
            assert indexed_item["Indexed Chunks"] == 27

            unindexed_item = next(it for it in items if it["Detected Date"] == "2026-08-31")
            assert unindexed_item["Status"] == "Unindexed"
            assert unindexed_item["Indexed Chunks"] == 0

    def test_cluster_diagnostics_online(self) -> None:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 54
        mock_client.get_collection.return_value = mock_collection_info

        record1 = MagicMock()
        record1.payload = {"edition_date": "2026-08-24"}
        record2 = MagicMock()
        record2.payload = {"edition_date": "2026-08-31"}
        mock_client.scroll.return_value = ([record1, record2], None)

        diag = get_cluster_diagnostics(mock_client)
        assert diag["online"] is True
        assert diag["points"] == 54
        assert diag["editions"] == 2
        assert diag["error"] is None

    def test_cluster_diagnostics_none_client(self) -> None:
        diag = get_cluster_diagnostics(None)
        assert diag["online"] is False
        assert diag["points"] == 0


class TestLRUQueryEmbeddingCache:
    """Verify in-memory LRU cache for query vectorization."""

    def test_dense_embedding_lru_cache_hits(self) -> None:
        clear_embedding_cache()
        query = "What is the capital gains tax for FY 2025-26?"

        # First call: computes and caches
        info_before = compute_query_dense_embedding.cache_info()
        vec1 = compute_query_dense_embedding(query)
        info_after1 = compute_query_dense_embedding.cache_info()

        assert len(vec1) == 384
        assert info_after1.misses == info_before.misses + 1

        # Second call: hit
        vec2 = compute_query_dense_embedding(query)
        info_after2 = compute_query_dense_embedding.cache_info()

        assert vec1 == vec2
        assert info_after2.hits == info_after1.hits + 1

    def test_sparse_embedding_lru_cache_hits(self) -> None:
        clear_embedding_cache()
        query = "FAST-DS foreign asset reporting limits"

        info_before = compute_query_sparse_embedding.cache_info()
        indices1, values1 = compute_query_sparse_embedding(query)
        info_after1 = compute_query_sparse_embedding.cache_info()

        assert len(indices1) == len(values1)
        assert len(indices1) > 0
        assert info_after1.misses == info_before.misses + 1

        # Second call: hit
        indices2, values2 = compute_query_sparse_embedding(query)
        info_after2 = compute_query_sparse_embedding.cache_info()

        assert indices1 == indices2
        assert values1 == values2
        assert info_after2.hits == info_after1.hits + 1

    def test_combined_embedding_cache(self) -> None:
        clear_embedding_cache()
        query = "Health insurance senior deductible"

        dense_vec, indices, values = compute_query_embeddings(query)
        assert len(dense_vec) == 384
        assert len(indices) == len(values)

        # Subsequent call hits cache
        info_before = compute_query_embeddings.cache_info()
        dense_vec2, indices2, values2 = compute_query_embeddings(query)
        info_after = compute_query_embeddings.cache_info()

        assert info_after.hits == info_before.hits + 1
        assert dense_vec == dense_vec2

    def test_hybrid_search_graceful_on_disconnected_client(self) -> None:
        # Mock client that raises connection error
        mock_client = MagicMock()
        mock_client.query_points.side_effect = ConnectionError("Qdrant Cloud unreachable")
        mock_client.search.side_effect = ConnectionError("Qdrant Cloud unreachable")

        results = hybrid_search(
            client=mock_client,
            query="test query",
        )
        assert results == []
