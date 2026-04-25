"""
Unit tests for SparseRetriever.
"""

import pytest
from unittest.mock import Mock

from src.core.query_engine.sparse_retriever import SparseRetriever, FakeSparseRetriever
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult


class TestFakeSparseRetriever:
    """Tests for FakeSparseRetriever."""

    def test_fake_retriever_returns_default_results(self):
        """Test fake retriever returns empty list by default."""
        retriever = FakeSparseRetriever()
        results = retriever.retrieve(["keyword1", "keyword2"])
        assert results == []

    def test_fake_retriever_returns_custom_results(self):
        """Test fake retriever returns custom results."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
        ]
        retriever = FakeSparseRetriever(results=custom_results)
        results = retriever.retrieve(["keyword"])
        assert len(results) == 2
        assert results[0].chunk_id == "1"

    def test_fake_retriever_respects_top_k(self):
        """Test fake retriever respects top_k."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
            RetrievalResult(chunk_id="3", score=0.7, text="result 3"),
        ]
        retriever = FakeSparseRetriever(results=custom_results)
        results = retriever.retrieve(["keyword"], top_k=2)
        assert len(results) == 2

    def test_fake_retriever_records_calls(self):
        """Test fake retriever records calls."""
        retriever = FakeSparseRetriever()
        retriever.retrieve(["keyword1"], top_k=5)
        retriever.retrieve(["keyword2", "keyword3"], top_k=10)

        assert len(retriever.retrieve_calls) == 2
        assert retriever.retrieve_calls[0]["keywords"] == ["keyword1"]
        assert retriever.retrieve_calls[1]["keywords"] == ["keyword2", "keyword3"]

    def test_fake_retriever_get_methods(self):
        """Test fake retriever get methods."""
        retriever = FakeSparseRetriever()
        assert retriever.get_bm25_indexer() is None
        assert retriever.get_vector_store() is None


class TestSparseRetriever:
    """Tests for SparseRetriever."""

    @pytest.fixture
    def mock_bm25_indexer(self):
        """Create mock BM25 indexer."""
        indexer = Mock()
        indexer.query.return_value = [
            ("chunk_1", 2.5),
            ("chunk_2", 1.8),
            ("chunk_3", 1.2),
        ]
        return indexer

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        store.get_by_ids.return_value = [
            {"id": "chunk_1", "text": "text 1", "metadata": {"page": 1}},
            {"id": "chunk_2", "text": "text 2", "metadata": {"page": 2}},
            {"id": "chunk_3", "text": "text 3", "metadata": {"page": 3}},
        ]
        return store

    @pytest.fixture
    def retriever(self, mock_bm25_indexer, mock_vector_store):
        """Create SparseRetriever with mocked dependencies."""
        return SparseRetriever(
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store,
        )

    def test_retrieve_returns_results(self, retriever):
        """Test retrieve returns results."""
        results = retriever.retrieve(["machine", "learning"])
        assert len(results) == 3
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score == 2.5
        assert results[0].text == "text 1"

    def test_retrieve_calls_bm25_indexer(self, retriever, mock_bm25_indexer):
        """Test retrieve calls BM25 indexer."""
        retriever.retrieve(["machine", "learning"], top_k=5)
        mock_bm25_indexer.query.assert_called_once_with(["machine", "learning"], 5)

    def test_retrieve_calls_vector_store(self, retriever, mock_vector_store):
        """Test retrieve calls vector store."""
        retriever.retrieve(["machine", "learning"])
        mock_vector_store.get_by_ids.assert_called_once_with(["chunk_1", "chunk_2", "chunk_3"])

    def test_retrieve_with_trace(self, retriever):
        """Test retrieve with trace context."""
        trace = TraceContext(trace_id="test-trace")
        retriever.retrieve(["machine", "learning"], trace=trace)

        assert len(trace.stages) == 1
        assert trace.stages[0]["stage"] == "sparse_retrieval"

    def test_retrieve_merges_score_and_text(self, retriever):
        """Test retrieve merges BM25 score with text from vector store."""
        results = retriever.retrieve(["machine", "learning"])

        assert results[0].chunk_id == "chunk_1"
        assert results[0].score == 2.5  # From BM25
        assert results[0].text == "text 1"  # From VectorStore
        assert results[0].metadata == {"page": 1}

    def test_retrieve_empty_keywords(self, retriever):
        """Test retrieve with empty keywords."""
        results = retriever.retrieve([])
        assert results == []

    def test_retrieve_no_bm25_results(self, mock_bm25_indexer, mock_vector_store):
        """Test retrieve with no BM25 results."""
        mock_bm25_indexer.query.return_value = []
        retriever = SparseRetriever(
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store,
        )

        results = retriever.retrieve(["nonexistent"])
        assert results == []

    def test_retrieve_without_bm25_indexer_raises(self, mock_vector_store):
        """Test retrieve without BM25 indexer raises error."""
        retriever = SparseRetriever(vector_store=mock_vector_store)

        with pytest.raises(RuntimeError, match="BM25 indexer not configured"):
            retriever.retrieve(["keyword"])

    def test_retrieve_without_vector_store_raises(self, mock_bm25_indexer):
        """Test retrieve without vector store raises error."""
        retriever = SparseRetriever(bm25_indexer=mock_bm25_indexer)

        with pytest.raises(RuntimeError, match="Vector store not configured"):
            retriever.retrieve(["keyword"])

    def test_retrieve_missing_records(self, mock_bm25_indexer, mock_vector_store):
        """Test retrieve handles missing records gracefully."""
        mock_bm25_indexer.query.return_value = [
            ("chunk_1", 2.5),
            ("chunk_missing", 1.8),
        ]
        mock_vector_store.get_by_ids.return_value = [
            {"id": "chunk_1", "text": "text 1", "metadata": {"page": 1}},
            # chunk_missing not returned
        ]
        retriever = SparseRetriever(
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store,
        )

        results = retriever.retrieve(["keyword"])
        assert len(results) == 2
        assert results[0].text == "text 1"
        assert results[1].text == ""  # Missing record has empty text

    def test_get_bm25_indexer(self, retriever, mock_bm25_indexer):
        """Test get_bm25_indexer returns indexer."""
        assert retriever.get_bm25_indexer() == mock_bm25_indexer

    def test_get_vector_store(self, retriever, mock_vector_store):
        """Test get_vector_store returns store."""
        assert retriever.get_vector_store() == mock_vector_store

    def test_retrieve_with_custom_top_k(self, retriever, mock_bm25_indexer):
        """Test retrieve with custom top_k."""
        retriever.retrieve(["machine"], top_k=20)
        mock_bm25_indexer.query.assert_called_once_with(["machine"], 20)

    def test_retrieve_preserves_order(self, mock_bm25_indexer, mock_vector_store):
        """Test retrieve preserves BM25 order."""
        mock_bm25_indexer.query.return_value = [
            ("chunk_3", 3.0),
            ("chunk_1", 2.0),
            ("chunk_2", 1.0),
        ]
        mock_vector_store.get_by_ids.return_value = [
            {"id": "chunk_3", "text": "text 3", "metadata": {}},
            {"id": "chunk_1", "text": "text 1", "metadata": {}},
            {"id": "chunk_2", "text": "text 2", "metadata": {}},
        ]
        retriever = SparseRetriever(
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store,
        )

        results = retriever.retrieve(["keyword"])
        assert results[0].chunk_id == "chunk_3"  # Highest score first
        assert results[1].chunk_id == "chunk_1"
        assert results[2].chunk_id == "chunk_2"


class TestSparseRetrieverIntegration:
    """Integration tests for SparseRetriever."""

    def test_retriever_with_settings(self):
        """Test retriever initialization with settings."""
        from src.core.settings import Settings

        settings = Settings()
        retriever = SparseRetriever(settings=settings)

        assert retriever._settings == settings

    def test_retriever_default_settings(self):
        """Test retriever with default settings."""
        retriever = SparseRetriever()
        assert retriever._settings is not None
