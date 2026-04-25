"""
Integration tests for HybridSearch.
"""

import pytest
from unittest.mock import Mock

from src.core.query_engine.hybrid_search import HybridSearch, FakeHybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFFusion
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult


class TestFakeHybridSearch:
    """Tests for FakeHybridSearch."""

    def test_fake_search_returns_default_results(self):
        """Test fake search returns empty list by default."""
        search = FakeHybridSearch()
        results = search.search("test query")
        assert results == []

    def test_fake_search_returns_custom_results(self):
        """Test fake search returns custom results."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
        ]
        search = FakeHybridSearch(results=custom_results)
        results = search.search("test query")
        assert len(results) == 2
        assert results[0].chunk_id == "1"

    def test_fake_search_respects_top_k(self):
        """Test fake search respects top_k."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
            RetrievalResult(chunk_id="3", score=0.7, text="result 3"),
        ]
        search = FakeHybridSearch(results=custom_results)
        results = search.search("test query", top_k=2)
        assert len(results) == 2

    def test_fake_search_records_calls(self):
        """Test fake search records calls."""
        search = FakeHybridSearch()
        search.search("query 1", top_k=5)
        search.search("query 2", top_k=10, filters={"collection": "test"})

        assert len(search.search_calls) == 2
        assert search.search_calls[0]["query"] == "query 1"
        assert search.search_calls[1]["filters"] == {"collection": "test"}

    def test_fake_search_get_methods(self):
        """Test fake search get methods."""
        search = FakeHybridSearch()
        assert search.get_query_processor() is None
        assert search.get_dense_retriever() is None
        assert search.get_sparse_retriever() is None
        assert search.get_fusion() is None


class TestHybridSearch:
    """Tests for HybridSearch."""

    @pytest.fixture
    def mock_dense_retriever(self):
        """Create mock dense retriever."""
        retriever = Mock(spec=DenseRetriever)
        retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="d1", score=0.95, text="dense text 1", metadata={"page": 1}),
            RetrievalResult(chunk_id="d2", score=0.85, text="dense text 2", metadata={"page": 2}),
        ]
        return retriever

    @pytest.fixture
    def mock_sparse_retriever(self):
        """Create mock sparse retriever."""
        retriever = Mock(spec=SparseRetriever)
        retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="s1", score=2.5, text="sparse text 1", metadata={"page": 3}),
            RetrievalResult(chunk_id="s2", score=1.8, text="sparse text 2", metadata={"page": 4}),
        ]
        return retriever

    @pytest.fixture
    def query_processor(self):
        """Create query processor."""
        return QueryProcessor()

    @pytest.fixture
    def fusion(self):
        """Create RRF fusion."""
        return RRFFusion()

    @pytest.fixture
    def hybrid_search(self, query_processor, mock_dense_retriever, mock_sparse_retriever, fusion):
        """Create HybridSearch with mocked dependencies."""
        return HybridSearch(
            query_processor=query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            fusion=fusion,
        )

    def test_search_returns_results(self, hybrid_search):
        """Test search returns results."""
        results = hybrid_search.search("machine learning")
        assert len(results) > 0

    def test_search_calls_dense_retriever(self, hybrid_search, mock_dense_retriever):
        """Test search calls dense retriever."""
        hybrid_search.search("machine learning")
        mock_dense_retriever.retrieve.assert_called_once()

    def test_search_calls_sparse_retriever(self, hybrid_search, mock_sparse_retriever):
        """Test search calls sparse retriever."""
        hybrid_search.search("machine learning")
        mock_sparse_retriever.retrieve.assert_called_once()

    def test_search_with_trace(self, hybrid_search):
        """Test search with trace context."""
        trace = TraceContext(trace_id="test-trace")
        hybrid_search.search("machine learning", trace=trace)

        # Should have stages from query_processor, retrievers, fusion, and hybrid_search
        stage_names = [s["stage"] for s in trace.stages]
        assert "hybrid_search" in stage_names

    def test_search_with_filters(self, hybrid_search, mock_dense_retriever):
        """Test search with filters."""
        hybrid_search.search("machine learning", filters={"collection": "test"})
        call_args = mock_dense_retriever.retrieve.call_args
        assert call_args[1]["filters"] == {"collection": "test"}

    def test_search_without_dense_retriever(self, query_processor, mock_sparse_retriever, fusion):
        """Test search without dense retriever."""
        hybrid_search = HybridSearch(
            query_processor=query_processor,
            dense_retriever=None,
            sparse_retriever=mock_sparse_retriever,
            fusion=fusion,
        )
        results = hybrid_search.search("machine learning")
        assert len(results) > 0

    def test_search_without_sparse_retriever(self, query_processor, mock_dense_retriever, fusion):
        """Test search without sparse retriever."""
        hybrid_search = HybridSearch(
            query_processor=query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=None,
            fusion=fusion,
        )
        results = hybrid_search.search("machine learning")
        assert len(results) > 0

    def test_search_without_both_retrievers(self, query_processor, fusion):
        """Test search without both retrievers."""
        hybrid_search = HybridSearch(
            query_processor=query_processor,
            dense_retriever=None,
            sparse_retriever=None,
            fusion=fusion,
        )
        results = hybrid_search.search("machine learning")
        assert results == []

    def test_search_dense_retriever_failure(self, query_processor, mock_sparse_retriever, fusion):
        """Test search handles dense retriever failure."""
        failing_dense = Mock(spec=DenseRetriever)
        failing_dense.retrieve.side_effect = Exception("Dense retriever failed")

        hybrid_search = HybridSearch(
            query_processor=query_processor,
            dense_retriever=failing_dense,
            sparse_retriever=mock_sparse_retriever,
            fusion=fusion,
        )
        results = hybrid_search.search("machine learning")
        # Should still get results from sparse retriever
        assert len(results) > 0

    def test_search_sparse_retriever_failure(self, query_processor, mock_dense_retriever, fusion):
        """Test search handles sparse retriever failure."""
        failing_sparse = Mock(spec=SparseRetriever)
        failing_sparse.retrieve.side_effect = Exception("Sparse retriever failed")

        hybrid_search = HybridSearch(
            query_processor=query_processor,
            dense_retriever=mock_dense_retriever,
            sparse_retriever=failing_sparse,
            fusion=fusion,
        )
        results = hybrid_search.search("machine learning")
        # Should still get results from dense retriever
        assert len(results) > 0

    def test_metadata_filter_collection(self, hybrid_search, mock_dense_retriever):
        """Test metadata filter for collection."""
        mock_dense_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1", metadata={"collection": "docs_a"}),
            RetrievalResult(chunk_id="2", score=0.8, text="text 2", metadata={"collection": "docs_b"}),
        ]

        results = hybrid_search.search("query", filters={"collection": "docs_a"})
        assert len(results) == 1
        assert results[0].metadata["collection"] == "docs_a"

    def test_metadata_filter_doc_type(self, hybrid_search, mock_dense_retriever):
        """Test metadata filter for doc_type."""
        mock_dense_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1", metadata={"doc_type": "pdf"}),
            RetrievalResult(chunk_id="2", score=0.8, text="text 2", metadata={"doc_type": "md"}),
        ]

        results = hybrid_search.search("query", filters={"doc_type": "pdf"})
        assert len(results) == 1
        assert results[0].metadata["doc_type"] == "pdf"

    def test_metadata_filter_custom_field(self, hybrid_search, mock_dense_retriever):
        """Test metadata filter for custom field."""
        mock_dense_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1", metadata={"category": "tech"}),
            RetrievalResult(chunk_id="2", score=0.8, text="text 2", metadata={"category": "news"}),
        ]

        results = hybrid_search.search("query", filters={"category": "tech"})
        assert len(results) == 1
        assert results[0].metadata["category"] == "tech"

    def test_get_methods(self, hybrid_search, query_processor, mock_dense_retriever, mock_sparse_retriever, fusion):
        """Test get methods."""
        assert hybrid_search.get_query_processor() == query_processor
        assert hybrid_search.get_dense_retriever() == mock_dense_retriever
        assert hybrid_search.get_sparse_retriever() == mock_sparse_retriever
        assert hybrid_search.get_fusion() == fusion

    def test_search_respects_top_k(self, hybrid_search, mock_dense_retriever, mock_sparse_retriever):
        """Test search respects top_k."""
        mock_dense_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id=f"d{i}", score=0.9 - i * 0.1, text=f"text {i}")
            for i in range(10)
        ]
        mock_sparse_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id=f"s{i}", score=2.5 - i * 0.1, text=f"text {i}")
            for i in range(10)
        ]

        results = hybrid_search.search("query", top_k=5)
        assert len(results) == 5


class TestHybridSearchIntegration:
    """Integration tests for HybridSearch."""

    def test_hybrid_search_with_settings(self):
        """Test hybrid search initialization with settings."""
        from src.core.settings import Settings

        settings = Settings()
        search = HybridSearch(settings=settings)

        assert search._settings == settings

    def test_hybrid_search_default_settings(self):
        """Test hybrid search with default settings."""
        search = HybridSearch()
        assert search._settings is not None
