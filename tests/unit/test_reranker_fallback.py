"""
Unit tests for Core layer QueryReranker fallback behavior.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.core.query_engine.reranker import QueryReranker, FakeQueryReranker
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    RerankerError,
    NoneReranker,
)


class TestFakeQueryReranker:
    """Tests for FakeQueryReranker."""

    def test_fake_reranker_returns_default_results(self):
        """Test fake reranker returns empty list by default."""
        reranker = FakeQueryReranker()
        results = reranker.rerank("test query", [])
        assert results.candidates == []

    def test_fake_reranker_returns_custom_results(self):
        """Test fake reranker returns custom results."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
        ]
        reranker = FakeQueryReranker(results=custom_results)
        result = reranker.rerank("test query", [])
        assert len(result.candidates) == 2
        assert result.candidates[0].id == "1"

    def test_fake_reranker_respects_top_k(self):
        """Test fake reranker respects top_k."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
            RetrievalResult(chunk_id="3", score=0.7, text="result 3"),
        ]
        reranker = FakeQueryReranker(results=custom_results)
        result = reranker.rerank("test query", [], top_k=2)
        assert len(result.candidates) == 2

    def test_fake_reranker_records_calls(self):
        """Test fake reranker records calls."""
        reranker = FakeQueryReranker()
        reranker.rerank("query 1", [], top_k=5)
        reranker.rerank("query 2", [RetrievalResult(chunk_id="1", score=0.9, text="text")], top_k=10)

        assert len(reranker.rerank_calls) == 2
        assert reranker.rerank_calls[0]["query"] == "query 1"
        assert reranker.rerank_calls[1]["result_count"] == 1

    def test_fake_reranker_rerank_to_results(self):
        """Test fake reranker rerank_to_results method."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
        ]
        reranker = FakeQueryReranker(results=custom_results)
        results = reranker.rerank_to_results("test query", [], top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "1"

    def test_fake_reranker_is_enabled(self):
        """Test fake reranker is_enabled returns True."""
        reranker = FakeQueryReranker()
        assert reranker.is_enabled() is True

    def test_fake_reranker_get_reranker(self):
        """Test fake reranker get_reranker returns None."""
        reranker = FakeQueryReranker()
        assert reranker.get_reranker() is None


class TestQueryReranker:
    """Tests for QueryReranker."""

    @pytest.fixture
    def mock_reranker(self):
        """Create mock reranker that succeeds."""
        reranker = Mock(spec=BaseReranker)
        reranker.get_backend_name.return_value = "mock"
        reranker.is_available.return_value = True
        reranker.rerank.return_value = RerankResult(
            candidates=[
                RerankCandidate(id="2", text="text 2", score=0.95),
                RerankCandidate(id="1", text="text 1", score=0.85),
            ],
            backend="mock",
            elapsed_ms=1.5,
        )
        return reranker

    @pytest.fixture
    def failing_reranker(self):
        """Create mock reranker that fails."""
        reranker = Mock(spec=BaseReranker)
        reranker.get_backend_name.return_value = "failing"
        reranker.is_available.return_value = True
        reranker.rerank.side_effect = RerankerError("Reranker failed", backend="failing")
        return reranker

    @pytest.fixture
    def unavailable_reranker(self):
        """Create mock reranker that is unavailable."""
        reranker = Mock(spec=BaseReranker)
        reranker.get_backend_name.return_value = "unavailable"
        reranker.is_available.return_value = False
        return reranker

    @pytest.fixture
    def retrieval_results(self):
        """Create sample retrieval results."""
        return [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1", metadata={"page": 1}),
            RetrievalResult(chunk_id="2", score=0.8, text="text 2", metadata={"page": 2}),
            RetrievalResult(chunk_id="3", score=0.7, text="text 3", metadata={"page": 3}),
        ]

    def test_rerank_returns_results(self, mock_reranker, retrieval_results):
        """Test rerank returns results."""
        query_reranker = QueryReranker(reranker=mock_reranker)
        result = query_reranker.rerank("test query", retrieval_results)
        assert len(result.candidates) == 2
        assert result.backend == "mock"

    def test_rerank_calls_underlying_reranker(self, mock_reranker, retrieval_results):
        """Test rerank calls underlying reranker."""
        query_reranker = QueryReranker(reranker=mock_reranker)
        query_reranker.rerank("test query", retrieval_results, top_k=5)
        mock_reranker.rerank.assert_called_once()

    def test_rerank_with_trace(self, mock_reranker, retrieval_results):
        """Test rerank with trace context."""
        trace = TraceContext(trace_id="test-trace")
        query_reranker = QueryReranker(reranker=mock_reranker)
        query_reranker.rerank("test query", retrieval_results, trace=trace)

        assert len(trace.stages) == 1
        assert trace.stages[0]["stage"] == "rerank"

    def test_rerank_fallback_on_reranker_error(self, failing_reranker, retrieval_results):
        """Test rerank fallback on RerankerError."""
        query_reranker = QueryReranker(reranker=failing_reranker)
        result = query_reranker.rerank("test query", retrieval_results)

        assert result.fallback_used is True
        assert "Reranker failed" in result.fallback_reason
        # Fallback should preserve original order
        assert result.candidates[0].id == "1"

    def test_rerank_fallback_on_unavailable(self, unavailable_reranker, retrieval_results):
        """Test rerank fallback when reranker is unavailable."""
        query_reranker = QueryReranker(reranker=unavailable_reranker)
        result = query_reranker.rerank("test query", retrieval_results)

        assert result.fallback_used is True
        assert "not available" in result.fallback_reason

    def test_rerank_fallback_on_unexpected_error(self, retrieval_results):
        """Test rerank fallback on unexpected error."""
        failing_reranker = Mock(spec=BaseReranker)
        failing_reranker.get_backend_name.return_value = "error"
        failing_reranker.is_available.return_value = True
        failing_reranker.rerank.side_effect = RuntimeError("Unexpected error")

        query_reranker = QueryReranker(reranker=failing_reranker)
        result = query_reranker.rerank("test query", retrieval_results)

        assert result.fallback_used is True
        assert "Unexpected error" in result.fallback_reason

    def test_rerank_fallback_on_none_reranker(self, retrieval_results):
        """Test rerank fallback when reranker is None."""
        query_reranker = QueryReranker(reranker=None)
        result = query_reranker.rerank("test query", retrieval_results)

        # Should use NoneReranker (no fallback flag since it's intentional)
        assert result.backend == "none"

    def test_rerank_to_results_preserves_metadata(self, mock_reranker, retrieval_results):
        """Test rerank_to_results preserves metadata from original results."""
        query_reranker = QueryReranker(reranker=mock_reranker)
        results = query_reranker.rerank_to_results("test query", retrieval_results)

        # Check that metadata is preserved
        assert results[0].metadata["page"] == 2  # From original result with chunk_id="2"
        assert results[1].metadata["page"] == 1  # From original result with chunk_id="1"

    def test_rerank_to_results_updates_scores(self, mock_reranker, retrieval_results):
        """Test rerank_to_results updates scores from reranker."""
        query_reranker = QueryReranker(reranker=mock_reranker)
        results = query_reranker.rerank_to_results("test query", retrieval_results)

        # Scores should be from reranker (0.95 and 0.85), not original (0.9 and 0.8)
        assert results[0].score == 0.95
        assert results[1].score == 0.85

    def test_rerank_respects_top_k(self, mock_reranker, retrieval_results):
        """Test rerank respects top_k."""
        query_reranker = QueryReranker(reranker=mock_reranker)
        query_reranker.rerank("test query", retrieval_results, top_k=2)
        mock_reranker.rerank.assert_called_once()
        call_args = mock_reranker.rerank.call_args
        # top_k is passed as positional argument (3rd arg)
        assert call_args[0][2] == 2  # args: (query, candidates, top_k)

    def test_rerank_with_empty_results(self):
        """Test rerank with empty results."""
        # Use NoneReranker for this test since mock reranker returns preset results
        query_reranker = QueryReranker(reranker=None)
        result = query_reranker.rerank("test query", [])
        assert result.candidates == []

    def test_get_reranker_returns_instance(self, mock_reranker):
        """Test get_reranker returns the reranker instance."""
        query_reranker = QueryReranker(reranker=mock_reranker)
        assert query_reranker.get_reranker() == mock_reranker

    def test_is_enabled_returns_true_for_real_reranker(self, mock_reranker):
        """Test is_enabled returns True for real reranker."""
        query_reranker = QueryReranker(reranker=mock_reranker)
        assert query_reranker.is_enabled() is True

    def test_is_enabled_returns_false_for_none_reranker(self):
        """Test is_enabled returns False for NoneReranker."""
        none_reranker = NoneReranker()
        query_reranker = QueryReranker(reranker=none_reranker)
        assert query_reranker.is_enabled() is False

    def test_is_enabled_returns_false_for_none_instance(self):
        """Test is_enabled returns False when reranker is None."""
        query_reranker = QueryReranker(reranker=None)
        assert query_reranker.is_enabled() is False


class TestQueryRerankerIntegration:
    """Integration tests for QueryReranker."""

    def test_reranker_with_settings(self):
        """Test reranker initialization with settings."""
        from src.core.settings import Settings

        settings = Settings()
        query_reranker = QueryReranker(settings=settings)

        assert query_reranker._settings == settings

    def test_reranker_default_settings(self):
        """Test reranker with default settings."""
        query_reranker = QueryReranker()
        assert query_reranker._settings is not None

    def test_reranker_lazy_initialization(self):
        """Test reranker lazy initialization."""
        query_reranker = QueryReranker()

        # Initially _reranker is None
        assert query_reranker._reranker is None

        # After get_reranker, it should be initialized
        reranker = query_reranker.get_reranker()
        assert reranker is not None
        assert query_reranker._reranker is not None


class TestFallbackBehavior:
    """Detailed tests for fallback behavior."""

    @pytest.fixture
    def retrieval_results(self):
        """Create sample retrieval results with different scores."""
        return [
            RetrievalResult(chunk_id="A", score=0.95, text="text A", metadata={"source": "doc1"}),
            RetrievalResult(chunk_id="B", score=0.85, text="text B", metadata={"source": "doc2"}),
            RetrievalResult(chunk_id="C", score=0.75, text="text C", metadata={"source": "doc3"}),
        ]

    def test_fallback_preserves_original_order(self, retrieval_results):
        """Test fallback preserves original order."""
        query_reranker = QueryReranker(reranker=None)
        result = query_reranker.rerank("test query", retrieval_results)

        # Original order: A, B, C
        assert result.candidates[0].id == "A"
        assert result.candidates[1].id == "B"
        assert result.candidates[2].id == "C"

    def test_fallback_preserves_original_scores(self, retrieval_results):
        """Test fallback preserves original scores."""
        query_reranker = QueryReranker(reranker=None)
        result = query_reranker.rerank("test query", retrieval_results)

        assert result.candidates[0].score == 0.95
        assert result.candidates[1].score == 0.85
        assert result.candidates[2].score == 0.75

    def test_fallback_preserves_metadata(self, retrieval_results):
        """Test fallback preserves metadata."""
        query_reranker = QueryReranker(reranker=None)
        results = query_reranker.rerank_to_results("test query", retrieval_results)

        assert results[0].metadata["source"] == "doc1"
        assert results[1].metadata["source"] == "doc2"
        assert results[2].metadata["source"] == "doc3"

    def test_fallback_trace_records_details(self, retrieval_results):
        """Test fallback trace records details."""
        trace = TraceContext(trace_id="test-trace")

        # Create a failing reranker
        failing_reranker = Mock(spec=BaseReranker)
        failing_reranker.get_backend_name.return_value = "failing"
        failing_reranker.is_available.return_value = True
        failing_reranker.rerank.side_effect = RerankerError("Test error", backend="failing")

        query_reranker = QueryReranker(reranker=failing_reranker)
        query_reranker.rerank("test query", retrieval_results, trace=trace)

        # Check trace details
        stage = trace.stages[0]
        assert stage["details"]["fallback_used"] is True
        assert "Test error" in stage["details"]["fallback_reason"]

    def test_multiple_rerank_calls_with_fallback(self, retrieval_results):
        """Test multiple rerank calls with fallback."""
        query_reranker = QueryReranker(reranker=None)

        # First call
        result1 = query_reranker.rerank("query 1", retrieval_results)
        assert result1.fallback_used is False  # NoneReranker is intentional, not fallback

        # Second call
        result2 = query_reranker.rerank("query 2", retrieval_results)
        assert result2.fallback_used is False

    def test_rerank_result_to_dict(self, retrieval_results):
        """Test RerankResult to_dict method."""
        query_reranker = QueryReranker(reranker=None)
        result = query_reranker.rerank("test query", retrieval_results)

        d = result.to_dict()
        assert d["backend"] == "none"
        assert d["count"] == 3
        assert d["fallback_used"] is False


class TestEdgeCases:
    """Edge case tests for QueryReranker."""

    def test_rerank_with_single_result(self):
        """Test rerank with single result."""
        results = [RetrievalResult(chunk_id="1", score=0.9, text="text 1")]
        query_reranker = QueryReranker(reranker=None)
        result = query_reranker.rerank("test query", results)
        assert len(result.candidates) == 1

    def test_rerank_with_top_k_zero(self):
        """Test rerank with top_k=0."""
        results = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="text 2"),
        ]
        # Use NoneReranker for this test
        query_reranker = QueryReranker(reranker=None)
        result = query_reranker.rerank("test query", results, top_k=0)
        assert len(result.candidates) == 0

    def test_rerank_with_top_k_greater_than_results(self):
        """Test rerank with top_k greater than results count."""
        results = [RetrievalResult(chunk_id="1", score=0.9, text="text 1")]
        query_reranker = QueryReranker(reranker=None)
        result = query_reranker.rerank("test query", results, top_k=10)
        assert len(result.candidates) == 1

    def test_rerank_with_missing_metadata(self):
        """Test rerank with results missing metadata."""
        results = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1", metadata={}),
        ]
        query_reranker = QueryReranker(reranker=None)
        rerank_result = query_reranker.rerank_to_results("test query", results)
        assert rerank_result[0].metadata == {}

    def test_rerank_candidate_not_in_original(self):
        """Test rerank when candidate is not in original results."""
        mock_reranker = Mock(spec=BaseReranker)
        mock_reranker.get_backend_name.return_value = "mock"
        mock_reranker.is_available.return_value = True
        mock_reranker.rerank.return_value = RerankResult(
            candidates=[
                RerankCandidate(id="new_id", text="new text", score=0.99, metadata={"new": True}),
            ],
            backend="mock",
        )

        original_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1", metadata={"page": 1}),
        ]

        query_reranker = QueryReranker(reranker=mock_reranker)
        results = query_reranker.rerank_to_results("test query", original_results)

        # Should use candidate's data since it's not in original
        assert results[0].chunk_id == "new_id"
        assert results[0].text == "new text"
        assert results[0].metadata["new"] is True

    def test_rerank_with_large_input(self):
        """Test rerank with large input."""
        results = [
            RetrievalResult(chunk_id=f"id_{i}", score=0.9 - i * 0.01, text=f"text {i}")
            for i in range(100)
        ]
        query_reranker = QueryReranker(reranker=None)
        result = query_reranker.rerank("test query", results, top_k=50)
        assert len(result.candidates) == 50