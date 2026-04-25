"""
Unit tests for DenseRetriever.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.core.query_engine.dense_retriever import DenseRetriever, FakeDenseRetriever
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.vector_store import QueryResult


class TestFakeDenseRetriever:
    """Tests for FakeDenseRetriever."""

    def test_fake_retriever_returns_default_results(self):
        """Test fake retriever returns empty list by default."""
        retriever = FakeDenseRetriever()
        results = retriever.retrieve("test query")
        assert results == []

    def test_fake_retriever_returns_custom_results(self):
        """Test fake retriever returns custom results."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
        ]
        retriever = FakeDenseRetriever(results=custom_results)
        results = retriever.retrieve("test query")
        assert len(results) == 2
        assert results[0].chunk_id == "1"

    def test_fake_retriever_respects_top_k(self):
        """Test fake retriever respects top_k."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
            RetrievalResult(chunk_id="3", score=0.7, text="result 3"),
        ]
        retriever = FakeDenseRetriever(results=custom_results)
        results = retriever.retrieve("test query", top_k=2)
        assert len(results) == 2

    def test_fake_retriever_records_calls(self):
        """Test fake retriever records calls."""
        retriever = FakeDenseRetriever()
        retriever.retrieve("query 1", top_k=5)
        retriever.retrieve("query 2", top_k=10, filters={"collection": "test"})

        assert len(retriever.retrieve_calls) == 2
        assert retriever.retrieve_calls[0]["query"] == "query 1"
        assert retriever.retrieve_calls[1]["filters"] == {"collection": "test"}

    def test_fake_retriever_get_methods(self):
        """Test fake retriever get methods."""
        retriever = FakeDenseRetriever()
        assert retriever.get_embedding_client() is None
        assert retriever.get_vector_store() is None


class TestDenseRetriever:
    """Tests for DenseRetriever."""

    @pytest.fixture
    def mock_embedding_client(self):
        """Create mock embedding client."""
        client = Mock()
        client.embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        return client

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        store.query.return_value = [
            QueryResult(id="chunk_1", score=0.95, text="text 1", metadata={"page": 1}),
            QueryResult(id="chunk_2", score=0.85, text="text 2", metadata={"page": 2}),
        ]
        return store

    @pytest.fixture
    def retriever(self, mock_embedding_client, mock_vector_store):
        """Create DenseRetriever with mocked dependencies."""
        return DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

    def test_retrieve_returns_results(self, retriever):
        """Test retrieve returns results."""
        results = retriever.retrieve("test query")
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score == 0.95
        assert results[0].text == "text 1"

    def test_retrieve_calls_embedding(self, retriever, mock_embedding_client):
        """Test retrieve calls embedding client."""
        retriever.retrieve("test query")
        mock_embedding_client.embed.assert_called_once_with(["test query"])

    def test_retrieve_calls_vector_store(self, retriever, mock_embedding_client, mock_vector_store):
        """Test retrieve calls vector store with correct params."""
        retriever.retrieve("test query", top_k=5, filters={"collection": "test"})

        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert call_args[1]["vector"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert call_args[1]["top_k"] == 5
        assert call_args[1]["filters"] == {"collection": "test"}

    def test_retrieve_with_trace(self, retriever):
        """Test retrieve with trace context."""
        trace = TraceContext(trace_id="test-trace")
        retriever.retrieve("test query", trace=trace)

        assert len(trace.stages) == 1
        assert trace.stages[0]["stage"] == "dense_retrieval"

    def test_retrieve_converts_query_result_to_retrieval_result(self, retriever):
        """Test retrieve converts QueryResult to RetrievalResult."""
        results = retriever.retrieve("test query")

        assert isinstance(results[0], RetrievalResult)
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score == 0.95
        assert results[0].text == "text 1"
        assert results[0].metadata == {"page": 1}

    def test_retrieve_empty_results(self, mock_embedding_client, mock_vector_store):
        """Test retrieve with empty results."""
        mock_vector_store.query.return_value = []
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        results = retriever.retrieve("test query")
        assert results == []

    def test_retrieve_without_embedding_client_raises(self, mock_vector_store):
        """Test retrieve without embedding client raises error."""
        retriever = DenseRetriever(vector_store=mock_vector_store)

        with pytest.raises(RuntimeError, match="Embedding client not configured"):
            retriever.retrieve("test query")

    def test_retrieve_without_vector_store_raises(self, mock_embedding_client):
        """Test retrieve without vector store raises error."""
        retriever = DenseRetriever(embedding_client=mock_embedding_client)

        with pytest.raises(AttributeError):
            retriever.retrieve("test query")

    def test_retrieve_embedding_failure(self, mock_embedding_client, mock_vector_store):
        """Test retrieve handles embedding failure."""
        mock_embedding_client.embed.return_value = []
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        with pytest.raises(RuntimeError, match="Failed to embed query"):
            retriever.retrieve("test query")

    def test_get_embedding_client(self, retriever, mock_embedding_client):
        """Test get_embedding_client returns client."""
        assert retriever.get_embedding_client() == mock_embedding_client

    def test_get_vector_store(self, retriever, mock_vector_store):
        """Test get_vector_store returns store."""
        assert retriever.get_vector_store() == mock_vector_store

    def test_retrieve_with_custom_top_k(self, retriever, mock_vector_store):
        """Test retrieve with custom top_k."""
        retriever.retrieve("test query", top_k=20)
        call_args = mock_vector_store.query.call_args
        assert call_args[1]["top_k"] == 20

    def test_retrieve_with_filters(self, retriever, mock_vector_store):
        """Test retrieve with filters."""
        filters = {"collection": "my_docs", "doc_type": "pdf"}
        retriever.retrieve("test query", filters=filters)
        call_args = mock_vector_store.query.call_args
        assert call_args[1]["filters"] == filters

    def test_retrieve_preserves_metadata(self, mock_embedding_client, mock_vector_store):
        """Test retrieve preserves all metadata."""
        mock_vector_store.query.return_value = [
            QueryResult(
                id="chunk_1",
                score=0.95,
                text="text 1",
                metadata={"page": 1, "source": "doc.pdf", "collection": "test"},
            ),
        ]
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        results = retriever.retrieve("test query")
        assert results[0].metadata["page"] == 1
        assert results[0].metadata["source"] == "doc.pdf"
        assert results[0].metadata["collection"] == "test"


class TestDenseRetrieverIntegration:
    """Integration tests for DenseRetriever."""

    def test_retriever_with_settings(self):
        """Test retriever initialization with settings."""
        from src.core.settings import Settings

        settings = Settings()
        retriever = DenseRetriever(settings=settings)

        assert retriever._settings == settings

    def test_retriever_default_settings(self):
        """Test retriever with default settings."""
        retriever = DenseRetriever()
        assert retriever._settings is not None
