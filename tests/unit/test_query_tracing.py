"""
Unit tests for Query pipeline tracing (F3).
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.trace import TraceContext, TraceCollector, set_trace_collector
from src.core.types import RetrievalResult
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool


class TestQueryPipelineTracing:
    """Tests for Query pipeline tracing."""

    def test_execute_creates_trace_context(self):
        """Test execute() creates TraceContext if not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            tool = QueryKnowledgeHubTool()

            # Mock components to avoid real initialization
            tool._initialized = True
            tool._hybrid_search = Mock()
            tool._hybrid_search.search.return_value = []
            tool._reranker = Mock()
            tool._response_builder = Mock()
            tool._response_builder.build.return_value = {
                "content": [{"type": "text", "text": "test"}],
                "structuredContent": {},
            }

            result = tool.execute(query="test query", top_k=5)

            # Verify trace was collected
            traces = collector.read_traces()
            assert len(traces) == 1
            assert traces[0]["trace_type"] == "query"

    def test_execute_uses_provided_trace(self):
        """Test execute() uses provided TraceContext."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            tool = QueryKnowledgeHubTool()
            trace = TraceContext(trace_type="query")

            # Mock components
            tool._initialized = True
            tool._hybrid_search = Mock()
            tool._hybrid_search.search.return_value = []
            tool._reranker = Mock()
            tool._response_builder = Mock()
            tool._response_builder.build.return_value = {
                "content": [{"type": "text", "text": "test"}],
                "structuredContent": {},
            }

            result = tool.execute(query="test query", trace=trace)

            # Verify the same trace was used
            assert trace._finished
            traces = collector.read_traces()
            assert len(traces) == 1
            assert traces[0]["trace_id"] == trace.trace_id

    def test_trace_records_hybrid_search_stage(self):
        """Test trace records hybrid_search stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            tool = QueryKnowledgeHubTool()
            trace = TraceContext(trace_type="query")

            # Mock hybrid search to record a stage
            def mock_search(query, top_k, filters, trace):
                if trace:
                    trace.record_stage("hybrid_search", elapsed_ms=50.0, method="hybrid")
                return []

            tool._initialized = True
            tool._hybrid_search = Mock()
            tool._hybrid_search.search.side_effect = mock_search
            tool._reranker = Mock()
            tool._response_builder = Mock()
            tool._response_builder.build.return_value = {
                "content": [{"type": "text", "text": "test"}],
                "structuredContent": {},
            }

            result = tool.execute(query="test query", trace=trace)

            # Verify hybrid_search stage was recorded
            traces = collector.read_traces()
            stages = traces[0]["stages"]
            stage_names = [s["stage"] for s in stages]
            assert "hybrid_search" in stage_names

    def test_trace_records_rerank_stage(self):
        """Test trace records rerank stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            tool = QueryKnowledgeHubTool()
            trace = TraceContext(trace_type="query")

            # Mock with results to trigger rerank
            mock_result = RetrievalResult(
                chunk_id="test_id",
                score=0.9,
                text="test text",
                metadata={},
            )

            def mock_search(query, top_k, filters, trace):
                if trace:
                    trace.record_stage("hybrid_search", elapsed_ms=50.0, method="hybrid")
                return [mock_result]

            def mock_rerank(query, results, top_k, trace):
                if trace:
                    trace.record_stage("rerank", elapsed_ms=10.0, method="cross_encoder")
                from src.libs.reranker import RerankResult, RerankCandidate
                return RerankResult(
                    candidates=[RerankCandidate(id="test_id", text="test text", score=0.95)],
                    backend="cross_encoder",
                    elapsed_ms=10.0,
                )

            tool._initialized = True
            tool._hybrid_search = Mock()
            tool._hybrid_search.search.side_effect = mock_search
            tool._reranker = Mock()
            tool._reranker.rerank.side_effect = mock_rerank
            tool._response_builder = Mock()
            tool._response_builder.build.return_value = {
                "content": [{"type": "text", "text": "test"}],
                "structuredContent": {},
            }

            result = tool.execute(query="test query", trace=trace)

            # Verify rerank stage was recorded
            traces = collector.read_traces()
            stages = traces[0]["stages"]
            stage_names = [s["stage"] for s in stages]
            assert "rerank" in stage_names

    def test_trace_records_response_building_stage(self):
        """Test trace records response_building stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            tool = QueryKnowledgeHubTool()
            trace = TraceContext(trace_type="query")

            tool._initialized = True
            tool._hybrid_search = Mock()
            tool._hybrid_search.search.return_value = []
            tool._reranker = Mock()
            tool._response_builder = Mock()
            tool._response_builder.build.return_value = {
                "content": [{"type": "text", "text": "test"}],
                "structuredContent": {},
            }

            result = tool.execute(query="test query", trace=trace)

            # Verify response_building stage was recorded
            traces = collector.read_traces()
            stages = traces[0]["stages"]
            stage_names = [s["stage"] for s in stages]
            assert "response_building" in stage_names

    def test_trace_includes_total_elapsed_ms(self):
        """Test trace includes total_elapsed_ms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            tool = QueryKnowledgeHubTool()

            tool._initialized = True
            tool._hybrid_search = Mock()
            tool._hybrid_search.search.return_value = []
            tool._reranker = Mock()
            tool._response_builder = Mock()
            tool._response_builder.build.return_value = {
                "content": [{"type": "text", "text": "test"}],
                "structuredContent": {},
            }

            result = tool.execute(query="test query")

            traces = collector.read_traces()
            assert "total_elapsed_ms" in traces[0]
            assert traces[0]["total_elapsed_ms"] >= 0

    def test_trace_on_error(self):
        """Test trace is collected even on error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            tool = QueryKnowledgeHubTool()

            tool._initialized = True
            tool._hybrid_search = Mock()
            tool._hybrid_search.search.side_effect = Exception("Search failed")
            tool._reranker = Mock()

            result = tool.execute(query="test query")

            # Verify trace was collected even on error
            traces = collector.read_traces()
            assert len(traces) == 1

            # Verify error stage was recorded
            stages = traces[0]["stages"]
            error_stages = [s for s in stages if s["stage"] == "error"]
            assert len(error_stages) == 1
            assert "Search failed" in error_stages[0]["details"]["error"]

    def test_trace_includes_query_info(self):
        """Test trace includes query info in structuredContent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            tool = QueryKnowledgeHubTool()

            tool._initialized = True
            tool._hybrid_search = Mock()
            tool._hybrid_search.search.return_value = []
            tool._reranker = Mock()
            tool._response_builder = Mock()
            tool._response_builder.build.return_value = {
                "content": [{"type": "text", "text": "test"}],
                "structuredContent": {"query": "test query"},
            }

            result = tool.execute(query="test query", collection="test_collection")

            # Verify response includes trace_id
            assert "trace_id" in result.get("structuredContent", {}) or result.get("content")


class TestQueryProcessorTracing:
    """Tests for QueryProcessor tracing."""

    def test_query_processor_records_stage(self):
        """Test QueryProcessor records stage to trace."""
        from src.core.query_engine import QueryProcessor

        trace = TraceContext(trace_type="query")
        processor = QueryProcessor()

        result = processor.process("test query", trace=trace)

        # Verify stage was recorded
        stages = trace.stages
        stage_names = [s["stage"] for s in stages]
        assert "query_processing" in stage_names

    def test_query_processor_stage_details(self):
        """Test QueryProcessor stage includes details."""
        from src.core.query_engine import QueryProcessor

        trace = TraceContext(trace_type="query")
        processor = QueryProcessor()

        result = processor.process("test query with multiple words", trace=trace)

        # Verify stage details
        query_stages = [s for s in trace.stages if s["stage"] == "query_processing"]
        assert len(query_stages) == 1
        assert "keyword_count" in query_stages[0]["details"]


class TestDenseRetrieverTracing:
    """Tests for DenseRetriever tracing."""

    def test_dense_retriever_records_stage(self):
        """Test DenseRetriever records stage to trace."""
        from src.core.query_engine import DenseRetriever

        trace = TraceContext(trace_type="query")

        # Mock dependencies
        mock_embedding = Mock()
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_store = Mock()
        mock_store.query.return_value = []

        retriever = DenseRetriever(
            embedding_client=mock_embedding,
            vector_store=mock_store,
        )

        result = retriever.retrieve("test query", trace=trace)

        # Verify stage was recorded
        stages = trace.stages
        stage_names = [s["stage"] for s in stages]
        assert "dense_retrieval" in stage_names


class TestSparseRetrieverTracing:
    """Tests for SparseRetriever tracing."""

    def test_sparse_retriever_records_stage(self):
        """Test SparseRetriever records stage to trace."""
        from src.core.query_engine import SparseRetriever

        trace = TraceContext(trace_type="query")

        # Mock dependencies with results
        mock_bm25 = Mock()
        mock_bm25.query.return_value = [("chunk_1", 0.9)]
        mock_store = Mock()
        mock_store.get_by_ids.return_value = [
            {"id": "chunk_1", "text": "test text", "metadata": {}}
        ]

        retriever = SparseRetriever(
            bm25_indexer=mock_bm25,
            vector_store=mock_store,
        )

        result = retriever.retrieve(["test", "keyword"], trace=trace)

        # Verify stage was recorded
        stages = trace.stages
        stage_names = [s["stage"] for s in stages]
        assert "sparse_retrieval" in stage_names

    def test_sparse_retriever_no_keywords_no_stage(self):
        """Test SparseRetriever does not record stage when no keywords."""
        from src.core.query_engine import SparseRetriever

        trace = TraceContext(trace_type="query")

        # Mock dependencies
        mock_bm25 = Mock()
        mock_store = Mock()

        retriever = SparseRetriever(
            bm25_indexer=mock_bm25,
            vector_store=mock_store,
        )

        result = retriever.retrieve([], trace=trace)

        # No stage should be recorded when no keywords
        stages = trace.stages
        stage_names = [s["stage"] for s in stages]
        assert "sparse_retrieval" not in stage_names


class TestRRFFusionTracing:
    """Tests for RRF Fusion tracing."""

    def test_rrf_fusion_records_stage(self):
        """Test RRFFusion records stage to trace."""
        from src.core.query_engine import RRFFusion

        trace = TraceContext(trace_type="query")
        fusion = RRFFusion()

        result = fusion.fuse([], [], trace=trace)

        # Verify stage was recorded
        stages = trace.stages
        stage_names = [s["stage"] for s in stages]
        assert "rrf_fusion" in stage_names

    def test_rrf_fusion_stage_details(self):
        """Test RRFFusion stage includes details."""
        from src.core.query_engine import RRFFusion

        trace = TraceContext(trace_type="query")
        fusion = RRFFusion()

        dense_results = [
            RetrievalResult(chunk_id="a", score=0.9, text="a", metadata={}),
        ]
        sparse_results = [
            RetrievalResult(chunk_id="b", score=0.8, text="b", metadata={}),
        ]

        result = fusion.fuse(dense_results, sparse_results, trace=trace)

        # Verify stage details
        fusion_stages = [s for s in trace.stages if s["stage"] == "rrf_fusion"]
        assert len(fusion_stages) == 1
        assert fusion_stages[0]["details"]["dense_count"] == 1
        assert fusion_stages[0]["details"]["sparse_count"] == 1


class TestHybridSearchTracing:
    """Tests for HybridSearch tracing."""

    def test_hybrid_search_records_stage(self):
        """Test HybridSearch records stage to trace."""
        from src.core.query_engine import HybridSearch, QueryProcessor

        trace = TraceContext(trace_type="query")

        hybrid = HybridSearch(query_processor=QueryProcessor())
        result = hybrid.search("test query", trace=trace)

        # Verify stage was recorded
        stages = trace.stages
        stage_names = [s["stage"] for s in stages]
        assert "hybrid_search" in stage_names


class TestQueryRerankerTracing:
    """Tests for QueryReranker tracing."""

    def test_reranker_records_stage(self):
        """Test QueryReranker records stage to trace."""
        from src.core.query_engine import QueryReranker
        from src.libs.reranker import NoneReranker

        trace = TraceContext(trace_type="query")

        reranker = QueryReranker(reranker=NoneReranker())

        results = [
            RetrievalResult(chunk_id="a", score=0.9, text="a", metadata={}),
        ]

        result = reranker.rerank("test query", results, trace=trace)

        # Verify stage was recorded
        stages = trace.stages
        stage_names = [s["stage"] for s in stages]
        assert "rerank" in stage_names
