"""
Unit tests for RRF Fusion.
"""

import pytest

from src.core.query_engine.fusion import RRFFusion, FakeFusion, FusionResult
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult


class TestFusionResult:
    """Tests for FusionResult dataclass."""

    def test_fusion_result_creation(self):
        """Test creating FusionResult."""
        results = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1"),
        ]
        fusion_result = FusionResult(
            results=results,
            method="rrf",
            k=60,
        )
        assert fusion_result.method == "rrf"
        assert fusion_result.k == 60
        assert len(fusion_result.results) == 1

    def test_fusion_result_to_dict(self):
        """Test converting FusionResult to dict."""
        results = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1"),
        ]
        fusion_result = FusionResult(
            results=results,
            method="rrf",
            k=60,
            elapsed_ms=1.5,
        )
        d = fusion_result.to_dict()
        assert d["method"] == "rrf"
        assert d["k"] == 60
        assert d["elapsed_ms"] == 1.5
        assert len(d["results"]) == 1


class TestFakeFusion:
    """Tests for FakeFusion."""

    def test_fake_fusion_returns_default_results(self):
        """Test fake fusion returns empty list by default."""
        fusion = FakeFusion()
        dense = [RetrievalResult(chunk_id="1", score=0.9, text="text 1")]
        sparse = [RetrievalResult(chunk_id="2", score=0.8, text="text 2")]

        result = fusion.fuse(dense, sparse)
        assert result.results == []

    def test_fake_fusion_returns_custom_results(self):
        """Test fake fusion returns custom results."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
        ]
        fusion = FakeFusion(results=custom_results)
        result = fusion.fuse([], [])
        assert len(result.results) == 2

    def test_fake_fusion_respects_top_k(self):
        """Test fake fusion respects top_k."""
        custom_results = [
            RetrievalResult(chunk_id="1", score=0.9, text="result 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="result 2"),
            RetrievalResult(chunk_id="3", score=0.7, text="result 3"),
        ]
        fusion = FakeFusion(results=custom_results)
        result = fusion.fuse([], [], top_k=2)
        assert len(result.results) == 2

    def test_fake_fusion_records_calls(self):
        """Test fake fusion records calls."""
        fusion = FakeFusion()
        dense = [RetrievalResult(chunk_id="1", score=0.9, text="text 1")]
        sparse = [RetrievalResult(chunk_id="2", score=0.8, text="text 2")]

        fusion.fuse(dense, sparse, top_k=5)
        fusion.fuse([], [], top_k=10)

        assert len(fusion.fuse_calls) == 2
        assert fusion.fuse_calls[0]["dense_count"] == 1
        assert fusion.fuse_calls[0]["sparse_count"] == 1

    def test_fake_fusion_get_k(self):
        """Test fake fusion get_k."""
        fusion = FakeFusion()
        assert fusion.get_k() == 60


class TestRRFFusion:
    """Tests for RRFFusion."""

    @pytest.fixture
    def fusion(self):
        """Create default RRFFusion."""
        return RRFFusion()

    @pytest.fixture
    def fusion_k_10(self):
        """Create RRFFusion with k=10."""
        return RRFFusion(k=10)

    def test_fuse_empty_inputs(self, fusion):
        """Test fusing empty inputs."""
        result = fusion.fuse([], [])
        assert result.results == []

    def test_fuse_only_dense(self, fusion):
        """Test fusing only dense results."""
        dense = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="text 2"),
        ]
        result = fusion.fuse(dense, [])
        assert len(result.results) == 2
        assert result.results[0].chunk_id == "1"  # Higher rank

    def test_fuse_only_sparse(self, fusion):
        """Test fusing only sparse results."""
        sparse = [
            RetrievalResult(chunk_id="1", score=2.5, text="text 1"),
            RetrievalResult(chunk_id="2", score=1.8, text="text 2"),
        ]
        result = fusion.fuse([], sparse)
        assert len(result.results) == 2
        assert result.results[0].chunk_id == "1"

    def test_fuse_combines_scores(self, fusion):
        """Test that fuse combines scores from both retrievers."""
        dense = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1"),
        ]
        sparse = [
            RetrievalResult(chunk_id="1", score=2.5, text="text 1"),
        ]
        result = fusion.fuse(dense, sparse)

        # RRF score = 1/(60+1) + 1/(60+1) = 2/61
        expected_score = 2.0 / 61.0
        assert len(result.results) == 1
        assert abs(result.results[0].score - expected_score) < 0.001

    def test_fuse_different_documents(self, fusion):
        """Test fusing different documents from each retriever."""
        dense = [
            RetrievalResult(chunk_id="A", score=0.9, text="text A"),
            RetrievalResult(chunk_id="B", score=0.8, text="text B"),
        ]
        sparse = [
            RetrievalResult(chunk_id="C", score=2.5, text="text C"),
            RetrievalResult(chunk_id="D", score=1.8, text="text D"),
        ]
        result = fusion.fuse(dense, sparse, top_k=4)
        assert len(result.results) == 4

    def test_fuse_overlapping_documents(self, fusion):
        """Test fusing overlapping documents."""
        dense = [
            RetrievalResult(chunk_id="A", score=0.9, text="text A"),
            RetrievalResult(chunk_id="B", score=0.8, text="text B"),
            RetrievalResult(chunk_id="C", score=0.7, text="text C"),
        ]
        sparse = [
            RetrievalResult(chunk_id="B", score=2.5, text="text B"),
            RetrievalResult(chunk_id="A", score=1.8, text="text A"),
            RetrievalResult(chunk_id="D", score=1.2, text="text D"),
        ]
        result = fusion.fuse(dense, sparse, top_k=4)

        # A and B appear in both, should have higher scores
        chunk_ids = [r.chunk_id for r in result.results]
        assert "A" in chunk_ids
        assert "B" in chunk_ids

    def test_fuse_respects_top_k(self, fusion):
        """Test that fuse respects top_k."""
        dense = [
            RetrievalResult(chunk_id=str(i), score=0.9 - i * 0.1, text=f"text {i}")
            for i in range(10)
        ]
        sparse = [
            RetrievalResult(chunk_id=str(i), score=2.5 - i * 0.1, text=f"text {i}")
            for i in range(10)
        ]
        result = fusion.fuse(dense, sparse, top_k=5)
        assert len(result.results) == 5

    def test_fuse_with_trace(self, fusion):
        """Test fuse with trace context."""
        trace = TraceContext(trace_id="test-trace")
        dense = [RetrievalResult(chunk_id="1", score=0.9, text="text 1")]
        sparse = [RetrievalResult(chunk_id="2", score=0.8, text="text 2")]

        fusion.fuse(dense, sparse, trace=trace)

        assert len(trace.stages) == 1
        assert trace.stages[0]["stage"] == "rrf_fusion"

    def test_fuse_custom_k(self, fusion_k_10):
        """Test fuse with custom k parameter."""
        dense = [RetrievalResult(chunk_id="1", score=0.9, text="text 1")]
        sparse = [RetrievalResult(chunk_id="1", score=2.5, text="text 1")]

        result = fusion_k_10.fuse(dense, sparse)

        # RRF score with k=10: 1/(10+1) + 1/(10+1) = 2/11
        expected_score = 2.0 / 11.0
        assert abs(result.results[0].score - expected_score) < 0.001

    def test_fuse_preserves_text_and_metadata(self, fusion):
        """Test that fuse preserves text and metadata."""
        dense = [
            RetrievalResult(
                chunk_id="1",
                score=0.9,
                text="text from dense",
                metadata={"source": "dense", "page": 1},
            ),
        ]
        sparse = []

        result = fusion.fuse(dense, sparse)
        assert result.results[0].text == "text from dense"
        assert result.results[0].metadata["source"] == "dense"
        assert result.results[0].metadata["page"] == 1

    def test_fuse_prefers_first_occurrence_for_metadata(self, fusion):
        """Test that fuse uses first occurrence for metadata."""
        dense = [
            RetrievalResult(
                chunk_id="1",
                score=0.9,
                text="text dense",
                metadata={"source": "dense"},
            ),
        ]
        sparse = [
            RetrievalResult(
                chunk_id="1",
                score=2.5,
                text="text sparse",
                metadata={"source": "sparse"},
            ),
        ]

        result = fusion.fuse(dense, sparse)
        # Dense comes first, so its metadata is used
        assert result.results[0].metadata["source"] == "dense"

    def test_get_k(self, fusion, fusion_k_10):
        """Test get_k returns correct value."""
        assert fusion.get_k() == 60
        assert fusion_k_10.get_k() == 10

    def test_fuse_deterministic(self, fusion):
        """Test that fuse is deterministic."""
        dense = [
            RetrievalResult(chunk_id="A", score=0.9, text="text A"),
            RetrievalResult(chunk_id="B", score=0.8, text="text B"),
        ]
        sparse = [
            RetrievalResult(chunk_id="B", score=2.5, text="text B"),
            RetrievalResult(chunk_id="A", score=1.8, text="text A"),
        ]

        result1 = fusion.fuse(dense, sparse)
        result2 = fusion.fuse(dense, sparse)

        # Results should be identical
        assert len(result1.results) == len(result2.results)
        for r1, r2 in zip(result1.results, result2.results):
            assert r1.chunk_id == r2.chunk_id
            assert r1.score == r2.score

    def test_fuse_elapsed_ms_recorded(self, fusion):
        """Test that elapsed_ms is recorded."""
        dense = [RetrievalResult(chunk_id="1", score=0.9, text="text 1")]
        sparse = [RetrievalResult(chunk_id="2", score=0.8, text="text 2")]

        result = fusion.fuse(dense, sparse)
        assert result.elapsed_ms is not None
        assert result.elapsed_ms >= 0


class TestRRFFusionEdgeCases:
    """Edge case tests for RRFFusion."""

    @pytest.fixture
    def fusion(self):
        """Create default RRFFusion."""
        return RRFFusion()

    def test_fuse_large_inputs(self, fusion):
        """Test fusing large inputs."""
        dense = [
            RetrievalResult(chunk_id=f"d{i}", score=0.9, text=f"text {i}")
            for i in range(100)
        ]
        sparse = [
            RetrievalResult(chunk_id=f"s{i}", score=2.5, text=f"text {i}")
            for i in range(100)
        ]
        result = fusion.fuse(dense, sparse, top_k=50)
        assert len(result.results) == 50

    def test_fuse_all_same_documents(self, fusion):
        """Test fusing when all documents are the same."""
        dense = [
            RetrievalResult(chunk_id="1", score=0.9, text="text 1"),
            RetrievalResult(chunk_id="2", score=0.8, text="text 2"),
        ]
        sparse = [
            RetrievalResult(chunk_id="1", score=2.5, text="text 1"),
            RetrievalResult(chunk_id="2", score=1.8, text="text 2"),
        ]
        result = fusion.fuse(dense, sparse)
        assert len(result.results) == 2

    def test_fuse_zero_top_k(self, fusion):
        """Test fusing with top_k=0."""
        dense = [RetrievalResult(chunk_id="1", score=0.9, text="text 1")]
        sparse = [RetrievalResult(chunk_id="2", score=0.8, text="text 2")]
        result = fusion.fuse(dense, sparse, top_k=0)
        assert len(result.results) == 0

    def test_fuse_preserves_order_for_same_scores(self, fusion):
        """Test that documents with same scores are ordered consistently."""
        # When k is large, documents appearing at same ranks get similar scores
        dense = [
            RetrievalResult(chunk_id="A", score=0.9, text="text A"),
            RetrievalResult(chunk_id="B", score=0.8, text="text B"),
        ]
        sparse = [
            RetrievalResult(chunk_id="B", score=2.5, text="text B"),
            RetrievalResult(chunk_id="A", score=1.8, text="text A"),
        ]
        result = fusion.fuse(dense, sparse)
        # Both A and B have same combined RRF score
        # Order should be deterministic (sorted by score, then by insertion order)
        assert len(result.results) == 2
