"""
BatchProcessor 单元测试。

测试批处理编排功能。
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.core.types import Chunk, ChunkRecord
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.batch_processor import BatchProcessor, FakeBatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_chunks():
    """创建测试用 Chunk 列表。"""
    return [
        Chunk(id=f"chunk_{i:03d}", text=f"Text {i}", metadata={"index": i})
        for i in range(5)
    ]


@pytest.fixture
def mock_dense_encoder():
    """创建 Mock DenseEncoder。"""
    encoder = Mock(spec=DenseEncoder)
    encoder.encode = Mock(side_effect=lambda chunks, trace=None: [
        ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
            dense_vector=[0.1] * 128,
        )
        for chunk in chunks
    ])
    return encoder


@pytest.fixture
def mock_sparse_encoder():
    """创建 Mock SparseEncoder。"""
    encoder = Mock(spec=SparseEncoder)
    encoder.encode = Mock(side_effect=lambda chunks, trace=None: [
        ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
            sparse_vector={"term": 1.0},
        )
        for chunk in chunks
    ])
    return encoder


@pytest.fixture
def trace_context():
    """创建 TraceContext。"""
    return TraceContext(trace_type="ingestion")


# ============================================================================
# BatchProcessor 测试
# ============================================================================


class TestBatchProcessor:
    """BatchProcessor 测试。"""

    def test_process_returns_correct_count(
        self, mock_dense_encoder, mock_sparse_encoder, sample_chunks
    ):
        """测试处理返回正确数量的记录。"""
        processor = BatchProcessor(
            dense_encoder=mock_dense_encoder,
            sparse_encoder=mock_sparse_encoder,
            batch_size=2,
        )
        records = processor.process(sample_chunks)

        assert len(records) == len(sample_chunks)

    def test_process_with_batch_size_2(self, sample_chunks):
        """测试 batch_size=2 时对 5 chunks 分成 3 批。"""
        processor = BatchProcessor(batch_size=2)

        batches = processor._split_batches(sample_chunks)

        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_process_batches_are_stable(self, sample_chunks):
        """测试批次顺序稳定。"""
        processor = BatchProcessor(batch_size=2)

        batches1 = processor._split_batches(sample_chunks)
        batches2 = processor._split_batches(sample_chunks)

        for b1, b2 in zip(batches1, batches2):
            assert [c.id for c in b1] == [c.id for c in b2]

    def test_process_empty_chunks(
        self, mock_dense_encoder, mock_sparse_encoder
    ):
        """测试处理空 Chunk 列表。"""
        processor = BatchProcessor(
            dense_encoder=mock_dense_encoder,
            sparse_encoder=mock_sparse_encoder,
        )
        records = processor.process([])

        assert records == []

    def test_process_calls_dense_encoder(
        self, mock_dense_encoder, mock_sparse_encoder, sample_chunks
    ):
        """测试处理调用 DenseEncoder。"""
        processor = BatchProcessor(
            dense_encoder=mock_dense_encoder,
            sparse_encoder=mock_sparse_encoder,
            batch_size=3,
        )
        processor.process(sample_chunks)

        # batch_size=3, 5 chunks -> 2 batches
        assert mock_dense_encoder.encode.call_count == 2

    def test_process_calls_sparse_encoder(
        self, mock_dense_encoder, mock_sparse_encoder, sample_chunks
    ):
        """测试处理调用 SparseEncoder。"""
        processor = BatchProcessor(
            dense_encoder=mock_dense_encoder,
            sparse_encoder=mock_sparse_encoder,
            batch_size=3,
        )
        processor.process(sample_chunks)

        # batch_size=3, 5 chunks -> 2 batches
        assert mock_sparse_encoder.encode.call_count == 2

    def test_process_without_dense_encoder(self, mock_sparse_encoder, sample_chunks):
        """测试没有 DenseEncoder 时只进行 Sparse 编码。"""
        processor = BatchProcessor(
            dense_encoder=None,
            sparse_encoder=mock_sparse_encoder,
        )
        records = processor.process(sample_chunks)

        for record in records:
            assert record.dense_vector is None
            assert record.sparse_vector is not None

    def test_process_without_sparse_encoder(self, mock_dense_encoder, sample_chunks):
        """测试没有 SparseEncoder 时只进行 Dense 编码。"""
        processor = BatchProcessor(
            dense_encoder=mock_dense_encoder,
            sparse_encoder=None,
        )
        records = processor.process(sample_chunks)

        for record in records:
            assert record.dense_vector is not None
            assert record.sparse_vector is None

    def test_process_without_encoders(self, sample_chunks):
        """测试没有编码器时返回基本 ChunkRecord。"""
        processor = BatchProcessor()
        records = processor.process(sample_chunks)

        assert len(records) == len(sample_chunks)
        for record in records:
            assert record.dense_vector is None
            assert record.sparse_vector is None

    def test_process_preserves_chunk_info(
        self, mock_dense_encoder, mock_sparse_encoder, sample_chunks
    ):
        """测试处理保留 Chunk 信息。"""
        processor = BatchProcessor(
            dense_encoder=mock_dense_encoder,
            sparse_encoder=mock_sparse_encoder,
        )
        records = processor.process(sample_chunks)

        for i, record in enumerate(records):
            assert record.id == sample_chunks[i].id
            assert record.text == sample_chunks[i].text
            assert record.metadata == sample_chunks[i].metadata

    def test_process_with_trace(
        self, mock_dense_encoder, mock_sparse_encoder, sample_chunks, trace_context
    ):
        """测试处理记录追踪。"""
        processor = BatchProcessor(
            dense_encoder=mock_dense_encoder,
            sparse_encoder=mock_sparse_encoder,
        )
        processor.process(sample_chunks, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "batch_processing" for s in stages)

    def test_process_trace_includes_details(
        self, mock_dense_encoder, mock_sparse_encoder, sample_chunks, trace_context
    ):
        """测试追踪包含详细信息。"""
        processor = BatchProcessor(
            dense_encoder=mock_dense_encoder,
            sparse_encoder=mock_sparse_encoder,
            batch_size=2,
        )
        processor.process(sample_chunks, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        stage = next((s for s in stages if s["stage"] == "batch_processing"), None)

        assert stage is not None
        assert "chunk_count" in stage.get("details", {})
        assert "batch_count" in stage.get("details", {})
        assert "batch_size" in stage.get("details", {})

    def test_get_batch_count(self):
        """测试计算批次数。"""
        processor = BatchProcessor(batch_size=10)

        assert processor.get_batch_count(0) == 0
        assert processor.get_batch_count(5) == 1
        assert processor.get_batch_count(10) == 1
        assert processor.get_batch_count(11) == 2
        assert processor.get_batch_count(25) == 3

    def test_get_batch_size(self):
        """测试获取批次大小。"""
        processor = BatchProcessor(batch_size=64)

        assert processor.get_batch_size() == 64


# ============================================================================
# FakeBatchProcessor 测试
# ============================================================================


class TestFakeBatchProcessor:
    """FakeBatchProcessor 测试。"""

    def test_process_returns_correct_count(self, sample_chunks):
        """测试处理返回正确数量的记录。"""
        processor = FakeBatchProcessor(batch_size=2)
        records = processor.process(sample_chunks)

        assert len(records) == len(sample_chunks)

    def test_process_returns_vectors(self, sample_chunks):
        """测试处理返回向量。"""
        processor = FakeBatchProcessor()
        records = processor.process(sample_chunks)

        for record in records:
            assert record.dense_vector is not None
            assert record.sparse_vector is not None

    def test_process_empty_chunks(self):
        """测试处理空 Chunk 列表。"""
        processor = FakeBatchProcessor()
        records = processor.process([])

        assert records == []

    def test_process_with_trace(self, sample_chunks, trace_context):
        """测试处理记录追踪。"""
        processor = FakeBatchProcessor()
        processor.process(sample_chunks, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "batch_processing" for s in stages)

    def test_get_batch_count(self):
        """测试计算批次数。"""
        processor = FakeBatchProcessor(batch_size=10)

        assert processor.get_batch_count(0) == 0
        assert processor.get_batch_count(5) == 1
        assert processor.get_batch_count(10) == 1
        assert processor.get_batch_count(11) == 2

    def test_get_batch_size(self):
        """测试获取批次大小。"""
        processor = FakeBatchProcessor(batch_size=32)

        assert processor.get_batch_size() == 32

    def test_split_batches(self, sample_chunks):
        """测试批次分割。"""
        processor = FakeBatchProcessor(batch_size=2)
        batches = processor._split_batches(sample_chunks)

        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1


# ============================================================================
# 边界条件测试
# ============================================================================


class TestEdgeCases:
    """边界条件测试。"""

    def test_large_batch_size(self, sample_chunks):
        """测试大批次大小。"""
        processor = BatchProcessor(batch_size=100)
        batches = processor._split_batches(sample_chunks)

        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_batch_size_1(self):
        """测试批次大小为 1。"""
        chunks = [Chunk(id=f"c_{i}", text=f"T{i}", metadata={}) for i in range(3)]
        processor = BatchProcessor(batch_size=1)
        batches = processor._split_batches(chunks)

        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_large_chunk_count(
        self, mock_dense_encoder, mock_sparse_encoder
    ):
        """测试大量 Chunk。"""
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Text {i}", metadata={})
            for i in range(100)
        ]

        processor = BatchProcessor(
            dense_encoder=mock_dense_encoder,
            sparse_encoder=mock_sparse_encoder,
            batch_size=32,
        )
        records = processor.process(chunks)

        assert len(records) == 100

    def test_chunk_with_existing_metadata(
        self, mock_dense_encoder, mock_sparse_encoder
    ):
        """测试 Chunk 已有元数据。"""
        chunks = [
            Chunk(
                id="meta",
                text="Text",
                metadata={"title": "Test", "tags": ["a", "b"]},
            )
        ]

        processor = BatchProcessor(
            dense_encoder=mock_dense_encoder,
            sparse_encoder=mock_sparse_encoder,
        )
        records = processor.process(chunks)

        assert records[0].metadata["title"] == "Test"
        assert records[0].metadata["tags"] == ["a", "b"]
