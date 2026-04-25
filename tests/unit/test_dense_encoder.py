"""
DenseEncoder 单元测试。

测试稠密向量编码功能。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.dense_encoder import DenseEncoder, FakeDenseEncoder
from src.libs.embedding.base_embedding import EmbeddingResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_settings():
    """创建 Mock Settings。"""
    settings = Mock()
    settings.embedding = Mock()
    settings.embedding.provider = "openai"
    settings.embedding.model = "text-embedding-ada-002"
    return settings


@pytest.fixture
def mock_embedding():
    """创建 Mock Embedding。"""
    embedding = Mock()
    embedding.get_dimensions = Mock(return_value=1536)
    embedding.get_model_name = Mock(return_value="text-embedding-ada-002")
    # embed 方法根据输入文本数量返回对应数量的向量
    def embed_side_effect(texts, trace=None):
        return EmbeddingResult(
            vectors=[[0.1] * 1536 for _ in range(len(texts))],
            model="text-embedding-ada-002",
            dimensions=1536,
        )
    embedding.embed = Mock(side_effect=embed_side_effect)
    return embedding


@pytest.fixture
def sample_chunks():
    """创建测试用 Chunk 列表。"""
    return [
        Chunk(
            id="chunk_001",
            text="This is the first chunk.",
            metadata={"source_path": "test.pdf", "chunk_index": 0},
        ),
        Chunk(
            id="chunk_002",
            text="This is the second chunk.",
            metadata={"source_path": "test.pdf", "chunk_index": 1},
        ),
        Chunk(
            id="chunk_003",
            text="This is the third chunk.",
            metadata={"source_path": "test.pdf", "chunk_index": 2},
        ),
    ]


@pytest.fixture
def trace_context():
    """创建 TraceContext。"""
    return TraceContext(trace_type="ingestion")


# ============================================================================
# DenseEncoder 测试
# ============================================================================


class TestDenseEncoder:
    """DenseEncoder 测试。"""

    def test_encode_returns_correct_count(
        self, mock_settings, mock_embedding, sample_chunks
    ):
        """测试编码返回正确数量的记录。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        records = encoder.encode(sample_chunks)

        assert len(records) == len(sample_chunks)

    def test_encode_returns_correct_dimensions(
        self, mock_settings, mock_embedding, sample_chunks
    ):
        """测试编码返回正确维度的向量。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        records = encoder.encode(sample_chunks)

        for record in records:
            assert record.dense_vector is not None
            assert len(record.dense_vector) == 1536

    def test_encode_preserves_chunk_info(
        self, mock_settings, mock_embedding, sample_chunks
    ):
        """测试编码保留 Chunk 信息。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        records = encoder.encode(sample_chunks)

        for i, record in enumerate(records):
            assert record.id == sample_chunks[i].id
            assert record.text == sample_chunks[i].text
            assert record.metadata == sample_chunks[i].metadata

    def test_encode_empty_chunks(
        self, mock_settings, mock_embedding
    ):
        """测试编码空 Chunk 列表。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        records = encoder.encode([])

        assert records == []

    def test_encode_calls_embedding_with_texts(
        self, mock_settings, mock_embedding, sample_chunks
    ):
        """测试编码调用 Embedding 并传入正确文本。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        encoder.encode(sample_chunks)

        # 验证 embed 被调用，且传入了正确的文本列表
        mock_embedding.embed.assert_called_once()
        call_args = mock_embedding.embed.call_args
        texts = call_args[0][0]  # 第一个位置参数

        assert texts == [chunk.text for chunk in sample_chunks]

    def test_encode_with_trace(
        self, mock_settings, mock_embedding, sample_chunks, trace_context
    ):
        """测试编码记录追踪。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        records = encoder.encode(sample_chunks, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "dense_encoding" for s in stages)

    def test_encode_trace_includes_details(
        self, mock_settings, mock_embedding, sample_chunks, trace_context
    ):
        """测试追踪包含详细信息。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        records = encoder.encode(sample_chunks, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        stage = next((s for s in stages if s["stage"] == "dense_encoding"), None)

        assert stage is not None
        assert "chunk_count" in stage.get("details", {})
        assert "dimensions" in stage.get("details", {})

    def test_encode_texts(
        self, mock_settings, mock_embedding
    ):
        """测试 encode_texts 方法。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        texts = ["text1", "text2"]

        vectors = encoder.encode_texts(texts)

        assert len(vectors) == 2
        assert len(vectors[0]) == 1536

    def test_encode_texts_empty(
        self, mock_settings, mock_embedding
    ):
        """测试 encode_texts 空列表。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        vectors = encoder.encode_texts([])

        assert vectors == []

    def test_encode_single(
        self, mock_settings, mock_embedding
    ):
        """测试 encode_single 方法。"""
        mock_embedding.embed_single = Mock(return_value=[0.5] * 1536)

        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        vector = encoder.encode_single("test text")

        assert len(vector) == 1536

    def test_get_dimensions(
        self, mock_settings, mock_embedding
    ):
        """测试获取向量维度。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)

        assert encoder.get_dimensions() == 1536

    def test_get_model_name(
        self, mock_settings, mock_embedding
    ):
        """测试获取模型名称。"""
        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)

        assert encoder.get_model_name() == "text-embedding-ada-002"

    def test_creates_embedding_if_not_provided(
        self, mock_settings
    ):
        """测试未提供 Embedding 时自动创建。"""
        with patch("src.ingestion.embedding.dense_encoder.EmbeddingFactory.create") as mock_create:
            mock_embedding = Mock()
            mock_embedding.get_dimensions = Mock(return_value=1536)
            mock_create.return_value = mock_embedding

            encoder = DenseEncoder(mock_settings)

            mock_create.assert_called_once_with(mock_settings)


# ============================================================================
# FakeDenseEncoder 测试
# ============================================================================


class TestFakeDenseEncoder:
    """FakeDenseEncoder 测试。"""

    def test_encode_returns_correct_count(self, sample_chunks):
        """测试编码返回正确数量的记录。"""
        encoder = FakeDenseEncoder(dimensions=768)
        records = encoder.encode(sample_chunks)

        assert len(records) == len(sample_chunks)

    def test_encode_returns_correct_dimensions(self, sample_chunks):
        """测试编码返回正确维度的向量。"""
        encoder = FakeDenseEncoder(dimensions=768)
        records = encoder.encode(sample_chunks)

        for record in records:
            assert record.dense_vector is not None
            assert len(record.dense_vector) == 768

    def test_encode_is_deterministic(self, sample_chunks):
        """测试编码是确定性的（相同输入产生相同输出）。"""
        encoder = FakeDenseEncoder(dimensions=768)

        records1 = encoder.encode(sample_chunks)
        records2 = encoder.encode(sample_chunks)

        for r1, r2 in zip(records1, records2):
            assert r1.dense_vector == r2.dense_vector

    def test_encode_different_ids_produce_different_vectors(self, sample_chunks):
        """测试不同 ID 产生不同向量。"""
        encoder = FakeDenseEncoder(dimensions=768)
        records = encoder.encode(sample_chunks)

        vectors = [r.dense_vector for r in records]

        # 每个向量应该不同
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                assert vectors[i] != vectors[j]

    def test_encode_preserves_chunk_info(self, sample_chunks):
        """测试编码保留 Chunk 信息。"""
        encoder = FakeDenseEncoder(dimensions=768)
        records = encoder.encode(sample_chunks)

        for i, record in enumerate(records):
            assert record.id == sample_chunks[i].id
            assert record.text == sample_chunks[i].text

    def test_encode_empty_chunks(self):
        """测试编码空 Chunk 列表。"""
        encoder = FakeDenseEncoder(dimensions=768)
        records = encoder.encode([])

        assert records == []

    def test_encode_with_trace(self, sample_chunks, trace_context):
        """测试编码记录追踪。"""
        encoder = FakeDenseEncoder(dimensions=768)
        records = encoder.encode(sample_chunks, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "dense_encoding" for s in stages)

    def test_encode_texts(self):
        """测试 encode_texts 方法。"""
        encoder = FakeDenseEncoder(dimensions=512)
        texts = ["a", "b", "c"]

        vectors = encoder.encode_texts(texts)

        assert len(vectors) == 3
        assert len(vectors[0]) == 512

    def test_encode_single(self):
        """测试 encode_single 方法。"""
        encoder = FakeDenseEncoder(dimensions=256)
        vector = encoder.encode_single("test")

        assert len(vector) == 256

    def test_get_dimensions(self):
        """测试获取向量维度。"""
        encoder = FakeDenseEncoder(dimensions=1024)

        assert encoder.get_dimensions() == 1024

    def test_get_model_name(self):
        """测试获取模型名称。"""
        encoder = FakeDenseEncoder(dimensions=768)

        assert encoder.get_model_name() == "fake-embedding"


# ============================================================================
# 边界条件测试
# ============================================================================


class TestEdgeCases:
    """边界条件测试。"""

    def test_large_batch(
        self, mock_settings, mock_embedding
    ):
        """测试大批量编码。"""
        # 创建大量 chunk
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Text {i}", metadata={})
            for i in range(100)
        ]

        # Mock 返回 100 个向量
        mock_embedding.embed = Mock(return_value=EmbeddingResult(
            vectors=[[0.1] * 1536 for _ in range(100)],
            model="text-embedding-ada-002",
            dimensions=1536,
        ))

        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        records = encoder.encode(chunks)

        assert len(records) == 100

    def test_chunk_with_empty_text(
        self, mock_settings, mock_embedding
    ):
        """测试空文本 Chunk。"""
        chunks = [Chunk(id="empty", text="", metadata={})]
        mock_embedding.embed = Mock(return_value=EmbeddingResult(
            vectors=[[0.0] * 1536],
            model="text-embedding-ada-002",
            dimensions=1536,
        ))

        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        records = encoder.encode(chunks)

        assert len(records) == 1
        assert records[0].dense_vector is not None

    def test_chunk_with_special_characters(
        self, mock_settings, mock_embedding
    ):
        """测试特殊字符文本。"""
        chunks = [
            Chunk(
                id="special",
                text="特殊字符 🎉 <script>alert('xss')</script>",
                metadata={},
            )
        ]
        mock_embedding.embed = Mock(return_value=EmbeddingResult(
            vectors=[[0.1] * 1536],
            model="text-embedding-ada-002",
            dimensions=1536,
        ))

        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        records = encoder.encode(chunks)

        assert len(records) == 1

    def test_chunk_with_existing_metadata(
        self, mock_settings, mock_embedding
    ):
        """测试 Chunk 已有元数据。"""
        chunks = [
            Chunk(
                id="meta",
                text="Text",
                metadata={"title": "Test", "tags": ["a", "b"]},
            )
        ]
        mock_embedding.embed = Mock(return_value=EmbeddingResult(
            vectors=[[0.1] * 1536],
            model="text-embedding-ada-002",
            dimensions=1536,
        ))

        encoder = DenseEncoder(mock_settings, embedding=mock_embedding)
        records = encoder.encode(chunks)

        assert records[0].metadata["title"] == "Test"
        assert records[0].metadata["tags"] == ["a", "b"]
