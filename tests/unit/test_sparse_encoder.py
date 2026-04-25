"""
SparseEncoder 单元测试。

测试稀疏向量编码功能。
"""

import pytest
from unittest.mock import Mock

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.sparse_encoder import SparseEncoder, FakeSparseEncoder


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_chunks():
    """创建测试用 Chunk 列表。"""
    return [
        Chunk(
            id="chunk_001",
            text="The quick brown fox jumps over the lazy dog.",
            metadata={"source_path": "test.pdf", "chunk_index": 0},
        ),
        Chunk(
            id="chunk_002",
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"source_path": "test.pdf", "chunk_index": 1},
        ),
        Chunk(
            id="chunk_003",
            text="Python is a popular programming language.",
            metadata={"source_path": "test.pdf", "chunk_index": 2},
        ),
    ]


@pytest.fixture
def trace_context():
    """创建 TraceContext。"""
    return TraceContext(trace_type="ingestion")


# ============================================================================
# SparseEncoder 测试
# ============================================================================


class TestSparseEncoder:
    """SparseEncoder 测试。"""

    def test_encode_returns_correct_count(self, sample_chunks):
        """测试编码返回正确数量的记录。"""
        encoder = SparseEncoder()
        records = encoder.encode(sample_chunks)

        assert len(records) == len(sample_chunks)

    def test_encode_returns_term_weights(self, sample_chunks):
        """测试编码返回 term weights。"""
        encoder = SparseEncoder()
        records = encoder.encode(sample_chunks)

        for record in records:
            assert record.sparse_vector is not None
            assert isinstance(record.sparse_vector, dict)
            # 所有值应该是浮点数
            for term, weight in record.sparse_vector.items():
                assert isinstance(term, str)
                assert isinstance(weight, float)
                assert weight > 0

    def test_encode_preserves_chunk_info(self, sample_chunks):
        """测试编码保留 Chunk 信息。"""
        encoder = SparseEncoder()
        records = encoder.encode(sample_chunks)

        for i, record in enumerate(records):
            assert record.id == sample_chunks[i].id
            assert record.text == sample_chunks[i].text
            assert record.metadata == sample_chunks[i].metadata

    def test_encode_empty_chunks(self):
        """测试编码空 Chunk 列表。"""
        encoder = SparseEncoder()
        records = encoder.encode([])

        assert records == []

    def test_encode_empty_text(self):
        """测试编码空文本。"""
        encoder = SparseEncoder()
        chunks = [Chunk(id="empty", text="", metadata={})]
        records = encoder.encode(chunks)

        assert len(records) == 1
        assert records[0].sparse_vector == {}

    def test_encode_filters_stopwords(self):
        """测试过滤停用词。"""
        encoder = SparseEncoder()
        # 只有停用词的文本
        chunks = [Chunk(id="stopwords", text="the a an the", metadata={})]
        records = encoder.encode(chunks)

        assert records[0].sparse_vector == {}

    def test_encode_filters_short_terms(self):
        """测试过滤短词。"""
        encoder = SparseEncoder(min_term_length=3)
        chunks = [Chunk(id="short", text="a big cat", metadata={})]
        records = encoder.encode(chunks)

        # "a" 被停用词过滤，"big" 和 "cat" 长度 >= 3
        terms = records[0].sparse_vector.keys()
        assert "big" in terms
        assert "cat" in terms

    def test_encode_lowercase(self):
        """测试小写转换。"""
        encoder = SparseEncoder(lowercase=True)
        chunks = [Chunk(id="case", text="Hello WORLD", metadata={})]
        records = encoder.encode(chunks)

        terms = records[0].sparse_vector.keys()
        assert "hello" in terms
        assert "world" in terms

    def test_encode_no_lowercase(self):
        """测试不转换小写。"""
        encoder = SparseEncoder(lowercase=False)
        chunks = [Chunk(id="case", text="Hello WORLD", metadata={})]
        records = encoder.encode(chunks)

        terms = records[0].sparse_vector.keys()
        # 注意：停用词过滤时也是区分大小写的
        # "Hello" 和 "WORLD" 不会被停用词过滤（因为停用词是小写）
        assert "Hello" in terms
        assert "WORLD" in terms

    def test_encode_with_trace(self, sample_chunks, trace_context):
        """测试编码记录追踪。"""
        encoder = SparseEncoder()
        records = encoder.encode(sample_chunks, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "sparse_encoding" for s in stages)

    def test_encode_trace_includes_details(self, sample_chunks, trace_context):
        """测试追踪包含详细信息。"""
        encoder = SparseEncoder()
        records = encoder.encode(sample_chunks, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        stage = next((s for s in stages if s["stage"] == "sparse_encoding"), None)

        assert stage is not None
        assert "chunk_count" in stage.get("details", {})
        assert "stopwords_count" in stage.get("details", {})

    def test_encode_texts(self):
        """测试 encode_texts 方法。"""
        encoder = SparseEncoder()
        texts = ["hello world", "machine learning"]

        weights_list = encoder.encode_texts(texts)

        assert len(weights_list) == 2
        assert isinstance(weights_list[0], dict)
        assert isinstance(weights_list[1], dict)

    def test_encode_texts_empty(self):
        """测试 encode_texts 空列表。"""
        encoder = SparseEncoder()
        weights_list = encoder.encode_texts([])

        assert weights_list == []

    def test_encode_single(self):
        """测试 encode_single 方法。"""
        encoder = SparseEncoder()
        weights = encoder.encode_single("hello world")

        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_encode_single_empty(self):
        """测试 encode_single 空文本。"""
        encoder = SparseEncoder()
        weights = encoder.encode_single("")

        assert weights == {}

    def test_get_stopwords(self):
        """测试获取停用词集合。"""
        encoder = SparseEncoder()
        stopwords = encoder.get_stopwords()

        assert isinstance(stopwords, set)
        assert "the" in stopwords
        assert "a" in stopwords

    def test_get_model_name(self):
        """测试获取模型名称。"""
        encoder = SparseEncoder()

        assert encoder.get_model_name() == "bm25"

    def test_custom_stopwords(self):
        """测试自定义停用词。"""
        custom_stopwords = {"hello", "world"}
        encoder = SparseEncoder(stopwords=custom_stopwords)

        chunks = [Chunk(id="custom", text="hello world test", metadata={})]
        records = encoder.encode(chunks)

        terms = records[0].sparse_vector.keys()
        assert "hello" not in terms
        assert "world" not in terms
        assert "test" in terms

    def test_chinese_text(self):
        """测试中文文本。"""
        # 中文按字符分割，需要设置 min_term_length=1
        encoder = SparseEncoder(min_term_length=1)
        chunks = [Chunk(id="chinese", text="机器学习是人工智能的分支", metadata={})]
        records = encoder.encode(chunks)

        # 应该能提取中文词
        assert len(records[0].sparse_vector) > 0

    def test_mixed_text(self):
        """测试中英文混合文本。"""
        # 中文按字符分割，需要设置 min_term_length=1
        encoder = SparseEncoder(min_term_length=1)
        chunks = [Chunk(id="mixed", text="Python 是一门流行的编程语言", metadata={})]
        records = encoder.encode(chunks)

        terms = records[0].sparse_vector.keys()
        # 应该包含英文词
        assert "python" in terms
        # 应该包含中文词
        assert len(terms) > 1


# ============================================================================
# FakeSparseEncoder 测试
# ============================================================================


class TestFakeSparseEncoder:
    """FakeSparseEncoder 测试。"""

    def test_encode_returns_correct_count(self, sample_chunks):
        """测试编码返回正确数量的记录。"""
        encoder = FakeSparseEncoder()
        records = encoder.encode(sample_chunks)

        assert len(records) == len(sample_chunks)

    def test_encode_returns_term_weights(self, sample_chunks):
        """测试编码返回 term weights。"""
        encoder = FakeSparseEncoder()
        records = encoder.encode(sample_chunks)

        for record in records:
            assert record.sparse_vector is not None
            assert isinstance(record.sparse_vector, dict)

    def test_encode_is_deterministic(self, sample_chunks):
        """测试编码是确定性的（相同输入产生相同输出）。"""
        encoder = FakeSparseEncoder()

        records1 = encoder.encode(sample_chunks)
        records2 = encoder.encode(sample_chunks)

        for r1, r2 in zip(records1, records2):
            assert r1.sparse_vector == r2.sparse_vector

    def test_encode_different_ids_produce_different_weights(self, sample_chunks):
        """测试不同 ID 产生不同 term weights。"""
        encoder = FakeSparseEncoder()
        records = encoder.encode(sample_chunks)

        weights = [r.sparse_vector for r in records]

        # 每个 term weights 应该不同
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                assert weights[i] != weights[j]

    def test_encode_preserves_chunk_info(self, sample_chunks):
        """测试编码保留 Chunk 信息。"""
        encoder = FakeSparseEncoder()
        records = encoder.encode(sample_chunks)

        for i, record in enumerate(records):
            assert record.id == sample_chunks[i].id
            assert record.text == sample_chunks[i].text

    def test_encode_empty_chunks(self):
        """测试编码空 Chunk 列表。"""
        encoder = FakeSparseEncoder()
        records = encoder.encode([])

        assert records == []

    def test_encode_with_trace(self, sample_chunks, trace_context):
        """测试编码记录追踪。"""
        encoder = FakeSparseEncoder()
        records = encoder.encode(sample_chunks, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "sparse_encoding" for s in stages)

    def test_encode_texts(self):
        """测试 encode_texts 方法。"""
        encoder = FakeSparseEncoder()
        texts = ["a", "b", "c"]

        weights_list = encoder.encode_texts(texts)

        assert len(weights_list) == 3
        assert all(isinstance(w, dict) for w in weights_list)

    def test_encode_single(self):
        """测试 encode_single 方法。"""
        encoder = FakeSparseEncoder()
        weights = encoder.encode_single("test")

        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_get_model_name(self):
        """测试获取模型名称。"""
        encoder = FakeSparseEncoder()

        assert encoder.get_model_name() == "fake-sparse"


# ============================================================================
# 边界条件测试
# ============================================================================


class TestEdgeCases:
    """边界条件测试。"""

    def test_large_batch(self):
        """测试大批量编码。"""
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Text {i}", metadata={})
            for i in range(100)
        ]

        encoder = SparseEncoder()
        records = encoder.encode(chunks)

        assert len(records) == 100

    def test_chunk_with_special_characters(self):
        """测试特殊字符文本。"""
        chunks = [
            Chunk(
                id="special",
                text="Hello! @World# $Test% &Data*",
                metadata={},
            )
        ]

        encoder = SparseEncoder()
        records = encoder.encode(chunks)

        assert len(records) == 1
        assert records[0].sparse_vector is not None

    def test_chunk_with_numbers(self):
        """测试包含数字的文本。"""
        chunks = [
            Chunk(
                id="numbers",
                text="Python 3.11 was released in 2022",
                metadata={},
            )
        ]

        encoder = SparseEncoder()
        records = encoder.encode(chunks)

        terms = records[0].sparse_vector.keys()
        # 数字应该被保留
        assert "3" in terms or "11" in terms or "2022" in terms

    def test_chunk_with_repeated_terms(self):
        """测试重复词。"""
        chunks = [
            Chunk(
                id="repeated",
                text="test test test hello hello",
                metadata={},
            )
        ]

        encoder = SparseEncoder()
        records = encoder.encode(chunks)

        weights = records[0].sparse_vector
        # test 出现 3 次，hello 出现 2 次
        assert weights.get("test") == 3.0
        assert weights.get("hello") == 2.0

    def test_chunk_with_existing_metadata(self):
        """测试 Chunk 已有元数据。"""
        chunks = [
            Chunk(
                id="meta",
                text="Test text",
                metadata={"title": "Test", "tags": ["a", "b"]},
            )
        ]

        encoder = SparseEncoder()
        records = encoder.encode(chunks)

        assert records[0].metadata["title"] == "Test"
        assert records[0].metadata["tags"] == ["a", "b"]
