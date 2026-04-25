"""
VectorUpserter 单元测试。

测试向量存储与幂等性保证。
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.core.types import ChunkRecord
from src.core.trace.trace_context import TraceContext
from src.ingestion.storage.vector_upserter import VectorUpserter, FakeVectorUpserter
from src.libs.vector_store import BaseVectorStore, UpsertResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_records():
    """创建测试用 ChunkRecord 列表。"""
    return [
        ChunkRecord(
            id="chunk_001",
            text="This is the first chunk.",
            metadata={"source_path": "test.pdf", "chunk_index": 0},
            dense_vector=[0.1] * 128,
        ),
        ChunkRecord(
            id="chunk_002",
            text="This is the second chunk.",
            metadata={"source_path": "test.pdf", "chunk_index": 1},
            dense_vector=[0.2] * 128,
        ),
        ChunkRecord(
            id="chunk_003",
            text="This is the third chunk.",
            metadata={"source_path": "test.pdf", "chunk_index": 2},
            dense_vector=[0.3] * 128,
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """创建 Mock VectorStore。"""
    store = Mock(spec=BaseVectorStore)
    store.upsert = Mock(return_value=UpsertResult(
        success=True,
        upserted_count=3,
        ids=["chunk_001", "chunk_002", "chunk_003"],
    ))
    return store


@pytest.fixture
def trace_context():
    """创建 TraceContext。"""
    return TraceContext(trace_type="ingestion")


# ============================================================================
# VectorUpserter 测试
# ============================================================================


class TestVectorUpserter:
    """VectorUpserter 测试。"""

    def test_upsert_returns_result(
        self, mock_vector_store, sample_records
    ):
        """测试写入返回结果。"""
        upserter = VectorUpserter(mock_vector_store)
        result = upserter.upsert(sample_records)

        assert result.success is True
        assert result.upserted_count == 3

    def test_upsert_calls_vector_store(
        self, mock_vector_store, sample_records
    ):
        """测试写入调用 VectorStore。"""
        upserter = VectorUpserter(mock_vector_store)
        upserter.upsert(sample_records)

        mock_vector_store.upsert.assert_called_once()
        call_args = mock_vector_store.upsert.call_args
        vector_records = call_args[0][0]

        assert len(vector_records) == 3

    def test_upsert_empty_records(self, mock_vector_store):
        """测试写入空记录列表。"""
        upserter = VectorUpserter(mock_vector_store)
        result = upserter.upsert([])

        assert result.success is True
        assert result.upserted_count == 0

    def test_upsert_skips_records_without_vector(
        self, mock_vector_store
    ):
        """测试跳过没有向量的记录。"""
        records = [
            ChunkRecord(id="no_vector", text="No vector", metadata={}),
            ChunkRecord(id="has_vector", text="Has vector", metadata={}, dense_vector=[0.1] * 128),
        ]

        upserter = VectorUpserter(mock_vector_store)
        upserter.upsert(records)

        call_args = mock_vector_store.upsert.call_args
        vector_records = call_args[0][0]

        assert len(vector_records) == 1
        assert vector_records[0].id == "has_vector"

    def test_upsert_with_collection(
        self, mock_vector_store, sample_records
    ):
        """测试写入带集合名称。"""
        upserter = VectorUpserter(mock_vector_store, collection="my_docs")
        upserter.upsert(sample_records)

        call_args = mock_vector_store.upsert.call_args
        vector_records = call_args[0][0]

        # 检查 metadata 中包含 collection
        for record in vector_records:
            assert record.metadata.get("collection") == "my_docs"

    def test_upsert_preserves_metadata(
        self, mock_vector_store, sample_records
    ):
        """测试写入保留元数据。"""
        upserter = VectorUpserter(mock_vector_store)
        upserter.upsert(sample_records)

        call_args = mock_vector_store.upsert.call_args
        vector_records = call_args[0][0]

        for i, record in enumerate(vector_records):
            assert record.metadata["source_path"] == sample_records[i].metadata["source_path"]
            assert record.metadata["chunk_index"] == sample_records[i].metadata["chunk_index"]

    def test_upsert_with_trace(
        self, mock_vector_store, sample_records, trace_context
    ):
        """测试写入记录追踪。"""
        upserter = VectorUpserter(mock_vector_store)
        upserter.upsert(sample_records, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "vector_upsert" for s in stages)

    def test_upsert_trace_includes_details(
        self, mock_vector_store, sample_records, trace_context
    ):
        """测试追踪包含详细信息。"""
        upserter = VectorUpserter(mock_vector_store)
        upserter.upsert(sample_records, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        stage = next((s for s in stages if s["stage"] == "vector_upsert"), None)

        assert stage is not None
        assert "record_count" in stage.get("details", {})
        assert "upserted_count" in stage.get("details", {})

    def test_upsert_single_success(
        self, mock_vector_store
    ):
        """测试写入单个记录成功。"""
        record = ChunkRecord(
            id="single",
            text="Single record",
            metadata={},
            dense_vector=[0.1] * 128,
        )

        # 为单个记录设置正确的 mock 返回值
        mock_vector_store.upsert = Mock(return_value=UpsertResult(
            success=True,
            upserted_count=1,
            ids=["single"],
        ))

        upserter = VectorUpserter(mock_vector_store)
        result = upserter.upsert_single(record)

        assert result is True

    def test_upsert_single_no_vector(
        self, mock_vector_store
    ):
        """测试写入单个记录无向量。"""
        record = ChunkRecord(
            id="no_vector",
            text="No vector",
            metadata={},
        )

        upserter = VectorUpserter(mock_vector_store)
        result = upserter.upsert_single(record)

        assert result is False

    def test_get_vector_store(self, mock_vector_store):
        """测试获取 VectorStore。"""
        upserter = VectorUpserter(mock_vector_store)

        assert upserter.get_vector_store() == mock_vector_store

    def test_get_collection(self, mock_vector_store):
        """测试获取集合名称。"""
        upserter = VectorUpserter(mock_vector_store, collection="test_collection")

        assert upserter.get_collection() == "test_collection"


# ============================================================================
# FakeVectorUpserter 测试
# ============================================================================


class TestFakeVectorUpserter:
    """FakeVectorUpserter 测试。"""

    def test_upsert_records(self, sample_records):
        """测试写入记录。"""
        upserter = FakeVectorUpserter()
        result = upserter.upsert(sample_records)

        assert result.success is True
        assert result.upserted_count == 3
        assert upserter.get_upserted_count() == 3

    def test_upsert_is_idempotent(self, sample_records):
        """测试写入幂等性。"""
        upserter = FakeVectorUpserter()

        # 第一次写入
        upserter.upsert(sample_records)
        count1 = upserter.get_upserted_count()

        # 第二次写入相同记录
        upserter.upsert(sample_records)
        count2 = upserter.get_upserted_count()

        # 数量应该相同（幂等）
        assert count1 == count2 == 3

    def test_upsert_updates_existing_record(self, sample_records):
        """测试更新已存在的记录。"""
        upserter = FakeVectorUpserter()

        # 第一次写入
        upserter.upsert(sample_records)

        # 修改记录
        modified_records = [
            ChunkRecord(
                id="chunk_001",
                text="Modified text",
                metadata={"source_path": "test.pdf"},
                dense_vector=[0.5] * 128,  # 不同的向量
            ),
        ]

        # 第二次写入
        upserter.upsert(modified_records)

        # 检查记录被更新
        records = upserter.get_upserted_records()
        chunk_001 = next(r for r in records if r.id == "chunk_001")
        assert chunk_001.text == "Modified text"
        assert chunk_001.dense_vector == [0.5] * 128

    def test_upsert_empty_records(self):
        """测试写入空记录列表。"""
        upserter = FakeVectorUpserter()
        result = upserter.upsert([])

        assert result.success is True
        assert result.upserted_count == 0

    def test_upsert_skips_records_without_vector(self):
        """测试跳过没有向量的记录。"""
        records = [
            ChunkRecord(id="no_vector", text="No vector", metadata={}),
            ChunkRecord(id="has_vector", text="Has vector", metadata={}, dense_vector=[0.1] * 128),
        ]

        upserter = FakeVectorUpserter()
        result = upserter.upsert(records)

        # 只有一条记录被写入
        assert result.upserted_count == 1
        assert "has_vector" in result.ids

    def test_upsert_with_trace(self, sample_records, trace_context):
        """测试写入记录追踪。"""
        upserter = FakeVectorUpserter()
        upserter.upsert(sample_records, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "vector_upsert" for s in stages)

    def test_upsert_single_success(self):
        """测试写入单个记录成功。"""
        record = ChunkRecord(
            id="single",
            text="Single record",
            metadata={},
            dense_vector=[0.1] * 128,
        )

        upserter = FakeVectorUpserter()
        result = upserter.upsert_single(record)

        assert result is True
        assert upserter.get_upserted_count() == 1

    def test_upsert_single_no_vector(self):
        """测试写入单个记录无向量。"""
        record = ChunkRecord(
            id="no_vector",
            text="No vector",
            metadata={},
        )

        upserter = FakeVectorUpserter()
        result = upserter.upsert_single(record)

        assert result is False

    def test_clear(self, sample_records):
        """测试清空记录。"""
        upserter = FakeVectorUpserter()
        upserter.upsert(sample_records)

        upserter.clear()

        assert upserter.get_upserted_count() == 0

    def test_get_collection(self):
        """测试获取集合名称。"""
        upserter = FakeVectorUpserter(collection="test_collection")

        assert upserter.get_collection() == "test_collection"


# ============================================================================
# 幂等性测试
# ============================================================================


class TestIdempotency:
    """幂等性测试。"""

    def test_same_content_same_id(self):
        """测试相同内容产生相同 ID。"""
        upserter = FakeVectorUpserter()

        records = [
            ChunkRecord(
                id="chunk_001",
                text="Same content",
                metadata={"source": "test.pdf"},
                dense_vector=[0.1] * 128,
            ),
        ]

        # 多次写入
        upserter.upsert(records)
        upserter.upsert(records)
        upserter.upsert(records)

        # 应该只有一条记录
        assert upserter.get_upserted_count() == 1

    def test_different_content_different_id(self):
        """测试不同内容产生不同 ID。"""
        upserter = FakeVectorUpserter()

        records1 = [
            ChunkRecord(
                id="chunk_001",
                text="Content 1",
                metadata={"source": "test.pdf"},
                dense_vector=[0.1] * 128,
            ),
        ]

        records2 = [
            ChunkRecord(
                id="chunk_002",
                text="Content 2",
                metadata={"source": "test.pdf"},
                dense_vector=[0.2] * 128,
            ),
        ]

        upserter.upsert(records1)
        upserter.upsert(records2)

        # 应该有两条记录
        assert upserter.get_upserted_count() == 2

    def test_batch_upsert_maintains_order(self):
        """测试批量写入保持顺序。"""
        upserter = FakeVectorUpserter()

        records = [
            ChunkRecord(id=f"chunk_{i:03d}", text=f"Text {i}", metadata={}, dense_vector=[float(i)] * 128)
            for i in range(10)
        ]

        upserter.upsert(records)

        stored_records = upserter.get_upserted_records()
        for i, record in enumerate(stored_records):
            assert record.id == f"chunk_{i:03d}"


# ============================================================================
# 边界条件测试
# ============================================================================


class TestEdgeCases:
    """边界条件测试。"""

    def test_large_batch(self):
        """测试大批量写入。"""
        upserter = FakeVectorUpserter()

        records = [
            ChunkRecord(
                id=f"chunk_{i}",
                text=f"Text {i}",
                metadata={},
                dense_vector=[0.1] * 128,
            )
            for i in range(100)
        ]

        result = upserter.upsert(records)

        assert result.upserted_count == 100

    def test_record_with_empty_metadata(self):
        """测试空元数据记录。"""
        upserter = FakeVectorUpserter()

        record = ChunkRecord(
            id="empty_meta",
            text="Text",
            metadata={},
            dense_vector=[0.1] * 128,
        )

        result = upserter.upsert([record])

        assert result.success is True

    def test_record_with_complex_metadata(self):
        """测试复杂元数据记录。"""
        upserter = FakeVectorUpserter()

        record = ChunkRecord(
            id="complex_meta",
            text="Text",
            metadata={
                "title": "Test Document",
                "tags": ["tag1", "tag2"],
                "nested": {"key": "value"},
            },
            dense_vector=[0.1] * 128,
        )

        result = upserter.upsert([record])

        assert result.success is True
        stored = upserter.get_upserted_records()[0]
        assert stored.metadata["title"] == "Test Document"
        assert stored.metadata["tags"] == ["tag1", "tag2"]
