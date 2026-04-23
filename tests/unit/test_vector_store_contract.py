"""
VectorStore Factory 单元测试。

验证工厂路由逻辑和契约约束。
"""

import pytest

from src.core.settings import Settings, VectorStoreSettings
from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    UpsertResult,
    VectorStoreError,
)
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    """Fake VectorStore 实现，用于测试。"""

    def __init__(self, settings):
        if hasattr(settings, "vector_store"):
            settings = settings.vector_store
        self.collection_name = getattr(settings, "collection_name", "test")

    def upsert(self, records, trace=None):
        return UpsertResult(
            success=True,
            upserted_count=len(records),
            ids=[r.id for r in records],
        )

    def query(self, vector, top_k=10, filters=None, trace=None):
        return [
            QueryResult(id=f"result_{i}", score=0.9 - i * 0.1, text=f"text_{i}", metadata={})
            for i in range(min(top_k, 3))
        ]

    def get_by_ids(self, ids):
        return [{"id": id_, "text": f"text_{id_}", "metadata": {}} for id_ in ids]

    def delete_by_ids(self, ids):
        return len(ids)

    def delete_by_metadata(self, filters):
        return 0

    def get_collection_stats(self):
        return {"count": 100}

    def get_backend_name(self) -> str:
        return "fake"


class TestVectorStoreFactory:
    """VectorStoreFactory 测试。"""

    def test_get_supported_providers(self):
        """应该返回支持的 provider 列表。"""
        providers = VectorStoreFactory.get_supported_providers()

        assert "chroma" in providers
        assert "qdrant" in providers
        assert "pinecone" in providers

    def test_unsupported_provider_raises_error(self):
        """不支持的 provider 应该抛出错误。"""
        settings = Settings(
            vector_store=VectorStoreSettings(
                provider="unsupported_provider",
                collection_name="test",
            )
        )

        with pytest.raises(VectorStoreError) as exc_info:
            VectorStoreFactory.create(settings)

        assert "Unsupported VectorStore provider" in str(exc_info.value)
        assert exc_info.value.backend == "unsupported_provider"

    def test_create_chroma_store(self):
        """应该能创建 ChromaStore。"""
        settings = Settings(
            vector_store=VectorStoreSettings(
                provider="chroma",
                persist_directory="./data/db/chroma",
                collection_name="test_collection",
            )
        )

        store = VectorStoreFactory.create(settings)

        assert store is not None
        assert store.get_backend_name() == "chroma"

    def test_create_chroma_store_case_insensitive(self):
        """Provider 名称应该不区分大小写。"""
        settings = Settings(
            vector_store=VectorStoreSettings(
                provider="Chroma",  # 大小写混合
                collection_name="test",
            )
        )

        store = VectorStoreFactory.create(settings)

        assert store is not None
        assert store.get_backend_name() == "chroma"

    def test_create_qdrant_store(self):
        """应该能创建 QdrantStore。"""
        settings = Settings(
            vector_store=VectorStoreSettings(
                provider="qdrant",
                collection_name="test",
            )
        )

        store = VectorStoreFactory.create(settings)

        assert store is not None
        assert store.get_backend_name() == "qdrant"

    def test_create_pinecone_store(self):
        """应该能创建 PineconeStore。"""
        settings = Settings(
            vector_store=VectorStoreSettings(
                provider="pinecone",
                collection_name="test",
            )
        )

        store = VectorStoreFactory.create(settings)

        assert store is not None
        assert store.get_backend_name() == "pinecone"

    def test_create_from_vector_store_settings(self):
        """应该能从 VectorStoreSettings 创建 VectorStore。"""
        vs_settings = VectorStoreSettings(
            provider="chroma",
            collection_name="test",
        )

        store = VectorStoreFactory.create_from_settings(vs_settings)

        assert store is not None


class TestVectorRecord:
    """VectorRecord 测试。"""

    def test_vector_record_to_dict(self):
        """VectorRecord 应该能转换为字典。"""
        record = VectorRecord(
            id="test_id",
            vector=[0.1, 0.2, 0.3],
            text="test text",
            metadata={"source": "test.pdf"},
        )

        data = record.to_dict()

        assert data["id"] == "test_id"
        assert data["text"] == "test text"
        assert data["metadata"]["source"] == "test.pdf"
        # vector 不应该在 to_dict 中暴露
        assert "vector" not in data


class TestQueryResult:
    """QueryResult 测试。"""

    def test_query_result_to_dict(self):
        """QueryResult 应该能转换为字典。"""
        result = QueryResult(
            id="result_1",
            score=0.95,
            text="matched text",
            metadata={"page": 5},
        )

        data = result.to_dict()

        assert data["id"] == "result_1"
        assert data["score"] == 0.95
        assert data["text"] == "matched text"
        assert data["metadata"]["page"] == 5


class TestUpsertResult:
    """UpsertResult 测试。"""

    def test_upsert_result_success(self):
        """UpsertResult 应该正确表示成功。"""
        result = UpsertResult(
            success=True,
            upserted_count=10,
            ids=["id1", "id2", "id3"],
        )

        assert result.success is True
        assert result.upserted_count == 10
        assert len(result.ids) == 3

    def test_upsert_result_with_message(self):
        """UpsertResult 应该能包含消息。"""
        result = UpsertResult(
            success=False,
            upserted_count=0,
            message="Connection failed",
        )

        assert result.success is False
        assert result.message == "Connection failed"


class TestVectorStoreError:
    """VectorStoreError 测试。"""

    def test_vector_store_error_basic(self):
        """基本错误信息。"""
        error = VectorStoreError("Something went wrong")

        assert "Something went wrong" in str(error)

    def test_vector_store_error_with_backend(self):
        """带后端的错误信息。"""
        error = VectorStoreError("Something went wrong", backend="chroma")

        assert "[chroma]" in str(error)
        assert "Something went wrong" in str(error)

    def test_vector_store_error_subclasses(self):
        """错误子类应该继承 VectorStoreError。"""
        from src.libs.vector_store.base_vector_store import (
            VectorStoreConnectionError,
            VectorStoreConfigError,
            VectorStoreQueryError,
            VectorStoreUpsertError,
        )

        conn_error = VectorStoreConnectionError("Connection failed", backend="qdrant")
        config_error = VectorStoreConfigError("Invalid config", backend="chroma")
        query_error = VectorStoreQueryError("Query failed", backend="pinecone")
        upsert_error = VectorStoreUpsertError("Upsert failed", backend="chroma")

        assert isinstance(conn_error, VectorStoreError)
        assert isinstance(config_error, VectorStoreError)
        assert isinstance(query_error, VectorStoreError)
        assert isinstance(upsert_error, VectorStoreError)


class TestBaseVectorStoreContract:
    """BaseVectorStore 契约测试。"""

    def test_upsert_empty_records(self):
        """upsert 空列表应该返回成功。"""
        fake_store = FakeVectorStore(VectorStoreSettings(provider="fake", collection_name="test"))

        result = fake_store.upsert([])

        assert result.success is True
        assert result.upserted_count == 0

    def test_query_returns_correct_count(self):
        """query 应该返回正确数量的结果。"""
        fake_store = FakeVectorStore(VectorStoreSettings(provider="fake", collection_name="test"))

        results = fake_store.query([0.1, 0.2], top_k=2)

        assert len(results) == 2

    def test_get_by_ids_returns_records(self):
        """get_by_ids 应该返回记录。"""
        fake_store = FakeVectorStore(VectorStoreSettings(provider="fake", collection_name="test"))

        records = fake_store.get_by_ids(["id1", "id2"])

        assert len(records) == 2
        assert records[0]["id"] == "id1"

    def test_delete_by_ids_returns_count(self):
        """delete_by_ids 应该返回删除数量。"""
        fake_store = FakeVectorStore(VectorStoreSettings(provider="fake", collection_name="test"))

        count = fake_store.delete_by_ids(["id1", "id2", "id3"])

        assert count == 3
