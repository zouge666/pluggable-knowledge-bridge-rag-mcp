"""
Embedding Factory 单元测试。

使用 Fake provider 验证工厂路由逻辑。
"""

import pytest

from src.core.settings import EmbeddingSettings, Settings
from src.libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResult,
    EmbeddingError,
)
from src.libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """Fake Embedding 实现，用于测试。"""

    def __init__(self, settings, dimensions: int = 768):
        if hasattr(settings, "embedding"):
            settings = settings.embedding
        self.model = settings.model
        self._dimensions = dimensions

    def embed(self, texts, trace=None):
        # 返回固定维度的零向量
        vectors = [[0.0] * self._dimensions for _ in texts]
        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            dimensions=self._dimensions,
        )

    def get_model_name(self) -> str:
        return self.model

    def get_dimensions(self) -> int:
        return self._dimensions


class TestEmbeddingFactory:
    """EmbeddingFactory 测试。"""

    def test_get_supported_providers(self):
        """应该返回支持的 provider 列表。"""
        providers = EmbeddingFactory.get_supported_providers()

        assert "openai" in providers
        assert "azure" in providers
        assert "ollama" in providers

    def test_unsupported_provider_raises_error(self):
        """不支持的 provider 应该抛出错误。"""
        settings = Settings(
            embedding=EmbeddingSettings(
                provider="unsupported_provider",
                model="test-model",
            )
        )

        with pytest.raises(EmbeddingError) as exc_info:
            EmbeddingFactory.create(settings)

        assert "Unsupported Embedding provider" in str(exc_info.value)
        assert exc_info.value.provider == "unsupported_provider"

    def test_create_openai_embedding(self):
        """应该能创建 OpenAI Embedding。"""
        settings = Settings(
            embedding=EmbeddingSettings(
                provider="openai",
                model="text-embedding-3-small",
                api_key="test-key",
                dimensions=1536,
            )
        )

        embedding = EmbeddingFactory.create(settings)

        assert embedding is not None
        assert embedding.get_model_name() == "text-embedding-3-small"
        assert embedding.get_dimensions() == 1536

    def test_create_openai_embedding_case_insensitive(self):
        """Provider 名称应该不区分大小写。"""
        settings = Settings(
            embedding=EmbeddingSettings(
                provider="OpenAI",  # 大小写混合
                model="text-embedding-3-small",
                api_key="test-key",
            )
        )

        embedding = EmbeddingFactory.create(settings)

        assert embedding is not None

    def test_create_azure_embedding(self):
        """应该能创建 Azure Embedding。"""
        settings = Settings(
            embedding=EmbeddingSettings(
                provider="azure",
                model="text-embedding-ada-002",
                deployment_name="my-embedding",
                azure_endpoint="https://my-resource.openai.azure.com",
                api_key="test-key",
            )
        )

        embedding = EmbeddingFactory.create(settings)

        assert embedding is not None
        assert embedding.get_model_name() == "text-embedding-ada-002"

    def test_create_ollama_embedding(self):
        """应该能创建 Ollama Embedding。"""
        settings = Settings(
            embedding=EmbeddingSettings(
                provider="ollama",
                model="nomic-embed-text",
            )
        )

        embedding = EmbeddingFactory.create(settings)

        assert embedding is not None
        assert embedding.get_model_name() == "nomic-embed-text"

    def test_create_from_embedding_settings(self):
        """应该能从 EmbeddingSettings 创建 Embedding。"""
        embedding_settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )

        embedding = EmbeddingFactory.create_from_settings(embedding_settings)

        assert embedding is not None
        assert embedding.get_model_name() == "text-embedding-3-small"


class TestBaseEmbedding:
    """BaseEmbedding 测试。"""

    def test_embedding_result_to_dict(self):
        """EmbeddingResult 应该能转换为字典。"""
        result = EmbeddingResult(
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            model="text-embedding-3-small",
            dimensions=2,
            usage={"total_tokens": 10},
        )

        data = result.to_dict()

        assert data["model"] == "text-embedding-3-small"
        assert data["dimensions"] == 2
        assert data["count"] == 2
        assert data["usage"]["total_tokens"] == 10

    def test_embedding_result_to_dict_minimal(self):
        """EmbeddingResult 最小字段应该能转换。"""
        result = EmbeddingResult(
            vectors=[],
            model="test-model",
            dimensions=768,
        )

        data = result.to_dict()

        assert data["model"] == "test-model"
        assert data["dimensions"] == 768
        assert data["count"] == 0
        assert "usage" not in data

    def test_embed_single(self):
        """embed_single 便捷方法应该工作。"""
        fake_embedding = FakeEmbedding(
            EmbeddingSettings(provider="fake", model="fake-model"),
            dimensions=128,
        )

        vector = fake_embedding.embed_single("test text")

        assert len(vector) == 128
        assert all(v == 0.0 for v in vector)

    def test_embed_empty_list(self):
        """空列表应该返回空结果。"""
        fake_embedding = FakeEmbedding(
            EmbeddingSettings(provider="fake", model="fake-model"),
        )

        result = fake_embedding.embed([])

        assert result.vectors == []
        assert result.model == "fake-model"


class TestEmbeddingError:
    """EmbeddingError 测试。"""

    def test_embedding_error_basic(self):
        """基本错误信息。"""
        error = EmbeddingError("Something went wrong")

        assert "Something went wrong" in str(error)

    def test_embedding_error_with_provider(self):
        """带 provider 的错误信息。"""
        error = EmbeddingError("Something went wrong", provider="openai")

        assert "[openai]" in str(error)
        assert "Something went wrong" in str(error)

    def test_embedding_error_with_all_fields(self):
        """带所有字段的错误信息。"""
        error = EmbeddingError(
            "Something went wrong",
            provider="azure",
            model="text-embedding-ada-002",
        )

        error_str = str(error)
        assert "[azure]" in error_str
        assert "model=text-embedding-ada-002" in error_str
        assert "Something went wrong" in error_str

    def test_embedding_error_subclasses(self):
        """错误子类应该继承 EmbeddingError。"""
        from src.libs.embedding.base_embedding import (
            EmbeddingAuthenticationError,
            EmbeddingConnectionError,
            EmbeddingRateLimitError,
            EmbeddingResponseError,
        )

        auth_error = EmbeddingAuthenticationError("Auth failed", provider="openai")
        conn_error = EmbeddingConnectionError("Connection failed", provider="azure")
        rate_error = EmbeddingRateLimitError("Rate limited", provider="openai")
        resp_error = EmbeddingResponseError("Bad response", provider="ollama")

        assert isinstance(auth_error, EmbeddingError)
        assert isinstance(conn_error, EmbeddingError)
        assert isinstance(rate_error, EmbeddingError)
        assert isinstance(resp_error, EmbeddingError)
