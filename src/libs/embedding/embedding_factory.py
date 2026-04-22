"""
Embedding 工厂。

根据配置创建对应的 Embedding 实例。
"""

from typing import TYPE_CHECKING

from src.core.settings import EmbeddingSettings, Settings
from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingError

if TYPE_CHECKING:
    pass


class EmbeddingFactory:
    """
    Embedding 工厂类。

    根据配置动态创建 Embedding 实例，支持多种 provider。
    """

    @classmethod
    def create(cls, settings: Settings) -> BaseEmbedding:
        """
        根据配置创建 Embedding 实例。

        Args:
            settings: 应用配置。

        Returns:
            BaseEmbedding: Embedding 实例。

        Raises:
            EmbeddingError: 不支持的 provider 或配置错误。
        """
        embedding_settings = settings.embedding
        provider = embedding_settings.provider.lower()

        if provider == "azure":
            from src.libs.embedding.azure_embedding import AzureEmbedding

            return AzureEmbedding(embedding_settings)
        elif provider == "openai":
            from src.libs.embedding.openai_embedding import OpenAIEmbedding

            return OpenAIEmbedding(embedding_settings)
        elif provider == "ollama":
            from src.libs.embedding.ollama_embedding import OllamaEmbedding

            return OllamaEmbedding(embedding_settings)
        else:
            raise EmbeddingError(
                f"Unsupported Embedding provider: {provider}",
                provider=provider,
            )

    @classmethod
    def create_from_settings(cls, embedding_settings: EmbeddingSettings) -> BaseEmbedding:
        """
        从 EmbeddingSettings 创建 Embedding 实例。

        Args:
            embedding_settings: Embedding 配置。

        Returns:
            BaseEmbedding: Embedding 实例。
        """
        provider = embedding_settings.provider.lower()

        if provider == "azure":
            from src.libs.embedding.azure_embedding import AzureEmbedding

            return AzureEmbedding(embedding_settings)
        elif provider == "openai":
            from src.libs.embedding.openai_embedding import OpenAIEmbedding

            return OpenAIEmbedding(embedding_settings)
        elif provider == "ollama":
            from src.libs.embedding.ollama_embedding import OllamaEmbedding

            return OllamaEmbedding(embedding_settings)
        else:
            raise EmbeddingError(
                f"Unsupported Embedding provider: {provider}",
                provider=provider,
            )

    @classmethod
    def get_supported_providers(cls) -> list:
        """获取支持的 provider 列表。"""
        return ["openai", "azure", "ollama"]
