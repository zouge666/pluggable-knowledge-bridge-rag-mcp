"""
VectorStore 工厂。

根据配置创建对应的 VectorStore 实例。
"""

from typing import TYPE_CHECKING

from src.core.settings import Settings, VectorStoreSettings
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError

if TYPE_CHECKING:
    pass


class VectorStoreFactory:
    """
    VectorStore 工厂类。

    根据配置动态创建 VectorStore 实例，支持多种后端。
    """

    @classmethod
    def create(cls, settings: Settings) -> BaseVectorStore:
        """
        根据配置创建 VectorStore 实例。

        Args:
            settings: 应用配置。

        Returns:
            BaseVectorStore: VectorStore 实例。

        Raises:
            VectorStoreError: 不支持的后端或配置错误。
        """
        vs_settings = settings.vector_store
        provider = vs_settings.provider.lower()

        if provider == "chroma":
            from src.libs.vector_store.chroma_store import ChromaStore

            return ChromaStore(vs_settings)
        elif provider == "qdrant":
            from src.libs.vector_store.qdrant_store import QdrantStore

            return QdrantStore(vs_settings)
        elif provider == "pinecone":
            from src.libs.vector_store.pinecone_store import PineconeStore

            return PineconeStore(vs_settings)
        else:
            raise VectorStoreError(
                f"Unsupported VectorStore provider: {provider}",
                backend=provider,
            )

    @classmethod
    def create_from_settings(cls, vs_settings: VectorStoreSettings) -> BaseVectorStore:
        """
        从 VectorStoreSettings 创建 VectorStore 实例。

        Args:
            vs_settings: VectorStore 配置。

        Returns:
            BaseVectorStore: VectorStore 实例。
        """
        provider = vs_settings.provider.lower()

        if provider == "chroma":
            from src.libs.vector_store.chroma_store import ChromaStore

            return ChromaStore(vs_settings)
        elif provider == "qdrant":
            from src.libs.vector_store.qdrant_store import QdrantStore

            return QdrantStore(vs_settings)
        elif provider == "pinecone":
            from src.libs.vector_store.pinecone_store import PineconeStore

            return PineconeStore(vs_settings)
        else:
            raise VectorStoreError(
                f"Unsupported VectorStore provider: {provider}",
                backend=provider,
            )

    @classmethod
    def get_supported_providers(cls) -> list:
        """获取支持的 provider 列表。"""
        return ["chroma", "qdrant", "pinecone"]
