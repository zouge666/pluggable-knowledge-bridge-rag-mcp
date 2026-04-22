"""
Embedding 模块。

提供可插拔的 Embedding 抽象接口和多种 provider 实现。
"""

from src.libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResult,
    EmbeddingError,
    EmbeddingAuthenticationError,
    EmbeddingConnectionError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
)
from src.libs.embedding.embedding_factory import EmbeddingFactory

__all__ = [
    "BaseEmbedding",
    "EmbeddingResult",
    "EmbeddingError",
    "EmbeddingAuthenticationError",
    "EmbeddingConnectionError",
    "EmbeddingRateLimitError",
    "EmbeddingResponseError",
    "EmbeddingFactory",
]
