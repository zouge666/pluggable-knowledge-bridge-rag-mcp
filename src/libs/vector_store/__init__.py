"""
VectorStore 模块。

提供可插拔的向量存储抽象接口和多种后端实现。
"""

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    UpsertResult,
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreConfigError,
    VectorStoreQueryError,
    VectorStoreUpsertError,
)
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

__all__ = [
    "BaseVectorStore",
    "VectorRecord",
    "QueryResult",
    "UpsertResult",
    "VectorStoreError",
    "VectorStoreConnectionError",
    "VectorStoreConfigError",
    "VectorStoreQueryError",
    "VectorStoreUpsertError",
    "VectorStoreFactory",
]
