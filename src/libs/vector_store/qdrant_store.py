"""
Qdrant Store 占位实现。

基于 Qdrant 的向量存储（占位，未来扩展）。
"""

from typing import Any, Dict, List, Optional

from src.core.trace.trace_context import TraceContext
from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    UpsertResult,
    VectorStoreError,
)


class QdrantStore(BaseVectorStore):
    """
    Qdrant Store 占位实现。

    基于 Qdrant 的向量存储，未来扩展。
    """

    def __init__(self, settings) -> None:
        if hasattr(settings, "vector_store"):
            settings = settings.vector_store

        self.url = getattr(settings, "url", None) or "http://localhost:6333"
        self.collection_name = getattr(settings, "collection_name", None) or "knowledge_hub"

    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional[TraceContext] = None,
    ) -> UpsertResult:
        raise VectorStoreError("QdrantStore not implemented yet", backend="qdrant")

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[QueryResult]:
        raise VectorStoreError("QdrantStore not implemented yet", backend="qdrant")

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        raise VectorStoreError("QdrantStore not implemented yet", backend="qdrant")

    def delete_by_ids(self, ids: List[str]) -> int:
        raise VectorStoreError("QdrantStore not implemented yet", backend="qdrant")

    def delete_by_metadata(self, filters: Dict[str, Any]) -> int:
        raise VectorStoreError("QdrantStore not implemented yet", backend="qdrant")

    def get_collection_stats(self) -> Dict[str, Any]:
        return {"count": 0, "note": "not implemented"}

    def get_backend_name(self) -> str:
        return "qdrant"