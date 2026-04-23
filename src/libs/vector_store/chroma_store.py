"""
ChromaStore 占位实现。

基于 ChromaDB 的向量存储实现（占位，B7.6 阶段完善）。
"""

from typing import Any, Dict, List, Optional

from src.core.trace.trace_context import TraceContext
from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    UpsertResult,
    VectorStoreConfigError,
)


class ChromaStore(BaseVectorStore):
    """
    ChromaStore 占位实现。

    基于 ChromaDB 的向量存储，支持本地持久化。
    当前为占位实现，B7.6 阶段完善。
    """

    def __init__(self, settings) -> None:
        """
        初始化 ChromaStore。

        Args:
            settings: VectorStore 配置。
        """
        if hasattr(settings, "vector_store"):
            settings = settings.vector_store

        self.persist_directory = getattr(settings, "persist_directory", None) or "./data/db/chroma"
        self.collection_name = getattr(settings, "collection_name", None) or "knowledge_hub"

        # 延迟初始化客户端
        self._client = None
        self._collection = None

    def _get_collection(self):
        """获取或创建 ChromaDB collection。"""
        if self._collection is None:
            try:
                import chromadb

                self._client = chromadb.PersistentClient(path=self.persist_directory)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                )
            except ImportError:
                raise ImportError(
                    "chromadb package is required. Install it with: pip install chromadb"
                )
        return self._collection

    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional[TraceContext] = None,
    ) -> UpsertResult:
        """
        批量插入或更新向量记录。

        当前为占位实现。
        """
        if not records:
            return UpsertResult(success=True, upserted_count=0)

        collection = self._get_collection()

        ids = [r.id for r in records]
        vectors = [r.vector for r in records]
        texts = [r.text for r in records]
        metadatas = [r.metadata for r in records]

        try:
            collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )

            return UpsertResult(
                success=True,
                upserted_count=len(records),
                ids=ids,
            )
        except Exception as e:
            raise VectorStoreConfigError(
                str(e),
                backend="chroma",
                original_error=e,
            )

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[QueryResult]:
        """
        向量相似度查询。

        当前为占位实现。
        """
        collection = self._get_collection()

        try:
            results = collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=filters if filters else None,
                include=["documents", "metadatas", "distances"],
            )

            # 解析结果
            query_results = []
            if results and results["ids"]:
                ids = results["ids"][0]
                docs = results["documents"][0] or []
                metas = results["metadatas"][0] or []
                distances = results["distances"][0] or []

                for i, id_ in enumerate(ids):
                    # Chroma 返回的是距离，转换为相似度分数
                    score = 1.0 - distances[i] if distances else 0.0
                    query_results.append(
                        QueryResult(
                            id=id_,
                            score=score,
                            text=docs[i] if i < len(docs) else "",
                            metadata=metas[i] if i < len(metas) else {},
                        )
                    )

            return query_results
        except Exception as e:
            raise VectorStoreConfigError(
                str(e),
                backend="chroma",
                original_error=e,
            )

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """根据 ID 批量获取记录。"""
        if not ids:
            return []

        collection = self._get_collection()

        try:
            results = collection.get(
                ids=ids,
                include=["documents", "metadatas"],
            )

            records = []
            if results and results["ids"]:
                docs = results["documents"] or []
                metas = results["metadatas"] or []

                for i, id_ in enumerate(results["ids"]):
                    records.append({
                        "id": id_,
                        "text": docs[i] if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                    })

            return records
        except Exception as e:
            raise VectorStoreConfigError(
                str(e),
                backend="chroma",
                original_error=e,
            )

    def delete_by_ids(self, ids: List[str]) -> int:
        """根据 ID 批量删除记录。"""
        if not ids:
            return 0

        collection = self._get_collection()

        try:
            collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            raise VectorStoreConfigError(
                str(e),
                backend="chroma",
                original_error=e,
            )

    def delete_by_metadata(self, filters: Dict[str, Any]) -> int:
        """根据元数据条件批量删除记录。"""
        collection = self._get_collection()

        try:
            # 先查询符合条件的记录
            results = collection.get(where=filters)
            ids = results["ids"] if results and results["ids"] else []

            if ids:
                collection.delete(ids=ids)
                return len(ids)
            return 0
        except Exception as e:
            raise VectorStoreConfigError(
                str(e),
                backend="chroma",
                original_error=e,
            )

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息。"""
        collection = self._get_collection()

        try:
            count = collection.count()
            return {
                "count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            raise VectorStoreConfigError(
                str(e),
                backend="chroma",
                original_error=e,
            )

    def get_backend_name(self) -> str:
        """获取后端名称。"""
        return "chroma"