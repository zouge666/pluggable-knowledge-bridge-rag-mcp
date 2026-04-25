"""
Dense Retriever - 稠密向量检索器。

使用 Embedding 模型将查询向量化，然后在 VectorStore 中检索。
"""

import time
from typing import List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.embedding import BaseEmbedding
from src.libs.vector_store import BaseVectorStore


class DenseRetriever:
    """
    稠密向量检索器。

    功能：
    1. 将查询文本向量化
    2. 在 VectorStore 中检索相似向量
    3. 返回检索结果列表
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        """
        初始化 DenseRetriever。

        Args:
            settings: 应用配置。
            embedding_client: Embedding 客户端（可选，用于依赖注入）。
            vector_store: 向量存储（可选，用于依赖注入）。
        """
        self._settings = settings or Settings()
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """
        执行稠密向量检索。

        Args:
            query: 查询文本。
            top_k: 返回结果数量。
            filters: 过滤条件。
            trace: 追踪上下文。

        Returns:
            List[RetrievalResult]: 检索结果列表。
        """
        start_time = time.time()

        # 1. 查询向量化
        query_vector = self._embed_query(query)

        # 2. 向量检索
        query_results = self._vector_store.query(
            vector=query_vector,
            top_k=top_k,
            filters=filters,
            trace=trace,
        )

        # 3. 转换为 RetrievalResult
        results = []
        for qr in query_results:
            results.append(
                RetrievalResult(
                    chunk_id=qr.id,
                    score=qr.score,
                    text=qr.text,
                    metadata=qr.metadata,
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="dense_retrieval",
                elapsed_ms=elapsed_ms,
                method="embedding",
                details={
                    "query_length": len(query),
                    "result_count": len(results),
                    "top_k": top_k,
                },
            )

        return results

    def _embed_query(self, query: str) -> List[float]:
        """
        将查询文本向量化。

        Args:
            query: 查询文本。

        Returns:
            List[float]: 查询向量。
        """
        if self._embedding_client is None:
            raise RuntimeError("Embedding client not configured")

        vectors = self._embedding_client.embed([query])
        if not vectors:
            raise RuntimeError("Failed to embed query")
        return vectors[0]

    def get_embedding_client(self) -> Optional[BaseEmbedding]:
        """获取 Embedding 客户端。"""
        return self._embedding_client

    def get_vector_store(self) -> Optional[BaseVectorStore]:
        """获取向量存储。"""
        return self._vector_store


class FakeDenseRetriever:
    """
    Fake Dense Retriever 用于测试。

    返回预设结果，不执行实际检索。
    """

    def __init__(self, results: Optional[List[RetrievalResult]] = None):
        """
        初始化 FakeDenseRetriever。

        Args:
            results: 预设的检索结果。
        """
        self._default_results = results or []
        self.retrieve_calls: List[dict] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """执行 fake 检索。"""
        self.retrieve_calls.append({
            "query": query,
            "top_k": top_k,
            "filters": filters,
        })

        # 返回 top_k 数量的结果
        return self._default_results[:top_k]

    def get_embedding_client(self) -> None:
        """获取 Embedding 客户端。"""
        return None

    def get_vector_store(self) -> None:
        """获取向量存储。"""
        return None
