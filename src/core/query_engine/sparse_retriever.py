"""
Sparse Retriever - 稀疏向量检索器。

使用 BM25 索引进行关键词检索。
"""

import time
from typing import Dict, List, Optional, Tuple

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store import BaseVectorStore


class SparseRetriever:
    """
    稀疏向量检索器。

    功能：
    1. 使用 BM25 索引进行关键词检索
    2. 从 VectorStore 获取文本和元数据
    3. 返回检索结果列表
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        """
        初始化 SparseRetriever。

        Args:
            settings: 应用配置。
            bm25_indexer: BM25 索引器（可选，用于依赖注入）。
            vector_store: 向量存储（可选，用于依赖注入）。
        """
        self._settings = settings or Settings()
        self._bm25_indexer = bm25_indexer
        self._vector_store = vector_store

    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """
        执行稀疏向量检索。

        Args:
            keywords: 关键词列表。
            top_k: 返回结果数量。
            trace: 追踪上下文。

        Returns:
            List[RetrievalResult]: 检索结果列表。
        """
        start_time = time.time()

        if not keywords:
            return []

        # 1. BM25 检索
        bm25_results = self._bm25_query(keywords, top_k)

        if not bm25_results:
            return []

        # 2. 获取文本和元数据
        chunk_ids = [chunk_id for chunk_id, _ in bm25_results]
        records = self._get_by_ids(chunk_ids)

        # 3. 合并结果
        results = []
        records_map = {r["id"]: r for r in records}

        for chunk_id, score in bm25_results:
            record = records_map.get(chunk_id, {})
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=record.get("text", ""),
                    metadata=record.get("metadata", {}),
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="sparse_retrieval",
                elapsed_ms=elapsed_ms,
                method="bm25",
                details={
                    "keyword_count": len(keywords),
                    "result_count": len(results),
                    "top_k": top_k,
                },
            )

        return results

    def _bm25_query(
        self,
        keywords: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        执行 BM25 查询。

        Args:
            keywords: 关键词列表。
            top_k: 返回结果数量。

        Returns:
            List[Tuple[str, float]]: (chunk_id, score) 列表。
        """
        if self._bm25_indexer is None:
            raise RuntimeError("BM25 indexer not configured")

        return self._bm25_indexer.query(keywords, top_k)

    def _get_by_ids(self, ids: List[str]) -> List[Dict]:
        """
        根据 ID 批量获取记录。

        Args:
            ids: ID 列表。

        Returns:
            List[Dict]: 记录列表。
        """
        if self._vector_store is None:
            raise RuntimeError("Vector store not configured")

        return self._vector_store.get_by_ids(ids)

    def get_bm25_indexer(self) -> Optional[BM25Indexer]:
        """获取 BM25 索引器。"""
        return self._bm25_indexer

    def get_vector_store(self) -> Optional[BaseVectorStore]:
        """获取向量存储。"""
        return self._vector_store


class FakeSparseRetriever:
    """
    Fake Sparse Retriever 用于测试。

    返回预设结果，不执行实际检索。
    """

    def __init__(self, results: Optional[List[RetrievalResult]] = None):
        """
        初始化 FakeSparseRetriever。

        Args:
            results: 预设的检索结果。
        """
        self._default_results = results or []
        self.retrieve_calls: List[dict] = []

    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """执行 fake 检索。"""
        self.retrieve_calls.append({
            "keywords": keywords,
            "top_k": top_k,
        })

        # 返回 top_k 数量的结果
        return self._default_results[:top_k]

    def get_bm25_indexer(self) -> None:
        """获取 BM25 索引器。"""
        return None

    def get_vector_store(self) -> None:
        """获取向量存储。"""
        return None
