"""
Hybrid Search - 混合检索编排器。

编排 Dense + Sparse + Fusion 的完整混合检索流程。
"""

import time
from typing import Any, Dict, List, Optional

from src.core.query_engine.query_processor import QueryProcessor, ProcessedQuery
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFFusion
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult


class HybridSearch:
    """
    混合检索编排器。

    功能：
    1. 查询预处理（关键词提取）
    2. 并行执行稠密和稀疏检索
    3. RRF 融合结果
    4. 元数据过滤
    5. 返回 Top-K 结果
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[RRFFusion] = None,
    ):
        """
        初始化 HybridSearch。

        Args:
            settings: 应用配置。
            query_processor: 查询预处理器。
            dense_retriever: 稠密检索器。
            sparse_retriever: 稀疏检索器。
            fusion: RRF 融合器。
        """
        self._settings = settings or Settings()
        self._query_processor = query_processor or QueryProcessor()
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._fusion = fusion or RRFFusion()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """
        执行混合检索。

        Args:
            query: 查询文本。
            top_k: 返回结果数量。
            filters: 过滤条件。
            trace: 追踪上下文。

        Returns:
            List[RetrievalResult]: 检索结果列表。
        """
        start_time = time.time()

        # 1. 查询预处理
        processed = self._query_processor.process(query, filters, trace)

        # 2. 执行检索
        dense_results = []
        sparse_results = []

        # 稠密检索
        if self._dense_retriever:
            try:
                dense_results = self._dense_retriever.retrieve(
                    query=query,
                    top_k=top_k * 2,  # 召回更多，融合后截断
                    filters=processed.filters,
                    trace=trace,
                )
            except Exception:
                pass  # 降级：继续使用稀疏检索

        # 稀疏检索
        if self._sparse_retriever and processed.keywords:
            try:
                sparse_results = self._sparse_retriever.retrieve(
                    keywords=processed.keywords,
                    top_k=top_k * 2,
                    trace=trace,
                )
            except Exception:
                pass  # 降级：继续使用稠密检索

        # 3. 融合结果
        if dense_results or sparse_results:
            fusion_result = self._fusion.fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=top_k,
                trace=trace,
            )
            results = fusion_result.results
        else:
            results = []

        # 4. 应用元数据过滤（后置过滤兜底）
        if filters:
            results = self._apply_metadata_filters(results, filters)

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="hybrid_search",
                elapsed_ms=elapsed_ms,
                method="hybrid",
                details={
                    "query_length": len(query),
                    "keyword_count": len(processed.keywords),
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "result_count": len(results),
                },
            )

        return results

    def _apply_metadata_filters(
        self,
        results: List[RetrievalResult],
        filters: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """
        应用元数据过滤。

        Args:
            results: 检索结果列表。
            filters: 过滤条件。

        Returns:
            List[RetrievalResult]: 过滤后的结果列表。
        """
        if not filters:
            return results

        filtered = []
        for result in results:
            match = True
            for key, value in filters.items():
                if key == "collection":
                    # 特殊处理 collection 过滤
                    if result.metadata.get("collection") != value:
                        match = False
                        break
                elif key == "doc_type":
                    # 特殊处理 doc_type 过滤
                    if result.metadata.get("doc_type") != value:
                        match = False
                        break
                else:
                    # 通用元数据过滤
                    if result.metadata.get(key) != value:
                        match = False
                        break

            if match:
                filtered.append(result)

        return filtered

    def get_query_processor(self) -> Optional[QueryProcessor]:
        """获取查询预处理器。"""
        return self._query_processor

    def get_dense_retriever(self) -> Optional[DenseRetriever]:
        """获取稠密检索器。"""
        return self._dense_retriever

    def get_sparse_retriever(self) -> Optional[SparseRetriever]:
        """获取稀疏检索器。"""
        return self._sparse_retriever

    def get_fusion(self) -> Optional[RRFFusion]:
        """获取融合器。"""
        return self._fusion


class FakeHybridSearch:
    """
    Fake Hybrid Search 用于测试。

    返回预设结果，不执行实际检索。
    """

    def __init__(self, results: Optional[List[RetrievalResult]] = None):
        """
        初始化 FakeHybridSearch。

        Args:
            results: 预设的检索结果。
        """
        self._default_results = results or []
        self.search_calls: List[dict] = []

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """执行 fake 检索。"""
        self.search_calls.append({
            "query": query,
            "top_k": top_k,
            "filters": filters,
        })

        return self._default_results[:top_k]

    def get_query_processor(self) -> None:
        """获取查询预处理器。"""
        return None

    def get_dense_retriever(self) -> None:
        """获取稠密检索器。"""
        return None

    def get_sparse_retriever(self) -> None:
        """获取稀疏检索器。"""
        return None

    def get_fusion(self) -> None:
        """获取融合器。"""
        return None
