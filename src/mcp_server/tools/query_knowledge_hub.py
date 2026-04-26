"""
query_knowledge_hub Tool。

MCP Tool 实现：查询知识库。
"""

import logging
import time
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.core.query_engine import (
    QueryProcessor,
    DenseRetriever,
    SparseRetriever,
    RRFFusion,
    HybridSearch,
    QueryReranker,
)
from src.core.response import ResponseBuilder
from src.core.trace import TraceContext, get_trace_collector
from src.core.types import RetrievalResult
from src.libs.embedding import EmbeddingFactory
from src.libs.vector_store import VectorStoreFactory
from src.ingestion.storage.bm25_indexer import BM25Indexer

logger = logging.getLogger("mcp_server.tools.query_knowledge_hub")


class QueryKnowledgeHubTool:
    """
    查询知识库 Tool。

    调用 HybridSearch + Reranker，构建带引用的响应。
    """

    def __init__(self, settings: Settings = None):
        """
        初始化 Tool。

        Args:
            settings: 配置对象（可选）。
        """
        self.settings = settings or Settings()
        self._initialized = False
        self._hybrid_search = None
        self._reranker = None
        self._response_builder = None

    def _lazy_init(self):
        """延迟初始化组件。"""
        if self._initialized:
            return

        try:
            # Embedding client
            embedding_client = EmbeddingFactory.create(self.settings)

            # Vector store
            vector_store = VectorStoreFactory.create(self.settings)

            # BM25 indexer
            bm25_indexer = BM25Indexer()

            # Query processor
            query_processor = QueryProcessor()

            # Dense retriever
            dense_retriever = DenseRetriever(
                settings=self.settings,
                embedding_client=embedding_client,
                vector_store=vector_store,
            )

            # Sparse retriever
            sparse_retriever = SparseRetriever(
                settings=self.settings,
                bm25_indexer=bm25_indexer,
                vector_store=vector_store,
            )

            # RRF fusion
            fusion = RRFFusion()

            # Hybrid search
            self._hybrid_search = HybridSearch(
                settings=self.settings,
                query_processor=query_processor,
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                fusion=fusion,
            )

            # Reranker
            self._reranker = QueryReranker(settings=self.settings)

            # Response builder
            self._response_builder = ResponseBuilder()

            self._initialized = True
            logger.info("QueryKnowledgeHubTool initialized")

        except Exception as e:
            logger.error(f"Failed to initialize QueryKnowledgeHubTool: {e}")
            raise

    def execute(
        self,
        query: str,
        top_k: int = 10,
        collection: Optional[str] = None,
        no_rerank: bool = False,
        trace: Optional[TraceContext] = None,
    ) -> Dict[str, Any]:
        """
        执行查询。

        Args:
            query: 查询文本。
            top_k: 返回结果数量。
            collection: 集合名称（可选）。
            no_rerank: 是否跳过重排序。
            trace: 追踪上下文（可选，用于外部传入）。

        Returns:
            MCP 格式响应。
        """
        self._lazy_init()

        # 创建 TraceContext（如果外部未传入）
        if trace is None:
            trace = TraceContext(trace_type="query")

        logger.info(f"Executing query: {query[:50]}... (top_k={top_k}, collection={collection}, trace_id={trace.trace_id})")

        try:
            # 构建过滤器
            filters = None
            if collection:
                filters = {"collection": collection}

            # 执行混合检索
            results = self._hybrid_search.search(
                query=query,
                top_k=top_k * 2,  # 召回更多，供 rerank 筛选
                filters=filters,
                trace=trace,
            )

            logger.info(f"Hybrid search returned {len(results)} results")

            # 重排序
            rerank_backend = "none"
            if not no_rerank and results:
                rerank_result = self._reranker.rerank(
                    query=query,
                    results=results,
                    top_k=top_k,
                    trace=trace,
                )
                rerank_backend = rerank_result.backend

                # 转换回 RetrievalResult
                original_map = {r.chunk_id: r for r in results}
                final_results = []
                for c in rerank_result.candidates:
                    original = original_map.get(c.id)
                    if original:
                        final_results.append(
                            RetrievalResult(
                                chunk_id=c.id,
                                score=c.score,
                                text=original.text,
                                metadata=original.metadata,
                            )
                        )

                logger.info(f"Reranked to {len(final_results)} results (backend: {rerank_backend})")
            else:
                final_results = results[:top_k]

            # 构建响应
            response = self._response_builder.build(
                results=final_results,
                query=query,
            )

            # 记录响应构建阶段
            trace.record_stage(
                stage_name="response_building",
                elapsed_ms=0.1,  # 响应构建通常很快
                method="markdown",
                details={
                    "result_count": len(final_results),
                    "has_citations": True,
                },
            )

            # Finish trace
            trace.finish()

            # Collect trace
            collector = get_trace_collector()
            collector.collect(trace)

            logger.info(f"Query completed in {trace.elapsed_ms():.2f}ms (trace_id={trace.trace_id})")

            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")

            # 记录错误到 trace
            trace.record_stage(
                stage_name="error",
                elapsed_ms=0,
                method="exception",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            trace.finish()

            # Collect trace even on error
            collector = get_trace_collector()
            collector.collect(trace)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"查询失败：{str(e)}",
                    }
                ],
                "structuredContent": {
                    "error": str(e),
                    "query": query,
                    "trace_id": trace.trace_id,
                },
            }

    def get_schema(self) -> Dict[str, Any]:
        """获取 Tool 的 JSON Schema。"""
        return {
            "name": "query_knowledge_hub",
            "description": "查询知识库，返回相关文档片段和引用信息。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查询文本",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量（默认 10）",
                        "default": 10,
                    },
                    "collection": {
                        "type": "string",
                        "description": "集合名称（可选）",
                    },
                    "no_rerank": {
                        "type": "boolean",
                        "description": "是否跳过重排序（默认 false）",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        }


def create_handler(settings: Settings = None) -> callable:
    """
    创建 Tool 处理函数。

    Args:
        settings: 配置对象（可选）。

    Returns:
        Tool 处理函数。
    """
    tool = QueryKnowledgeHubTool(settings=settings)

    def handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Tool 处理函数。"""
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 10)
        collection = arguments.get("collection")
        no_rerank = arguments.get("no_rerank", False)

        return tool.execute(
            query=query,
            top_k=top_k,
            collection=collection,
            no_rerank=no_rerank,
        )

    return handler
