"""
Core 层 Reranker 编排器。

封装 libs.reranker 后端，提供 fallback 机制和追踪支持。
"""

import time
from typing import List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult
from src.libs.reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    RerankerError,
    RerankerFactory,
    NoneReranker,
)


class QueryReranker:
    """
    Core 层 Reranker 编排器。

    功能：
    1. 封装 libs.reranker 后端调用
    2. 提供 fallback 机制（失败时回退到原始排序）
    3. 支持追踪记录
    4. 转换 RetrievalResult ↔ RerankCandidate
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        reranker: Optional[BaseReranker] = None,
    ):
        """
        初始化 QueryReranker。

        Args:
            settings: 应用配置。
            reranker: Reranker 实例（可选，用于依赖注入）。
        """
        self._settings = settings or Settings()
        self._reranker = reranker
        self._fallback_reranker = NoneReranker()

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> RerankResult:
        """
        对检索结果进行重排序。

        Args:
            query: 查询文本。
            results: 检索结果列表。
            top_k: 返回结果数量（可选）。
            trace: 追踪上下文。

        Returns:
            RerankResult: 重排序结果（包含 fallback 信息）。
        """
        start_time = time.time()

        # 转换 RetrievalResult → RerankCandidate
        candidates = self._to_candidates(results)

        # 尝试使用配置的 Reranker
        reranker = self._get_reranker()
        fallback_used = False
        fallback_reason = None

        try:
            if reranker is None or not reranker.is_available():
                # Reranker 不可用，使用 fallback
                fallback_used = True
                fallback_reason = "Reranker not available"
                result = self._fallback_reranker.rerank(query, candidates, top_k)
            else:
                # 调用 Reranker
                result = reranker.rerank(query, candidates, top_k, trace)

        except RerankerError as e:
            # Reranker 失败，使用 fallback
            fallback_used = True
            fallback_reason = str(e)
            result = self._fallback_reranker.rerank(query, candidates, top_k)

        except Exception as e:
            # 其他异常，使用 fallback
            fallback_used = True
            fallback_reason = f"Unexpected error: {e}"
            result = self._fallback_reranker.rerank(query, candidates, top_k)

        # 更新结果中的 fallback 信息
        if fallback_used:
            result.fallback_used = True
            result.fallback_reason = fallback_reason

        elapsed_ms = (time.time() - start_time) * 1000
        result.elapsed_ms = elapsed_ms

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="rerank",
                elapsed_ms=elapsed_ms,
                method=reranker.get_backend_name() if reranker else "none",
                details={
                    "candidate_count": len(candidates),
                    "result_count": len(result.candidates),
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                },
            )

        return result

    def rerank_to_results(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """
        对检索结果进行重排序，返回 RetrievalResult 列表。

        Args:
            query: 查询文本。
            results: 检索结果列表。
            top_k: 返回结果数量（可选）。
            trace: 追踪上下文。

        Returns:
            List[RetrievalResult]: 重排序后的检索结果列表。
        """
        rerank_result = self.rerank(query, results, top_k, trace)
        return self._to_retrieval_results(rerank_result.candidates, results)

    def _get_reranker(self) -> Optional[BaseReranker]:
        """获取 Reranker 实例。"""
        if self._reranker is not None:
            return self._reranker

        # 延迟初始化：从配置创建
        try:
            self._reranker = RerankerFactory.create(self._settings)
            return self._reranker
        except Exception:
            return None

    def _to_candidates(
        self, results: List[RetrievalResult]
    ) -> List[RerankCandidate]:
        """将 RetrievalResult 列表转换为 RerankCandidate 列表。"""
        candidates = []
        for r in results:
            candidates.append(
                RerankCandidate(
                    id=r.chunk_id,
                    text=r.text,
                    score=r.score,
                    metadata=r.metadata,
                )
            )
        return candidates

    def _to_retrieval_results(
        self,
        candidates: List[RerankCandidate],
        original_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        将 RerankCandidate 列表转换为 RetrievalResult 列表。

        保留原始结果的 metadata（因为 rerank 可能只返回 id 和 score）。
        """
        # 构建原始结果的映射（用于补充 metadata）
        original_map = {r.chunk_id: r for r in original_results}

        results = []
        for c in candidates:
            original = original_map.get(c.id)
            if original:
                # 使用重排序后的分数，保留原始的 text 和 metadata
                results.append(
                    RetrievalResult(
                        chunk_id=c.id,
                        score=c.score,
                        text=original.text,
                        metadata=original.metadata,
                    )
                )
            else:
                # 没有原始结果，使用 candidate 的数据
                results.append(
                    RetrievalResult(
                        chunk_id=c.id,
                        score=c.score,
                        text=c.text,
                        metadata=c.metadata,
                    )
                )

        return results

    def get_reranker(self) -> Optional[BaseReranker]:
        """获取底层 Reranker 实例。"""
        return self._get_reranker()

    def is_enabled(self) -> bool:
        """检查 Reranker 是否启用。"""
        reranker = self._get_reranker()
        return reranker is not None and not isinstance(reranker, NoneReranker)


class FakeQueryReranker:
    """
    Fake Query Reranker 用于测试。

    不执行实际重排序，只记录调用。
    """

    def __init__(self, results: Optional[List[RetrievalResult]] = None):
        """
        初始化 FakeQueryReranker。

        Args:
            results: 预设的重排序结果。
        """
        self._default_results = results or []
        self.rerank_calls: List[dict] = []

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> RerankResult:
        """执行 fake 重排序。"""
        self.rerank_calls.append({
            "query": query,
            "result_count": len(results),
            "top_k": top_k,
        })

        # 返回预设结果（转换为 RerankCandidate）
        candidates = []
        for r in self._default_results[: (top_k or len(self._default_results))]:
            candidates.append(
                RerankCandidate(
                    id=r.chunk_id,
                    text=r.text,
                    score=r.score,
                    metadata=r.metadata,
                )
            )

        return RerankResult(
            candidates=candidates,
            backend="fake",
            elapsed_ms=0.1,
            fallback_used=False,
        )

    def rerank_to_results(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[RetrievalResult]:
        """执行 fake 重排序，返回 RetrievalResult 列表。"""
        rerank_result = self.rerank(query, results, top_k, trace)
        return self._default_results[: (top_k or len(self._default_results))]

    def get_reranker(self) -> None:
        """获取底层 Reranker 实例。"""
        return None

    def is_enabled(self) -> bool:
        """检查 Reranker 是否启用。"""
        return True
