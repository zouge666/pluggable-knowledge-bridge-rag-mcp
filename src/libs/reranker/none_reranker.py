"""
None Reranker 实现。

不做任何重排序，保持原始顺序。作为默认回退方案。
"""

from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
)


class NoneReranker(BaseReranker):
    """
    None Reranker 实现。

    不做任何重排序，直接返回原始候选项。
    作为默认回退方案，确保系统在 Reranker 不可用时仍能正常工作。
    """

    def __init__(self) -> None:
        """初始化 None Reranker。"""
        pass

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> RerankResult:
        """
        不做重排序，直接返回原始候选项。

        Args:
            query: 查询文本（忽略）。
            candidates: 候选项列表。
            top_k: 返回结果数量。
            trace: 追踪上下文。

        Returns:
            RerankResult: 原始候选项（保持原顺序）。
        """
        import time

        start_time = time.time()

        # 不做任何排序，直接返回原始顺序
        # 注意：top_k=0 时应返回空列表，而不是全部候选
        if top_k is not None:
            result_candidates = candidates[:top_k]
        else:
            result_candidates = candidates

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="rerank",
                elapsed_ms=elapsed_ms,
                method="none",
                provider="none",
                details={
                    "input_count": len(candidates),
                    "output_count": len(result_candidates),
                },
            )

        return RerankResult(
            candidates=result_candidates,
            backend="none",
            elapsed_ms=elapsed_ms,
            fallback_used=False,
        )

    def get_backend_name(self) -> str:
        """获取后端名称。"""
        return "none"

    def is_available(self) -> bool:
        """None Reranker 始终可用。"""
        return True
