"""
LLM Reranker 占位实现。

基于 LLM 的重排序（占位，B7.7 阶段完善）。
"""

from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    RerankerUnavailableError,
)


class LLMReranker(BaseReranker):
    """
    LLM Reranker 占位实现。

    使用 LLM 对候选项进行智能重排序。
    当前为占位实现，B7.7 阶段完善。
    """

    def __init__(self, settings) -> None:
        """
        初始化 LLM Reranker。

        Args:
            settings: Rerank 配置。
        """
        if hasattr(settings, "rerank"):
            settings = settings.rerank

        self.top_k = getattr(settings, "top_k", None) or 5

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> RerankResult:
        """
        对候选项进行重排序。

        当前为占位实现，返回原始顺序。
        """
        raise RerankerUnavailableError(
            "LLMReranker not implemented yet. Will be completed in B7.7",
            backend="llm",
        )

    def get_backend_name(self) -> str:
        """获取后端名称。"""
        return "llm"

    def is_available(self) -> bool:
        """LLM Reranker 当前不可用。"""
        return False
