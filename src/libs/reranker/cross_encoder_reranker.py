"""
Cross-Encoder Reranker 占位实现。

基于 Cross-Encoder 模型的重排序（占位，B7.8 阶段完善）。
"""

from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    RerankerUnavailableError,
)


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder Reranker 占位实现。

    使用 Cross-Encoder 模型对候选项进行精细打分排序。
    当前为占位实现，B7.8 阶段完善。
    """

    def __init__(self, settings) -> None:
        """
        初始化 Cross-Encoder Reranker。

        Args:
            settings: Rerank 配置。
        """
        if hasattr(settings, "rerank"):
            settings = settings.rerank

        self.model = getattr(settings, "model", None) or "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
            "CrossEncoderReranker not implemented yet. Will be completed in B7.8",
            backend="cross_encoder",
        )

    def get_backend_name(self) -> str:
        """获取后端名称。"""
        return "cross_encoder"

    def is_available(self) -> bool:
        """Cross-Encoder 当前不可用。"""
        return False
