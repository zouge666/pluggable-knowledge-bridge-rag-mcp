"""
Reranker 抽象基类。

定义统一的重排序接口，支持多种后端（None/CrossEncoder/LLM）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.trace.trace_context import TraceContext


@dataclass
class RerankCandidate:
    """重排序候选项。"""

    id: str  # 候选项 ID
    text: str  # 候选项文本
    score: float  # 原始分数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class RerankResult:
    """重排序结果。"""

    candidates: List[RerankCandidate]  # 重排序后的候选项列表
    backend: str  # 使用的后端
    elapsed_ms: Optional[float] = None  # 耗时（毫秒）
    fallback_used: bool = False  # 是否使用了回退
    fallback_reason: Optional[str] = None  # 回退原因

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "backend": self.backend,
            "count": len(self.candidates),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "elapsed_ms": self.elapsed_ms,
        }


class BaseReranker(ABC):
    """
    Reranker 抽象基类。

    所有 Reranker 实现（None/CrossEncoder/LLM）都必须实现此接口。
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> RerankResult:
        """
        对候选项进行重排序。

        Args:
            query: 查询文本。
            candidates: 候选项列表。
            top_k: 返回结果数量（可选，默认返回全部）。
            trace: 追踪上下文（可选）。

        Returns:
            RerankResult: 重排序结果。

        Raises:
            RerankerError: 重排序失败时抛出。
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """获取后端名称。"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查 Reranker 是否可用。"""
        pass


class RerankerError(Exception):
    """Reranker 错误基类。"""

    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.backend = backend
        self.original_error = original_error

        error_parts = []
        if backend:
            error_parts.append(f"[{backend}]")
        error_parts.append(message)

        super().__init__(" ".join(error_parts))


class RerankerConfigError(RerankerError):
    """Reranker 配置错误。"""
    pass


class RerankerUnavailableError(RerankerError):
    """Reranker 不可用错误。"""
    pass


class RerankerTimeoutError(RerankerError):
    """Reranker 超时错误。"""
    pass
