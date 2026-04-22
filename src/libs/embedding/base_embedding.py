"""
Embedding 抽象基类。

定义统一的 Embedding 接口，支持多种 provider（OpenAI/Azure/Ollama）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.trace.trace_context import TraceContext


@dataclass
class EmbeddingResult:
    """Embedding 结果。"""

    vectors: List[List[float]]  # 向量列表
    model: str  # 使用的模型
    dimensions: int  # 向量维度
    usage: Optional[Dict[str, int]] = None  # {"total_tokens": x}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        result: Dict[str, Any] = {
            "model": self.model,
            "dimensions": self.dimensions,
            "count": len(self.vectors),
        }
        if self.usage:
            result["usage"] = self.usage
        return result


class BaseEmbedding(ABC):
    """
    Embedding 抽象基类。

    所有 Embedding 实现（OpenAI/Azure/Ollama）都必须实现此接口。
    """

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        trace: Optional[TraceContext] = None,
    ) -> EmbeddingResult:
        """
        批量生成文本的向量表示。

        Args:
            texts: 文本列表。
            trace: 追踪上下文（可选）。

        Returns:
            EmbeddingResult: 包含向量和元数据的结果。

        Raises:
            EmbeddingError: Embedding 调用失败时抛出。
        """
        pass

    def embed_single(
        self,
        text: str,
        trace: Optional[TraceContext] = None,
    ) -> List[float]:
        """
        便捷方法：生成单个文本的向量。

        Args:
            text: 输入文本。
            trace: 追踪上下文（可选）。

        Returns:
            List[float]: 向量表示。
        """
        result = self.embed([text], trace)
        return result.vectors[0]

    @abstractmethod
    def get_model_name(self) -> str:
        """获取当前使用的模型名称。"""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """获取向量维度。"""
        pass


class EmbeddingError(Exception):
    """Embedding 调用错误基类。"""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.provider = provider
        self.model = model
        self.original_error = original_error

        error_parts = []
        if provider:
            error_parts.append(f"[{provider}]")
        if model:
            error_parts.append(f"model={model}")
        error_parts.append(message)

        super().__init__(" ".join(error_parts))


class EmbeddingConnectionError(EmbeddingError):
    """Embedding 连接错误。"""
    pass


class EmbeddingAuthenticationError(EmbeddingError):
    """Embedding 认证错误。"""
    pass


class EmbeddingRateLimitError(EmbeddingError):
    """Embedding 速率限制错误。"""
    pass


class EmbeddingResponseError(EmbeddingError):
    """Embedding 响应错误。"""
    pass
