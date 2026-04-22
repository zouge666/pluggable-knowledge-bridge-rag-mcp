"""
Splitter 抽象基类。

定义统一的文本切分接口，支持多种切分策略（Recursive/Semantic/Fixed）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.trace.trace_context import TraceContext


@dataclass
class SplitResult:
    """切分结果。"""

    chunks: List[str]  # 切分后的文本片段
    splitter_type: str  # 使用的切分器类型
    chunk_size: int  # 目标 chunk 大小
    overlap: int  # 重叠大小
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        result: Dict[str, Any] = {
            "splitter_type": self.splitter_type,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "chunk_count": len(self.chunks),
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class BaseSplitter(ABC):
    """
    Splitter 抽象基类。

    所有 Splitter 实现（Recursive/Semantic/Fixed）都必须实现此接口。
    """

    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Optional[TraceContext] = None,
    ) -> SplitResult:
        """
        将文本切分为多个片段。

        Args:
            text: 输入文本。
            trace: 追踪上下文（可选）。

        Returns:
            SplitResult: 包含切分结果和元数据。

        Raises:
            SplitterError: 切分失败时抛出。
        """
        pass

    @abstractmethod
    def get_splitter_type(self) -> str:
        """获取切分器类型名称。"""
        pass

    @abstractmethod
    def get_chunk_size(self) -> int:
        """获取目标 chunk 大小。"""
        pass

    @abstractmethod
    def get_overlap(self) -> int:
        """获取重叠大小。"""
        pass


class SplitterError(Exception):
    """Splitter 错误基类。"""

    def __init__(
        self,
        message: str,
        splitter_type: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.splitter_type = splitter_type
        self.original_error = original_error

        error_parts = []
        if splitter_type:
            error_parts.append(f"[{splitter_type}]")
        error_parts.append(message)

        super().__init__(" ".join(error_parts))


class SplitterConfigError(SplitterError):
    """Splitter 配置错误。"""
    pass


class SplitterProcessingError(SplitterError):
    """Splitter 处理错误。"""
    pass