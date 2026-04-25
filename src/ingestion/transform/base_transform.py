"""
Transform 抽象基类。

定义数据增强处理的统一接口。
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """
    Transform 抽象基类。

    所有数据增强处理器必须实现此接口。
    """

    @abstractmethod
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """
        对 Chunk 列表进行增强处理。

        Args:
            chunks: 待处理的 Chunk 列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[Chunk]: 处理后的 Chunk 列表。
        """
        pass

    def get_name(self) -> str:
        """
        获取处理器名称。

        Returns:
            str: 处理器名称。
        """
        return self.__class__.__name__