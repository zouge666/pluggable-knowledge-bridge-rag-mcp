"""
Fixed Length Splitter 实现。

按固定字符长度切分文本。
"""

from typing import Optional

from src.core.trace.trace_context import TraceContext
from src.libs.splitter.base_splitter import BaseSplitter, SplitResult


class FixedLengthSplitter(BaseSplitter):
    """
    Fixed Length Splitter 实现。

    按固定字符长度切分文本，简单但可能破坏语义完整性。
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> None:
        """
        初始化 Fixed Length Splitter。

        Args:
            chunk_size: 目标 chunk 大小（字符数）。
            overlap: 重叠大小（字符数）。
        """
        self._chunk_size = chunk_size
        self._overlap = overlap

    def split_text(
        self,
        text: str,
        trace: Optional[TraceContext] = None,
    ) -> SplitResult:
        """
        将文本切分为多个片段。
        """
        if not text or not text.strip():
            return SplitResult(
                chunks=[],
                splitter_type="fixed",
                chunk_size=self._chunk_size,
                overlap=self._overlap,
            )

        chunks = []
        start = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # 下一个 chunk 的起始位置（考虑重叠）
            start = end - self._overlap if self._overlap > 0 else end

        return SplitResult(
            chunks=chunks,
            splitter_type="fixed",
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            metadata={
                "input_length": len(text),
                "overlap_applied": self._overlap > 0,
            },
        )

    def get_splitter_type(self) -> str:
        return "fixed"

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def get_overlap(self) -> int:
        return self._overlap