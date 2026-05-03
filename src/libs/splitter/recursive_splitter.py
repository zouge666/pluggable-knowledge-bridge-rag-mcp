"""
Recursive Character Text Splitter 实现。

基于 LangChain 的 RecursiveCharacterTextSplitter，按层级分隔符递归切分。
"""

from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.libs.splitter.base_splitter import (
    BaseSplitter,
    SplitResult,
    SplitterProcessingError,
)

try:
    # LangChain 0.2+ / 1.x: splitters are in a dedicated package.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        # Older LangChain releases exposed splitters from langchain directly.
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None


# Markdown 优化的分隔符层级
MARKDOWN_SEPARATORS = [
    "\n\n",  # 段落
    "\n",    # 行
    "。",    # 中文句号
    ".",     # 英文句号
    "！",    # 中文感叹号
    "!",     # 英文感叹号
    "？",    # 中文问号
    "?",     # 英文问号
    "；",    # 中文分号
    ";",     # 英文分号
    "，",    # 中文逗号
    ",",     # 英文逗号
    " ",     # 空格
    "",      # 字符
]


class RecursiveSplitter(BaseSplitter):
    """
    Recursive Character Text Splitter 实现。

    按层级分隔符递归切分文本，对 Markdown 文档结构有良好适配性。
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: Optional[List[str]] = None,
    ) -> None:
        """
        初始化 Recursive Splitter。

        Args:
            chunk_size: 目标 chunk 大小（字符数）。
            overlap: 重叠大小（字符数）。
            separators: 自定义分隔符层级（可选）。
        """
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._separators = separators or MARKDOWN_SEPARATORS

        # 延迟初始化 LangChain splitter
        self._splitter = None

    def _get_splitter(self):
        """获取或创建 LangChain splitter。"""
        if self._splitter is None:
            if RecursiveCharacterTextSplitter is None:
                raise ImportError(
                    "Recursive splitter requires langchain_text_splitters (or legacy langchain). "
                    "Install with: python -m pip install langchain-text-splitters"
                )
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._overlap,
                separators=self._separators,
                length_function=len,
            )
        return self._splitter

    def split_text(
        self,
        text: str,
        trace: Optional[TraceContext] = None,
    ) -> SplitResult:
        """
        将文本切分为多个片段。

        Args:
            text: 输入文本。
            trace: 追踪上下文。

        Returns:
            SplitResult: 包含切分结果和元数据。
        """
        import time

        start_time = time.time()

        if not text or not text.strip():
            return SplitResult(
                chunks=[],
                splitter_type="recursive",
                chunk_size=self._chunk_size,
                overlap=self._overlap,
            )

        try:
            splitter = self._get_splitter()
            chunks = splitter.split_text(text)

            elapsed_ms = (time.time() - start_time) * 1000

            # 记录追踪
            if trace:
                trace.record_stage(
                    stage_name="split",
                    elapsed_ms=elapsed_ms,
                    method="recursive",
                    provider="langchain",
                    details={
                        "chunk_size": self._chunk_size,
                        "overlap": self._overlap,
                        "input_length": len(text),
                        "output_chunks": len(chunks),
                    },
                )

            # 计算统计信息
            avg_chunk_length = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

            return SplitResult(
                chunks=chunks,
                splitter_type="recursive",
                chunk_size=self._chunk_size,
                overlap=self._overlap,
                metadata={
                    "input_length": len(text),
                    "avg_chunk_length": avg_chunk_length,
                },
            )

        except Exception as e:
            raise SplitterProcessingError(
                str(e),
                splitter_type="recursive",
                original_error=e,
            )

    def get_splitter_type(self) -> str:
        """获取切分器类型。"""
        return "recursive"

    def get_chunk_size(self) -> int:
        """获取目标 chunk 大小。"""
        return self._chunk_size

    def get_overlap(self) -> int:
        """获取重叠大小。"""
        return self._overlap
