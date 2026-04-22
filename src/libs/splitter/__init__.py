"""
Splitter 模块。

提供可插拔的文本切分抽象接口和多种切分策略实现。
"""

from src.libs.splitter.base_splitter import (
    BaseSplitter,
    SplitResult,
    SplitterError,
    SplitterConfigError,
    SplitterProcessingError,
)
from src.libs.splitter.splitter_factory import SplitterFactory

__all__ = [
    "BaseSplitter",
    "SplitResult",
    "SplitterError",
    "SplitterConfigError",
    "SplitterProcessingError",
    "SplitterFactory",
]
