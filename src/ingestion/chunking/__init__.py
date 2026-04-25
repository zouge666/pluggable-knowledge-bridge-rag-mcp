"""
Chunking 模块。

提供文档切分功能。
"""

from src.ingestion.chunking.document_chunker import (
    DocumentChunker,
    FakeDocumentChunker,
)

__all__ = [
    "DocumentChunker",
    "FakeDocumentChunker",
]