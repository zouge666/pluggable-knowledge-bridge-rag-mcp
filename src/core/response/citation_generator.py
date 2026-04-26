"""
Citation Generator。

生成引用信息，用于在响应中标注来源。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult


@dataclass
class Citation:
    """
    引用信息。

    表示一个检索结果的引用信息。
    """

    # 引用序号（从 1 开始）
    index: int
    # 来源文档路径
    source: str
    # 页码（可选）
    page: Optional[int]
    # Chunk ID
    chunk_id: str
    # 相似度分数
    score: float
    # 文本摘要（可选）
    snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        result = {
            "index": self.index,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "score": self.score,
        }
        if self.page is not None:
            result["page"] = self.page
        if self.snippet is not None:
            result["snippet"] = self.snippet
        return result


class CitationGenerator:
    """
    引用生成器。

    从检索结果生成引用列表。
    """

    def __init__(self, snippet_length: int = 100):
        """
        初始化引用生成器。

        Args:
            snippet_length: 摘要长度。
        """
        self.snippet_length = snippet_length

    def generate(
        self,
        results: List[RetrievalResult],
        include_snippet: bool = True,
    ) -> List[Citation]:
        """
        生成引用列表。

        Args:
            results: 检索结果列表。
            include_snippet: 是否包含文本摘要。

        Returns:
            引用列表。
        """
        citations = []

        for i, result in enumerate(results, 1):
            # 提取来源信息
            source = result.metadata.get("source_path", "unknown")
            page = result.metadata.get("page")

            # 生成摘要
            snippet = None
            if include_snippet and result.text:
                snippet = self._truncate_text(result.text, self.snippet_length)

            citation = Citation(
                index=i,
                source=source,
                page=page,
                chunk_id=result.chunk_id,
                score=result.score,
                snippet=snippet,
            )
            citations.append(citation)

        return citations

    def _truncate_text(self, text: str, max_length: int) -> str:
        """截断文本。"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
