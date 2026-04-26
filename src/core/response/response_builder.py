"""
Response Builder。

构建 MCP 协议格式的响应。
"""

from typing import Any, Dict, List

from src.core.types import RetrievalResult
from src.core.response.citation_generator import Citation, CitationGenerator


class ResponseBuilder:
    """
    MCP 响应构建器。

    将检索结果转换为 MCP 协议格式的响应。
    """

    def __init__(self, citation_generator: CitationGenerator = None):
        """
        初始化响应构建器。

        Args:
            citation_generator: 引用生成器（可选）。
        """
        self.citation_generator = citation_generator or CitationGenerator()

    def build(
        self,
        results: List[RetrievalResult],
        query: str,
        include_citations: bool = True,
    ) -> Dict[str, Any]:
        """
        构建 MCP 格式响应。

        Args:
            results: 检索结果列表。
            query: 用户查询。
            include_citations: 是否包含引用信息。

        Returns:
            MCP 格式响应字典。
        """
        if not results:
            return self._build_empty_response(query)

        # 生成引用
        citations = []
        if include_citations:
            citations = self.citation_generator.generate(results)

        # 构建 Markdown 内容
        markdown_content = self._build_markdown(results, citations, query)

        # 构建 structuredContent
        structured_content = {
            "citations": [c.to_dict() for c in citations],
            "total_results": len(results),
            "query": query,
        }

        return {
            "content": [
                {
                    "type": "text",
                    "text": markdown_content,
                }
            ],
            "structuredContent": structured_content,
        }

    def _build_empty_response(self, query: str) -> Dict[str, Any]:
        """构建空结果响应。"""
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"未找到与 \"{query}\" 相关的文档。\n\n建议：\n- 尝试使用不同的关键词\n- 检查是否有数据已摄取（运行 ingest.py）\n- 确认 collection 名称是否正确",
                }
            ],
            "structuredContent": {
                "citations": [],
                "total_results": 0,
                "query": query,
            },
        }

    def _build_markdown(
        self,
        results: List[RetrievalResult],
        citations: List[Citation],
        query: str,
    ) -> str:
        """
        构建 Markdown 格式内容。

        Args:
            results: 检索结果列表。
            citations: 引用列表。
            query: 用户查询。

        Returns:
            Markdown 文本。
        """
        lines = []
        lines.append(f"# 查询结果：{query}")
        lines.append("")
        lines.append(f"找到 {len(results)} 个相关文档片段：")
        lines.append("")
        lines.append("---")
        lines.append("")

        for i, result in enumerate(results, 1):
            citation = citations[i - 1] if i <= len(citations) else None

            # 添加引用标注
            citation_marker = f"[{i}]" if citation else ""

            # 提取来源信息
            source = result.metadata.get("source_path", "unknown")
            page = result.metadata.get("page")

            # 构建来源行
            source_info = f"来源：{source}"
            if page is not None:
                source_info += f" (第 {page} 页)"
            source_info += f" {citation_marker}"

            lines.append(f"## 结果 {i}")
            lines.append("")
            lines.append(source_info)
            lines.append(f"相似度：{result.score:.4f}")
            lines.append("")
            lines.append("**内容：**")
            lines.append("")
            lines.append(result.text)
            lines.append("")
            lines.append("---")
            lines.append("")

        # 添加引用列表
        if citations:
            lines.append("## 引用来源")
            lines.append("")
            for citation in citations:
                page_info = ""
                if citation.page is not None:
                    page_info = f", 第 {citation.page} 页"
                lines.append(f"[{citation.index}] {citation.source}{page_info} (score: {citation.score:.4f})")
            lines.append("")

        return "\n".join(lines)