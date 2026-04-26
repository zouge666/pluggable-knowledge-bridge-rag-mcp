"""
get_document_summary Tool。

MCP Tool 实现：获取文档摘要信息。
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.settings import Settings

logger = logging.getLogger("mcp_server.tools.get_document_summary")


class GetDocumentSummaryTool:
    """
    获取文档摘要 Tool。

    按 doc_id 返回 title/summary/tags（从 metadata 获取）。
    """

    def __init__(self, settings: Settings = None):
        """
        初始化 Tool。

        Args:
            settings: 配置对象（可选）。
        """
        self.settings = settings or Settings()
        self._documents_path = Path("data/documents")

    def execute(
        self,
        doc_id: str,
    ) -> Dict[str, Any]:
        """
        执行获取文档摘要。

        Args:
            doc_id: 文档 ID。

        Returns:
            MCP 格式响应。
        """
        logger.info(f"Getting document summary for: {doc_id}")

        if not doc_id:
            return self._build_error_response("缺少 doc_id 参数")

        try:
            # 查找文档
            doc_info = self._find_document(doc_id)

            if doc_info is None:
                return self._build_not_found_response(doc_id)

            return self._build_response(doc_id, doc_info)

        except Exception as e:
            logger.error(f"Failed to get document summary: {e}")
            return self._build_error_response(str(e))

    def _find_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        查找文档信息。

        Args:
            doc_id: 文档 ID。

        Returns:
            文档信息，如果未找到返回 None。
        """
        # 在 documents 目录下搜索
        if not self._documents_path.exists():
            return None

        # 遍历所有集合目录
        for collection_dir in self._documents_path.iterdir():
            if collection_dir.is_dir():
                # 在集合目录下查找文件
                for file_path in collection_dir.rglob("*"):
                    if file_path.is_file():
                        # 使用文件名（不含扩展名）作为 doc_id 匹配
                        file_stem = file_path.stem
                        if file_stem == doc_id or file_path.name == doc_id:
                            return self._extract_doc_info(file_path, collection_dir.name)

        return None

    def _extract_doc_info(self, file_path: Path, collection: str) -> Dict[str, Any]:
        """
        提取文档信息。

        Args:
            file_path: 文件路径。
            collection: 集合名称。

        Returns:
            文档信息。
        """
        # 基本信息
        stat = file_path.stat()

        doc_info = {
            "title": file_path.stem,
            "source_path": str(file_path),
            "collection": collection,
            "file_type": file_path.suffix.lstrip("."),
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 2),
            "modified_at": stat.st_mtime,
        }

        # 尝试从 metadata 文件读取更多信息
        metadata_path = file_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    doc_info.update({
                        "summary": metadata.get("summary"),
                        "tags": metadata.get("tags", []),
                        "title": metadata.get("title", doc_info["title"]),
                    })
            except Exception as e:
                logger.warning(f"Failed to read metadata file: {e}")

        return doc_info

    def _build_not_found_response(self, doc_id: str) -> Dict[str, Any]:
        """构建未找到响应。"""
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"未找到文档：{doc_id}\n\n请检查 doc_id 是否正确，或确认文档已被摄取。",
                }
            ],
            "structuredContent": {
                "found": False,
                "doc_id": doc_id,
                "error": "Document not found",
            },
        }

    def _build_error_response(self, error: str) -> Dict[str, Any]:
        """构建错误响应。"""
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"获取文档摘要失败：{error}",
                }
            ],
            "structuredContent": {
                "found": False,
                "error": error,
            },
        }

    def _build_response(self, doc_id: str, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建响应。

        Args:
            doc_id: 文档 ID。
            doc_info: 文档信息。

        Returns:
            MCP 格式响应。
        """
        lines = []
        lines.append(f"# 文档摘要：{doc_info.get('title', doc_id)}")
        lines.append("")

        # 基本信息
        lines.append("## 基本信息")
        lines.append(f"- **标题**：{doc_info.get('title', 'N/A')}")
        lines.append(f"- **集合**：{doc_info.get('collection', 'N/A')}")
        lines.append(f"- **类型**：{doc_info.get('file_type', 'N/A')}")
        lines.append(f"- **大小**：{doc_info.get('size_kb', 0)} KB")
        lines.append("")

        # 摘要
        if doc_info.get("summary"):
            lines.append("## 摘要")
            lines.append(doc_info["summary"])
            lines.append("")

        # 标签
        if doc_info.get("tags"):
            lines.append("## 标签")
            tags = doc_info["tags"]
            if isinstance(tags, list):
                lines.append(", ".join(tags))
            else:
                lines.append(str(tags))
            lines.append("")

        # 来源
        lines.append("## 来源")
        lines.append(f"- **路径**：{doc_info.get('source_path', 'N/A')}")
        lines.append("")

        markdown_content = "\n".join(lines)

        return {
            "content": [
                {
                    "type": "text",
                    "text": markdown_content,
                }
            ],
            "structuredContent": {
                "found": True,
                "doc_id": doc_id,
                "title": doc_info.get("title"),
                "collection": doc_info.get("collection"),
                "file_type": doc_info.get("file_type"),
                "size_kb": doc_info.get("size_kb"),
                "summary": doc_info.get("summary"),
                "tags": doc_info.get("tags", []),
            },
        }

    def get_schema(self) -> Dict[str, Any]:
        """获取 Tool 的 JSON Schema。"""
        return {
            "name": "get_document_summary",
            "description": "获取文档的摘要信息，包括标题、摘要、标签等。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "文档 ID（文件名，不含扩展名）",
                    },
                },
                "required": ["doc_id"],
            },
        }


def create_handler(settings: Settings = None) -> callable:
    """
    创建 Tool 处理函数。

    Args:
        settings: 配置对象（可选）。

    Returns:
        Tool 处理函数。
    """
    tool = GetDocumentSummaryTool(settings=settings)

    def handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Tool 处理函数。"""
        doc_id = arguments.get("doc_id", "")
        return tool.execute(doc_id=doc_id)

    return handler
