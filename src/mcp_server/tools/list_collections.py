"""
list_collections Tool。

MCP Tool 实现：列出知识库中的集合。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import Settings

logger = logging.getLogger("mcp_server.tools.list_collections")


class ListCollectionsTool:
    """
    列出集合 Tool。

    列出 data/documents/ 下的集合并附带统计信息。
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
        include_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        执行列出集合。

        Args:
            include_stats: 是否包含统计信息。

        Returns:
            MCP 格式响应。
        """
        logger.info(f"Listing collections from {self._documents_path}")

        try:
            collections = self._list_collections(include_stats)

            if not collections:
                return self._build_empty_response()

            return self._build_response(collections)

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"列出集合失败：{str(e)}",
                    }
                ],
                "structuredContent": {
                    "error": str(e),
                },
            }

    def _list_collections(self, include_stats: bool) -> List[Dict[str, Any]]:
        """
        列出集合。

        Args:
            include_stats: 是否包含统计信息。

        Returns:
            集合列表。
        """
        collections = []

        if not self._documents_path.exists():
            return collections

        for collection_dir in self._documents_path.iterdir():
            if collection_dir.is_dir():
                collection_name = collection_dir.name
                collection_info = {
                    "name": collection_name,
                }

                if include_stats:
                    stats = self._get_collection_stats(collection_dir)
                    collection_info.update(stats)

                collections.append(collection_info)

        # 按名称排序
        collections.sort(key=lambda x: x["name"])

        return collections

    def _get_collection_stats(self, collection_dir: Path) -> Dict[str, Any]:
        """
        获取集合统计信息。

        Args:
            collection_dir: 集合目录。

        Returns:
            统计信息。
        """
        document_count = 0
        total_size = 0

        for file_path in collection_dir.rglob("*"):
            if file_path.is_file():
                document_count += 1
                total_size += file_path.stat().st_size

        return {
            "document_count": document_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def _build_empty_response(self) -> Dict[str, Any]:
        """构建空结果响应。"""
        return {
            "content": [
                {
                    "type": "text",
                    "text": "当前没有集合。\n\n提示：运行 ingest.py 摄取文档以创建集合。",
                }
            ],
            "structuredContent": {
                "collections": [],
                "total_collections": 0,
            },
        }

    def _build_response(self, collections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        构建响应。

        Args:
            collections: 集合列表。

        Returns:
            MCP 格式响应。
        """
        lines = []
        lines.append("# 知识库集合列表")
        lines.append("")
        lines.append(f"共 {len(collections)} 个集合：")
        lines.append("")

        for collection in collections:
            name = collection["name"]
            lines.append(f"## {name}")

            if "document_count" in collection:
                doc_count = collection["document_count"]
                size_mb = collection.get("total_size_mb", 0)
                lines.append(f"- 文档数：{doc_count}")
                lines.append(f"- 大小：{size_mb} MB")

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
                "collections": collections,
                "total_collections": len(collections),
            },
        }

    def get_schema(self) -> Dict[str, Any]:
        """获取 Tool 的 JSON Schema。"""
        return {
            "name": "list_collections",
            "description": "列出知识库中的所有集合及其统计信息。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include_stats": {
                        "type": "boolean",
                        "description": "是否包含统计信息（默认 true）",
                        "default": True,
                    },
                },
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
    tool = ListCollectionsTool(settings=settings)

    def handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Tool 处理函数。"""
        include_stats = arguments.get("include_stats", True)
        return tool.execute(include_stats=include_stats)

    return handler
