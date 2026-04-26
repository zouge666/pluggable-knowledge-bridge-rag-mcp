"""
MCP Tools 模块。

提供 MCP Server 的 Tool 实现。
"""

from src.mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubTool,
    create_handler as create_query_handler,
)
from src.mcp_server.tools.list_collections import (
    ListCollectionsTool,
    create_handler as create_list_collections_handler,
)
from src.mcp_server.tools.get_document_summary import (
    GetDocumentSummaryTool,
    create_handler as create_get_document_summary_handler,
)

__all__ = [
    "QueryKnowledgeHubTool",
    "create_query_handler",
    "ListCollectionsTool",
    "create_list_collections_handler",
    "GetDocumentSummaryTool",
    "create_get_document_summary_handler",
]