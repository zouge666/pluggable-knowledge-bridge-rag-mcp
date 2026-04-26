"""
MCP Protocol Handler。

封装 JSON-RPC 2.0 协议解析，处理 MCP 核心方法：
- initialize: 初始化连接，返回 server capabilities
- tools/list: 返回可用 tools 列表
- tools/call: 调用指定 tool
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("mcp_server.protocol_handler")

# JSON-RPC 2.0 错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ProtocolHandler:
    """
    MCP Protocol Handler。

    处理 JSON-RPC 2.0 请求，路由到对应的处理方法。
    """

    def __init__(self):
        """初始化 Protocol Handler。"""
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._tool_handlers: Dict[str, Callable] = {}
        self._initialized = False

        # 注册默认方法处理器
        self._method_handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }

        logger.info("ProtocolHandler initialized")

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理 JSON-RPC 请求。

        Args:
            request: JSON-RPC 请求字典。

        Returns:
            响应字典，如果是 notification 则返回 None。
        """
        # 验证请求格式
        if not self._validate_request(request):
            # 对于非 dict 请求，无法获取 id
            request_id = request.get("id") if isinstance(request, dict) else None
            return self._error_response(
                request_id,
                INVALID_REQUEST,
                "Invalid Request",
            )

        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        # 检查是否是 notification（无 id）
        is_notification = "id" not in request

        # 查找方法处理器
        handler = self._method_handlers.get(method)
        if handler is None:
            if is_notification:
                logger.warning(f"Unknown notification method: {method}")
                return None
            return self._error_response(
                request_id,
                METHOD_NOT_FOUND,
                f"Method not found: {method}",
            )

        # 执行处理器
        try:
            result = handler(params)

            # notification 不需要响应
            if is_notification:
                return None

            return self._success_response(request_id, result)

        except ValueError as e:
            logger.error(f"Invalid params for {method}: {e}")
            if is_notification:
                return None
            return self._error_response(request_id, INVALID_PARAMS, str(e))

        except Exception as e:
            logger.error(f"Internal error handling {method}: {e}")
            if is_notification:
                return None
            return self._error_response(
                request_id,
                INTERNAL_ERROR,
                "Internal error",
            )

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
    ):
        """
        注册一个 tool。

        Args:
            name: Tool 名称。
            description: Tool 描述。
            input_schema: 输入参数 JSON Schema。
            handler: Tool 处理函数。
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
        }
        self._tool_handlers[name] = handler
        logger.info(f"Registered tool: {name}")

    def _validate_request(self, request: Dict[str, Any]) -> bool:
        """
        验证 JSON-RPC 请求格式。

        Args:
            request: 请求字典。

        Returns:
            bool: 是否有效。
        """
        if not isinstance(request, dict):
            return False

        # 必须有 jsonrpc 和 method
        if request.get("jsonrpc") != "2.0":
            return False

        if "method" not in request:
            return False

        return True

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 initialize 请求。

        Args:
            params: 请求参数。

        Returns:
            初始化结果。
        """
        self._initialized = True

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "pluggable-knowledge-bridge",
                "version": "0.1.0",
            },
        }

    def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 tools/list 请求。

        Args:
            params: 请求参数。

        Returns:
            Tools 列表。
        """
        tools = list(self._tools.values())
        return {"tools": tools}

    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 tools/call 请求。

        Args:
            params: 请求参数，包含 name 和 arguments。

        Returns:
            Tool 执行结果。
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise ValueError("Missing tool name")

        handler = self._tool_handlers.get(tool_name)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        # 执行 tool
        result = handler(arguments)

        # 包装为 MCP content 格式
        return {
            "content": [
                {
                    "type": "text",
                    "text": result if isinstance(result, str) else str(result),
                }
            ],
        }

    def _success_response(
        self,
        request_id: Any,
        result: Any,
    ) -> Dict[str, Any]:
        """
        构建成功响应。

        Args:
            request_id: 请求 ID。
            result: 结果。

        Returns:
            响应字典。
        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def _error_response(
        self,
        request_id: Any,
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        构建错误响应。

        Args:
            request_id: 请求 ID。
            code: 错误码。
            message: 错误消息。
            data: 错误数据（可选）。

        Returns:
            响应字典。
        """
        error = {
            "code": code,
            "message": message,
        }
        if data is not None:
            error["data"] = data

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error,
        }
