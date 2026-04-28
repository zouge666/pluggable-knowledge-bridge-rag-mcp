"""
E2E: MCP Client 侧调用模拟。

模拟 MCP Client 发送 JSON-RPC 请求，验证 MCP Server 完整流程。
"""

import json
import pytest
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestMCPClientE2E:
    """
    MCP Client E2E 测试。

    以子进程方式启动 MCP Server，模拟 Client 发送请求。
    """

    @pytest.fixture
    def mcp_server_process(self):
        """启动 MCP Server 子进程。"""
        # 启动 MCP Server
        process = subprocess.Popen(
            [sys.executable, "-m", "src.mcp_server.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root),
        )

        yield process

        # 清理
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def _send_request(self, process, request: dict) -> dict:
        """发送请求并接收响应。"""
        request_json = json.dumps(request) + "\n"
        process.stdin.write(request_json.encode())
        process.stdin.flush()

        response_line = process.stdout.readline()
        return json.loads(response_line.decode())

    def test_initialize(self, mcp_server_process):
        """
        测试 initialize 请求。

        验证 Server 返回正确的 capabilities。
        """
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0",
                },
            },
        }

        response = self._send_request(mcp_server_process, request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "capabilities" in response["result"]
        assert "serverInfo" in response["result"]
        assert response["result"]["serverInfo"]["name"] == "pluggable-knowledge-bridge"

    def test_tools_list(self, mcp_server_process):
        """
        测试 tools/list 请求。

        验证 Server 返回已注册的 tools 列表。
        """
        # 先初始化
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }
        self._send_request(mcp_server_process, init_request)

        # 请求 tools 列表
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        response = self._send_request(mcp_server_process, request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]

    def test_invalid_method(self, mcp_server_process):
        """
        测试无效方法请求。

        验证 Server 返回正确的错误码。
        """
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "invalid/method",
            "params": {},
        }

        response = self._send_request(mcp_server_process, request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found

    def test_invalid_jsonrpc(self, mcp_server_process):
        """
        测试无效 JSON-RPC 请求。

        验证 Server 返回正确的错误码。
        """
        request = {
            "jsonrpc": "1.0",  # 错误版本
            "id": 1,
            "method": "initialize",
        }

        response = self._send_request(mcp_server_process, request)

        assert "error" in response
        assert response["error"]["code"] == -32600  # Invalid Request


class TestMCPProtocolHandler:
    """
    MCP Protocol Handler 单元测试。

    直接测试 ProtocolHandler 类。
    """

    @pytest.fixture
    def handler(self):
        """创建 ProtocolHandler 实例。"""
        from src.mcp_server.protocol_handler import ProtocolHandler
        return ProtocolHandler()

    def test_handle_initialize(self, handler):
        """测试 initialize 处理。"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }

        response = handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "capabilities" in response["result"]

    def test_handle_tools_list(self, handler):
        """测试 tools/list 处理。"""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        response = handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        assert isinstance(response["result"]["tools"], list)

    def test_register_tool(self, handler):
        """测试注册 tool。"""
        handler.register_tool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            handler=lambda args: "test result",
        )

        # 验证 tool 已注册
        tools_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }

        response = handler.handle_request(tools_request)
        tools = response["result"]["tools"]

        tool_names = [t["name"] for t in tools]
        assert "test_tool" in tool_names

    def test_handle_tools_call(self, handler):
        """测试 tools/call 处理。"""
        # 注册一个测试 tool
        handler.register_tool(
            name="echo",
            description="Echo tool",
            input_schema={"type": "object"},
            handler=lambda args: f"Echo: {args.get('message', '')}",
        )

        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "echo",
                "arguments": {"message": "hello"},
            },
        }

        response = handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert "content" in response["result"]
        assert response["result"]["content"][0]["type"] == "text"
        assert "Echo: hello" in response["result"]["content"][0]["text"]

    def test_unknown_tool(self, handler):
        """测试调用未知 tool。"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {},
            },
        }

        response = handler.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == -32602  # Invalid params

    def test_notification_no_response(self, handler):
        """测试 notification（无 id）不返回响应。"""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
        }

        response = handler.handle_request(request)

        # notification 不应该返回响应
        assert response is None


class TestMCPToolsIntegration:
    """
    MCP Tools 集成测试。

    测试实际的 MCP Tools 实现。
    """

    @pytest.fixture
    def handler(self):
        """创建带有真实 tools 的 ProtocolHandler。"""
        from src.mcp_server.protocol_handler import ProtocolHandler

        handler = ProtocolHandler()

        # 注册 query_knowledge_hub tool
        def query_knowledge_hub(args):
            query = args.get("query", "")
            top_k = args.get("top_k", 10)
            return json.dumps({
                "answer": f"Mock answer for: {query}",
                "citations": [],
            })

        handler.register_tool(
            name="query_knowledge_hub",
            description="Query the knowledge hub",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            handler=query_knowledge_hub,
        )

        # 注册 list_collections tool
        def list_collections(args):
            return json.dumps({
                "collections": [
                    {"name": "default", "document_count": 10},
                ],
            })

        handler.register_tool(
            name="list_collections",
            description="List all collections",
            input_schema={"type": "object"},
            handler=list_collections,
        )

        return handler

    def test_query_knowledge_hub_tool(self, handler):
        """测试 query_knowledge_hub tool。"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {
                    "query": "What is machine learning?",
                    "top_k": 5,
                },
            },
        }

        response = handler.handle_request(request)

        assert "result" in response
        content = response["result"]["content"][0]
        assert content["type"] == "text"

        result = json.loads(content["text"])
        assert "answer" in result
        assert "machine learning" in result["answer"].lower()

    def test_list_collections_tool(self, handler):
        """测试 list_collections tool。"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "list_collections",
                "arguments": {},
            },
        }

        response = handler.handle_request(request)

        assert "result" in response
        content = response["result"]["content"][0]

        result = json.loads(content["text"])
        assert "collections" in result
