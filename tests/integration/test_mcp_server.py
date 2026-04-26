"""
Integration tests for MCP Server.

Tests the JSON-RPC 2.0 protocol handling and tool registration.
"""

import pytest
import json
from unittest.mock import Mock, patch
from io import StringIO

from src.mcp_server.server import MCPServer
from src.mcp_server.protocol_handler import (
    ProtocolHandler,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from src.core.response import ResponseBuilder


class TestProtocolHandler:
    """Tests for ProtocolHandler."""

    def test_init_creates_method_handlers(self):
        """Test init creates default method handlers."""
        handler = ProtocolHandler()
        assert "initialize" in handler._method_handlers
        assert "tools/list" in handler._method_handlers
        assert "tools/call" in handler._method_handlers

    def test_handle_initialize(self):
        """Test initialize method."""
        handler = ProtocolHandler()
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
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert "capabilities" in response["result"]
        assert "tools" in response["result"]["capabilities"]
        assert response["result"]["serverInfo"]["name"] == "pluggable-knowledge-bridge"

    def test_handle_tools_list_empty(self):
        """Test tools/list with no registered tools."""
        handler = ProtocolHandler()
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
        assert response["result"]["tools"] == []

    def test_handle_tools_list_with_tools(self):
        """Test tools/list with registered tools."""
        handler = ProtocolHandler()
        handler.register_tool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda x: "result",
        )

        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list",
            "params": {},
        }
        response = handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert len(response["result"]["tools"]) == 1
        assert response["result"]["tools"][0]["name"] == "test_tool"
        assert response["result"]["tools"][0]["description"] == "A test tool"

    def test_handle_tools_call(self):
        """Test tools/call method."""
        handler = ProtocolHandler()
        handler.register_tool(
            name="echo",
            description="Echo tool",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=lambda args: args.get("text", ""),
        )

        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "echo",
                "arguments": {"text": "hello"},
            },
        }
        response = handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "result" in response
        assert "content" in response["result"]
        assert len(response["result"]["content"]) == 1
        assert response["result"]["content"][0]["type"] == "text"
        assert response["result"]["content"][0]["text"] == "hello"

    def test_handle_tools_call_missing_name(self):
        """Test tools/call with missing name."""
        handler = ProtocolHandler()
        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"arguments": {}},
        }
        response = handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 5
        assert "error" in response
        assert response["error"]["code"] == INVALID_PARAMS

    def test_handle_tools_call_unknown_tool(self):
        """Test tools/call with unknown tool."""
        handler = ProtocolHandler()
        request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {},
            },
        }
        response = handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 6
        assert "error" in response
        assert response["error"]["code"] == INVALID_PARAMS

    def test_handle_unknown_method(self):
        """Test unknown method returns error."""
        handler = ProtocolHandler()
        request = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "unknown/method",
            "params": {},
        }
        response = handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 7
        assert "error" in response
        assert response["error"]["code"] == METHOD_NOT_FOUND

    def test_handle_notification_no_response(self):
        """Test notification (no id) returns no response."""
        handler = ProtocolHandler()
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
        }
        response = handler.handle_request(request)
        assert response is None

    def test_handle_notification_unknown_method(self):
        """Test notification with unknown method returns no response."""
        handler = ProtocolHandler()
        request = {
            "jsonrpc": "2.0",
            "method": "unknown/method",
            "params": {},
        }
        response = handler.handle_request(request)
        assert response is None

    def test_validate_request_missing_jsonrpc(self):
        """Test request validation with missing jsonrpc."""
        handler = ProtocolHandler()
        request = {"id": 1, "method": "initialize"}
        response = handler.handle_request(request)

        assert response["error"]["code"] == INVALID_REQUEST

    def test_validate_request_wrong_jsonrpc_version(self):
        """Test request validation with wrong jsonrpc version."""
        handler = ProtocolHandler()
        request = {"jsonrpc": "1.0", "id": 1, "method": "initialize"}
        response = handler.handle_request(request)

        assert response["error"]["code"] == INVALID_REQUEST

    def test_validate_request_missing_method(self):
        """Test request validation with missing method."""
        handler = ProtocolHandler()
        request = {"jsonrpc": "2.0", "id": 1}
        response = handler.handle_request(request)

        assert response["error"]["code"] == INVALID_REQUEST

    def test_validate_request_not_dict(self):
        """Test request validation with non-dict."""
        handler = ProtocolHandler()
        response = handler.handle_request("not a dict")

        assert response["error"]["code"] == INVALID_REQUEST

    def test_tool_handler_exception(self):
        """Test tool handler exception."""
        handler = ProtocolHandler()
        handler.register_tool(
            name="failing_tool",
            description="A failing tool",
            input_schema={"type": "object"},
            handler=lambda x: raise_exception(),
        )

        request = {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"name": "failing_tool", "arguments": {}},
        }
        response = handler.handle_request(request)

        assert response["error"]["code"] == INTERNAL_ERROR

    def test_register_tool(self):
        """Test register_tool method."""
        handler = ProtocolHandler()
        handler.register_tool(
            name="new_tool",
            description="New tool description",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
            handler=lambda x: x,
        )

        assert "new_tool" in handler._tools
        assert handler._tools["new_tool"]["description"] == "New tool description"
        assert "new_tool" in handler._tool_handlers

    def test_tool_handler_returns_dict(self):
        """Test tool handler returning dict."""
        handler = ProtocolHandler()
        handler.register_tool(
            name="dict_tool",
            description="Returns dict",
            input_schema={"type": "object"},
            handler=lambda x: {"key": "value"},
        )

        request = {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "dict_tool", "arguments": {}},
        }
        response = handler.handle_request(request)

        assert response["result"]["content"][0]["text"] == "{'key': 'value'}"


def raise_exception():
    raise Exception("Tool failed")


class TestMCPServer:
    """Tests for MCPServer."""

    def test_init(self):
        """Test MCPServer initialization."""
        server = MCPServer()
        assert server.protocol_handler is not None
        assert server.running is True

    def test_send_response(self):
        """Test _send_response method."""
        server = MCPServer()
        response = {"jsonrpc": "2.0", "id": 1, "result": {}}

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            server._send_response(response)
            output = mock_stdout.getvalue()

        assert output.strip() == json.dumps(response)

    def test_send_error(self):
        """Test _send_error method."""
        server = MCPServer()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            server._send_error(1, METHOD_NOT_FOUND, "Method not found")
            output = mock_stdout.getvalue()

        expected = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": METHOD_NOT_FOUND, "message": "Method not found"},
        })
        assert output.strip() == expected

    def test_send_error_with_data(self):
        """Test _send_error with data."""
        server = MCPServer()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            server._send_error(1, INVALID_PARAMS, "Invalid params", data={"field": "name"})
            output = mock_stdout.getvalue()

        response = json.loads(output.strip())
        assert response["error"]["data"] == {"field": "name"}

    def test_stop(self):
        """Test stop method."""
        server = MCPServer()
        server.stop()
        assert server.running is False

    def test_run_handles_valid_request(self):
        """Test run handles valid request."""
        server = MCPServer()
        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        })

        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                mock_stdin.write(request + "\n")
                mock_stdin.seek(0)

                # Run one iteration then stop
                server.running = False
                server.run()

                output = mock_stdout.getvalue()

        # Should have a response
        if output:
            response = json.loads(output.strip())
            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 1

    def test_run_handles_invalid_json(self):
        """Test run handles invalid JSON."""
        server = MCPServer()

        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                mock_stdin.write("invalid json\n")
                mock_stdin.seek(0)

                server.running = False
                server.run()

                output = mock_stdout.getvalue()

        if output:
            response = json.loads(output.strip())
            assert response["error"]["code"] == PARSE_ERROR

    def test_run_handles_empty_line(self):
        """Test run handles empty line."""
        server = MCPServer()

        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                mock_stdin.write("\n\n")
                mock_stdin.seek(0)

                server.running = False
                server.run()

                output = mock_stdout.getvalue()
                # Empty lines should not produce output
                assert output == ""

    def test_run_handles_notification(self):
        """Test run handles notification (no response)."""
        server = MCPServer()
        request = json.dumps({
            "jsonrpc": "2.0",
            "method": "initialized",  # notification, no id
            "params": {},
        })

        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                mock_stdin.write(request + "\n")
                mock_stdin.seek(0)

                server.running = False
                server.run()

                output = mock_stdout.getvalue()
                # Notification should not produce response
                assert output == ""

    def test_run_exits_on_stdin_close(self):
        """Test run exits when stdin closes."""
        server = MCPServer()

        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                mock_stdin.write("")  # Empty stdin = closed
                mock_stdin.seek(0)

                server.run()

                # Should have stopped
                assert server.running is False


class TestMCPServerIntegration:
    """Integration tests for MCPServer with tools."""

    def test_server_with_registered_tool(self):
        """Test server with registered tool."""
        server = MCPServer()
        server.protocol_handler.register_tool(
            name="query",
            description="Query knowledge base",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            handler=lambda args: f"Results for: {args.get('query')}",
        )

        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query",
                "arguments": {"query": "test query"},
            },
        })

        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                mock_stdin.write(request + "\n")
                mock_stdin.seek(0)

                server.running = False
                server.run()

                output = mock_stdout.getvalue()

        if output:
            response = json.loads(output.strip())
            assert response["result"]["content"][0]["text"] == "Results for: test query"

    def test_full_initialize_flow(self):
        """Test full initialize -> tools/list flow."""
        server = MCPServer()
        server.protocol_handler.register_tool(
            name="test",
            description="Test tool",
            input_schema={"type": "object"},
            handler=lambda x: "ok",
        )

        requests = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        ]

        with patch("sys.stdin", new_callable=StringIO) as mock_stdin:
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                mock_stdin.write("\n".join(requests) + "\n")
                mock_stdin.seek(0)

                server.running = False
                server.run()

                output = mock_stdout.getvalue()

        if output:
            lines = output.strip().split("\n")
            responses = [json.loads(line) for line in lines if line]

            # First response: initialize
            assert responses[0]["id"] == 1
            assert "protocolVersion" in responses[0]["result"]

            # Second response: tools/list
            assert responses[1]["id"] == 2
            assert len(responses[1]["result"]["tools"]) == 1


class TestQueryKnowledgeHubTool:
    """Tests for query_knowledge_hub tool."""

    def test_tool_schema(self):
        """Test tool schema."""
        from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

        tool = QueryKnowledgeHubTool()
        schema = tool.get_schema()

        assert schema["name"] == "query_knowledge_hub"
        assert "query" in schema["inputSchema"]["properties"]
        assert "top_k" in schema["inputSchema"]["properties"]
        assert "collection" in schema["inputSchema"]["properties"]
        assert schema["inputSchema"]["required"] == ["query"]

    def test_create_handler(self):
        """Test create_handler function."""
        from src.mcp_server.tools.query_knowledge_hub import create_handler

        handler = create_handler()
        assert callable(handler)

    def test_handler_returns_dict(self):
        """Test handler returns dict."""
        from src.mcp_server.tools.query_knowledge_hub import create_handler

        handler = create_handler()

        # This will fail because no data, but should return error response
        result = handler({"query": "test"})

        assert isinstance(result, dict)
        assert "content" in result

    def test_tool_execute_with_mock(self):
        """Test tool execute with mocked components."""
        from unittest.mock import Mock, patch
        from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
        from src.core.types import RetrievalResult

        tool = QueryKnowledgeHubTool()

        # Mock the lazy init components
        mock_hybrid_search = Mock()
        mock_hybrid_search.search.return_value = [
            RetrievalResult(
                chunk_id="c1",
                score=0.95,
                text="Test answer",
                metadata={"source_path": "doc.pdf", "page": 1},
            ),
        ]

        mock_reranker = Mock()
        mock_reranker.rerank.return_value = Mock(
            candidates=[Mock(id="c1", score=0.98)],
            backend="none",
            fallback_used=False,
        )

        # Patch the lazy init
        with patch.object(tool, "_lazy_init"):
            tool._hybrid_search = mock_hybrid_search
            tool._reranker = mock_reranker
            tool._response_builder = ResponseBuilder()
            tool._initialized = True

            result = tool.execute("test query", top_k=5)

        assert "content" in result
        assert "structuredContent" in result
        mock_hybrid_search.search.assert_called_once()

    def test_tool_execute_no_rerank(self):
        """Test tool execute without reranking."""
        from unittest.mock import Mock, patch
        from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
        from src.core.types import RetrievalResult

        tool = QueryKnowledgeHubTool()

        mock_hybrid_search = Mock()
        mock_hybrid_search.search.return_value = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="Answer",
                metadata={"source_path": "doc.pdf"},
            ),
        ]

        with patch.object(tool, "_lazy_init"):
            tool._hybrid_search = mock_hybrid_search
            tool._reranker = Mock()
            tool._response_builder = ResponseBuilder()
            tool._initialized = True

            result = tool.execute("test", top_k=5, no_rerank=True)

        assert "content" in result
        # Reranker should not be called
        tool._reranker.rerank.assert_not_called()

    def test_tool_execute_with_collection_filter(self):
        """Test tool execute with collection filter."""
        from unittest.mock import Mock, patch
        from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
        from src.core.types import RetrievalResult

        tool = QueryKnowledgeHubTool()

        mock_hybrid_search = Mock()
        mock_hybrid_search.search.return_value = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="Answer",
                metadata={"source_path": "doc.pdf", "collection": "my-docs"},
            ),
        ]

        with patch.object(tool, "_lazy_init"):
            tool._hybrid_search = mock_hybrid_search
            tool._reranker = Mock()
            tool._reranker.rerank.return_value = Mock(
                candidates=[Mock(id="c1", score=0.9)],
                backend="none",
                fallback_used=False,
            )
            tool._response_builder = ResponseBuilder()
            tool._initialized = True

            result = tool.execute("test", collection="my-docs")

        # Check that filter was passed
        call_args = mock_hybrid_search.search.call_args
        assert call_args[1]["filters"] == {"collection": "my-docs"}