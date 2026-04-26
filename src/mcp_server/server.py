"""
MCP Server 入口。

遵循 MCP 协议规范：
- stdout 只输出 MCP JSON-RPC 消息
- stderr 用于日志和调试信息
- 使用 stdio transport（标准输入/输出通信）
"""

import json
import sys
import logging
from typing import Any, Dict, Optional

from src.mcp_server.protocol_handler import ProtocolHandler

# 配置日志到 stderr（stdout 只用于 MCP 消息）
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.INFO)

# 创建 stderr handler
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.INFO)
stderr_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(stderr_handler)


class MCPServer:
    """
    MCP Server 实现。

    功能：
    1. 从 stdin 读取 JSON-RPC 请求
    2. 通过 ProtocolHandler 处理请求
    3. 将响应写入 stdout（纯 MCP 消息）
    4. 日志输出到 stderr
    """

    def __init__(self):
        """初始化 MCP Server。"""
        self.protocol_handler = ProtocolHandler()
        self.running = True
        logger.info("MCP Server initialized")

    def run(self):
        """
        运行 MCP Server 主循环。

        从 stdin 读取请求，处理后写入 stdout。
        """
        logger.info("MCP Server starting, waiting for requests...")

        try:
            while self.running:
                # 从 stdin 读取一行
                line = sys.stdin.readline()
                if not line:
                    # stdin 关闭，退出
                    logger.info("stdin closed, shutting down")
                    self.running = False
                    break

                # 去除换行符
                line = line.strip()
                if not line:
                    continue

                # 解析请求
                try:
                    request = json.loads(line)
                    logger.debug(f"Received request: {request.get('method', 'unknown')}")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    self._send_error(None, -32700, f"Parse error: {e}")
                    continue

                # 处理请求
                response = self.protocol_handler.handle_request(request)

                # 发送响应
                if response is not None:
                    self._send_response(response)

        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, shutting down")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

        logger.info("MCP Server stopped")

    def _send_response(self, response: Dict[str, Any]):
        """
        发送响应到 stdout。

        Args:
            response: 响应字典。
        """
        # 确保输出是纯 JSON，不带额外内容
        output = json.dumps(response, ensure_ascii=False)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
        logger.debug(f"Sent response for request id: {response.get('id')}")

    def _send_error(
        self,
        request_id: Optional[Any],
        code: int,
        message: str,
        data: Optional[Any] = None,
    ):
        """
        发送错误响应。

        Args:
            request_id: 请求 ID。
            code: 错误码。
            message: 错误消息。
            data: 错误数据（可选）。
        """
        error = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
        if data is not None:
            error["error"]["data"] = data

        self._send_response(error)

    def stop(self):
        """停止 MCP Server。"""
        self.running = False
        logger.info("MCP Server stop requested")


def main():
    """MCP Server 主入口。"""
    server = MCPServer()
    server.run()


if __name__ == "__main__":
    main()