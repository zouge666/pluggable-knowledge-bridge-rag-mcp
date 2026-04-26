"""
Logging utilities.

Provides structured logging with stderr output for MCP server compatibility.
Supports JSON Lines format for trace persistence.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    JSON 格式日志格式化器。

    输出 JSON Lines 格式的日志。
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录为 JSON。

        Args:
            record: 日志记录。

        Returns:
            JSON 字符串。
        """
        log_dict: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 添加额外字段
        if hasattr(record, "trace_id"):
            log_dict["trace_id"] = record.trace_id

        if hasattr(record, "trace_type"):
            log_dict["trace_type"] = record.trace_type

        if hasattr(record, "stage"):
            log_dict["stage"] = record.stage

        if hasattr(record, "elapsed_ms"):
            log_dict["elapsed_ms"] = record.elapsed_ms

        # 添加异常信息
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict, ensure_ascii=False)


def get_logger(name: str = "rag_mcp", level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Logs are written to stderr to avoid polluting stdout
    (which is used for MCP protocol messages).

    Args:
        name: Logger name.
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    return logger


def get_trace_logger(
    name: str = "rag_mcp.trace",
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Get a trace logger with JSON Lines format.

    Args:
        name: Logger name.
        level: Log level. Defaults to INFO.

    Returns:
        Configured logger instance with JSON formatter.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    return logger


def write_trace(
    trace_dict: Dict[str, Any],
    output_path: str = "logs/traces.jsonl",
) -> None:
    """
    Write trace dictionary to JSON Lines file.

    Args:
        trace_dict: Trace dictionary to write.
        output_path: Output file path.
    """
    output_file = Path(output_path)

    # 确保目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 追加写入
    trace_json = json.dumps(trace_dict, ensure_ascii=False)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(trace_json + "\n")


class TraceLoggerAdapter(logging.LoggerAdapter):
    """
    Trace 日志适配器。

    自动添加 trace_id 和 trace_type 到日志记录。
    """

    def __init__(
        self,
        logger: logging.Logger,
        trace_id: str,
        trace_type: str = "query",
    ):
        """
        初始化适配器。

        Args:
            logger: 底层 logger。
            trace_id: 追踪 ID。
            trace_type: 追踪类型（query/ingestion）。
        """
        super().__init__(logger, {
            "trace_id": trace_id,
            "trace_type": trace_type,
        })

    def process(
        self,
        msg: str,
        kwargs: Dict[str, Any],
    ) -> tuple:
        """
        处理日志消息，添加额外字段。

        Args:
            msg: 日志消息。
            kwargs: 关键字参数。

        Returns:
            处理后的消息和参数。
        """
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs
