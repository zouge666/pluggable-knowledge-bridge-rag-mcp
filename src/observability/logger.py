"""
Logging utilities.

Provides structured logging with stderr output for MCP server compatibility.
"""

import logging
import sys
from typing import Optional


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
