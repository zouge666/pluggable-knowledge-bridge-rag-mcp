#!/usr/bin/env python3
"""
Main Entry Point

This is the entry point for the MCP Server that exposes
RAG capabilities through the Model Context Protocol.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from core.settings import ConfigurationError, load_settings
from observability.logger import get_logger


def main() -> int:
    """Main entry point for the MCP Server."""
    logger = get_logger("main")

    try:
        settings = load_settings()
        logger.info(f"Configuration loaded successfully")
        logger.info(f"LLM provider: {settings.llm.provider}/{settings.llm.model}")
        logger.info(f"Embedding provider: {settings.embedding.provider}/{settings.embedding.model}")
        logger.info(f"Vector store: {settings.vector_store.provider}")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    print("Modular RAG MCP Server - Starting...")
    print("Project structure initialized successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
