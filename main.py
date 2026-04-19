#!/usr/bin/env python3
"""
Modular RAG MCP Server - Main Entry Point

This is the entry point for the MCP Server that exposes
RAG capabilities through the Model Context Protocol.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def main() -> int:
    """Main entry point for the MCP Server."""
    print("Modular RAG MCP Server - Starting...")
    print("Project structure initialized successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
