#!/usr/bin/env python3
"""
Query Script

Usage:
    python scripts/query.py --query <text> [--top-k N] [--verbose]
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the knowledge base")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"Query: {args.query}")
    print(f"Top-K: {args.top_k}")
    print("Note: Query engine not yet implemented (A1 skeleton only)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
