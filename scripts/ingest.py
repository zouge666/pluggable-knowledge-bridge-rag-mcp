#!/usr/bin/env python3
"""
Data Ingestion Script

Usage:
    python scripts/ingest.py --path <path> --collection <name> [--force]
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the knowledge base")
    parser.add_argument("--path", required=True, help="Path to document or directory")
    parser.add_argument("--collection", default="default", help="Collection name")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    args = parser.parse_args()

    print(f"Ingesting: {args.path}")
    print(f"Collection: {args.collection}")
    print(f"Force: {args.force}")
    print("Note: Ingestion pipeline not yet implemented (A1 skeleton only)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
