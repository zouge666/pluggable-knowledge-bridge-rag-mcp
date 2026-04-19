#!/usr/bin/env python3
"""
Evaluation Script

Usage:
    python scripts/evaluate.py [--test-set <path>]
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--test-set", help="Path to test set JSON")
    args = parser.parse_args()

    print("Running evaluation...")
    print("Note: Evaluation module not yet implemented (A1 skeleton only)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
