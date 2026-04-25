#!/usr/bin/env python3
"""
Data Ingestion Script

Usage:
    python scripts/ingest.py --path <path> --collection <name> [--force]
    python scripts/ingest.py --path ./docs --collection my-docs
    python scripts/ingest.py --path ./doc.pdf --collection my-docs --force
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.core.settings import Settings
from src.ingestion.pipeline import IngestionPipeline, PipelineResult


def format_elapsed(ms: float) -> str:
    """Format elapsed time in human-readable format."""
    if ms < 1000:
        return f"{ms:.1f}ms"
    elif ms < 60000:
        return f"{ms/1000:.2f}s"
    else:
        return f"{ms/60000:.2f}m"


def print_result(result: PipelineResult, verbose: bool = False):
    """Print ingestion result."""
    if result.skipped:
        print(f"  ⏭️  Skipped (already processed)")
        return

    if result.success:
        print(f"  ✅ Success")
        print(f"     Chunks: {result.chunks_count}")
        print(f"     Images: {result.images_count}")
        print(f"     Time: {format_elapsed(result.elapsed_ms)}")

        if verbose and result.stages:
            print("     Stages:")
            for stage in result.stages:
                stage_name = stage.get("stage", "unknown")
                stage_time = stage.get("elapsed_ms", 0)
                print(f"       - {stage_name}: {format_elapsed(stage_time)}")
    else:
        print(f"  ❌ Failed: {result.error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ingest a single file
    python scripts/ingest.py --path ./document.pdf --collection my-docs

    # Ingest all files in a directory
    python scripts/ingest.py --path ./documents/ --collection my-docs

    # Force re-ingestion
    python scripts/ingest.py --path ./document.pdf --collection my-docs --force
        """,
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to document file or directory",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Collection name (default: default)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if file was already processed",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )
    args = parser.parse_args()

    # Resolve path
    input_path = Path(args.path).resolve()
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return 1

    # Collect files to process
    files = []
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        # Find all PDF files in directory
        files = list(input_path.glob("**/*.pdf"))
        if not files:
            print(f"No PDF files found in: {input_path}")
            return 0
    else:
        print(f"Error: Invalid path: {input_path}")
        return 1

    print(f"Ingesting {len(files)} file(s) into collection '{args.collection}'")
    print(f"Force: {args.force}")
    print()

    # Create pipeline
    settings = Settings()
    pipeline = IngestionPipeline(settings=settings)

    # Process files
    start_time = time.time()
    results = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
    }

    for file_path in files:
        print(f"📄 {file_path.name}")
        result = pipeline.ingest(
            file_path=file_path,
            collection=args.collection,
            force=args.force,
        )
        print_result(result, verbose=args.verbose)

        if result.skipped:
            results["skipped"] += 1
        elif result.success:
            results["success"] += 1
        else:
            results["failed"] += 1

    # Summary
    total_time = (time.time() - start_time) * 1000
    print()
    print("=" * 50)
    print(f"Summary:")
    print(f"  ✅ Success: {results['success']}")
    print(f"  ⏭️  Skipped: {results['skipped']}")
    print(f"  ❌ Failed: {results['failed']}")
    print(f"  ⏱️  Total time: {format_elapsed(total_time)}")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
