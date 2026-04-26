#!/usr/bin/env python3
"""
Query Script - 在线查询命令行入口

Usage:
    python scripts/query.py --query "问题" [--top-k 10] [--collection xxx] [--verbose] [--no-rerank]
    python scripts/query.py --query "如何配置 Azure？" --top-k 5 --verbose
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.core.settings import Settings
from src.core.query_engine import (
    QueryProcessor,
    DenseRetriever,
    SparseRetriever,
    RRFFusion,
    HybridSearch,
    QueryReranker,
)
from src.core.types import RetrievalResult
from src.libs.embedding import EmbeddingFactory
from src.libs.vector_store import VectorStoreFactory
from src.libs.reranker import RerankerFactory
from src.ingestion.storage.bm25_indexer import BM25Indexer


def format_score(score: float) -> str:
    """Format score for display."""
    return f"{score:.4f}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for display."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def print_result(result: RetrievalResult, index: int, verbose: bool = False):
    """Print a single retrieval result."""
    score_str = format_score(result.score)
    text_summary = truncate_text(result.text, 80)

    # Extract metadata
    source = result.metadata.get("source_path", "unknown")
    page = result.metadata.get("page", "-")

    print(f"  [{index}] Score: {score_str}")
    print(f"      Text: {text_summary}")
    if verbose:
        print(f"      Source: {source}")
        print(f"      Page: {page}")
        print(f"      Chunk ID: {result.chunk_id}")
    print()


def print_results(
    results: list,
    title: str,
    verbose: bool = False,
    max_display: int = 5,
):
    """Print a list of results."""
    print(f"\n{title} ({len(results)} results):")
    print("-" * 50)

    if not results:
        print("  (无结果)")
        return

    display_count = min(len(results), max_display)
    for i, result in enumerate(results[:display_count], 1):
        print_result(result, i, verbose=verbose)

    if len(results) > display_count:
        print(f"  ... 还有 {len(results) - display_count} 个结果")


def check_data_available(settings: Settings) -> bool:
    """Check if there is data available for querying."""
    try:
        vector_store = VectorStoreFactory.create(settings)
        stats = vector_store.get_collection_stats()
        return stats.get("count", 0) > 0
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query the knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic query
    python scripts/query.py --query "如何配置 Azure？"

    # Query with more results
    python scripts/query.py --query "机器学习" --top-k 20

    # Query specific collection
    python scripts/query.py --query "部署流程" --collection my-docs

    # Verbose mode (show intermediate results)
    python scripts/query.py --query "API 文档" --verbose

    # Skip reranking
    python scripts/query.py --query "测试" --no-rerank
        """,
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Query text",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    parser.add_argument(
        "--collection", "-c",
        help="Limit search to specific collection",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including intermediate results",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking stage",
    )
    args = parser.parse_args()

    # Load settings
    settings = Settings()

    # Check if data is available
    if not check_data_available(settings):
        print("❌ 未找到相关文档")
        print()
        print("提示：请先运行 ingest.py 摄取数据：")
        print("  python scripts/ingest.py --path ./docs --collection my-docs")
        return 0

    print(f"🔍 查询: {args.query}")
    print(f"   Top-K: {args.top_k}")
    if args.collection:
        print(f"   Collection: {args.collection}")
    print()

    try:
        # Initialize components
        if args.verbose:
            print("初始化组件...")

        # Embedding client
        embedding_client = EmbeddingFactory.create(settings)

        # Vector store
        vector_store = VectorStoreFactory.create(settings)

        # BM25 indexer
        bm25_indexer = BM25Indexer(settings=settings)

        # Query processor
        query_processor = QueryProcessor()

        # Dense retriever
        dense_retriever = DenseRetriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )

        # Sparse retriever
        sparse_retriever = SparseRetriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )

        # RRF fusion
        fusion = RRFFusion()

        # Hybrid search
        hybrid_search = HybridSearch(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion,
        )

        # Build filters
        filters = {}
        if args.collection:
            filters["collection"] = args.collection

        # Execute hybrid search
        if args.verbose:
            print("执行混合检索...")

        results = hybrid_search.search(
            query=args.query,
            top_k=args.top_k * 2,  # 召回更多，供 rerank 筛选
            filters=filters if filters else None,
        )

        if args.verbose:
            print(f"  混合检索返回 {len(results)} 个候选结果")

        # Reranking
        if not args.no_rerank and results:
            if args.verbose:
                print("执行重排序...")

            reranker = QueryReranker(settings=settings)
            rerank_result = reranker.rerank(
                query=args.query,
                results=results,
                top_k=args.top_k,
            )

            # Convert back to RetrievalResult
            original_map = {r.chunk_id: r for r in results}
            final_results = []
            for c in rerank_result.candidates:
                original = original_map.get(c.id)
                if original:
                    final_results.append(
                        RetrievalResult(
                            chunk_id=c.id,
                            score=c.score,
                            text=original.text,
                            metadata=original.metadata,
                        )
                    )

            if args.verbose:
                backend = rerank_result.backend
                fallback = " (fallback)" if rerank_result.fallback_used else ""
                print(f"  重排序后端: {backend}{fallback}")
        else:
            final_results = results[:args.top_k]

        # Output results
        print()
        print("=" * 50)
        print(f"检索结果 ({len(final_results)} 个):")
        print("=" * 50)

        if not final_results:
            print()
            print("  未找到相关文档")
            print()
            return 0

        for i, result in enumerate(final_results, 1):
            print_result(result, i, verbose=args.verbose)

        # Summary
        print("-" * 50)
        print(f"共 {len(final_results)} 个结果")

        return 0

    except Exception as e:
        print(f"❌ 查询失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())