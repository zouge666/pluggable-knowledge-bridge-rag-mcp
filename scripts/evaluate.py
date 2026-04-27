#!/usr/bin/env python3
"""
评估运行脚本。

运行 RAG 评估并输出指标报告。

Usage:
    python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json
    python scripts/evaluate.py --verbose
    python scripts/evaluate.py --output report.json
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main() -> int:
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="运行 RAG 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test-set",
        type=str,
        default="tests/fixtures/golden_test_set.json",
        help="黄金测试集文件路径 (默认: tests/fixtures/golden_test_set.json)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="配置文件路径 (默认: config/settings.yaml)",
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="限定检索的集合名称",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="检索返回数量 (默认: 10)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告文件路径 (JSON 格式)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出",
    )

    args = parser.parse_args()

    # 加载配置
    try:
        from src.core.settings import load_settings
        settings = load_settings(args.config)
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return 1

    # 创建评估器
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory
    evaluator = EvaluatorFactory.create(settings)

    # 检查是否有 HybridSearch 可用
    try:
        from src.core.query_engine.hybrid_search import HybridSearch, FakeHybridSearch
        from src.core.query_engine.dense_retriever import DenseRetriever
        from src.core.query_engine.sparse_retriever import SparseRetriever
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        from src.ingestion.storage.bm25_indexer import BM25Indexer

        # 创建组件
        embedding_client = EmbeddingFactory.create(settings)
        vector_store = VectorStoreFactory.create(settings)
        bm25_indexer = BM25Indexer()

        # 创建 HybridSearch
        hybrid_search = HybridSearch(
            settings=settings,
            dense_retriever=DenseRetriever(
                embedding_client=embedding_client,
                vector_store=vector_store,
            ),
            sparse_retriever=SparseRetriever(
                bm25_indexer=bm25_indexer,
                vector_store=vector_store,
            ),
        )

    except Exception as e:
        print(f"⚠️ HybridSearch 初始化失败，使用 Mock 模式: {e}")
        from src.core.query_engine.hybrid_search import FakeHybridSearch
        hybrid_search = FakeHybridSearch()

    # 创建 EvalRunner
    from src.observability.evaluation.eval_runner import EvalRunner
    runner = EvalRunner(
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        top_k=args.top_k,
    )

    # 运行评估
    print(f"\n🚀 开始评估...")
    print(f"   测试集: {args.test_set}")
    print(f"   评估器: {evaluator.get_evaluator_name()}")
    print(f"   Top-K: {args.top_k}")
    if args.collection:
        print(f"   集合: {args.collection}")
    print()

    try:
        if args.verbose:
            result = runner.run_with_details(args.test_set, args.collection)
            print_report_verbose(result)
            report_data = result
        else:
            report = runner.run(args.test_set, args.collection)
            print_report(report)
            report_data = {
                "total_queries": report.total_queries,
                "avg_metrics": report.avg_metrics,
                "elapsed_ms": report.elapsed_ms,
            }

        # 保存报告
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"\n📄 报告已保存到: {args.output}")

        return 0

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def print_report(report):
    """打印评估报告。"""
    print("\n" + "=" * 50)
    print("📊 评估报告")
    print("=" * 50)
    print(f"总查询数: {report.total_queries}")
    print(f"总耗时: {report.elapsed_ms:.2f} ms")
    print()
    print("平均指标:")
    for name, value in report.avg_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("=" * 50)


def print_report_verbose(result):
    """打印详细评估报告。"""
    print("\n" + "=" * 60)
    print("📊 详细评估报告")
    print("=" * 60)
    print(f"测试集: {result['test_set_name']} (v{result['test_set_version']})")
    print(f"总查询数: {result['total_queries']}")
    print(f"总耗时: {result['elapsed_ms']:.2f} ms")
    print()
    print("平均指标:")
    for name, value in result['avg_metrics'].items():
        print(f"  {name}: {value:.4f}")
    print()
    print("-" * 60)
    print("详细结果:")
    print("-" * 60)

    for detail in result['detailed_results']:
        print(f"\n[{detail['index'] + 1}] 查询: {detail['query']}")
        print(f"    预期 IDs: {detail['expected_ids']}")
        print(f"    指标:")
        for name, value in detail['metrics'].items():
            print(f"      {name}: {value:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    sys.exit(main())