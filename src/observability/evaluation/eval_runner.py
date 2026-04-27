"""
EvalRunner - 评估执行器。

读取 golden test set，执行检索评估，产出评估报告。
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.types import RetrievalResult
from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
    EvalReport,
)


@dataclass
class TestCase:
    """单个测试用例。"""

    query: str  # 查询文本
    expected_chunk_ids: List[str]  # 预期的 chunk ID 列表
    expected_sources: Optional[List[str]] = None  # 预期的来源文件列表
    expected_answer: Optional[str] = None  # 预期的答案（可选）
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "query": self.query,
            "expected_chunk_ids": self.expected_chunk_ids,
            "expected_sources": self.expected_sources or [],
            "expected_answer": self.expected_answer,
            "metadata": self.metadata or {},
        }


@dataclass
class GoldenTestSet:
    """黄金测试集。"""

    name: str  # 测试集名称
    version: str  # 版本号
    test_cases: List[TestCase]  # 测试用例列表
    description: Optional[str] = None  # 描述

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }

    @classmethod
    def from_json(cls, path: str) -> "GoldenTestSet":
        """
        从 JSON 文件加载测试集。

        Args:
            path: JSON 文件路径。

        Returns:
            GoldenTestSet: 测试集对象。

        Raises:
            ValueError: 文件格式错误。
        """
        file_path = Path(path)

        if not file_path.exists():
            raise ValueError(f"Golden test set file not found: {path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        test_cases = []
        for tc_data in data.get("test_cases", []):
            test_cases.append(TestCase(
                query=tc_data.get("query", ""),
                expected_chunk_ids=tc_data.get("expected_chunk_ids", []),
                expected_sources=tc_data.get("expected_sources"),
                expected_answer=tc_data.get("expected_answer"),
                metadata=tc_data.get("metadata"),
            ))

        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", "1.0"),
            test_cases=test_cases,
            description=data.get("description"),
        )


class EvalRunner:
    """
    评估执行器。

    读取 golden test set，执行检索评估，产出评估报告。
    """

    def __init__(
        self,
        hybrid_search: HybridSearch,
        evaluator: BaseEvaluator,
        top_k: int = 10,
    ):
        """
        初始化 EvalRunner。

        Args:
            hybrid_search: 混合检索器。
            evaluator: 评估器。
            top_k: 检索返回数量。
        """
        self._hybrid_search = hybrid_search
        self._evaluator = evaluator
        self._top_k = top_k

    def run(
        self,
        test_set_path: str,
        collection: Optional[str] = None,
    ) -> EvalReport:
        """
        运行评估。

        Args:
            test_set_path: 测试集文件路径。
            collection: 可选的集合过滤。

        Returns:
            EvalReport: 评估报告。
        """
        start_time = time.time()

        # 加载测试集
        test_set = GoldenTestSet.from_json(test_set_path)

        # 执行评估
        eval_inputs = []
        results = []

        for test_case in test_set.test_cases:
            # 执行检索
            filters = {}
            if collection:
                filters["collection"] = collection

            retrieval_results = self._hybrid_search.search(
                query=test_case.query,
                top_k=self._top_k,
                filters=filters,
            )

            # 构建 EvalInput
            retrieved_ids = [r.chunk_id for r in retrieval_results]
            retrieved_texts = [r.text for r in retrieval_results]

            eval_input = EvalInput(
                query=test_case.query,
                retrieved_ids=retrieved_ids,
                retrieved_texts=retrieved_texts,
                golden_ids=test_case.expected_chunk_ids,
                golden_texts=[test_case.expected_answer] if test_case.expected_answer else None,
            )

            eval_inputs.append(eval_input)

            # 执行评估
            result = self._evaluator.evaluate(eval_input)
            results.append(result)

        elapsed_ms = (time.time() - start_time) * 1000

        # 计算平均指标
        all_metrics: Dict[str, List[float]] = {}
        for result in results:
            for name, value in result.metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)

        avg_metrics = {
            name: sum(values) / len(values)
            for name, values in all_metrics.items()
        }

        return EvalReport(
            results=results,
            avg_metrics=avg_metrics,
            total_queries=len(test_set.test_cases),
            elapsed_ms=elapsed_ms,
        )

    def run_with_details(
        self,
        test_set_path: str,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行评估并返回详细信息。

        Args:
            test_set_path: 测试集文件路径。
            collection: 可选的集合过滤。

        Returns:
            Dict: 包含评估报告和详细结果。
        """
        report = self.run(test_set_path, collection)

        # 加载测试集获取详细信息
        test_set = GoldenTestSet.from_json(test_set_path)

        detailed_results = []
        for i, (test_case, result) in enumerate(zip(test_set.test_cases, report.results)):
            detailed_results.append({
                "index": i,
                "query": test_case.query,
                "expected_ids": test_case.expected_chunk_ids,
                "metrics": result.metrics,
                "elapsed_ms": result.elapsed_ms,
            })

        return {
            "test_set_name": test_set.name,
            "test_set_version": test_set.version,
            "total_queries": report.total_queries,
            "avg_metrics": report.avg_metrics,
            "elapsed_ms": report.elapsed_ms,
            "detailed_results": detailed_results,
        }