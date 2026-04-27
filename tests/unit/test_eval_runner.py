"""
EvalRunner 单元测试。

验证评估执行器实现正确性。
"""

import json
import pytest
import tempfile
from pathlib import Path

from src.core.query_engine.hybrid_search import FakeHybridSearch
from src.core.types import RetrievalResult
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.eval_runner import (
    EvalRunner,
    TestCase,
    GoldenTestSet,
)


class TestTestCase:
    """TestCase 测试。"""

    def test_create_test_case(self):
        """应该能创建测试用例。"""
        tc = TestCase(
            query="测试查询",
            expected_chunk_ids=["id1", "id2"],
            expected_sources=["doc.pdf"],
        )

        assert tc.query == "测试查询"
        assert tc.expected_chunk_ids == ["id1", "id2"]
        assert tc.expected_sources == ["doc.pdf"]

    def test_to_dict(self):
        """应该能转换为字典。"""
        tc = TestCase(
            query="测试查询",
            expected_chunk_ids=["id1"],
            expected_sources=["doc.pdf"],
            expected_answer="预期答案",
            metadata={"category": "test"},
        )

        data = tc.to_dict()

        assert data["query"] == "测试查询"
        assert data["expected_chunk_ids"] == ["id1"]
        assert data["expected_sources"] == ["doc.pdf"]
        assert data["expected_answer"] == "预期答案"
        assert data["metadata"]["category"] == "test"


class TestGoldenTestSet:
    """GoldenTestSet 测试。"""

    def test_create_test_set(self):
        """应该能创建测试集。"""
        test_cases = [
            TestCase(query="q1", expected_chunk_ids=["id1"]),
            TestCase(query="q2", expected_chunk_ids=["id2"]),
        ]

        test_set = GoldenTestSet(
            name="test_set",
            version="1.0",
            test_cases=test_cases,
            description="测试描述",
        )

        assert test_set.name == "test_set"
        assert test_set.version == "1.0"
        assert len(test_set.test_cases) == 2
        assert test_set.description == "测试描述"

    def test_to_dict(self):
        """应该能转换为字典。"""
        test_cases = [
            TestCase(query="q1", expected_chunk_ids=["id1"]),
        ]

        test_set = GoldenTestSet(
            name="test_set",
            version="1.0",
            test_cases=test_cases,
        )

        data = test_set.to_dict()

        assert data["name"] == "test_set"
        assert data["version"] == "1.0"
        assert len(data["test_cases"]) == 1

    def test_from_json(self):
        """应该能从 JSON 文件加载。"""
        # 创建临时 JSON 文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "name": "test_set",
                "version": "2.0",
                "description": "测试描述",
                "test_cases": [
                    {
                        "query": "查询1",
                        "expected_chunk_ids": ["id1", "id2"],
                        "expected_sources": ["doc1.pdf"],
                    },
                    {
                        "query": "查询2",
                        "expected_chunk_ids": ["id3"],
                        "expected_answer": "答案",
                    },
                ],
            }, f)
            temp_path = f.name

        try:
            test_set = GoldenTestSet.from_json(temp_path)

            assert test_set.name == "test_set"
            assert test_set.version == "2.0"
            assert len(test_set.test_cases) == 2
            assert test_set.test_cases[0].query == "查询1"
            assert test_set.test_cases[1].expected_answer == "答案"
        finally:
            Path(temp_path).unlink()

    def test_from_json_file_not_found(self):
        """文件不存在时应该抛出错误。"""
        with pytest.raises(ValueError) as exc_info:
            GoldenTestSet.from_json("nonexistent.json")

        assert "not found" in str(exc_info.value)

    def test_from_json_invalid_json(self):
        """JSON 格式错误时应该抛出错误。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json {")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                GoldenTestSet.from_json(temp_path)

            assert "Failed to parse JSON" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()


class TestEvalRunner:
    """EvalRunner 测试。"""

    def test_run_with_fake_search(self):
        """使用 FakeHybridSearch 运行评估。"""
        # 创建 FakeHybridSearch 返回预设结果
        results = [
            RetrievalResult(
                chunk_id="id1",
                score=0.9,
                text="文本1",
                metadata={},
            ),
            RetrievalResult(
                chunk_id="id2",
                score=0.8,
                text="文本2",
                metadata={},
            ),
        ]
        hybrid_search = FakeHybridSearch(results=results)
        evaluator = CustomEvaluator()

        # 创建测试集
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "name": "test",
                "version": "1.0",
                "test_cases": [
                    {
                        "query": "查询1",
                        "expected_chunk_ids": ["id1"],
                    },
                    {
                        "query": "查询2",
                        "expected_chunk_ids": ["id3"],  # 不在结果中
                    },
                ],
            }, f)
            temp_path = f.name

        try:
            runner = EvalRunner(
                hybrid_search=hybrid_search,
                evaluator=evaluator,
                top_k=10,
            )

            report = runner.run(temp_path)

            assert report.total_queries == 2
            assert "hit_rate" in report.avg_metrics
            # 第一个查询命中，第二个未命中，平均 hit_rate = 0.5
            assert report.avg_metrics["hit_rate"] == 0.5
        finally:
            Path(temp_path).unlink()

    def test_run_with_details(self):
        """运行评估并返回详细信息。"""
        results = [
            RetrievalResult(
                chunk_id="id1",
                score=0.9,
                text="文本1",
                metadata={},
            ),
        ]
        hybrid_search = FakeHybridSearch(results=results)
        evaluator = CustomEvaluator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "name": "detailed_test",
                "version": "1.0",
                "test_cases": [
                    {
                        "query": "查询1",
                        "expected_chunk_ids": ["id1"],
                    },
                ],
            }, f)
            temp_path = f.name

        try:
            runner = EvalRunner(
                hybrid_search=hybrid_search,
                evaluator=evaluator,
                top_k=5,
            )

            result = runner.run_with_details(temp_path)

            assert result["test_set_name"] == "detailed_test"
            assert result["total_queries"] == 1
            assert len(result["detailed_results"]) == 1
            assert result["detailed_results"][0]["query"] == "查询1"
            assert "hit_rate" in result["detailed_results"][0]["metrics"]
        finally:
            Path(temp_path).unlink()

    def test_run_with_collection_filter(self):
        """使用集合过滤运行评估。"""
        results = [
            RetrievalResult(
                chunk_id="id1",
                score=0.9,
                text="文本1",
                metadata={"collection": "test_collection"},
            ),
        ]
        hybrid_search = FakeHybridSearch(results=results)
        evaluator = CustomEvaluator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "name": "test",
                "version": "1.0",
                "test_cases": [
                    {
                        "query": "查询1",
                        "expected_chunk_ids": ["id1"],
                    },
                ],
            }, f)
            temp_path = f.name

        try:
            runner = EvalRunner(
                hybrid_search=hybrid_search,
                evaluator=evaluator,
                top_k=10,
            )

            report = runner.run(temp_path, collection="test_collection")

            # 验证 search 被调用且传入了 collection 过滤
            assert len(hybrid_search.search_calls) == 1
            assert hybrid_search.search_calls[0]["filters"]["collection"] == "test_collection"
        finally:
            Path(temp_path).unlink()

    def test_run_empty_test_set(self):
        """空测试集应该返回空报告。"""
        hybrid_search = FakeHybridSearch(results=[])
        evaluator = CustomEvaluator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "name": "empty",
                "version": "1.0",
                "test_cases": [],
            }, f)
            temp_path = f.name

        try:
            runner = EvalRunner(
                hybrid_search=hybrid_search,
                evaluator=evaluator,
                top_k=10,
            )

            report = runner.run(temp_path)

            assert report.total_queries == 0
            assert len(report.avg_metrics) == 0
        finally:
            Path(temp_path).unlink()


class TestEvalRunnerWithMRR:
    """EvalRunner MRR 计算测试。"""

    def test_mrr_calculation(self):
        """应该正确计算 MRR。"""
        # 结果顺序：id1, id2, id3
        results = [
            RetrievalResult(chunk_id="id1", score=0.9, text="t1", metadata={}),
            RetrievalResult(chunk_id="id2", score=0.8, text="t2", metadata={}),
            RetrievalResult(chunk_id="id3", score=0.7, text="t3", metadata={}),
        ]
        hybrid_search = FakeHybridSearch(results=results)
        evaluator = CustomEvaluator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "name": "mrr_test",
                "version": "1.0",
                "test_cases": [
                    {
                        "query": "预期在第一位",
                        "expected_chunk_ids": ["id1"],  # MRR = 1.0
                    },
                    {
                        "query": "预期在第二位",
                        "expected_chunk_ids": ["id2"],  # MRR = 0.5
                    },
                    {
                        "query": "预期在第三位",
                        "expected_chunk_ids": ["id3"],  # MRR = 0.33
                    },
                ],
            }, f)
            temp_path = f.name

        try:
            runner = EvalRunner(
                hybrid_search=hybrid_search,
                evaluator=evaluator,
                top_k=10,
            )

            report = runner.run(temp_path)

            # 平均 MRR = (1.0 + 0.5 + 0.333...) / 3 ≈ 0.611
            expected_mrr = (1.0 + 0.5 + 1/3) / 3
            assert abs(report.avg_metrics["mrr"] - expected_mrr) < 0.01
        finally:
            Path(temp_path).unlink()
