"""
Evaluator 契约测试。

验证所有 Evaluator 实现遵循 BaseEvaluator 接口契约。
"""

import pytest
from typing import List

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
    EvalReport,
    EvaluatorError,
    EvaluatorConfigError,
    EvaluatorUnavailableError,
)
from src.libs.evaluator.custom_evaluator import CustomEvaluator


class TestEvalInput:
    """EvalInput 数据类测试。"""

    def test_eval_input_creation(self):
        """应该能创建评估输入。"""
        eval_input = EvalInput(
            query="What is machine learning?",
            retrieved_ids=["doc_1", "doc_2", "doc_3"],
            golden_ids=["doc_1"],
        )

        assert eval_input.query == "What is machine learning?"
        assert len(eval_input.retrieved_ids) == 3
        assert eval_input.golden_ids == ["doc_1"]

    def test_eval_input_to_dict(self):
        """应该能转换为字典。"""
        eval_input = EvalInput(
            query="Test query",
            retrieved_ids=["doc_1"],
            golden_ids=["doc_1"],
            generated_answer="Test answer",
        )

        data = eval_input.to_dict()

        assert data["query"] == "Test query"
        assert data["retrieved_count"] == 1
        assert data["has_golden"] is True
        assert data["has_generated"] is True

    def test_eval_input_optional_fields(self):
        """可选字段应该有默认值。"""
        eval_input = EvalInput(
            query="Test query",
            retrieved_ids=["doc_1"],
        )

        assert eval_input.golden_ids is None
        assert eval_input.golden_texts is None
        assert eval_input.generated_answer is None
        assert eval_input.golden_answer is None


class TestEvalResult:
    """EvalResult 数据类测试。"""

    def test_eval_result_creation(self):
        """应该能创建评估结果。"""
        result = EvalResult(
            metrics={"hit_rate": 1.0, "mrr": 0.5},
            evaluator_name="custom",
        )

        assert result.metrics["hit_rate"] == 1.0
        assert result.metrics["mrr"] == 0.5
        assert result.evaluator_name == "custom"

    def test_eval_result_to_dict(self):
        """应该能转换为字典。"""
        result = EvalResult(
            metrics={"hit_rate": 1.0},
            evaluator_name="custom",
            elapsed_ms=10.5,
        )

        data = result.to_dict()

        assert data["evaluator"] == "custom"
        assert data["metrics"]["hit_rate"] == 1.0
        assert data["elapsed_ms"] == 10.5

    def test_eval_result_get_metric(self):
        """应该能获取指定指标。"""
        result = EvalResult(
            metrics={"hit_rate": 1.0, "mrr": 0.5},
            evaluator_name="custom",
        )

        assert result.get_metric("hit_rate") == 1.0
        assert result.get_metric("mrr") == 0.5
        assert result.get_metric("unknown", default=0.0) == 0.0

    def test_eval_result_default_values(self):
        """默认值应该正确。"""
        result = EvalResult(
            metrics={"hit_rate": 1.0},
            evaluator_name="custom",
        )

        assert result.details == {}
        assert result.elapsed_ms is None


class TestEvalReport:
    """EvalReport 数据类测试。"""

    def test_eval_report_creation(self):
        """应该能创建评估报告。"""
        results = [
            EvalResult(metrics={"hit_rate": 1.0}, evaluator_name="custom"),
            EvalResult(metrics={"hit_rate": 0.0}, evaluator_name="custom"),
        ]

        report = EvalReport(
            results=results,
            avg_metrics={"hit_rate": 0.5},
            total_queries=2,
        )

        assert len(report.results) == 2
        assert report.avg_metrics["hit_rate"] == 0.5
        assert report.total_queries == 2

    def test_eval_report_to_dict(self):
        """应该能转换为字典。"""
        report = EvalReport(
            results=[],
            avg_metrics={"hit_rate": 0.5},
            total_queries=2,
            elapsed_ms=100.0,
        )

        data = report.to_dict()

        assert data["total_queries"] == 2
        assert data["avg_metrics"]["hit_rate"] == 0.5
        assert data["elapsed_ms"] == 100.0


class TestEvaluatorErrors:
    """Evaluator 错误类测试。"""

    def test_evaluator_error_basic(self):
        """基本错误信息。"""
        error = EvaluatorError("Something went wrong")

        assert "Something went wrong" in str(error)

    def test_evaluator_error_with_evaluator(self):
        """带评估器的错误信息。"""
        error = EvaluatorError("Something went wrong", evaluator="ragas")

        assert "[ragas]" in str(error)
        assert "Something went wrong" in str(error)

    def test_evaluator_error_with_original(self):
        """带原始错误的错误信息。"""
        original = ValueError("Original error")
        error = EvaluatorError("Wrapped", evaluator="custom", original_error=original)

        assert error.original_error == original

    def test_config_error_is_evaluator_error(self):
        """ConfigError 应该是 EvaluatorError 的子类。"""
        error = EvaluatorConfigError("Invalid config", evaluator="ragas")

        assert isinstance(error, EvaluatorError)

    def test_unavailable_error_is_evaluator_error(self):
        """UnavailableError 应该是 EvaluatorError 的子类。"""
        error = EvaluatorUnavailableError("Service unavailable", evaluator="ragas")

        assert isinstance(error, EvaluatorError)


class TestCustomEvaluatorContract:
    """CustomEvaluator 契约测试。"""

    @pytest.fixture
    def evaluator(self):
        """创建 CustomEvaluator 实例。"""
        return CustomEvaluator()

    def test_implements_base_evaluator(self, evaluator):
        """应该实现 BaseEvaluator 接口。"""
        assert isinstance(evaluator, BaseEvaluator)

    def test_get_evaluator_name(self, evaluator):
        """应该返回正确的评估器名称。"""
        assert evaluator.get_evaluator_name() == "custom"

    def test_get_supported_metrics(self, evaluator):
        """应该返回支持的指标列表。"""
        metrics = evaluator.get_supported_metrics()

        assert "hit_rate" in metrics
        assert "mrr" in metrics
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "recall@10" in metrics

    def test_evaluate_returns_eval_result(self, evaluator):
        """evaluate 应该返回 EvalResult。"""
        eval_input = EvalInput(
            query="Test query",
            retrieved_ids=["doc_1", "doc_2"],
            golden_ids=["doc_1"],
        )

        result = evaluator.evaluate(eval_input)

        assert isinstance(result, EvalResult)

    def test_evaluate_hit_rate_hit(self, evaluator):
        """应该正确计算 hit_rate（命中）。"""
        eval_input = EvalInput(
            query="Test query",
            retrieved_ids=["doc_1", "doc_2"],
            golden_ids=["doc_1"],
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["hit_rate"] == 1.0

    def test_evaluate_hit_rate_miss(self, evaluator):
        """应该正确计算 hit_rate（未命中）。"""
        eval_input = EvalInput(
            query="Test query",
            retrieved_ids=["doc_3", "doc_4"],
            golden_ids=["doc_1"],
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["hit_rate"] == 0.0

    def test_evaluate_mrr(self, evaluator):
        """应该正确计算 MRR。"""
        # 第一个位置命中
        eval_input_1 = EvalInput(
            query="Test query",
            retrieved_ids=["doc_1", "doc_2"],
            golden_ids=["doc_1"],
        )
        result_1 = evaluator.evaluate(eval_input_1)
        assert result_1.metrics["mrr"] == 1.0

        # 第二个位置命中
        eval_input_2 = EvalInput(
            query="Test query",
            retrieved_ids=["doc_2", "doc_1"],
            golden_ids=["doc_1"],
        )
        result_2 = evaluator.evaluate(eval_input_2)
        assert result_2.metrics["mrr"] == 0.5

    def test_evaluate_no_golden_ids(self, evaluator):
        """没有 golden_ids 时应该返回零值。"""
        eval_input = EvalInput(
            query="Test query",
            retrieved_ids=["doc_1", "doc_2"],
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["hit_rate"] == 0.0
        assert result.metrics["mrr"] == 0.0

    def test_evaluate_result_has_elapsed_ms(self, evaluator):
        """结果应该包含耗时。"""
        eval_input = EvalInput(
            query="Test query",
            retrieved_ids=["doc_1"],
            golden_ids=["doc_1"],
        )

        result = evaluator.evaluate(eval_input)

        assert result.elapsed_ms is not None
        assert result.elapsed_ms >= 0

    def test_evaluate_batch(self, evaluator):
        """应该支持批量评估。"""
        eval_inputs = [
            EvalInput(
                query="Query 1",
                retrieved_ids=["doc_1"],
                golden_ids=["doc_1"],
            ),
            EvalInput(
                query="Query 2",
                retrieved_ids=["doc_2"],
                golden_ids=["doc_1"],
            ),
        ]

        report = evaluator.evaluate_batch(eval_inputs)

        assert isinstance(report, EvalReport)
        assert report.total_queries == 2
        assert len(report.results) == 2
        assert "hit_rate" in report.avg_metrics

    def test_custom_k_values(self):
        """应该支持自定义 k 值。"""
        evaluator = CustomEvaluator(k_values=[1, 3, 5])

        metrics = evaluator.get_supported_metrics()

        assert "recall@1" in metrics
        assert "recall@3" in metrics
        assert "recall@5" in metrics
        assert "recall@10" not in metrics


class TestBaseEvaluatorContract:
    """BaseEvaluator 接口契约测试。

    任何实现 BaseEvaluator 的类都应该通过这些测试。
    """

    def test_custom_evaluator_satisfies_contract(self):
        """CustomEvaluator 应该满足契约。"""
        evaluator = CustomEvaluator()

        # 必须实现的方法
        assert hasattr(evaluator, "evaluate")
        assert hasattr(evaluator, "evaluate_batch")
        assert hasattr(evaluator, "get_evaluator_name")
        assert hasattr(evaluator, "get_supported_metrics")

        # 方法必须是可调用的
        assert callable(evaluator.evaluate)
        assert callable(evaluator.evaluate_batch)
        assert callable(evaluator.get_evaluator_name)
        assert callable(evaluator.get_supported_metrics)

        # get_evaluator_name 必须返回字符串
        name = evaluator.get_evaluator_name()
        assert isinstance(name, str)
        assert len(name) > 0

        # get_supported_metrics 必须返回字符串列表
        metrics = evaluator.get_supported_metrics()
        assert isinstance(metrics, list)
        assert all(isinstance(m, str) for m in metrics)

        # evaluate 必须接受 EvalInput 并返回 EvalResult
        eval_input = EvalInput(
            query="Test query",
            retrieved_ids=["doc_1"],
        )
        result = evaluator.evaluate(eval_input)
        assert isinstance(result, EvalResult)

        # evaluate_batch 必须返回 EvalReport
        report = evaluator.evaluate_batch([eval_input])
        assert isinstance(report, EvalReport)
