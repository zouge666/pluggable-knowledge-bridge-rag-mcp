"""
Custom Evaluator 单元测试。

验证评估指标计算正确性。
"""

import pytest

from src.core.settings import EvaluationSettings, Settings
from src.libs.evaluator.base_evaluator import EvalInput, EvalResult, EvaluatorError
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory


class TestCustomEvaluator:
    """CustomEvaluator 测试。"""

    def test_hit_rate_when_hit(self):
        """检索结果包含标准答案时，hit_rate 应该为 1。"""
        evaluator = CustomEvaluator()

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1", "id2", "id3"],
            golden_ids=["id2"],
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["hit_rate"] == 1.0

    def test_hit_rate_when_miss(self):
        """检索结果不包含标准答案时，hit_rate 应该为 0。"""
        evaluator = CustomEvaluator()

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1", "id2", "id3"],
            golden_ids=["id4"],
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["hit_rate"] == 0.0

    def test_mrr_first_position(self):
        """标准答案在第一位时，MRR 应该为 1.0。"""
        evaluator = CustomEvaluator()

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1", "id2", "id3"],
            golden_ids=["id1"],
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["mrr"] == 1.0

    def test_mrr_second_position(self):
        """标准答案在第二位时，MRR 应该为 0.5。"""
        evaluator = CustomEvaluator()

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1", "id2", "id3"],
            golden_ids=["id2"],
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["mrr"] == 0.5

    def test_mrr_not_found(self):
        """标准答案不在检索结果中时，MRR 应该为 0。"""
        evaluator = CustomEvaluator()

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1", "id2", "id3"],
            golden_ids=["id4"],
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["mrr"] == 0.0

    def test_recall_at_k(self):
        """recall@k 应该正确计算。"""
        evaluator = CustomEvaluator(k_values=[1, 2, 3])

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1", "id2", "id3"],
            golden_ids=["id2", "id3"],
        )

        result = evaluator.evaluate(eval_input)

        # recall@1: 前 1 个中有 0 个标准答案
        assert result.metrics["recall@1"] == 0.0
        # recall@2: 前 2 个中有 1 个标准答案（id2）
        assert result.metrics["recall@2"] == 0.5
        # recall@3: 前 3 个中有 2 个标准答案
        assert result.metrics["recall@3"] == 1.0

    def test_no_golden_ids(self):
        """没有标准答案时，应该返回 0 分。"""
        evaluator = CustomEvaluator()

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1", "id2", "id3"],
            golden_ids=None,
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["hit_rate"] == 0.0
        assert result.metrics["mrr"] == 0.0

    def test_get_evaluator_name(self):
        """应该返回正确的评估器名称。"""
        evaluator = CustomEvaluator()

        assert evaluator.get_evaluator_name() == "custom"

    def test_get_supported_metrics(self):
        """应该返回支持的指标列表。"""
        evaluator = CustomEvaluator(k_values=[1, 5, 10])

        metrics = evaluator.get_supported_metrics()

        assert "hit_rate" in metrics
        assert "mrr" in metrics
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "recall@10" in metrics


class TestEvaluatorFactory:
    """EvaluatorFactory 测试。"""

    def test_get_supported_providers(self):
        """应该返回支持的 provider 列表。"""
        providers = EvaluatorFactory.get_supported_providers()

        assert "custom" in providers
        assert "ragas" in providers
        assert "deepeval" in providers

    def test_create_custom_evaluator(self):
        """应该能创建 Custom Evaluator。"""
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True,
                provider="custom",
            )
        )

        evaluator = EvaluatorFactory.create(settings)

        assert isinstance(evaluator, CustomEvaluator)

    def test_create_custom_when_disabled(self):
        """未启用时应该返回 Custom Evaluator。"""
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=False,
                provider="ragas",
            )
        )

        evaluator = EvaluatorFactory.create(settings)

        assert isinstance(evaluator, CustomEvaluator)

    def test_unsupported_provider_raises_error(self):
        """不支持的 provider 应该抛出错误。"""
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True,
                provider="unsupported",
            )
        )

        with pytest.raises(EvaluatorError) as exc_info:
            EvaluatorFactory.create(settings)

        assert "Unsupported Evaluator provider" in str(exc_info.value)

    def test_create_ragas_evaluator(self):
        """应该能创建 Ragas Evaluator。"""
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True,
                provider="ragas",
            )
        )

        evaluator = EvaluatorFactory.create(settings)

        assert evaluator.get_evaluator_name() == "ragas"


class TestEvalInput:
    """EvalInput 测试。"""

    def test_to_dict(self):
        """EvalInput 应该能转换为字典。"""
        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1", "id2"],
            golden_ids=["id1"],
        )

        data = eval_input.to_dict()

        assert data["query"] == "测试查询"
        assert data["retrieved_count"] == 2
        assert data["has_golden"] is True
        assert data["has_generated"] is False


class TestEvalResult:
    """EvalResult 测试。"""

    def test_to_dict(self):
        """EvalResult 应该能转换为字典。"""
        result = EvalResult(
            metrics={"hit_rate": 1.0, "mrr": 0.5},
            evaluator_name="custom",
            elapsed_ms=1.5,
        )

        data = result.to_dict()

        assert data["evaluator"] == "custom"
        assert data["metrics"]["hit_rate"] == 1.0
        assert data["elapsed_ms"] == 1.5

    def test_get_metric(self):
        """应该能获取指定指标的值。"""
        result = EvalResult(
            metrics={"hit_rate": 1.0, "mrr": 0.5},
            evaluator_name="custom",
        )

        assert result.get_metric("hit_rate") == 1.0
        assert result.get_metric("mrr") == 0.5
        assert result.get_metric("unknown", default=0.0) == 0.0


class TestBatchEvaluation:
    """批量评估测试。"""

    def test_evaluate_batch(self):
        """批量评估应该返回正确的报告。"""
        evaluator = CustomEvaluator()

        eval_inputs = [
            EvalInput(
                query="查询1",
                retrieved_ids=["id1", "id2"],
                golden_ids=["id1"],
            ),
            EvalInput(
                query="查询2",
                retrieved_ids=["id3", "id4"],
                golden_ids=["id5"],
            ),
        ]

        report = evaluator.evaluate_batch(eval_inputs)

        assert report.total_queries == 2
        assert "hit_rate" in report.avg_metrics
        # 平均 hit_rate = (1 + 0) / 2 = 0.5
        assert report.avg_metrics["hit_rate"] == 0.5

    def test_empty_batch(self):
        """空批量应该返回空报告。"""
        evaluator = CustomEvaluator()

        report = evaluator.evaluate_batch([])

        assert report.total_queries == 0
        assert len(report.avg_metrics) == 0
