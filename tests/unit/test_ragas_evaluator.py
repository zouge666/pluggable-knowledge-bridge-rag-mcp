"""
Ragas Evaluator 单元测试。

验证 Ragas 评估器实现正确性。
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.core.settings import EvaluationSettings, Settings
from src.libs.evaluator.base_evaluator import EvalInput, EvalResult, EvaluatorError, EvaluatorUnavailableError
from src.libs.evaluator.ragas_evaluator import RagasEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory


class TestRagasEvaluatorInit:
    """RagasEvaluator 初始化测试。"""

    def test_init_with_settings(self):
        """应该正确初始化。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness", "answer_relevancy"],
        )

        evaluator = RagasEvaluator(settings)

        assert evaluator.get_evaluator_name() == "ragas"
        assert "faithfulness" in evaluator.metrics
        assert "answer_relevancy" in evaluator.metrics

    def test_init_with_full_settings(self):
        """应该能处理完整 Settings 对象。"""
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True,
                provider="ragas",
                metrics=["context_precision"],
            )
        )

        evaluator = RagasEvaluator(settings)

        assert evaluator.metrics == ["context_precision"]

    def test_default_metrics(self):
        """默认指标应该包含 faithfulness, answer_relevancy, context_precision。"""
        # 当配置中没有 Ragas 支持的指标时，使用 Ragas 默认指标
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=[],  # 空列表，使用默认
        )

        evaluator = RagasEvaluator(settings)

        assert "faithfulness" in evaluator.metrics
        assert "answer_relevancy" in evaluator.metrics
        assert "context_precision" in evaluator.metrics

    def test_get_supported_metrics(self):
        """应该返回支持的指标列表。"""
        settings = EvaluationSettings(enabled=True, provider="ragas")
        evaluator = RagasEvaluator(settings)

        metrics = evaluator.get_supported_metrics()

        assert "faithfulness" in metrics
        assert "answer_relevancy" in metrics
        assert "context_precision" in metrics
        assert "context_recall" in metrics


class TestRagasEvaluatorAvailability:
    """Ragas 可用性测试。"""

    def test_is_available_when_ragas_installed(self):
        """Ragas 已安装时 is_available 应该返回 True。"""
        settings = EvaluationSettings(enabled=True, provider="ragas")
        evaluator = RagasEvaluator(settings)

        # 检查 Ragas 是否实际安装
        try:
            import ragas
            assert evaluator.is_available() is True
        except ImportError:
            assert evaluator.is_available() is False

    def test_raises_error_when_ragas_not_installed(self):
        """Ragas 未安装时应该抛出 EvaluatorUnavailableError。"""
        settings = EvaluationSettings(enabled=True, provider="ragas")
        evaluator = RagasEvaluator(settings)

        # 如果 Ragas 未安装，评估应该抛出错误
        if not evaluator.is_available():
            eval_input = EvalInput(
                query="测试查询",
                retrieved_ids=["id1"],
                retrieved_texts=["文本1"],
                generated_answer="生成的答案",
            )

            with pytest.raises(EvaluatorUnavailableError) as exc_info:
                evaluator.evaluate(eval_input)

            assert "Ragas is not installed" in str(exc_info.value)


class TestRagasEvaluatorValidation:
    """RagasEvaluator 输入验证测试。"""

    @pytest.mark.skipif(
        not RagasEvaluator(EvaluationSettings(enabled=True, provider="ragas")).is_available(),
        reason="Ragas not installed"
    )
    def test_empty_query_returns_zero_metrics(self):
        """空查询应该返回零分。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness"],
        )
        evaluator = RagasEvaluator(settings)

        eval_input = EvalInput(
            query="",
            retrieved_ids=["id1"],
            retrieved_texts=["文本1"],
            generated_answer="答案",
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["faithfulness"] == 0.0
        assert "error" in result.details

    @pytest.mark.skipif(
        not RagasEvaluator(EvaluationSettings(enabled=True, provider="ragas")).is_available(),
        reason="Ragas not installed"
    )
    def test_missing_generated_answer_for_faithfulness(self):
        """计算 faithfulness 时缺少 generated_answer 应该返回零分。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness"],
        )
        evaluator = RagasEvaluator(settings)

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1"],
            retrieved_texts=["文本1"],
            generated_answer=None,  # 缺少答案
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["faithfulness"] == 0.0
        assert "error" in result.details

    @pytest.mark.skipif(
        not RagasEvaluator(EvaluationSettings(enabled=True, provider="ragas")).is_available(),
        reason="Ragas not installed"
    )
    def test_missing_retrieved_texts_for_context_metrics(self):
        """计算 context 指标时缺少 retrieved_texts 应该返回零分。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["context_precision"],
        )
        evaluator = RagasEvaluator(settings)

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1"],
            retrieved_texts=None,  # 缺少文本
            generated_answer="答案",
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["context_precision"] == 0.0
        assert "error" in result.details


class TestRagasEvaluatorMetrics:
    """RagasEvaluator 指标计算测试。"""

    @pytest.mark.skipif(
        not RagasEvaluator(EvaluationSettings(enabled=True, provider="ragas")).is_available(),
        reason="Ragas not installed"
    )
    def test_evaluate_returns_correct_structure(self):
        """评估结果应该包含正确的结构。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness"],
        )
        evaluator = RagasEvaluator(settings)

        eval_input = EvalInput(
            query="什么是机器学习？",
            retrieved_ids=["id1", "id2"],
            retrieved_texts=[
                "机器学习是人工智能的一个分支。",
                "它使用算法从数据中学习。",
            ],
            generated_answer="机器学习是人工智能的分支，使用算法从数据中学习。",
        )

        result = evaluator.evaluate(eval_input)

        assert isinstance(result, EvalResult)
        assert result.evaluator_name == "ragas"
        assert "faithfulness" in result.metrics
        assert result.elapsed_ms is not None
        assert result.elapsed_ms >= 0

    @pytest.mark.skipif(
        not RagasEvaluator(EvaluationSettings(enabled=True, provider="ragas")).is_available(),
        reason="Ragas not installed"
    )
    def test_evaluate_with_golden_texts(self):
        """评估时包含标准答案文本。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["context_recall"],
        )
        evaluator = RagasEvaluator(settings)

        eval_input = EvalInput(
            query="什么是机器学习？",
            retrieved_ids=["id1"],
            retrieved_texts=["机器学习是人工智能的一个分支。"],
            generated_answer="机器学习是AI的分支。",
            golden_texts=["机器学习是让计算机从数据中学习的技术。"],
        )

        result = evaluator.evaluate(eval_input)

        assert "context_recall" in result.metrics
        assert result.details["has_golden"] is True

    @pytest.mark.skipif(
        not RagasEvaluator(EvaluationSettings(enabled=True, provider="ragas")).is_available(),
        reason="Ragas not installed"
    )
    def test_multiple_metrics(self):
        """应该能同时计算多个指标。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness", "answer_relevancy", "context_precision"],
        )
        evaluator = RagasEvaluator(settings)

        eval_input = EvalInput(
            query="什么是深度学习？",
            retrieved_ids=["id1", "id2"],
            retrieved_texts=[
                "深度学习使用神经网络。",
                "它是机器学习的子领域。",
            ],
            generated_answer="深度学习是使用神经网络的机器学习子领域。",
        )

        result = evaluator.evaluate(eval_input)

        assert "faithfulness" in result.metrics
        assert "answer_relevancy" in result.metrics
        assert "context_precision" in result.metrics

    @pytest.mark.skipif(
        not RagasEvaluator(EvaluationSettings(enabled=True, provider="ragas")).is_available(),
        reason="Ragas not installed"
    )
    def test_unsupported_metric_returns_zero(self):
        """不支持的指标应该返回零分。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["unsupported_metric"],
        )
        evaluator = RagasEvaluator(settings)

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1"],
            retrieved_texts=["文本"],
            generated_answer="答案",
        )

        result = evaluator.evaluate(eval_input)

        assert result.metrics["unsupported_metric"] == 0.0


class TestRagasEvaluatorBatch:
    """RagasEvaluator 批量评估测试。"""

    @pytest.mark.skipif(
        not RagasEvaluator(EvaluationSettings(enabled=True, provider="ragas")).is_available(),
        reason="Ragas not installed"
    )
    def test_evaluate_batch(self):
        """批量评估应该返回正确的报告。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness"],
        )
        evaluator = RagasEvaluator(settings)

        eval_inputs = [
            EvalInput(
                query="查询1",
                retrieved_ids=["id1"],
                retrieved_texts=["文本1"],
                generated_answer="答案1",
            ),
            EvalInput(
                query="查询2",
                retrieved_ids=["id2"],
                retrieved_texts=["文本2"],
                generated_answer="答案2",
            ),
        ]

        report = evaluator.evaluate_batch(eval_inputs)

        assert report.total_queries == 2
        assert "faithfulness" in report.avg_metrics
        assert report.elapsed_ms is not None


class TestEvaluatorFactoryWithRagas:
    """EvaluatorFactory 与 Ragas 集成测试。"""

    def test_create_ragas_evaluator(self):
        """工厂应该能创建 Ragas Evaluator。"""
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True,
                provider="ragas",
                metrics=["faithfulness"],
            )
        )

        evaluator = EvaluatorFactory.create(settings)

        assert evaluator.get_evaluator_name() == "ragas"

    def test_create_ragas_with_custom_metrics(self):
        """工厂应该能创建带自定义指标的 Ragas Evaluator。"""
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True,
                provider="ragas",
                metrics=["answer_relevancy", "context_precision"],
            )
        )

        evaluator = EvaluatorFactory.create(settings)

        assert "answer_relevancy" in evaluator.metrics
        assert "context_precision" in evaluator.metrics


class TestRagasEvaluatorMockLLM:
    """使用 Mock LLM 的 RagasEvaluator 测试。"""

    def test_evaluate_with_mock_llm_client(self):
        """应该能使用注入的 Mock LLM 客户端。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness"],
        )

        # 创建 Mock LLM 客户端
        mock_llm = MagicMock()

        evaluator = RagasEvaluator(settings, llm_client=mock_llm)

        # 如果 Ragas 未安装，跳过
        if not evaluator.is_available():
            pytest.skip("Ragas not installed")

        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1"],
            retrieved_texts=["测试文本"],
            generated_answer="测试答案",
        )

        result = evaluator.evaluate(eval_input)

        assert isinstance(result, EvalResult)
        assert result.evaluator_name == "ragas"


class TestRagasEvaluatorErrorHandling:
    """RagasEvaluator 错误处理测试。"""

    @pytest.mark.skipif(
        not RagasEvaluator(EvaluationSettings(enabled=True, provider="ragas")).is_available(),
        reason="Ragas not installed"
    )
    def test_metric_error_does_not_affect_others(self):
        """单个指标错误不应影响其他指标。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness", "answer_relevancy"],
        )
        evaluator = RagasEvaluator(settings)

        # 提供完整输入
        eval_input = EvalInput(
            query="测试查询",
            retrieved_ids=["id1"],
            retrieved_texts=["测试文本"],
            generated_answer="测试答案",
        )

        result = evaluator.evaluate(eval_input)

        # 两个指标都应该有值（即使可能为 0）
        assert "faithfulness" in result.metrics
        assert "answer_relevancy" in result.metrics
