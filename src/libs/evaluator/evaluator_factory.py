"""
Evaluator 工厂。

根据配置创建对应的 Evaluator 实例。
"""

from typing import TYPE_CHECKING, List

from src.core.settings import EvaluationSettings, Settings
from src.libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError
from src.libs.evaluator.custom_evaluator import CustomEvaluator

if TYPE_CHECKING:
    pass


class EvaluatorFactory:
    """
    Evaluator 工厂类。

    根据配置动态创建 Evaluator 实例，支持多种评估框架。
    """

    @classmethod
    def create(cls, settings: Settings) -> BaseEvaluator:
        """
        根据配置创建 Evaluator 实例。

        Args:
            settings: 应用配置。

        Returns:
            BaseEvaluator: Evaluator 实例。

        Raises:
            EvaluatorError: 不支持的评估器或配置错误。
        """
        eval_settings = settings.evaluation

        # 如果未启用，返回 Custom Evaluator
        if not eval_settings.enabled:
            return CustomEvaluator()

        provider = eval_settings.provider.lower() if eval_settings.provider else "custom"

        if provider == "custom":
            return CustomEvaluator()
        elif provider == "ragas":
            from src.libs.evaluator.ragas_evaluator import RagasEvaluator

            return RagasEvaluator(eval_settings)
        elif provider == "deepeval":
            from src.libs.evaluator.deepeval_evaluator import DeepEvalEvaluator

            return DeepEvalEvaluator(eval_settings)
        else:
            raise EvaluatorError(
                f"Unsupported Evaluator provider: {provider}",
                evaluator=provider,
            )

    @classmethod
    def create_from_settings(cls, eval_settings: EvaluationSettings) -> BaseEvaluator:
        """
        从 EvaluationSettings 创建 Evaluator 实例。

        Args:
            eval_settings: Evaluation 配置。

        Returns:
            BaseEvaluator: Evaluator 实例。
        """
        if not eval_settings.enabled:
            return CustomEvaluator()

        provider = eval_settings.provider.lower() if eval_settings.provider else "custom"

        if provider == "custom":
            return CustomEvaluator()
        elif provider == "ragas":
            from src.libs.evaluator.ragas_evaluator import RagasEvaluator

            return RagasEvaluator(eval_settings)
        elif provider == "deepeval":
            from src.libs.evaluator.deepeval_evaluator import DeepEvalEvaluator

            return DeepEvalEvaluator(eval_settings)
        else:
            raise EvaluatorError(
                f"Unsupported Evaluator provider: {provider}",
                evaluator=provider,
            )

    @classmethod
    def create_composite(
        cls,
        evaluators: List[BaseEvaluator],
    ) -> "CompositeEvaluator":
        """
        创建组合评估器（并行执行多个评估器）。

        Args:
            evaluators: 评估器列表。

        Returns:
            CompositeEvaluator: 组合评估器。
        """
        from src.libs.evaluator.composite_evaluator import CompositeEvaluator

        return CompositeEvaluator(evaluators)

    @classmethod
    def get_supported_providers(cls) -> list:
        """获取支持的 provider 列表。"""
        return ["custom", "ragas", "deepeval"]
