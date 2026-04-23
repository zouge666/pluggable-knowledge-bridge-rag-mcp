"""
DeepEval Evaluator 占位实现。

基于 DeepEval 框架的评估（占位，未来扩展）。
"""

from typing import List

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
    EvaluatorUnavailableError,
)


class DeepEvalEvaluator(BaseEvaluator):
    """
    DeepEval Evaluator 占位实现。

    基于 DeepEval 框架的 LLM 评估，支持 LLM-as-Judge 模式。
    当前为占位实现，未来扩展。
    """

    def __init__(self, settings) -> None:
        """
        初始化 DeepEval Evaluator。

        Args:
            settings: Evaluation 配置。
        """
        if hasattr(settings, "evaluation"):
            settings = settings.evaluation

        self.metrics = getattr(settings, "metrics", None) or [
            "answer_correctness",
            "hallucination",
        ]

    def evaluate(self, eval_input: EvalInput) -> EvalResult:
        """
        评估单个查询。

        当前为占位实现。
        """
        raise EvaluatorUnavailableError(
            "DeepEvalEvaluator not implemented yet",
            evaluator="deepeval",
        )

    def get_evaluator_name(self) -> str:
        """获取评估器名称。"""
        return "deepeval"

    def get_supported_metrics(self) -> List[str]:
        """获取支持的指标列表。"""
        return [
            "answer_correctness",
            "hallucination",
            "bias",
            "toxicity",
        ]
