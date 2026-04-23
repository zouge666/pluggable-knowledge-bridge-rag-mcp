"""
Ragas Evaluator 占位实现。

基于 Ragas 框架的评估（占位，H1 阶段完善）。
"""

from typing import List

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
    EvaluatorUnavailableError,
)


class RagasEvaluator(BaseEvaluator):
    """
    Ragas Evaluator 占位实现。

    基于 Ragas 框架的 RAG 评估，支持：
    - Faithfulness（忠实度）
    - Answer Relevancy（答案相关性）
    - Context Precision（上下文精确度）
    - Context Recall（上下文召回率）

    当前为占位实现，H1 阶段完善。
    """

    def __init__(self, settings) -> None:
        """
        初始化 Ragas Evaluator。

        Args:
            settings: Evaluation 配置。
        """
        if hasattr(settings, "evaluation"):
            settings = settings.evaluation

        self.metrics = getattr(settings, "metrics", None) or [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
        ]

    def evaluate(self, eval_input: EvalInput) -> EvalResult:
        """
        评估单个查询。

        当前为占位实现。
        """
        raise EvaluatorUnavailableError(
            "RagasEvaluator not implemented yet. Will be completed in H1",
            evaluator="ragas",
        )

    def get_evaluator_name(self) -> str:
        """获取评估器名称。"""
        return "ragas"

    def get_supported_metrics(self) -> List[str]:
        """获取支持的指标列表。"""
        return [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
