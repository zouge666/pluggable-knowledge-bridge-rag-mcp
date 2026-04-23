"""
Composite Evaluator 实现。

组合多个评估器并行执行，汇总结果。
"""

from typing import Any, Dict, List

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
    EvalReport,
)


class CompositeEvaluator(BaseEvaluator):
    """
    Composite Evaluator 实现。

    组合多个评估器，并行执行并汇总结果。
    """

    def __init__(self, evaluators: List[BaseEvaluator]) -> None:
        """
        初始化 Composite Evaluator。

        Args:
            evaluators: 评估器列表。
        """
        self.evaluators = evaluators

    def evaluate(self, eval_input: EvalInput) -> EvalResult:
        """
        使用所有评估器评估单个查询。

        Args:
            eval_input: 评估输入。

        Returns:
            EvalResult: 汇总的评估结果。
        """
        import time

        start_time = time.time()

        all_metrics: Dict[str, float] = {}
        all_details: Dict[str, Any] = {}

        for evaluator in self.evaluators:
            result = evaluator.evaluate(eval_input)

            # 添加前缀避免指标名冲突
            prefix = evaluator.get_evaluator_name()
            for name, value in result.metrics.items():
                key = f"{prefix}.{name}"
                all_metrics[key] = value

            # 汇总详细信息
            all_details[prefix] = result.details

        elapsed_ms = (time.time() - start_time) * 1000

        return EvalResult(
            metrics=all_metrics,
            evaluator_name="composite",
            details=all_details,
            elapsed_ms=elapsed_ms,
        )

    def evaluate_batch(self, eval_inputs: List[EvalInput]) -> EvalReport:
        """
        批量评估多个查询。

        Args:
            eval_inputs: 评估输入列表。

        Returns:
            EvalReport: 评估报告。
        """
        import time

        start_time = time.time()
        results = []

        for eval_input in eval_inputs:
            result = self.evaluate(eval_input)
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
            total_queries=len(eval_inputs),
            elapsed_ms=elapsed_ms,
        )

    def get_evaluator_name(self) -> str:
        """获取评估器名称。"""
        return "composite"

    def get_supported_metrics(self) -> List[str]:
        """获取支持的指标列表。"""
        metrics = []
        for evaluator in self.evaluators:
            prefix = evaluator.get_evaluator_name()
            for name in evaluator.get_supported_metrics():
                metrics.append(f"{prefix}.{name}")
        return metrics
