"""
Custom Evaluator 实现。

提供自定义的轻量级评估指标（hit_rate、mrr）。
"""

from typing import List

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
)


class CustomEvaluator(BaseEvaluator):
    """
    Custom Evaluator 实现。

    提供轻量级的检索评估指标，不需要调用 LLM：
    - hit_rate: 检索结果中是否包含标准答案
    - mrr: 平均倒数排名
    - recall@k: 前 k 个结果中标准答案的比例
    """

    def __init__(self, k_values: List[int] = None) -> None:
        """
        初始化 Custom Evaluator。

        Args:
            k_values: 计算 recall@k 的 k 值列表，默认 [1, 5, 10]
        """
        self.k_values = k_values or [1, 5, 10]

    def evaluate(self, eval_input: EvalInput) -> EvalResult:
        """
        评估单个查询。

        Args:
            eval_input: 评估输入。

        Returns:
            EvalResult: 评估结果。
        """
        import time

        start_time = time.time()

        metrics = {}

        # 如果没有标准答案，返回空指标
        if not eval_input.golden_ids:
            return EvalResult(
                metrics={"hit_rate": 0.0, "mrr": 0.0},
                evaluator_name="custom",
                details={"error": "No golden_ids provided"},
            )

        retrieved_ids = eval_input.retrieved_ids
        golden_ids = set(eval_input.golden_ids)

        # 计算 hit_rate（检索结果中是否包含任一标准答案）
        hit = any(rid in golden_ids for rid in retrieved_ids)
        metrics["hit_rate"] = 1.0 if hit else 0.0

        # 计算 MRR（平均倒数排名）
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in golden_ids:
                mrr = 1.0 / rank
                break
        metrics["mrr"] = mrr

        # 计算 recall@k
        for k in self.k_values:
            top_k_ids = set(retrieved_ids[:k])
            recall = len(top_k_ids & golden_ids) / len(golden_ids) if golden_ids else 0.0
            metrics[f"recall@{k}"] = recall

        elapsed_ms = (time.time() - start_time) * 1000

        return EvalResult(
            metrics=metrics,
            evaluator_name="custom",
            details={
                "retrieved_count": len(retrieved_ids),
                "golden_count": len(golden_ids),
                "hit": hit,
            },
            elapsed_ms=elapsed_ms,
        )

    def get_evaluator_name(self) -> str:
        """获取评估器名称。"""
        return "custom"

    def get_supported_metrics(self) -> List[str]:
        """获取支持的指标列表。"""
        metrics = ["hit_rate", "mrr"]
        for k in self.k_values:
            metrics.append(f"recall@{k}")
        return metrics
