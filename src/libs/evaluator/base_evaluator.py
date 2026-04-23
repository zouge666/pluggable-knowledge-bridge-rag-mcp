"""
Evaluator 抽象基类。

定义统一的评估接口，支持多种评估框架（Ragas/DeepEval/Custom）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalInput:
    """评估输入。"""

    query: str  # 查询文本
    retrieved_ids: List[str]  # 检索返回的文档 ID 列表
    retrieved_texts: Optional[List[str]] = None  # 检索返回的文档文本（可选）
    golden_ids: Optional[List[str]] = None  # 标准答案的文档 ID（可选）
    golden_texts: Optional[List[str]] = None  # 标准答案的文本（可选）
    generated_answer: Optional[str] = None  # LLM 生成的答案（可选）
    golden_answer: Optional[str] = None  # 标准答案（可选）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "query": self.query,
            "retrieved_count": len(self.retrieved_ids),
            "has_golden": self.golden_ids is not None,
            "has_generated": self.generated_answer is not None,
        }


@dataclass
class EvalResult:
    """评估结果。"""

    metrics: Dict[str, float]  # 指标名称 -> 分数
    evaluator_name: str  # 评估器名称
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    elapsed_ms: Optional[float] = None  # 耗时（毫秒）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "evaluator": self.evaluator_name,
            "metrics": self.metrics,
            "elapsed_ms": self.elapsed_ms,
        }

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """获取指定指标的值。"""
        return self.metrics.get(name, default)


@dataclass
class EvalReport:
    """评估报告（多个查询的汇总）。"""

    results: List[EvalResult]  # 每个查询的评估结果
    avg_metrics: Dict[str, float]  # 平均指标
    total_queries: int  # 总查询数
    elapsed_ms: Optional[float] = None  # 总耗时

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "total_queries": self.total_queries,
            "avg_metrics": self.avg_metrics,
            "elapsed_ms": self.elapsed_ms,
        }


class BaseEvaluator(ABC):
    """
    Evaluator 抽象基类。

    所有 Evaluator 实现（Ragas/DeepEval/Custom）都必须实现此接口。
    """

    @abstractmethod
    def evaluate(
        self,
        eval_input: EvalInput,
    ) -> EvalResult:
        """
        评估单个查询。

        Args:
            eval_input: 评估输入。

        Returns:
            EvalResult: 评估结果。

        Raises:
            EvaluatorError: 评估失败时抛出。
        """
        pass

    def evaluate_batch(
        self,
        eval_inputs: List[EvalInput],
    ) -> EvalReport:
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

    @abstractmethod
    def get_evaluator_name(self) -> str:
        """获取评估器名称。"""
        pass

    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """获取支持的指标列表。"""
        pass


class EvaluatorError(Exception):
    """Evaluator 错误基类。"""

    def __init__(
        self,
        message: str,
        evaluator: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.evaluator = evaluator
        self.original_error = original_error

        error_parts = []
        if evaluator:
            error_parts.append(f"[{evaluator}]")
        error_parts.append(message)

        super().__init__(" ".join(error_parts))


class EvaluatorConfigError(EvaluatorError):
    """Evaluator 配置错误。"""
    pass


class EvaluatorUnavailableError(EvaluatorError):
    """Evaluator 不可用错误。"""
    pass
