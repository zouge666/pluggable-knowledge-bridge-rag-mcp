"""
Ragas Evaluator 实现。

基于 Ragas 框架的 RAG 评估，支持：
- Faithfulness（忠实度）
- Answer Relevancy（答案相关性）
- Context Precision（上下文精确度）
- Context Recall（上下文召回率）
"""

import time
from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
    EvaluatorError,
    EvaluatorUnavailableError,
)
from src.core.settings import EvaluationSettings


class RagasEvaluator(BaseEvaluator):
    """
    Ragas Evaluator 实现。

    基于 Ragas 框架的 RAG 评估，使用 LLM-as-Judge 模式评估：
    - Faithfulness: 答案是否基于检索上下文生成
    - Answer Relevancy: 答案与问题的相关性
    - Context Precision: 检索上下文的精确度
    - Context Recall: 检索上下文的召回率

    需要配置 LLM 用于评估。
    """

    def __init__(
        self,
        settings: EvaluationSettings,
        llm_client: Optional[Any] = None,
    ) -> None:
        """
        初始化 Ragas Evaluator。

        Args:
            settings: Evaluation 配置。
            llm_client: 可选的 LLM 客户端（用于依赖注入测试）。

        Raises:
            EvaluatorUnavailableError: Ragas 未安装时抛出。
        """
        if hasattr(settings, "evaluation"):
            settings = settings.evaluation

        self.settings = settings

        # Ragas 默认指标
        ragas_default_metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
        ]

        # 获取配置的指标，过滤出 Ragas 支持的指标
        config_metrics = getattr(settings, "metrics", None) or []
        supported = set(self.get_supported_metrics())

        # 如果配置中有 Ragas 支持的指标，使用配置的
        # 否则使用 Ragas 默认指标
        ragas_metrics = [m for m in config_metrics if m in supported]
        self.metrics = ragas_metrics if ragas_metrics else ragas_default_metrics

        # LLM 客户端（用于测试注入或从工厂创建）
        self._llm_client = llm_client

        # 检查 Ragas 是否可用
        self._ragas_available = self._check_ragas_available()

        # 初始化 Ragas 组件（延迟加载）
        self._ragas_metrics: Dict[str, Any] = {}
        self._initialized = False

    def _check_ragas_available(self) -> bool:
        """检查 Ragas 是否已安装。"""
        try:
            import ragas
            return True
        except ImportError:
            return False

    def _ensure_initialized(self) -> None:
        """确保 Ragas 组件已初始化。"""
        if self._initialized:
            return

        if not self._ragas_available:
            raise EvaluatorUnavailableError(
                "Ragas is not installed. Install with: pip install ragas",
                evaluator="ragas",
            )

        # 初始化 Ragas 指标
        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )

            self._ragas_metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }
            self._initialized = True
        except ImportError as e:
            raise EvaluatorUnavailableError(
                f"Failed to import Ragas metrics: {e}",
                evaluator="ragas",
            )

    def evaluate(self, eval_input: EvalInput) -> EvalResult:
        """
        评估单个查询。

        Args:
            eval_input: 评估输入。

        Returns:
            EvalResult: 评估结果。

        Raises:
            EvaluatorUnavailableError: Ragas 未安装时抛出。
            EvaluatorError: 评估失败时抛出。
        """
        start_time = time.time()

        # 确保 Ragas 已初始化
        self._ensure_initialized()

        # 验证必要输入
        if not eval_input.query:
            return EvalResult(
                metrics={m: 0.0 for m in self.metrics},
                evaluator_name="ragas",
                details={"error": "Empty query"},
                elapsed_ms=0.0,
            )

        # 检查必要字段
        if eval_input.generated_answer is None and "faithfulness" in self.metrics:
            return EvalResult(
                metrics={m: 0.0 for m in self.metrics},
                evaluator_name="ragas",
                details={"error": "generated_answer required for faithfulness metric"},
                elapsed_ms=0.0,
            )

        if eval_input.retrieved_texts is None and any(
            m in self.metrics for m in ["context_precision", "context_recall"]
        ):
            return EvalResult(
                metrics={m: 0.0 for m in self.metrics},
                evaluator_name="ragas",
                details={"error": "retrieved_texts required for context metrics"},
                elapsed_ms=0.0,
            )

        try:
            # 构建 Ragas 评估数据
            metrics_to_compute = self._compute_metrics(eval_input)

            elapsed_ms = (time.time() - start_time) * 1000

            return EvalResult(
                metrics=metrics_to_compute,
                evaluator_name="ragas",
                details={
                    "query_length": len(eval_input.query),
                    "retrieved_count": len(eval_input.retrieved_ids) if eval_input.retrieved_ids else 0,
                    "has_golden": eval_input.golden_texts is not None,
                    "metrics_requested": self.metrics,
                },
                elapsed_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            raise EvaluatorError(
                f"Ragas evaluation failed: {e}",
                evaluator="ragas",
                original_error=e,
            )

    def _compute_metrics(self, eval_input: EvalInput) -> Dict[str, float]:
        """
        使用 Ragas 计算指标。

        Args:
            eval_input: 评估输入。

        Returns:
            Dict[str, float]: 指标名称 -> 分数。
        """
        results = {}

        # 构建 Ragas SingleTurnSample
        try:
            from ragas import SingleTurnSample, EvaluationDataset
        except ImportError:
            # 旧版本 Ragas 兼容
            from ragas.dataset_schema import SingleTurnSample

        # 构建样本
        sample_data = {
            "user_input": eval_input.query,
        }

        # 添加可选字段
        if eval_input.retrieved_texts:
            sample_data["retrieved_contexts"] = eval_input.retrieved_texts

        if eval_input.generated_answer:
            sample_data["response"] = eval_input.generated_answer

        if eval_input.golden_texts:
            sample_data["reference"] = "\n".join(eval_input.golden_texts)

        sample = SingleTurnSample(**sample_data)

        # 计算每个指标
        for metric_name in self.metrics:
            if metric_name not in self._ragas_metrics:
                results[metric_name] = 0.0
                continue

            metric = self._ragas_metrics[metric_name]

            try:
                # 使用 Ragas 的单样本评估
                score = self._evaluate_single_metric(sample, metric)
                results[metric_name] = float(score) if score is not None else 0.0
            except Exception as e:
                # 单个指标失败不影响其他指标
                results[metric_name] = 0.0
                results[f"{metric_name}_error"] = str(e)

        return results

    def _evaluate_single_metric(self, sample: Any, metric: Any) -> Optional[float]:
        """
        评估单个指标。

        Args:
            sample: Ragas 样本。
            metric: Ragas 指标对象。

        Returns:
            Optional[float]: 指标分数。
        """
        # 尝试使用 Ragas 的异步评估方法
        try:
            import asyncio

            # 检查是否是异步方法
            if hasattr(metric, 'ascore'):
                # 使用异步方法
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 如果已有运行中的循环，创建任务
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                metric.ascore(sample)
                            )
                            return future.result(timeout=60)
                    else:
                        # 直接运行
                        return loop.run_until_complete(metric.ascore(sample))
                except RuntimeError:
                    # 没有事件循环，创建新的
                    return asyncio.run(metric.ascore(sample))
            elif hasattr(metric, 'score'):
                # 使用同步方法（旧版本）
                return metric.score(sample)
            else:
                return None

        except Exception:
            return None

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

    def is_available(self) -> bool:
        """检查 Ragas 是否可用。"""
        return self._ragas_available