"""
Cross-Encoder Reranker 实现。

基于 Cross-Encoder 模型的重排序，使用 sentence-transformers 进行精细打分。
"""

import time
from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    RerankerError,
    RerankerUnavailableError,
)

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder Reranker 实现。

    使用 Cross-Encoder 模型对候选项进行精细打分排序。
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEFAULT_MAX_LENGTH = 512

    def __init__(
        self,
        settings=None,
        model: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> None:
        """
        初始化 Cross-Encoder Reranker。

        Args:
            settings: Rerank 配置（可选）。
            model: 模型名称（可选）。
            max_length: 最大序列长度（可选）。
        """
        if settings:
            if hasattr(settings, "rerank"):
                settings = settings.rerank
            self.model_name = getattr(settings, "model", None) or self.DEFAULT_MODEL
            self.top_k = getattr(settings, "top_k", None) or 5
        else:
            self.model_name = model or self.DEFAULT_MODEL
            self.top_k = 5

        self.max_length = max_length or self.DEFAULT_MAX_LENGTH

        # 延迟加载模型
        self._model = None
        self._available: Optional[bool] = None

    def _get_model(self):
        """获取或加载 Cross-Encoder 模型。"""
        if self._model is None:
            if CrossEncoder is None:
                raise ImportError(
                    "sentence-transformers package is required. "
                    "Install it with: pip install sentence-transformers"
                )
            try:
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                )
                self._available = True
            except Exception as e:
                self._available = False
                raise RerankerUnavailableError(
                    f"Failed to load Cross-Encoder model: {e}",
                    backend="cross_encoder",
                )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> RerankResult:
        """
        对候选项进行重排序。

        Args:
            query: 查询文本。
            candidates: 候选项列表。
            top_k: 返回结果数量。
            trace: 追踪上下文。

        Returns:
            RerankResult: 重排序结果。
        """
        start_time = time.time()

        # 确定返回数量
        k = top_k or self.top_k or len(candidates)

        # 空候选项处理
        if not candidates:
            return RerankResult(
                candidates=[],
                backend="cross_encoder",
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        try:
            model = self._get_model()

            # 构造 query-document 对
            pairs = [(query, candidate.text) for candidate in candidates]

            # 批量打分
            scores = model.predict(pairs)

            # 按分数降序排序
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # 构建重排序后的候选项
            reranked = []
            for candidate, score in scored_candidates[:k]:
                reranked.append(RerankCandidate(
                    id=candidate.id,
                    text=candidate.text,
                    score=float(score),
                    metadata=candidate.metadata,
                ))

            elapsed_ms = (time.time() - start_time) * 1000

            # 记录追踪
            if trace:
                trace.record_stage(
                    stage_name="rerank",
                    elapsed_ms=elapsed_ms,
                    method="cross_encoder",
                    provider="cross_encoder",
                    details={
                        "model": self.model_name,
                        "candidate_count": len(candidates),
                        "result_count": len(reranked),
                    },
                )

            return RerankResult(
                candidates=reranked,
                backend="cross_encoder",
                elapsed_ms=elapsed_ms,
            )

        except ImportError as e:
            raise RerankerUnavailableError(
                str(e),
                backend="cross_encoder",
                original_error=e,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            # 检查是否是模型加载错误
            if "not found" in str(e).lower() or "load" in str(e).lower():
                self._available = False
                raise RerankerUnavailableError(
                    f"Failed to load model: {e}",
                    backend="cross_encoder",
                    original_error=e,
                )

            raise RerankerError(
                f"Rerank failed: {e}",
                backend="cross_encoder",
                original_error=e,
            )

    def get_backend_name(self) -> str:
        """获取后端名称。"""
        return "cross_encoder"

    def is_available(self) -> bool:
        """检查 Cross-Encoder 是否可用。"""
        if self._available is not None:
            return self._available

        try:
            # 尝试加载模型
            self._get_model()
            return True
        except Exception:
            self._available = False
            return False
