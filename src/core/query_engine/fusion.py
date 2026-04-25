"""
RRF Fusion - Reciprocal Rank Fusion 融合算法。

将多个检索器的排名结果融合为统一排序。
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult


@dataclass
class FusionResult:
    """
    融合结果。

    包含融合后的检索结果列表。
    """

    # 融合后的结果
    results: List[RetrievalResult] = field(default_factory=list)
    # 使用的融合方法
    method: str = "rrf"
    # RRF k 参数
    k: int = 60
    # 处理耗时（毫秒）
    elapsed_ms: Optional[float] = None

    def to_dict(self) -> Dict:
        """转换为字典。"""
        return {
            "results": [r.to_dict() for r in self.results],
            "method": self.method,
            "k": self.k,
            "elapsed_ms": self.elapsed_ms,
        }


class RRFFusion:
    """
    Reciprocal Rank Fusion 融合器。

    RRF 公式：
        score(d) = Σ 1 / (k + rank_i(d))

    其中：
    - d 是文档
    - rank_i(d) 是文档 d 在第 i 个检索器中的排名（从 1 开始）
    - k 是平滑参数（默认 60）

    特点：
    1. 简单高效，无需训练
    2. 对排名位置敏感，而非分数
    3. 能有效融合不同检索器的结果
    """

    def __init__(self, k: int = 60):
        """
        初始化 RRFFusion。

        Args:
            k: RRF 平滑参数（默认 60）。
        """
        self._k = k

    def fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int = 10,
        trace: Optional[TraceContext] = None,
    ) -> FusionResult:
        """
        融合稠密和稀疏检索结果。

        Args:
            dense_results: 稠密检索结果。
            sparse_results: 稀疏检索结果。
            top_k: 返回结果数量。
            trace: 追踪上下文。

        Returns:
            FusionResult: 融合结果。
        """
        start_time = time.time()

        # 1. 计算每个文档的 RRF 分数
        rrf_scores: Dict[str, float] = {}
        doc_info: Dict[str, RetrievalResult] = {}

        # 处理稠密检索结果
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (self._k + rank)
            if chunk_id not in doc_info:
                doc_info[chunk_id] = result

        # 处理稀疏检索结果
        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (self._k + rank)
            if chunk_id not in doc_info:
                doc_info[chunk_id] = result

        # 2. 按 RRF 分数排序
        sorted_items = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # 3. 构建融合结果
        fused_results = []
        for chunk_id, rrf_score in sorted_items[:top_k]:
            original = doc_info[chunk_id]
            fused_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=rrf_score,
                    text=original.text,
                    metadata=original.metadata,
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="rrf_fusion",
                elapsed_ms=elapsed_ms,
                method="rrf",
                details={
                    "k": self._k,
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "fused_count": len(fused_results),
                },
            )

        return FusionResult(
            results=fused_results,
            method="rrf",
            k=self._k,
            elapsed_ms=elapsed_ms,
        )

    def get_k(self) -> int:
        """获取 k 参数。"""
        return self._k


class FakeFusion:
    """
    Fake Fusion 用于测试。

    返回预设结果，不执行实际融合。
    """

    def __init__(self, results: Optional[List[RetrievalResult]] = None):
        """
        初始化 FakeFusion。

        Args:
            results: 预设的融合结果。
        """
        self._default_results = results or []
        self.fuse_calls: List[dict] = []

    def fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int = 10,
        trace: Optional[TraceContext] = None,
    ) -> FusionResult:
        """执行 fake 融合。"""
        self.fuse_calls.append({
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "top_k": top_k,
        })

        return FusionResult(
            results=self._default_results[:top_k],
            method="fake",
            k=60,
            elapsed_ms=0.1,
        )

    def get_k(self) -> int:
        """获取 k 参数。"""
        return 60
