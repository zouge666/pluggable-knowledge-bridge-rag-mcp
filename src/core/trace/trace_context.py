"""
追踪上下文。

用于记录请求各阶段的处理数据。
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TraceContext:
    """
    追踪上下文。

    记录一次请求（query/ingestion）的完整处理过程。
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_type: str = "query"  # "query" | "ingestion"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    stages: List[Dict[str, Any]] = field(default_factory=list)
    _start_time: float = field(default_factory=time.perf_counter, repr=False)
    _finished: bool = False
    _finish_time: Optional[float] = None

    def record_stage(
        self,
        stage_name: str,
        elapsed_ms: Optional[float] = None,
        method: Optional[str] = None,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录一个处理阶段。

        Args:
            stage_name: 阶段名称（如 "dense_retrieval", "rerank"）。
            elapsed_ms: 阶段耗时（毫秒）。
            method: 具体方法（如 "bm25", "chroma"）。
            provider: 提供者（如 "openai", "azure"）。
            details: 阶段详情。
        """
        stage: Dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
        }
        if elapsed_ms is not None:
            stage["elapsed_ms"] = elapsed_ms
        if method:
            stage["method"] = method
        if provider:
            stage["provider"] = provider
        if details:
            stage["details"] = details

        self.stages.append(stage)

    def finish(self) -> None:
        """标记追踪结束，计算总耗时。"""
        if not self._finished:
            self._finish_time = time.perf_counter()
            self.finished_at = datetime.now().isoformat()
            self._finished = True

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """
        获取指定阶段或总耗时。

        Args:
            stage_name: 阶段名称。如果为 None，返回总耗时。

        Returns:
            耗时（毫秒）。
        """
        if stage_name is not None:
            # 查找指定阶段的耗时
            for stage in self.stages:
                if stage.get("stage") == stage_name:
                    return stage.get("elapsed_ms", 0.0)
            return 0.0

        # 返回总耗时
        if self._finished and self._finish_time is not None:
            return (self._finish_time - self._start_time) * 1000

        # 如果未 finish，返回当前耗时
        return (time.perf_counter() - self._start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        result: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "stages": self.stages,
        }

        if self._finished and hasattr(self, "finished_at"):
            result["finished_at"] = self.finished_at
            result["total_elapsed_ms"] = round(self.elapsed_ms(), 2)

        return result
