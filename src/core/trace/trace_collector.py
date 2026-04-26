"""
Trace Collector。

收集并持久化 trace 数据。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.trace.trace_context import TraceContext

logger = logging.getLogger("core.trace.collector")


class TraceCollector:
    """
    Trace 收集器。

    收集 TraceContext 并持久化到文件。
    """

    def __init__(
        self,
        output_path: str = "logs/traces.jsonl",
        enabled: bool = True,
    ):
        """
        初始化收集器。

        Args:
            output_path: 输出文件路径。
            enabled: 是否启用收集。
        """
        self._output_path = Path(output_path)
        self._enabled = enabled

        # 确保目录存在
        if self._enabled:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def collect(self, trace: TraceContext) -> None:
        """
        收集并持久化 trace。

        Args:
            trace: TraceContext 实例。
        """
        if not self._enabled:
            return

        try:
            # 确保 trace 已 finish
            if not trace._finished:
                trace.finish()

            # 序列化为 JSON
            trace_dict = trace.to_dict()
            trace_json = json.dumps(trace_dict, ensure_ascii=False)

            # 追加写入文件
            with open(self._output_path, "a", encoding="utf-8") as f:
                f.write(trace_json + "\n")

            logger.debug(f"Collected trace: {trace.trace_id}")

        except Exception as e:
            logger.error(f"Failed to collect trace: {e}")

    def collect_batch(self, traces: List[TraceContext]) -> None:
        """
        批量收集 trace。

        Args:
            traces: TraceContext 列表。
        """
        for trace in traces:
            self.collect(trace)

    def read_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        读取已收集的 traces。

        Args:
            trace_type: 过滤 trace 类型（可选）。
            limit: 最大返回数量。

        Returns:
            trace 字典列表。
        """
        if not self._output_path.exists():
            return []

        traces = []
        try:
            with open(self._output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        trace_dict = json.loads(line)

                        # 过滤 trace_type
                        if trace_type and trace_dict.get("trace_type") != trace_type:
                            continue

                        traces.append(trace_dict)

                        if len(traces) >= limit:
                            break

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Failed to read traces: {e}")

        # 按时间倒序
        traces.reverse()
        return traces

    def clear(self) -> None:
        """清空 trace 文件。"""
        if self._output_path.exists():
            self._output_path.unlink()
            logger.info(f"Cleared trace file: {self._output_path}")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取 trace 统计信息。

        Returns:
            统计信息字典。
        """
        if not self._output_path.exists():
            return {
                "total_traces": 0,
                "query_traces": 0,
                "ingestion_traces": 0,
            }

        total = 0
        query_count = 0
        ingestion_count = 0

        try:
            with open(self._output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        trace_dict = json.loads(line)
                        total += 1

                        if trace_dict.get("trace_type") == "query":
                            query_count += 1
                        elif trace_dict.get("trace_type") == "ingestion":
                            ingestion_count += 1

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Failed to get trace stats: {e}")

        return {
            "total_traces": total,
            "query_traces": query_count,
            "ingestion_traces": ingestion_count,
        }


# 全局默认收集器实例
_default_collector: Optional[TraceCollector] = None


def get_trace_collector() -> TraceCollector:
    """获取默认 Trace 收集器。"""
    global _default_collector
    if _default_collector is None:
        _default_collector = TraceCollector()
    return _default_collector


def set_trace_collector(collector: TraceCollector) -> None:
    """设置默认 Trace 收集器。"""
    global _default_collector
    _default_collector = collector
