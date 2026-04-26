"""
Trace service for Dashboard.

Reads and parses traces.jsonl for trace display.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TraceStage:
    """Trace stage data."""

    stage: str
    elapsed_ms: float
    method: Optional[str] = None
    provider: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class TraceRecord:
    """Trace record data."""

    trace_id: str
    trace_type: str
    total_elapsed_ms: float
    stages: List[TraceStage]
    timestamp: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TraceService:
    """Service for reading and parsing traces."""

    def __init__(self, trace_file: str = "logs/traces.jsonl"):
        """
        Initialize TraceService.

        Args:
            trace_file: Path to traces.jsonl file.
        """
        self._trace_file = Path(trace_file)

    def read_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[TraceRecord]:
        """
        Read traces from file.

        Args:
            trace_type: Filter by trace type (optional).
            limit: Maximum number of traces to return.

        Returns:
            List[TraceRecord]: Trace records.
        """
        if not self._trace_file.exists():
            return []

        traces = []
        try:
            with open(self._trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        record = self._parse_trace(data)

                        # Filter by type
                        if trace_type and record.trace_type != trace_type:
                            continue

                        traces.append(record)

                    except json.JSONDecodeError:
                        continue

            # Sort by timestamp (most recent first)
            traces.sort(key=lambda t: t.timestamp or "", reverse=True)

            return traces[:limit]

        except Exception:
            return []

    def _parse_trace(self, data: Dict[str, Any]) -> TraceRecord:
        """Parse trace data from JSON.

        Args:
            data: Raw trace data.

        Returns:
            TraceRecord: Parsed trace record.
        """
        stages = []
        for stage_data in data.get("stages", []):
            stages.append(TraceStage(
                stage=stage_data.get("stage", ""),
                elapsed_ms=stage_data.get("elapsed_ms", 0.0),
                method=stage_data.get("method"),
                provider=stage_data.get("provider"),
                details=stage_data.get("details"),
            ))

        # Determine success
        success = True
        error = None
        for stage in stages:
            if stage.stage == "error":
                success = False
                if stage.details:
                    error = stage.details.get("error")

        return TraceRecord(
            trace_id=data.get("trace_id", ""),
            trace_type=data.get("trace_type", ""),
            total_elapsed_ms=data.get("total_elapsed_ms", 0.0),
            stages=stages,
            timestamp=data.get("timestamp"),
            success=success,
            error=error,
            metadata=data.get("metadata"),
        )

    def get_trace_by_id(self, trace_id: str) -> Optional[TraceRecord]:
        """
        Get trace by ID.

        Args:
            trace_id: Trace ID.

        Returns:
            Optional[TraceRecord]: Trace record.
        """
        traces = self.read_traces()
        for trace in traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def get_ingestion_traces(self, limit: int = 50) -> List[TraceRecord]:
        """
        Get ingestion traces.

        Args:
            limit: Maximum number of traces.

        Returns:
            List[TraceRecord]: Ingestion traces.
        """
        return self.read_traces(trace_type="ingestion", limit=limit)

    def get_query_traces(self, limit: int = 50) -> List[TraceRecord]:
        """
        Get query traces.

        Args:
            limit: Maximum number of traces.

        Returns:
            List[TraceRecord]: Query traces.
        """
        return self.read_traces(trace_type="query", limit=limit)

    def get_stage_summary(self, trace: TraceRecord) -> Dict[str, float]:
        """
        Get stage elapsed time summary.

        Args:
            trace: Trace record.

        Returns:
            Dict[str, float]: Stage elapsed times.
        """
        summary = {}
        for stage in trace.stages:
            summary[stage.stage] = stage.elapsed_ms
        return summary

    def get_total_elapsed(self, trace: TraceRecord) -> float:
        """
        Get total elapsed time.

        Args:
            trace: Trace record.

        Returns:
            float: Total elapsed time in ms.
        """
        return trace.total_elapsed_ms