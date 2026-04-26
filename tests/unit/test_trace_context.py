"""
Unit tests for TraceContext enhancement and TraceCollector.
"""

import pytest
import json
import tempfile
from pathlib import Path
import time

from src.core.trace import TraceContext, TraceCollector


class TestTraceContextEnhancement:
    """Tests for enhanced TraceContext."""

    def test_trace_type_query(self):
        """Test trace_type defaults to query."""
        trace = TraceContext()
        assert trace.trace_type == "query"

    def test_trace_type_ingestion(self):
        """Test trace_type can be set to ingestion."""
        trace = TraceContext(trace_type="ingestion")
        assert trace.trace_type == "ingestion"

    def test_finish_marks_finished(self):
        """Test finish() marks trace as finished."""
        trace = TraceContext()
        assert not trace._finished

        trace.finish()
        assert trace._finished

    def test_finish_sets_finished_at(self):
        """Test finish() sets finished_at timestamp."""
        trace = TraceContext()
        trace.finish()

        assert hasattr(trace, "finished_at")
        assert trace.finished_at is not None

    def test_elapsed_ms_before_finish(self):
        """Test elapsed_ms() before finish returns current elapsed."""
        trace = TraceContext()
        time.sleep(0.01)  # 10ms

        elapsed = trace.elapsed_ms()
        assert elapsed > 0
        assert elapsed < 1000  # Should be less than 1 second

    def test_elapsed_ms_after_finish(self):
        """Test elapsed_ms() after finish returns total elapsed."""
        trace = TraceContext()
        time.sleep(0.02)  # 20ms
        trace.finish()

        elapsed = trace.elapsed_ms()
        assert elapsed > 15  # At least 15ms
        assert elapsed < 100  # Should be less than 100ms

    def test_elapsed_ms_for_stage(self):
        """Test elapsed_ms(stage_name) returns stage elapsed."""
        trace = TraceContext()
        trace.record_stage("test_stage", elapsed_ms=50.5)

        elapsed = trace.elapsed_ms("test_stage")
        assert elapsed == 50.5

    def test_elapsed_ms_for_missing_stage(self):
        """Test elapsed_ms(stage_name) returns 0 for missing stage."""
        trace = TraceContext()
        elapsed = trace.elapsed_ms("nonexistent")
        assert elapsed == 0.0

    def test_to_dict_includes_trace_type(self):
        """Test to_dict() includes trace_type."""
        trace = TraceContext(trace_type="ingestion")
        trace.finish()

        result = trace.to_dict()
        assert result["trace_type"] == "ingestion"

    def test_to_dict_includes_total_elapsed_ms(self):
        """Test to_dict() includes total_elapsed_ms after finish."""
        trace = TraceContext()
        time.sleep(0.01)
        trace.finish()

        result = trace.to_dict()
        assert "total_elapsed_ms" in result
        assert result["total_elapsed_ms"] > 0

    def test_to_dict_without_finish_no_total_elapsed(self):
        """Test to_dict() without finish has no total_elapsed_ms."""
        trace = TraceContext()
        result = trace.to_dict()

        assert "total_elapsed_ms" not in result

    def test_to_dict_serializable(self):
        """Test to_dict() is JSON serializable."""
        trace = TraceContext(trace_type="query")
        trace.record_stage("test", elapsed_ms=10, method="test_method")
        trace.finish()

        result = trace.to_dict()
        json_str = json.dumps(result)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["trace_id"] == trace.trace_id

    def test_multiple_finish_calls(self):
        """Test multiple finish() calls don't change elapsed time."""
        trace = TraceContext()
        time.sleep(0.01)
        trace.finish()

        elapsed1 = trace.elapsed_ms()
        time.sleep(0.01)
        trace.finish()  # Second call

        elapsed2 = trace.elapsed_ms()
        # elapsed should not change after first finish
        assert abs(elapsed2 - elapsed1) < 5  # Allow small variance


class TestTraceCollector:
    """Tests for TraceCollector."""

    def test_init_creates_output_directory(self):
        """Test init creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            assert Path(output_path).parent.exists()

    def test_collect_disabled(self):
        """Test collect() when disabled does nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path, enabled=False)

            trace = TraceContext()
            trace.finish()
            collector.collect(trace)

            assert not Path(output_path).exists()

    def test_collect_writes_to_file(self):
        """Test collect() writes trace to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            trace = TraceContext(trace_type="query")
            trace.record_stage("test_stage", elapsed_ms=10)
            trace.finish()
            collector.collect(trace)

            assert Path(output_path).exists()

            # Read and verify
            with open(output_path, "r") as f:
                line = f.readline()
                trace_dict = json.loads(line)
                assert trace_dict["trace_type"] == "query"

    def test_collect_auto_finish(self):
        """Test collect() auto-finishes trace if not finished."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            trace = TraceContext()
            # Don't call finish()
            collector.collect(trace)

            # Verify trace was finished
            assert trace._finished

            # Verify file has total_elapsed_ms
            with open(output_path, "r") as f:
                line = f.readline()
                trace_dict = json.loads(line)
                assert "total_elapsed_ms" in trace_dict

    def test_collect_batch(self):
        """Test collect_batch() collects multiple traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            traces = [
                TraceContext(trace_type="query"),
                TraceContext(trace_type="ingestion"),
            ]

            collector.collect_batch(traces)

            # Verify two lines
            with open(output_path, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2

    def test_read_traces_empty_file(self):
        """Test read_traces() with no file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            traces = collector.read_traces()
            assert traces == []

    def test_read_traces_with_filter(self):
        """Test read_traces() with trace_type filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            # Collect traces of different types
            collector.collect(TraceContext(trace_type="query"))
            collector.collect(TraceContext(trace_type="ingestion"))
            collector.collect(TraceContext(trace_type="query"))

            # Read only query traces
            traces = collector.read_traces(trace_type="query")
            assert len(traces) == 2

            for t in traces:
                assert t["trace_type"] == "query"

    def test_read_traces_limit(self):
        """Test read_traces() respects limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            # Collect 10 traces
            for i in range(10):
                collector.collect(TraceContext())

            traces = collector.read_traces(limit=5)
            assert len(traces) == 5

    def test_read_traces_reverse_order(self):
        """Test read_traces() returns traces in reverse order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            # Collect traces with different IDs
            trace1 = TraceContext()
            trace2 = TraceContext()
            trace3 = TraceContext()

            collector.collect(trace1)
            collector.collect(trace2)
            collector.collect(trace3)

            traces = collector.read_traces()

            # Should be in reverse order (most recent first)
            assert traces[0]["trace_id"] == trace3.trace_id
            assert traces[1]["trace_id"] == trace2.trace_id
            assert traces[2]["trace_id"] == trace1.trace_id

    def test_clear(self):
        """Test clear() removes trace file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            collector.collect(TraceContext())
            assert Path(output_path).exists()

            collector.clear()
            assert not Path(output_path).exists()

    def test_get_stats(self):
        """Test get_stats() returns correct statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            # Collect traces
            collector.collect(TraceContext(trace_type="query"))
            collector.collect(TraceContext(trace_type="query"))
            collector.collect(TraceContext(trace_type="ingestion"))

            stats = collector.get_stats()
            assert stats["total_traces"] == 3
            assert stats["query_traces"] == 2
            assert stats["ingestion_traces"] == 1

    def test_get_stats_empty(self):
        """Test get_stats() with no traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            stats = collector.get_stats()
            assert stats["total_traces"] == 0


class TestGlobalCollector:
    """Tests for global collector functions."""

    def test_get_trace_collector_returns_instance(self):
        """Test get_trace_collector() returns instance."""
        from src.core.trace import get_trace_collector

        collector = get_trace_collector()
        assert collector is not None
        assert isinstance(collector, TraceCollector)

    def test_set_trace_collector(self):
        """Test set_trace_collector() sets global instance."""
        from src.core.trace import get_trace_collector, set_trace_collector

        custom = TraceCollector(output_path="custom/path.jsonl")
        set_trace_collector(custom)

        retrieved = get_trace_collector()
        assert retrieved == custom


class TestTraceContextIntegration:
    """Integration tests for TraceContext with Collector."""

    def test_full_flow(self):
        """Test full trace flow: create → record → finish → collect."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)

            # Create trace
            trace = TraceContext(trace_type="query")

            # Record stages
            trace.record_stage("query_processing", elapsed_ms=5.2, method="keyword_extract")
            trace.record_stage("dense_retrieval", elapsed_ms=50.3, method="chroma", provider="openai")
            trace.record_stage("sparse_retrieval", elapsed_ms=10.1, method="bm25")
            trace.record_stage("fusion", elapsed_ms=2.5, method="rrf")
            trace.record_stage("rerank", elapsed_ms=100.0, method="llm", provider="azure")

            # Finish and collect
            trace.finish()
            collector.collect(trace)

            # Read and verify
            traces = collector.read_traces()
            assert len(traces) == 1

            result = traces[0]
            assert result["trace_type"] == "query"
            assert len(result["stages"]) == 5
            assert "total_elapsed_ms" in result

            # Verify stage details
            stage_names = [s["stage"] for s in result["stages"]]
            assert "dense_retrieval" in stage_names
            assert "rerank" in stage_names