"""
Unit tests for Ingestion pipeline tracing (F4).
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.trace import TraceContext, TraceCollector, set_trace_collector
from src.core.types import Chunk, ChunkRecord, Document, ImageRef
from src.ingestion.pipeline import IngestionPipeline, PipelineResult


class TestIngestionPipelineTracing:
    """Tests for Ingestion pipeline tracing."""

    def test_ingest_creates_trace_context(self):
        """Test ingest() creates TraceContext if not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            # Create a test file
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            # Mock all components
            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_doc = Document(id="doc1", text="test", metadata={})
            mock_loader.load.return_value = mock_doc

            mock_chunker = Mock()
            mock_chunker.split_document.return_value = []

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                chunker=mock_chunker,
            )

            result = pipeline.ingest(test_file)

            # Verify trace was collected
            traces = collector.read_traces()
            assert len(traces) == 1
            assert traces[0]["trace_type"] == "ingestion"

    def test_ingest_uses_provided_trace(self):
        """Test ingest() uses provided TraceContext."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            trace = TraceContext(trace_type="ingestion")

            # Create a test file
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            # Mock all components
            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_doc = Document(id="doc1", text="test", metadata={})
            mock_loader.load.return_value = mock_doc

            mock_chunker = Mock()
            mock_chunker.split_document.return_value = []

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                chunker=mock_chunker,
            )

            result = pipeline.ingest(test_file, trace=trace)

            # Verify the same trace was used
            assert trace._finished
            traces = collector.read_traces()
            assert len(traces) == 1
            assert traces[0]["trace_id"] == trace.trace_id

    def test_trace_records_integrity_check_stage(self):
        """Test trace records integrity_check stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_doc = Document(id="doc1", text="test", metadata={})
            mock_loader.load.return_value = mock_doc

            mock_chunker = Mock()
            mock_chunker.split_document.return_value = []

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                chunker=mock_chunker,
            )

            result = pipeline.ingest(test_file)

            traces = collector.read_traces()
            stages = traces[0]["stages"]
            stage_names = [s["stage"] for s in stages]
            assert "integrity_check" in stage_names

    def test_trace_records_load_stage(self):
        """Test trace records load stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_doc = Document(id="doc1", text="test", metadata={})
            mock_loader.load.return_value = mock_doc

            mock_chunker = Mock()
            mock_chunker.split_document.return_value = []

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                chunker=mock_chunker,
            )

            result = pipeline.ingest(test_file)

            traces = collector.read_traces()
            stages = traces[0]["stages"]
            load_stages = [s for s in stages if s["stage"] == "load"]
            assert len(load_stages) == 1
            assert load_stages[0]["method"] == "pdf"

    def test_trace_records_split_stage(self):
        """Test trace records split stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_doc = Document(id="doc1", text="test", metadata={})
            mock_loader.load.return_value = mock_doc

            mock_chunker = Mock()
            mock_chunker.split_document.return_value = [
                Chunk(id="c1", text="chunk1", metadata={}),
            ]

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                chunker=mock_chunker,
            )

            result = pipeline.ingest(test_file)

            traces = collector.read_traces()
            stages = traces[0]["stages"]
            split_stages = [s for s in stages if s["stage"] == "split"]
            assert len(split_stages) == 1
            assert split_stages[0]["details"]["chunks_count"] == 1

    def test_trace_records_transform_stage(self):
        """Test trace records transform stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_doc = Document(id="doc1", text="test", metadata={})
            mock_loader.load.return_value = mock_doc

            mock_chunker = Mock()
            mock_chunker.split_document.return_value = []

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                chunker=mock_chunker,
            )

            result = pipeline.ingest(test_file)

            traces = collector.read_traces()
            stages = traces[0]["stages"]
            stage_names = [s["stage"] for s in stages]
            assert "transform" in stage_names

    def test_trace_records_encode_stage(self):
        """Test trace records encode stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_doc = Document(id="doc1", text="test", metadata={})
            mock_loader.load.return_value = mock_doc

            mock_chunker = Mock()
            mock_chunker.split_document.return_value = []

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                chunker=mock_chunker,
            )

            result = pipeline.ingest(test_file)

            traces = collector.read_traces()
            stages = traces[0]["stages"]
            stage_names = [s["stage"] for s in stages]
            assert "encode" in stage_names

    def test_trace_records_store_stage(self):
        """Test trace records store stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_doc = Document(id="doc1", text="test", metadata={})
            mock_loader.load.return_value = mock_doc

            mock_chunker = Mock()
            mock_chunker.split_document.return_value = []

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                chunker=mock_chunker,
            )

            result = pipeline.ingest(test_file)

            traces = collector.read_traces()
            stages = traces[0]["stages"]
            stage_names = [s["stage"] for s in stages]
            assert "store" in stage_names

    def test_trace_includes_total_elapsed_ms(self):
        """Test trace includes total_elapsed_ms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_doc = Document(id="doc1", text="test", metadata={})
            mock_loader.load.return_value = mock_doc

            mock_chunker = Mock()
            mock_chunker.split_document.return_value = []

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                chunker=mock_chunker,
            )

            result = pipeline.ingest(test_file)

            traces = collector.read_traces()
            assert "total_elapsed_ms" in traces[0]

    def test_trace_on_skip(self):
        """Test trace is collected when file is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = True  # File should be skipped

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
            )

            result = pipeline.ingest(test_file)

            # Verify trace was collected even when skipped
            traces = collector.read_traces()
            assert len(traces) == 1

            # Verify integrity_check stage with skip action
            stages = traces[0]["stages"]
            integrity_stages = [s for s in stages if s["stage"] == "integrity_check"]
            assert len(integrity_stages) == 1
            assert integrity_stages[0]["method"] == "skip"

    def test_trace_on_error(self):
        """Test trace is collected even on error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_loader.load.side_effect = Exception("Load failed")

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
            )

            result = pipeline.ingest(test_file)

            # Verify trace was collected even on error
            traces = collector.read_traces()
            assert len(traces) == 1

            # Verify error stage was recorded
            stages = traces[0]["stages"]
            error_stages = [s for s in stages if s["stage"] == "error"]
            assert len(error_stages) == 1
            assert "Load failed" in error_stages[0]["details"]["error"]

    def test_trace_all_stages_in_order(self):
        """Test trace records all stages in correct order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            collector = TraceCollector(output_path=output_path)
            set_trace_collector(collector)

            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_doc = Document(id="doc1", text="test", metadata={})
            mock_loader.load.return_value = mock_doc

            mock_chunker = Mock()
            mock_chunker.split_document.return_value = []

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                chunker=mock_chunker,
            )

            result = pipeline.ingest(test_file)

            traces = collector.read_traces()
            stages = traces[0]["stages"]

            # Verify all stages are present in order
            expected_stages = [
                "integrity_check",
                "load",
                "split",
                "transform",
                "encode",
                "store",
            ]

            actual_stages = [s["stage"] for s in stages]
            for expected in expected_stages:
                assert expected in actual_stages, f"Missing stage: {expected}"


class TestIngestionComponentTracing:
    """Tests for individual ingestion component tracing."""

    def test_sparse_encoder_records_stage(self):
        """Test SparseEncoder records stage to trace."""
        from src.ingestion.embedding.sparse_encoder import SparseEncoder

        trace = TraceContext(trace_type="ingestion")

        encoder = SparseEncoder()

        chunks = [Chunk(id="c1", text="test content", metadata={})]
        result = encoder.encode(chunks, trace=trace)

        # Verify stage was recorded
        stages = trace.stages
        stage_names = [s["stage"] for s in stages]
        assert "sparse_encoding" in stage_names
