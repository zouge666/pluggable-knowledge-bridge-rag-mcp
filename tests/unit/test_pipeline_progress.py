"""
Unit tests for Pipeline progress callback (F5).
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from src.core.trace import TraceContext
from src.core.types import Document, Chunk
from src.ingestion.pipeline import IngestionPipeline, PipelineResult


class TestPipelineProgressCallback:
    """Tests for Pipeline progress callback."""

    def test_on_progress_called_for_each_stage(self):
        """Test on_progress is called for each stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            # Track progress calls
            progress_calls = []

            def on_progress(stage, current, total):
                progress_calls.append({
                    "stage": stage,
                    "current": current,
                    "total": total,
                })

            # Mock components
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
                on_progress=on_progress,
            )

            result = pipeline.ingest(test_file)

            # Verify progress was called for each stage
            assert len(progress_calls) == 6

            # Verify stages
            stages = [p["stage"] for p in progress_calls]
            assert stages == [
                "integrity_check",
                "load",
                "split",
                "transform",
                "encode",
                "store",
            ]

            # Verify progress counts
            for i, call in enumerate(progress_calls):
                assert call["current"] == i + 1
                assert call["total"] == 6

    def test_on_progress_not_called_when_skipped(self):
        """Test on_progress is not called when file is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            # Track progress calls
            progress_calls = []

            def on_progress(stage, current, total):
                progress_calls.append({
                    "stage": stage,
                    "current": current,
                    "total": total,
                })

            # Mock components - file should be skipped
            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = True

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                on_progress=on_progress,
            )

            result = pipeline.ingest(test_file)

            # Verify no progress calls when skipped
            assert len(progress_calls) == 0

    def test_on_progress_with_custom_total(self):
        """Test on_progress respects custom total stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            # Track progress calls
            progress_calls = []

            def on_progress(stage, current, total):
                progress_calls.append({
                    "stage": stage,
                    "current": current,
                    "total": total,
                })

            # Mock components
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
                on_progress=on_progress,
            )

            result = pipeline.ingest(test_file)

            # Verify total is always 6 (standard pipeline stages)
            for call in progress_calls:
                assert call["total"] == 6

    def test_on_progress_can_be_none(self):
        """Test pipeline works when on_progress is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            # Mock components
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
                on_progress=None,  # No callback
            )

            result = pipeline.ingest(test_file)

            # Should complete without error
            assert result.success

    def test_on_progress_on_error(self):
        """Test on_progress stops on error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            # Track progress calls
            progress_calls = []

            def on_progress(stage, current, total):
                progress_calls.append({
                    "stage": stage,
                    "current": current,
                    "total": total,
                })

            # Mock components - load will fail
            mock_integrity = Mock()
            mock_integrity.compute_sha256.return_value = "test_hash"
            mock_integrity.should_skip.return_value = False

            mock_loader = Mock()
            mock_loader.load.side_effect = Exception("Load failed")

            pipeline = IngestionPipeline(
                integrity_checker=mock_integrity,
                loader=mock_loader,
                on_progress=on_progress,
            )

            result = pipeline.ingest(test_file)

            # Verify only integrity_check progress was called before error
            assert len(progress_calls) == 1
            assert progress_calls[0]["stage"] == "integrity_check"

    def test_progress_callback_signature(self):
        """Test progress callback has correct signature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            # Track callback arguments
            callback_args = []

            def on_progress(*args):
                callback_args.append(args)

            # Mock components
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
                on_progress=on_progress,
            )

            result = pipeline.ingest(test_file)

            # Verify callback signature: (stage_name, current, total)
            for args in callback_args:
                assert len(args) == 3
                assert isinstance(args[0], str)  # stage_name
                assert isinstance(args[1], int)  # current
                assert isinstance(args[2], int)  # total


class TestProgressIntegration:
    """Integration tests for progress callback with real components."""

    def test_progress_with_streamlit_style_callback(self):
        """Test progress callback works like Streamlit progress bar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            # Simulate Streamlit-style progress tracking
            progress_state = {"value": 0.0, "text": ""}

            def update_progress(stage, current, total):
                progress_state["value"] = current / total
                progress_state["text"] = f"Processing: {stage} ({current}/{total})"

            # Mock components
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
                on_progress=update_progress,
            )

            result = pipeline.ingest(test_file)

            # Verify final progress state
            assert progress_state["value"] == 1.0  # 6/6 = 100%
            assert "store" in progress_state["text"]

    def test_progress_with_logging_callback(self):
        """Test progress callback works with logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(b"%PDF-1.4\ntest content")

            # Capture log messages
            log_messages = []

            def log_progress(stage, current, total):
                log_messages.append(f"[{current}/{total}] {stage}")

            # Mock components
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
                on_progress=log_progress,
            )

            result = pipeline.ingest(test_file)

            # Verify log messages
            assert len(log_messages) == 6
            assert "[1/6] integrity_check" in log_messages
            assert "[6/6] store" in log_messages