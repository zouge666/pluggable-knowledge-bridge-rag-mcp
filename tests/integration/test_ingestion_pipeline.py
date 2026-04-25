"""
Integration tests for Ingestion Pipeline.

Tests the complete pipeline flow with mocked components.
"""

import tempfile
import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.ingestion.pipeline import IngestionPipeline, PipelineResult, FakeIngestionPipeline
from src.core.trace.trace_context import TraceContext
from src.core.types import Document, Chunk, ChunkRecord, ImageRef
from src.libs.loader.file_integrity import SQLiteIntegrityChecker


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_pipeline_result_creation(self):
        """Test creating PipelineResult."""
        result = PipelineResult(
            success=True,
            doc_hash="abc123",
            file_path="/path/to/file.pdf",
            collection="test-collection",
            chunks_count=10,
            images_count=2,
            elapsed_ms=100.0,
        )
        assert result.success is True
        assert result.doc_hash == "abc123"
        assert result.chunks_count == 10
        assert result.images_count == 2
        assert result.skipped is False

    def test_pipeline_result_with_error(self):
        """Test creating PipelineResult with error."""
        result = PipelineResult(
            success=False,
            doc_hash="abc123",
            file_path="/path/to/file.pdf",
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_pipeline_result_skipped(self):
        """Test creating PipelineResult for skipped file."""
        result = PipelineResult(
            success=True,
            doc_hash="abc123",
            file_path="/path/to/file.pdf",
            skipped=True,
        )
        assert result.skipped is True


class TestFakeIngestionPipeline:
    """Tests for FakeIngestionPipeline."""

    def test_fake_pipeline_returns_default_result(self):
        """Test fake pipeline returns default result."""
        pipeline = FakeIngestionPipeline()
        result = pipeline.ingest("/path/to/file.pdf", collection="test")

        assert result.success is True
        assert result.chunks_count == 10
        assert pipeline.ingest_calls[0]["file_path"] == "/path/to/file.pdf"

    def test_fake_pipeline_returns_custom_result(self):
        """Test fake pipeline returns custom result."""
        custom_result = PipelineResult(
            success=False,
            doc_hash="custom-hash",
            file_path="/custom/path.pdf",
            error="Custom error",
        )
        pipeline = FakeIngestionPipeline(result=custom_result)
        result = pipeline.ingest("/path/to/file.pdf")

        assert result.success is False
        assert result.error == "Custom error"

    def test_fake_pipeline_records_calls(self):
        """Test fake pipeline records all calls."""
        pipeline = FakeIngestionPipeline()
        pipeline.ingest("/file1.pdf", collection="col1")
        pipeline.ingest("/file2.pdf", collection="col2", force=True)

        assert len(pipeline.ingest_calls) == 2
        assert pipeline.ingest_calls[0]["collection"] == "col1"
        assert pipeline.ingest_calls[1]["force"] is True


class TestIngestionPipeline:
    """Tests for IngestionPipeline with mocked components."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def mock_integrity_checker(self, temp_dir):
        """Create mock integrity checker."""
        db_path = os.path.join(temp_dir, "integrity.db")
        checker = SQLiteIntegrityChecker(db_path=db_path)
        return checker

    @pytest.fixture
    def mock_loader(self):
        """Create mock loader that returns a Document."""
        loader = Mock()

        def load_side_effect(path, collection=None, trace=None):
            return Document(
                id="doc-123",
                text="This is sample document text.\n\nSecond paragraph here.",
                metadata={
                    "source_path": str(path),
                    "doc_type": "pdf",
                    "collection": collection,
                },
            )

        loader.load.side_effect = load_side_effect
        return loader

    @pytest.fixture
    def mock_chunker(self):
        """Create mock chunker that returns Chunks."""
        chunker = Mock()

        def split_side_effect(document, trace=None):
            text = document.text
            paragraphs = text.split("\n\n")
            chunks = []
            for i, para in enumerate(paragraphs):
                chunk = Chunk(
                    id=f"chunk-{i}",
                    text=para,
                    metadata={"chunk_index": i},
                    source_ref=document.id,
                )
                chunks.append(chunk)
            return chunks

        chunker.split_document.side_effect = split_side_effect
        return chunker

    @pytest.fixture
    def mock_batch_processor(self):
        """Create mock batch processor."""
        processor = Mock()

        def process_side_effect(chunks, trace=None):
            records = []
            for chunk in chunks:
                record = ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    dense_vector=[0.1] * 128,  # Mock 128-dim vector
                    sparse_vector={"term": 1.0},
                    metadata=chunk.metadata,
                )
                records.append(record)
            return records

        processor.process.side_effect = process_side_effect
        return processor

    @pytest.fixture
    def mock_vector_upserter(self):
        """Create mock vector upserter."""
        upserter = Mock()
        upserter.upsert.return_value = None
        return upserter

    @pytest.fixture
    def mock_bm25_indexer(self):
        """Create mock BM25 indexer."""
        indexer = Mock()
        indexer.index.return_value = None
        return indexer

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        return store

    @pytest.fixture
    def sample_pdf(self, temp_dir):
        """Create a sample PDF file (minimal)."""
        # Create a minimal text file as placeholder
        # In real tests, we would use actual PDF
        file_path = os.path.join(temp_dir, "sample.pdf")
        # Minimal PDF header (not valid but enough for hash)
        with open(file_path, "wb") as f:
            f.write(b"%PDF-1.4\n%fake pdf content\n")
        return file_path

    def test_pipeline_ingest_success(
        self,
        temp_dir,
        mock_integrity_checker,
        mock_loader,
        mock_chunker,
        mock_batch_processor,
        mock_vector_upserter,
        mock_bm25_indexer,
        mock_vector_store,
        sample_pdf,
    ):
        """Test successful pipeline ingestion."""
        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_vector_upserter,
            bm25_indexer=mock_bm25_indexer,
            vector_store=mock_vector_store,
        )

        result = pipeline.ingest(sample_pdf, collection="test-collection")

        assert result.success is True
        assert result.skipped is False
        assert result.chunks_count == 2  # Two paragraphs
        assert len(result.stages) == 6  # 6 stages

        # Verify components were called
        mock_loader.load.assert_called_once()
        mock_chunker.split_document.assert_called_once()
        mock_batch_processor.process.assert_called_once()
        # Vector upserter and BM25 indexer are called in _store_records
        # which requires both to be set

    def test_pipeline_skips_already_processed(
        self,
        temp_dir,
        mock_integrity_checker,
        mock_loader,
        sample_pdf,
    ):
        """Test pipeline skips already processed files."""
        # First, mark the file as already processed
        doc_hash = mock_integrity_checker.compute_sha256(sample_pdf)
        mock_integrity_checker.mark_success(doc_hash, sample_pdf, "test-collection")

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
        )

        result = pipeline.ingest(sample_pdf, collection="test-collection")

        assert result.success is True
        assert result.skipped is True
        # Loader should not be called
        mock_loader.load.assert_not_called()

    def test_pipeline_force_reprocess(
        self,
        temp_dir,
        mock_integrity_checker,
        mock_loader,
        mock_chunker,
        mock_batch_processor,
        sample_pdf,
    ):
        """Test pipeline force reprocesses already processed files."""
        # Mark file as processed
        doc_hash = mock_integrity_checker.compute_sha256(sample_pdf)
        mock_integrity_checker.mark_success(doc_hash, sample_pdf, "test-collection")

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
        )

        result = pipeline.ingest(sample_pdf, collection="test-collection", force=True)

        assert result.success is True
        assert result.skipped is False
        # Loader should be called despite being processed
        mock_loader.load.assert_called_once()

    def test_pipeline_handles_load_error(
        self,
        temp_dir,
        mock_integrity_checker,
        mock_loader,
        sample_pdf,
    ):
        """Test pipeline handles load errors."""
        mock_loader.load.side_effect = Exception("Load failed")

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
        )

        result = pipeline.ingest(sample_pdf, collection="test-collection")

        assert result.success is False
        assert result.error == "Load failed"

        # Check integrity checker marked as failed
        doc_hash = mock_integrity_checker.compute_sha256(sample_pdf)
        record = mock_integrity_checker.get_record(doc_hash)
        assert record["status"] == "failed"

    def test_pipeline_handles_chunk_error(
        self,
        temp_dir,
        mock_integrity_checker,
        mock_loader,
        mock_chunker,
        sample_pdf,
    ):
        """Test pipeline handles chunk errors."""
        mock_chunker.split_document.side_effect = Exception("Chunk failed")

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
        )

        result = pipeline.ingest(sample_pdf, collection="test-collection")

        assert result.success is False
        assert result.error == "Chunk failed"

    def test_pipeline_with_trace(
        self,
        temp_dir,
        mock_integrity_checker,
        mock_loader,
        mock_chunker,
        mock_batch_processor,
        sample_pdf,
    ):
        """Test pipeline with trace context."""
        trace = TraceContext(trace_id="test-trace")

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
        )

        result = pipeline.ingest(sample_pdf, collection="test-collection", trace=trace)

        assert result.success is True
        # Trace should have recorded stages from components

    def test_pipeline_progress_callback(
        self,
        temp_dir,
        mock_integrity_checker,
        mock_loader,
        mock_chunker,
        mock_batch_processor,
        sample_pdf,
    ):
        """Test pipeline progress callback."""
        progress_calls = []

        def on_progress(stage, current, total):
            progress_calls.append((stage, current, total))

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            on_progress=on_progress,
        )

        result = pipeline.ingest(sample_pdf, collection="test-collection")

        assert result.success is True
        assert len(progress_calls) == 6  # 6 stages
        # Check first and last progress calls
        assert progress_calls[0] == ("integrity_check", 1, 6)
        assert progress_calls[-1] == ("store", 6, 6)

    def test_pipeline_with_images(
        self,
        temp_dir,
        mock_integrity_checker,
        mock_loader,
        mock_chunker,
        mock_batch_processor,
        mock_vector_store,
        sample_pdf,
    ):
        """Test pipeline with image extraction."""
        # Create a fake image file
        image_path = os.path.join(temp_dir, "img-1.png")
        with open(image_path, "wb") as f:
            f.write(b"fake image data")

        # Create mock loader that returns document with images
        mock_loader_with_images = Mock()

        def load_with_images(path, collection=None, trace=None):
            doc = Document(
                id="doc-123",
                text="Document text with image.",
                metadata={"source_path": str(path)},
            )
            # Add image reference (ImageRef has path, not data)
            img = ImageRef(
                id="img-1",
                path=image_path,
                page=1,
            )
            doc.metadata["images"] = [img.to_dict()]
            return doc

        mock_loader_with_images.load.side_effect = load_with_images

        # Create mock image storage
        mock_image_storage = Mock()
        mock_image_storage.save.return_value = None

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader_with_images,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            image_storage=mock_image_storage,
            vector_store=mock_vector_store,
        )

        result = pipeline.ingest(sample_pdf, collection="test-collection")

        assert result.success is True
        assert result.images_count == 1
        mock_image_storage.save.assert_called_once()

    def test_pipeline_empty_chunks(
        self,
        temp_dir,
        mock_integrity_checker,
        mock_loader,
        mock_chunker,
        mock_batch_processor,
        mock_vector_store,
        sample_pdf,
    ):
        """Test pipeline handles empty chunks."""
        # Create a chunker that returns empty list
        empty_chunker = Mock()
        empty_chunker.split_document.return_value = []

        pipeline = IngestionPipeline(
            integrity_checker=mock_integrity_checker,
            loader=mock_loader,
            chunker=empty_chunker,
            batch_processor=mock_batch_processor,
            vector_store=mock_vector_store,
        )

        result = pipeline.ingest(sample_pdf, collection="test-collection")

        assert result.success is True
        assert result.chunks_count == 0


class TestIngestionPipelineIntegration:
    """Integration tests with real components (where possible)."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_pipeline_with_real_integrity_checker(self, temp_dir):
        """Test pipeline with real SQLite integrity checker."""
        db_path = os.path.join(temp_dir, "integrity.db")
        checker = SQLiteIntegrityChecker(db_path=db_path)

        # Create a test file
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        # Compute hash
        hash1 = checker.compute_sha256(test_file)

        # Modify file
        with open(test_file, "w") as f:
            f.write("modified content")

        hash2 = checker.compute_sha256(test_file)

        # Hashes should be different
        assert hash1 != hash2

        # Mark first hash as success
        checker.mark_success(hash1, test_file, "test")

        # Should skip first hash
        assert checker.should_skip(hash1) is True

        # Should not skip second hash
        assert checker.should_skip(hash2) is False