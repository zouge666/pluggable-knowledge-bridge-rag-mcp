"""
Unit tests for DocumentManager (G2).
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from src.ingestion.document_manager import (
    DocumentManager,
    DocumentInfo,
    DocumentDetail,
    DeleteResult,
    CollectionStats,
    ChunkDetail,
)
from src.libs.vector_store.chroma_store import ChromaStore
from src.ingestion.storage.bm25_indexer import FakeBM25Indexer
from src.ingestion.storage.image_storage import FakeImageStorage
from src.libs.loader.file_integrity import SQLiteIntegrityChecker


class TestDocumentManager:
    """Tests for DocumentManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_chroma(self):
        """Create mock ChromaStore."""
        mock = Mock(spec=ChromaStore)
        mock.get_collection_stats.return_value = {"count": 10}
        mock.delete_by_ids.return_value = 5
        mock.get_by_metadata.return_value = []
        mock.query_by_metadata.return_value = []
        mock.delete_by_metadata.return_value = 0
        return mock

    @pytest.fixture
    def fake_bm25(self):
        """Create FakeBM25Indexer."""
        return FakeBM25Indexer()

    @pytest.fixture
    def fake_images(self):
        """Create FakeImageStorage."""
        return FakeImageStorage()

    @pytest.fixture
    def file_integrity(self, temp_dir):
        """Create SQLiteIntegrityChecker with temp database."""
        db_path = Path(temp_dir) / "test_history.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))
        yield checker
        checker.close()

    @pytest.fixture
    def document_manager(self, mock_chroma, fake_bm25, fake_images, file_integrity):
        """Create DocumentManager with test fixtures."""
        return DocumentManager(
            chroma_store=mock_chroma,
            bm25_indexer=fake_bm25,
            image_storage=fake_images,
            file_integrity=file_integrity,
        )

    def test_init(self, document_manager):
        """Test DocumentManager initialization."""
        assert document_manager._chroma is not None
        assert document_manager._bm25 is not None
        assert document_manager._images is not None
        assert document_manager._integrity is not None

    def test_list_documents_empty(self, document_manager):
        """Test list_documents returns empty list when no documents."""
        result = document_manager.list_documents()
        assert result == []

    def test_list_documents_with_records(
        self,
        document_manager,
        file_integrity,
        mock_chroma,
    ):
        """Test list_documents returns documents from history."""
        # Add a record to history
        file_integrity.mark_success(
            file_hash="hash1",
            file_path="/path/to/doc.pdf",
            collection="test_collection",
            metadata={"chunk_count": 5},
        )

        # Mock chroma to return chunks
        mock_chroma.query_by_metadata.return_value = [
            {"id": "chunk1"},
            {"id": "chunk2"},
        ]

        result = document_manager.list_documents()

        assert len(result) == 1
        assert result[0].source_path == "/path/to/doc.pdf"
        assert result[0].file_hash == "hash1"
        assert result[0].collection == "test_collection"
        assert result[0].status == "success"

    def test_list_documents_with_collection_filter(
        self,
        document_manager,
        file_integrity,
    ):
        """Test list_documents with collection filter."""
        # Add records to different collections
        file_integrity.mark_success(
            file_hash="hash1",
            file_path="/path/to/doc1.pdf",
            collection="collection_a",
        )
        file_integrity.mark_success(
            file_hash="hash2",
            file_path="/path/to/doc2.pdf",
            collection="collection_b",
        )

        result = document_manager.list_documents(collection="collection_a")

        assert len(result) == 1
        assert result[0].collection == "collection_a"

    def test_list_documents_with_status_filter(
        self,
        document_manager,
        file_integrity,
    ):
        """Test list_documents with status filter."""
        # Add success and failed records
        file_integrity.mark_success(
            file_hash="hash1",
            file_path="/path/to/doc1.pdf",
            collection="test",
        )
        file_integrity.mark_failed(
            file_hash="hash2",
            error_msg="Failed to process",
        )

        result_success = document_manager.list_documents(status="success")
        assert len(result_success) == 1
        assert result_success[0].status == "success"

    def test_get_document_detail_not_found(self, document_manager):
        """Test get_document_detail returns None for unknown document."""
        result = document_manager.get_document_detail("/unknown/path.pdf")
        assert result is None

    def test_get_document_detail_found(
        self,
        document_manager,
        file_integrity,
        mock_chroma,
        fake_images,
    ):
        """Test get_document_detail returns document details."""
        # Add record to history
        file_integrity.mark_success(
            file_hash="hash1",
            file_path="/path/to/doc.pdf",
            collection="test_collection",
        )

        # Mock chroma to return chunks
        mock_chroma.get_by_metadata.return_value = [
            {"id": "chunk1", "text": "text1", "metadata": {"source": "/path/to/doc.pdf"}},
            {"id": "chunk2", "text": "text2", "metadata": {"source": "/path/to/doc.pdf"}},
        ]

        # Add images
        fake_images.save("img1", b"image_data", collection="test_collection", doc_hash="hash1")

        result = document_manager.get_document_detail("/path/to/doc.pdf")

        assert result is not None
        assert result.source_path == "/path/to/doc.pdf"
        assert result.file_hash == "hash1"
        assert result.collection == "test_collection"
        assert len(result.chunks) == 2
        assert "img1" in result.images

    def test_delete_document_not_found(self, document_manager):
        """Test delete_document returns error for unknown document."""
        result = document_manager.delete_document("/unknown/path.pdf")

        assert result.success is False
        assert result.error == "Document not found"

    def test_delete_document_success(
        self,
        document_manager,
        file_integrity,
        mock_chroma,
        fake_images,
        fake_bm25,
    ):
        """Test delete_document removes from all stores."""
        # Add record to history
        file_integrity.mark_success(
            file_hash="hash1",
            file_path="/path/to/doc.pdf",
            collection="test_collection",
        )

        # Mock chroma to return chunks
        mock_chroma.get_by_metadata.return_value = [
            {"id": "chunk1", "text": "text1", "metadata": {}},
            {"id": "chunk2", "text": "text2", "metadata": {}},
        ]
        mock_chroma.delete_by_ids.return_value = 2

        # Add images
        fake_images.save("img1", b"image_data", collection="test_collection", doc_hash="hash1")

        # Build BM25 index
        fake_bm25.build({
            "chunk1": {"term1": 1.0},
            "chunk2": {"term2": 1.0},
        })

        result = document_manager.delete_document("/path/to/doc.pdf")

        assert result.success is True
        assert result.chunks_deleted == 2
        assert result.images_deleted == 1
        assert result.history_removed is True

        # Verify document is no longer in list
        documents = document_manager.list_documents()
        assert len(documents) == 0

    def test_delete_document_partial_failure(
        self,
        document_manager,
        file_integrity,
        mock_chroma,
    ):
        """Test delete_document handles partial failures."""
        # Add record to history
        file_integrity.mark_success(
            file_hash="hash1",
            file_path="/path/to/doc.pdf",
            collection="test_collection",
        )

        # Mock chroma to throw error
        mock_chroma.get_by_metadata.return_value = [
            {"id": "chunk1", "text": "text1", "metadata": {}},
        ]
        mock_chroma.delete_by_ids.side_effect = Exception("Delete failed")

        result = document_manager.delete_document("/path/to/doc.pdf")

        assert result.success is False
        assert "Delete failed" in result.error

    def test_get_collection_stats(
        self,
        document_manager,
        file_integrity,
        mock_chroma,
        fake_images,
    ):
        """Test get_collection_stats returns correct stats."""
        # Add records
        file_integrity.mark_success(
            file_hash="hash1",
            file_path="/path/to/doc1.pdf",
            collection="test_collection",
        )
        file_integrity.mark_success(
            file_hash="hash2",
            file_path="/path/to/doc2.pdf",
            collection="test_collection",
        )

        # Mock chroma stats
        mock_chroma.get_collection_stats.return_value = {"count": 10}

        # Add images
        fake_images.save("img1", b"data1", collection="test_collection")
        fake_images.save("img2", b"data2", collection="test_collection")

        result = document_manager.get_collection_stats("test_collection")

        assert result.collection_name == "test_collection"
        assert result.document_count == 2
        assert result.chunk_count == 10
        assert result.image_count == 2

    def test_delete_by_collection(
        self,
        document_manager,
        file_integrity,
        mock_chroma,
        fake_images,
    ):
        """Test delete_by_collection removes all documents."""
        # Add records to collection
        file_integrity.mark_success(
            file_hash="hash1",
            file_path="/path/to/doc1.pdf",
            collection="test_collection",
        )
        file_integrity.mark_success(
            file_hash="hash2",
            file_path="/path/to/doc2.pdf",
            collection="test_collection",
        )

        # Mock chroma
        mock_chroma.get_by_metadata.return_value = [
            {"id": "chunk1", "text": "text1", "metadata": {}},
        ]
        mock_chroma.delete_by_ids.return_value = 1

        # Add images
        fake_images.save("img1", b"data1", collection="test_collection", doc_hash="hash1")

        result = document_manager.delete_by_collection("test_collection")

        assert result["collection"] == "test_collection"
        assert result["documents_deleted"] == 2
        assert result["chunks_deleted"] == 2  # 1 chunk per doc
        assert result["images_deleted"] == 1

    def test_chunk_detail_dataclass(self):
        """Test ChunkDetail dataclass."""
        chunk = ChunkDetail(
            chunk_id="chunk1",
            text="sample text",
            metadata={"source": "test.pdf"},
            score=0.95,
        )
        assert chunk.chunk_id == "chunk1"
        assert chunk.text == "sample text"
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.score == 0.95

    def test_document_info_dataclass(self):
        """Test DocumentInfo dataclass."""
        info = DocumentInfo(
            source_path="/path/to/doc.pdf",
            file_hash="abc123",
            collection="test",
            chunk_count=5,
            image_count=2,
            status="success",
            created_at="2026-04-26T10:00:00",
        )
        assert info.source_path == "/path/to/doc.pdf"
        assert info.chunk_count == 5
        assert info.image_count == 2

    def test_delete_result_dataclass(self):
        """Test DeleteResult dataclass."""
        result = DeleteResult(
            success=True,
            source_path="/path/to/doc.pdf",
            chunks_deleted=5,
            images_deleted=2,
            history_removed=True,
        )
        assert result.success is True
        assert result.chunks_deleted == 5

    def test_collection_stats_dataclass(self):
        """Test CollectionStats dataclass."""
        stats = CollectionStats(
            collection_name="test",
            document_count=10,
            chunk_count=50,
            image_count=5,
        )
        assert stats.collection_name == "test"
        assert stats.document_count == 10


class TestDocumentManagerIntegration:
    """Integration tests for DocumentManager with real components."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.skip(reason="Requires chromadb package")
    def test_full_document_lifecycle(self, temp_dir):
        """Test full document lifecycle: add, list, detail, delete."""
        # Setup real components
        chroma_path = Path(temp_dir) / "chroma"
        bm25_path = Path(temp_dir) / "bm25.db"
        image_dir = Path(temp_dir) / "images"
        history_path = Path(temp_dir) / "history.db"

        # Create settings mock for ChromaStore
        from dataclasses import dataclass
        @dataclass
        class MockVectorStoreSettings:
            persist_directory: str
            collection_name: str

        # Create ChromaStore
        chroma = ChromaStore(MockVectorStoreSettings(
            persist_directory=str(chroma_path),
            collection_name="test",
        ))

        # Create BM25Indexer
        bm25 = FakeBM25Indexer()

        # Create ImageStorage
        images = FakeImageStorage()

        # Create FileIntegrityChecker
        integrity = SQLiteIntegrityChecker(db_path=str(history_path))

        # Create DocumentManager
        manager = DocumentManager(
            chroma_store=chroma,
            bm25_indexer=bm25,
            image_storage=images,
            file_integrity=integrity,
        )

        # 1. Mark document as processed
        integrity.mark_success(
            file_hash="hash1",
            file_path="/path/to/doc.pdf",
            collection="test",
        )

        # 2. Add chunks to Chroma
        from src.libs.vector_store.base_vector_store import VectorRecord
        chroma.upsert([
            VectorRecord(
                id="chunk1",
                vector=[0.1] * 1536,
                text="text1",
                metadata={"source": "/path/to/doc.pdf"},
            ),
            VectorRecord(
                id="chunk2",
                vector=[0.2] * 1536,
                text="text2",
                metadata={"source": "/path/to/doc.pdf"},
            ),
        ])

        # 3. Add images
        images.save("img1", b"image_data", collection="test", doc_hash="hash1")

        # 4. Build BM25 index
        bm25.build({
            "chunk1": {"term1": 1.0},
            "chunk2": {"term2": 1.0},
        })

        # 5. List documents
        docs = manager.list_documents(collection="test")
        assert len(docs) == 1
        assert docs[0].source_path == "/path/to/doc.pdf"

        # 6. Get document detail
        detail = manager.get_document_detail("/path/to/doc.pdf")
        assert detail is not None
        assert len(detail.chunks) == 2
        assert "img1" in detail.images

        # 7. Delete document
        result = manager.delete_document("/path/to/doc.pdf")
        assert result.success is True
        assert result.chunks_deleted == 2
        assert result.images_deleted == 1

        # 8. Verify deletion
        docs_after = manager.list_documents(collection="test")
        assert len(docs_after) == 0

        integrity.close()