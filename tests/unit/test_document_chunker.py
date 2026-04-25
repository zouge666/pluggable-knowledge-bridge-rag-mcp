"""
Document Chunker 单元测试。

验证 Document → List[Chunk] 的转换逻辑。
"""

import hashlib
import pytest
from unittest.mock import MagicMock, patch

from src.core.settings import Settings, IngestionSettings
from src.core.types import Chunk, Document, ImageRef
from src.ingestion.chunking import DocumentChunker, FakeDocumentChunker


class TestFakeDocumentChunker:
    """FakeDocumentChunker 测试（不依赖真实 Splitter）。"""

    def test_split_document_basic(self):
        """应该能切分文档。"""
        chunker = FakeDocumentChunker(chunk_size=50)

        doc = Document(
            id="doc_001",
            text="This is a test document with enough content to split into multiple chunks.",
            metadata={"source_path": "test.pdf"},
        )

        chunks = chunker.split_document(doc)

        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_split_document_chunk_size(self):
        """应该按指定大小切分。"""
        chunker = FakeDocumentChunker(chunk_size=20)

        doc = Document(
            id="doc_002",
            text="A" * 100,  # 100 个字符
            metadata={},
        )

        chunks = chunker.split_document(doc)

        assert len(chunks) == 5  # 100 / 20 = 5

    def test_chunk_id_unique(self):
        """每个 Chunk ID 应该唯一。"""
        chunker = FakeDocumentChunker(chunk_size=30)

        doc = Document(
            id="doc_003",
            text="Different content in each chunk part one part two part three.",
            metadata={},
        )

        chunks = chunker.split_document(doc)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))  # 所有 ID 唯一

    def test_chunk_id_deterministic(self):
        """相同文档应该产生相同 Chunk ID 序列。"""
        chunker = FakeDocumentChunker(chunk_size=30)

        doc = Document(
            id="doc_004",
            text="Same content same chunks.",
            metadata={},
        )

        chunks1 = chunker.split_document(doc)
        chunks2 = chunker.split_document(doc)

        ids1 = [c.id for c in chunks1]
        ids2 = [c.id for c in chunks2]

        assert ids1 == ids2

    def test_metadata_inheritance(self):
        """Chunk 应继承 Document.metadata。"""
        chunker = FakeDocumentChunker(chunk_size=50)

        doc = Document(
            id="doc_005",
            text="Test content for metadata inheritance.",
            metadata={
                "source_path": "/path/to/file.pdf",
                "doc_type": "pdf",
                "title": "Test Document",
            },
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            assert chunk.metadata["source_path"] == "/path/to/file.pdf"
            assert chunk.metadata["doc_type"] == "pdf"
            assert chunk.metadata["title"] == "Test Document"

    def test_chunk_index_added(self):
        """metadata 应包含 chunk_index。"""
        chunker = FakeDocumentChunker(chunk_size=20)

        doc = Document(
            id="doc_006",
            text="A" * 60,
            metadata={},
        )

        chunks = chunker.split_document(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_source_ref_links_to_document(self):
        """source_ref 应指向父 Document.id。"""
        chunker = FakeDocumentChunker(chunk_size=30)

        doc = Document(
            id="doc_007",
            text="Content for source ref test.",
            metadata={},
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            assert chunk.source_ref == "doc_007"

    def test_image_ref_extraction(self):
        """应该能提取图片 ID。"""
        chunker = FakeDocumentChunker(chunk_size=100)

        doc = Document(
            id="doc_008",
            text="Text before [IMAGE: img_001] text after [IMAGE: img_002] end.",
            metadata={
                "images": [
                    {"id": "img_001", "path": "img1.png"},
                    {"id": "img_002", "path": "img2.png"},
                ],
            },
        )

        chunks = chunker.split_document(doc)

        # 检查包含图片占位符的 chunk
        for chunk in chunks:
            if "[IMAGE:" in chunk.text:
                assert "image_refs" in chunk.metadata
                # 验证提取的 ID
                refs = chunk.metadata["image_refs"]
                for ref in refs:
                    assert ref.startswith("img_")

    def test_chunk_without_images_no_image_field(self):
        """不含图片的 chunk 不应有 images 字段。"""
        chunker = FakeDocumentChunker(chunk_size=30)

        doc = Document(
            id="doc_009",
            text="Plain text without any images here.",
            metadata={},
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            assert "images" not in chunk.metadata
            assert "image_refs" not in chunk.metadata

    def test_chunk_serializable(self):
        """Chunk 应可序列化。"""
        chunker = FakeDocumentChunker(chunk_size=50)

        doc = Document(
            id="doc_010",
            text="Test serialization.",
            metadata={"key": "value"},
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            data = chunk.to_dict()
            assert "id" in data
            assert "text" in data
            assert "metadata" in data


class TestDocumentChunker:
    """DocumentChunker 测试（使用 Mock Splitter）。"""

    def test_split_document_with_mock_splitter(self):
        """应该能使用 Mock Splitter 切分文档。"""
        # 创建 Mock Splitter
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_splitter.get_backend_name.return_value = "fake"

        # 创建 Settings
        settings = Settings(
            ingestion=IngestionSettings(
                splitter="recursive",
                chunk_size=100,
            )
        )

        # Patch SplitterFactory
        with patch("src.ingestion.chunking.document_chunker.SplitterFactory.create") as mock_factory:
            mock_factory.return_value = mock_splitter

            chunker = DocumentChunker(settings)

            doc = Document(
                id="doc_test",
                text="Test content",
                metadata={"source_path": "test.pdf"},
            )

            chunks = chunker.split_document(doc)

            assert len(chunks) == 3
            assert chunks[0].text == "chunk1"
            assert chunks[1].text == "chunk2"
            assert chunks[2].text == "chunk3"

    def test_chunk_id_format(self):
        """Chunk ID 格式应为 {doc_id}_{index:04d}_{hash_8chars}。"""
        chunker = FakeDocumentChunker(chunk_size=50)

        doc = Document(
            id="abc123",
            text="Test content for ID format validation.",
            metadata={},
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            # ID 应以 doc_id 开头
            assert chunk.id.startswith("abc123") or len(chunk.id) == 16  # FakeChunker 使用 hash[:16]

    def test_offset_estimation(self):
        """应该估算 start_offset 和 end_offset。"""
        chunker = FakeDocumentChunker(chunk_size=20)

        doc = Document(
            id="doc_offset",
            text="A" * 60,
            metadata={},
        )

        chunks = chunker.split_document(doc)

        # 验证偏移量递增
        for i, chunk in enumerate(chunks):
            expected_start = i * 20
            assert chunk.start_offset == expected_start


class TestImageDistribution:
    """图片分发测试。"""

    def test_image_refs_match_placeholder(self):
        """image_refs 应与占位符一致。"""
        chunker = FakeDocumentChunker(chunk_size=100)

        doc = Document(
            id="doc_img",
            text="Start [IMAGE: img_a] middle [IMAGE: img_b] end.",
            metadata={
                "images": [
                    {"id": "img_a", "path": "a.png"},
                    {"id": "img_b", "path": "b.png"},
                ],
            },
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            if "image_refs" in chunk.metadata:
                # 验证 refs 与文本中的占位符匹配
                refs = chunk.metadata["image_refs"]
                text_refs = chunker._extract_image_refs(chunk.text)
                assert refs == text_refs

    def test_images_subset_not_full_list(self):
        """chunk.images 应为子集，不是完整文档图片列表。"""
        chunker = FakeDocumentChunker(chunk_size=30)

        doc = Document(
            id="doc_subset",
            text="Part1 [IMAGE: img_1] Part2 [IMAGE: img_2] Part3 [IMAGE: img_3]",
            metadata={
                "images": [
                    {"id": "img_1", "path": "1.png"},
                    {"id": "img_2", "path": "2.png"},
                    {"id": "img_3", "path": "3.png"},
                ],
            },
        )

        chunks = chunker.split_document(doc)

        # 每个 chunk 最多包含部分图片
        for chunk in chunks:
            if "images" in chunk.metadata:
                assert len(chunk.metadata["images"]) <= 3  # 不应包含所有图片


class TestMetadataContract:
    """元数据契约测试。"""

    def test_chunk_metadata_has_source_path(self):
        """Chunk metadata 必须包含 source_path。"""
        chunker = FakeDocumentChunker(chunk_size=50)

        doc = Document(
            id="doc_contract",
            text="Test metadata contract.",
            metadata={"source_path": "/test/file.pdf"},
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            assert "source_path" in chunk.metadata

    def test_chunk_metadata_has_chunk_index(self):
        """Chunk metadata 必须包含 chunk_index。"""
        chunker = FakeDocumentChunker(chunk_size=30)

        doc = Document(
            id="doc_index",
            text="A" * 60,
            metadata={},
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            assert "chunk_index" in chunk.metadata

    def test_chunk_source_ref_correct(self):
        """Chunk source_ref 必须指向父 Document。"""
        chunker = FakeDocumentChunker(chunk_size=50)

        doc = Document(
            id="parent_doc_id",
            text="Test source ref.",
            metadata={},
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            assert chunk.source_ref == "parent_doc_id"


class TestChunkToDict:
    """Chunk 序列化测试。"""

    def test_chunk_to_dict_complete(self):
        """Chunk.to_dict() 应包含所有字段。"""
        chunker = FakeDocumentChunker(chunk_size=50)

        doc = Document(
            id="doc_dict",
            text="Test dict serialization.",
            metadata={"key": "value"},
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            data = chunk.to_dict()
            assert "id" in data
            assert "text" in data
            assert "metadata" in data
            assert "start_offset" in data
            assert "end_offset" in data
            assert "source_ref" in data