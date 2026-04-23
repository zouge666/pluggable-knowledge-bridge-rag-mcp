"""
核心数据类型单元测试。

验证 Document/Chunk/ChunkRecord/RetrievalResult/ImageRef 的序列化和字段约束。
"""

import pytest
from src.core.types import (
    Document,
    Chunk,
    ChunkRecord,
    RetrievalResult,
    ImageRef,
    compute_sha256,
    compute_file_hash,
    generate_chunk_id,
    generate_image_id,
)


class TestImageRef:
    """ImageRef 测试。"""

    def test_image_ref_creation(self):
        """应该能创建 ImageRef。"""
        img = ImageRef(
            id="abc123_0_1",
            path="data/images/default/abc123_0_1.png",
            page=0,
            text_offset=100,
            text_length=20,
        )

        assert img.id == "abc123_0_1"
        assert img.path == "data/images/default/abc123_0_1.png"
        assert img.page == 0
        assert img.text_offset == 100
        assert img.text_length == 20

    def test_image_ref_to_dict(self):
        """应该能转换为字典。"""
        img = ImageRef(
            id="test_id",
            path="data/images/test.png",
            page=1,
            text_offset=50,
            text_length=15,
            position={"x": 100, "y": 200},
        )

        result = img.to_dict()

        assert result["id"] == "test_id"
        assert result["path"] == "data/images/test.png"
        assert result["page"] == 1
        assert result["text_offset"] == 50
        assert result["text_length"] == 15
        assert result["position"] == {"x": 100, "y": 200}

    def test_image_ref_to_dict_minimal(self):
        """应该能转换最简 ImageRef 为字典。"""
        img = ImageRef(id="test", path="test.png")

        result = img.to_dict()

        assert result["id"] == "test"
        assert result["path"] == "test.png"
        assert "page" not in result  # None 值不包含
        assert "position" not in result

    def test_image_ref_from_dict(self):
        """应该能从字典创建 ImageRef。"""
        data = {
            "id": "test_id",
            "path": "test.png",
            "page": 2,
            "text_offset": 100,
            "text_length": 20,
        }

        img = ImageRef.from_dict(data)

        assert img.id == "test_id"
        assert img.path == "test.png"
        assert img.page == 2


class TestDocument:
    """Document 测试。"""

    def test_document_creation(self):
        """应该能创建 Document。"""
        doc = Document(
            id="doc_001",
            text="This is a test document.",
            metadata={"source_path": "/path/to/file.pdf"},
        )

        assert doc.id == "doc_001"
        assert doc.text == "This is a test document."
        assert doc.metadata["source_path"] == "/path/to/file.pdf"

    def test_document_to_dict(self):
        """应该能转换为字典。"""
        doc = Document(
            id="doc_001",
            text="Test content",
            metadata={"source_path": "test.pdf", "title": "Test Doc"},
        )

        result = doc.to_dict()

        assert result["id"] == "doc_001"
        assert result["text"] == "Test content"
        assert result["metadata"]["source_path"] == "test.pdf"

    def test_document_from_dict(self):
        """应该能从字典创建 Document。"""
        data = {
            "id": "doc_002",
            "text": "Another doc",
            "metadata": {"source_path": "another.pdf"},
        }

        doc = Document.from_dict(data)

        assert doc.id == "doc_002"
        assert doc.text == "Another doc"

    def test_document_json_serialization(self):
        """应该能序列化为 JSON。"""
        doc = Document(
            id="doc_003",
            text="JSON test",
            metadata={"key": "value"},
        )

        json_str = doc.to_json()
        restored = Document.from_json(json_str)

        assert restored.id == doc.id
        assert restored.text == doc.text
        assert restored.metadata == doc.metadata

    def test_document_get_source_path(self):
        """应该能获取来源路径。"""
        doc = Document(
            id="doc",
            text="text",
            metadata={"source_path": "/path/to/file.pdf"},
        )

        assert doc.get_source_path() == "/path/to/file.pdf"

    def test_document_get_images(self):
        """应该能获取图片引用列表。"""
        doc = Document(
            id="doc",
            text="Text [IMAGE: img_001] more text",
            metadata={
                "source_path": "test.pdf",
                "images": [
                    {"id": "img_001", "path": "data/images/img_001.png", "page": 1},
                ],
            },
        )

        images = doc.get_images()

        assert len(images) == 1
        assert images[0].id == "img_001"

    def test_document_with_image_placeholder(self):
        """文档文本应支持图片占位符。"""
        doc = Document(
            id="doc_with_images",
            text="This is text before image.\n[IMAGE: abc123_0_1]\nThis is text after image.",
            metadata={
                "source_path": "test.pdf",
                "images": [
                    {
                        "id": "abc123_0_1",
                        "path": "data/images/default/abc123_0_1.png",
                        "page": 0,
                        "text_offset": 28,
                        "text_length": 19,
                    },
                ],
            },
        )

        assert "[IMAGE: abc123_0_1]" in doc.text
        images = doc.get_images()
        assert len(images) == 1
        assert images[0].text_offset == 28


class TestChunk:
    """Chunk 测试。"""

    def test_chunk_creation(self):
        """应该能创建 Chunk。"""
        chunk = Chunk(
            id="doc_001_0000_abcd1234",
            text="This is a chunk.",
            metadata={"source_path": "test.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=16,
            source_ref="doc_001",
        )

        assert chunk.id == "doc_001_0000_abcd1234"
        assert chunk.text == "This is a chunk."
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.source_ref == "doc_001"

    def test_chunk_to_dict(self):
        """应该能转换为字典。"""
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={"key": "value"},
            start_offset=100,
            end_offset=200,
            source_ref="doc_001",
        )

        result = chunk.to_dict()

        assert result["id"] == "chunk_001"
        assert result["start_offset"] == 100
        assert result["end_offset"] == 200
        assert result["source_ref"] == "doc_001"

    def test_chunk_from_dict(self):
        """应该能从字典创建 Chunk。"""
        data = {
            "id": "chunk_002",
            "text": "Text",
            "metadata": {"chunk_index": 1},
            "start_offset": 50,
            "end_offset": 100,
            "source_ref": "doc_002",
        }

        chunk = Chunk.from_dict(data)

        assert chunk.id == "chunk_002"
        assert chunk.start_offset == 50

    def test_chunk_json_serialization(self):
        """应该能序列化为 JSON。"""
        chunk = Chunk(
            id="chunk_003",
            text="JSON chunk",
            metadata={"index": 0},
        )

        json_str = chunk.to_json()
        restored = Chunk.from_json(json_str)

        assert restored.id == chunk.id
        assert restored.text == chunk.text

    def test_chunk_get_chunk_index(self):
        """应该能获取 chunk 序号。"""
        chunk = Chunk(
            id="chunk",
            text="text",
            metadata={"chunk_index": 5},
        )

        assert chunk.get_chunk_index() == 5

    def test_chunk_get_images(self):
        """应该能获取 chunk 中的图片引用。"""
        chunk = Chunk(
            id="chunk_with_images",
            text="Text [IMAGE: img_001] more",
            metadata={
                "images": [{"id": "img_001", "path": "img.png"}],
                "image_refs": ["img_001"],
            },
        )

        images = chunk.get_images()
        assert len(images) == 1
        assert chunk.get_image_refs() == ["img_001"]


class TestChunkRecord:
    """ChunkRecord 测试。"""

    def test_chunk_record_creation(self):
        """应该能创建 ChunkRecord。"""
        record = ChunkRecord(
            id="chunk_001",
            text="Chunk text",
            metadata={"source": "test.pdf"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"term1": 0.5, "term2": 0.3},
        )

        assert record.id == "chunk_001"
        assert record.dense_vector == [0.1, 0.2, 0.3]
        assert record.sparse_vector["term1"] == 0.5

    def test_chunk_record_to_dict(self):
        """应该能转换为字典（不含向量）。"""
        record = ChunkRecord(
            id="record_001",
            text="Text",
            metadata={},
            dense_vector=[0.1, 0.2],
            sparse_vector={"term": 0.5},
        )

        result = record.to_dict()

        # to_dict 不包含向量数据，只包含标记
        assert "dense_vector" not in result
        assert "sparse_vector" not in result
        assert result["has_dense_vector"] is True
        assert result["has_sparse_vector"] is True

    def test_chunk_record_to_storage_dict(self):
        """应该能转换为存储格式（含向量）。"""
        record = ChunkRecord(
            id="record_001",
            text="Text",
            metadata={},
            dense_vector=[0.1, 0.2],
            sparse_vector={"term": 0.5},
        )

        result = record.to_storage_dict()

        assert result["dense_vector"] == [0.1, 0.2]
        assert result["sparse_vector"] == {"term": 0.5}

    def test_chunk_record_from_chunk(self):
        """应该能从 Chunk 创建 ChunkRecord。"""
        chunk = Chunk(
            id="chunk_001",
            text="Chunk text",
            metadata={"source_path": "test.pdf", "chunk_index": 0},
        )

        record = ChunkRecord.from_chunk(chunk)

        assert record.id == "chunk_001"
        assert record.text == "Chunk text"
        assert record.dense_vector is None
        assert record.sparse_vector is None


class TestRetrievalResult:
    """RetrievalResult 测试。"""

    def test_retrieval_result_creation(self):
        """应该能创建 RetrievalResult。"""
        result = RetrievalResult(
            chunk_id="chunk_001",
            score=0.95,
            text="Retrieved text",
            metadata={"source_path": "test.pdf"},
        )

        assert result.chunk_id == "chunk_001"
        assert result.score == 0.95
        assert result.text == "Retrieved text"

    def test_retrieval_result_to_dict(self):
        """应该能转换为字典。"""
        result = RetrievalResult(
            chunk_id="chunk_001",
            score=0.85,
            text="Text",
            metadata={"page": 5},
        )

        data = result.to_dict()

        assert data["chunk_id"] == "chunk_001"
        assert data["score"] == 0.85

    def test_retrieval_result_from_dict(self):
        """应该能从字典创建 RetrievalResult。"""
        data = {
            "chunk_id": "chunk_002",
            "score": 0.75,
            "text": "Text",
            "metadata": {},
        }

        result = RetrievalResult.from_dict(data)

        assert result.chunk_id == "chunk_002"
        assert result.score == 0.75


class TestUtilityFunctions:
    """工具函数测试。"""

    def test_compute_sha256_string(self):
        """应该能计算字符串的 SHA256。"""
        hash1 = compute_sha256("hello world")
        hash2 = compute_sha256("hello world")

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 输出 64 个十六进制字符

    def test_compute_sha256_bytes(self):
        """应该能计算字节的 SHA256。"""
        hash1 = compute_sha256(b"hello world")
        hash2 = compute_sha256("hello world")

        assert hash1 == hash2

    def test_compute_sha256_different(self):
        """不同内容应该产生不同哈希。"""
        hash1 = compute_sha256("content 1")
        hash2 = compute_sha256("content 2")

        assert hash1 != hash2

    def test_compute_file_hash(self, tmp_path):
        """应该能计算文件的 SHA256。"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")

        file_hash = compute_file_hash(file_path)
        expected_hash = compute_sha256("hello world")

        assert file_hash == expected_hash

    def test_generate_chunk_id(self):
        """应该能生成 Chunk ID。"""
        chunk_id = generate_chunk_id("doc_001", 0, "This is chunk text.")

        assert chunk_id.startswith("doc_001_0000_")
        assert len(chunk_id) == len("doc_001_0000_") + 8

    def test_generate_chunk_id_deterministic(self):
        """相同输入应该生成相同 Chunk ID。"""
        id1 = generate_chunk_id("doc_001", 0, "text")
        id2 = generate_chunk_id("doc_001", 0, "text")

        assert id1 == id2

    def test_generate_chunk_id_different_index(self):
        """不同序号应该生成不同 Chunk ID。"""
        id1 = generate_chunk_id("doc_001", 0, "text")
        id2 = generate_chunk_id("doc_001", 1, "text")

        assert id1 != id2

    def test_generate_image_id(self):
        """应该能生成图片 ID。"""
        image_id = generate_image_id("abc123", 0, 1)

        assert image_id == "abc123_0_1"


class TestMetadataContract:
    """元数据契约测试。"""

    def test_document_metadata_has_source_path(self):
        """Document metadata 必须包含 source_path。"""
        doc = Document(
            id="doc_001",
            text="text",
            metadata={"source_path": "/path/to/file.pdf"},
        )

        assert "source_path" in doc.metadata

    def test_chunk_metadata_inherits_source_path(self):
        """Chunk metadata 应继承 source_path。"""
        chunk = Chunk(
            id="chunk_001",
            text="text",
            metadata={"source_path": "/path/to/file.pdf", "chunk_index": 0},
        )

        assert "source_path" in chunk.metadata
        assert "chunk_index" in chunk.metadata

    def test_image_metadata_structure(self):
        """图片元数据结构应符合规范。"""
        images = [
            {
                "id": "abc123_0_1",
                "path": "data/images/default/abc123_0_1.png",
                "page": 0,
                "text_offset": 100,
                "text_length": 19,
                "position": {"x": 100, "y": 200, "width": 300, "height": 200},
            },
        ]

        doc = Document(
            id="doc_001",
            text="Text [IMAGE: abc123_0_1] more",
            metadata={"source_path": "test.pdf", "images": images},
        )

        restored_images = doc.get_images()
        assert len(restored_images) == 1
        img = restored_images[0]
        assert img.id == "abc123_0_1"
        assert img.page == 0
        assert img.text_offset == 100
        assert img.position is not None
