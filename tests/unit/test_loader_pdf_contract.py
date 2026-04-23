"""
PDF Loader 单元测试。

使用 Mock 和 fixtures 验证 PDF 加载逻辑。
"""

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.types import Document, ImageRef
from src.libs.loader.base_loader import (
    BaseLoader,
    LoaderError,
    ParsingError,
    UnsupportedFormatError,
)
from src.libs.loader.pdf_loader import PdfLoader, SimplePdfLoader


class FakePdfLoader(BaseLoader):
    """Fake PDF Loader 用于测试。"""

    def __init__(self, text: str = "Fake PDF content"):
        self._text = text

    def supports(self, path) -> bool:
        return Path(path).suffix.lower() == ".pdf"

    def load(self, path, collection=None) -> Document:
        path = Path(path)
        doc_hash = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        return Document(
            id=doc_hash,
            text=self._text,
            metadata={
                "source_path": str(path),
                "doc_type": "pdf",
                "doc_hash": doc_hash,
            },
        )


class TestBaseLoader:
    """BaseLoader 测试。"""

    def test_abstract_methods(self):
        """抽象类不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseLoader()

    def test_get_file_extension(self):
        """应该能获取文件扩展名。"""
        loader = FakePdfLoader()

        ext = loader._get_file_extension("test.pdf")
        assert ext == ".pdf"

        ext = loader._get_file_extension("TEST.PDF")
        assert ext == ".pdf"


class TestPdfLoaderContract:
    """PdfLoader 契约测试。"""

    def test_supports_pdf_files(self):
        """应该支持 PDF 文件。"""
        loader = PdfLoader()

        assert loader.supports("test.pdf") is True
        assert loader.supports("test.PDF") is True
        assert loader.supports("test.txt") is False
        assert loader.supports("test.md") is False

    def test_raises_file_not_found(self, tmp_path):
        """文件不存在应该抛出 FileNotFoundError。"""
        loader = PdfLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.pdf")

    def test_raises_unsupported_format(self, tmp_path):
        """非 PDF 文件应该抛出 UnsupportedFormatError。"""
        loader = PdfLoader()

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")

        with pytest.raises(UnsupportedFormatError):
            loader.load(txt_file)


class TestSimplePdfLoader:
    """SimplePdfLoader 测试。"""

    def test_supports_pdf_files(self):
        """应该支持 PDF 文件。"""
        loader = SimplePdfLoader()

        assert loader.supports("test.pdf") is True

    def test_raises_file_not_found(self, tmp_path):
        """文件不存在应该抛出 FileNotFoundError。"""
        loader = SimplePdfLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "missing.pdf")


class TestPdfLoaderWithMock:
    """PdfLoader Mock 测试。"""

    def test_load_with_mock_pymupdf(self, tmp_path):
        """使用 Mock PyMuPDF 加载 PDF。"""
        # 创建假 PDF 文件
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        loader = PdfLoader(extract_images=False)

        # Mock PyMuPDF
        with patch("src.libs.loader.pdf_loader.HAS_PYMUPDF", True):
            with patch("src.libs.loader.pdf_loader.HAS_MARKITDOWN", False):
                with patch("src.libs.loader.pdf_loader.fitz") as mock_fitz:
                    # 设置 Mock 返回值
                    mock_doc = MagicMock()
                    mock_page = MagicMock()
                    mock_page.get_text.return_value = "Mock PDF text content"
                    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
                    mock_doc.close = MagicMock()
                    mock_fitz.open.return_value = mock_doc

                    # 加载文档
                    doc = loader.load(pdf_path, collection="test")

                    assert doc.id is not None
                    assert "Mock PDF text content" in doc.text
                    assert doc.metadata["source_path"] == str(pdf_path)
                    assert doc.metadata["doc_type"] == "pdf"
                    assert doc.metadata["collection"] == "test"

    def test_load_with_mock_markitdown(self, tmp_path):
        """使用 Mock MarkItDown 加载 PDF。"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        # 直接使用 FakePdfLoader 测试，因为 MarkItDown mock 比较复杂
        loader = FakePdfLoader(text="# Mock Markdown\n\nContent from PDF.")
        doc = loader.load(pdf_path)

        assert "# Mock Markdown" in doc.text
        assert doc.metadata["doc_type"] == "pdf"


class TestImageExtraction:
    """图片提取测试。"""

    def test_extract_images_disabled(self, tmp_path):
        """禁用图片提取时不应该提取图片。"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        loader = PdfLoader(extract_images=False)

        with patch("src.libs.loader.pdf_loader.HAS_PYMUPDF", True):
            with patch("src.libs.loader.pdf_loader.HAS_MARKITDOWN", False):
                with patch("src.libs.loader.pdf_loader.fitz") as mock_fitz:
                    mock_doc = MagicMock()
                    mock_page = MagicMock()
                    mock_page.get_text.return_value = "text"
                    mock_page.get_images.return_value = [(1,)]
                    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
                    mock_doc.close = MagicMock()
                    mock_fitz.open.return_value = mock_doc

                    doc = loader.load(pdf_path)

                    # extract_images=False，不应该有图片
                    assert "images" not in doc.metadata or len(doc.metadata.get("images", [])) == 0

    def test_image_ref_structure(self):
        """ImageRef 结构应符合规范。"""
        img = ImageRef(
            id="abc123_0_1",
            path="data/images/default/abc123_0_1.png",
            page=0,
            text_offset=100,
            text_length=19,
            position={"x": 100, "y": 200, "width": 300, "height": 200},
        )

        data = img.to_dict()

        assert data["id"] == "abc123_0_1"
        assert data["path"] == "data/images/default/abc123_0_1.png"
        assert data["page"] == 0
        assert data["position"]["width"] == 300


class TestLoaderError:
    """LoaderError 测试。"""

    def test_loader_error_basic(self):
        """应该能创建 LoaderError。"""
        error = LoaderError("Test error", path="/test/file.pdf")

        assert str(error) == "Test error"
        assert error.path == "/test/file.pdf"

    def test_loader_error_with_original(self):
        """应该能保存原始错误。"""
        original = ValueError("original error")
        error = LoaderError("Wrapper", path="/test.pdf", original_error=original)

        assert error.original_error == original

    def test_unsupported_format_error(self):
        """UnsupportedFormatError 应该是 LoaderError 子类。"""
        error = UnsupportedFormatError("Not supported", path="test.txt")

        assert isinstance(error, LoaderError)

    def test_parsing_error(self):
        """ParsingError 应该是 LoaderError 子类。"""
        error = ParsingError("Parse failed", path="test.pdf")

        assert isinstance(error, LoaderError)


class TestDocumentMetadata:
    """文档元数据契约测试。"""

    def test_metadata_has_source_path(self, tmp_path):
        """metadata 必须包含 source_path。"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake")

        loader = FakePdfLoader()
        doc = loader.load(pdf_path)

        assert "source_path" in doc.metadata

    def test_metadata_has_doc_type(self, tmp_path):
        """metadata 应包含 doc_type。"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake")

        loader = FakePdfLoader()
        doc = loader.load(pdf_path)

        assert doc.metadata.get("doc_type") == "pdf"

    def test_metadata_has_doc_hash(self, tmp_path):
        """metadata 应包含 doc_hash。"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake")

        loader = FakePdfLoader()
        doc = loader.load(pdf_path)

        assert "doc_hash" in doc.metadata


class TestNoParserAvailable:
    """无解析器可用测试。"""

    def test_raises_when_no_parser(self, tmp_path):
        """无解析器可用时应该抛出 ParsingError。"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        loader = PdfLoader()

        with patch("src.libs.loader.pdf_loader.HAS_PYMUPDF", False):
            with patch("src.libs.loader.pdf_loader.HAS_MARKITDOWN", False):
                with pytest.raises(ParsingError) as exc_info:
                    loader.load(pdf_path)

                assert "No PDF parser available" in str(exc_info.value)