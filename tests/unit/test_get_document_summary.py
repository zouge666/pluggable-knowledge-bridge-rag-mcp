"""
Unit tests for get_document_summary tool.
"""

import pytest
from pathlib import Path
import tempfile
import json

from src.mcp_server.tools.get_document_summary import GetDocumentSummaryTool, create_handler


class TestGetDocumentSummaryTool:
    """Tests for GetDocumentSummaryTool."""

    def test_tool_schema(self):
        """Test tool schema."""
        tool = GetDocumentSummaryTool()
        schema = tool.get_schema()

        assert schema["name"] == "get_document_summary"
        assert "doc_id" in schema["inputSchema"]["properties"]
        assert schema["inputSchema"]["required"] == ["doc_id"]

    def test_create_handler(self):
        """Test create_handler function."""
        handler = create_handler()
        assert callable(handler)

    def test_execute_missing_doc_id(self):
        """Test execute with missing doc_id."""
        tool = GetDocumentSummaryTool()
        result = tool.execute("")

        assert "content" in result
        assert "缺少" in result["content"][0]["text"] or "失败" in result["content"][0]["text"]

    def test_execute_not_found(self):
        """Test execute with non-existent doc_id."""
        tool = GetDocumentSummaryTool()
        result = tool.execute("nonexistent_doc_12345")

        assert "content" in result
        assert result["structuredContent"]["found"] is False

    def test_execute_with_document(self):
        """Test execute with existing document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create collection and document
            col = Path(tmpdir) / "my_collection"
            col.mkdir()
            doc_path = col / "test_doc.pdf"
            doc_path.write_bytes(b"PDF content here")

            tool = GetDocumentSummaryTool()
            tool._documents_path = Path(tmpdir)

            result = tool.execute("test_doc")

        assert "content" in result
        assert result["structuredContent"]["found"] is True
        assert result["structuredContent"]["title"] == "test_doc"
        assert result["structuredContent"]["collection"] == "my_collection"
        assert result["structuredContent"]["file_type"] == "pdf"

    def test_execute_with_metadata(self):
        """Test execute with metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create collection and document
            col = Path(tmpdir) / "my_collection"
            col.mkdir()
            doc_path = col / "test_doc.md"
            doc_path.write_text("# Test Document")

            # Create metadata file
            metadata_path = col / "test_doc.metadata.json"
            metadata = {
                "title": "Custom Title",
                "summary": "This is a test summary.",
                "tags": ["test", "document"],
            }
            metadata_path.write_text(json.dumps(metadata))

            tool = GetDocumentSummaryTool()
            tool._documents_path = Path(tmpdir)

            result = tool.execute("test_doc")

        assert result["structuredContent"]["found"] is True
        assert result["structuredContent"]["title"] == "Custom Title"
        assert result["structuredContent"]["summary"] == "This is a test summary."
        assert result["structuredContent"]["tags"] == ["test", "document"]

    def test_execute_with_full_filename(self):
        """Test execute with full filename as doc_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            col = Path(tmpdir) / "collection"
            col.mkdir()
            doc_path = col / "report.pdf"
            doc_path.write_bytes(b"content")

            tool = GetDocumentSummaryTool()
            tool._documents_path = Path(tmpdir)

            # Use full filename
            result = tool.execute("report.pdf")

        assert result["structuredContent"]["found"] is True

    def test_handler_returns_dict(self):
        """Test handler returns dict."""
        handler = create_handler()
        result = handler({"doc_id": "test"})

        assert isinstance(result, dict)
        assert "content" in result

    def test_markdown_output_format(self):
        """Test markdown output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            col = Path(tmpdir) / "collection"
            col.mkdir()
            doc_path = col / "mydoc.md"
            doc_path.write_text("Content")

            tool = GetDocumentSummaryTool()
            tool._documents_path = Path(tmpdir)

            result = tool.execute("mydoc")

        text = result["content"][0]["text"]
        assert "# 文档摘要" in text
        assert "mydoc" in text
        assert "## 基本信息" in text

    def test_size_calculation(self):
        """Test size calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            col = Path(tmpdir) / "collection"
            col.mkdir()
            doc_path = col / "sizedoc.txt"
            doc_path.write_bytes(b"x" * 2048)  # 2 KB

            tool = GetDocumentSummaryTool()
            tool._documents_path = Path(tmpdir)

            result = tool.execute("sizedoc")

        assert result["structuredContent"]["size_kb"] == 2.0

    def test_search_across_collections(self):
        """Test search across multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two collections
            col1 = Path(tmpdir) / "collection1"
            col2 = Path(tmpdir) / "collection2"
            col1.mkdir()
            col2.mkdir()

            # Document in collection2
            doc_path = col2 / "target.md"
            doc_path.write_text("Target document")

            tool = GetDocumentSummaryTool()
            tool._documents_path = Path(tmpdir)

            result = tool.execute("target")

        assert result["structuredContent"]["found"] is True
        assert result["structuredContent"]["collection"] == "collection2"
