"""
Unit tests for list_collections tool.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.mcp_server.tools.list_collections import ListCollectionsTool, create_handler


class TestListCollectionsTool:
    """Tests for ListCollectionsTool."""

    def test_tool_schema(self):
        """Test tool schema."""
        tool = ListCollectionsTool()
        schema = tool.get_schema()

        assert schema["name"] == "list_collections"
        assert "include_stats" in schema["inputSchema"]["properties"]

    def test_create_handler(self):
        """Test create_handler function."""
        handler = create_handler()
        assert callable(handler)

    def test_execute_empty_collections(self):
        """Test execute with no collections."""
        with patch("pathlib.Path.exists", return_value=False):
            tool = ListCollectionsTool()
            result = tool.execute()

        assert "content" in result
        assert "structuredContent" in result
        assert result["structuredContent"]["total_collections"] == 0

    def test_execute_with_collections(self):
        """Test execute with collections."""
        from pathlib import Path
        import tempfile
        import os

        # Create temp directory with collections
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create collection directories
            col1 = Path(tmpdir) / "collection1"
            col2 = Path(tmpdir) / "collection2"
            col1.mkdir()
            col2.mkdir()

            # Create some files
            (col1 / "doc1.pdf").write_bytes(b"content1")
            (col2 / "doc2.md").write_bytes(b"content2")

            tool = ListCollectionsTool()
            tool._documents_path = Path(tmpdir)

            result = tool.execute()

        assert "content" in result
        assert result["structuredContent"]["total_collections"] == 2
        collections = result["structuredContent"]["collections"]
        assert len(collections) == 2
        names = [c["name"] for c in collections]
        assert "collection1" in names
        assert "collection2" in names

    def test_execute_with_stats(self):
        """Test execute with stats."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            col = Path(tmpdir) / "my_collection"
            col.mkdir()
            (col / "doc.pdf").write_bytes(b"x" * 1000)

            tool = ListCollectionsTool()
            tool._documents_path = Path(tmpdir)

            result = tool.execute(include_stats=True)

        collections = result["structuredContent"]["collections"]
        assert collections[0]["document_count"] == 1
        assert "total_size_bytes" in collections[0]
        assert "total_size_mb" in collections[0]

    def test_execute_without_stats(self):
        """Test execute without stats."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            col = Path(tmpdir) / "my_collection"
            col.mkdir()
            (col / "doc.pdf").write_bytes(b"content")

            tool = ListCollectionsTool()
            tool._documents_path = Path(tmpdir)

            result = tool.execute(include_stats=False)

        collections = result["structuredContent"]["collections"]
        assert "document_count" not in collections[0]

    def test_handler_returns_dict(self):
        """Test handler returns dict."""
        handler = create_handler()
        result = handler({})

        assert isinstance(result, dict)
        assert "content" in result

    def test_handler_with_arguments(self):
        """Test handler with arguments."""
        handler = create_handler()
        result = handler({"include_stats": False})

        assert isinstance(result, dict)
        assert "content" in result

    def test_collections_sorted_by_name(self):
        """Test collections are sorted by name."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create in non-alphabetical order
            (Path(tmpdir) / "zebra").mkdir()
            (Path(tmpdir) / "alpha").mkdir()
            (Path(tmpdir) / "middle").mkdir()

            tool = ListCollectionsTool()
            tool._documents_path = Path(tmpdir)

            result = tool.execute()

        collections = result["structuredContent"]["collections"]
        names = [c["name"] for c in collections]
        assert names == ["alpha", "middle", "zebra"]

    def test_markdown_output_format(self):
        """Test markdown output format."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            col = Path(tmpdir) / "test_collection"
            col.mkdir()
            (col / "doc.pdf").write_bytes(b"content")

            tool = ListCollectionsTool()
            tool._documents_path = Path(tmpdir)

            result = tool.execute()

        text = result["content"][0]["text"]
        assert "# 知识库集合列表" in text
        assert "test_collection" in text
        assert "共 1 个集合" in text
