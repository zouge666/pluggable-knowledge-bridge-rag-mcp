"""
Unit tests for ResponseBuilder and CitationGenerator.
"""

import pytest
from pathlib import Path

from src.core.types import RetrievalResult
from src.core.response import Citation, CitationGenerator, ResponseBuilder


class TestCitation:
    """Tests for Citation dataclass."""

    def test_to_dict_basic(self):
        """Test to_dict with basic fields."""
        citation = Citation(
            index=1,
            source="doc.pdf",
            page=5,
            chunk_id="chunk_123",
            score=0.95,
        )
        result = citation.to_dict()

        assert result["index"] == 1
        assert result["source"] == "doc.pdf"
        assert result["page"] == 5
        assert result["chunk_id"] == "chunk_123"
        assert result["score"] == 0.95
        assert "snippet" not in result

    def test_to_dict_with_snippet(self):
        """Test to_dict with snippet."""
        citation = Citation(
            index=2,
            source="doc.md",
            page=None,
            chunk_id="chunk_456",
            score=0.88,
            snippet="This is a snippet...",
        )
        result = citation.to_dict()

        assert result["snippet"] == "This is a snippet..."
        assert "page" not in result


class TestCitationGenerator:
    """Tests for CitationGenerator."""

    def test_generate_empty_results(self):
        """Test generate with empty results."""
        generator = CitationGenerator()
        citations = generator.generate([])
        assert citations == []

    def test_generate_single_result(self):
        """Test generate with single result."""
        generator = CitationGenerator()
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.95,
                text="This is the text content.",
                metadata={"source_path": "doc.pdf", "page": 1},
            ),
        ]

        citations = generator.generate(results)

        assert len(citations) == 1
        assert citations[0].index == 1
        assert citations[0].source == "doc.pdf"
        assert citations[0].page == 1
        assert citations[0].chunk_id == "c1"
        assert citations[0].score == 0.95

    def test_generate_multiple_results(self):
        """Test generate with multiple results."""
        generator = CitationGenerator()
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.95,
                text="Text 1",
                metadata={"source_path": "doc1.pdf"},
            ),
            RetrievalResult(
                chunk_id="c2",
                score=0.88,
                text="Text 2",
                metadata={"source_path": "doc2.pdf", "page": 5},
            ),
            RetrievalResult(
                chunk_id="c3",
                score=0.75,
                text="Text 3",
                metadata={"source_path": "doc3.md"},
            ),
        ]

        citations = generator.generate(results)

        assert len(citations) == 3
        assert citations[0].index == 1
        assert citations[1].index == 2
        assert citations[2].index == 3

    def test_generate_without_snippet(self):
        """Test generate without snippet."""
        generator = CitationGenerator()
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="Long text content here.",
                metadata={"source_path": "doc.pdf"},
            ),
        ]

        citations = generator.generate(results, include_snippet=False)

        assert citations[0].snippet is None

    def test_generate_with_snippet(self):
        """Test generate with snippet."""
        generator = CitationGenerator(snippet_length=20)
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="This is a very long text that should be truncated.",
                metadata={"source_path": "doc.pdf"},
            ),
        ]

        citations = generator.generate(results, include_snippet=True)

        assert citations[0].snippet is not None
        assert len(citations[0].snippet) <= 23  # 20 + "..."

    def test_generate_missing_metadata(self):
        """Test generate with missing metadata."""
        generator = CitationGenerator()
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="Text",
                metadata={},  # No source_path
            ),
        ]

        citations = generator.generate(results)

        assert citations[0].source == "unknown"
        assert citations[0].page is None


class TestResponseBuilder:
    """Tests for ResponseBuilder."""

    def test_build_empty_results(self):
        """Test build with empty results."""
        builder = ResponseBuilder()
        response = builder.build([], "test query")

        assert "content" in response
        assert "structuredContent" in response
        assert "未找到" in response["content"][0]["text"]
        assert response["structuredContent"]["total_results"] == 0
        assert response["structuredContent"]["citations"] == []

    def test_build_single_result(self):
        """Test build with single result."""
        builder = ResponseBuilder()
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.95,
                text="This is the answer.",
                metadata={"source_path": "doc.pdf", "page": 1},
            ),
        ]

        response = builder.build(results, "test query")

        assert "content" in response
        assert "structuredContent" in response
        assert response["structuredContent"]["total_results"] == 1
        assert len(response["structuredContent"]["citations"]) == 1

        # Check markdown content
        text = response["content"][0]["text"]
        assert "# 查询结果" in text
        assert "test query" in text
        assert "[1]" in text
        assert "doc.pdf" in text

    def test_build_multiple_results(self):
        """Test build with multiple results."""
        builder = ResponseBuilder()
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.95,
                text="Answer 1",
                metadata={"source_path": "doc1.pdf"},
            ),
            RetrievalResult(
                chunk_id="c2",
                score=0.88,
                text="Answer 2",
                metadata={"source_path": "doc2.pdf", "page": 5},
            ),
        ]

        response = builder.build(results, "query")

        assert response["structuredContent"]["total_results"] == 2
        assert len(response["structuredContent"]["citations"]) == 2

        text = response["content"][0]["text"]
        assert "[1]" in text
        assert "[2]" in text

    def test_build_without_citations(self):
        """Test build without citations."""
        builder = ResponseBuilder()
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="Text",
                metadata={"source_path": "doc.pdf"},
            ),
        ]

        response = builder.build(results, "query", include_citations=False)

        assert response["structuredContent"]["citations"] == []

    def test_build_markdown_structure(self):
        """Test markdown structure."""
        builder = ResponseBuilder()
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.95,
                text="The answer is 42.",
                metadata={"source_path": "guide.pdf", "page": 10},
            ),
        ]

        response = builder.build(results, "What is the answer?")
        text = response["content"][0]["text"]

        # Check structure
        assert "# 查询结果" in text
        assert "What is the answer?" in text
        assert "## 结果 1" in text
        assert "guide.pdf" in text
        assert "第 10 页" in text
        assert "0.95" in text
        assert "The answer is 42." in text
        assert "## 引用来源" in text

    def test_build_with_custom_citation_generator(self):
        """Test build with custom citation generator."""
        custom_generator = CitationGenerator(snippet_length=10)
        builder = ResponseBuilder(citation_generator=custom_generator)

        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="This is a very long text for testing.",
                metadata={"source_path": "doc.pdf"},
            ),
        ]

        response = builder.build(results, "query")
        citation = response["structuredContent"]["citations"][0]

        # Should have snippet truncated to 10 chars
        assert citation["snippet"] is not None
        assert len(citation["snippet"]) <= 13  # 10 + "..."


class TestResponseBuilderIntegration:
    """Integration tests for ResponseBuilder."""

    def test_full_flow(self):
        """Test full flow from results to response."""
        builder = ResponseBuilder()

        # Simulate retrieval results
        results = [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.95,
                text="Azure 配置需要以下步骤：首先创建资源组...",
                metadata={
                    "source_path": "docs/azure-setup.md",
                    "page": 1,
                    "collection": "tech-docs",
                },
            ),
            RetrievalResult(
                chunk_id="chunk_002",
                score=0.88,
                text="部署流程包括：1. 构建镜像 2. 推送仓库...",
                metadata={
                    "source_path": "docs/deploy.md",
                    "page": 5,
                    "collection": "tech-docs",
                },
            ),
        ]

        response = builder.build(results, "如何配置 Azure？")

        # Validate structure
        assert response["content"][0]["type"] == "text"
        assert response["structuredContent"]["query"] == "如何配置 Azure？"
        assert response["structuredContent"]["total_results"] == 2

        # Validate citations
        citations = response["structuredContent"]["citations"]
        assert len(citations) == 2
        assert citations[0]["source"] == "docs/azure-setup.md"
        assert citations[1]["source"] == "docs/deploy.md"

        # Validate markdown
        text = response["content"][0]["text"]
        assert "Azure" in text
        assert "部署流程" in text


class TestMultimodalAssembler:
    """Tests for MultimodalAssembler."""

    def test_assemble_text_only(self):
        """Test assemble with text only."""
        from src.core.response import MultimodalAssembler

        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="Text content",
                metadata={"source_path": "doc.pdf"},
            ),
        ]

        contents = assembler.assemble(results)

        assert len(contents) == 1
        assert contents[0]["type"] == "text"

    def test_assemble_with_images(self):
        """Test assemble with images."""
        import tempfile
        from src.core.response import MultimodalAssembler
        from src.core.types import ImageRef

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image file
            img_path = Path(tmpdir) / "test_image.png"
            img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 100)

            results = [
                RetrievalResult(
                    chunk_id="c1",
                    score=0.9,
                    text="Text with image",
                    metadata={
                        "source_path": "doc.pdf",
                        "images": [
                            {
                                "id": "test_image",
                                "path": str(img_path),
                            }
                        ],
                    },
                ),
            ]

            assembler = MultimodalAssembler()
            contents = assembler.assemble(results, include_images=True)

        assert len(contents) == 2
        assert contents[0]["type"] == "text"
        assert contents[1]["type"] == "image"
        assert contents[1]["mimeType"] == "image/png"

    def test_assemble_max_images(self):
        """Test assemble respects max_images."""
        import tempfile
        from src.core.response import MultimodalAssembler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple images
            for i in range(5):
                img_path = Path(tmpdir) / f"img{i}.png"
                img_path.write_bytes(b"\x89PNG" + b"x" * 50)

            results = [
                RetrievalResult(
                    chunk_id="c1",
                    score=0.9,
                    text="Text",
                    metadata={
                        "source_path": "doc.pdf",
                        "images": [
                            {"id": f"img{i}", "path": str(Path(tmpdir) / f"img{i}.png")}
                            for i in range(5)
                        ],
                    },
                ),
            ]

            assembler = MultimodalAssembler()
            contents = assembler.assemble(results, max_images=2)

        # 1 text + 2 images
        image_count = sum(1 for c in contents if c["type"] == "image")
        assert image_count == 2

    def test_assemble_without_images(self):
        """Test assemble with include_images=False."""
        from src.core.response import MultimodalAssembler

        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="Text",
                metadata={
                    "source_path": "doc.pdf",
                    "images": [{"id": "img1", "path": "/nonexistent.png"}],
                },
            ),
        ]

        assembler = MultimodalAssembler()
        contents = assembler.assemble(results, include_images=False)

        assert len(contents) == 1
        assert contents[0]["type"] == "text"

    def test_assemble_missing_image(self):
        """Test assemble with missing image file."""
        from src.core.response import MultimodalAssembler

        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="Text",
                metadata={
                    "source_path": "doc.pdf",
                    "images": [{"id": "missing", "path": "/nonexistent.png"}],
                },
            ),
        ]

        assembler = MultimodalAssembler()
        contents = assembler.assemble(results)

        # Only text, image not found
        assert len(contents) == 1
        assert contents[0]["type"] == "text"

    def test_get_mime_type(self):
        """Test MIME type detection."""
        from src.core.response import MultimodalAssembler

        assembler = MultimodalAssembler()

        assert assembler._get_mime_type(Path("test.png")) == "image/png"
        assert assembler._get_mime_type(Path("test.jpg")) == "image/jpeg"
        assert assembler._get_mime_type(Path("test.jpeg")) == "image/jpeg"
        assert assembler._get_mime_type(Path("test.gif")) == "image/gif"
        assert assembler._get_mime_type(Path("test.webp")) == "image/webp"
        assert assembler._get_mime_type(Path("test.unknown")) == "image/png"  # default


class TestAssembleMultimodalResponse:
    """Tests for assemble_multimodal_response function."""

    def test_assemble_response(self):
        """Test assemble_multimodal_response function."""
        from src.core.response import assemble_multimodal_response

        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.9,
                text="Answer text",
                metadata={"source_path": "doc.pdf"},
            ),
        ]

        response = assemble_multimodal_response(results, "test query")

        assert "content" in response
        assert "structuredContent" in response
        assert response["structuredContent"]["query"] == "test query"
        assert response["structuredContent"]["total_results"] == 1
        assert response["structuredContent"]["has_images"] is False

    def test_assemble_response_with_images(self):
        """Test assemble_multimodal_response with images."""
        import tempfile
        from src.core.response import assemble_multimodal_response

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "img.png"
            img_path.write_bytes(b"\x89PNG" + b"x" * 50)

            results = [
                RetrievalResult(
                    chunk_id="c1",
                    score=0.9,
                    text="Text",
                    metadata={
                        "source_path": "doc.pdf",
                        "images": [{"id": "img", "path": str(img_path)}],
                    },
                ),
            ]

            response = assemble_multimodal_response(results, "query")

        assert response["structuredContent"]["has_images"] is True
        assert len(response["content"]) == 3  # title + text + image