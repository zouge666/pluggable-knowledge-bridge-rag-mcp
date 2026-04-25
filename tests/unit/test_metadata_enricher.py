"""
MetadataEnricher 单元测试。

测试规则增强和 LLM 增强功能。
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.metadata_enricher import (
    MetadataEnricher,
    FakeMetadataEnricher,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_settings():
    """创建 Mock Settings。"""
    settings = Mock()
    settings.ingestion = Mock()
    settings.ingestion.metadata_enricher = {"use_llm": False}
    return settings


@pytest.fixture
def mock_settings_with_llm():
    """创建启用 LLM 的 Mock Settings。"""
    settings = Mock()
    settings.ingestion = Mock()
    settings.ingestion.metadata_enricher = {"use_llm": True}
    return settings


@pytest.fixture
def mock_llm():
    """创建 Mock LLM。"""
    llm = Mock()
    llm.chat = Mock(return_value=Mock(content=json.dumps({
        "title": "Test Title",
        "summary": "Test summary content.",
        "tags": ["test", "example"],
    })))
    return llm


@pytest.fixture
def sample_chunks():
    """创建测试用 Chunk 列表。"""
    return [
        Chunk(
            id="chunk_001",
            text="# Introduction\n\nThis is the introduction section.",
            metadata={"source_path": "test.pdf", "chunk_index": 0},
        ),
        Chunk(
            id="chunk_002",
            text="## Configuration\n\nTo configure the system, edit the config file.",
            metadata={"source_path": "test.pdf", "chunk_index": 1},
        ),
        Chunk(
            id="chunk_003",
            text="```python\ndef hello():\n    print('Hello')\n```",
            metadata={"source_path": "test.pdf", "chunk_index": 2},
        ),
        Chunk(
            id="chunk_004",
            text="",
            metadata={"source_path": "test.pdf", "chunk_index": 3},
        ),
    ]


@pytest.fixture
def trace_context():
    """创建 TraceContext。"""
    return TraceContext(trace_type="ingestion")


# ============================================================================
# 规则增强测试
# ============================================================================


class TestRuleBasedEnrichment:
    """规则增强测试。"""

    def test_extract_title_from_markdown_heading(self, mock_settings):
        """测试从 Markdown 标题提取标题。"""
        enricher = MetadataEnricher(mock_settings)

        text = "# API Reference\n\nThis section describes the API."
        result = enricher._rule_based_enrich(text)

        assert result["title"] == "API Reference"

    def test_extract_title_from_subheading(self, mock_settings):
        """测试从子标题提取标题。"""
        enricher = MetadataEnricher(mock_settings)

        text = "## Installation Guide\n\nFollow these steps."
        result = enricher._rule_based_enrich(text)

        assert result["title"] == "Installation Guide"

    def test_extract_title_from_first_line(self, mock_settings):
        """测试从第一行提取标题（无 Markdown 标题时）。"""
        enricher = MetadataEnricher(mock_settings)

        text = "This is the first line.\nThis is the second line."
        result = enricher._rule_based_enrich(text)

        assert result["title"] == "This is the first line."

    def test_extract_title_empty_text(self, mock_settings):
        """测试空文本的标题提取。"""
        enricher = MetadataEnricher(mock_settings)

        result = enricher._rule_based_enrich("")

        assert result["title"] == ""

    def test_generate_summary_short_text(self, mock_settings):
        """测试短文本摘要生成。"""
        enricher = MetadataEnricher(mock_settings)

        text = "This is a short text."
        result = enricher._rule_based_enrich(text)

        assert result["summary"] == "This is a short text."

    def test_generate_summary_long_text(self, mock_settings):
        """测试长文本摘要生成。"""
        enricher = MetadataEnricher(mock_settings)

        text = "A" * 300
        result = enricher._rule_based_enrich(text)

        assert len(result["summary"]) <= 203  # 200 + "..."
        assert result["summary"].endswith("...")

    def test_extract_tags_from_headings(self, mock_settings):
        """测试从标题提取标签。"""
        enricher = MetadataEnricher(mock_settings)

        text = "# API Reference\n\n## Configuration\n\n### Security"
        result = enricher._rule_based_enrich(text)

        assert "api" in result["tags"]
        assert "reference" in result["tags"]
        assert "configuration" in result["tags"]
        assert "security" in result["tags"]

    def test_extract_tags_from_code_blocks(self, mock_settings):
        """测试从代码块提取标签。"""
        enricher = MetadataEnricher(mock_settings)

        text = "```python\ndef hello():\n    pass\n```\n\n```javascript\nconsole.log('hi');\n```"
        result = enricher._rule_based_enrich(text)

        assert "code:python" in result["tags"]
        assert "code:javascript" in result["tags"]

    def test_extract_tags_from_keywords(self, mock_settings):
        """测试从关键词提取标签。"""
        enricher = MetadataEnricher(mock_settings)

        text = "This section describes the API configuration and installation process."
        result = enricher._rule_based_enrich(text)

        assert "api" in result["tags"]
        assert "configuration" in result["tags"]
        assert "installation" in result["tags"]

    def test_tags_limit(self, mock_settings):
        """测试标签数量限制。"""
        enricher = MetadataEnricher(mock_settings)

        # 包含多个关键词的文本
        text = """
        # API Configuration Installation Setup Example Tutorial
        ## Guide Reference Architecture Deployment Security Authentication
        ### Database Model Training Inference
        """
        result = enricher._rule_based_enrich(text)

        assert len(result["tags"]) <= 10

    def test_transform_chunks(self, mock_settings, sample_chunks):
        """测试转换 Chunk 列表。"""
        enricher = MetadataEnricher(mock_settings)
        result = enricher.transform(sample_chunks)

        assert len(result) == len(sample_chunks)

        # 验证第一个 chunk
        chunk = result[0]
        assert chunk.metadata["title"] == "Introduction"
        assert "introduction" in chunk.metadata["tags"]
        assert chunk.metadata["enriched_by"] == "rule"

        # 验证第二个 chunk
        chunk = result[1]
        assert chunk.metadata["title"] == "Configuration"
        assert "configuration" in chunk.metadata["tags"]

        # 验证第三个 chunk（代码块）
        chunk = result[2]
        assert "code:python" in chunk.metadata["tags"]

    def test_transform_preserves_chunk_fields(self, mock_settings, sample_chunks):
        """测试转换保留 Chunk 其他字段。"""
        enricher = MetadataEnricher(mock_settings)
        result = enricher.transform(sample_chunks)

        for i, chunk in enumerate(result):
            assert chunk.id == sample_chunks[i].id
            assert chunk.text == sample_chunks[i].text
            assert chunk.start_offset == sample_chunks[i].start_offset
            assert chunk.end_offset == sample_chunks[i].end_offset
            assert chunk.source_ref == sample_chunks[i].source_ref

    def test_transform_with_trace(self, mock_settings, sample_chunks, trace_context):
        """测试转换时记录追踪。"""
        enricher = MetadataEnricher(mock_settings)
        result = enricher.transform(sample_chunks, trace_context)

        # 验证追踪记录
        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "metadata_enrichment" for s in stages)


# ============================================================================
# LLM 增强测试
# ============================================================================


class TestLLMEnrichment:
    """LLM 增强测试。"""

    def test_llm_enrich_success(self, mock_settings_with_llm, mock_llm):
        """测试 LLM 增强成功。"""
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)

        text = "# Test Document\n\nThis is a test document."
        result = enricher._llm_enrich(text)

        assert result is not None
        assert result["title"] == "Test Title"
        assert result["summary"] == "Test summary content."
        assert result["tags"] == ["test", "example"]

    def test_llm_enrich_empty_text(self, mock_settings_with_llm, mock_llm):
        """测试 LLM 增强空文本。"""
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)

        result = enricher._llm_enrich("")

        assert result is None

    def test_llm_enrich_json_with_markdown(self, mock_settings_with_llm):
        """测试 LLM 返回带 markdown 代码块的 JSON。"""
        llm = Mock()
        llm.chat = Mock(return_value=Mock(content="""```json
{
    "title": "Markdown Title",
    "summary": "Markdown summary.",
    "tags": ["markdown", "test"]
}
```"""))

        enricher = MetadataEnricher(mock_settings_with_llm, llm=llm)

        result = enricher._llm_enrich("Test text")

        assert result is not None
        assert result["title"] == "Markdown Title"

    def test_llm_enrich_parse_error(self, mock_settings_with_llm):
        """测试 LLM 返回无效 JSON 时的降级。"""
        llm = Mock()
        llm.chat = Mock(return_value=Mock(content="Invalid JSON"))

        enricher = MetadataEnricher(mock_settings_with_llm, llm=llm)

        result = enricher._llm_enrich("Test text")

        assert result is None

    def test_llm_enrich_api_error(self, mock_settings_with_llm):
        """测试 LLM API 错误时的降级。"""
        llm = Mock()
        llm.chat = Mock(side_effect=Exception("API Error"))

        enricher = MetadataEnricher(mock_settings_with_llm, llm=llm)

        result = enricher._llm_enrich("Test text")

        assert result is None

    def test_transform_with_llm_success(self, mock_settings_with_llm, mock_llm, sample_chunks):
        """测试使用 LLM 转换 Chunk 列表。"""
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)
        result = enricher.transform(sample_chunks[:1])  # 只测试一个

        chunk = result[0]
        assert chunk.metadata["title"] == "Test Title"
        assert chunk.metadata["summary"] == "Test summary content."
        assert chunk.metadata["tags"] == ["test", "example"]
        assert chunk.metadata["enriched_by"] == "llm"

    def test_transform_llm_fallback_to_rule(self, mock_settings_with_llm, sample_chunks):
        """测试 LLM 失败时降级到规则模式。"""
        llm = Mock()
        llm.chat = Mock(side_effect=Exception("API Error"))

        enricher = MetadataEnricher(mock_settings_with_llm, llm=llm)
        result = enricher.transform(sample_chunks[:1])

        chunk = result[0]
        assert chunk.metadata["enriched_by"] == "rule"
        assert chunk.metadata["title"] == "Introduction"  # 规则提取的标题


# ============================================================================
# 配置测试
# ============================================================================


class TestConfiguration:
    """配置测试。"""

    def test_use_llm_false(self, mock_settings):
        """测试 use_llm=False 时不使用 LLM。"""
        enricher = MetadataEnricher(mock_settings)

        assert enricher.use_llm is False

    def test_use_llm_true_with_llm(self, mock_settings_with_llm, mock_llm):
        """测试 use_llm=True 且提供 LLM 时使用 LLM。"""
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)

        assert enricher.use_llm is True

    def test_use_llm_true_without_llm_fallback(self, mock_settings_with_llm):
        """测试 use_llm=True 但 LLM 创建失败时降级。"""
        with patch("src.ingestion.transform.metadata_enricher.LLMFactory.create") as mock_create:
            mock_create.side_effect = Exception("LLM creation failed")

            enricher = MetadataEnricher(mock_settings_with_llm)

            assert enricher.use_llm is False

    def test_custom_prompt_path(self, mock_settings, tmp_path):
        """测试自定义 Prompt 路径。"""
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text("Custom prompt: {text}")

        enricher = MetadataEnricher(mock_settings, prompt_path=str(prompt_file))

        assert "Custom prompt" in enricher._prompt_template


# ============================================================================
# FakeMetadataEnricher 测试
# ============================================================================


class TestFakeMetadataEnricher:
    """Fake MetadataEnricher 测试。"""

    def test_transform(self, sample_chunks):
        """测试 Fake 转换。"""
        enricher = FakeMetadataEnricher()
        result = enricher.transform(sample_chunks)

        assert len(result) == len(sample_chunks)

        for chunk in result:
            assert "title" in chunk.metadata
            assert "summary" in chunk.metadata
            assert "tags" in chunk.metadata
            assert chunk.metadata["enriched_by"] == "rule"

    def test_transform_with_trace(self, sample_chunks, trace_context):
        """测试 Fake 转换记录追踪。"""
        enricher = FakeMetadataEnricher()
        result = enricher.transform(sample_chunks, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "metadata_enrichment" for s in stages)

    def test_transform_empty_chunk(self):
        """测试 Fake 转换空 Chunk。"""
        enricher = FakeMetadataEnricher()

        chunks = [Chunk(id="empty", text="", metadata={})]
        result = enricher.transform(chunks)

        assert result[0].metadata["title"] == ""
        assert result[0].metadata["summary"] == ""
        assert result[0].metadata["tags"] == []


# ============================================================================
# 边界条件测试
# ============================================================================


class TestEdgeCases:
    """边界条件测试。"""

    def test_empty_chunks_list(self, mock_settings):
        """测试空 Chunk 列表。"""
        enricher = MetadataEnricher(mock_settings)
        result = enricher.transform([])

        assert result == []

    def test_chunk_with_existing_metadata(self, mock_settings):
        """测试 Chunk 已有元数据时的合并。"""
        enricher = MetadataEnricher(mock_settings)

        chunk = Chunk(
            id="test",
            text="# Title\n\nContent",
            metadata={"existing_field": "value", "source_path": "test.pdf"},
        )

        result = enricher.transform([chunk])

        assert result[0].metadata["existing_field"] == "value"
        assert result[0].metadata["source_path"] == "test.pdf"
        assert result[0].metadata["title"] == "Title"

    def test_very_long_title(self, mock_settings):
        """测试超长标题截断。"""
        enricher = MetadataEnricher(mock_settings)

        text = "# " + "A" * 200
        result = enricher._rule_based_enrich(text)

        assert len(result["title"]) <= 100

    def test_special_characters_in_text(self, mock_settings):
        """测试特殊字符处理。"""
        enricher = MetadataEnricher(mock_settings)

        text = "# Title with 特殊字符 & symbols!@#$%\n\nContent with émojis 🎉"
        result = enricher._rule_based_enrich(text)

        assert "特殊字符" in result["title"]
        assert result["summary"]  # 应该能生成摘要

    def test_only_whitespace(self, mock_settings):
        """测试只有空白字符的文本。"""
        enricher = MetadataEnricher(mock_settings)

        result = enricher._rule_based_enrich("   \n\n   \t\t   ")

        assert result["title"] == ""
        assert result["summary"] == ""
