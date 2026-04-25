"""
Chunk Refiner 单元测试。

验证规则去噪和 LLM 增强功能。
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.settings import Settings, IngestionSettings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform import ChunkRefiner, FakeChunkRefiner, BaseTransform


# 加载测试 fixtures
FIXTURES_PATH = Path(__file__).parent.parent.parent / "fixtures" / "noisy_chunks.json"


def load_test_chunks():
    """加载测试数据。"""
    try:
        # 使用绝对路径
        abs_path = FIXTURES_PATH.resolve()
        if abs_path.exists():
            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["test_chunks"]
        else:
            # 尝试备用路径
            alt_path = Path("tests/fixtures/noisy_chunks.json").resolve()
            if alt_path.exists():
                with open(alt_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data["test_chunks"]
    except Exception as e:
        print(f"Error loading fixtures: {e}")
    return []


class TestFakeChunkRefiner:
    """FakeChunkRefiner 测试（不依赖真实 LLM）。"""

    def test_refine_basic(self):
        """应该能处理基本文本。"""
        refiner = FakeChunkRefiner()

        chunk = Chunk(
            id="test_001",
            text="This is a test chunk.",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert len(refined) == 1
        assert refined[0].text == "This is a test chunk."
        assert refined[0].metadata["refined_by"] == "rule"

    def test_refine_page_markers(self):
        """应该移除页眉页脚标记。"""
        refiner = FakeChunkRefiner()

        chunk = Chunk(
            id="test_002",
            text="--- Page 1 ---\n\nMain content here.\n\n--- Page 1 ---",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert "--- Page 1 ---" not in refined[0].text
        assert "Main content here." in refined[0].text

    def test_refine_whitespace(self):
        """应该处理多余空白。"""
        refiner = FakeChunkRefiner()

        chunk = Chunk(
            id="test_003",
            text="This    has    too    many    spaces.",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert "    " not in refined[0].text
        assert "This has too many spaces." in refined[0].text

    def test_refine_multiple_chunks(self):
        """应该处理多个 Chunk。"""
        refiner = FakeChunkRefiner()

        chunks = [
            Chunk(id="test_004a", text="First chunk.", metadata={}),
            Chunk(id="test_004b", text="Second chunk.", metadata={}),
            Chunk(id="test_004c", text="Third chunk.", metadata={}),
        ]

        refined = refiner.transform(chunks)

        assert len(refined) == 3
        assert all(c.metadata["refined_by"] == "rule" for c in refined)

    def test_metadata_preserved(self):
        """应该保留原有元数据。"""
        refiner = FakeChunkRefiner()

        chunk = Chunk(
            id="test_005",
            text="Test content.",
            metadata={"source_path": "test.pdf", "page": 5},
        )

        refined = refiner.transform([chunk])

        assert refined[0].metadata["source_path"] == "test.pdf"
        assert refined[0].metadata["page"] == 5
        assert refined[0].metadata["refined_by"] == "rule"

    def test_empty_text(self):
        """应该处理空文本。"""
        refiner = FakeChunkRefiner()

        chunk = Chunk(
            id="test_006",
            text="",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert len(refined) == 1
        assert refined[0].text == ""

    def test_trace_recorded(self):
        """应该记录追踪信息。"""
        refiner = FakeChunkRefiner()
        trace = TraceContext(trace_type="ingestion")

        chunk = Chunk(id="test_007", text="Test.", metadata={})
        refiner.transform([chunk], trace=trace)

        assert len(trace.stages) == 1
        assert trace.stages[0]["stage"] == "chunk_refinement"


class TestChunkRefinerRuleBased:
    """ChunkRefiner 规则模式测试。"""

    def test_rule_based_refine_page_markers(self):
        """规则模式应该移除页眉页脚。"""
        settings = Settings(
            ingestion=IngestionSettings(
                chunk_refiner={"use_llm": False}
            )
        )
        refiner = ChunkRefiner(settings)

        chunk = Chunk(
            id="test_008",
            text="--- Page 1 ---\n\nContent here.\n\nPage 1 of 10",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert "--- Page 1 ---" not in refined[0].text
        assert "Page 1 of 10" not in refined[0].text
        assert "Content here." in refined[0].text

    def test_rule_based_refine_html_comments(self):
        """规则模式应该移除 HTML 注释。"""
        settings = Settings(
            ingestion=IngestionSettings(
                chunk_refiner={"use_llm": False}
            )
        )
        refiner = ChunkRefiner(settings)

        chunk = Chunk(
            id="test_009",
            text="<!-- This is a comment -->\nReal content here.",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert "<!--" not in refined[0].text
        assert "Real content here." in refined[0].text

    def test_rule_based_refine_html_tags(self):
        """规则模式应该移除简单 HTML 标签。"""
        settings = Settings(
            ingestion=IngestionSettings(
                chunk_refiner={"use_llm": False}
            )
        )
        refiner = ChunkRefiner(settings)

        chunk = Chunk(
            id="test_010",
            text="<div><p>Content inside tags.</p></div>",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert "<div>" not in refined[0].text
        assert "<p>" not in refined[0].text
        assert "Content inside tags." in refined[0].text

    def test_rule_based_preserve_code_blocks(self):
        """规则模式应该保留代码块。"""
        settings = Settings(
            ingestion=IngestionSettings(
                chunk_refiner={"use_llm": False}
            )
        )
        refiner = ChunkRefiner(settings)

        chunk = Chunk(
            id="test_011",
            text="```python\ndef hello():\n    print('Hello')\n```\n\nText after code.",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert "```python" in refined[0].text
        assert "def hello():" in refined[0].text
        assert "print('Hello')" in refined[0].text

    def test_rule_based_multiple_blank_lines(self):
        """规则模式应该处理多个空行。"""
        settings = Settings(
            ingestion=IngestionSettings(
                chunk_refiner={"use_llm": False}
            )
        )
        refiner = ChunkRefiner(settings)

        chunk = Chunk(
            id="test_012",
            text="Line 1.\n\n\n\n\n\nLine 2.",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert "\n\n\n" not in refined[0].text
        assert "Line 1." in refined[0].text
        assert "Line 2." in refined[0].text


class TestChunkRefinerLLM:
    """ChunkRefiner LLM 模式测试。"""

    def test_llm_mode_with_mock(self):
        """LLM 模式应该调用 LLM。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(content="LLM refined text.")

        settings = Settings(
            ingestion=IngestionSettings(
                chunk_refiner={"use_llm": True}
            )
        )
        refiner = ChunkRefiner(settings, llm=mock_llm)

        chunk = Chunk(
            id="test_013",
            text="Original text.",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert mock_llm.chat.called
        assert refined[0].text == "LLM refined text."
        assert refined[0].metadata["refined_by"] == "llm"

    def test_llm_failure_fallback(self):
        """LLM 失败时应该降级到规则模式。"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("LLM error")

        settings = Settings(
            ingestion=IngestionSettings(
                chunk_refiner={"use_llm": True}
            )
        )
        refiner = ChunkRefiner(settings, llm=mock_llm)

        chunk = Chunk(
            id="test_014",
            text="Original text.",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert refined[0].metadata["refined_by"] == "rule"

    def test_llm_disabled_uses_rule(self):
        """禁用 LLM 时应该使用规则模式。"""
        mock_llm = MagicMock()

        settings = Settings(
            ingestion=IngestionSettings(
                chunk_refiner={"use_llm": False}
            )
        )
        refiner = ChunkRefiner(settings, llm=mock_llm)

        chunk = Chunk(
            id="test_015",
            text="Original text.",
            metadata={},
        )

        refined = refiner.transform([chunk])

        assert not mock_llm.chat.called
        assert refined[0].metadata["refined_by"] == "rule"


class TestNoisyChunksFixtures:
    """使用 fixtures 测试各种噪声场景。"""

    @pytest.fixture
    def test_chunks(self):
        """加载测试数据。"""
        chunks = load_test_chunks()
        return chunks

    def test_noisy_chunk_refinement(self, test_chunks):
        """测试各种噪声场景。"""
        if not test_chunks:
            pytest.skip("No test data available")

        settings = Settings(
            ingestion=IngestionSettings(
                chunk_refiner={"use_llm": False}
            )
        )
        refiner = ChunkRefiner(settings)

        for test_data in test_chunks:
            chunk = Chunk(
                id=test_data["id"],
                text=test_data["text"],
                metadata={},
            )

            refined = refiner.transform([chunk])

            # 验证精炼后的文本不为空
            assert refined[0].text.strip() != ""

            # 验证元数据正确
            assert refined[0].metadata["refined_by"] == "rule"


class TestBaseTransform:
    """BaseTransform 抽象类测试。"""

    def test_get_name(self):
        """应该返回类名。"""
        refiner = FakeChunkRefiner()
        assert refiner.get_name() == "FakeChunkRefiner"


class TestTraceIntegration:
    """追踪集成测试。"""

    def test_trace_records_elapsed_time(self):
        """追踪应该记录耗时。"""
        refiner = FakeChunkRefiner()
        trace = TraceContext(trace_type="ingestion")

        chunk = Chunk(id="test_trace", text="Test content.", metadata={})
        refiner.transform([chunk], trace=trace)

        assert len(trace.stages) == 1
        assert "elapsed_ms" in trace.stages[0]
        assert trace.stages[0]["elapsed_ms"] >= 0

    def test_trace_records_chunk_count(self):
        """追踪应该记录 Chunk 数量。"""
        refiner = FakeChunkRefiner()
        trace = TraceContext(trace_type="ingestion")

        chunks = [
            Chunk(id="test_1", text="Content 1.", metadata={}),
            Chunk(id="test_2", text="Content 2.", metadata={}),
        ]
        refiner.transform(chunks, trace=trace)

        assert trace.stages[0]["details"]["chunk_count"] == 2

    def test_trace_records_method(self):
        """追踪应该记录处理方法。"""
        refiner = FakeChunkRefiner()
        trace = TraceContext(trace_type="ingestion")

        chunk = Chunk(id="test_method", text="Content.", metadata={})
        refiner.transform([chunk], trace=trace)

        assert trace.stages[0]["method"] == "rule"
