"""
ImageCaptioner 单元测试。

测试图片描述生成和降级功能。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.core.types import Chunk, ImageRef
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.image_captioner import (
    ImageCaptioner,
    FakeImageCaptioner,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_settings_disabled():
    """创建禁用 Vision LLM 的 Mock Settings。"""
    settings = Mock()
    settings.vision_llm = Mock()
    settings.vision_llm.enabled = False
    return settings


@pytest.fixture
def mock_settings_enabled():
    """创建启用 Vision LLM 的 Mock Settings。"""
    settings = Mock()
    settings.vision_llm = Mock()
    settings.vision_llm.enabled = True
    settings.vision_llm.provider = "openai"
    settings.vision_llm.model = "gpt-4o"
    return settings


@pytest.fixture
def mock_vision_llm():
    """创建 Mock Vision LLM。"""
    vision_llm = Mock()
    vision_llm.chat_with_image = Mock(return_value=Mock(
        content="This is a diagram showing the system architecture with three layers."
    ))
    return vision_llm


@pytest.fixture
def sample_image_ref():
    """创建测试用图片引用。"""
    return ImageRef(
        id="test_doc_001_0",
        path="/tmp/test_image.png",
        page=1,
        text_offset=100,
        text_length=20,
    )


@pytest.fixture
def chunk_with_images(sample_image_ref):
    """创建包含图片的 Chunk。"""
    return Chunk(
        id="chunk_001",
        text="# Introduction\n\nThis is shown in [IMAGE: test_doc_001_0] the diagram above.",
        metadata={
            "source_path": "test.pdf",
            "chunk_index": 0,
            "images": [sample_image_ref.to_dict()],
            "image_refs": ["test_doc_001_0"],
        },
    )


@pytest.fixture
def chunk_without_images():
    """创建不包含图片的 Chunk。"""
    return Chunk(
        id="chunk_002",
        text="# Text Only\n\nThis chunk has no images.",
        metadata={
            "source_path": "test.pdf",
            "chunk_index": 1,
        },
    )


@pytest.fixture
def trace_context():
    """创建 TraceContext。"""
    return TraceContext(trace_type="ingestion")


# ============================================================================
# 禁用模式测试
# ============================================================================


class TestDisabledMode:
    """禁用模式测试。"""

    def test_disabled_mode_skips_captioning(
        self, mock_settings_disabled, chunk_with_images
    ):
        """测试禁用模式下跳过描述生成。"""
        captioner = ImageCaptioner(mock_settings_disabled)
        result = captioner.transform([chunk_with_images])

        chunk = result[0]
        assert "image_captions" not in chunk.metadata
        assert chunk.metadata.get("has_unprocessed_images") is True
        assert chunk.metadata.get("captioned_by") == "disabled"

    def test_disabled_mode_preserves_image_refs(
        self, mock_settings_disabled, chunk_with_images
    ):
        """测试禁用模式保留图片引用。"""
        captioner = ImageCaptioner(mock_settings_disabled)
        result = captioner.transform([chunk_with_images])

        chunk = result[0]
        assert "images" in chunk.metadata
        assert "image_refs" in chunk.metadata

    def test_disabled_mode_no_images(
        self, mock_settings_disabled, chunk_without_images
    ):
        """测试禁用模式处理无图片 Chunk。"""
        captioner = ImageCaptioner(mock_settings_disabled)
        result = captioner.transform([chunk_without_images])

        chunk = result[0]
        assert "has_unprocessed_images" not in chunk.metadata
        assert "captioned_by" not in chunk.metadata


# ============================================================================
# 启用模式测试
# ============================================================================


class TestEnabledMode:
    """启用模式测试。"""

    def test_enabled_mode_with_vision_llm(
        self, mock_settings_enabled, mock_vision_llm, chunk_with_images, tmp_path
    ):
        """测试启用模式下生成描述。"""
        # 创建临时图片文件
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(b"fake image data")

        # 更新 image_ref 路径
        chunk = chunk_with_images
        chunk.metadata["images"][0]["path"] = str(image_path)

        captioner = ImageCaptioner(mock_settings_enabled, vision_llm=mock_vision_llm)
        captioner._image_base_path = tmp_path

        result = captioner.transform([chunk])

        processed_chunk = result[0]
        assert "image_captions" in processed_chunk.metadata
        assert processed_chunk.metadata["captioned_by"] == "vision_llm"

    def test_caption_appended_to_text(
        self, mock_settings_enabled, mock_vision_llm, chunk_with_images, tmp_path
    ):
        """测试描述追加到文本中。"""
        # 创建临时图片文件
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(b"fake image data")

        chunk = chunk_with_images
        chunk.metadata["images"][0]["path"] = str(image_path)

        captioner = ImageCaptioner(mock_settings_enabled, vision_llm=mock_vision_llm)
        captioner._image_base_path = tmp_path

        result = captioner.transform([chunk])

        processed_chunk = result[0]
        assert "图片描述:" in processed_chunk.text

    def test_vision_llm_failure_fallback(
        self, mock_settings_enabled, chunk_with_images, tmp_path
    ):
        """测试 Vision LLM 调用失败时的降级。"""
        # 创建 Mock Vision LLM，模拟失败
        vision_llm = Mock()
        vision_llm.chat_with_image = Mock(side_effect=Exception("API Error"))

        # 创建临时图片文件
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(b"fake image data")

        chunk = chunk_with_images
        chunk.metadata["images"][0]["path"] = str(image_path)

        captioner = ImageCaptioner(mock_settings_enabled, vision_llm=vision_llm)
        captioner._image_base_path = tmp_path

        result = captioner.transform([chunk])

        processed_chunk = result[0]
        assert processed_chunk.metadata.get("has_unprocessed_images") is True
        assert processed_chunk.metadata.get("captioned_by") == "failed"

    def test_image_not_found_fallback(
        self, mock_settings_enabled, mock_vision_llm, chunk_with_images
    ):
        """测试图片文件不存在时的降级。"""
        # 图片路径指向不存在的文件
        chunk = chunk_with_images
        chunk.metadata["images"][0]["path"] = "/nonexistent/image.png"

        captioner = ImageCaptioner(mock_settings_enabled, vision_llm=mock_vision_llm)

        result = captioner.transform([chunk])

        processed_chunk = result[0]
        assert processed_chunk.metadata.get("has_unprocessed_images") is True

    def test_short_caption_rejected(
        self, mock_settings_enabled, chunk_with_images, tmp_path
    ):
        """测试过短描述被拒绝。"""
        # 创建返回过短描述的 Mock
        vision_llm = Mock()
        vision_llm.chat_with_image = Mock(return_value=Mock(content="short"))

        # 创建临时图片文件
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(b"fake image data")

        chunk = chunk_with_images
        chunk.metadata["images"][0]["path"] = str(image_path)

        captioner = ImageCaptioner(mock_settings_enabled, vision_llm=vision_llm)
        captioner._image_base_path = tmp_path

        result = captioner.transform([chunk])

        processed_chunk = result[0]
        # 过短描述应该被拒绝
        assert "image_captions" not in processed_chunk.metadata or \
               processed_chunk.metadata.get("has_unprocessed_images") is True


# ============================================================================
# 配置测试
# ============================================================================


class TestConfiguration:
    """配置测试。"""

    def test_vision_llm_creation_failure_fallback(self, mock_settings_enabled):
        """测试 Vision LLM 创建失败时降级。"""
        with patch("src.ingestion.transform.image_captioner.LLMFactory.create_vision_llm") as mock_create:
            mock_create.side_effect = Exception("Creation failed")

            captioner = ImageCaptioner(mock_settings_enabled)

            assert captioner.enabled is False

    def test_custom_prompt_path(self, mock_settings_disabled, tmp_path):
        """测试自定义 Prompt 路径。"""
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text("Custom image analysis prompt.")

        captioner = ImageCaptioner(mock_settings_disabled, prompt_path=str(prompt_file))

        assert "Custom image analysis prompt" in captioner._prompt_template


# ============================================================================
# Trace 测试
# ============================================================================


class TestTracing:
    """追踪测试。"""

    def test_trace_recorded(
        self, mock_settings_disabled, chunk_with_images, trace_context
    ):
        """测试追踪记录。"""
        captioner = ImageCaptioner(mock_settings_disabled)
        result = captioner.transform([chunk_with_images], trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "image_captioning" for s in stages)

    def test_trace_includes_stats(
        self, mock_settings_enabled, mock_vision_llm, chunk_with_images, trace_context, tmp_path
    ):
        """测试追踪包含统计信息。"""
        # 创建临时图片文件
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(b"fake image data")

        chunk = chunk_with_images
        chunk.metadata["images"][0]["path"] = str(image_path)

        captioner = ImageCaptioner(mock_settings_enabled, vision_llm=mock_vision_llm)
        captioner._image_base_path = tmp_path

        result = captioner.transform([chunk], trace_context)

        stages = trace_context.to_dict().get("stages", [])
        stage = next((s for s in stages if s["stage"] == "image_captioning"), None)

        assert stage is not None
        assert "total_images" in stage.get("details", {})
        assert "captioned_images" in stage.get("details", {})


# ============================================================================
# FakeImageCaptioner 测试
# ============================================================================


class TestFakeImageCaptioner:
    """Fake ImageCaptioner 测试。"""

    def test_fake_disabled_mode(self, chunk_with_images):
        """测试 Fake 禁用模式。"""
        captioner = FakeImageCaptioner(enabled=False)
        result = captioner.transform([chunk_with_images])

        chunk = result[0]
        assert chunk.metadata.get("has_unprocessed_images") is True
        assert chunk.metadata.get("captioned_by") == "disabled"

    def test_fake_enabled_mode(self, chunk_with_images):
        """测试 Fake 启用模式。"""
        captioner = FakeImageCaptioner(enabled=True)
        result = captioner.transform([chunk_with_images])

        chunk = result[0]
        assert "image_captions" in chunk.metadata
        assert chunk.metadata.get("captioned_by") == "fake_vision_llm"

    def test_fake_no_images(self, chunk_without_images):
        """测试 Fake 处理无图片 Chunk。"""
        captioner = FakeImageCaptioner(enabled=True)
        result = captioner.transform([chunk_without_images])

        chunk = result[0]
        assert "image_captions" not in chunk.metadata

    def test_fake_trace_recorded(self, chunk_with_images, trace_context):
        """测试 Fake 追踪记录。"""
        captioner = FakeImageCaptioner(enabled=True)
        result = captioner.transform([chunk_with_images], trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "image_captioning" for s in stages)


# ============================================================================
# 边界条件测试
# ============================================================================


class TestEdgeCases:
    """边界条件测试。"""

    def test_empty_chunks_list(self, mock_settings_disabled):
        """测试空 Chunk 列表。"""
        captioner = ImageCaptioner(mock_settings_disabled)
        result = captioner.transform([])

        assert result == []

    def test_chunk_with_empty_image_refs(self, mock_settings_disabled):
        """测试空图片引用列表。"""
        chunk = Chunk(
            id="test",
            text="Text content",
            metadata={"images": [], "image_refs": []},
        )

        captioner = ImageCaptioner(mock_settings_disabled)
        result = captioner.transform([chunk])

        assert result[0].metadata.get("has_unprocessed_images") is not True

    def test_multiple_images_in_chunk(
        self, mock_settings_enabled, mock_vision_llm, tmp_path
    ):
        """测试 Chunk 包含多张图片。"""
        # 创建两个图片文件
        image1 = tmp_path / "image1.png"
        image1.write_bytes(b"image1")
        image2 = tmp_path / "image2.png"
        image2.write_bytes(b"image2")

        chunk = Chunk(
            id="multi",
            text="Multiple images: [IMAGE: img1] and [IMAGE: img2]",
            metadata={
                "images": [
                    ImageRef(id="img1", path=str(image1)).to_dict(),
                    ImageRef(id="img2", path=str(image2)).to_dict(),
                ],
                "image_refs": ["img1", "img2"],
            },
        )

        captioner = ImageCaptioner(mock_settings_enabled, vision_llm=mock_vision_llm)
        captioner._image_base_path = tmp_path

        result = captioner.transform([chunk])

        processed = result[0]
        assert len(processed.metadata.get("image_captions", {})) == 2

    def test_image_ref_without_path(self, mock_settings_enabled, mock_vision_llm):
        """测试图片引用无路径时的处理。"""
        chunk = Chunk(
            id="no_path",
            text="Image without path: [IMAGE: missing]",
            metadata={
                "images": [
                    ImageRef(id="missing", path="").to_dict(),
                ],
                "image_refs": ["missing"],
            },
        )

        captioner = ImageCaptioner(mock_settings_enabled, vision_llm=mock_vision_llm)

        result = captioner.transform([chunk])

        # 无路径应该降级
        processed = result[0]
        assert processed.metadata.get("has_unprocessed_images") is True

    def test_preserves_existing_metadata(
        self, mock_settings_disabled, chunk_with_images
    ):
        """测试保留现有元数据。"""
        chunk = chunk_with_images
        chunk.metadata["custom_field"] = "custom_value"
        chunk.metadata["title"] = "Test Title"

        captioner = ImageCaptioner(mock_settings_disabled)
        result = captioner.transform([chunk])

        processed = result[0]
        assert processed.metadata.get("custom_field") == "custom_value"
        assert processed.metadata.get("title") == "Test Title"
