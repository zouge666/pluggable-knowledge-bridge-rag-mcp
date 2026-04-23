"""
Vision LLM Factory 单元测试。

使用 Fake Vision LLM 验证工厂路由逻辑。
"""

import pytest

from src.core.settings import Settings, VisionLLMSettings
from src.libs.llm.base_llm import ChatResponse, LLMError
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageContent
from src.libs.llm.llm_factory import LLMFactory


class FakeVisionLLM(BaseVisionLLM):
    """Fake Vision LLM 实现，用于测试。"""

    def __init__(self, settings, response: str = "fake vision response"):
        if hasattr(settings, "vision_llm"):
            settings = settings.vision_llm
        self.model = settings.model
        self._response = response
        self.max_image_size = getattr(settings, "max_image_size", 2048)

    def chat_with_image(
        self,
        text,
        image,
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        trace=None,
    ):
        return ChatResponse(
            content=self._response,
            model=self.model,
        )

    def chat_with_images(
        self,
        text,
        images,
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        trace=None,
    ):
        return ChatResponse(
            content=self._response,
            model=self.model,
        )

    def get_model_name(self) -> str:
        return self.model


class TestVisionLLMFactory:
    """Vision LLM Factory 测试。"""

    def test_get_supported_vision_providers(self):
        """应该返回支持的 Vision LLM provider 列表。"""
        providers = LLMFactory.get_supported_vision_providers()

        assert "azure" in providers
        assert "openai" in providers

    def test_create_vision_llm_disabled_raises_error(self):
        """Vision LLM 未启用时应该抛出错误。"""
        settings = Settings(
            vision_llm=VisionLLMSettings(
                enabled=False,
                provider="azure",
            )
        )

        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create_vision_llm(settings)

        assert "not enabled" in str(exc_info.value)

    def test_create_vision_llm_unsupported_provider(self):
        """不支持的 provider 应该抛出错误。"""
        settings = Settings(
            vision_llm=VisionLLMSettings(
                enabled=True,
                provider="unsupported",
            )
        )

        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create_vision_llm(settings)

        assert "Unsupported Vision LLM provider" in str(exc_info.value)

    def test_create_vision_llm_from_settings_disabled(self):
        """从 VisionLLMSettings 创建时未启用应该抛出错误。"""
        vision_settings = VisionLLMSettings(
            enabled=False,
            provider="azure",
        )

        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create_vision_llm_from_settings(vision_settings)

        assert "not enabled" in str(exc_info.value)


class TestBaseVisionLLM:
    """BaseVisionLLM 测试。"""

    def test_image_content_to_base64_from_bytes(self):
        """应该能从 bytes 转换为 base64。"""
        image_data = b"fake image data"
        image = ImageContent(source=image_data, mime_type="image/png")

        base64_str = image.to_base64()

        assert base64_str is not None
        assert len(base64_str) > 0

    def test_image_content_to_base64_from_base64_string(self):
        """应该能处理已有的 base64 字符串。"""
        base64_str = "YWJjMTIz"  # "abc123" 的 base64
        image = ImageContent(source=base64_str, mime_type="image/png")

        result = image.to_base64()

        assert result == base64_str

    def test_image_content_to_dict(self):
        """应该能转换为字典。"""
        image = ImageContent(
            source=b"test",
            mime_type="image/png",
            compressed=True,
            original_size=(1000, 800),
        )

        result = image.to_dict()

        assert result["mime_type"] == "image/png"
        assert result["compressed"] is True
        assert result["original_size"] == (1000, 800)

    def test_normalize_image_from_string(self):
        """应该能从字符串路径标准化图片。"""
        fake_llm = FakeVisionLLM(
            VisionLLMSettings(enabled=True, provider="fake", model="fake-model")
        )

        image = fake_llm._normalize_image("/path/to/image.png")

        assert isinstance(image, ImageContent)
        assert image.mime_type == "image/png"

    def test_normalize_image_from_bytes(self):
        """应该能从 bytes 标准化图片。"""
        fake_llm = FakeVisionLLM(
            VisionLLMSettings(enabled=True, provider="fake", model="fake-model")
        )

        image = fake_llm._normalize_image(b"image data")

        assert isinstance(image, ImageContent)
        assert image.mime_type is None  # bytes 无法推断 MIME 类型

    def test_normalize_image_from_image_content(self):
        """应该能保持 ImageContent 不变。"""
        fake_llm = FakeVisionLLM(
            VisionLLMSettings(enabled=True, provider="fake", model="fake-model")
        )

        original = ImageContent(source=b"test", mime_type="image/jpeg")
        image = fake_llm._normalize_image(original)

        assert image is original


class TestFakeVisionLLM:
    """FakeVisionLLM 测试。"""

    def test_chat_with_image(self):
        """应该能发送带图片的聊天请求。"""
        llm = FakeVisionLLM(
            VisionLLMSettings(enabled=True, provider="fake", model="gpt-4o"),
            response="This is an image of a cat.",
        )

        response = llm.chat_with_image(
            text="What is in this image?",
            image=b"fake image data",
        )

        assert response.content == "This is an image of a cat."
        assert response.model == "gpt-4o"

    def test_chat_with_images(self):
        """应该能发送带多张图片的聊天请求。"""
        llm = FakeVisionLLM(
            VisionLLMSettings(enabled=True, provider="fake", model="gpt-4o"),
            response="Multiple images analyzed.",
        )

        response = llm.chat_with_images(
            text="Compare these images.",
            images=[b"image1", b"image2"],
        )

        assert response.content == "Multiple images analyzed."

    def test_get_model_name(self):
        """应该返回正确的模型名称。"""
        llm = FakeVisionLLM(
            VisionLLMSettings(enabled=True, provider="fake", model="gpt-4-vision"),
        )

        assert llm.get_model_name() == "gpt-4-vision"


class TestVisionLLMSettings:
    """VisionLLMSettings 测试。"""

    def test_default_settings(self):
        """默认设置应该正确。"""
        settings = VisionLLMSettings()

        assert settings.enabled is False
        assert settings.provider == "openai"
        assert settings.model == "gpt-4o"
        assert settings.max_image_size == 2048

    def test_custom_settings(self):
        """自定义设置应该正确。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4-vision",
            max_image_size=1024,
        )

        assert settings.enabled is True
        assert settings.provider == "azure"
        assert settings.model == "gpt-4-vision"
        assert settings.max_image_size == 1024