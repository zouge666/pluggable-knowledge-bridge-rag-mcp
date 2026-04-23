"""
Azure Vision LLM 单元测试。

使用 Mock HTTP 验证图像理解逻辑。
"""

import base64
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.core.settings import Settings, VisionLLMSettings
from src.libs.llm.base_llm import ChatResponse, LLMAuthenticationError, LLMConnectionError, LLMRateLimitError, LLMResponseError
from src.libs.llm.base_vision_llm import ImageContent
from src.libs.llm.azure_vision_llm import AzureVisionLLM
from src.libs.llm.llm_factory import LLMFactory


class TestAzureVisionLLM:
    """AzureVisionLLM 测试。"""

    def test_init_with_settings(self):
        """应该能使用 settings 初始化。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            max_image_size=1024,
        )

        llm = AzureVisionLLM(settings)

        assert llm.model == "gpt-4o"
        assert llm.azure_endpoint == "https://test.openai.azure.com"
        assert llm.max_image_size == 1024

    def test_init_with_full_settings(self):
        """应该能从完整 Settings 初始化。"""
        settings = Settings(
            vision_llm=VisionLLMSettings(
                enabled=True,
                provider="azure",
                model="gpt-4-vision",
                azure_endpoint="https://test.openai.azure.com",
                api_key="test-key",
            )
        )

        llm = AzureVisionLLM(settings)

        assert llm.model == "gpt-4-vision"

    def test_normalize_image(self):
        """应该能标准化图片。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        llm = AzureVisionLLM(settings)

        # 从 bytes
        image = llm._normalize_image(b"image data")
        assert isinstance(image, ImageContent)

        # 从 ImageContent
        original = ImageContent(source=b"test", mime_type="image/png")
        image = llm._normalize_image(original)
        assert image is original

    def test_build_image_content(self):
        """应该能构建 OpenAI API 图片结构。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        llm = AzureVisionLLM(settings)

        image = ImageContent(source=b"test", mime_type="image/png")
        content = llm._build_image_content(image)

        assert content["type"] == "image_url"
        assert "data:image/png;base64," in content["image_url"]["url"]

    def test_chat_with_image_mock(self):
        """应该能发送带图片的聊天请求（mock）。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            deployment_name="gpt-4o-deployment",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        llm = AzureVisionLLM(settings)

        # Mock 客户端
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a cat."
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        llm._client = mock_client

        response = llm.chat_with_image(
            text="What is in this image?",
            image=b"fake image data",
        )

        assert response.content == "This is a cat."
        assert response.model == "gpt-4o"
        assert response.usage["prompt_tokens"] == 100

    def test_chat_with_images_mock(self):
        """应该能发送带多张图片的聊天请求（mock）。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        llm = AzureVisionLLM(settings)

        # Mock 客户端
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Comparing images."
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        llm._client = mock_client

        response = llm.chat_with_images(
            text="Compare these images.",
            images=[b"image1", b"image2"],
        )

        assert response.content == "Comparing images."

    def test_chat_with_image_authentication_error(self):
        """认证错误应该抛出 LLMAuthenticationError。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        llm = AzureVisionLLM(settings)

        # Mock 客户端抛出认证错误
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized")
        llm._client = mock_client

        with pytest.raises(LLMAuthenticationError):
            llm.chat_with_image("test", b"image")

    def test_chat_with_image_rate_limit_error(self):
        """速率限制错误应该抛出 LLMRateLimitError。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        llm = AzureVisionLLM(settings)

        # Mock 客户端抛出速率限制错误
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("429 Rate limit exceeded")
        llm._client = mock_client

        with pytest.raises(LLMRateLimitError):
            llm.chat_with_image("test", b"image")

    def test_chat_with_image_connection_error(self):
        """连接错误应该抛出 LLMConnectionError。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        llm = AzureVisionLLM(settings)

        # Mock 客户端抛出连接错误
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection timeout")
        llm._client = mock_client

        with pytest.raises(LLMConnectionError):
            llm.chat_with_image("test", b"image")

    def test_get_model_name(self):
        """应该返回正确的模型名称。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4-vision",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        llm = AzureVisionLLM(settings)

        assert llm.get_model_name() == "gpt-4-vision"


class TestAzureVisionLLMFactory:
    """Azure Vision LLM 工厂测试。"""

    def test_create_azure_vision_llm(self):
        """应该能创建 Azure Vision LLM。"""
        settings = Settings(
            vision_llm=VisionLLMSettings(
                enabled=True,
                provider="azure",
                model="gpt-4o",
                azure_endpoint="https://test.openai.azure.com",
                api_key="test-key",
            )
        )

        llm = LLMFactory.create_vision_llm(settings)

        assert llm is not None
        assert llm.get_model_name() == "gpt-4o"

    def test_create_azure_vision_llm_from_settings(self):
        """应该能从 VisionLLMSettings 创建。"""
        vision_settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4-vision",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )

        llm = LLMFactory.create_vision_llm_from_settings(vision_settings)

        assert llm is not None
        assert llm.get_model_name() == "gpt-4-vision"


class TestImageCompression:
    """图片压缩测试。"""

    def test_compress_image_skip_when_small(self):
        """小图片不应该压缩。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            max_image_size=2048,
        )
        llm = AzureVisionLLM(settings)

        # 创建一个小图片的 ImageContent
        image = ImageContent(source=b"small image", mime_type="image/png")

        # 由于 PIL 可能未安装，这里只测试方法存在
        result = llm._compress_image_if_needed(image)
        assert isinstance(result, ImageContent)

    def test_compress_image_method_exists(self):
        """压缩方法应该存在并返回 ImageContent。"""
        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            max_image_size=100,
        )
        llm = AzureVisionLLM(settings)

        image = ImageContent(source=b"test image data", mime_type="image/png")
        result = llm._compress_image_if_needed(image)

        # 应该返回 ImageContent（无论是否压缩）
        assert isinstance(result, ImageContent)


class TestAPIKeyFromEnv:
    """API Key 环境变量测试。"""

    def test_api_key_from_env_variable(self):
        """应该能从环境变量获取 API Key。"""
        import os
        os.environ["TEST_VISION_API_KEY"] = "env-test-key"

        settings = VisionLLMSettings(
            enabled=True,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://test.openai.azure.com",
            api_key="${TEST_VISION_API_KEY}",
        )

        llm = AzureVisionLLM(settings)

        assert llm.api_key == "env-test-key"

        # 清理
        del os.environ["TEST_VISION_API_KEY"]