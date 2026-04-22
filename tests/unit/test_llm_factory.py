"""
LLM Factory 单元测试。

使用 Fake provider 验证工厂路由逻辑。
"""

import pytest

from src.core.settings import LLMSettings, Settings
from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError
from src.libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    """Fake LLM 实现，用于测试。"""

    def __init__(self, settings, response: str = "fake response"):
        if hasattr(settings, "llm"):
            settings = settings.llm
        self.model = settings.model
        self._response = response

    def chat(self, messages, temperature=None, max_tokens=None, trace=None):
        return ChatResponse(
            content=self._response,
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    def get_model_name(self) -> str:
        return self.model


class TestLLMFactory:
    """LLMFactory 测试。"""

    def test_get_supported_providers(self):
        """应该返回支持的 provider 列表。"""
        providers = LLMFactory.get_supported_providers()

        assert "openai" in providers
        assert "azure" in providers
        assert "ollama" in providers
        assert "deepseek" in providers

    def test_unsupported_provider_raises_error(self):
        """不支持的 provider 应该抛出错误。"""
        settings = Settings(
            llm=LLMSettings(
                provider="unsupported_provider",
                model="test-model",
                api_key="test-key",
            )
        )

        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create(settings)

        assert "Unsupported LLM provider" in str(exc_info.value)
        assert exc_info.value.provider == "unsupported_provider"

    def test_register_custom_provider(self):
        """应该支持注册自定义 provider。"""
        LLMFactory.register("fake", FakeLLM)

        assert "fake" in LLMFactory._registry

    def test_create_openai_llm(self):
        """应该能创建 OpenAI LLM。"""
        settings = Settings(
            llm=LLMSettings(
                provider="openai",
                model="gpt-4o",
                api_key="test-key",
            )
        )

        llm = LLMFactory.create(settings)

        assert llm is not None
        assert llm.get_model_name() == "gpt-4o"

    def test_create_openai_llm_case_insensitive(self):
        """Provider 名称应该不区分大小写。"""
        settings = Settings(
            llm=LLMSettings(
                provider="OpenAI",  # 大小写混合
                model="gpt-4o",
                api_key="test-key",
            )
        )

        llm = LLMFactory.create(settings)

        assert llm is not None

    def test_create_azure_llm(self):
        """应该能创建 Azure LLM。"""
        settings = Settings(
            llm=LLMSettings(
                provider="azure",
                model="gpt-4o",
                deployment_name="my-deployment",
                azure_endpoint="https://my-resource.openai.azure.com",
                api_key="test-key",
            )
        )

        llm = LLMFactory.create(settings)

        assert llm is not None
        assert llm.get_model_name() == "gpt-4o"

    def test_create_ollama_llm(self):
        """应该能创建 Ollama LLM。"""
        settings = Settings(
            llm=LLMSettings(
                provider="ollama",
                model="llama3",
            )
        )

        llm = LLMFactory.create(settings)

        assert llm is not None
        assert llm.get_model_name() == "llama3"

    def test_create_deepseek_llm(self):
        """应该能创建 DeepSeek LLM。"""
        settings = Settings(
            llm=LLMSettings(
                provider="deepseek",
                model="deepseek-chat",
                api_key="test-key",
            )
        )

        llm = LLMFactory.create(settings)

        assert llm is not None
        assert llm.get_model_name() == "deepseek-chat"

    def test_create_from_llm_settings(self):
        """应该能从 LLMSettings 创建 LLM。"""
        llm_settings = LLMSettings(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
        )

        llm = LLMFactory.create_from_settings(llm_settings)

        assert llm is not None
        assert llm.get_model_name() == "gpt-4o"


class TestBaseLLM:
    """BaseLLM 测试。"""

    def test_chat_message_to_dict(self):
        """ChatMessage 应该能转换为字典。"""
        msg = ChatMessage(role="user", content="Hello")

        result = msg.to_dict()

        assert result == {"role": "user", "content": "Hello"}

    def test_chat_response_to_dict(self):
        """ChatResponse 应该能转换为字典。"""
        response = ChatResponse(
            content="Hello",
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
        )

        result = response.to_dict()

        assert result["content"] == "Hello"
        assert result["model"] == "gpt-4o"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["finish_reason"] == "stop"

    def test_chat_response_to_dict_minimal(self):
        """ChatResponse 最小字段应该能转换。"""
        response = ChatResponse(content="Hello", model="gpt-4o")

        result = response.to_dict()

        assert result == {"content": "Hello", "model": "gpt-4o"}
        assert "usage" not in result
        assert "finish_reason" not in result

    def test_chat_with_str(self):
        """chat_with_str 便捷方法应该工作。"""
        fake_llm = FakeLLM(
            LLMSettings(provider="fake", model="fake-model"),
            response="test response",
        )

        result = fake_llm.chat_with_str("Hello")

        assert result == "test response"

    def test_chat_with_str_with_system_prompt(self):
        """chat_with_str 应该支持系统提示。"""
        fake_llm = FakeLLM(
            LLMSettings(provider="fake", model="fake-model"),
            response="test response",
        )

        result = fake_llm.chat_with_str("Hello", system_prompt="You are helpful.")

        assert result == "test response"


class TestLLMError:
    """LLMError 测试。"""

    def test_llm_error_basic(self):
        """基本错误信息。"""
        error = LLMError("Something went wrong")

        assert "Something went wrong" in str(error)

    def test_llm_error_with_provider(self):
        """带 provider 的错误信息。"""
        error = LLMError("Something went wrong", provider="openai")

        assert "[openai]" in str(error)
        assert "Something went wrong" in str(error)

    def test_llm_error_with_all_fields(self):
        """带所有字段的错误信息。"""
        error = LLMError(
            "Something went wrong",
            provider="azure",
            model="gpt-4o",
        )

        error_str = str(error)
        assert "[azure]" in error_str
        assert "model=gpt-4o" in error_str
        assert "Something went wrong" in error_str

    def test_llm_error_subclasses(self):
        """错误子类应该继承 LLMError。"""
        from src.libs.llm.base_llm import (
            LLMAuthenticationError,
            LLMConnectionError,
            LLMRateLimitError,
            LLMResponseError,
        )

        auth_error = LLMAuthenticationError("Auth failed", provider="openai")
        conn_error = LLMConnectionError("Connection failed", provider="azure")
        rate_error = LLMRateLimitError("Rate limited", provider="deepseek")
        resp_error = LLMResponseError("Bad response", provider="ollama")

        assert isinstance(auth_error, LLMError)
        assert isinstance(conn_error, LLMError)
        assert isinstance(rate_error, LLMError)
        assert isinstance(resp_error, LLMError)
