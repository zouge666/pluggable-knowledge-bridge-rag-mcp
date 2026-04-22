"""
LLM 工厂。

根据配置创建对应的 LLM 实例。
"""

from typing import TYPE_CHECKING, Optional

from src.core.settings import LLMSettings, Settings
from src.libs.llm.base_llm import BaseLLM, LLMError

if TYPE_CHECKING:
    pass


class LLMFactory:
    """
    LLM 工厂类。

    根据配置动态创建 LLM 实例，支持多种 provider。
    """

    # Provider 注册表（延迟加载）
    _registry: dict = {}

    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        """
        根据配置创建 LLM 实例。

        Args:
            settings: 应用配置。

        Returns:
            BaseLLM: LLM 实例。

        Raises:
            LLMError: 不支持的 provider 或配置错误。
        """
        llm_settings = settings.llm
        provider = llm_settings.provider.lower()

        # 延迟加载具体实现
        if provider == "azure":
            from src.libs.llm.azure_llm import AzureLLM

            return AzureLLM(llm_settings)
        elif provider == "openai":
            from src.libs.llm.openai_llm import OpenAILLM

            return OpenAILLM(llm_settings)
        elif provider == "ollama":
            from src.libs.llm.ollama_llm import OllamaLLM

            return OllamaLLM(llm_settings)
        elif provider == "deepseek":
            from src.libs.llm.deepseek_llm import DeepSeekLLM

            return DeepSeekLLM(llm_settings)
        else:
            raise LLMError(
                f"Unsupported LLM provider: {provider}",
                provider=provider,
            )

    @classmethod
    def create_from_settings(cls, llm_settings: LLMSettings) -> BaseLLM:
        """
        从 LLMSettings 创建 LLM 实例。

        Args:
            llm_settings: LLM 配置。

        Returns:
            BaseLLM: LLM 实例。
        """
        provider = llm_settings.provider.lower()

        if provider == "azure":
            from src.libs.llm.azure_llm import AzureLLM

            return AzureLLM(llm_settings)
        elif provider == "openai":
            from src.libs.llm.openai_llm import OpenAILLM

            return OpenAILLM(llm_settings)
        elif provider == "ollama":
            from src.libs.llm.ollama_llm import OllamaLLM

            return OllamaLLM(llm_settings)
        elif provider == "deepseek":
            from src.libs.llm.deepseek_llm import DeepSeekLLM

            return DeepSeekLLM(llm_settings)
        else:
            raise LLMError(
                f"Unsupported LLM provider: {provider}",
                provider=provider,
            )

    @classmethod
    def register(cls, provider: str, llm_class: type) -> None:
        """
        注册自定义 LLM provider。

        Args:
            provider: Provider 名称。
            llm_class: LLM 实现类。
        """
        cls._registry[provider.lower()] = llm_class

    @classmethod
    def get_supported_providers(cls) -> list:
        """获取支持的 provider 列表。"""
        return ["azure", "openai", "ollama", "deepseek"]
