"""
Reranker 工厂。

根据配置创建对应的 Reranker 实例。
"""

from typing import TYPE_CHECKING

from src.core.settings import RerankSettings, Settings
from src.libs.reranker.base_reranker import BaseReranker, RerankerError
from src.libs.reranker.none_reranker import NoneReranker

if TYPE_CHECKING:
    pass


class RerankerFactory:
    """
    Reranker 工厂类。

    根据配置动态创建 Reranker 实例，支持多种后端。
    """

    @classmethod
    def create(cls, settings: Settings) -> BaseReranker:
        """
        根据配置创建 Reranker 实例。

        Args:
            settings: 应用配置。

        Returns:
            BaseReranker: Reranker 实例。

        Raises:
            RerankerError: 不支持的后端或配置错误。
        """
        rerank_settings = settings.rerank

        # 如果未启用，返回 None Reranker
        if not rerank_settings.enabled:
            return NoneReranker()

        provider = rerank_settings.provider.lower() if rerank_settings.provider else "none"

        if provider == "none" or provider == "":
            return NoneReranker()
        elif provider == "cross_encoder":
            from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker

            return CrossEncoderReranker(rerank_settings)
        elif provider == "llm":
            from src.libs.reranker.llm_reranker import LLMReranker

            return LLMReranker(rerank_settings)
        else:
            raise RerankerError(
                f"Unsupported Reranker provider: {provider}",
                backend=provider,
            )

    @classmethod
    def create_from_settings(cls, rerank_settings: RerankSettings) -> BaseReranker:
        """
        从 RerankSettings 创建 Reranker 实例。

        Args:
            rerank_settings: Rerank 配置。

        Returns:
            BaseReranker: Reranker 实例。
        """
        if not rerank_settings.enabled:
            return NoneReranker()

        provider = rerank_settings.provider.lower() if rerank_settings.provider else "none"

        if provider == "none" or provider == "":
            return NoneReranker()
        elif provider == "cross_encoder":
            from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker

            return CrossEncoderReranker(rerank_settings)
        elif provider == "llm":
            from src.libs.reranker.llm_reranker import LLMReranker

            return LLMReranker(rerank_settings)
        else:
            raise RerankerError(
                f"Unsupported Reranker provider: {provider}",
                backend=provider,
            )

    @classmethod
    def get_supported_providers(cls) -> list:
        """获取支持的 provider 列表。"""
        return ["none", "cross_encoder", "llm"]

    @classmethod
    def get_fallback(cls) -> NoneReranker:
        """获取默认回退 Reranker。"""
        return NoneReranker()
