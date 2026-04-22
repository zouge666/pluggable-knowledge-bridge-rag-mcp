"""
OpenAI Embedding 实现。
"""

import os
from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResult,
    EmbeddingConnectionError,
    EmbeddingAuthenticationError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embedding 实现。

    支持 OpenAI 官方 API 的 text-embedding-3-small/large 等模型。
    """

    def __init__(self, settings) -> None:
        """
        初始化 OpenAI Embedding。

        Args:
            settings: Embedding 配置（EmbeddingSettings 或 Settings）。
        """
        if hasattr(settings, "embedding"):
            settings = settings.embedding

        self.model = settings.model or "text-embedding-3-small"
        self.dimensions = settings.dimensions or 1536

        # 获取 API Key
        self.api_key = settings.api_key
        if self.api_key and self.api_key.startswith("${") and self.api_key.endswith("}"):
            env_var = self.api_key[2:-1]
            self.api_key = os.environ.get(env_var, "")

        # 自定义 base_url
        self.base_url = getattr(settings, "base_url", None)

        # 延迟初始化客户端
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """获取或创建 OpenAI 客户端。"""
        if self._client is None:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required. Install it with: pip install openai"
                )
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def embed(
        self,
        texts: List[str],
        trace: Optional[TraceContext] = None,
    ) -> EmbeddingResult:
        """
        批量生成文本的向量表示。

        Args:
            texts: 文本列表。
            trace: 追踪上下文。

        Returns:
            EmbeddingResult: 包含向量和元数据的结果。
        """
        import time

        start_time = time.time()
        client = self._get_client()

        # 过滤空文本
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model=self.model,
                dimensions=self.dimensions,
            )

        try:
            # OpenAI API 调用
            response = client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # 记录追踪
            if trace:
                trace.record_stage(
                    stage_name="embedding",
                    elapsed_ms=elapsed_ms,
                    method="openai",
                    provider="openai",
                    details={
                        "model": self.model,
                        "text_count": len(texts),
                        "dimensions": self.dimensions,
                    },
                )

            # 提取向量
            vectors = [item.embedding for item in response.data]

            return EmbeddingResult(
                vectors=vectors,
                model=response.model,
                dimensions=len(vectors[0]) if vectors else self.dimensions,
                usage={
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None,
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "api key" in error_msg:
                raise EmbeddingAuthenticationError(
                    str(e), provider="openai", model=self.model, original_error=e
                )
            elif "rate limit" in error_msg or "429" in error_msg:
                raise EmbeddingRateLimitError(
                    str(e), provider="openai", model=self.model, original_error=e
                )
            elif "connection" in error_msg or "timeout" in error_msg:
                raise EmbeddingConnectionError(
                    str(e), provider="openai", model=self.model, original_error=e
                )
            else:
                raise EmbeddingResponseError(
                    str(e), provider="openai", model=self.model, original_error=e
                )

    def get_model_name(self) -> str:
        """获取模型名称。"""
        return self.model

    def get_dimensions(self) -> int:
        """获取向量维度。"""
        return self.dimensions
