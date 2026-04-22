"""
Ollama Embedding 实现。

支持通过 Ollama HTTP API 调用本地部署的 Embedding 模型。
"""

import json
from typing import List, Optional

import requests

from src.core.trace.trace_context import TraceContext
from src.libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResult,
    EmbeddingConnectionError,
    EmbeddingResponseError,
)


class OllamaEmbedding(BaseEmbedding):
    """
    Ollama Embedding 实现。

    支持通过 Ollama HTTP API 调用本地部署的 Embedding 模型
    （如 nomic-embed-text、mxbai-embed-large 等）。
    """

    def __init__(self, settings) -> None:
        """
        初始化 Ollama Embedding。

        Args:
            settings: Embedding 配置（EmbeddingSettings 或 Settings）。
        """
        if hasattr(settings, "embedding"):
            settings = settings.embedding

        self.model = settings.model or "nomic-embed-text"
        # Ollama 模型维度由模型决定，常见值：
        # - nomic-embed-text: 768
        # - mxbai-embed-large: 1024
        self.dimensions = settings.dimensions or 768

        # Ollama 服务地址
        self.base_url = getattr(settings, "base_url", None) or "http://localhost:11434"

        # 请求超时
        self.timeout = 60

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

        # 过滤空文本
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model=self.model,
                dimensions=self.dimensions,
            )

        vectors = []
        total_tokens = 0

        try:
            # Ollama 的 /api/embeddings 接口一次处理一个文本
            for text in texts:
                payload = {
                    "model": self.model,
                    "prompt": text,
                }

                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                data = response.json()
                embedding = data.get("embedding", [])
                vectors.append(embedding)

                # 估算 token 数（Ollama 不返回 token 数）
                total_tokens += len(text.split())

            elapsed_ms = (time.time() - start_time) * 1000

            # 记录追踪
            if trace:
                trace.record_stage(
                    stage_name="embedding",
                    elapsed_ms=elapsed_ms,
                    method="ollama",
                    provider="ollama",
                    details={
                        "model": self.model,
                        "text_count": len(texts),
                    },
                )

            # 从第一个向量推断维度
            actual_dimensions = len(vectors[0]) if vectors else self.dimensions

            return EmbeddingResult(
                vectors=vectors,
                model=self.model,
                dimensions=actual_dimensions,
                usage={"total_tokens": total_tokens},
            )

        except requests.exceptions.ConnectionError as e:
            raise EmbeddingConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running (ollama serve)",
                provider="ollama",
                model=self.model,
                original_error=e,
            )
        except requests.exceptions.Timeout as e:
            raise EmbeddingConnectionError(
                f"Ollama request timed out after {self.timeout}s",
                provider="ollama",
                model=self.model,
                original_error=e,
            )
        except requests.exceptions.HTTPError as e:
            raise EmbeddingResponseError(
                str(e),
                provider="ollama",
                model=self.model,
                original_error=e,
            )
        except json.JSONDecodeError as e:
            raise EmbeddingResponseError(
                f"Failed to parse Ollama response: {e}",
                provider="ollama",
                model=self.model,
                original_error=e,
            )

    def get_model_name(self) -> str:
        """获取模型名称。"""
        return self.model

    def get_dimensions(self) -> int:
        """获取向量维度。"""
        return self.dimensions
