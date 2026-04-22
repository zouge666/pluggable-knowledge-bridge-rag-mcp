"""
DeepSeek LLM 实现。

DeepSeek API 兼容 OpenAI API 格式。
"""

import os
from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.libs.llm.base_llm import (
    BaseLLM,
    ChatMessage,
    ChatResponse,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMResponseError,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# DeepSeek API 基础地址
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekLLM(BaseLLM):
    """
    DeepSeek LLM 实现。

    DeepSeek API 兼容 OpenAI API 格式，使用 OpenAI SDK 调用。
    """

    def __init__(self, settings) -> None:
        """
        初始化 DeepSeek LLM。

        Args:
            settings: LLM 配置（LLMSettings 或 Settings）。
        """
        if hasattr(settings, "llm"):
            settings = settings.llm

        self.model = settings.model or "deepseek-chat"
        self.temperature = settings.temperature or 0.0
        self.max_tokens = settings.max_tokens or 4096

        # 获取 API Key
        self.api_key = settings.api_key
        if self.api_key and self.api_key.startswith("${") and self.api_key.endswith("}"):
            env_var = self.api_key[2:-1]
            self.api_key = os.environ.get(env_var, "")

        # 自定义 base_url（可选）
        self.base_url = getattr(settings, "base_url", None) or DEEPSEEK_BASE_URL

        # 延迟初始化客户端
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """获取或创建 OpenAI 客户端（兼容 DeepSeek API）。"""
        if self._client is None:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required. Install it with: pip install openai"
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> ChatResponse:
        """
        发送聊天请求。

        Args:
            messages: 聊天消息列表。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。
            trace: 追踪上下文。

        Returns:
            ChatResponse: 聊天响应。
        """
        import time

        start_time = time.time()
        client = self._get_client()

        # 转换消息格式
        formatted_messages = [msg.to_dict() for msg in messages]

        # 使用传入参数或默认配置
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temp,
                max_tokens=tokens,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # 记录追踪
            if trace:
                trace.record_stage(
                    stage_name="llm_chat",
                    elapsed_ms=elapsed_ms,
                    method="deepseek",
                    provider="deepseek",
                    details={
                        "model": self.model,
                        "message_count": len(messages),
                    },
                )

            # 构建响应
            choice = response.choices[0]
            return ChatResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
                if response.usage
                else None,
                finish_reason=choice.finish_reason,
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "api key" in error_msg or "401" in error_msg:
                raise LLMAuthenticationError(
                    str(e), provider="deepseek", model=self.model, original_error=e
                )
            elif "rate limit" in error_msg or "429" in error_msg:
                raise LLMRateLimitError(
                    str(e), provider="deepseek", model=self.model, original_error=e
                )
            elif "connection" in error_msg or "timeout" in error_msg:
                raise LLMConnectionError(
                    str(e), provider="deepseek", model=self.model, original_error=e
                )
            else:
                raise LLMResponseError(
                    str(e), provider="deepseek", model=self.model, original_error=e
                )

    def get_model_name(self) -> str:
        """获取模型名称。"""
        return self.model
