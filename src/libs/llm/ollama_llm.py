"""
Ollama LLM 实现。
"""

import json
from typing import List, Optional

import requests

from src.core.trace.trace_context import TraceContext
from src.libs.llm.base_llm import (
    BaseLLM,
    ChatMessage,
    ChatResponse,
    LLMConnectionError,
    LLMResponseError,
)


class OllamaLLM(BaseLLM):
    """
    Ollama LLM 实现。

    支持通过 Ollama HTTP API 调用本地部署的模型。
    """

    def __init__(self, settings) -> None:
        """
        初始化 Ollama LLM。

        Args:
            settings: LLM 配置（LLMSettings 或 Settings）。
        """
        if hasattr(settings, "llm"):
            settings = settings.llm

        self.model = settings.model or "llama3"
        self.temperature = settings.temperature or 0.0
        self.max_tokens = settings.max_tokens or 4096

        # Ollama 服务地址
        self.base_url = getattr(settings, "base_url", None) or "http://localhost:11434"

        # 请求超时
        self.timeout = 120

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

        # 转换消息格式
        formatted_messages = [msg.to_dict() for msg in messages]

        # 使用传入参数或默认配置
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # 构建请求
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "options": {
                "temperature": temp,
                "num_predict": tokens,
            },
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            elapsed_ms = (time.time() - start_time) * 1000

            # 记录追踪
            if trace:
                trace.record_stage(
                    stage_name="llm_chat",
                    elapsed_ms=elapsed_ms,
                    method="ollama",
                    provider="ollama",
                    details={
                        "model": self.model,
                        "message_count": len(messages),
                    },
                )

            # 解析响应
            data = response.json()
            message = data.get("message", {})

            return ChatResponse(
                content=message.get("content", ""),
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                finish_reason="stop" if data.get("done") else None,
            )

        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running (ollama serve)",
                provider="ollama",
                model=self.model,
                original_error=e,
            )
        except requests.exceptions.Timeout as e:
            raise LLMConnectionError(
                f"Ollama request timed out after {self.timeout}s",
                provider="ollama",
                model=self.model,
                original_error=e,
            )
        except requests.exceptions.HTTPError as e:
            raise LLMResponseError(
                str(e),
                provider="ollama",
                model=self.model,
                original_error=e,
            )
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"Failed to parse Ollama response: {e}",
                provider="ollama",
                model=self.model,
                original_error=e,
            )

    def get_model_name(self) -> str:
        """获取模型名称。"""
        return self.model
