"""
LLM 抽象基类。

定义统一的 LLM 接口，支持多种 provider（Azure/OpenAI/Ollama/DeepSeek）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.trace.trace_context import TraceContext


@dataclass
class ChatMessage:
    """聊天消息。"""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式。"""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """聊天响应。"""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # {"prompt_tokens": x, "completion_tokens": y}
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        result: Dict[str, Any] = {
            "content": self.content,
            "model": self.model,
        }
        if self.usage:
            result["usage"] = self.usage
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        return result


class BaseLLM(ABC):
    """
    LLM 抽象基类。

    所有 LLM 实现（Azure/OpenAI/Ollama/DeepSeek）都必须实现此接口。
    """

    @abstractmethod
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
            temperature: 采样温度（可选，覆盖默认配置）。
            max_tokens: 最大生成 token 数（可选，覆盖默认配置）。
            trace: 追踪上下文（可选）。

        Returns:
            ChatResponse: 聊天响应。

        Raises:
            LLMError: LLM 调用失败时抛出。
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """获取当前使用的模型名称。"""
        pass

    def chat_with_str(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> str:
        """
        便捷方法：使用字符串发送聊天请求。

        Args:
            prompt: 用户输入。
            system_prompt: 系统提示（可选）。
            temperature: 采样温度（可选）。
            max_tokens: 最大生成 token 数（可选）。
            trace: 追踪上下文（可选）。

        Returns:
            str: 生成的文本内容。
        """
        messages: List[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        response = self.chat(messages, temperature, max_tokens, trace)
        return response.content


class LLMError(Exception):
    """LLM 调用错误基类。"""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.provider = provider
        self.model = model
        self.original_error = original_error

        error_parts = []
        if provider:
            error_parts.append(f"[{provider}]")
        if model:
            error_parts.append(f"model={model}")
        error_parts.append(message)

        super().__init__(" ".join(error_parts))


class LLMConnectionError(LLMError):
    """LLM 连接错误。"""
    pass


class LLMAuthenticationError(LLMError):
    """LLM 认证错误。"""
    pass


class LLMRateLimitError(LLMError):
    """LLM 速率限制错误。"""
    pass


class LLMResponseError(LLMError):
    """LLM 响应错误。"""
    pass
