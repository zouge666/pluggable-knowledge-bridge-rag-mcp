"""
Azure Vision LLM 实现。

支持通过 Azure OpenAI 调用 GPT-4o/GPT-4-Vision 进行图像理解。
"""

import base64
import io
import os
from pathlib import Path
from typing import List, Optional, Union

from src.core.trace.trace_context import TraceContext
from src.libs.llm.base_llm import ChatResponse, LLMAuthenticationError, LLMConnectionError, LLMRateLimitError, LLMResponseError
from src.libs.llm.base_vision_llm import (
    BaseVisionLLM,
    ImageContent,
    VisionLLMError,
    VisionLLMImageError,
    VisionLLMSizeError,
)

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


class AzureVisionLLM(BaseVisionLLM):
    """
    Azure Vision LLM 实现。

    支持通过 Azure OpenAI 调用 GPT-4o/GPT-4-Vision-Preview 进行图像理解。
    """

    def __init__(self, settings) -> None:
        """
        初始化 Azure Vision LLM。

        Args:
            settings: Vision LLM 配置（VisionLLMSettings 或 Settings）。
        """
        if hasattr(settings, "vision_llm"):
            settings = settings.vision_llm

        self.model = settings.model or "gpt-4o"
        self.deployment_name = getattr(settings, "deployment_name", None) or self.model
        self.temperature = getattr(settings, "temperature", 0.0) or 0.0
        self.max_tokens = getattr(settings, "max_tokens", 4096) or 4096

        # Azure 特有配置
        self.azure_endpoint = settings.azure_endpoint
        self.api_version = getattr(settings, "api_version", None) or "2024-02-15-preview"

        # 获取 API Key
        self.api_key = settings.api_key
        if self.api_key and self.api_key.startswith("${") and self.api_key.endswith("}"):
            env_var = self.api_key[2:-1]
            self.api_key = os.environ.get(env_var, "")

        # 图片处理配置
        self.max_image_size = getattr(settings, "max_image_size", 2048) or 2048

        # 延迟初始化客户端
        self._client = None

    def _get_client(self):
        """获取或创建 Azure OpenAI 客户端。"""
        if self._client is None:
            if AzureOpenAI is None:
                raise ImportError(
                    "openai package is required. Install it with: pip install openai"
                )
            self._client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
            )
        return self._client

    def _compress_image_if_needed(
        self,
        image: ImageContent,
    ) -> ImageContent:
        """
        如果图片过大，进行压缩。

        Args:
            image: 图片内容。

        Returns:
            ImageContent: 可能压缩后的图片。
        """
        try:
            from PIL import Image
        except ImportError:
            # PIL 未安装，跳过压缩
            return image

        # 获取图片数据
        if isinstance(image.source, bytes):
            img_data = image.source
        elif isinstance(image.source, (str, Path)):
            path = Path(image.source)
            if path.exists():
                with open(path, "rb") as f:
                    img_data = f.read()
            else:
                return image
        else:
            return image

        try:
            # 打开图片
            img = Image.open(io.BytesIO(img_data))
            original_size = img.size

            # 检查是否需要压缩
            max_dim = max(original_size)
            if max_dim <= self.max_image_size:
                return image

            # 计算缩放比例
            scale = self.max_image_size / max_dim
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))

            # 缩放图片
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

            # 转换为 bytes
            output = io.BytesIO()
            format_map = {"PNG": "PNG", "JPEG": "JPEG", "JPG": "JPEG", "GIF": "GIF", "WEBP": "WEBP"}
            img_format = format_map.get(img.format, "PNG")
            img_resized.save(output, format=img_format)
            compressed_data = output.getvalue()

            return ImageContent(
                source=compressed_data,
                mime_type=f"image/{img_format.lower()}",
                compressed=True,
                original_size=original_size,
            )

        except Exception:
            # 压缩失败，返回原图
            return image

    def _build_image_content(
        self,
        image: ImageContent,
    ) -> dict:
        """
        构建 OpenAI API 所需的图片内容结构。

        Args:
            image: 图片内容。

        Returns:
            dict: OpenAI API 图片内容结构。
        """
        # 确保有 MIME 类型
        mime_type = image.mime_type or "image/png"

        # 获取 base64 数据
        base64_data = image.to_base64()

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_data}",
            },
        }

    def chat_with_image(
        self,
        text: str,
        image: Union[str, bytes, Path, ImageContent],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> ChatResponse:
        """
        发送带图片的聊天请求。

        Args:
            text: 文本内容。
            image: 图片。
            system_prompt: 系统提示。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。
            trace: 追踪上下文。

        Returns:
            ChatResponse: 聊天响应。
        """
        import time

        start_time = time.time()

        # 标准化图片
        normalized_image = self._normalize_image(image)

        # 压缩图片（如果需要）
        compressed_image = self._compress_image_if_needed(normalized_image)

        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 构建用户消息（文本 + 图片）
        user_content = [
            {"type": "text", "text": text},
            self._build_image_content(compressed_image),
        ]
        messages.append({"role": "user", "content": user_content})

        client = self._get_client()
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # 记录追踪
            if trace:
                trace.record_stage(
                    stage_name="vision_llm_chat",
                    elapsed_ms=elapsed_ms,
                    method="azure",
                    provider="azure",
                    details={
                        "model": self.model,
                        "deployment": self.deployment_name,
                        "image_compressed": compressed_image.compressed,
                    },
                )

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
                    str(e), provider="azure", model=self.model, original_error=e
                )
            elif "rate limit" in error_msg or "429" in error_msg:
                raise LLMRateLimitError(
                    str(e), provider="azure", model=self.model, original_error=e
                )
            elif "connection" in error_msg or "timeout" in error_msg:
                raise LLMConnectionError(
                    str(e), provider="azure", model=self.model, original_error=e
                )
            else:
                raise LLMResponseError(
                    str(e), provider="azure", model=self.model, original_error=e
                )

    def chat_with_images(
        self,
        text: str,
        images: List[Union[str, bytes, Path, ImageContent]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> ChatResponse:
        """
        发送带多张图片的聊天请求。

        Args:
            text: 文本内容。
            images: 图片列表。
            system_prompt: 系统提示。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。
            trace: 追踪上下文。

        Returns:
            ChatResponse: 聊天响应。
        """
        import time

        start_time = time.time()

        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 构建用户消息（文本 + 多张图片）
        user_content = [{"type": "text", "text": text}]
        for image in images:
            normalized = self._normalize_image(image)
            compressed = self._compress_image_if_needed(normalized)
            user_content.append(self._build_image_content(compressed))

        messages.append({"role": "user", "content": user_content})

        client = self._get_client()
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # 记录追踪
            if trace:
                trace.record_stage(
                    stage_name="vision_llm_chat",
                    elapsed_ms=elapsed_ms,
                    method="azure",
                    provider="azure",
                    details={
                        "model": self.model,
                        "deployment": self.deployment_name,
                        "image_count": len(images),
                    },
                )

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
                    str(e), provider="azure", model=self.model, original_error=e
                )
            elif "rate limit" in error_msg or "429" in error_msg:
                raise LLMRateLimitError(
                    str(e), provider="azure", model=self.model, original_error=e
                )
            elif "connection" in error_msg or "timeout" in error_msg:
                raise LLMConnectionError(
                    str(e), provider="azure", model=self.model, original_error=e
                )
            else:
                raise LLMResponseError(
                    str(e), provider="azure", model=self.model, original_error=e
                )

    def get_model_name(self) -> str:
        """获取模型名称。"""
        return self.model
