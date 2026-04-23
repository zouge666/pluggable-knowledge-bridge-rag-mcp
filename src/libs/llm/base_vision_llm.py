"""
Vision LLM 抽象基类。

定义统一的多模态 LLM 接口，支持文本+图片输入。
"""

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.trace.trace_context import TraceContext
from src.libs.llm.base_llm import ChatResponse, LLMError


@dataclass
class ImageContent:
    """图片内容。"""

    # 图片数据：可以是文件路径、base64 字符串或 bytes
    source: Union[str, bytes, Path]
    # 图片 MIME 类型
    mime_type: Optional[str] = None
    # 是否已压缩
    compressed: bool = False
    # 原始尺寸（压缩前）
    original_size: Optional[tuple] = None

    def to_base64(self) -> str:
        """
        将图片转换为 base64 字符串。

        Returns:
            str: base64 编码的图片数据。
        """
        if isinstance(self.source, bytes):
            return base64.b64encode(self.source).decode("utf-8")
        elif isinstance(self.source, str):
            # 检查是否已经是 base64
            if self.source.startswith("data:"):
                # data:image/png;base64,xxxxx 格式
                return self.source.split(",", 1)[1]
            # 假设是文件路径
            path = Path(self.source)
            if path.exists():
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            # 假设已经是 base64
            return self.source
        elif isinstance(self.source, Path):
            with open(self.source, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image source type: {type(self.source)}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "mime_type": self.mime_type,
            "compressed": self.compressed,
            "original_size": self.original_size,
        }


class BaseVisionLLM(ABC):
    """
    Vision LLM 抽象基类。

    所有 Vision LLM 实现（Azure/OpenAI/Ollama）都必须实现此接口。
    支持文本+图片的多模态输入。
    """

    @abstractmethod
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
            image: 图片（路径、bytes 或 ImageContent）。
            system_prompt: 系统提示（可选）。
            temperature: 采样温度（可选）。
            max_tokens: 最大生成 token 数（可选）。
            trace: 追踪上下文（可选）。

        Returns:
            ChatResponse: 聊天响应。

        Raises:
            LLMError: LLM 调用失败时抛出。
        """
        pass

    @abstractmethod
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
            system_prompt: 系统提示（可选）。
            temperature: 采样温度（可选）。
            max_tokens: 最大生成 token 数（可选）。
            trace: 追踪上下文（可选）。

        Returns:
            ChatResponse: 聊天响应。
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """获取当前使用的模型名称。"""
        pass

    def _normalize_image(
        self,
        image: Union[str, bytes, Path, ImageContent],
    ) -> ImageContent:
        """
        将各种格式的图片转换为 ImageContent。

        Args:
            image: 图片输入。

        Returns:
            ImageContent: 标准化的图片内容。
        """
        if isinstance(image, ImageContent):
            return image

        # 推断 MIME 类型
        mime_type = None
        if isinstance(image, (str, Path)):
            path = Path(image)
            suffix = path.suffix.lower()
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(suffix, "image/png")

        return ImageContent(source=image, mime_type=mime_type)

    def _compress_image(
        self,
        image: ImageContent,
        max_size: int = 2048,
    ) -> ImageContent:
        """
        压缩图片（如果需要）。

        Args:
            image: 图片内容。
            max_size: 最大边长（像素）。

        Returns:
            ImageContent: 压缩后的图片。
        """
        # 默认实现：不做压缩，子类可覆盖
        # 如果需要压缩，可以使用 PIL 或其他库
        return image


class VisionLLMError(LLMError):
    """Vision LLM 错误基类。"""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, provider, model, original_error)


class VisionLLMImageError(VisionLLMError):
    """图片处理错误。"""
    pass


class VisionLLMSizeError(VisionLLMError):
    """图片尺寸错误。"""
    pass
