"""
Image Captioner - 图片描述生成处理器。

当启用 Vision LLM 且存在 image_refs 时生成 caption 并写回 chunk metadata。
当禁用/不可用/异常时走降级路径，不阻塞 ingestion。
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from src.core.settings import Settings
from src.core.types import Chunk, ImageRef
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageContent
from src.libs.llm.llm_factory import LLMFactory


class ImageCaptioner(BaseTransform):
    """
    图片描述生成处理器。

    功能：
    1. 检查 Chunk 是否包含图片引用 (image_refs)
    2. 启用 Vision LLM 时，为每张图片生成描述
    3. 将描述写入 chunk metadata 或追加到文本
    4. 降级机制：Vision LLM 禁用/不可用/异常时，标记 has_unprocessed_images
    """

    def __init__(
        self,
        settings: Settings,
        vision_llm: Optional[BaseVisionLLM] = None,
        prompt_path: Optional[str] = None,
    ):
        """
        初始化 ImageCaptioner。

        Args:
            settings: 应用配置。
            vision_llm: Vision LLM 实例（可选）。
            prompt_path: Prompt 模板路径（可选）。
        """
        self.settings = settings
        self.enabled = settings.vision_llm.enabled if hasattr(settings, 'vision_llm') else False

        # Vision LLM 实例（可选）
        self._vision_llm = vision_llm
        if self.enabled and self._vision_llm is None:
            try:
                self._vision_llm = LLMFactory.create_vision_llm(settings)
            except Exception:
                # Vision LLM 创建失败，降级为禁用模式
                self.enabled = False

        # Prompt 模板
        self._prompt_template = self._load_prompt(prompt_path)

        # 图片存储基础路径
        self._image_base_path = Path("data/images")

    def _load_prompt(self, prompt_path: Optional[str] = None) -> str:
        """
        加载 Prompt 模板。

        Args:
            prompt_path: Prompt 文件路径（可选）。

        Returns:
            str: Prompt 模板。
        """
        default_path = "config/prompts/image_captioning.txt"

        if prompt_path:
            path = Path(prompt_path)
        else:
            path = Path(default_path)

        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass

        # 默认 Prompt
        return """You are an AI assistant specialized in analyzing and describing images from technical documents.

## Task

Analyze the provided image and generate a detailed text description that captures:
1. The main content and purpose of the image
2. Key elements, labels, or text visible in the image
3. Relationships between components (for diagrams, charts, etc.)
4. Any data or numerical information (for charts, tables)

## Output Format

Provide a clear, concise description in 2-4 sentences."""

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """
        对 Chunk 列表进行图片描述生成。

        Args:
            chunks: 待处理的 Chunk 列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[Chunk]: 处理后的 Chunk 列表。
        """
        start_time = time.time()
        processed_chunks = []

        total_images = 0
        captioned_images = 0
        failed_images = 0

        for chunk in chunks:
            processed_chunk, stats = self._process_chunk_images(chunk, trace)
            processed_chunks.append(processed_chunk)

            total_images += stats.get("total", 0)
            captioned_images += stats.get("captioned", 0)
            failed_images += stats.get("failed", 0)

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="image_captioning",
                elapsed_ms=elapsed_ms,
                method="vision_llm" if self.enabled else "disabled",
                details={
                    "chunk_count": len(chunks),
                    "total_images": total_images,
                    "captioned_images": captioned_images,
                    "failed_images": failed_images,
                    "enabled": self.enabled,
                },
            )

        return processed_chunks

    def _process_chunk_images(
        self,
        chunk: Chunk,
        trace: Optional[TraceContext] = None,
    ) -> tuple:
        """
        处理单个 Chunk 中的图片。

        Args:
            chunk: 待处理的 Chunk。
            trace: 追踪上下文（可选）。

        Returns:
            tuple: (处理后的 Chunk, 处理统计)
        """
        # 获取图片引用
        image_refs = chunk.get_images()
        image_ids = chunk.get_image_refs()

        if not image_refs and not image_ids:
            # 无图片，直接返回
            return chunk, {"total": 0, "captioned": 0, "failed": 0}

        stats = {
            "total": len(image_refs) if image_refs else len(image_ids),
            "captioned": 0,
            "failed": 0,
        }

        metadata = chunk.metadata.copy()
        captions: Dict[str, str] = {}

        # 如果启用 Vision LLM，尝试生成描述
        if self.enabled and self._vision_llm:
            for image_ref in image_refs:
                caption = self._generate_caption(image_ref, chunk.text, trace)

                if caption:
                    captions[image_ref.id] = caption
                    stats["captioned"] += 1
                else:
                    stats["failed"] += 1

        # 写入 metadata
        if captions:
            metadata["image_captions"] = captions
            metadata["captioned_by"] = "vision_llm"
        elif self.enabled and stats["total"] > 0:
            # 启用但未能生成描述
            metadata["has_unprocessed_images"] = True
            metadata["captioned_by"] = "failed"
        else:
            # 禁用模式
            metadata["has_unprocessed_images"] = True
            metadata["captioned_by"] = "disabled"

        # 可选：将描述追加到文本中
        text = chunk.text
        if captions:
            for image_id, caption in captions.items():
                # 替换占位符或追加描述
                placeholder = f"[IMAGE: {image_id}]"
                if placeholder in text:
                    text = text.replace(placeholder, f"[IMAGE: {image_id}]\n图片描述: {caption}")
                else:
                    # 追加到文本末尾
                    if "图片描述:" not in text:
                        text += "\n\n--- 图片描述 ---\n"
                    text += f"\n{image_id}: {caption}"

        return Chunk(
            id=chunk.id,
            text=text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        ), stats

    def _generate_caption(
        self,
        image_ref: ImageRef,
        context_text: str,
        trace: Optional[TraceContext] = None,
    ) -> Optional[str]:
        """
        为单张图片生成描述。

        Args:
            image_ref: 图片引用信息。
            context_text: Chunk 文本（作为上下文）。
            trace: 追踪上下文（可选）。

        Returns:
            Optional[str]: 图片描述，失败返回 None。
        """
        if not self._vision_llm:
            return None

        # 构建图片路径
        image_path = self._resolve_image_path(image_ref)

        if not image_path or not image_path.exists():
            return None

        try:
            # 构建带上下文的 prompt
            prompt = self._build_prompt(context_text)

            # 调用 Vision LLM
            response = self._vision_llm.chat_with_image(
                text=prompt,
                image=image_path,
                system_prompt=self._prompt_template,
            )

            caption = response.content.strip()

            # 验证描述质量
            if len(caption) < 10:
                return None

            return caption

        except Exception:
            # Vision LLM 调用失败
            return None

    def _resolve_image_path(self, image_ref: ImageRef) -> Optional[Path]:
        """
        解析图片路径。

        Args:
            image_ref: 图片引用信息。

        Returns:
            Optional[Path]: 图片文件路径。
        """
        # 优先使用 image_ref.path
        if image_ref.path:
            path = Path(image_ref.path)
            if path.exists():
                return path

        # 尝试从 image_id 推断路径
        # 格式：{doc_hash}_{page}_{seq}
        image_id = image_ref.id

        # 搜索可能的路径
        possible_paths = [
            self._image_base_path / "default" / f"{image_id}.png",
            self._image_base_path / "default" / f"{image_id}.jpg",
            self._image_base_path / f"{image_id}.png",
            self._image_base_path / f"{image_id}.jpg",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def _build_prompt(self, context_text: str) -> str:
        """
        构建带上下文的 prompt。

        Args:
            context_text: Chunk 文本。

        Returns:
            str: 构建的 prompt。
        """
        # 截取上下文（避免过长）
        context = context_text[:500] if context_text else ""

        if context:
            return f"""请分析这张图片并生成详细描述。

图片上下文：
{context}

请描述图片的主要内容、关键元素和目的。"""
        else:
            return "请分析这张图片并生成详细描述。"


class FakeImageCaptioner(BaseTransform):
    """
    Fake Image Captioner 用于测试。

    不依赖真实 Vision LLM，直接返回固定描述或标记未处理。
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """处理 Chunk 列表。"""
        start_time = time.time()
        processed_chunks = []

        for chunk in chunks:
            image_refs = chunk.get_images()

            metadata = chunk.metadata.copy()

            if image_refs:
                if self.enabled:
                    # 模拟生成描述
                    captions = {}
                    for img_ref in image_refs:
                        captions[img_ref.id] = f"Fake caption for image {img_ref.id}"

                    metadata["image_captions"] = captions
                    metadata["captioned_by"] = "fake_vision_llm"
                else:
                    # 禁用模式
                    metadata["has_unprocessed_images"] = True
                    metadata["captioned_by"] = "disabled"

            processed_chunks.append(Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata=metadata,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                source_ref=chunk.source_ref,
            ))

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                stage_name="image_captioning",
                elapsed_ms=elapsed_ms,
                method="fake" if self.enabled else "disabled",
                details={
                    "chunk_count": len(chunks),
                    "enabled": self.enabled,
                },
            )

        return processed_chunks