"""
Multimodal Assembler。

组装多模态响应（Text + Image）。
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult, ImageRef

logger = logging.getLogger("core.response.multimodal_assembler")


class MultimodalAssembler:
    """
    多模态组装器。

    当命中 chunk 含 image_refs 时，读取图片并 base64 返回 ImageContent。
    """

    def __init__(self, images_path: str = "data/images"):
        """
        初始化组装器。

        Args:
            images_path: 图片存储路径。
        """
        self._images_path = Path(images_path)

    def assemble(
        self,
        results: List[RetrievalResult],
        include_images: bool = True,
        max_images: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        组装多模态内容。

        Args:
            results: 检索结果列表。
            include_images: 是否包含图片。
            max_images: 最大图片数量。

        Returns:
            MCP content 列表。
        """
        contents = []
        image_count = 0

        for result in results:
            # 添加文本内容
            text_content = self._build_text_content(result)
            contents.append(text_content)

            # 添加图片内容
            if include_images and image_count < max_images:
                images = self._get_images_from_result(result)
                for image_ref in images:
                    if image_count >= max_images:
                        break

                    image_content = self._build_image_content(image_ref)
                    if image_content:
                        contents.append(image_content)
                        image_count += 1

        return contents

    def _build_text_content(self, result: RetrievalResult) -> Dict[str, Any]:
        """
        构建文本内容。

        Args:
            result: 检索结果。

        Returns:
            MCP text content。
        """
        # 提取来源信息
        source = result.metadata.get("source_path", "unknown")
        page = result.metadata.get("page")

        source_info = f"来源：{source}"
        if page is not None:
            source_info += f" (第 {page} 页)"

        text = f"{source_info}\n\n{result.text}"

        return {
            "type": "text",
            "text": text,
        }

    def _get_images_from_result(self, result: RetrievalResult) -> List[ImageRef]:
        """
        从检索结果获取图片引用。

        Args:
            result: 检索结果。

        Returns:
            图片引用列表。
        """
        images_data = result.metadata.get("images", [])
        image_refs = []

        for img_data in images_data:
            if isinstance(img_data, dict):
                image_refs.append(ImageRef.from_dict(img_data))
            elif isinstance(img_data, ImageRef):
                image_refs.append(img_data)

        return image_refs

    def _build_image_content(self, image_ref: ImageRef) -> Optional[Dict[str, Any]]:
        """
        构建图片内容。

        Args:
            image_ref: 图片引用。

        Returns:
            MCP image content，如果图片不存在返回 None。
        """
        # 查找图片文件
        image_path = self._find_image_file(image_ref)

        if image_path is None or not image_path.exists():
            logger.warning(f"Image file not found: {image_ref.id}")
            return None

        try:
            # 读取并 base64 编码
            with open(image_path, "rb") as f:
                image_data = f.read()

            base64_data = base64.b64encode(image_data).decode("utf-8")

            # 确定 MIME 类型
            mime_type = self._get_mime_type(image_path)

            return {
                "type": "image",
                "data": base64_data,
                "mimeType": mime_type,
            }

        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            return None

    def _find_image_file(self, image_ref: ImageRef) -> Optional[Path]:
        """
        查找图片文件。

        Args:
            image_ref: 图片引用。

        Returns:
            图片文件路径。
        """
        # 尝试从 path 字段查找
        if image_ref.path:
            path = Path(image_ref.path)
            if path.exists():
                return path

            # 尝试相对于 images_path
            relative_path = self._images_path / image_ref.path
            if relative_path.exists():
                return relative_path

        # 尝试根据 id 查找
        # 格式：{doc_hash}_{page}_{seq}
        image_id = image_ref.id

        # 在 images 目录下搜索
        for collection_dir in self._images_path.iterdir():
            if collection_dir.is_dir():
                for image_file in collection_dir.iterdir():
                    if image_file.stem == image_id or image_file.name.startswith(image_id):
                        return image_file

        return None

    def _get_mime_type(self, image_path: Path) -> str:
        """
        获取 MIME 类型。

        Args:
            image_path: 图片路径。

        Returns:
            MIME 类型字符串。
        """
        suffix = image_path.suffix.lower()

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }

        return mime_types.get(suffix, "image/png")


def assemble_multimodal_response(
    results: List[RetrievalResult],
    query: str,
    include_images: bool = True,
    max_images: int = 5,
) -> Dict[str, Any]:
    """
    组装多模态响应。

    Args:
        results: 检索结果列表。
        query: 用户查询。
        include_images: 是否包含图片。
        max_images: 最大图片数量。

    Returns:
        MCP 格式响应。
    """
    assembler = MultimodalAssembler()
    contents = assembler.assemble(results, include_images, max_images)

    # 添加标题
    if contents:
        title = {
            "type": "text",
            "text": f"# 查询结果：{query}\n\n找到 {len(results)} 个相关文档片段。",
        }
        contents.insert(0, title)

    return {
        "content": contents,
        "structuredContent": {
            "total_results": len(results),
            "query": query,
            "has_images": any(c["type"] == "image" for c in contents),
        },
    }