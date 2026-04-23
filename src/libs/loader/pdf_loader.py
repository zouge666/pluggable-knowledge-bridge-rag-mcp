"""
PDF Loader 实现。

使用 MarkItDown 或 PyMuPDF 解析 PDF 文档。
"""

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from src.core.trace.trace_context import TraceContext
from src.core.types import Document, ImageRef, generate_image_id
from src.libs.loader.base_loader import (
    BaseLoader,
    LoaderError,
    ParsingError,
    UnsupportedFormatError,
)

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None  # type: ignore

try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False

if TYPE_CHECKING:
    import fitz as fitz_types


class PdfLoader(BaseLoader):
    """
    PDF 文档加载器。

    支持两种解析方式：
    1. MarkItDown（优先）：将 PDF 转为 Markdown，保留结构
    2. PyMuPDF（备选）：直接提取文本和图片

    图片处理：
    - 提取 PDF 中的图片并保存到指定目录
    - 在文本中插入 [IMAGE: {image_id}] 占位符
    - 在 metadata.images 中记录图片信息
    """

    def __init__(
        self,
        image_output_dir: str = "data/images",
        use_markitdown: bool = True,
        extract_images: bool = True,
    ):
        """
        初始化 PDF Loader。

        Args:
            image_output_dir: 图片输出目录。
            use_markitdown: 是否优先使用 MarkItDown。
            extract_images: 是否提取图片。
        """
        self.image_output_dir = Path(image_output_dir)
        self.use_markitdown = use_markitdown and HAS_MARKITDOWN
        self.extract_images = extract_images and HAS_PYMUPDF

        # 确保图片输出目录存在
        if self.extract_images:
            self.image_output_dir.mkdir(parents=True, exist_ok=True)

    def supports(self, path: Union[str, Path]) -> bool:
        """判断是否为 PDF 文件。"""
        ext = self._get_file_extension(path)
        return ext == ".pdf"

    def load(
        self,
        path: Union[str, Path],
        collection: Optional[str] = None,
        trace: Optional[TraceContext] = None,
    ) -> Document:
        """
        加载 PDF 文档。

        Args:
            path: PDF 文件路径。
            collection: 集合名称（可选）。
            trace: 追踪上下文（可选）。

        Returns:
            Document: 文档对象。

        Raises:
            FileNotFoundError: 文件不存在。
            UnsupportedFormatError: 不是 PDF 文件。
            ParsingError: 解析失败。
        """
        import time
        start_time = time.time()

        path = Path(path)

        # 检查文件存在
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        # 检查文件格式
        if not self.supports(path):
            raise UnsupportedFormatError(
                f"Not a PDF file: {path}",
                path=str(path),
            )

        # 计算文件哈希作为文档 ID
        doc_hash = self._compute_file_hash(path)

        # 确定图片保存目录
        image_dir = self.image_output_dir / (collection or "default") / doc_hash
        if self.extract_images:
            image_dir.mkdir(parents=True, exist_ok=True)

        # 解析 PDF
        try:
            if self.use_markitdown:
                text, images = self._parse_with_markitdown(path, image_dir, doc_hash)
            elif HAS_PYMUPDF:
                text, images = self._parse_with_pymupdf(path, image_dir, doc_hash)
            else:
                raise ParsingError(
                    "No PDF parser available. Install markitdown or pymupdf.",
                    path=str(path),
                )
        except Exception as e:
            if isinstance(e, LoaderError):
                raise
            raise ParsingError(
                f"Failed to parse PDF: {e}",
                path=str(path),
                original_error=e,
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="pdf_load",
                elapsed_ms=elapsed_ms,
                method="markitdown" if self.use_markitdown else "pymupdf",
                details={
                    "doc_hash": doc_hash,
                    "image_count": len(images),
                    "text_length": len(text),
                },
            )

        # 构建 metadata
        metadata = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "file_name": path.name,
            "file_size": path.stat().st_size,
        }

        if images:
            metadata["images"] = [img.to_dict() for img in images]

        if collection:
            metadata["collection"] = collection

        return Document(
            id=doc_hash,
            text=text,
            metadata=metadata,
        )

    def _compute_file_hash(self, path: Path) -> str:
        """计算文件 SHA256 哈希。"""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _parse_with_markitdown(
        self,
        path: Path,
        image_dir: Path,
        doc_hash: str,
    ) -> tuple:
        """
        使用 MarkItDown 解析 PDF。

        MarkItDown 将 PDF 转为 Markdown，保留结构。
        图片提取仍使用 PyMuPDF（如果可用）。

        Returns:
            tuple: (text, images)
        """
        md = MarkItDown()
        result = md.convert(str(path))
        text = result.text_content

        # 使用 PyMuPDF 提取图片（如果可用）
        images = []
        if self.extract_images and HAS_PYMUPDF:
            images = self._extract_images_pymupdf(path, image_dir, doc_hash, text)

        return text, images

    def _parse_with_pymupdf(
        self,
        path: Path,
        image_dir: Path,
        doc_hash: str,
    ) -> tuple:
        """
        使用 PyMuPDF 解析 PDF。

        直接提取文本和图片。

        Returns:
            tuple: (text, images)
        """
        doc = fitz.open(str(path))

        text_parts = []
        images = []

        for page_num, page in enumerate(doc):
            # 提取文本
            page_text = page.get_text()
            text_parts.append(page_text)

            # 提取图片
            if self.extract_images:
                page_images = self._extract_page_images(
                    page, page_num, image_dir, doc_hash
                )
                images.extend(page_images)

        doc.close()

        text = "\n\n".join(text_parts)
        return text, images

    def _extract_images_pymupdf(
        self,
        path: Path,
        image_dir: Path,
        doc_hash: str,
        text: str,
    ) -> List[ImageRef]:
        """
        使用 PyMuPDF 提取图片。

        Args:
            path: PDF 文件路径。
            image_dir: 图片保存目录。
            doc_hash: 文档哈希。
            text: 已提取的文本（用于插入占位符）。

        Returns:
            List[ImageRef]: 图片引用列表。
        """
        doc = fitz.open(str(path))
        images = []

        for page_num, page in enumerate(doc):
            page_images = self._extract_page_images(
                page, page_num, image_dir, doc_hash
            )
            images.extend(page_images)

        doc.close()
        return images

    def _extract_page_images(
        self,
        page: Any,  # fitz.Page (use Any to avoid runtime error when fitz not installed)
        page_num: int,
        image_dir: Path,
        doc_hash: str,
    ) -> List[ImageRef]:
        """
        提取单页中的图片。

        Args:
            page: PyMuPDF 页面对象。
            page_num: 页码。
            image_dir: 图片保存目录。
            doc_hash: 文档哈希。

        Returns:
            List[ImageRef]: 图片引用列表。
        """
        images = []
        image_list = page.get_images(full=True)

        for seq, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)

                if base_image is None:
                    continue

                # 生成图片 ID
                image_id = generate_image_id(doc_hash, page_num, seq)

                # 确定图片格式和文件名
                ext = base_image.get("ext", "png")
                image_filename = f"{image_id}.{ext}"
                image_path = image_dir / image_filename

                # 保存图片
                image_bytes = base_image.get("image")
                if image_bytes:
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    # 获取图片位置信息
                    rects = page.get_image_rects(xref)
                    position = None
                    if rects:
                        rect = rects[0]
                        position = {
                            "x": rect.x0,
                            "y": rect.y0,
                            "width": rect.width,
                            "height": rect.height,
                        }

                    images.append(ImageRef(
                        id=image_id,
                        path=str(image_path),
                        page=page_num,
                        position=position,
                    ))
            except Exception:
                # 图片提取失败，继续处理其他图片
                continue

        return images


class SimplePdfLoader(BaseLoader):
    """
    简单 PDF Loader（仅提取文本，不提取图片）。

    用于不需要图片处理的场景。
    """

    def supports(self, path: Union[str, Path]) -> bool:
        """判断是否为 PDF 文件。"""
        ext = self._get_file_extension(path)
        return ext == ".pdf"

    def load(
        self,
        path: Union[str, Path],
        collection: Optional[str] = None,
    ) -> Document:
        """
        加载 PDF 文档（仅文本）。

        Args:
            path: PDF 文件路径。
            collection: 集合名称（可选）。

        Returns:
            Document: 文档对象。
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if not self.supports(path):
            raise UnsupportedFormatError(
                f"Not a PDF file: {path}",
                path=str(path),
            )

        # 计算文件哈希
        doc_hash = self._compute_file_hash(path)

        # 提取文本
        if HAS_PYMUPDF:
            text = self._extract_text_pymupdf(path)
        elif HAS_MARKITDOWN:
            text = self._extract_text_markitdown(path)
        else:
            raise ParsingError(
                "No PDF parser available. Install markitdown or pymupdf.",
                path=str(path),
            )

        metadata = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "file_name": path.name,
        }

        if collection:
            metadata["collection"] = collection

        return Document(
            id=doc_hash,
            text=text,
            metadata=metadata,
        )

    def _compute_file_hash(self, path: Path) -> str:
        """计算文件哈希。"""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_text_pymupdf(self, path: Path) -> str:
        """使用 PyMuPDF 提取文本。"""
        doc = fitz.open(str(path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)

    def _extract_text_markitdown(self, path: Path) -> str:
        """使用 MarkItDown 提取文本。"""
        md = MarkItDown()
        result = md.convert(str(path))
        return result.text_content