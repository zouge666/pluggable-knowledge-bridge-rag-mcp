"""
Document Chunker - 文档切分适配器。

作为 libs.splitter 和 Ingestion Pipeline 之间的适配器层，
完成 Document → List[Chunk] 的业务对象转换。
"""

import hashlib
import re
from typing import List, Optional

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, Document, ImageRef
from src.libs.splitter.base_splitter import SplitResult
from src.libs.splitter import SplitterFactory


class DocumentChunker:
    """
    文档切分器。

    职责边界：
    - libs.splitter：纯文本切分工具（str → List[str]）
    - DocumentChunker：业务适配器（Document → List[Chunk]）

    增值功能：
    1. Chunk ID 生成（确定性）
    2. 元数据继承
    3. chunk_index 记录
    4. source_ref 溯源链接
    5. 图片引用按需分发
    6. 类型转换
    """

    def __init__(self, settings: Settings):
        """
        初始化 DocumentChunker。

        Args:
            settings: 应用配置。
        """
        self.settings = settings
        self._splitter = SplitterFactory.create(settings)

    def split_document(
        self,
        document: Document,
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """
        将 Document 切分为多个 Chunk。

        Args:
            document: 文档对象。
            trace: 追踪上下文（可选）。

        Returns:
            List[Chunk]: Chunk 列表。
        """
        import time
        start_time = time.time()

        # 使用 libs.splitter 进行纯文本切分
        split_output = self._splitter.split_text(document.text)
        if isinstance(split_output, SplitResult):
            text_chunks = split_output.chunks
        else:
            # Backward compatibility for legacy/mocked splitters returning List[str].
            text_chunks = split_output

        # 获取文档级图片列表
        doc_images = document.get_images()
        doc_images_map = {img.id: img for img in doc_images}

        # 转换为 Chunk 对象
        chunks = []
        for index, chunk_text in enumerate(text_chunks):
            # 生成 Chunk ID
            chunk_id = self._generate_chunk_id(document.id, index, chunk_text)

            # 计算文本偏移量（近似）
            start_offset = self._estimate_offset(document.text, chunk_text, index, text_chunks)
            end_offset = start_offset + len(chunk_text)

            # 继承元数据 + 图片分发
            metadata = self._inherit_metadata(document, index, chunk_text, doc_images_map)

            # 创建 Chunk
            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata=metadata,
                start_offset=start_offset,
                end_offset=end_offset,
                source_ref=document.id,
            )
            chunks.append(chunk)

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            splitter_method = getattr(self._splitter, "get_splitter_type", None)
            method_name = splitter_method() if callable(splitter_method) else "unknown"
            trace.record_stage(
                stage_name="document_chunking",
                elapsed_ms=elapsed_ms,
                method=method_name,
                details={
                    "doc_id": document.id,
                    "chunk_count": len(chunks),
                    "doc_length": len(document.text),
                },
            )

        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """
        生成确定性 Chunk ID。

        格式：{doc_id}_{index:04d}_{hash_8chars}

        Args:
            doc_id: 文档 ID。
            index: Chunk 序号。
            text: Chunk 文本。

        Returns:
            str: Chunk ID。
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{text_hash}"

    def _inherit_metadata(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str,
        doc_images_map: dict,
    ) -> dict:
        """
        继承文档元数据并分发图片引用。

        Args:
            document: 文档对象。
            chunk_index: Chunk 序号。
            chunk_text: Chunk 文本。
            doc_images_map: 文档图片映射（id → ImageRef）。

        Returns:
            dict: Chunk 元数据。
        """
        # 复制文档级元数据
        metadata = document.metadata.copy()

        # 添加 chunk_index
        metadata["chunk_index"] = chunk_index

        # 移除文档级 images（后续按需分发）
        if "images" in metadata:
            del metadata["images"]

        # 图片引用按需分发
        image_refs = self._extract_image_refs(chunk_text)
        if image_refs:
            # 仅包含该 chunk 引用的图片
            chunk_images = []
            for img_id in image_refs:
                if img_id in doc_images_map:
                    chunk_images.append(doc_images_map[img_id])

            if chunk_images:
                metadata["images"] = [img.to_dict() for img in chunk_images]
            metadata["image_refs"] = image_refs

        return metadata

    def _extract_image_refs(self, text: str) -> List[str]:
        """
        从文本中提取图片 ID。

        扫描 [IMAGE: {id}] 占位符。

        Args:
            text: Chunk 文本。

        Returns:
            List[str]: 图片 ID 列表。
        """
        pattern = r"\[IMAGE:\s*([^\]]+)\]"
        matches = re.findall(pattern, text)
        return [m.strip() for m in matches]

    def _estimate_offset(
        self,
        full_text: str,
        chunk_text: str,
        index: int,
        all_chunks: List[str],
    ) -> int:
        """
        估算 Chunk 在原文中的起始偏移量。

        Args:
            full_text: 完整文档文本。
            chunk_text: Chunk 文本。
            index: Chunk 序号。
            all_chunks: 所有 Chunk 文本列表。

        Returns:
            int: 起始偏移量。
        """
        # 简单估算：累加前面所有 chunk 的长度
        offset = 0
        for i in range(index):
            offset += len(all_chunks[i])
        return offset


class FakeDocumentChunker:
    """
    Fake Document Chunker 用于测试。

    不依赖真实 Splitter，直接按固定长度切分。
    """

    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size

    def split_document(self, document: Document) -> List[Chunk]:
        """切分文档。"""
        text = document.text
        chunks = []

        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i:i + self.chunk_size]
            index = i // self.chunk_size

            chunk_id = hashlib.sha256(
                f"{document.id}_{index}_{chunk_text}".encode()
            ).hexdigest()[:16]

            metadata = document.metadata.copy()
            metadata["chunk_index"] = index

            # 图片分发
            image_refs = self._extract_image_refs(chunk_text)
            if image_refs:
                metadata["image_refs"] = image_refs

            chunks.append(Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata=metadata,
                start_offset=i,
                end_offset=i + len(chunk_text),
                source_ref=document.id,
            ))

        return chunks

    def _extract_image_refs(self, text: str) -> List[str]:
        """提取图片 ID。"""
        pattern = r"\[IMAGE:\s*([^\]]+)\]"
        matches = re.findall(pattern, text)
        return [m.strip() for m in matches]
