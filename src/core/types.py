"""
核心数据类型/契约。

定义全链路（ingestion → retrieval → mcp tools）共用的数据结构。
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ImageRef:
    """
    图片引用信息。

    用于记录文档中图片的位置和存储信息。
    """

    # 全局唯一图片标识符（格式：{doc_hash}_{page}_{seq}）
    id: str
    # 图片文件存储路径（约定：data/images/{collection}/{image_id}.png）
    path: str
    # 图片在原文档中的页码（可选，适用于PDF等分页文档）
    page: Optional[int] = None
    # 占位符在 Document.text 中的起始字符位置（从0开始计数）
    text_offset: int = 0
    # 占位符的字符长度（通常为 len("[IMAGE: {image_id}]")）
    text_length: int = 0
    # 图片在原文档中的物理位置信息（可选，如PDF坐标、像素位置、尺寸等）
    position: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        result = {
            "id": self.id,
            "path": self.path,
            "text_offset": self.text_offset,
            "text_length": self.text_length,
        }
        if self.page is not None:
            result["page"] = self.page
        if self.position is not None:
            result["position"] = self.position
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageRef":
        """从字典创建 ImageRef。"""
        return cls(
            id=data["id"],
            path=data["path"],
            page=data.get("page"),
            text_offset=data.get("text_offset", 0),
            text_length=data.get("text_length", 0),
            position=data.get("position"),
        )


@dataclass
class Document:
    """
    文档对象。

    表示一个完整的文档（如 PDF、Markdown 文件等）。
    """

    # 文档唯一标识符（通常是文件内容的 SHA256）
    id: str
    # 文档文本内容（图片位置使用 [IMAGE: {image_id}] 占位符）
    text: str
    # 元数据（至少包含 source_path，可增量扩展）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """从字典创建 Document。"""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "Document":
        """从 JSON 字符串创建 Document。"""
        return cls.from_dict(json.loads(json_str))

    def get_source_path(self) -> Optional[str]:
        """获取文档来源路径。"""
        return self.metadata.get("source_path")

    def get_images(self) -> List[ImageRef]:
        """获取文档中的图片引用列表。"""
        images_data = self.metadata.get("images", [])
        return [ImageRef.from_dict(img) if isinstance(img, dict) else img for img in images_data]


@dataclass
class Chunk:
    """
    文档片段（Chunk）。

    表示文档的一个片段，由 Splitter 切分产生。
    """

    # Chunk 唯一标识符（格式：{doc_id}_{index:04d}_{hash_8chars}）
    id: str
    # Chunk 文本内容
    text: str
    # 元数据（继承自 Document.metadata + chunk_index）
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 在原文档中的起始字符偏移量
    start_offset: int = 0
    # 在原文档中的结束字符偏移量
    end_offset: int = 0
    # 指向父 Document.id，用于溯源
    source_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典创建 Chunk。"""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            start_offset=data.get("start_offset", 0),
            end_offset=data.get("end_offset", 0),
            source_ref=data.get("source_ref"),
        )

    def to_json(self) -> str:
        """转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "Chunk":
        """从 JSON 字符串创建 Chunk。"""
        return cls.from_dict(json.loads(json_str))

    def get_source_path(self) -> Optional[str]:
        """获取来源文档路径。"""
        return self.metadata.get("source_path")

    def get_chunk_index(self) -> Optional[int]:
        """获取 chunk 在文档中的序号。"""
        return self.metadata.get("chunk_index")

    def get_images(self) -> List[ImageRef]:
        """获取 chunk 中引用的图片列表。"""
        images_data = self.metadata.get("images", [])
        return [ImageRef.from_dict(img) if isinstance(img, dict) else img for img in images_data]

    def get_image_refs(self) -> List[str]:
        """获取 chunk 中引用的图片 ID 列表。"""
        return self.metadata.get("image_refs", [])


@dataclass
class ChunkRecord:
    """
    Chunk 存储记录。

    用于向量数据库存储/检索，包含向量和稀疏向量。
    """

    # Chunk 唯一标识符
    id: str
    # Chunk 文本内容
    text: str
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 稠密向量（Embedding）
    dense_vector: Optional[List[float]] = None
    # 稀疏向量（BM25 term weights）
    sparse_vector: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（不含向量，用于日志/展示）。"""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "has_dense_vector": self.dense_vector is not None,
            "has_sparse_vector": self.sparse_vector is not None,
        }

    def to_storage_dict(self) -> Dict[str, Any]:
        """转换为存储格式（含向量）。"""
        result = {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }
        if self.dense_vector is not None:
            result["dense_vector"] = self.dense_vector
        if self.sparse_vector is not None:
            result["sparse_vector"] = self.sparse_vector
        return result

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkRecord":
        """从 Chunk 创建 ChunkRecord（不含向量）。"""
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
        )


@dataclass
class RetrievalResult:
    """
    检索结果。

    表示单次检索返回的一个结果项。
    """

    # Chunk ID
    chunk_id: str
    # 相似度分数
    score: float
    # Chunk 文本内容
    text: str
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """从字典创建 RetrievalResult。"""
        return cls(
            chunk_id=data["chunk_id"],
            score=data["score"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# 工具函数
# ============================================================================


def compute_sha256(content: Union[str, bytes]) -> str:
    """
    计算内容的 SHA256 哈希值。

    Args:
        content: 文本或字节内容。

    Returns:
        str: SHA256 哈希值（十六进制字符串）。
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """
    计算文件的 SHA256 哈希值。

    Args:
        file_path: 文件路径。

    Returns:
        str: SHA256 哈希值（十六进制字符串）。
    """
    path = Path(file_path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def generate_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    """
    生成 Chunk ID。

    格式：{doc_id}_{index:04d}_{hash_8chars}

    Args:
        doc_id: 文档 ID。
        chunk_index: Chunk 在文档中的序号（从 0 开始）。
        chunk_text: Chunk 文本内容。

    Returns:
        str: Chunk ID。
    """
    text_hash = compute_sha256(chunk_text)[:8]
    return f"{doc_id}_{chunk_index:04d}_{text_hash}"


def generate_image_id(doc_hash: str, page: int, seq: int) -> str:
    """
    生成图片 ID。

    格式：{doc_hash}_{page}_{seq}

    Args:
        doc_hash: 文档哈希值。
        page: 页码。
        seq: 序号（同一页中的图片序号）。

    Returns:
        str: 图片 ID。
    """
    return f"{doc_hash}_{page}_{seq}"
