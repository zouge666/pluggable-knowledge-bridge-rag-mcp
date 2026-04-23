"""
VectorStore 抽象基类。

定义统一的向量存储接口，支持多种后端（Chroma/Qdrant/Pinecone）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.trace.trace_context import TraceContext


@dataclass
class VectorRecord:
    """向量记录。"""

    id: str  # 记录唯一标识
    vector: List[float]  # 向量
    text: str  # 原文本
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class QueryResult:
    """查询结果。"""

    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class UpsertResult:
    """Upsert 结果。"""

    success: bool
    upserted_count: int
    ids: List[str] = field(default_factory=list)
    message: Optional[str] = None


class BaseVectorStore(ABC):
    """
    VectorStore 抽象基类。

    所有向量存储实现（Chroma/Qdrant/Pinecone）都必须实现此接口。
    """

    @abstractmethod
    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional[TraceContext] = None,
    ) -> UpsertResult:
        """
        批量插入或更新向量记录。

        Args:
            records: 向量记录列表。
            trace: 追踪上下文（可选）。

        Returns:
            UpsertResult: Upsert 结果。

        Raises:
            VectorStoreError: 操作失败时抛出。
        """
        pass

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> List[QueryResult]:
        """
        向量相似度查询。

        Args:
            vector: 查询向量。
            top_k: 返回结果数量。
            filters: 元数据过滤条件（可选）。
            trace: 追踪上下文（可选）。

        Returns:
            List[QueryResult]: 查询结果列表。

        Raises:
            VectorStoreError: 查询失败时抛出。
        """
        pass

    @abstractmethod
    def get_by_ids(
        self,
        ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        根据 ID 批量获取记录。

        Args:
            ids: ID 列表。

        Returns:
            List[Dict]: 记录列表（包含 id, text, metadata）。
        """
        pass

    @abstractmethod
    def delete_by_ids(
        self,
        ids: List[str],
    ) -> int:
        """
        根据 ID 批量删除记录。

        Args:
            ids: ID 列表。

        Returns:
            int: 删除的记录数量。
        """
        pass

    @abstractmethod
    def delete_by_metadata(
        self,
        filters: Dict[str, Any],
    ) -> int:
        """
        根据元数据条件批量删除记录。

        Args:
            filters: 元数据过滤条件。

        Returns:
            int: 删除的记录数量。
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息。

        Returns:
            Dict: 统计信息（记录数、维度等）。
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """获取后端名称。"""
        pass


class VectorStoreError(Exception):
    """VectorStore 错误基类。"""

    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.backend = backend
        self.original_error = original_error

        error_parts = []
        if backend:
            error_parts.append(f"[{backend}]")
        error_parts.append(message)

        super().__init__(" ".join(error_parts))


class VectorStoreConnectionError(VectorStoreError):
    """VectorStore 连接错误。"""
    pass


class VectorStoreConfigError(VectorStoreError):
    """VectorStore 配置错误。"""
    pass


class VectorStoreQueryError(VectorStoreError):
    """VectorStore 查询错误。"""
    pass


class VectorStoreUpsertError(VectorStoreError):
    """VectorStore Upsert 错误。"""
    pass