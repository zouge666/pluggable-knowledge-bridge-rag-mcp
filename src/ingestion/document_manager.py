"""
Document Manager - 跨存储的文档生命周期管理。

协调 ChromaStore、BM25Indexer、ImageStorage、FileIntegrityChecker
实现文档的 list/delete/stats 操作。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.libs.vector_store.chroma_store import ChromaStore
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import SQLiteIntegrityChecker


@dataclass
class DocumentInfo:
    """文档信息。"""

    source_path: str
    file_hash: str
    collection: Optional[str]
    chunk_count: int
    image_count: int
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChunkDetail:
    """Chunk 详情。"""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


@dataclass
class DocumentDetail:
    """文档详情。"""

    source_path: str
    file_hash: str
    collection: Optional[str]
    chunks: List[ChunkDetail]
    images: List[str]  # image_id 列表
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class DeleteResult:
    """删除结果。"""

    success: bool
    source_path: str
    chunks_deleted: int
    images_deleted: int
    history_removed: bool
    error: Optional[str] = None


@dataclass
class CollectionStats:
    """集合统计信息。"""

    collection_name: str
    document_count: int
    chunk_count: int
    image_count: int


class DocumentManager:
    """
    文档管理器。

    协调四个存储模块实现文档生命周期管理：
    1. ChromaStore：向量存储
    2. BM25Indexer：倒排索引
    3. ImageStorage：图片存储
    4. FileIntegrityChecker：摄入历史
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_indexer: BM25Indexer,
        image_storage: ImageStorage,
        file_integrity: SQLiteIntegrityChecker,
    ):
        """
        初始化 DocumentManager。

        Args:
            chroma_store: 向量存储。
            bm25_indexer: BM25 索引器。
            image_storage: 图片存储。
            file_integrity: 文件完整性检查器。
        """
        self._chroma = chroma_store
        self._bm25 = bm25_indexer
        self._images = image_storage
        self._integrity = file_integrity

    def list_documents(
        self,
        collection: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[DocumentInfo]:
        """
        列出已摄入的文档。

        Args:
            collection: 集合名称（可选，None 表示所有集合）。
            status: 状态过滤（可选，"success" 或 "failed"）。

        Returns:
            List[DocumentInfo]: 文档信息列表。
        """
        # 从摄入历史获取记录
        records = self._integrity.list_processed(collection=collection, status=status)

        documents = []
        for record in records:
            source_path = record["file_path"]
            file_hash = record["file_hash"]
            doc_collection = record.get("collection")

            # 统计 chunk 数量
            chunk_count = self._count_chunks_by_source(source_path, doc_collection)

            # 统计图片数量
            image_count = self._images.count(collection=doc_collection) if doc_collection else 0

            documents.append(DocumentInfo(
                source_path=source_path,
                file_hash=file_hash,
                collection=doc_collection,
                chunk_count=chunk_count,
                image_count=image_count,
                status=record["status"],
                created_at=record.get("created_at"),
                updated_at=record.get("updated_at"),
                metadata=record.get("metadata"),
            ))

        return documents

    def _count_chunks_by_source(self, source_path: str, collection: Optional[str]) -> int:
        """
        统计指定源文件的 chunk 数量。

        Args:
            source_path: 源文件路径。
            collection: 集合名称。

        Returns:
            int: chunk 数量。
        """
        try:
            # Primary: 尝试使用完整的 source_path (+ collection if provided)
            if collection:
                filters = {"$and": [{"source_path": source_path}, {"collection": collection}]}
            else:
                filters = {"source_path": source_path}

            results = self._chroma.query_by_metadata(filters)

            # Backward compatibility: 如果没有结果，先尝试去掉 collection 限制
            if not results and collection:
                results = self._chroma.query_by_metadata({"source_path": source_path})

            # Fallback 1: 尝试按文件名匹配（有些存储只记录 file_name）
            if not results:
                file_name = Path(source_path).name
                results = self._chroma.query_by_metadata({"file_name": file_name})

            # Fallback 2: 尝试常见替代字段 'source'
            if not results:
                results = self._chroma.query_by_metadata({"source": source_path})

            return len(results) if results else 0
        except Exception:
            return 0

    def get_document_detail(self, source_path: str) -> Optional[DocumentDetail]:
        """
        获取文档详情。

        Args:
            source_path: 源文件路径。

        Returns:
            Optional[DocumentDetail]: 文档详情，不存在返回 None。
        """
        # 从摄入历史查找记录
        records = self._integrity.list_processed()
        record = None
        for r in records:
            if r["file_path"] == source_path:
                record = r
                break

        if record is None:
            return None

        file_hash = record["file_hash"]
        collection = record.get("collection")

        # 获取 chunks
        chunks = self._get_chunks_by_source(source_path, collection)

        # 获取图片
        images = []
        if collection:
            image_infos = self._images.list_by_doc_hash(file_hash)
            images = [info.image_id for info in image_infos]

        return DocumentDetail(
            source_path=source_path,
            file_hash=file_hash,
            collection=collection,
            chunks=chunks,
            images=images,
            status=record["status"],
            created_at=record.get("created_at"),
            updated_at=record.get("updated_at"),
        )

    def _get_chunks_by_source(
        self,
        source_path: str,
        collection: Optional[str],
    ) -> List[ChunkDetail]:
        """
        获取指定源文件的所有 chunks。

        Args:
            source_path: 源文件路径。
            collection: 集合名称。

        Returns:
            List[ChunkDetail]: chunk 详情列表。
        """
        try:
            # Primary: 使用完整的 source_path (+ collection if provided)
            if collection:
                filters = {"$and": [{"source_path": source_path}, {"collection": collection}]}
            else:
                filters = {"source_path": source_path}

            results = self._chroma.get_by_metadata(filters)

            # Backward compatibility: 去掉 collection 限制再试一次
            if not results and collection:
                results = self._chroma.get_by_metadata({"source_path": source_path})

            # Fallback 1: 按 file_name 查找
            if not results:
                file_name = Path(source_path).name
                results = self._chroma.get_by_metadata({"file_name": file_name})

            # Fallback 2: 常见替代字段 'source'
            if not results:
                results = self._chroma.get_by_metadata({"source": source_path})

            if not results:
                return []

            return [
                ChunkDetail(
                    chunk_id=r["id"],
                    text=r.get("text", ""),
                    metadata=r.get("metadata", {}),
                )
                for r in results
            ]
        except Exception:
            return []

    def delete_document(
        self,
        source_path: str,
        collection: Optional[str] = None,
    ) -> DeleteResult:
        """
        删除文档。

        协调删除 Chroma + BM25 + ImageStorage + FileIntegrity 四个存储。

        Args:
            source_path: 源文件路径。
            collection: 集合名称（可选）。

        Returns:
            DeleteResult: 删除结果。
        """
        try:
            # 1. 获取文档详情
            detail = self.get_document_detail(source_path)
            if detail is None:
                return DeleteResult(
                    success=False,
                    source_path=source_path,
                    chunks_deleted=0,
                    images_deleted=0,
                    history_removed=False,
                    error="Document not found",
                )

            file_hash = detail.file_hash
            chunk_ids = [c.chunk_id for c in detail.chunks]
            image_ids = detail.images

            # 2. 从 Chroma 删除 chunks
            chunks_deleted = 0
            if chunk_ids:
                chunks_deleted = self._chroma.delete_by_ids(chunk_ids)

            # 3. 从 BM25 索引删除
            if chunk_ids:
                self._bm25.remove_documents(chunk_ids)

            # 4. 删除图片
            images_deleted = 0
            for image_id in image_ids:
                if self._images.delete(image_id):
                    images_deleted += 1

            # 5. 删除摄入历史记录
            history_removed = self._integrity.remove_record(file_hash)

            return DeleteResult(
                success=True,
                source_path=source_path,
                chunks_deleted=chunks_deleted,
                images_deleted=images_deleted,
                history_removed=history_removed,
            )

        except Exception as e:
            return DeleteResult(
                success=False,
                source_path=source_path,
                chunks_deleted=0,
                images_deleted=0,
                history_removed=False,
                error=str(e),
            )

    def get_collection_stats(
        self,
        collection: Optional[str] = None,
    ) -> CollectionStats:
        """
        获取集合统计信息。

        Args:
            collection: 集合名称（可选，None 表示默认集合）。

        Returns:
            CollectionStats: 集合统计信息。
        """
        collection_name = collection or "default"

        # 获取 chunk 数量
        chroma_stats = self._chroma.get_collection_stats()
        chunk_count = chroma_stats.get("count", 0)

        # 获取文档数量
        records = self._integrity.list_processed(collection=collection, status="success")
        document_count = len(records)

        # 获取图片数量
        image_count = self._images.count(collection=collection)

        return CollectionStats(
            collection_name=collection_name,
            document_count=document_count,
            chunk_count=chunk_count,
            image_count=image_count,
        )

    def delete_by_collection(self, collection: str) -> Dict[str, Any]:
        """
        删除集合中的所有文档。

        Args:
            collection: 集合名称。

        Returns:
            Dict[str, Any]: 删除统计信息。
        """
        # 获取集合中的所有文档
        documents = self.list_documents(collection=collection, status="success")

        chunks_deleted = 0
        images_deleted = 0
        docs_deleted = 0

        for doc in documents:
            result = self.delete_document(doc.source_path, collection)
            if result.success:
                chunks_deleted += result.chunks_deleted
                images_deleted += result.images_deleted
                docs_deleted += 1

        return {
            "collection": collection,
            "documents_deleted": docs_deleted,
            "chunks_deleted": chunks_deleted,
            "images_deleted": images_deleted,
        }
