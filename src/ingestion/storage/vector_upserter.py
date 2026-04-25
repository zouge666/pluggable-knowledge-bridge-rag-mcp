"""
Vector Upserter - 向量存储与幂等性保证。

接收 DenseEncoder 的向量输出，生成稳定的 chunk_id，调用 VectorStore 进行幂等写入。
"""

import hashlib
import time
from typing import Dict, List, Optional

from src.core.types import ChunkRecord
from src.core.trace.trace_context import TraceContext
from src.libs.vector_store import BaseVectorStore, VectorRecord, UpsertResult


class VectorUpserter:
    """
    向量存储器。

    功能：
    1. 接收 DenseEncoder 的向量输出
    2. 生成稳定的 chunk_id
    3. 调用 VectorStore 进行幂等写入
    4. 支持追踪记录
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        collection: Optional[str] = None,
    ):
        """
        初始化 VectorUpserter。

        Args:
            vector_store: 向量存储实例。
            collection: 集合名称（可选）。
        """
        self._vector_store = vector_store
        self._collection = collection

    def upsert(
        self,
        records: List[ChunkRecord],
        trace: Optional[TraceContext] = None,
    ) -> UpsertResult:
        """
        批量写入向量记录。

        Args:
            records: 包含向量的 ChunkRecord 列表。
            trace: 追踪上下文（可选）。

        Returns:
            UpsertResult: 写入结果。
        """
        if not records:
            return UpsertResult(success=True, upserted_count=0, ids=[])

        start_time = time.time()

        # 转换为 VectorRecord
        vector_records = []
        for record in records:
            if record.dense_vector is None:
                continue

            vector_record = VectorRecord(
                id=record.id,
                vector=record.dense_vector,
                text=record.text,
                metadata=self._enrich_metadata(record),
            )
            vector_records.append(vector_record)

        # 调用 VectorStore.upsert
        result = self._vector_store.upsert(vector_records, trace)

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                stage_name="vector_upsert",
                elapsed_ms=elapsed_ms,
                method="vector_store",
                details={
                    "record_count": len(records),
                    "upserted_count": result.upserted_count,
                },
            )

        return result

    def upsert_single(
        self,
        record: ChunkRecord,
        trace: Optional[TraceContext] = None,
    ) -> bool:
        """
        写入单个向量记录。

        Args:
            record: 包含向量的 ChunkRecord。
            trace: 追踪上下文（可选）。

        Returns:
            bool: 是否成功。
        """
        if record.dense_vector is None:
            return False

        result = self.upsert([record], trace)
        return result.success and result.upserted_count == 1

    def _enrich_metadata(self, record: ChunkRecord) -> Dict:
        """
        丰富元数据。

        Args:
            record: ChunkRecord。

        Returns:
            Dict: 丰富后的元数据。
        """
        metadata = record.metadata.copy()

        # 添加集合信息
        if self._collection:
            metadata["collection"] = self._collection

        return metadata

    def get_vector_store(self) -> BaseVectorStore:
        """获取向量存储实例。"""
        return self._vector_store

    def get_collection(self) -> Optional[str]:
        """获取集合名称。"""
        return self._collection


class FakeVectorUpserter:
    """
    Fake Vector Upserter 用于测试。

    不依赖真实 VectorStore，记录写入操作。
    """

    def __init__(self, collection: Optional[str] = None):
        """初始化 FakeVectorUpserter。"""
        self._collection = collection
        self._upserted_records: List[ChunkRecord] = []

    def upsert(
        self,
        records: List[ChunkRecord],
        trace: Optional[TraceContext] = None,
    ) -> UpsertResult:
        """批量写入向量记录。"""
        start_time = time.time()

        upserted_count = 0
        ids = []

        for record in records:
            if record.dense_vector is None:
                continue

            # 检查是否已存在（幂等性）
            existing = next(
                (r for r in self._upserted_records if r.id == record.id), None
            )
            if existing is None:
                self._upserted_records.append(record)
            else:
                # 更新现有记录
                idx = self._upserted_records.index(existing)
                self._upserted_records[idx] = record

            ids.append(record.id)
            upserted_count += 1

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                stage_name="vector_upsert",
                elapsed_ms=elapsed_ms,
                method="fake",
                details={
                    "record_count": len(records),
                    "upserted_count": upserted_count,
                },
            )

        return UpsertResult(
            success=True,
            upserted_count=upserted_count,
            ids=ids,
        )

    def upsert_single(
        self,
        record: ChunkRecord,
        trace: Optional[TraceContext] = None,
    ) -> bool:
        """写入单个向量记录。"""
        if record.dense_vector is None:
            return False

        result = self.upsert([record], trace)
        return result.success and result.upserted_count == 1

    def get_upserted_records(self) -> List[ChunkRecord]:
        """获取已写入的记录。"""
        return self._upserted_records

    def get_upserted_count(self) -> int:
        """获取已写入的记录数量。"""
        return len(self._upserted_records)

    def clear(self):
        """清空已写入的记录。"""
        self._upserted_records = []

    def get_collection(self) -> Optional[str]:
        """获取集合名称。"""
        return self._collection
