"""
Batch Processor - 批处理编排器。

将 chunks 分 batch，驱动 dense/sparse 编码，记录批次耗时。
"""

import time
from typing import List, Optional, Tuple

from src.core.types import Chunk, ChunkRecord
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
    """
    批处理编排器。

    功能：
    1. 将 chunks 分成多个 batch
    2. 驱动 dense/sparse 编码
    3. 合并编码结果
    4. 记录批次耗时（为 trace 预留）
    """

    def __init__(
        self,
        dense_encoder: Optional[DenseEncoder] = None,
        sparse_encoder: Optional[SparseEncoder] = None,
        batch_size: int = 32,
    ):
        """
        初始化 BatchProcessor。

        Args:
            dense_encoder: 稠密向量编码器（可选）。
            sparse_encoder: 稀疏向量编码器（可选）。
            batch_size: 批次大小。
        """
        self._dense_encoder = dense_encoder
        self._sparse_encoder = sparse_encoder
        self._batch_size = batch_size

    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """
        批量处理 Chunk 列表。

        Args:
            chunks: 待处理的 Chunk 列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[ChunkRecord]: 包含向量的 ChunkRecord 列表。
        """
        if not chunks:
            return []

        start_time = time.time()

        # 分批
        batches = self._split_batches(chunks)

        # 处理每个批次
        all_records = []
        for batch_index, batch_chunks in enumerate(batches):
            batch_records = self._process_batch(batch_chunks, batch_index, trace)
            all_records.extend(batch_records)

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="batch_processing",
                elapsed_ms=elapsed_ms,
                method="batch",
                details={
                    "chunk_count": len(chunks),
                    "batch_count": len(batches),
                    "batch_size": self._batch_size,
                },
            )

        return all_records

    def _split_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """
        将 chunks 分成多个批次。

        Args:
            chunks: Chunk 列表。

        Returns:
            List[List[Chunk]]: 批次列表。
        """
        batches = []
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i:i + self._batch_size]
            batches.append(batch)
        return batches

    def _process_batch(
        self,
        chunks: List[Chunk],
        batch_index: int,
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """
        处理单个批次。

        Args:
            chunks: 批次中的 Chunk 列表。
            batch_index: 批次索引。
            trace: 追踪上下文（可选）。

        Returns:
            List[ChunkRecord]: 包含向量的 ChunkRecord 列表。
        """
        # 先创建 ChunkRecord
        records = [ChunkRecord.from_chunk(chunk) for chunk in chunks]

        # Dense 编码
        if self._dense_encoder is not None:
            dense_records = self._dense_encoder.encode(chunks, trace)
            # 合并 dense_vector
            for i, record in enumerate(records):
                record.dense_vector = dense_records[i].dense_vector

        # Sparse 编码
        if self._sparse_encoder is not None:
            sparse_records = self._sparse_encoder.encode(chunks, trace)
            # 合并 sparse_vector
            for i, record in enumerate(records):
                record.sparse_vector = sparse_records[i].sparse_vector

        return records

    def get_batch_count(self, chunk_count: int) -> int:
        """
        计算批次数。

        Args:
            chunk_count: Chunk 数量。

        Returns:
            int: 批次数。
        """
        if chunk_count == 0:
            return 0
        return (chunk_count + self._batch_size - 1) // self._batch_size

    def get_batch_size(self) -> int:
        """获取批次大小。"""
        return self._batch_size


class FakeBatchProcessor:
    """
    Fake Batch Processor 用于测试。

    不依赖真实编码器，生成固定的结果。
    """

    def __init__(self, batch_size: int = 32):
        """
        初始化 FakeBatchProcessor。

        Args:
            batch_size: 批次大小。
        """
        self._batch_size = batch_size

    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """
        批量处理 Chunk 列表。

        Args:
            chunks: 待处理的 Chunk 列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[ChunkRecord]: ChunkRecord 列表。
        """
        start_time = time.time()

        records = []
        for i, chunk in enumerate(chunks):
            record = ChunkRecord.from_chunk(chunk)
            # 生成假向量
            record.dense_vector = [float(i)] * 128
            record.sparse_vector = {f"term_{i}": 1.0}
            records.append(record)

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                stage_name="batch_processing",
                elapsed_ms=elapsed_ms,
                method="fake",
                details={
                    "chunk_count": len(chunks),
                    "batch_size": self._batch_size,
                },
            )

        return records

    def _split_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """将 chunks 分成多个批次。"""
        batches = []
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i:i + self._batch_size]
            batches.append(batch)
        return batches

    def get_batch_count(self, chunk_count: int) -> int:
        """计算批次数。"""
        if chunk_count == 0:
            return 0
        return (chunk_count + self._batch_size - 1) // self._batch_size

    def get_batch_size(self) -> int:
        """获取批次大小。"""
        return self._batch_size