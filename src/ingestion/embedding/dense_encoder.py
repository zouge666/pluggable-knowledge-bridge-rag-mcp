"""
Dense Encoder - 稠密向量编码器。

把 Chunk 文本批量送入 Embedding 模型生成向量。
"""

import time
from typing import List, Optional

from src.core.settings import Settings
from src.core.types import Chunk, ChunkRecord
from src.core.trace.trace_context import TraceContext
from src.libs.embedding import EmbeddingFactory, BaseEmbedding


class DenseEncoder:
    """
    稠密向量编码器。

    功能：
    1. 批量将 Chunk 文本转换为向量
    2. 输出 ChunkRecord（包含 dense_vector）
    3. 支持追踪记录
    """

    def __init__(
        self,
        settings: Settings,
        embedding: Optional[BaseEmbedding] = None,
    ):
        """
        初始化 DenseEncoder。

        Args:
            settings: 应用配置。
            embedding: Embedding 实例（可选）。
        """
        self.settings = settings

        # Embedding 实例
        self._embedding = embedding
        if self._embedding is None:
            self._embedding = EmbeddingFactory.create(settings)

        # 向量维度
        self._dimensions = self._embedding.get_dimensions()

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """
        批量编码 Chunk 列表。

        Args:
            chunks: 待编码的 Chunk 列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[ChunkRecord]: 包含向量的 ChunkRecord 列表。
        """
        if not chunks:
            return []

        start_time = time.time()

        # 提取文本
        texts = [chunk.text for chunk in chunks]

        # 批量生成向量
        embedding_result = self._embedding.embed(texts, trace)
        vectors = embedding_result.vectors

        # 构建 ChunkRecord
        records = []
        for chunk, vector in zip(chunks, vectors):
            record = ChunkRecord.from_chunk(chunk)
            record.dense_vector = vector
            records.append(record)

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="dense_encoding",
                elapsed_ms=elapsed_ms,
                method=self._embedding.get_model_name(),
                details={
                    "chunk_count": len(chunks),
                    "dimensions": self._dimensions,
                    "model": embedding_result.model,
                },
            )

        return records

    def encode_texts(
        self,
        texts: List[str],
        trace: Optional[TraceContext] = None,
    ) -> List[List[float]]:
        """
        批量编码文本列表（便捷方法）。

        Args:
            texts: 文本列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[List[float]]: 向量列表。
        """
        if not texts:
            return []

        result = self._embedding.embed(texts, trace)
        return result.vectors

    def encode_single(
        self,
        text: str,
        trace: Optional[TraceContext] = None,
    ) -> List[float]:
        """
        编码单个文本。

        Args:
            text: 输入文本。
            trace: 追踪上下文（可选）。

        Returns:
            List[float]: 向量。
        """
        return self._embedding.embed_single(text, trace)

    def get_dimensions(self) -> int:
        """
        获取向量维度。

        Returns:
            int: 向量维度。
        """
        return self._dimensions

    def get_model_name(self) -> str:
        """
        获取模型名称。

        Returns:
            str: 模型名称。
        """
        return self._embedding.get_model_name()


class FakeDenseEncoder:
    """
    Fake Dense Encoder 用于测试。

    不依赖真实 Embedding，生成固定维度的随机向量。
    """

    def __init__(self, dimensions: int = 1536):
        """
        初始化 FakeDenseEncoder。

        Args:
            dimensions: 向量维度。
        """
        self._dimensions = dimensions

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[ChunkRecord]:
        """
        批量编码 Chunk 列表。

        Args:
            chunks: 待编码的 Chunk 列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[ChunkRecord]: 包含向量的 ChunkRecord 列表。
        """
        start_time = time.time()

        records = []
        for i, chunk in enumerate(chunks):
            record = ChunkRecord.from_chunk(chunk)
            # 生成确定性向量（基于 chunk id 的 hash）
            vector = self._generate_vector(chunk.id)
            record.dense_vector = vector
            records.append(record)

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                stage_name="dense_encoding",
                elapsed_ms=elapsed_ms,
                method="fake",
                details={
                    "chunk_count": len(chunks),
                    "dimensions": self._dimensions,
                },
            )

        return records

    def encode_texts(
        self,
        texts: List[str],
        trace: Optional[TraceContext] = None,
    ) -> List[List[float]]:
        """
        批量编码文本列表。

        Args:
            texts: 文本列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[List[float]]: 向量列表。
        """
        return [self._generate_vector(str(i)) for i in range(len(texts))]

    def encode_single(
        self,
        text: str,
        trace: Optional[TraceContext] = None,
    ) -> List[float]:
        """
        编码单个文本。

        Args:
            text: 输入文本。
            trace: 追踪上下文（可选）。

        Returns:
            List[float]: 向量。
        """
        return self._generate_vector(text)

    def _generate_vector(self, seed: str) -> List[float]:
        """
        生成确定性向量。

        Args:
            seed: 种子字符串。

        Returns:
            List[float]: 向量。
        """
        import hashlib

        # 使用 hash 作为种子
        hash_bytes = hashlib.sha256(seed.encode()).digest()

        # 生成向量
        vector = []
        for i in range(self._dimensions):
            # 使用 hash 的不同字节生成值
            byte_val = hash_bytes[i % len(hash_bytes)]
            # 归一化到 [-1, 1]
            vector.append((byte_val / 128.0) - 1.0)

        return vector

    def get_dimensions(self) -> int:
        """获取向量维度。"""
        return self._dimensions

    def get_model_name(self) -> str:
        """获取模型名称。"""
        return "fake-embedding"
