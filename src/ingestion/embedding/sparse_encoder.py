"""
Sparse Encoder - 稀疏向量编码器。

基于 BM25 统计，生成 term weights 结构。
"""

import re
import time
from collections import Counter
from typing import Dict, List, Optional

from src.core.types import Chunk, ChunkRecord
from src.core.trace.trace_context import TraceContext


# 默认停用词（英文）
DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "same", "so", "than", "too",
    "very", "just", "also", "now", "here", "there", "then", "once",
}


class SparseEncoder:
    """
    稀疏向量编码器。

    功能：
    1. 对 Chunk 文本进行分词
    2. 计算 term frequency (TF)
    3. 输出 term weights 结构（用于 BM25 索引构建）
    4. 支持追踪记录

    注意：IDF 计算在 BM25Indexer 中完成（需要全局统计）。
    """

    def __init__(
        self,
        stopwords: Optional[set] = None,
        min_term_length: int = 2,
        lowercase: bool = True,
    ):
        """
        初始化 SparseEncoder。

        Args:
            stopwords: 停用词集合（可选）。
            min_term_length: 最小词长度（过滤短词）。
            lowercase: 是否转换为小写。
        """
        self._stopwords = stopwords or DEFAULT_STOPWORDS
        self._min_term_length = min_term_length
        self._lowercase = lowercase

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
            List[ChunkRecord]: 包含 sparse_vector 的 ChunkRecord 列表。
        """
        start_time = time.time()

        records = []
        for chunk in chunks:
            record = ChunkRecord.from_chunk(chunk)
            # 计算 term weights
            term_weights = self._compute_term_weights(chunk.text)
            record.sparse_vector = term_weights
            records.append(record)

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                stage_name="sparse_encoding",
                elapsed_ms=elapsed_ms,
                method="bm25",
                details={
                    "chunk_count": len(chunks),
                    "stopwords_count": len(self._stopwords),
                },
            )

        return records

    def encode_texts(
        self,
        texts: List[str],
        trace: Optional[TraceContext] = None,
    ) -> List[Dict[str, float]]:
        """
        批量编码文本列表（便捷方法）。

        Args:
            texts: 文本列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[Dict[str, float]]: term weights 列表。
        """
        return [self._compute_term_weights(text) for text in texts]

    def encode_single(
        self,
        text: str,
        trace: Optional[TraceContext] = None,
    ) -> Dict[str, float]:
        """
        编码单个文本。

        Args:
            text: 输入文本。
            trace: 追踪上下文（可选）。

        Returns:
            Dict[str, float]: term weights。
        """
        return self._compute_term_weights(text)

    def _compute_term_weights(self, text: str) -> Dict[str, float]:
        """
        计算文本的 term weights。

        Args:
            text: 输入文本。

        Returns:
            Dict[str, float]: term -> tf (词频)。
        """
        if not text:
            return {}

        # 分词
        tokens = self._tokenize(text)

        # 过滤停用词和短词
        tokens = [
            t for t in tokens
            if t not in self._stopwords and len(t) >= self._min_term_length
        ]

        # 计算词频
        term_counts = Counter(tokens)

        # 归一化（可选：这里先输出原始 TF，BM25 计算在 indexer 中）
        # 输出结构：{term: tf}
        return {term: float(count) for term, count in term_counts.items()}

    def _tokenize(self, text: str) -> List[str]:
        """
        分词。

        Args:
            text: 输入文本。

        Returns:
            List[str]: 词列表。
        """
        if self._lowercase:
            text = text.lower()

        tokens = []

        # 英文/数字分词
        english_pattern = re.compile(r"[a-zA-Z0-9]+")
        english_tokens = english_pattern.findall(text)
        tokens.extend(english_tokens)

        # 中文分词（简单按字符分割，实际应用可用 jieba）
        # 使用正确的 Unicode 范围：一-鿿 是基本 CJK 统一汉字
        chinese_pattern = re.compile(r"[一-鿿]+")
        chinese_matches = chinese_pattern.findall(text)
        # 中文按字符分割（简单策略）
        for match in chinese_matches:
            # 如果长度较短，整体作为一个词
            if len(match) <= 4:
                tokens.append(match)
            else:
                # 较长的中文串，按字符分割
                tokens.extend(list(match))

        return tokens

    def get_stopwords(self) -> set:
        """获取停用词集合。"""
        return self._stopwords

    def get_model_name(self) -> str:
        """获取模型名称。"""
        return "bm25"


class FakeSparseEncoder:
    """
    Fake Sparse Encoder 用于测试。

    不依赖真实分词，生成固定的 term weights。
    """

    def __init__(self):
        """初始化 FakeSparseEncoder。"""
        pass

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
            List[ChunkRecord]: 包含 sparse_vector 的 ChunkRecord 列表。
        """
        start_time = time.time()

        records = []
        for i, chunk in enumerate(chunks):
            record = ChunkRecord.from_chunk(chunk)
            # 生成确定性 term weights（基于 chunk id）
            term_weights = self._generate_term_weights(chunk.id)
            record.sparse_vector = term_weights
            records.append(record)

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                stage_name="sparse_encoding",
                elapsed_ms=elapsed_ms,
                method="fake",
                details={
                    "chunk_count": len(chunks),
                },
            )

        return records

    def encode_texts(
        self,
        texts: List[str],
        trace: Optional[TraceContext] = None,
    ) -> List[Dict[str, float]]:
        """
        批量编码文本列表。

        Args:
            texts: 文本列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[Dict[str, float]]: term weights 列表。
        """
        return [self._generate_term_weights(str(i)) for i in range(len(texts))]

    def encode_single(
        self,
        text: str,
        trace: Optional[TraceContext] = None,
    ) -> Dict[str, float]:
        """
        编码单个文本。

        Args:
            text: 输入文本。
            trace: 追踪上下文（可选）。

        Returns:
            Dict[str, float]: term weights。
        """
        return self._generate_term_weights(text)

    def _generate_term_weights(self, seed: str) -> Dict[str, float]:
        """
        生成确定性 term weights。

        Args:
            seed: 种子字符串。

        Returns:
            Dict[str, float]: term weights。
        """
        import hashlib

        # 使用 hash 生成确定性 term weights
        hash_bytes = hashlib.sha256(seed.encode()).digest()

        # 生成 3-5 个 term
        term_count = 3 + (hash_bytes[0] % 3)
        weights = {}

        for i in range(term_count):
            # 生成 term 名称
            term = f"term_{hash_bytes[i % len(hash_bytes)] % 100}"
            # 生成权重 (1.0 - 5.0)
            weight = 1.0 + (hash_bytes[(i + 1) % len(hash_bytes)] / 255.0) * 4.0
            weights[term] = weight

        return weights

    def get_model_name(self) -> str:
        """获取模型名称。"""
        return "fake-sparse"