"""
BM25 Indexer - 倒排索引构建与持久化。

接收 SparseEncoder 的 term statistics 输出，计算 IDF，构建倒排索引。
"""

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.core.trace.trace_context import TraceContext


@dataclass
class Posting:
    """倒排索引项。"""

    chunk_id: str
    tf: float  # term frequency
    doc_length: int  # document length (in terms)


@dataclass
class TermIndex:
    """词项索引。"""

    idf: float  # inverse document frequency
    postings: List[Posting]  # posting list


@dataclass
class BM25Index:
    """BM25 索引结构。"""

    # 词项到索引的映射
    terms: Dict[str, TermIndex] = field(default_factory=dict)
    # 文档总数
    total_docs: int = 0
    # 平均文档长度
    avg_doc_length: float = 0.0
    # 文档长度映射 {chunk_id: doc_length}
    doc_lengths: Dict[str, int] = field(default_factory=dict)
    # 所有 chunk_id 集合
    chunk_ids: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "terms": {
                term: {
                    "idf": term_index.idf,
                    "postings": [asdict(p) for p in term_index.postings],
                }
                for term, term_index in self.terms.items()
            },
            "total_docs": self.total_docs,
            "avg_doc_length": self.avg_doc_length,
            "doc_lengths": self.doc_lengths,
            "chunk_ids": list(self.chunk_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        """从字典创建索引。"""
        index = cls()
        for term, term_data in data.get("terms", {}).items():
            postings = [
                Posting(
                    chunk_id=p["chunk_id"],
                    tf=p["tf"],
                    doc_length=p["doc_length"],
                )
                for p in term_data.get("postings", [])
            ]
            index.terms[term] = TermIndex(
                idf=term_data["idf"],
                postings=postings,
            )
        index.total_docs = data.get("total_docs", 0)
        index.avg_doc_length = data.get("avg_doc_length", 0.0)
        index.doc_lengths = data.get("doc_lengths", {})
        index.chunk_ids = set(data.get("chunk_ids", []))
        return index


class BM25Indexer:
    """
    BM25 索引器。

    功能：
    1. 接收 SparseEncoder 的 term weights 输出
    2. 计算 IDF
    3. 构建倒排索引
    4. 持久化到 SQLite 数据库
    5. 支持查询
    """

    # BM25 参数
    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75

    def __init__(
        self,
        db_path: str = "data/db/bm25/bm25_index.db",
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ):
        """
        初始化 BM25Indexer。

        Args:
            db_path: 数据库路径。
            k1: BM25 k1 参数（词频饱和参数）。
            b: BM25 b 参数（文档长度归一化参数）。
        """
        self._db_path = Path(db_path)
        self._k1 = k1
        self._b = b
        self._index: Optional[BM25Index] = None

        # 确保目录存在
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self._init_db()

    def _init_db(self):
        """初始化数据库表。"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # 创建元数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # 创建倒排索引表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inverted_index (
                term TEXT PRIMARY KEY,
                idf REAL,
                postings TEXT
            )
        """)

        # 创建文档长度表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doc_lengths (
                chunk_id TEXT PRIMARY KEY,
                doc_length INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def build(
        self,
        chunk_term_weights: Dict[str, Dict[str, float]],
        trace: Optional[TraceContext] = None,
    ) -> BM25Index:
        """
        构建 BM25 索引。

        Args:
            chunk_term_weights: {chunk_id: {term: weight}} 格式的词项权重。
            trace: 追踪上下文（可选）。

        Returns:
            BM25Index: 构建的索引。
        """
        start_time = time.time()

        # 创建新索引
        index = BM25Index()

        # 统计文档信息
        total_doc_length = 0
        doc_count = len(chunk_term_weights)

        # 第一遍：收集文档长度和词项文档频率
        term_doc_freq: Dict[str, int] = {}  # term -> document frequency

        for chunk_id, term_weights in chunk_term_weights.items():
            doc_length = len(term_weights)
            index.doc_lengths[chunk_id] = doc_length
            index.chunk_ids.add(chunk_id)
            total_doc_length += doc_length

            # 统计词项文档频率
            for term in term_weights.keys():
                term_doc_freq[term] = term_doc_freq.get(term, 0) + 1

        # 计算平均文档长度
        index.avg_doc_length = total_doc_length / doc_count if doc_count > 0 else 0.0
        index.total_docs = doc_count

        # 第二遍：计算 IDF 并构建倒排索引
        for chunk_id, term_weights in chunk_term_weights.items():
            doc_length = index.doc_lengths[chunk_id]

            for term, tf in term_weights.items():
                # 计算 IDF
                df = term_doc_freq[term]
                idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

                # 添加到倒排索引
                if term not in index.terms:
                    index.terms[term] = TermIndex(idf=idf, postings=[])

                posting = Posting(
                    chunk_id=chunk_id,
                    tf=tf,
                    doc_length=doc_length,
                )
                index.terms[term].postings.append(posting)

        # 持久化到数据库
        self._save_index(index)

        self._index = index

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                stage_name="bm25_indexing",
                elapsed_ms=elapsed_ms,
                method="bm25",
                details={
                    "chunk_count": doc_count,
                    "term_count": len(index.terms),
                    "avg_doc_length": index.avg_doc_length,
                },
            )

        return index

    def _save_index(self, index: BM25Index):
        """
        保存索引到数据库。

        Args:
            index: BM25 索引。
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # 清空现有数据
        cursor.execute("DELETE FROM inverted_index")
        cursor.execute("DELETE FROM doc_lengths")
        cursor.execute("DELETE FROM metadata")

        # 保存元数据
        metadata = {
            "total_docs": str(index.total_docs),
            "avg_doc_length": str(index.avg_doc_length),
        }
        for key, value in metadata.items():
            cursor.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )

        # 保存倒排索引
        for term, term_index in index.terms.items():
            postings_json = json.dumps([asdict(p) for p in term_index.postings])
            cursor.execute(
                "INSERT OR REPLACE INTO inverted_index (term, idf, postings) VALUES (?, ?, ?)",
                (term, term_index.idf, postings_json),
            )

        # 保存文档长度
        for chunk_id, doc_length in index.doc_lengths.items():
            cursor.execute(
                "INSERT OR REPLACE INTO doc_lengths (chunk_id, doc_length) VALUES (?, ?)",
                (chunk_id, doc_length),
            )

        conn.commit()
        conn.close()

    def load(self) -> BM25Index:
        """
        从数据库加载索引。

        Returns:
            BM25Index: 加载的索引。
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        index = BM25Index()

        # 加载元数据
        cursor.execute("SELECT key, value FROM metadata")
        for key, value in cursor.fetchall():
            if key == "total_docs":
                index.total_docs = int(value)
            elif key == "avg_doc_length":
                index.avg_doc_length = float(value)

        # 加载倒排索引
        cursor.execute("SELECT term, idf, postings FROM inverted_index")
        for term, idf, postings_json in cursor.fetchall():
            postings_data = json.loads(postings_json)
            postings = [
                Posting(
                    chunk_id=p["chunk_id"],
                    tf=p["tf"],
                    doc_length=p["doc_length"],
                )
                for p in postings_data
            ]
            index.terms[term] = TermIndex(idf=idf, postings=postings)

        # 加载文档长度
        cursor.execute("SELECT chunk_id, doc_length FROM doc_lengths")
        for chunk_id, doc_length in cursor.fetchall():
            index.doc_lengths[chunk_id] = doc_length
            index.chunk_ids.add(chunk_id)

        conn.close()

        self._index = index
        return index

    def query(
        self,
        terms: List[str],
        top_k: int = 10,
    ) -> List[tuple]:
        """
        查询 BM25 索引。

        Args:
            terms: 查询词项列表。
            top_k: 返回的 top-k 结果数量。

        Returns:
            List[tuple]: [(chunk_id, score), ...] 格式的结果列表。
        """
        if self._index is None:
            self.load()

        index = self._index

        # 计算每个文档的 BM25 分数
        scores: Dict[str, float] = {}

        for term in terms:
            if term not in index.terms:
                continue

            term_index = index.terms[term]
            idf = term_index.idf

            for posting in term_index.postings:
                chunk_id = posting.chunk_id
                tf = posting.tf
                doc_length = posting.doc_length

                # BM25 分数计算
                # score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * doc_length / index.avg_doc_length
                )
                score = idf * numerator / denominator

                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        # 排序并返回 top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def get_index(self) -> Optional[BM25Index]:
        """获取当前索引。"""
        return self._index

    def get_stats(self) -> dict:
        """
        获取索引统计信息。

        Returns:
            dict: 统计信息。
        """
        if self._index is None:
            return {
                "loaded": False,
                "total_docs": 0,
                "total_terms": 0,
                "avg_doc_length": 0.0,
            }

        return {
            "loaded": True,
            "total_docs": self._index.total_docs,
            "total_terms": len(self._index.terms),
            "avg_doc_length": self._index.avg_doc_length,
        }

    def clear(self):
        """清空索引。"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM inverted_index")
        cursor.execute("DELETE FROM doc_lengths")
        cursor.execute("DELETE FROM metadata")
        conn.commit()
        conn.close()

        self._index = None


class FakeBM25Indexer:
    """
    Fake BM25 Indexer 用于测试。

    不依赖真实数据库，使用内存存储。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """初始化 FakeBM25Indexer。"""
        self._k1 = k1
        self._b = b
        self._index: Optional[BM25Index] = None

    def build(
        self,
        chunk_term_weights: Dict[str, Dict[str, float]],
        trace: Optional[TraceContext] = None,
    ) -> BM25Index:
        """构建 BM25 索引。"""
        start_time = time.time()

        index = BM25Index()

        total_doc_length = 0
        doc_count = len(chunk_term_weights)
        term_doc_freq: Dict[str, int] = {}

        for chunk_id, term_weights in chunk_term_weights.items():
            doc_length = len(term_weights)
            index.doc_lengths[chunk_id] = doc_length
            index.chunk_ids.add(chunk_id)
            total_doc_length += doc_length

            for term in term_weights.keys():
                term_doc_freq[term] = term_doc_freq.get(term, 0) + 1

        index.avg_doc_length = total_doc_length / doc_count if doc_count > 0 else 0.0
        index.total_docs = doc_count

        for chunk_id, term_weights in chunk_term_weights.items():
            doc_length = index.doc_lengths[chunk_id]

            for term, tf in term_weights.items():
                df = term_doc_freq[term]
                idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

                if term not in index.terms:
                    index.terms[term] = TermIndex(idf=idf, postings=[])

                posting = Posting(
                    chunk_id=chunk_id,
                    tf=tf,
                    doc_length=doc_length,
                )
                index.terms[term].postings.append(posting)

        self._index = index

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                stage_name="bm25_indexing",
                elapsed_ms=elapsed_ms,
                method="fake",
                details={
                    "chunk_count": doc_count,
                    "term_count": len(index.terms),
                },
            )

        return index

    def load(self) -> BM25Index:
        """加载索引（Fake 实现直接返回内存中的索引）。"""
        if self._index is None:
            return BM25Index()
        return self._index

    def query(
        self,
        terms: List[str],
        top_k: int = 10,
    ) -> List[tuple]:
        """查询 BM25 索引。"""
        if self._index is None:
            return []

        index = self._index
        scores: Dict[str, float] = {}

        for term in terms:
            if term not in index.terms:
                continue

            term_index = index.terms[term]
            idf = term_index.idf

            for posting in term_index.postings:
                chunk_id = posting.chunk_id
                tf = posting.tf
                doc_length = posting.doc_length

                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * doc_length / index.avg_doc_length
                )
                score = idf * numerator / denominator

                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def get_index(self) -> Optional[BM25Index]:
        """获取当前索引。"""
        return self._index

    def get_stats(self) -> dict:
        """获取索引统计信息。"""
        if self._index is None:
            return {
                "loaded": False,
                "total_docs": 0,
                "total_terms": 0,
                "avg_doc_length": 0.0,
            }

        return {
            "loaded": True,
            "total_docs": self._index.total_docs,
            "total_terms": len(self._index.terms),
            "avg_doc_length": self._index.avg_doc_length,
        }
