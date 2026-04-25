"""
BM25Indexer 单元测试。

测试倒排索引构建与查询功能。
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.core.trace.trace_context import TraceContext
from src.ingestion.storage.bm25_indexer import (
    BM25Indexer,
    FakeBM25Indexer,
    BM25Index,
    Posting,
    TermIndex,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_term_weights():
    """创建测试用词项权重。"""
    return {
        "chunk_001": {"machine": 1.0, "learning": 1.0, "ai": 1.0},
        "chunk_002": {"deep": 1.0, "learning": 1.0, "neural": 1.0},
        "chunk_003": {"machine": 1.0, "vision": 1.0, "ai": 1.0},
    }


@pytest.fixture
def trace_context():
    """创建 TraceContext。"""
    return TraceContext(trace_type="ingestion")


@pytest.fixture
def temp_db_path():
    """创建临时数据库路径。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "bm25_index.db")


# ============================================================================
# BM25Index 测试
# ============================================================================


class TestBM25Index:
    """BM25Index 数据结构测试。"""

    def test_empty_index(self):
        """测试空索引。"""
        index = BM25Index()

        assert index.total_docs == 0
        assert index.avg_doc_length == 0.0
        assert len(index.terms) == 0
        assert len(index.chunk_ids) == 0

    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化。"""
        index = BM25Index()
        index.total_docs = 3
        index.avg_doc_length = 3.0
        index.doc_lengths = {"c1": 3, "c2": 4}
        index.chunk_ids = {"c1", "c2"}
        index.terms = {
            "test": TermIndex(
                idf=1.5,
                postings=[
                    Posting(chunk_id="c1", tf=2.0, doc_length=3),
                ],
            )
        }

        data = index.to_dict()
        restored = BM25Index.from_dict(data)

        assert restored.total_docs == 3
        assert restored.avg_doc_length == 3.0
        assert restored.doc_lengths == {"c1": 3, "c2": 4}
        assert restored.chunk_ids == {"c1", "c2"}
        assert "test" in restored.terms
        assert restored.terms["test"].idf == 1.5


# ============================================================================
# BM25Indexer 测试
# ============================================================================


class TestBM25Indexer:
    """BM25Indexer 测试。"""

    def test_build_creates_index(self, temp_db_path, sample_term_weights):
        """测试构建索引。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        index = indexer.build(sample_term_weights)

        assert index.total_docs == 3
        assert len(index.terms) > 0
        assert len(index.chunk_ids) == 3

    def test_build_calculates_idf(self, temp_db_path, sample_term_weights):
        """测试 IDF 计算。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        index = indexer.build(sample_term_weights)

        # "learning" 出现在 2 个文档中，df=2
        # IDF = log((3 - 2 + 0.5) / (2 + 0.5) + 1) = log(1.5 / 2.5 + 1) = log(1.6)
        assert "learning" in index.terms
        expected_idf = 0.470  # log(1.6) ≈ 0.47
        assert abs(index.terms["learning"].idf - expected_idf) < 0.01

    def test_build_creates_postings(self, temp_db_path, sample_term_weights):
        """测试创建 posting 列表。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        index = indexer.build(sample_term_weights)

        # "learning" 出现在 chunk_001 和 chunk_002
        assert len(index.terms["learning"].postings) == 2

        chunk_ids = [p.chunk_id for p in index.terms["learning"].postings]
        assert "chunk_001" in chunk_ids
        assert "chunk_002" in chunk_ids

    def test_build_persists_to_db(self, temp_db_path, sample_term_weights):
        """测试持久化到数据库。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(sample_term_weights)

        # 验证数据库文件存在
        assert os.path.exists(temp_db_path)

        # 创建新的 indexer 加载索引
        new_indexer = BM25Indexer(db_path=temp_db_path)
        loaded_index = new_indexer.load()

        assert loaded_index.total_docs == 3
        assert "learning" in loaded_index.terms

    def test_load_returns_stable_index(self, temp_db_path, sample_term_weights):
        """测试加载返回稳定索引。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        index1 = indexer.build(sample_term_weights)

        new_indexer = BM25Indexer(db_path=temp_db_path)
        index2 = new_indexer.load()

        assert index1.total_docs == index2.total_docs
        assert set(index1.terms.keys()) == set(index2.terms.keys())

    def test_query_returns_results(self, temp_db_path, sample_term_weights):
        """测试查询返回结果。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(sample_term_weights)

        results = indexer.query(["machine", "learning"], top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[1], float) for r in results)  # score is float

    def test_query_returns_stable_results(self, temp_db_path, sample_term_weights):
        """测试查询返回稳定结果。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(sample_term_weights)

        results1 = indexer.query(["machine", "learning"], top_k=3)
        results2 = indexer.query(["machine", "learning"], top_k=3)

        assert results1 == results2

    def test_query_with_unknown_term(self, temp_db_path, sample_term_weights):
        """测试查询未知词项。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(sample_term_weights)

        results = indexer.query(["unknown_term_xyz"], top_k=10)

        assert results == []

    def test_query_with_empty_terms(self, temp_db_path, sample_term_weights):
        """测试查询空词项列表。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(sample_term_weights)

        results = indexer.query([], top_k=10)

        assert results == []

    def test_build_with_trace(self, temp_db_path, sample_term_weights, trace_context):
        """测试构建记录追踪。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(sample_term_weights, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "bm25_indexing" for s in stages)

    def test_build_trace_includes_details(
        self, temp_db_path, sample_term_weights, trace_context
    ):
        """测试追踪包含详细信息。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(sample_term_weights, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        stage = next((s for s in stages if s["stage"] == "bm25_indexing"), None)

        assert stage is not None
        assert "chunk_count" in stage.get("details", {})
        assert "term_count" in stage.get("details", {})

    def test_get_stats(self, temp_db_path, sample_term_weights):
        """测试获取统计信息。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(sample_term_weights)

        stats = indexer.get_stats()

        assert stats["loaded"] is True
        assert stats["total_docs"] == 3
        assert stats["total_terms"] > 0

    def test_get_stats_before_build(self, temp_db_path):
        """测试构建前获取统计信息。"""
        indexer = BM25Indexer(db_path=temp_db_path)

        stats = indexer.get_stats()

        assert stats["loaded"] is False

    def test_clear(self, temp_db_path, sample_term_weights):
        """测试清空索引。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(sample_term_weights)

        indexer.clear()

        stats = indexer.get_stats()
        assert stats["loaded"] is False

    def test_build_empty_term_weights(self, temp_db_path):
        """测试构建空词项权重。"""
        indexer = BM25Indexer(db_path=temp_db_path)
        index = indexer.build({})

        assert index.total_docs == 0
        assert len(index.terms) == 0


# ============================================================================
# FakeBM25Indexer 测试
# ============================================================================


class TestFakeBM25Indexer:
    """FakeBM25Indexer 测试。"""

    def test_build_creates_index(self, sample_term_weights):
        """测试构建索引。"""
        indexer = FakeBM25Indexer()
        index = indexer.build(sample_term_weights)

        assert index.total_docs == 3
        assert len(index.terms) > 0

    def test_query_returns_results(self, sample_term_weights):
        """测试查询返回结果。"""
        indexer = FakeBM25Indexer()
        indexer.build(sample_term_weights)

        results = indexer.query(["machine", "learning"], top_k=2)

        assert len(results) <= 2

    def test_query_returns_stable_results(self, sample_term_weights):
        """测试查询返回稳定结果。"""
        indexer = FakeBM25Indexer()
        indexer.build(sample_term_weights)

        results1 = indexer.query(["machine", "learning"], top_k=3)
        results2 = indexer.query(["machine", "learning"], top_k=3)

        assert results1 == results2

    def test_load_returns_index(self, sample_term_weights):
        """测试加载返回索引。"""
        indexer = FakeBM25Indexer()
        indexer.build(sample_term_weights)

        loaded = indexer.load()

        assert loaded.total_docs == 3

    def test_get_stats(self, sample_term_weights):
        """测试获取统计信息。"""
        indexer = FakeBM25Indexer()
        indexer.build(sample_term_weights)

        stats = indexer.get_stats()

        assert stats["loaded"] is True
        assert stats["total_docs"] == 3

    def test_build_with_trace(self, sample_term_weights, trace_context):
        """测试构建记录追踪。"""
        indexer = FakeBM25Indexer()
        indexer.build(sample_term_weights, trace_context)

        stages = trace_context.to_dict().get("stages", [])
        assert any(s["stage"] == "bm25_indexing" for s in stages)


# ============================================================================
# Roundtrip 测试
# ============================================================================


class TestRoundtrip:
    """Roundtrip 测试（构建 → 持久化 → 加载 → 查询）。"""

    def test_build_load_query_roundtrip(self, temp_db_path):
        """测试完整的 roundtrip。"""
        # 准备数据
        term_weights = {
            "doc1": {"apple": 2.0, "banana": 1.0, "fruit": 1.0},
            "doc2": {"apple": 1.0, "orange": 2.0, "fruit": 1.0},
            "doc3": {"banana": 2.0, "orange": 1.0, "fruit": 1.0},
        }

        # 构建索引
        indexer1 = BM25Indexer(db_path=temp_db_path)
        indexer1.build(term_weights)

        # 加载索引
        indexer2 = BM25Indexer(db_path=temp_db_path)
        indexer2.load()

        # 查询
        results = indexer2.query(["apple", "fruit"], top_k=3)

        # 验证结果稳定
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_rebuild_updates_index(self, temp_db_path):
        """测试重建更新索引。"""
        # 第一次构建
        term_weights1 = {
            "doc1": {"apple": 1.0},
        }
        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(term_weights1)

        stats1 = indexer.get_stats()
        assert stats1["total_docs"] == 1

        # 第二次构建（重建）
        term_weights2 = {
            "doc1": {"apple": 1.0},
            "doc2": {"banana": 1.0},
        }
        indexer.build(term_weights2)

        stats2 = indexer.get_stats()
        assert stats2["total_docs"] == 2


# ============================================================================
# 边界条件测试
# ============================================================================


class TestEdgeCases:
    """边界条件测试。"""

    def test_single_document(self, temp_db_path):
        """测试单个文档。"""
        term_weights = {
            "doc1": {"test": 1.0, "document": 1.0},
        }

        indexer = BM25Indexer(db_path=temp_db_path)
        index = indexer.build(term_weights)

        assert index.total_docs == 1
        assert len(index.terms) == 2

    def test_single_term(self, temp_db_path):
        """测试单个词项。"""
        term_weights = {
            "doc1": {"test": 1.0},
            "doc2": {"test": 2.0},
        }

        indexer = BM25Indexer(db_path=temp_db_path)
        index = indexer.build(term_weights)

        assert len(index.terms) == 1
        assert "test" in index.terms

    def test_high_term_frequency(self, temp_db_path):
        """测试高词频。"""
        term_weights = {
            "doc1": {"test": 100.0},
        }

        indexer = BM25Indexer(db_path=temp_db_path)
        index = indexer.build(term_weights)

        results = indexer.query(["test"], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_many_documents(self, temp_db_path):
        """测试大量文档。"""
        term_weights = {
            f"doc_{i}": {f"term_{i % 10}": 1.0, "common": 1.0}
            for i in range(100)
        }

        indexer = BM25Indexer(db_path=temp_db_path)
        index = indexer.build(term_weights)

        assert index.total_docs == 100
        assert "common" in index.terms

    def test_query_top_k_larger_than_results(self, temp_db_path):
        """测试 top_k 大于结果数量。"""
        term_weights = {
            "doc1": {"test": 1.0},
        }

        indexer = BM25Indexer(db_path=temp_db_path)
        indexer.build(term_weights)

        results = indexer.query(["test"], top_k=100)
        assert len(results) == 1
