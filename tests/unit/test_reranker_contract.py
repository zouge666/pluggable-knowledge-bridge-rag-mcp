"""
Reranker 契约测试。

验证所有 Reranker 实现遵循 BaseReranker 接口契约。
"""

import pytest
from typing import List, Optional

from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    RerankerError,
    RerankerConfigError,
    RerankerUnavailableError,
)
from src.libs.reranker.none_reranker import NoneReranker


class TestRerankCandidate:
    """RerankCandidate 数据类测试。"""

    def test_candidate_creation(self):
        """应该能创建候选项。"""
        candidate = RerankCandidate(
            id="doc_1",
            text="Sample text",
            score=0.95,
            metadata={"source": "test.pdf"},
        )

        assert candidate.id == "doc_1"
        assert candidate.text == "Sample text"
        assert candidate.score == 0.95
        assert candidate.metadata["source"] == "test.pdf"

    def test_candidate_to_dict(self):
        """应该能转换为字典。"""
        candidate = RerankCandidate(
            id="doc_1",
            text="Sample text",
            score=0.95,
        )

        data = candidate.to_dict()

        assert data["id"] == "doc_1"
        assert data["text"] == "Sample text"
        assert data["score"] == 0.95

    def test_candidate_default_metadata(self):
        """metadata 应该默认为空字典。"""
        candidate = RerankCandidate(
            id="doc_1",
            text="Sample text",
            score=0.95,
        )

        assert candidate.metadata == {}


class TestRerankResult:
    """RerankResult 数据类测试。"""

    def test_result_creation(self):
        """应该能创建结果。"""
        candidates = [
            RerankCandidate(id="doc_1", text="Text 1", score=0.9),
            RerankCandidate(id="doc_2", text="Text 2", score=0.8),
        ]

        result = RerankResult(
            candidates=candidates,
            backend="none",
            elapsed_ms=1.5,
        )

        assert len(result.candidates) == 2
        assert result.backend == "none"
        assert result.elapsed_ms == 1.5
        assert result.fallback_used is False

    def test_result_to_dict(self):
        """应该能转换为字典。"""
        candidates = [
            RerankCandidate(id="doc_1", text="Text 1", score=0.9),
        ]

        result = RerankResult(
            candidates=candidates,
            backend="none",
            elapsed_ms=1.5,
            fallback_used=True,
            fallback_reason="CrossEncoder unavailable",
        )

        data = result.to_dict()

        assert data["backend"] == "none"
        assert data["count"] == 1
        assert data["fallback_used"] is True
        assert data["fallback_reason"] == "CrossEncoder unavailable"


class TestRerankerErrors:
    """Reranker 错误类测试。"""

    def test_reranker_error_basic(self):
        """基本错误信息。"""
        error = RerankerError("Something went wrong")

        assert "Something went wrong" in str(error)

    def test_reranker_error_with_backend(self):
        """带后端的错误信息。"""
        error = RerankerError("Something went wrong", backend="cross_encoder")

        assert "[cross_encoder]" in str(error)
        assert "Something went wrong" in str(error)

    def test_reranker_error_with_original(self):
        """带原始错误的错误信息。"""
        original = ValueError("Original error")
        error = RerankerError("Wrapped", backend="llm", original_error=original)

        assert error.original_error == original

    def test_config_error_is_reranker_error(self):
        """ConfigError 应该是 RerankerError 的子类。"""
        error = RerankerConfigError("Invalid config", backend="cross_encoder")

        assert isinstance(error, RerankerError)

    def test_unavailable_error_is_reranker_error(self):
        """UnavailableError 应该是 RerankerError 的子类。"""
        error = RerankerUnavailableError("Service unavailable", backend="llm")

        assert isinstance(error, RerankerError)


class TestNoneRerankerContract:
    """NoneReranker 契约测试。"""

    @pytest.fixture
    def reranker(self):
        """创建 NoneReranker 实例。"""
        return NoneReranker()

    def test_implements_base_reranker(self, reranker):
        """应该实现 BaseReranker 接口。"""
        assert isinstance(reranker, BaseReranker)

    def test_get_backend_name(self, reranker):
        """应该返回正确的后端名称。"""
        assert reranker.get_backend_name() == "none"

    def test_is_available(self, reranker):
        """应该始终可用。"""
        assert reranker.is_available() is True

    def test_rerank_returns_rerank_result(self, reranker):
        """rerank 应该返回 RerankResult。"""
        candidates = [
            RerankCandidate(id="doc_1", text="Text 1", score=0.9),
            RerankCandidate(id="doc_2", text="Text 2", score=0.8),
        ]

        result = reranker.rerank("query", candidates)

        assert isinstance(result, RerankResult)

    def test_rerank_preserves_order(self, reranker):
        """应该保持原始顺序。"""
        candidates = [
            RerankCandidate(id="doc_1", text="Text 1", score=0.9),
            RerankCandidate(id="doc_2", text="Text 2", score=0.8),
            RerankCandidate(id="doc_3", text="Text 3", score=0.7),
        ]

        result = reranker.rerank("query", candidates)

        assert result.candidates[0].id == "doc_1"
        assert result.candidates[1].id == "doc_2"
        assert result.candidates[2].id == "doc_3"

    def test_rerank_with_top_k(self, reranker):
        """应该支持 top_k 参数。"""
        candidates = [
            RerankCandidate(id="doc_1", text="Text 1", score=0.9),
            RerankCandidate(id="doc_2", text="Text 2", score=0.8),
            RerankCandidate(id="doc_3", text="Text 3", score=0.7),
        ]

        result = reranker.rerank("query", candidates, top_k=2)

        assert len(result.candidates) == 2
        assert result.candidates[0].id == "doc_1"
        assert result.candidates[1].id == "doc_2"

    def test_rerank_empty_candidates(self, reranker):
        """应该处理空候选项列表。"""
        result = reranker.rerank("query", [])

        assert len(result.candidates) == 0
        assert result.backend == "none"

    def test_rerank_top_k_zero(self, reranker):
        """top_k=0 应该返回空列表。"""
        candidates = [
            RerankCandidate(id="doc_1", text="Text 1", score=0.9),
        ]

        result = reranker.rerank("query", candidates, top_k=0)

        assert len(result.candidates) == 0

    def test_rerank_top_k_greater_than_candidates(self, reranker):
        """top_k 大于候选项数量时应该返回全部。"""
        candidates = [
            RerankCandidate(id="doc_1", text="Text 1", score=0.9),
        ]

        result = reranker.rerank("query", candidates, top_k=10)

        assert len(result.candidates) == 1

    def test_rerank_result_has_elapsed_ms(self, reranker):
        """结果应该包含耗时。"""
        candidates = [
            RerankCandidate(id="doc_1", text="Text 1", score=0.9),
        ]

        result = reranker.rerank("query", candidates)

        assert result.elapsed_ms is not None
        assert result.elapsed_ms >= 0

    def test_rerank_result_no_fallback(self, reranker):
        """NoneReranker 不应该使用回退。"""
        candidates = [
            RerankCandidate(id="doc_1", text="Text 1", score=0.9),
        ]

        result = reranker.rerank("query", candidates)

        assert result.fallback_used is False
        assert result.fallback_reason is None


class TestBaseRerankerContract:
    """BaseReranker 接口契约测试。

    任何实现 BaseReranker 的类都应该通过这些测试。
    """

    def test_none_reranker_satisfies_contract(self):
        """NoneReranker 应该满足契约。"""
        reranker = NoneReranker()

        # 必须实现的方法
        assert hasattr(reranker, "rerank")
        assert hasattr(reranker, "get_backend_name")
        assert hasattr(reranker, "is_available")

        # 方法必须是可调用的
        assert callable(reranker.rerank)
        assert callable(reranker.get_backend_name)
        assert callable(reranker.is_available)

        # get_backend_name 必须返回字符串
        backend_name = reranker.get_backend_name()
        assert isinstance(backend_name, str)
        assert len(backend_name) > 0

        # is_available 必须返回布尔值
        is_available = reranker.is_available()
        assert isinstance(is_available, bool)

        # rerank 必须接受正确的参数
        candidates = [
            RerankCandidate(id="doc_1", text="Text 1", score=0.9),
        ]
        result = reranker.rerank("query", candidates)
        assert isinstance(result, RerankResult)
