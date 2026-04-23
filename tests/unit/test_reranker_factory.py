"""
Reranker Factory 单元测试。

验证工厂路由逻辑和 None Reranker 回退行为。
"""

import pytest

from src.core.settings import RerankSettings, Settings
from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    RerankerError,
)
from src.libs.reranker.none_reranker import NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory


class FakeReranker(BaseReranker):
    """Fake Reranker 实现，用于测试。"""

    def __init__(self, available: bool = True):
        self._available = available

    def rerank(self, query, candidates, top_k=None, trace=None):
        # 简单按分数降序排列
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        return RerankResult(
            candidates=sorted_candidates[:top_k] if top_k else sorted_candidates,
            backend="fake",
        )

    def get_backend_name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return self._available


class TestRerankerFactory:
    """RerankerFactory 测试。"""

    def test_get_supported_providers(self):
        """应该返回支持的 provider 列表。"""
        providers = RerankerFactory.get_supported_providers()

        assert "none" in providers
        assert "cross_encoder" in providers
        assert "llm" in providers

    def test_unsupported_provider_raises_error(self):
        """不支持的 provider 应该抛出错误。"""
        settings = Settings(
            rerank=RerankSettings(
                enabled=True,
                provider="unsupported_provider",
            )
        )

        with pytest.raises(RerankerError) as exc_info:
            RerankerFactory.create(settings)

        assert "Unsupported Reranker provider" in str(exc_info.value)
        assert exc_info.value.backend == "unsupported_provider"

    def test_create_none_reranker_when_disabled(self):
        """未启用时应该返回 None Reranker。"""
        settings = Settings(
            rerank=RerankSettings(
                enabled=False,
                provider="cross_encoder",
            )
        )

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, NoneReranker)
        assert reranker.get_backend_name() == "none"

    def test_create_none_reranker_explicitly(self):
        """应该能显式创建 None Reranker。"""
        settings = Settings(
            rerank=RerankSettings(
                enabled=True,
                provider="none",
            )
        )

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, NoneReranker)

    def test_create_none_reranker_empty_provider(self):
        """空 provider 应该返回 None Reranker。"""
        settings = Settings(
            rerank=RerankSettings(
                enabled=True,
                provider="",
            )
        )

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, NoneReranker)

    def test_create_cross_encoder_reranker(self):
        """应该能创建 Cross-Encoder Reranker。"""
        settings = Settings(
            rerank=RerankSettings(
                enabled=True,
                provider="cross_encoder",
                model="test-model",
            )
        )

        reranker = RerankerFactory.create(settings)

        assert reranker is not None
        assert reranker.get_backend_name() == "cross_encoder"

    def test_create_llm_reranker(self):
        """应该能创建 LLM Reranker。"""
        settings = Settings(
            rerank=RerankSettings(
                enabled=True,
                provider="llm",
            )
        )

        reranker = RerankerFactory.create(settings)

        assert reranker is not None
        assert reranker.get_backend_name() == "llm"

    def test_get_fallback(self):
        """应该能获取默认回退 Reranker。"""
        fallback = RerankerFactory.get_fallback()

        assert isinstance(fallback, NoneReranker)
        assert fallback.is_available() is True


class TestNoneReranker:
    """NoneReranker 测试。"""

    def test_rerank_preserves_order(self):
        """None Reranker 应该保持原始顺序。"""
        reranker = NoneReranker()

        candidates = [
            RerankCandidate(id="1", text="text1", score=0.5),
            RerankCandidate(id="2", text="text2", score=0.9),
            RerankCandidate(id="3", text="text3", score=0.3),
        ]

        result = reranker.rerank("query", candidates)

        assert len(result.candidates) == 3
        assert result.candidates[0].id == "1"
        assert result.candidates[1].id == "2"
        assert result.candidates[2].id == "3"

    def test_rerank_with_top_k(self):
        """应该正确截取 top_k 结果。"""
        reranker = NoneReranker()

        candidates = [
            RerankCandidate(id="1", text="text1", score=0.5),
            RerankCandidate(id="2", text="text2", score=0.9),
            RerankCandidate(id="3", text="text3", score=0.3),
        ]

        result = reranker.rerank("query", candidates, top_k=2)

        assert len(result.candidates) == 2
        assert result.candidates[0].id == "1"
        assert result.candidates[1].id == "2"

    def test_rerank_empty_candidates(self):
        """空候选项列表应该返回空结果。"""
        reranker = NoneReranker()

        result = reranker.rerank("query", [])

        assert len(result.candidates) == 0
        assert result.backend == "none"

    def test_is_always_available(self):
        """None Reranker 应该始终可用。"""
        reranker = NoneReranker()

        assert reranker.is_available() is True

    def test_result_no_fallback_used(self):
        """None Reranker 结果不应该标记为回退。"""
        reranker = NoneReranker()

        result = reranker.rerank("query", [
            RerankCandidate(id="1", text="text1", score=0.5),
        ])

        assert result.fallback_used is False


class TestRerankCandidate:
    """RerankCandidate 测试。"""

    def test_to_dict(self):
        """RerankCandidate 应该能转换为字典。"""
        candidate = RerankCandidate(
            id="test_id",
            text="test text",
            score=0.95,
            metadata={"source": "doc.pdf"},
        )

        data = candidate.to_dict()

        assert data["id"] == "test_id"
        assert data["text"] == "test text"
        assert data["score"] == 0.95
        assert data["metadata"]["source"] == "doc.pdf"


class TestRerankResult:
    """RerankResult 测试。"""

    def test_to_dict(self):
        """RerankResult 应该能转换为字典。"""
        result = RerankResult(
            candidates=[
                RerankCandidate(id="1", text="text1", score=0.9),
            ],
            backend="none",
            elapsed_ms=1.5,
            fallback_used=False,
        )

        data = result.to_dict()

        assert data["backend"] == "none"
        assert data["count"] == 1
        assert data["fallback_used"] is False
        assert data["elapsed_ms"] == 1.5

    def test_to_dict_with_fallback(self):
        """带回退信息的结果应该正确转换。"""
        result = RerankResult(
            candidates=[],
            backend="none",
            fallback_used=True,
            fallback_reason="Cross-Encoder timeout",
        )

        data = result.to_dict()

        assert data["fallback_used"] is True
        assert data["fallback_reason"] == "Cross-Encoder timeout"


class TestRerankerError:
    """RerankerError 测试。"""

    def test_reranker_error_basic(self):
        """基本错误信息。"""
        error = RerankerError("Something went wrong")

        assert "Something went wrong" in str(error)

    def test_reranker_error_with_backend(self):
        """带后端的错误信息。"""
        error = RerankerError("Something went wrong", backend="cross_encoder")

        assert "[cross_encoder]" in str(error)
        assert "Something went wrong" in str(error)

    def test_reranker_error_subclasses(self):
        """错误子类应该继承 RerankerError。"""
        from src.libs.reranker.base_reranker import (
            RerankerConfigError,
            RerankerUnavailableError,
            RerankerTimeoutError,
        )

        config_error = RerankerConfigError("Invalid config", backend="llm")
        unavail_error = RerankerUnavailableError("Not available", backend="cross_encoder")
        timeout_error = RerankerTimeoutError("Timeout", backend="llm")

        assert isinstance(config_error, RerankerError)
        assert isinstance(unavail_error, RerankerError)
        assert isinstance(timeout_error, RerankerError)
