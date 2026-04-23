"""
Cross-Encoder Reranker 单元测试。

使用 Mock scorer 验证重排序逻辑。
"""

import pytest
from unittest.mock import MagicMock, patch

from src.core.settings import RerankSettings
from src.libs.reranker.base_reranker import RerankCandidate, RerankerError, RerankerUnavailableError
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """CrossEncoderReranker 测试。"""

    def test_init_default_settings(self):
        """应该能使用默认设置初始化。"""
        reranker = CrossEncoderReranker()

        assert reranker.get_backend_name() == "cross_encoder"
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_init_with_settings(self):
        """应该能使用 settings 初始化。"""
        settings = RerankSettings(
            enabled=True,
            provider="cross_encoder",
            model="custom-model",
            top_k=10,
        )
        reranker = CrossEncoderReranker(settings=settings)

        assert reranker.model_name == "custom-model"
        assert reranker.top_k == 10

    def test_init_with_model_name(self):
        """应该能使用模型名称初始化。"""
        reranker = CrossEncoderReranker(model="my-custom-model")

        assert reranker.model_name == "my-custom-model"

    def test_rerank_empty_candidates(self):
        """空候选项列表应该返回空结果。"""
        reranker = CrossEncoderReranker()

        # Mock 模型加载
        reranker._model = MagicMock()
        reranker._available = True

        result = reranker.rerank("test query", [])

        assert result.candidates == []
        assert result.backend == "cross_encoder"

    def test_rerank_with_mock_model(self):
        """应该能使用 mock 模型进行重排序。"""
        reranker = CrossEncoderReranker()

        # Mock 模型
        mock_model = MagicMock()
        # 返回分数：第二个最高，第一个次之，第三个最低
        mock_model.predict.return_value = [0.5, 0.9, 0.3]
        reranker._model = mock_model
        reranker._available = True

        candidates = [
            RerankCandidate(id="1", text="First document", score=0.5),
            RerankCandidate(id="2", text="Second document", score=0.6),
            RerankCandidate(id="3", text="Third document", score=0.4),
        ]

        result = reranker.rerank("test query", candidates)

        # 应该按分数降序排列
        assert len(result.candidates) == 3
        assert result.candidates[0].id == "2"  # 最高分 0.9
        assert result.candidates[0].score == 0.9
        assert result.candidates[1].id == "1"  # 次高分 0.5
        assert result.candidates[1].score == 0.5
        assert result.candidates[2].id == "3"  # 最低分 0.3
        assert result.candidates[2].score == 0.3

    def test_rerank_with_top_k(self):
        """应该能限制返回数量。"""
        reranker = CrossEncoderReranker(settings=RerankSettings(top_k=2))

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.3]
        reranker._model = mock_model
        reranker._available = True

        candidates = [
            RerankCandidate(id="1", text="First", score=0.5),
            RerankCandidate(id="2", text="Second", score=0.6),
            RerankCandidate(id="3", text="Third", score=0.4),
        ]

        result = reranker.rerank("test query", candidates)

        assert len(result.candidates) == 2

    def test_rerank_preserves_metadata(self):
        """应该保留原始元数据。"""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]
        reranker._model = mock_model
        reranker._available = True

        candidates = [
            RerankCandidate(
                id="1",
                text="Document",
                score=0.5,
                metadata={"source": "test.pdf", "page": 5},
            ),
        ]

        result = reranker.rerank("test query", candidates)

        assert result.candidates[0].metadata == {"source": "test.pdf", "page": 5}

    def test_rerank_without_sentence_transformers(self):
        """没有 sentence-transformers 应该抛出错误。"""
        with patch('src.libs.reranker.cross_encoder_reranker.CrossEncoder', None):
            reranker = CrossEncoderReranker()

            candidates = [
                RerankCandidate(id="1", text="First", score=0.5),
            ]

            with pytest.raises(RerankerUnavailableError) as exc_info:
                reranker.rerank("test query", candidates)

            assert "sentence-transformers" in str(exc_info.value)

    def test_get_backend_name(self):
        """应该返回正确的后端名称。"""
        reranker = CrossEncoderReranker()

        assert reranker.get_backend_name() == "cross_encoder"

    def test_is_available_with_mock_model(self):
        """应该正确检查可用性。"""
        reranker = CrossEncoderReranker()

        # Mock 模型已加载
        reranker._model = MagicMock()
        reranker._available = True

        assert reranker.is_available() is True

        # Mock 模型不可用
        reranker._available = False
        assert reranker.is_available() is False

    def test_rerank_result_structure(self):
        """结果结构应该正确。"""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.5]
        reranker._model = mock_model
        reranker._available = True

        candidates = [
            RerankCandidate(id="1", text="First", score=0.5),
            RerankCandidate(id="2", text="Second", score=0.6),
        ]

        result = reranker.rerank("test query", candidates)

        assert result.backend == "cross_encoder"
        assert result.elapsed_ms is not None
        assert result.elapsed_ms >= 0
        assert result.fallback_used is False

    def test_predict_called_with_correct_pairs(self):
        """应该使用正确的 query-document 对调用 predict。"""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9]
        reranker._model = mock_model
        reranker._available = True

        candidates = [
            RerankCandidate(id="1", text="First document", score=0.5),
            RerankCandidate(id="2", text="Second document", score=0.6),
        ]

        reranker.rerank("my query", candidates)

        # 验证调用参数
        mock_model.predict.assert_called_once()
        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs) == 2
        assert pairs[0] == ("my query", "First document")
        assert pairs[1] == ("my query", "Second document")
