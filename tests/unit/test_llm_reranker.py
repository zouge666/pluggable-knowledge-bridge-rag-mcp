"""
LLM Reranker 单元测试。

使用 Mock LLM 验证重排序逻辑。
"""

import pytest
from unittest.mock import MagicMock, patch

from src.core.settings import RerankSettings, Settings
from src.libs.llm.base_llm import BaseLLM, ChatResponse, LLMError
from src.libs.reranker.base_reranker import RerankCandidate, RerankerError, RerankerUnavailableError
from src.libs.reranker.llm_reranker import LLMReranker


class FakeLLM(BaseLLM):
    """Fake LLM 实现，用于测试。"""

    def __init__(self, response: str = ""):
        self._response = response
        self.model = "fake-model"

    def chat(self, messages, temperature=None, max_tokens=None, trace=None):
        return ChatResponse(
            content=self._response,
            model=self.model,
        )

    def get_model_name(self) -> str:
        return self.model


class TestLLMReranker:
    """LLMReranker 测试。"""

    def test_init_with_llm(self):
        """应该能使用 LLM 实例初始化。"""
        llm = FakeLLM()
        reranker = LLMReranker(llm=llm)

        assert reranker.get_backend_name() == "llm"
        assert reranker.is_available() is True

    def test_init_with_settings(self):
        """应该能使用 settings 初始化。"""
        llm = FakeLLM()
        settings = RerankSettings(
            enabled=True,
            provider="llm",
            top_k=10,
        )
        reranker = LLMReranker(llm=llm, settings=settings)

        assert reranker.top_k == 10

    def test_rerank_empty_candidates(self):
        """空候选项列表应该返回空结果。"""
        llm = FakeLLM()
        reranker = LLMReranker(llm=llm)

        result = reranker.rerank("test query", [])

        assert result.candidates == []
        assert result.backend == "llm"

    def test_rerank_with_valid_response(self):
        """应该能解析有效的 LLM 响应。"""
        llm = FakeLLM(response="""```json
{
  "ranked_indices": [1, 0, 2],
  "scores": [0.95, 0.80, 0.60]
}
```""")
        reranker = LLMReranker(llm=llm)

        candidates = [
            RerankCandidate(id="1", text="First document", score=0.5),
            RerankCandidate(id="2", text="Second document", score=0.6),
            RerankCandidate(id="3", text="Third document", score=0.4),
        ]

        result = reranker.rerank("test query", candidates)

        assert len(result.candidates) == 3
        # 第一个应该是原来的索引 1（Second document）
        assert result.candidates[0].id == "2"
        assert result.candidates[0].score == 0.95
        assert result.candidates[1].id == "1"
        assert result.candidates[1].score == 0.80

    def test_rerank_with_top_k(self):
        """应该能限制返回数量。"""
        llm = FakeLLM(response="""```json
{
  "ranked_indices": [2, 1, 0],
  "scores": [0.95, 0.80, 0.60]
}
```""")
        reranker = LLMReranker(llm=llm, settings=RerankSettings(top_k=2))

        candidates = [
            RerankCandidate(id="1", text="First", score=0.5),
            RerankCandidate(id="2", text="Second", score=0.6),
            RerankCandidate(id="3", text="Third", score=0.4),
        ]

        result = reranker.rerank("test query", candidates)

        assert len(result.candidates) == 2

    def test_rerank_with_invalid_json_response(self):
        """无效 JSON 响应应该返回原始顺序。"""
        llm = FakeLLM(response="This is not valid JSON")
        reranker = LLMReranker(llm=llm)

        candidates = [
            RerankCandidate(id="1", text="First", score=0.5),
            RerankCandidate(id="2", text="Second", score=0.6),
        ]

        result = reranker.rerank("test query", candidates)

        # 应该返回原始候选项
        assert len(result.candidates) == 2

    def test_rerank_with_partial_response(self):
        """部分有效的响应应该被正确处理。"""
        llm = FakeLLM(response="""```json
{
  "ranked_indices": [1, 0],
  "scores": [0.95]
}
```""")
        reranker = LLMReranker(llm=llm)

        candidates = [
            RerankCandidate(id="1", text="First", score=0.5),
            RerankCandidate(id="2", text="Second", score=0.6),
        ]

        result = reranker.rerank("test query", candidates)

        assert len(result.candidates) == 2
        # 第一个应该有分数 0.95
        assert result.candidates[0].score == 0.95
        # 第二个应该使用倒数排名作为分数
        assert result.candidates[1].score == 0.5  # 1/2

    def test_rerank_llm_error(self):
        """LLM 错误应该抛出 RerankerUnavailableError。"""
        error_llm = MagicMock(spec=BaseLLM)
        error_llm.chat_with_str.side_effect = LLMError("API error")

        reranker = LLMReranker(llm=error_llm)

        candidates = [
            RerankCandidate(id="1", text="First", score=0.5),
        ]

        with pytest.raises(RerankerUnavailableError):
            reranker.rerank("test query", candidates)

    def test_format_candidates(self):
        """应该正确格式化候选项。"""
        llm = FakeLLM()
        reranker = LLMReranker(llm=llm)

        candidates = [
            RerankCandidate(id="1", text="First document", score=0.5),
            RerankCandidate(id="2", text="Second document", score=0.6),
        ]

        formatted = reranker._format_candidates(candidates)

        assert "[0] First document" in formatted
        assert "[1] Second document" in formatted

    def test_format_candidates_truncation(self):
        """过长的文本应该被截断。"""
        llm = FakeLLM()
        reranker = LLMReranker(llm=llm)

        long_text = "x" * 600
        candidates = [
            RerankCandidate(id="1", text=long_text, score=0.5),
        ]

        formatted = reranker._format_candidates(candidates)

        assert len(formatted) < len(long_text) + 10
        assert "..." in formatted

    def test_get_backend_name(self):
        """应该返回正确的后端名称。"""
        llm = FakeLLM()
        reranker = LLMReranker(llm=llm)

        assert reranker.get_backend_name() == "llm"

    def test_is_available(self):
        """应该正确检查可用性。"""
        llm = FakeLLM()
        reranker = LLMReranker(llm=llm)

        assert reranker.is_available() is True

        # None LLM 应该不可用
        reranker_none = LLMReranker(llm=None)
        assert reranker_none.is_available() is False

    def test_load_prompt_from_file(self):
        """应该能从文件加载 prompt。"""
        llm = FakeLLM()
        reranker = LLMReranker(llm=llm, prompt_path="config/prompts/rerank.txt")

        assert "{query}" in reranker._prompt_template
        assert "{candidates}" in reranker._prompt_template

    def test_default_prompt(self):
        """应该有默认 prompt 模板。"""
        llm = FakeLLM()
        reranker = LLMReranker(llm=llm, prompt_path="nonexistent/path.txt")

        assert "{query}" in reranker._prompt_template
        assert "{candidates}" in reranker._prompt_template
