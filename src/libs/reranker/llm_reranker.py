"""
LLM Reranker 实现。

基于 LLM 的重排序，读取 rerank prompt 并调用 LLM 进行智能排序。
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional

from src.core.trace.trace_context import TraceContext
from src.libs.llm.base_llm import BaseLLM, LLMError
from src.libs.reranker.base_reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    RerankerError,
    RerankerTimeoutError,
    RerankerUnavailableError,
)


class LLMReranker(BaseReranker):
    """
    LLM Reranker 实现。

    使用 LLM 对候选项进行智能重排序。
    """

    DEFAULT_PROMPT_PATH = "config/prompts/rerank.txt"
    DEFAULT_TIMEOUT = 30  # 秒

    def __init__(
        self,
        llm: BaseLLM,
        settings=None,
        prompt_path: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        初始化 LLM Reranker。

        Args:
            llm: LLM 实例。
            settings: Rerank 配置（可选）。
            prompt_path: Prompt 模板路径（可选）。
            timeout: 超时时间（秒，可选）。
        """
        self.llm = llm

        if settings:
            if hasattr(settings, "rerank"):
                settings = settings.rerank
            self.top_k = getattr(settings, "top_k", None) or 5
            self.model = getattr(settings, "model", None) or "default"
        else:
            self.top_k = 5
            self.model = "default"

        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.prompt_path = prompt_path or self.DEFAULT_PROMPT_PATH

        # 加载 prompt 模板
        self._prompt_template = self._load_prompt(self.prompt_path)

    def _load_prompt(self, path: str) -> str:
        """加载 prompt 模板。"""
        prompt_path = Path(path)
        if not prompt_path.exists():
            # 使用默认模板
            return self._get_default_prompt()

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 转义 JSON 示例中的大括号（非占位符的大括号）
            # 保留 {query} 和 {candidates} 作为占位符
            import re
            # 将非占位符的 { 和 } 转义为 {{ 和 }}
            content = re.sub(r'\{(?!query|candidates)', '{{', content)
            content = re.sub(r'(?<!query|candidates)\}', '}}', content)
            return content
        except Exception:
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """获取默认 prompt 模板。"""
        return """# Rerank Prompt

You are an AI assistant specialized in ranking search results by relevance.

## Task

Given a query and a list of candidate documents, rank them by relevance to the query.

## Input

Query: {query}

Candidates:
{candidates}

## Guidelines

1. Evaluate each candidate's relevance to the query
2. Consider semantic similarity, not just keyword matching
3. Prioritize candidates that directly answer the query
4. Penalize candidates that are tangentially related

## Output Format

Return a JSON array of candidate indices sorted by relevance (most relevant first):
```json
{{
  "ranked_indices": [2, 0, 3, 1],
  "scores": [0.95, 0.82, 0.75, 0.60]
}}
```"""

    def _format_candidates(self, candidates: List[RerankCandidate]) -> str:
        """格式化候选项列表。"""
        lines = []
        for i, candidate in enumerate(candidates):
            # 截断过长的文本
            text = candidate.text
            if len(text) > 500:
                text = text[:500] + "..."
            lines.append(f"[{i}] {text}")
        return "\n".join(lines)

    def _parse_llm_response(self, response: str, candidates: List[RerankCandidate]) -> List[RerankCandidate]:
        """
        解析 LLM 响应，提取排序后的候选项。

        Args:
            response: LLM 响应文本。
            candidates: 原始候选项列表。

        Returns:
            排序后的候选项列表。
        """
        # 尝试从响应中提取 JSON
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析整个响应
            json_str = response.strip()

        try:
            data = json.loads(json_str)
            ranked_indices = data.get("ranked_indices", [])
            scores = data.get("scores", [])

            # 根据排序索引重新排列候选项
            reranked = []
            for i, idx in enumerate(ranked_indices):
                if 0 <= idx < len(candidates):
                    candidate = candidates[idx]
                    # 更新分数
                    if i < len(scores):
                        new_score = float(scores[i])
                    else:
                        # 如果没有对应分数，使用倒数排名作为分数
                        new_score = 1.0 / (i + 1)
                    reranked.append(RerankCandidate(
                        id=candidate.id,
                        text=candidate.text,
                        score=new_score,
                        metadata=candidate.metadata,
                    ))

            # 如果解析成功但没有结果，返回原始顺序
            if not reranked:
                return candidates

            return reranked

        except (json.JSONDecodeError, KeyError, ValueError):
            # 解析失败，返回原始顺序
            return candidates

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> RerankResult:
        """
        对候选项进行重排序。

        Args:
            query: 查询文本。
            candidates: 候选项列表。
            top_k: 返回结果数量。
            trace: 追踪上下文。

        Returns:
            RerankResult: 重排序结果。
        """
        import time

        start_time = time.time()

        # 确定返回数量
        k = top_k or self.top_k or len(candidates)

        # 空候选项处理
        if not candidates:
            return RerankResult(
                candidates=[],
                backend="llm",
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        try:
            # 构造 prompt
            candidates_text = self._format_candidates(candidates)
            prompt = self._prompt_template.format(
                query=query,
                candidates=candidates_text,
            )

            # 调用 LLM
            response = self.llm.chat_with_str(
                prompt=prompt,
                temperature=0.0,  # 低温度确保稳定输出
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # 解析响应
            reranked = self._parse_llm_response(response, candidates)

            # 截取 top_k
            reranked = reranked[:k]

            # 记录追踪
            if trace:
                trace.record_stage(
                    stage_name="rerank",
                    elapsed_ms=elapsed_ms,
                    method="llm",
                    provider="llm",
                    details={
                        "model": self.llm.get_model_name(),
                        "candidate_count": len(candidates),
                        "result_count": len(reranked),
                    },
                )

            return RerankResult(
                candidates=reranked,
                backend="llm",
                elapsed_ms=elapsed_ms,
            )

        except LLMError as e:
            elapsed_ms = (time.time() - start_time) * 1000

            # LLM 错误，抛出不可用异常
            raise RerankerUnavailableError(
                f"LLM call failed: {e}",
                backend="llm",
                original_error=e,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            # 其他错误
            raise RerankerError(
                f"Rerank failed: {e}",
                backend="llm",
                original_error=e,
            )

    def get_backend_name(self) -> str:
        """获取后端名称。"""
        return "llm"

    def is_available(self) -> bool:
        """LLM Reranker 可用性检查。"""
        return self.llm is not None
