"""
Chunk Refiner - 文本精炼处理器。

提供规则去噪和可选 LLM 增强，失败时自动降级。
"""

import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm import LLMFactory, BaseLLM


class ChunkRefiner(BaseTransform):
    """
    Chunk 精炼处理器。

    功能：
    1. 规则去噪：去除页眉页脚、多余空白、格式标记
    2. LLM 增强：可选的智能重写（需配置 LLM）
    3. 降级机制：LLM 失败时回退到规则结果
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None,
    ):
        """
        初始化 ChunkRefiner。

        Args:
            settings: 应用配置。
            llm: LLM 实例（可选，用于 LLM 增强）。
            prompt_path: Prompt 模板路径（可选）。
        """
        self.settings = settings
        self.use_llm = settings.ingestion.chunk_refiner.get("use_llm", False)

        # LLM 实例（可选）
        self._llm = llm
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(settings)
            except Exception:
                # LLM 创建失败，降级为规则模式
                self.use_llm = False

        # Prompt 模板
        self._prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: Optional[str] = None) -> str:
        """
        加载 Prompt 模板。

        Args:
            prompt_path: Prompt 文件路径（可选）。

        Returns:
            str: Prompt 模板。
        """
        default_path = "config/prompts/chunk_refinement.txt"

        if prompt_path:
            path = Path(prompt_path)
        else:
            path = Path(default_path)

        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass

        # 默认 Prompt
        return """You are an AI assistant specialized in cleaning and improving text chunks from technical documents.

## Task

Analyze and refine the following text chunk to:
1. Remove noise (headers, footers, page numbers, formatting artifacts)
2. Fix OCR errors or typos
3. Improve readability while preserving original meaning
4. Maintain technical accuracy

## Input Text

{text}

## Output

Provide only the refined text, without explanations or comments."""

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """
        对 Chunk 列表进行精炼处理。

        Args:
            chunks: 待处理的 Chunk 列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[Chunk]: 精炼后的 Chunk 列表。
        """
        start_time = time.time()
        refined_chunks = []

        for chunk in chunks:
            refined_chunk = self._refine_chunk(chunk, trace)
            refined_chunks.append(refined_chunk)

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="chunk_refinement",
                elapsed_ms=elapsed_ms,
                method="rule" if not self.use_llm else "llm",
                details={
                    "chunk_count": len(chunks),
                    "use_llm": self.use_llm,
                },
            )

        return refined_chunks

    def _refine_chunk(
        self,
        chunk: Chunk,
        trace: Optional[TraceContext] = None,
    ) -> Chunk:
        """
        精炼单个 Chunk。

        Args:
            chunk: 待处理的 Chunk。
            trace: 追踪上下文（可选）。

        Returns:
            Chunk: 精炼后的 Chunk。
        """
        # 先进行规则去噪
        rule_refined_text = self._rule_based_refine(chunk.text)

        # 如果启用 LLM，尝试 LLM 增强
        if self.use_llm and self._llm:
            llm_refined_text = self._llm_refine(rule_refined_text, trace)

            if llm_refined_text:
                # LLM 成功
                metadata = chunk.metadata.copy()
                metadata["refined_by"] = "llm"
                return Chunk(
                    id=chunk.id,
                    text=llm_refined_text,
                    metadata=metadata,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref,
                )

        # 规则模式或 LLM 降级
        metadata = chunk.metadata.copy()
        metadata["refined_by"] = "rule"

        return Chunk(
            id=chunk.id,
            text=rule_refined_text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    def _rule_based_refine(self, text: str) -> str:
        """
        规则去噪。

        Args:
            text: 原始文本。

        Returns:
            str: 去噪后的文本。
        """
        if not text:
            return text

        result = text

        # 1. 移除页眉页脚模式
        # 移除 "--- Page X ---" 格式
        result = re.sub(r"---\s*Page\s*\d+\s*---", "", result)

        # 移除 "Page X of Y" 格式
        result = re.sub(r"Page\s*\d+\s*of\s*\d+", "", result)

        # 移除常见的页眉标记
        result = re.sub(r"CONFIDENTIAL.*(?:\n|$)", "", result, flags=re.IGNORECASE)
        result = re.sub(r"Internal Use Only.*(?:\n|$)", "", result, flags=re.IGNORECASE)

        # 2. 移除 HTML 注释
        result = re.sub(r"<!--.*?-->", "", result, flags=re.DOTALL)

        # 3. 移除简单 HTML 标签（保留内容）
        result = re.sub(r"<div[^>]*>", "", result)
        result = re.sub(r"</div>", "", result)
        result = re.sub(r"<p[^>]*>", "", result)
        result = re.sub(r"</p>", "", result)

        # 4. 处理多余空白
        # 多个连续空格变为单个
        result = re.sub(r"[ \t]+", " ", result)

        # 多个连续空行变为单个
        result = re.sub(r"\n{3,}", "\n\n", result)

        # 移除行首行尾空白
        lines = result.split("\n")
        lines = [line.strip() for line in lines]
        result = "\n".join(lines)

        # 5. 移除分隔线
        result = re.sub(r"^---+$", "", result, flags=re.MULTILINE)

        # 6. 清理空行
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = result.strip()

        return result

    def _llm_refine(
        self,
        text: str,
        trace: Optional[TraceContext] = None,
    ) -> Optional[str]:
        """
        LLM 增强。

        Args:
            text: 规则去噪后的文本。
            trace: 追踪上下文（可选）。

        Returns:
            Optional[str]: LLM 精炼后的文本，失败返回 None。
        """
        if not self._llm or not text:
            return None

        try:
            prompt = self._prompt_template.replace("{text}", text)
            response = self._llm.chat([{"role": "user", "content": prompt}])
            return response.content.strip()
        except Exception:
            # LLM 失败，返回 None 触发降级
            return None


class FakeChunkRefiner(BaseTransform):
    """
    Fake Chunk Refiner 用于测试。

    不依赖真实 LLM，直接返回原文或简单处理。
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """精炼 Chunk 列表。"""
        import time
        start_time = time.time()

        refined_chunks = []

        for chunk in chunks:
            refined_text = self._rule_based_refine(chunk.text)
            metadata = chunk.metadata.copy()
            metadata["refined_by"] = "rule"

            refined_chunks.append(Chunk(
                id=chunk.id,
                text=refined_text,
                metadata=metadata,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                source_ref=chunk.source_ref,
            ))

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="chunk_refinement",
                elapsed_ms=elapsed_ms,
                method="rule",
                details={
                    "chunk_count": len(chunks),
                    "use_llm": self.use_llm,
                },
            )

        return refined_chunks

        return refined_chunks

    def _rule_based_refine(self, text: str) -> str:
        """简单规则去噪。"""
        if not text:
            return text

        result = text

        # 移除页眉页脚
        import re
        result = re.sub(r"---\s*Page\s*\d+\s*---", "", result)
        result = re.sub(r"Page\s*\d+\s*of\s*\d+", "", result)

        # 处理空白
        result = re.sub(r"[ \t]+", " ", result)
        result = re.sub(r"\n{3,}", "\n\n", result)

        # 移除行首行尾空白
        lines = result.split("\n")
        lines = [line.strip() for line in lines]
        result = "\n".join(lines)

        return result.strip()