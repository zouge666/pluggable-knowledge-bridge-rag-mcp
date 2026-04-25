"""
Metadata Enricher - 元数据增强处理器。

提供规则增强和可选 LLM 增强，为 Chunk 生成 title、summary、tags。
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


class MetadataEnricher(BaseTransform):
    """
    元数据增强处理器。

    功能：
    1. 规则增强：从文本中提取标题、生成简单摘要和标签
    2. LLM 增强：使用 LLM 生成语义丰富的 title、summary、tags
    3. 降级机制：LLM 失败时回退到规则结果
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None,
    ):
        """
        初始化 MetadataEnricher。

        Args:
            settings: 应用配置。
            llm: LLM 实例（可选，用于 LLM 增强）。
            prompt_path: Prompt 模板路径（可选）。
        """
        self.settings = settings
        self.use_llm = settings.ingestion.metadata_enricher.get("use_llm", False)

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
        default_path = "config/prompts/metadata_enrichment.txt"

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
        return """You are an AI assistant specialized in analyzing document chunks and extracting metadata.

## Task

Analyze the following text chunk and extract:
1. A concise title (max 10 words) that captures the main topic
2. A brief summary (max 50 words) of the key content
3. 3-5 relevant tags for categorization

## Input Text

{text}

## Output Format

Provide your response in the following JSON format:
{
  "title": "Your title here",
  "summary": "Your summary here",
  "tags": ["tag1", "tag2", "tag3"]
}

Only output the JSON, no additional text."""

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """
        对 Chunk 列表进行元数据增强。

        Args:
            chunks: 待处理的 Chunk 列表。
            trace: 追踪上下文（可选）。

        Returns:
            List[Chunk]: 增强后的 Chunk 列表。
        """
        start_time = time.time()
        enriched_chunks = []

        for chunk in chunks:
            enriched_chunk = self._enrich_chunk(chunk, trace)
            enriched_chunks.append(enriched_chunk)

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="metadata_enrichment",
                elapsed_ms=elapsed_ms,
                method="rule" if not self.use_llm else "llm",
                details={
                    "chunk_count": len(chunks),
                    "use_llm": self.use_llm,
                },
            )

        return enriched_chunks

    def _enrich_chunk(
        self,
        chunk: Chunk,
        trace: Optional[TraceContext] = None,
    ) -> Chunk:
        """
        增强单个 Chunk 的元数据。

        Args:
            chunk: 待处理的 Chunk。
            trace: 追踪上下文（可选）。

        Returns:
            Chunk: 增强后的 Chunk。
        """
        # 如果启用 LLM，尝试 LLM 增强
        if self.use_llm and self._llm:
            llm_metadata = self._llm_enrich(chunk.text, trace)

            if llm_metadata:
                # LLM 成功
                metadata = chunk.metadata.copy()
                metadata["title"] = llm_metadata.get("title", "")
                metadata["summary"] = llm_metadata.get("summary", "")
                metadata["tags"] = llm_metadata.get("tags", [])
                metadata["enriched_by"] = "llm"

                return Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref,
                )

        # 规则模式或 LLM 降级
        rule_metadata = self._rule_based_enrich(chunk.text)
        metadata = chunk.metadata.copy()
        metadata["title"] = rule_metadata.get("title", "")
        metadata["summary"] = rule_metadata.get("summary", "")
        metadata["tags"] = rule_metadata.get("tags", [])
        metadata["enriched_by"] = "rule"

        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        """
        规则增强。

        Args:
            text: Chunk 文本。

        Returns:
            Dict[str, Any]: 提取的元数据。
        """
        if not text:
            return {"title": "", "summary": "", "tags": []}

        # 1. 提取标题
        title = self._extract_title(text)

        # 2. 生成摘要
        summary = self._generate_summary(text)

        # 3. 提取标签
        tags = self._extract_tags(text)

        return {"title": title, "summary": summary, "tags": tags}

    def _extract_title(self, text: str) -> str:
        """
        从文本中提取标题。

        优先级：
        1. Markdown 标题 (# 开头)
        2. 第一行非空文本（截断至 50 字符）

        Args:
            text: 文本内容。

        Returns:
            str: 提取的标题。
        """
        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查 Markdown 标题
            if line.startswith("#"):
                # 移除 # 符号
                title = re.sub(r"^#+\s*", "", line).strip()
                if title:
                    return title[:100]  # 限制长度

            # 使用第一个非空行作为标题
            return line[:100]

        return ""

    def _generate_summary(self, text: str) -> str:
        """
        生成简单摘要。

        策略：取前 200 个字符作为摘要。

        Args:
            text: 文本内容。

        Returns:
            str: 摘要。
        """
        # 清理文本
        cleaned = re.sub(r"\s+", " ", text.strip())

        # 截断至 200 字符
        if len(cleaned) <= 200:
            return cleaned

        # 在单词边界截断
        truncated = cleaned[:200]
        last_space = truncated.rfind(" ")
        if last_space > 100:
            truncated = truncated[:last_space]

        return truncated.strip() + "..."

    def _extract_tags(self, text: str) -> List[str]:
        """
        从文本中提取标签。

        策略：
        1. 提取 Markdown 标题作为标签
        2. 提取代码块语言标识
        3. 提取常见关键词（如 API、配置、安装等）

        Args:
            text: 文本内容。

        Returns:
            List[str]: 标签列表。
        """
        tags = set()

        # 1. 提取 Markdown 标题
        for match in re.finditer(r"^#+\s+(.+)$", text, re.MULTILINE):
            title = match.group(1).strip().lower()
            # 分词并添加
            words = re.findall(r"\b\w+\b", title)
            for word in words:
                if len(word) >= 3 and word not in {"the", "and", "for", "with", "from"}:
                    tags.add(word)

        # 2. 提取代码块语言
        for match in re.finditer(r"```(\w+)", text):
            lang = match.group(1).lower()
            tags.add(f"code:{lang}")

        # 3. 提取常见关键词
        keyword_patterns = [
            (r"\bapi\b", "api"),
            (r"\bconfig(?:uration)?\b", "configuration"),
            (r"\binstall(?:ation)?\b", "installation"),
            (r"\bsetup\b", "setup"),
            (r"\bexample\b", "example"),
            (r"\btutorial\b", "tutorial"),
            (r"\bguide\b", "guide"),
            (r"\breference\b", "reference"),
            (r"\barchitecture\b", "architecture"),
            (r"\bdeployment\b", "deployment"),
            (r"\bsecurity\b", "security"),
            (r"\bauthentication\b", "auth"),
            (r"\bdatabase\b", "database"),
            (r"\bmodel\b", "model"),
            (r"\btraining\b", "training"),
            (r"\binference\b", "inference"),
        ]

        text_lower = text.lower()
        for pattern, tag in keyword_patterns:
            if re.search(pattern, text_lower):
                tags.add(tag)

        # 限制标签数量
        return list(tags)[:10]

    def _llm_enrich(
        self,
        text: str,
        trace: Optional[TraceContext] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        LLM 增强。

        Args:
            text: Chunk 文本。
            trace: 追踪上下文（可选）。

        Returns:
            Optional[Dict[str, Any]]: LLM 提取的元数据，失败返回 None。
        """
        if not self._llm or not text:
            return None

        try:
            prompt = self._prompt_template.replace("{text}", text[:4000])  # 限制输入长度
            response = self._llm.chat([{"role": "user", "content": prompt}])

            # 解析 JSON 响应
            content = response.content.strip()

            # 移除可能的 markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # 解析 JSON
            import json
            result = json.loads(content)

            # 验证字段
            return {
                "title": result.get("title", "")[:100],
                "summary": result.get("summary", "")[:300],
                "tags": result.get("tags", [])[:10],
            }

        except Exception:
            # LLM 失败，返回 None 触发降级
            return None


class FakeMetadataEnricher(BaseTransform):
    """
    Fake Metadata Enricher 用于测试。

    不依赖真实 LLM，直接使用规则增强。
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """增强 Chunk 列表的元数据。"""
        import time
        start_time = time.time()

        enriched_chunks = []

        for chunk in chunks:
            metadata = chunk.metadata.copy()

            # 规则增强
            rule_metadata = self._rule_based_enrich(chunk.text)
            metadata["title"] = rule_metadata.get("title", "")
            metadata["summary"] = rule_metadata.get("summary", "")
            metadata["tags"] = rule_metadata.get("tags", [])
            metadata["enriched_by"] = "rule"

            enriched_chunks.append(Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata=metadata,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                source_ref=chunk.source_ref,
            ))

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="metadata_enrichment",
                elapsed_ms=elapsed_ms,
                method="rule",
                details={
                    "chunk_count": len(chunks),
                    "use_llm": self.use_llm,
                },
            )

        return enriched_chunks

    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        """规则增强。"""
        if not text:
            return {"title": "", "summary": "", "tags": []}

        # 简单实现
        lines = text.strip().split("\n")
        title = ""
        for line in lines:
            line = line.strip()
            if line:
                title = line[:100]
                break

        summary = text[:200].strip()
        if len(text) > 200:
            summary += "..."

        return {"title": title, "summary": summary, "tags": []}
