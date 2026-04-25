"""
Query Processor - 查询预处理器。

对用户查询进行关键词提取和过滤器解析。
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

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


@dataclass
class ProcessedQuery:
    """
    处理后的查询结构。

    包含提取的关键词和解析的过滤器。
    """

    # 原始查询文本
    raw_query: str
    # 提取的关键词列表
    keywords: List[str] = field(default_factory=list)
    # 过滤器字典（如 collection, doc_type 等）
    filters: Dict[str, Any] = field(default_factory=dict)
    # 处理耗时（毫秒）
    elapsed_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "raw_query": self.raw_query,
            "keywords": self.keywords,
            "filters": self.filters,
            "elapsed_ms": self.elapsed_ms,
        }


class QueryProcessor:
    """
    查询预处理器。

    功能：
    1. 关键词提取（规则/分词）
    2. 过滤器解析
    3. 停用词过滤
    4. 支持追踪记录
    """

    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        min_keyword_length: int = 2,
        max_keywords: int = 20,
        lowercase: bool = True,
    ):
        """
        初始化 QueryProcessor。

        Args:
            stopwords: 停用词集合（可选）。
            min_keyword_length: 最小关键词长度。
            max_keywords: 最大关键词数量。
            lowercase: 是否转换为小写。
        """
        self._stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self._min_keyword_length = min_keyword_length
        self._max_keywords = max_keywords
        self._lowercase = lowercase

    def process(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> ProcessedQuery:
        """
        处理查询文本。

        Args:
            query: 原始查询文本。
            filters: 过滤器字典（可选）。
            trace: 追踪上下文（可选）。

        Returns:
            ProcessedQuery: 处理后的查询结构。
        """
        start_time = time.time()

        # 提取关键词
        keywords = self._extract_keywords(query)

        # 解析过滤器
        parsed_filters = self._parse_filters(filters)

        elapsed_ms = (time.time() - start_time) * 1000

        # 记录追踪
        if trace:
            trace.record_stage(
                stage_name="query_processing",
                elapsed_ms=elapsed_ms,
                method="rule",
                details={
                    "keyword_count": len(keywords),
                    "has_filters": bool(parsed_filters),
                },
            )

        return ProcessedQuery(
            raw_query=query,
            keywords=keywords,
            filters=parsed_filters,
            elapsed_ms=elapsed_ms,
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """
        从查询文本中提取关键词。

        Args:
            query: 查询文本。

        Returns:
            List[str]: 关键词列表。
        """
        if not query:
            return []

        # 预处理
        text = query.strip()
        if self._lowercase:
            text = text.lower()

        # 分词：按空格和标点分割
        # 匹配字母数字组合（支持中文字符）
        tokens = re.findall(r"[a-zA-Z0-9]+|[一-鿿]+", text)

        # 过滤并收集关键词
        keywords = []
        seen = set()  # 去重

        for token in tokens:
            # 长度过滤
            if len(token) < self._min_keyword_length:
                continue

            # 停用词过滤（只对英文）
            if token.isalpha() and token in self._stopwords:
                continue

            # 去重
            if token in seen:
                continue

            seen.add(token)
            keywords.append(token)

            # 限制数量
            if len(keywords) >= self._max_keywords:
                break

        return keywords

    def _parse_filters(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        解析过滤器。

        Args:
            filters: 原始过滤器字典。

        Returns:
            Dict[str, Any]: 解析后的过滤器。
        """
        if not filters:
            return {}

        # 复制过滤器，避免修改原始数据
        parsed = dict(filters)

        # 过滤空值
        parsed = {k: v for k, v in parsed.items() if v is not None}

        return parsed

    def get_stopwords(self) -> Set[str]:
        """获取当前停用词集合。"""
        return self._stopwords.copy()


class FakeQueryProcessor:
    """
    Fake Query Processor 用于测试。

    返回预设结果，不执行实际处理。
    """

    def __init__(self, keywords: Optional[List[str]] = None):
        """
        初始化 FakeQueryProcessor。

        Args:
            keywords: 预设的关键词列表。
        """
        self._default_keywords = keywords or ["fake", "keywords"]
        self.process_calls: List[dict] = []

    def process(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
    ) -> ProcessedQuery:
        """执行 fake 处理。"""
        self.process_calls.append({
            "query": query,
            "filters": filters,
        })

        return ProcessedQuery(
            raw_query=query,
            keywords=self._default_keywords.copy(),
            filters=filters or {},
            elapsed_ms=0.1,
        )

    def get_stopwords(self) -> Set[str]:
        """获取停用词集合。"""
        return set()
