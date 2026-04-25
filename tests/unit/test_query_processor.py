"""
Unit tests for QueryProcessor.
"""

import pytest

from src.core.query_engine.query_processor import (
    QueryProcessor,
    FakeQueryProcessor,
    ProcessedQuery,
    DEFAULT_STOPWORDS,
)
from src.core.trace.trace_context import TraceContext


class TestProcessedQuery:
    """Tests for ProcessedQuery dataclass."""

    def test_processed_query_creation(self):
        """Test creating ProcessedQuery."""
        result = ProcessedQuery(
            raw_query="test query",
            keywords=["test", "query"],
            filters={"collection": "my_docs"},
        )
        assert result.raw_query == "test query"
        assert result.keywords == ["test", "query"]
        assert result.filters == {"collection": "my_docs"}

    def test_processed_query_to_dict(self):
        """Test converting ProcessedQuery to dict."""
        result = ProcessedQuery(
            raw_query="test query",
            keywords=["test"],
            filters={},
            elapsed_ms=1.5,
        )
        d = result.to_dict()
        assert d["raw_query"] == "test query"
        assert d["keywords"] == ["test"]
        assert d["elapsed_ms"] == 1.5


class TestQueryProcessor:
    """Tests for QueryProcessor."""

    @pytest.fixture
    def processor(self):
        """Create default QueryProcessor."""
        return QueryProcessor()

    @pytest.fixture
    def processor_no_stopwords(self):
        """Create QueryProcessor without stopwords."""
        return QueryProcessor(stopwords=set())

    @pytest.fixture
    def processor_short_min_length(self):
        """Create QueryProcessor with short min length."""
        return QueryProcessor(min_keyword_length=1)

    def test_default_stopwords_not_empty(self):
        """Test default stopwords are defined."""
        assert len(DEFAULT_STOPWORDS) > 0
        assert "the" in DEFAULT_STOPWORDS
        assert "is" in DEFAULT_STOPWORDS

    def test_extract_keywords_simple(self, processor):
        """Test extracting keywords from simple query."""
        result = processor.process("machine learning algorithms")
        assert "machine" in result.keywords
        assert "learning" in result.keywords
        assert "algorithms" in result.keywords

    def test_extract_keywords_filters_stopwords(self, processor):
        """Test that stopwords are filtered."""
        result = processor.process("the quick brown fox")
        # "the" is a stopword
        assert "the" not in result.keywords
        assert "quick" in result.keywords
        assert "brown" in result.keywords
        assert "fox" in result.keywords

    def test_extract_keywords_filters_short_words(self, processor):
        """Test that short words are filtered."""
        result = processor.process("a big cat")
        # "a" is stopword and short, "big" and "cat" are kept
        assert "big" in result.keywords
        assert "cat" in result.keywords

    def test_extract_keywords_preserves_short_with_config(self, processor_short_min_length):
        """Test that short words are kept with min_length=1."""
        result = processor_short_min_length.process("a big cat")
        # "a" is still stopword, but "big" and "cat" are kept
        assert "big" in result.keywords
        assert "cat" in result.keywords

    def test_extract_keywords_lowercase(self, processor):
        """Test that keywords are lowercased."""
        result = processor.process("Machine Learning Algorithms")
        assert "machine" in result.keywords
        assert "learning" in result.keywords

    def test_extract_keywords_deduplication(self, processor):
        """Test that duplicate keywords are removed."""
        result = processor.process("test test test query query")
        assert result.keywords.count("test") == 1
        assert result.keywords.count("query") == 1

    def test_extract_keywords_chinese(self, processor):
        """Test extracting Chinese keywords."""
        result = processor.process("机器学习是人工智能的分支")
        # Chinese characters are extracted individually
        assert len(result.keywords) > 0

    def test_extract_keywords_mixed_language(self, processor):
        """Test extracting mixed language keywords."""
        result = processor.process("machine learning 机器学习")
        assert "machine" in result.keywords
        assert "learning" in result.keywords

    def test_extract_keywords_max_limit(self):
        """Test that keyword count is limited."""
        processor = QueryProcessor(max_keywords=3)
        result = processor.process("one two three four five six seven")
        assert len(result.keywords) == 3

    def test_extract_keywords_empty_query(self, processor):
        """Test handling empty query."""
        result = processor.process("")
        assert result.keywords == []

    def test_extract_keywords_whitespace_only(self, processor):
        """Test handling whitespace-only query."""
        result = processor.process("   \n\t  ")
        assert result.keywords == []

    def test_extract_keywords_punctuation(self, processor):
        """Test handling punctuation."""
        result = processor.process("hello, world! how are you?")
        assert "hello" in result.keywords
        assert "world" in result.keywords

    def test_parse_filters_none(self, processor):
        """Test parsing None filters."""
        result = processor.process("test query", filters=None)
        assert result.filters == {}

    def test_parse_filters_empty_dict(self, processor):
        """Test parsing empty filters."""
        result = processor.process("test query", filters={})
        assert result.filters == {}

    def test_parse_filters_preserves_values(self, processor):
        """Test that filters are preserved."""
        filters = {
            "collection": "my_docs",
            "doc_type": "pdf",
        }
        result = processor.process("test query", filters=filters)
        assert result.filters["collection"] == "my_docs"
        assert result.filters["doc_type"] == "pdf"

    def test_parse_filters_removes_none_values(self, processor):
        """Test that None values are removed."""
        filters = {
            "collection": "my_docs",
            "doc_type": None,
        }
        result = processor.process("test query", filters=filters)
        assert "collection" in result.filters
        assert "doc_type" not in result.filters

    def test_process_with_trace(self, processor):
        """Test processing with trace context."""
        trace = TraceContext(trace_id="test-trace")
        result = processor.process("test query", trace=trace)

        assert result is not None
        assert len(trace.stages) == 1
        assert trace.stages[0]["stage"] == "query_processing"

    def test_elapsed_ms_recorded(self, processor):
        """Test that elapsed_ms is recorded."""
        result = processor.process("test query")
        assert result.elapsed_ms is not None
        assert result.elapsed_ms >= 0

    def test_get_stopwords(self, processor):
        """Test getting stopwords."""
        stopwords = processor.get_stopwords()
        assert isinstance(stopwords, set)
        assert "the" in stopwords

    def test_custom_stopwords(self):
        """Test using custom stopwords."""
        custom_stopwords = {"foo", "bar"}
        processor = QueryProcessor(stopwords=custom_stopwords)
        result = processor.process("foo baz bar qux")
        assert "foo" not in result.keywords
        assert "bar" not in result.keywords
        assert "baz" in result.keywords
        assert "qux" in result.keywords

    def test_no_stopwords(self, processor_no_stopwords):
        """Test with no stopwords."""
        result = processor_no_stopwords.process("the quick brown fox")
        assert "the" in result.keywords
        assert "quick" in result.keywords

    def test_numbers_in_query(self, processor):
        """Test handling numbers in query."""
        result = processor.process("python 3.11 release 2024")
        assert "python" in result.keywords
        # Numbers are extracted as tokens
        assert "3" in result.keywords or "11" in result.keywords

    def test_special_characters(self, processor):
        """Test handling special characters."""
        result = processor.process("API @endpoint #hashtag")
        assert "api" in result.keywords
        # Special chars are stripped
        assert "@" not in result.keywords
        assert "#" not in result.keywords


class TestFakeQueryProcessor:
    """Tests for FakeQueryProcessor."""

    def test_fake_processor_returns_default_keywords(self):
        """Test fake processor returns default keywords."""
        processor = FakeQueryProcessor()
        result = processor.process("any query")
        assert result.keywords == ["fake", "keywords"]

    def test_fake_processor_returns_custom_keywords(self):
        """Test fake processor returns custom keywords."""
        processor = FakeQueryProcessor(keywords=["custom", "words"])
        result = processor.process("any query")
        assert result.keywords == ["custom", "words"]

    def test_fake_processor_records_calls(self):
        """Test fake processor records calls."""
        processor = FakeQueryProcessor()
        processor.process("query 1", filters={"a": 1})
        processor.process("query 2", filters={"b": 2})

        assert len(processor.process_calls) == 2
        assert processor.process_calls[0]["query"] == "query 1"
        assert processor.process_calls[1]["query"] == "query 2"

    def test_fake_processor_preserves_filters(self):
        """Test fake processor preserves filters."""
        processor = FakeQueryProcessor()
        result = processor.process("query", filters={"collection": "test"})
        assert result.filters == {"collection": "test"}

    def test_fake_processor_elapsed_ms(self):
        """Test fake processor elapsed_ms."""
        processor = FakeQueryProcessor()
        result = processor.process("query")
        assert result.elapsed_ms == 0.1

    def test_fake_processor_get_stopwords(self):
        """Test fake processor get_stopwords."""
        processor = FakeQueryProcessor()
        stopwords = processor.get_stopwords()
        assert stopwords == set()


class TestQueryProcessorEdgeCases:
    """Edge case tests for QueryProcessor."""

    @pytest.fixture
    def processor(self):
        """Create default QueryProcessor."""
        return QueryProcessor()

    def test_very_long_query(self, processor):
        """Test handling very long query."""
        long_query = " ".join(["word"] * 1000)
        result = processor.process(long_query)
        # Should still work, but limited by max_keywords
        assert len(result.keywords) <= 20

    def test_query_with_only_stopwords(self, processor):
        """Test query with only stopwords."""
        result = processor.process("the a an is are was were")
        assert result.keywords == []

    def test_query_with_only_short_words(self, processor):
        """Test query with only short words."""
        result = processor.process("a b c d e")
        assert result.keywords == []

    def test_unicode_query(self, processor):
        """Test handling unicode query."""
        result = processor.process("hello 世界 🌍")
        assert "hello" in result.keywords

    def test_query_with_newlines(self, processor):
        """Test handling query with newlines."""
        result = processor.process("hello\nworld\ntest")
        assert "hello" in result.keywords
        assert "world" in result.keywords

    def test_query_with_tabs(self, processor):
        """Test handling query with tabs."""
        result = processor.process("hello\tworld\ttest")
        assert "hello" in result.keywords
        assert "world" in result.keywords

    def test_query_with_multiple_spaces(self, processor):
        """Test handling query with multiple spaces."""
        result = processor.process("hello    world     test")
        assert "hello" in result.keywords
        assert "world" in result.keywords

    def test_camel_case_query(self, processor):
        """Test handling camelCase query."""
        result = processor.process("machineLearning deepLearning")
        # CamelCase is split by regex
        assert "machine" in result.keywords or "machinelearning" in result.keywords

    def test_hyphenated_words(self, processor):
        """Test handling hyphenated words."""
        result = processor.process("state-of-the-art technology")
        # Hyphens are treated as separators
        assert "state" in result.keywords or "art" in result.keywords
