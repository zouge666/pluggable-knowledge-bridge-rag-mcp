"""
Splitter Factory 单元测试。

使用 Fake provider 验证工厂路由逻辑。
"""

import pytest

from src.core.settings import IngestionSettings, Settings
from src.libs.splitter.base_splitter import BaseSplitter, SplitResult, SplitterError
from src.libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    """Fake Splitter 实现，用于测试。"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def split_text(self, text, trace=None):
        # 简单按换行切分
        chunks = text.split("\n") if text else []
        return SplitResult(
            chunks=chunks,
            splitter_type="fake",
            chunk_size=self._chunk_size,
            overlap=self._overlap,
        )

    def get_splitter_type(self) -> str:
        return "fake"

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def get_overlap(self) -> int:
        return self._overlap


class TestSplitterFactory:
    """SplitterFactory 测试。"""

    def test_get_supported_types(self):
        """应该返回支持的切分器类型列表。"""
        types = SplitterFactory.get_supported_types()

        assert "recursive" in types
        assert "semantic" in types
        assert "fixed" in types

    def test_unsupported_type_raises_error(self):
        """不支持的切分器类型应该抛出错误。"""
        settings = Settings(
            ingestion=IngestionSettings(
                splitter="unsupported_type",
                chunk_size=1000,
                chunk_overlap=200,
                batch_size=100,
            )
        )

        with pytest.raises(SplitterError) as exc_info:
            SplitterFactory.create(settings)

        assert "Unsupported Splitter type" in str(exc_info.value)
        assert exc_info.value.splitter_type == "unsupported_type"

    def test_create_recursive_splitter(self):
        """应该能创建 Recursive Splitter。"""
        settings = Settings(
            ingestion=IngestionSettings(
                splitter="recursive",
                chunk_size=500,
                chunk_overlap=100,
                batch_size=100,
            )
        )

        splitter = SplitterFactory.create(settings)

        assert splitter is not None
        assert splitter.get_splitter_type() == "recursive"
        assert splitter.get_chunk_size() == 500
        assert splitter.get_overlap() == 100

    def test_create_recursive_splitter_case_insensitive(self):
        """切分器类型应该不区分大小写。"""
        settings = Settings(
            ingestion=IngestionSettings(
                splitter="Recursive",  # 大小写混合
                chunk_size=1000,
                chunk_overlap=200,
                batch_size=100,
            )
        )

        splitter = SplitterFactory.create(settings)

        assert splitter is not None
        assert splitter.get_splitter_type() == "recursive"

    def test_create_semantic_splitter(self):
        """应该能创建 Semantic Splitter。"""
        settings = Settings(
            ingestion=IngestionSettings(
                splitter="semantic",
                chunk_size=800,
                chunk_overlap=150,
                batch_size=100,
            )
        )

        splitter = SplitterFactory.create(settings)

        assert splitter is not None
        assert splitter.get_splitter_type() == "semantic"
        assert splitter.get_chunk_size() == 800

    def test_create_fixed_splitter(self):
        """应该能创建 Fixed Length Splitter。"""
        settings = Settings(
            ingestion=IngestionSettings(
                splitter="fixed",
                chunk_size=600,
                chunk_overlap=100,
                batch_size=100,
            )
        )

        splitter = SplitterFactory.create(settings)

        assert splitter is not None
        assert splitter.get_splitter_type() == "fixed"
        assert splitter.get_chunk_size() == 600

    def test_create_with_params(self):
        """应该能使用参数创建 Splitter。"""
        splitter = SplitterFactory.create_with_params(
            splitter_type="recursive",
            chunk_size=300,
            overlap=50,
        )

        assert splitter is not None
        assert splitter.get_chunk_size() == 300
        assert splitter.get_overlap() == 50


class TestBaseSplitter:
    """BaseSplitter 测试。"""

    def test_split_result_to_dict(self):
        """SplitResult 应该能转换为字典。"""
        result = SplitResult(
            chunks=["chunk1", "chunk2", "chunk3"],
            splitter_type="recursive",
            chunk_size=1000,
            overlap=200,
            metadata={"input_length": 2500},
        )

        data = result.to_dict()

        assert data["splitter_type"] == "recursive"
        assert data["chunk_size"] == 1000
        assert data["overlap"] == 200
        assert data["chunk_count"] == 3
        assert data["metadata"]["input_length"] == 2500

    def test_split_result_to_dict_minimal(self):
        """SplitResult 最小字段应该能转换。"""
        result = SplitResult(
            chunks=["chunk1"],
            splitter_type="fixed",
            chunk_size=500,
            overlap=0,
        )

        data = result.to_dict()

        assert data["splitter_type"] == "fixed"
        assert data["chunk_size"] == 500
        assert data["overlap"] == 0
        assert data["chunk_count"] == 1
        assert "metadata" not in data


class TestFixedLengthSplitter:
    """FixedLengthSplitter 测试。"""

    def test_split_empty_text(self):
        """空文本应该返回空结果。"""
        splitter = SplitterFactory.create_with_params("fixed", chunk_size=100, overlap=0)

        result = splitter.split_text("")

        assert result.chunks == []

    def test_split_short_text(self):
        """短文本应该返回单个 chunk。"""
        splitter = SplitterFactory.create_with_params("fixed", chunk_size=100, overlap=0)

        result = splitter.split_text("Short text")

        assert len(result.chunks) == 1
        assert result.chunks[0] == "Short text"

    def test_split_long_text(self):
        """长文本应该被切分为多个 chunk。"""
        splitter = SplitterFactory.create_with_params("fixed", chunk_size=10, overlap=0)

        result = splitter.split_text("This is a longer text that needs splitting")

        assert len(result.chunks) > 1
        for chunk in result.chunks:
            assert len(chunk) <= 10

    def test_split_with_overlap(self):
        """有重叠的切分应该正确工作。"""
        splitter = SplitterFactory.create_with_params("fixed", chunk_size=10, overlap=3)

        result = splitter.split_text("01234567890123456789")

        assert len(result.chunks) > 1
        # 检查重叠
        if len(result.chunks) > 1:
            # 第一个 chunk 的末尾应该与第二个 chunk 的开头有重叠
            assert result.chunks[0][-3:] == result.chunks[1][:3]


class TestSplitterError:
    """SplitterError 测试。"""

    def test_splitter_error_basic(self):
        """基本错误信息。"""
        error = SplitterError("Something went wrong")

        assert "Something went wrong" in str(error)

    def test_splitter_error_with_type(self):
        """带切分器类型的错误信息。"""
        error = SplitterError("Something went wrong", splitter_type="recursive")

        assert "[recursive]" in str(error)
        assert "Something went wrong" in str(error)

    def test_splitter_error_subclasses(self):
        """错误子类应该继承 SplitterError。"""
        from src.libs.splitter.base_splitter import (
            SplitterConfigError,
            SplitterProcessingError,
        )

        config_error = SplitterConfigError("Config error", splitter_type="semantic")
        proc_error = SplitterProcessingError("Processing error", splitter_type="fixed")

        assert isinstance(config_error, SplitterError)
        assert isinstance(proc_error, SplitterError)
