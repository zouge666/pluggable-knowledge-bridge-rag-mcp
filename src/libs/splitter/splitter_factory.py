"""
Splitter 工厂。

根据配置创建对应的 Splitter 实例。
"""

from typing import TYPE_CHECKING

from src.core.settings import Settings
from src.libs.splitter.base_splitter import BaseSplitter, SplitterError

if TYPE_CHECKING:
    pass


class SplitterFactory:
    """
    Splitter 工厂类。

    根据配置动态创建 Splitter 实例，支持多种切分策略。
    """

    @classmethod
    def create(cls, settings: Settings) -> BaseSplitter:
        """
        根据配置创建 Splitter 实例。

        Args:
            settings: 应用配置。

        Returns:
            BaseSplitter: Splitter 实例。

        Raises:
            SplitterError: 不支持的切分器类型或配置错误。
        """
        ingestion_settings = settings.ingestion
        splitter_type = ingestion_settings.splitter.lower()

        chunk_size = ingestion_settings.chunk_size
        overlap = ingestion_settings.chunk_overlap

        if splitter_type == "recursive":
            from src.libs.splitter.recursive_splitter import RecursiveSplitter

            return RecursiveSplitter(chunk_size=chunk_size, overlap=overlap)
        elif splitter_type == "semantic":
            from src.libs.splitter.semantic_splitter import SemanticSplitter

            return SemanticSplitter(chunk_size=chunk_size, overlap=overlap)
        elif splitter_type == "fixed":
            from src.libs.splitter.fixed_length_splitter import FixedLengthSplitter

            return FixedLengthSplitter(chunk_size=chunk_size, overlap=overlap)
        else:
            raise SplitterError(
                f"Unsupported Splitter type: {splitter_type}",
                splitter_type=splitter_type,
            )

    @classmethod
    def create_with_params(
        cls,
        splitter_type: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> BaseSplitter:
        """
        使用参数创建 Splitter 实例。

        Args:
            splitter_type: 切分器类型。
            chunk_size: 目标 chunk 大小。
            overlap: 重叠大小。

        Returns:
            BaseSplitter: Splitter 实例。
        """
        splitter_type = splitter_type.lower()

        if splitter_type == "recursive":
            from src.libs.splitter.recursive_splitter import RecursiveSplitter

            return RecursiveSplitter(chunk_size=chunk_size, overlap=overlap)
        elif splitter_type == "semantic":
            from src.libs.splitter.semantic_splitter import SemanticSplitter

            return SemanticSplitter(chunk_size=chunk_size, overlap=overlap)
        elif splitter_type == "fixed":
            from src.libs.splitter.fixed_length_splitter import FixedLengthSplitter

            return FixedLengthSplitter(chunk_size=chunk_size, overlap=overlap)
        else:
            raise SplitterError(
                f"Unsupported Splitter type: {splitter_type}",
                splitter_type=splitter_type,
            )

    @classmethod
    def get_supported_types(cls) -> list:
        """获取支持的切分器类型列表。"""
        return ["recursive", "semantic", "fixed"]
