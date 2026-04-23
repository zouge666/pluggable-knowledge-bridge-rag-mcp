"""
Loader 抽象基类。

定义文档加载器的统一接口。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from src.core.types import Document


class BaseLoader(ABC):
    """
    文档加载器抽象基类。

    所有文档加载器（PDF/Markdown/HTML 等）都必须实现此接口。
    """

    @abstractmethod
    def load(
        self,
        path: Union[str, Path],
        collection: Optional[str] = None,
    ) -> Document:
        """
        加载文档并返回 Document 对象。

        Args:
            path: 文档路径。
            collection: 集合名称（可选）。

        Returns:
            Document: 文档对象。

        Raises:
            FileNotFoundError: 文件不存在。
            LoaderError: 加载失败。
        """
        pass

    @abstractmethod
    def supports(self, path: Union[str, Path]) -> bool:
        """
        判断是否支持该文件类型。

        Args:
            path: 文件路径。

        Returns:
            bool: True 表示支持，False 表示不支持。
        """
        pass

    def _get_file_extension(self, path: Union[str, Path]) -> str:
        """获取文件扩展名（小写）。"""
        return Path(path).suffix.lower()


class LoaderError(Exception):
    """文档加载错误基类。"""

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.path = path
        self.original_error = original_error


class UnsupportedFormatError(LoaderError):
    """不支持的文件格式错误。"""
    pass


class ParsingError(LoaderError):
    """解析错误。"""
    pass
