"""
Core 层导出。

统一 re-export 以简化导入路径。
"""

from src.core.settings import Settings, load_settings, validate_settings
from src.core.types import (
    Document,
    Chunk,
    ChunkRecord,
    RetrievalResult,
    ImageRef,
    compute_sha256,
    compute_file_hash,
    generate_chunk_id,
    generate_image_id,
)

__all__ = [
    # Settings
    "Settings",
    "load_settings",
    "validate_settings",
    # Types
    "Document",
    "Chunk",
    "ChunkRecord",
    "RetrievalResult",
    "ImageRef",
    # Utils
    "compute_sha256",
    "compute_file_hash",
    "generate_chunk_id",
    "generate_image_id",
]