"""
Image Storage - 图片文件存储与索引管理。

保存图片到 data/images/{collection}/，并使用 SQLite 记录 image_id→path 映射。
"""

import base64
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.core.trace.trace_context import TraceContext


@dataclass
class ImageInfo:
    """图片信息。"""

    image_id: str
    file_path: str
    collection: Optional[str] = None
    doc_hash: Optional[str] = None
    page_num: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典。"""
        return asdict(self)


class ImageStorage:
    """
    图片存储管理器。

    功能：
    1. 保存图片到文件系统
    2. 使用 SQLite 记录 image_id→path 映射
    3. 支持按 collection 查询
    4. 支持并发访问（WAL 模式）
    """

    def __init__(
        self,
        image_dir: str = "data/images",
        db_path: str = "data/db/image_index.db",
    ):
        """
        初始化 ImageStorage。

        Args:
            image_dir: 图片存储目录。
            db_path: 索引数据库路径。
        """
        self._image_dir = Path(image_dir)
        self._db_path = Path(db_path)

        # 确保目录存在
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self._init_db()

    def _init_db(self):
        """初始化数据库表。"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # 创建图片索引表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_index (
                image_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                collection TEXT,
                doc_hash TEXT,
                page_num INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)
        """)

        # 启用 WAL 模式
        cursor.execute("PRAGMA journal_mode=WAL")

        conn.commit()
        conn.close()

    def save(
        self,
        image_id: str,
        image_data: bytes,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> ImageInfo:
        """
        保存图片。

        Args:
            image_id: 图片 ID。
            image_data: 图片二进制数据。
            collection: 集合名称。
            doc_hash: 文档哈希。
            page_num: 页码。
            trace: 追踪上下文（可选）。

        Returns:
            ImageInfo: 图片信息。
        """
        start_time = time.time()

        # 确定存储路径
        collection_dir = self._image_dir / (collection or "default")
        collection_dir.mkdir(parents=True, exist_ok=True)

        # 确定文件扩展名（默认 .png）
        file_name = f"{image_id}.png"
        file_path = collection_dir / file_name

        # 保存图片文件
        with open(file_path, "wb") as f:
            f.write(image_data)

        # 记录到数据库
        created_at = datetime.now().isoformat()
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO image_index
            (image_id, file_path, collection, doc_hash, page_num, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (image_id, str(file_path), collection, doc_hash, page_num, created_at))
        conn.commit()
        conn.close()

        elapsed_ms = (time.time() - start_time) * 1000

        if trace:
            trace.record_stage(
                stage_name="image_storage",
                elapsed_ms=elapsed_ms,
                method="file",
                details={
                    "image_id": image_id,
                    "collection": collection,
                },
            )

        return ImageInfo(
            image_id=image_id,
            file_path=str(file_path),
            collection=collection,
            doc_hash=doc_hash,
            page_num=page_num,
            created_at=created_at,
        )

    def save_from_base64(
        self,
        image_id: str,
        base64_data: str,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> ImageInfo:
        """
        从 base64 数据保存图片。

        Args:
            image_id: 图片 ID。
            base64_data: base64 编码的图片数据。
            collection: 集合名称。
            doc_hash: 文档哈希。
            page_num: 页码。
            trace: 追踪上下文（可选）。

        Returns:
            ImageInfo: 图片信息。
        """
        image_data = base64.b64decode(base64_data)
        return self.save(image_id, image_data, collection, doc_hash, page_num, trace)

    def get(self, image_id: str) -> Optional[ImageInfo]:
        """
        获取图片信息。

        Args:
            image_id: 图片 ID。

        Returns:
            Optional[ImageInfo]: 图片信息，不存在返回 None。
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT image_id, file_path, collection, doc_hash, page_num, created_at
            FROM image_index
            WHERE image_id = ?
        """, (image_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return ImageInfo(
            image_id=row[0],
            file_path=row[1],
            collection=row[2],
            doc_hash=row[3],
            page_num=row[4],
            created_at=row[5],
        )

    def get_path(self, image_id: str) -> Optional[str]:
        """
        获取图片文件路径。

        Args:
            image_id: 图片 ID。

        Returns:
            Optional[str]: 文件路径，不存在返回 None。
        """
        info = self.get(image_id)
        return info.file_path if info else None

    def load(self, image_id: str) -> Optional[bytes]:
        """
        加载图片数据。

        Args:
            image_id: 图片 ID。

        Returns:
            Optional[bytes]: 图片二进制数据，不存在返回 None。
        """
        path = self.get_path(image_id)
        if path is None or not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            return f.read()

    def load_as_base64(self, image_id: str) -> Optional[str]:
        """
        加载图片数据为 base64。

        Args:
            image_id: 图片 ID。

        Returns:
            Optional[str]: base64 编码的图片数据，不存在返回 None。
        """
        data = self.load(image_id)
        if data is None:
            return None
        return base64.b64encode(data).decode("utf-8")

    def list_by_collection(self, collection: str) -> List[ImageInfo]:
        """
        按集合列出图片。

        Args:
            collection: 集合名称。

        Returns:
            List[ImageInfo]: 图片信息列表。
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT image_id, file_path, collection, doc_hash, page_num, created_at
            FROM image_index
            WHERE collection = ?
        """, (collection,))
        rows = cursor.fetchall()
        conn.close()

        return [
            ImageInfo(
                image_id=row[0],
                file_path=row[1],
                collection=row[2],
                doc_hash=row[3],
                page_num=row[4],
                created_at=row[5],
            )
            for row in rows
        ]

    def list_by_doc_hash(self, doc_hash: str) -> List[ImageInfo]:
        """
        按文档哈希列出图片。

        Args:
            doc_hash: 文档哈希。

        Returns:
            List[ImageInfo]: 图片信息列表。
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT image_id, file_path, collection, doc_hash, page_num, created_at
            FROM image_index
            WHERE doc_hash = ?
        """, (doc_hash,))
        rows = cursor.fetchall()
        conn.close()

        return [
            ImageInfo(
                image_id=row[0],
                file_path=row[1],
                collection=row[2],
                doc_hash=row[3],
                page_num=row[4],
                created_at=row[5],
            )
            for row in rows
        ]

    def delete(self, image_id: str) -> bool:
        """
        删除图片。

        Args:
            image_id: 图片 ID。

        Returns:
            bool: 是否成功删除。
        """
        # 获取图片信息
        info = self.get(image_id)
        if info is None:
            return False

        # 删除文件
        if os.path.exists(info.file_path):
            os.remove(info.file_path)

        # 删除数据库记录
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM image_index WHERE image_id = ?", (image_id,))
        conn.commit()
        conn.close()

        return True

    def delete_by_collection(self, collection: str) -> int:
        """
        删除集合中的所有图片。

        Args:
            collection: 集合名称。

        Returns:
            int: 删除的图片数量。
        """
        # 获取图片列表
        images = self.list_by_collection(collection)

        # 删除文件和记录
        for info in images:
            if os.path.exists(info.file_path):
                os.remove(info.file_path)

        # 批量删除数据库记录
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM image_index WHERE collection = ?", (collection,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count

    def exists(self, image_id: str) -> bool:
        """
        检查图片是否存在。

        Args:
            image_id: 图片 ID。

        Returns:
            bool: 是否存在。
        """
        return self.get(image_id) is not None

    def count(self, collection: Optional[str] = None) -> int:
        """
        统计图片数量。

        Args:
            collection: 集合名称（可选）。

        Returns:
            int: 图片数量。
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        if collection:
            cursor.execute("SELECT COUNT(*) FROM image_index WHERE collection = ?", (collection,))
        else:
            cursor.execute("SELECT COUNT(*) FROM image_index")

        count = cursor.fetchone()[0]
        conn.close()

        return count


class FakeImageStorage:
    """
    Fake Image Storage 用于测试。

    不依赖真实文件系统，使用内存存储。
    """

    def __init__(self):
        """初始化 FakeImageStorage。"""
        self._images: Dict[str, ImageInfo] = {}
        self._image_data: Dict[str, bytes] = {}

    def save(
        self,
        image_id: str,
        image_data: bytes,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> ImageInfo:
        """保存图片。"""
        created_at = datetime.now().isoformat()
        info = ImageInfo(
            image_id=image_id,
            file_path=f"/fake/{collection or 'default'}/{image_id}.png",
            collection=collection,
            doc_hash=doc_hash,
            page_num=page_num,
            created_at=created_at,
        )
        self._images[image_id] = info
        self._image_data[image_id] = image_data
        return info

    def save_from_base64(
        self,
        image_id: str,
        base64_data: str,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> ImageInfo:
        """从 base64 数据保存图片。"""
        image_data = base64.b64decode(base64_data)
        return self.save(image_id, image_data, collection, doc_hash, page_num, trace)

    def get(self, image_id: str) -> Optional[ImageInfo]:
        """获取图片信息。"""
        return self._images.get(image_id)

    def get_path(self, image_id: str) -> Optional[str]:
        """获取图片文件路径。"""
        info = self.get(image_id)
        return info.file_path if info else None

    def load(self, image_id: str) -> Optional[bytes]:
        """加载图片数据。"""
        return self._image_data.get(image_id)

    def load_as_base64(self, image_id: str) -> Optional[str]:
        """加载图片数据为 base64。"""
        data = self.load(image_id)
        if data is None:
            return None
        return base64.b64encode(data).decode("utf-8")

    def list_by_collection(self, collection: str) -> List[ImageInfo]:
        """按集合列出图片。"""
        return [info for info in self._images.values() if info.collection == collection]

    def list_by_doc_hash(self, doc_hash: str) -> List[ImageInfo]:
        """按文档哈希列出图片。"""
        return [info for info in self._images.values() if info.doc_hash == doc_hash]

    def delete(self, image_id: str) -> bool:
        """删除图片。"""
        if image_id not in self._images:
            return False
        del self._images[image_id]
        if image_id in self._image_data:
            del self._image_data[image_id]
        return True

    def delete_by_collection(self, collection: str) -> int:
        """删除集合中的所有图片。"""
        to_delete = [image_id for image_id, info in self._images.items() if info.collection == collection]
        for image_id in to_delete:
            self.delete(image_id)
        return len(to_delete)

    def exists(self, image_id: str) -> bool:
        """检查图片是否存在。"""
        return image_id in self._images

    def count(self, collection: Optional[str] = None) -> int:
        """统计图片数量。"""
        if collection:
            return len([info for info in self._images.values() if info.collection == collection])
        return len(self._images)

    def clear(self):
        """清空所有图片。"""
        self._images.clear()
        self._image_data.clear()
