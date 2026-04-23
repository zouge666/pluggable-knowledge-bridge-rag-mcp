"""
文件完整性检查模块。

计算文件 hash，提供"是否跳过"的判定接口。
使用 SQLite 作为默认存储，支持后续替换为 Redis/PostgreSQL。
"""

import hashlib
import json
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional


class FileIntegrityChecker(ABC):
    """
    文件完整性检查器抽象基类。

    定义文件 hash 计算和状态管理的接口。
    """

    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        """
        计算文件的 SHA256 哈希值。

        Args:
            path: 文件路径。

        Returns:
            str: SHA256 哈希值（十六进制字符串）。
        """
        pass

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """
        判断文件是否应该跳过（已成功处理过）。

        Args:
            file_hash: 文件哈希值。

        Returns:
            bool: True 表示应该跳过，False 表示需要处理。
        """
        pass

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        collection: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        标记文件处理成功。

        Args:
            file_hash: 文件哈希值。
            file_path: 文件路径。
            collection: 集合名称（可选）。
            metadata: 其他元数据（可选）。
        """
        pass

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """
        标记文件处理失败。

        Args:
            file_hash: 文件哈希值。
            error_msg: 错误信息。
        """
        pass

    @abstractmethod
    def get_record(self, file_hash: str) -> Optional[dict]:
        """
        获取文件处理记录。

        Args:
            file_hash: 文件哈希值。

        Returns:
            Optional[dict]: 处理记录，不存在返回 None。
        """
        pass

    @abstractmethod
    def remove_record(self, file_hash: str) -> bool:
        """
        删除文件处理记录。

        Args:
            file_hash: 文件哈希值。

        Returns:
            bool: True 表示删除成功，False 表示记录不存在。
        """
        pass

    @abstractmethod
    def list_processed(
        self,
        collection: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list:
        """
        列出已处理的文件记录。

        Args:
            collection: 集合名称（可选，None 表示所有集合）。
            status: 状态过滤（可选，"success" 或 "failed"）。

        Returns:
            list: 处理记录列表。
        """
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """
    基于 SQLite 的文件完整性检查器。

    默认实现，支持并发写入（WAL 模式）。
    """

    def __init__(
        self,
        db_path: str = "data/db/ingestion_history.db",
        enable_wal: bool = True,
    ):
        """
        初始化 SQLite 完整性检查器。

        Args:
            db_path: 数据库文件路径。
            enable_wal: 是否启用 WAL 模式（支持并发写入）。
        """
        self.db_path = Path(db_path)
        self.enable_wal = enable_wal
        self._conn: Optional[sqlite3.Connection] = None

        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接。"""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

            # 启用 WAL 模式（支持并发写入）
            if self.enable_wal:
                self._conn.execute("PRAGMA journal_mode=WAL")

        return self._conn

    def _init_db(self) -> None:
        """初始化数据库表。"""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_history (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                collection TEXT,
                status TEXT NOT NULL DEFAULT 'success',
                error_msg TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_collection
            ON ingestion_history(collection)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status
            ON ingestion_history(status)
        """)

        conn.commit()

    def compute_sha256(self, path: str) -> str:
        """
        计算文件的 SHA256 哈希值。

        使用流式读取，支持大文件。

        Args:
            path: 文件路径。

        Returns:
            str: SHA256 哈希值（十六进制字符串）。
        """
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        """
        判断文件是否应该跳过（已成功处理过）。

        Args:
            file_hash: 文件哈希值。

        Returns:
            bool: True 表示应该跳过，False 表示需要处理。
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT status FROM ingestion_history WHERE file_hash = ?",
            (file_hash,),
        )
        row = cursor.fetchone()
        return row is not None and row["status"] == "success"

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        collection: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        标记文件处理成功。

        Args:
            file_hash: 文件哈希值。
            file_path: 文件路径。
            collection: 集合名称（可选）。
            metadata: 其他元数据（可选）。
        """
        conn = self._get_connection()
        metadata_json = json.dumps(metadata) if metadata else None

        conn.execute("""
            INSERT INTO ingestion_history (file_hash, file_path, collection, status, metadata, updated_at)
            VALUES (?, ?, ?, 'success', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(file_hash) DO UPDATE SET
                file_path = excluded.file_path,
                collection = excluded.collection,
                status = 'success',
                error_msg = NULL,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
        """, (file_hash, file_path, collection, metadata_json))

        conn.commit()

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """
        标记文件处理失败。

        Args:
            file_hash: 文件哈希值。
            error_msg: 错误信息。
        """
        conn = self._get_connection()

        conn.execute("""
            INSERT INTO ingestion_history (file_hash, file_path, status, error_msg, updated_at)
            VALUES (?, '', 'failed', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(file_hash) DO UPDATE SET
                status = 'failed',
                error_msg = excluded.error_msg,
                updated_at = CURRENT_TIMESTAMP
        """, (file_hash, error_msg))

        conn.commit()

    def get_record(self, file_hash: str) -> Optional[dict]:
        """
        获取文件处理记录。

        Args:
            file_hash: 文件哈希值。

        Returns:
            Optional[dict]: 处理记录，不存在返回 None。
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM ingestion_history WHERE file_hash = ?",
            (file_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        result = dict(row)
        if result.get("metadata"):
            result["metadata"] = json.loads(result["metadata"])
        return result

    def remove_record(self, file_hash: str) -> bool:
        """
        删除文件处理记录。

        Args:
            file_hash: 文件哈希值。

        Returns:
            bool: True 表示删除成功，False 表示记录不存在。
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM ingestion_history WHERE file_hash = ?",
            (file_hash,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def list_processed(
        self,
        collection: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list:
        """
        列出已处理的文件记录。

        Args:
            collection: 集合名称（可选，None 表示所有集合）。
            status: 状态过滤（可选，"success" 或 "failed"）。

        Returns:
            list: 处理记录列表。
        """
        conn = self._get_connection()

        query = "SELECT * FROM ingestion_history WHERE 1=1"
        params = []

        if collection is not None:
            query += " AND collection = ?"
            params.append(collection)

        if status is not None:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"

        cursor = conn.execute(query, params)
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("metadata"):
                result["metadata"] = json.loads(result["metadata"])
            results.append(result)

        return results

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteIntegrityChecker":
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口。"""
        self.close()
