"""
文件完整性检查单元测试。

验证 SQLiteIntegrityChecker 的 hash 计算、状态管理和并发支持。
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
)


class TestSQLiteIntegrityChecker:
    """SQLiteIntegrityChecker 测试。"""

    def test_init_creates_db(self, tmp_path):
        """初始化应该创建数据库文件。"""
        db_path = tmp_path / "test_history.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        assert db_path.exists()
        checker.close()

    def test_init_creates_directory(self, tmp_path):
        """初始化应该创建目录结构。"""
        db_path = tmp_path / "nested" / "dir" / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        assert db_path.parent.exists()
        assert db_path.exists()
        checker.close()

    def test_compute_sha256_consistent(self, tmp_path):
        """同一文件多次计算 hash 结果一致。"""
        # 创建测试文件
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        hash1 = checker.compute_sha256(str(file_path))
        hash2 = checker.compute_sha256(str(file_path))

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 输出 64 个十六进制字符
        checker.close()

    def test_compute_sha256_different_files(self, tmp_path):
        """不同文件应该产生不同 hash。"""
        file1 = tmp_path / "file1.txt"
        file1.write_text("content 1")
        file2 = tmp_path / "file2.txt"
        file2.write_text("content 2")

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        hash1 = checker.compute_sha256(str(file1))
        hash2 = checker.compute_sha256(str(file2))

        assert hash1 != hash2
        checker.close()

    def test_compute_sha256_large_file(self, tmp_path):
        """应该能计算大文件的 hash（流式读取）。"""
        file_path = tmp_path / "large.bin"
        # 创建 1MB 文件
        file_path.write_bytes(os.urandom(1024 * 1024))

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        hash1 = checker.compute_sha256(str(file_path))
        hash2 = checker.compute_sha256(str(file_path))

        assert hash1 == hash2
        checker.close()

    def test_should_skip_returns_false_for_new_file(self, tmp_path):
        """新文件（未处理过）should_skip 返回 False。"""
        file_path = tmp_path / "new.txt"
        file_path.write_text("new content")

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        file_hash = checker.compute_sha256(str(file_path))
        result = checker.should_skip(file_hash)

        assert result is False
        checker.close()

    def test_should_skip_returns_true_after_mark_success(self, tmp_path):
        """标记 success 后，should_skip 返回 True。"""
        file_path = tmp_path / "processed.txt"
        file_path.write_text("processed content")

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        file_hash = checker.compute_sha256(str(file_path))

        # 标记成功
        checker.mark_success(file_hash, str(file_path), collection="test")

        # 应该跳过
        result = checker.should_skip(file_hash)
        assert result is True
        checker.close()

    def test_should_skip_returns_false_after_mark_failed(self, tmp_path):
        """标记 failed 后，should_skip 返回 False（需要重新处理）。"""
        file_path = tmp_path / "failed.txt"
        file_path.write_text("failed content")

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        file_hash = checker.compute_sha256(str(file_path))

        # 标记失败
        checker.mark_failed(file_hash, "Processing error")

        # 不应该跳过
        result = checker.should_skip(file_hash)
        assert result is False
        checker.close()

    def test_mark_success_stores_metadata(self, tmp_path):
        """mark_success 应该存储元数据。"""
        file_path = tmp_path / "meta.txt"
        file_path.write_text("content")

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        file_hash = checker.compute_sha256(str(file_path))
        metadata = {"chunk_count": 10, "image_count": 2}

        checker.mark_success(
            file_hash,
            str(file_path),
            collection="test_collection",
            metadata=metadata,
        )

        record = checker.get_record(file_hash)
        assert record is not None
        assert record["collection"] == "test_collection"
        assert record["metadata"] == metadata
        checker.close()

    def test_mark_failed_stores_error_msg(self, tmp_path):
        """mark_failed 应该存储错误信息。"""
        file_path = tmp_path / "error.txt"
        file_path.write_text("content")

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        file_hash = checker.compute_sha256(str(file_path))
        error_msg = "PDF parsing failed: corrupted file"

        checker.mark_failed(file_hash, error_msg)

        record = checker.get_record(file_hash)
        assert record is not None
        assert record["status"] == "failed"
        assert record["error_msg"] == error_msg
        checker.close()

    def test_get_record_returns_none_for_unknown_hash(self, tmp_path):
        """未知 hash 的 get_record 返回 None。"""
        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        record = checker.get_record("unknown_hash_12345")
        assert record is None
        checker.close()

    def test_remove_record_success(self, tmp_path):
        """应该能删除记录。"""
        file_path = tmp_path / "remove.txt"
        file_path.write_text("content")

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        file_hash = checker.compute_sha256(str(file_path))
        checker.mark_success(file_hash, str(file_path))

        # 删除记录
        result = checker.remove_record(file_hash)
        assert result is True

        # 记录不存在
        record = checker.get_record(file_hash)
        assert record is None
        checker.close()

    def test_remove_record_returns_false_for_unknown(self, tmp_path):
        """删除未知记录返回 False。"""
        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        result = checker.remove_record("unknown_hash")
        assert result is False
        checker.close()

    def test_list_processed_all(self, tmp_path):
        """应该能列出所有处理记录。"""
        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        # 创建多个文件并标记
        for i in range(3):
            file_path = tmp_path / f"file{i}.txt"
            file_path.write_text(f"content {i}")
            file_hash = checker.compute_sha256(str(file_path))
            checker.mark_success(file_hash, str(file_path), collection=f"col{i}")

        records = checker.list_processed()
        assert len(records) == 3
        checker.close()

    def test_list_processed_by_collection(self, tmp_path):
        """应该能按集合过滤记录。"""
        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        # 创建文件并标记到不同集合
        for i in range(3):
            file_path = tmp_path / f"file{i}.txt"
            file_path.write_text(f"content {i}")
            file_hash = checker.compute_sha256(str(file_path))
            checker.mark_success(file_hash, str(file_path), collection="collection_a")

        for i in range(3, 5):
            file_path = tmp_path / f"file{i}.txt"
            file_path.write_text(f"content {i}")
            file_hash = checker.compute_sha256(str(file_path))
            checker.mark_success(file_hash, str(file_path), collection="collection_b")

        # 按集合过滤
        records_a = checker.list_processed(collection="collection_a")
        records_b = checker.list_processed(collection="collection_b")

        assert len(records_a) == 3
        assert len(records_b) == 2
        checker.close()

    def test_list_processed_by_status(self, tmp_path):
        """应该能按状态过滤记录。"""
        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        # 创建成功和失败的记录
        for i in range(2):
            file_path = tmp_path / f"success{i}.txt"
            file_path.write_text(f"success {i}")
            file_hash = checker.compute_sha256(str(file_path))
            checker.mark_success(file_hash, str(file_path))

        for i in range(2):
            file_path = tmp_path / f"failed{i}.txt"
            file_path.write_text(f"failed {i}")
            file_hash = checker.compute_sha256(str(file_path))
            checker.mark_failed(file_hash, f"error {i}")

        success_records = checker.list_processed(status="success")
        failed_records = checker.list_processed(status="failed")

        assert len(success_records) == 2
        assert len(failed_records) == 2
        checker.close()

    def test_update_existing_record(self, tmp_path):
        """更新已存在的记录应该覆盖旧值。"""
        file_path = tmp_path / "update.txt"
        file_path.write_text("original")

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        file_hash = checker.compute_sha256(str(file_path))

        # 第一次标记
        checker.mark_success(file_hash, str(file_path), collection="old_collection")

        # 更新记录
        checker.mark_success(file_hash, str(file_path), collection="new_collection")

        record = checker.get_record(file_hash)
        assert record["collection"] == "new_collection"
        checker.close()

    def test_mark_success_after_failure(self, tmp_path):
        """失败后重新成功应该更新状态。"""
        file_path = tmp_path / "retry.txt"
        file_path.write_text("retry content")

        db_path = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        file_hash = checker.compute_sha256(str(file_path))

        # 先标记失败
        checker.mark_failed(file_hash, "first attempt failed")

        # 重新标记成功
        checker.mark_success(file_hash, str(file_path))

        record = checker.get_record(file_hash)
        assert record["status"] == "success"
        assert record["error_msg"] is None
        checker.close()

    def test_context_manager(self, tmp_path):
        """应该支持上下文管理器。"""
        db_path = tmp_path / "test.db"

        with SQLiteIntegrityChecker(db_path=str(db_path)) as checker:
            file_path = tmp_path / "ctx.txt"
            file_path.write_text("context test")
            file_hash = checker.compute_sha256(str(file_path))
            checker.mark_success(file_hash, str(file_path))

            assert checker.should_skip(file_hash) is True

        # 连接已关闭（无法直接验证，但不应抛异常）

    def test_wal_mode_enabled(self, tmp_path):
        """应该启用 WAL 模式。"""
        db_path = tmp_path / "wal.db"
        checker = SQLiteIntegrityChecker(db_path=str(db_path), enable_wal=True)

        # 检查 WAL 模式
        conn = checker._get_connection()
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]

        assert mode.lower() == "wal"
        checker.close()


class TestFileIntegrityCheckerAbstract:
    """FileIntegrityChecker 抽象类测试。"""

    def test_abstract_methods(self):
        """抽象类不能直接实例化。"""
        with pytest.raises(TypeError):
            FileIntegrityChecker()