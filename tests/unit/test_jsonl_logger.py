"""
Unit tests for JSON Lines logger enhancements.
"""

import pytest
import json
import logging
import tempfile
from pathlib import Path

from src.observability.logger import (
    JSONFormatter,
    get_logger,
    get_trace_logger,
    write_trace,
    TraceLoggerAdapter,
)


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_format_basic_record(self):
        """Test formatting a basic log record."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert log_dict["level"] == "INFO"
        assert log_dict["logger"] == "test.logger"
        assert log_dict["message"] == "Test message"
        assert "timestamp" in log_dict

    def test_format_with_trace_id(self):
        """Test formatting with trace_id."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.trace_id = "abc123"

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert log_dict["trace_id"] == "abc123"

    def test_format_with_trace_type(self):
        """Test formatting with trace_type."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.trace_type = "query"

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert log_dict["trace_type"] == "query"

    def test_format_with_stage(self):
        """Test formatting with stage."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.stage = "dense_retrieval"

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert log_dict["stage"] == "dense_retrieval"

    def test_format_with_elapsed_ms(self):
        """Test formatting with elapsed_ms."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.elapsed_ms = 50.5

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert log_dict["elapsed_ms"] == 50.5

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = logging.sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert "exception" in log_dict
        assert "ValueError: Test error" in log_dict["exception"]

    def test_format_chinese_message(self):
        """Test formatting Chinese message."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="测试消息",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert log_dict["message"] == "测试消息"


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_logger_instance(self):
        """Test returns a logger instance."""
        logger = get_logger("test.logger")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handler(self):
        """Test logger has a handler."""
        logger = get_logger("test.unique.name")
        assert len(logger.handlers) > 0

    def test_logger_level_default(self):
        """Test logger default level is INFO."""
        logger = get_logger("test.logger.default")
        assert logger.level == logging.INFO

    def test_logger_level_custom(self):
        """Test logger with custom level."""
        logger = get_logger("test.logger.debug", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_logger_writes_to_stderr(self, capsys):
        """Test logger writes to stderr."""
        logger = get_logger("test.logger.stderr")
        logger.info("Test message to stderr")

        captured = capsys.readouterr()
        assert "Test message to stderr" in captured.err


class TestGetTraceLogger:
    """Tests for get_trace_logger."""

    def test_returns_logger_with_json_formatter(self):
        """Test returns logger with JSON formatter."""
        logger = get_trace_logger("test.trace.logger")
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) > 0

        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_trace_logger_level_default(self):
        """Test trace logger default level is INFO."""
        logger = get_trace_logger("test.trace.default")
        assert logger.level == logging.INFO

    def test_trace_logger_level_custom(self):
        """Test trace logger with custom level."""
        logger = get_trace_logger("test.trace.debug", level="DEBUG")
        assert logger.level == logging.DEBUG


class TestWriteTrace:
    """Tests for write_trace."""

    def test_write_trace_creates_file(self):
        """Test write_trace creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces/test.jsonl"
            trace_dict = {"trace_id": "abc123", "trace_type": "query"}

            write_trace(trace_dict, output_path)

            assert Path(output_path).exists()

    def test_write_trace_appends_json_line(self):
        """Test write_trace appends JSON line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            trace_dict = {"trace_id": "abc123", "trace_type": "query"}

            write_trace(trace_dict, output_path)
            write_trace(trace_dict, output_path)

            with open(output_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 2

    def test_write_trace_valid_json(self):
        """Test write_trace writes valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            trace_dict = {
                "trace_id": "abc123",
                "trace_type": "query",
                "stages": [{"stage": "test", "elapsed_ms": 10.5}],
            }

            write_trace(trace_dict, output_path)

            with open(output_path, "r") as f:
                line = f.readline()

            parsed = json.loads(line)
            assert parsed["trace_id"] == "abc123"
            assert parsed["trace_type"] == "query"

    def test_write_trace_includes_trace_type(self):
        """Test write_trace includes trace_type field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            trace_dict = {"trace_id": "abc123", "trace_type": "ingestion"}

            write_trace(trace_dict, output_path)

            with open(output_path, "r") as f:
                line = f.readline()

            parsed = json.loads(line)
            assert "trace_type" in parsed
            assert parsed["trace_type"] == "ingestion"

    def test_write_trace_chinese_content(self):
        """Test write_trace with Chinese content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"
            trace_dict = {"trace_id": "abc123", "message": "测试追踪"}

            write_trace(trace_dict, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                line = f.readline()

            parsed = json.loads(line)
            assert parsed["message"] == "测试追踪"


class TestTraceLoggerAdapter:
    """Tests for TraceLoggerAdapter."""

    def test_adapter_adds_trace_id(self):
        """Test adapter adds trace_id to log record."""
        logger = get_logger("test.adapter")
        adapter = TraceLoggerAdapter(logger, trace_id="test123")

        # Use adapter to log
        adapter.info("Test message")

        # The extra should be added
        assert adapter.extra["trace_id"] == "test123"

    def test_adapter_adds_trace_type_default(self):
        """Test adapter adds default trace_type."""
        logger = get_logger("test.adapter.default")
        adapter = TraceLoggerAdapter(logger, trace_id="test123")

        assert adapter.extra["trace_type"] == "query"

    def test_adapter_adds_trace_type_custom(self):
        """Test adapter adds custom trace_type."""
        logger = get_logger("test.adapter.custom")
        adapter = TraceLoggerAdapter(logger, trace_id="test123", trace_type="ingestion")

        assert adapter.extra["trace_type"] == "ingestion"

    def test_process_adds_extra(self):
        """Test process() adds extra to kwargs."""
        logger = get_logger("test.adapter.process")
        adapter = TraceLoggerAdapter(logger, trace_id="test123", trace_type="query")

        msg, kwargs = adapter.process("Test message", {})

        assert msg == "Test message"
        assert "extra" in kwargs
        assert kwargs["extra"]["trace_id"] == "test123"
        assert kwargs["extra"]["trace_type"] == "query"

    def test_process_preserves_existing_extra(self):
        """Test process() preserves existing extra."""
        logger = get_logger("test.adapter.preserve")
        adapter = TraceLoggerAdapter(logger, trace_id="test123", trace_type="query")

        msg, kwargs = adapter.process("Test message", {"extra": {"custom": "value"}})

        assert kwargs["extra"]["custom"] == "value"
        assert kwargs["extra"]["trace_id"] == "test123"


class TestIntegration:
    """Integration tests for logger with trace."""

    def test_trace_logger_with_adapter(self, capsys):
        """Test trace logger with adapter outputs JSON."""
        logger = get_trace_logger("test.integration")
        adapter = TraceLoggerAdapter(logger, trace_id="abc123", trace_type="query")

        adapter.info("Processing query")

        captured = capsys.readouterr()
        # Output should be JSON
        output = captured.err.strip()
        log_dict = json.loads(output)

        assert log_dict["trace_id"] == "abc123"
        assert log_dict["trace_type"] == "query"
        assert log_dict["message"] == "Processing query"

    def test_full_trace_flow(self):
        """Test full trace flow: create → log → write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/traces.jsonl"

            # Create trace dict
            trace_dict = {
                "trace_id": "abc123",
                "trace_type": "query",
                "started_at": "2026-04-26T10:00:00",
                "stages": [
                    {"stage": "dense_retrieval", "elapsed_ms": 50.5},
                    {"stage": "sparse_retrieval", "elapsed_ms": 10.2},
                ],
                "total_elapsed_ms": 65.0,
            }

            # Write trace
            write_trace(trace_dict, output_path)

            # Verify file
            with open(output_path, "r") as f:
                line = f.readline()

            parsed = json.loads(line)
            assert parsed["trace_id"] == "abc123"
            assert parsed["trace_type"] == "query"
            assert len(parsed["stages"]) == 2
