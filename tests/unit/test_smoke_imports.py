"""
Smoke test for package imports.

Validates that all key packages can be imported successfully.
This test runs fast and provides early feedback on module structure.
"""

import pytest


class TestSmokeImports:
    """Verify key packages are importable."""

    def test_import_core_package(self) -> None:
        """Core package should be importable."""
        from src import core

        assert core is not None

    def test_import_ingestion_package(self) -> None:
        """Ingestion package should be importable."""
        from src import ingestion

        assert ingestion is not None

    def test_import_libs_package(self) -> None:
        """Libs package should be importable."""
        from src import libs

        assert libs is not None

    def test_import_mcp_server_package(self) -> None:
        """MCP server package should be importable."""
        from src import mcp_server

        assert mcp_server is not None

    def test_import_observability_package(self) -> None:
        """Observability package should be importable."""
        from src import observability

        assert observability is not None
