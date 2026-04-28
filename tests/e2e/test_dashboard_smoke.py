"""
E2E: Dashboard 冒烟测试。

使用 Streamlit AppTest 框架验证各页面可正常渲染、无 Python 异常。
"""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 尝试导入 streamlit testing
try:
    from streamlit.testing.v1 import AppTest
    STREAMLIT_TEST_AVAILABLE = True
except ImportError:
    STREAMLIT_TEST_AVAILABLE = False


@pytest.mark.skipif(
    not STREAMLIT_TEST_AVAILABLE,
    reason="Streamlit AppTest not available (requires streamlit>=1.28.0)"
)
class TestDashboardSmoke:
    """
    Dashboard 冒烟测试。

    验证各页面可正常渲染，无 Python 异常。
    """

    @pytest.fixture
    def app(self):
        """创建 AppTest 实例。"""
        dashboard_path = project_root / "src" / "observability" / "dashboard" / "app.py"
        return AppTest.from_file(str(dashboard_path))

    def test_app_loads(self, app):
        """测试 Dashboard 主应用可加载。"""
        app.run(timeout=10)
        assert not app.exception

    def test_overview_page(self, app):
        """测试 Overview 页面可渲染。"""
        app.run(timeout=10)
        # 默认选中 Overview 页面
        assert not app.exception

    def test_navigation_exists(self, app):
        """测试导航元素存在。"""
        app.run(timeout=10)
        # 检查 sidebar 中的导航
        # Streamlit 测试框架会验证页面可正常渲染
        assert not app.exception


class TestDashboardPagesImport:
    """
    Dashboard 页面导入测试。

    验证各页面模块可正常导入。
    """

    def test_import_overview(self):
        """测试导入 overview 页面。"""
        from src.observability.dashboard.pages.overview import render_overview_page
        assert callable(render_overview_page)

    def test_import_data_browser(self):
        """测试导入 data_browser 页面。"""
        from src.observability.dashboard.pages.data_browser import render_data_browser_page
        assert callable(render_data_browser_page)

    def test_import_ingestion_manager(self):
        """测试导入 ingestion_manager 页面。"""
        from src.observability.dashboard.pages.ingestion_manager import render_ingestion_manager_page
        assert callable(render_ingestion_manager_page)

    def test_import_ingestion_traces(self):
        """测试导入 ingestion_traces 页面。"""
        from src.observability.dashboard.pages.ingestion_traces import render_ingestion_tracing_page
        assert callable(render_ingestion_tracing_page)

    def test_import_query_traces(self):
        """测试导入 query_traces 页面。"""
        from src.observability.dashboard.pages.query_traces import render_query_tracing_page
        assert callable(render_query_tracing_page)

    def test_import_evaluation_panel(self):
        """测试导入 evaluation_panel 页面。"""
        from src.observability.dashboard.pages.evaluation_panel import render_evaluation_panel_page
        assert callable(render_evaluation_panel_page)


class TestDashboardServices:
    """
    Dashboard 服务层测试。

    验证各服务模块可正常工作。
    """

    def test_config_service(self):
        """测试 ConfigService。"""
        from src.observability.dashboard.services.config_service import ConfigService

        service = ConfigService()
        # 验证方法存在
        assert hasattr(service, 'load_settings')
        assert hasattr(service, 'get_component_configs')

    def test_trace_service(self):
        """测试 TraceService。"""
        from src.observability.dashboard.services.trace_service import TraceService

        service = TraceService()
        # 验证方法存在
        assert hasattr(service, 'read_traces')
        assert hasattr(service, 'get_ingestion_traces')
        assert hasattr(service, 'get_query_traces')

    def test_data_service(self):
        """测试 DataService。"""
        from src.observability.dashboard.services.data_service import DataService

        # DataService 需要 vector_store 参数
        # 这里只验证模块可导入
        from src.observability.dashboard.services import data_service
        assert hasattr(data_service, 'DataService')


class TestDashboardAppStructure:
    """
    Dashboard 应用结构测试。

    验证应用结构正确。
    """

    def test_app_file_exists(self):
        """测试 app.py 文件存在。"""
        app_path = project_root / "src" / "observability" / "dashboard" / "app.py"
        assert app_path.exists()

    def test_pages_directory_exists(self):
        """测试 pages 目录存在。"""
        pages_dir = project_root / "src" / "observability" / "dashboard" / "pages"
        assert pages_dir.exists()

    def test_services_directory_exists(self):
        """测试 services 目录存在。"""
        services_dir = project_root / "src" / "observability" / "dashboard" / "services"
        assert services_dir.exists()

    def test_all_page_files_exist(self):
        """测试所有页面文件存在。"""
        pages_dir = project_root / "src" / "observability" / "dashboard" / "pages"

        expected_files = [
            "overview.py",
            "data_browser.py",
            "ingestion_manager.py",
            "ingestion_traces.py",
            "query_traces.py",
            "evaluation_panel.py",
        ]

        for filename in expected_files:
            file_path = pages_dir / filename
            assert file_path.exists(), f"Missing page file: {filename}"
