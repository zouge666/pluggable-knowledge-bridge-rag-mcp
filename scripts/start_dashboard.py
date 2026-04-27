#!/usr/bin/env python3
"""
Dashboard Startup Script

Usage:
    python scripts/start_dashboard.py
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent


def main() -> int:
    dashboard_path = project_root / "src" / "observability" / "dashboard" / "app.py"

    if not dashboard_path.exists():
        print("Dashboard not yet implemented (A1 skeleton only)")
        print(f"Expected path: {dashboard_path}")
        return 1

    print("Starting Streamlit Dashboard...")
    print(f"Project root: {project_root}")

    # 使用 streamlit run，并确保工作目录是项目根目录
    return subprocess.call(
        [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
        ],
        cwd=str(project_root),  # 关键：设置工作目录为项目根目录
    )


if __name__ == "__main__":
    sys.exit(main())