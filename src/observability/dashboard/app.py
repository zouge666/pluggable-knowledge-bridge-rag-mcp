"""
Main Streamlit Dashboard application.

Multi-page navigation architecture for the RAG Knowledge Hub.
"""

import streamlit as st

from src.observability.dashboard.pages.overview import render_overview_page
from src.observability.dashboard.pages.data_browser import render_data_browser_page as render_data_browser_page_real
from src.observability.dashboard.pages.ingestion_manager import render_ingestion_manager_page as render_ingestion_manager_page_real
from src.observability.dashboard.pages.ingestion_traces import render_ingestion_tracing_page as render_ingestion_tracing_page_real
from src.observability.dashboard.pages.query_traces import render_query_tracing_page as render_query_tracing_page_real


def render_placeholder_page(page_name: str) -> None:
    """Render a placeholder page for unimplemented features.

    Args:
        page_name: Name of the page to display.
    """
    st.title(f"🚧 {page_name}")
    st.info(f"This page is under construction. Please check back later.")
    st.markdown(
        """
        **Planned Features:**
        - Evaluation metrics
        - Test result history

        For now, please use the **Overview** page to view system configuration.
        """
    )


def render_data_browser_page() -> None:
    """Render the data browser page."""
    render_data_browser_page_real()


def render_ingestion_manager_page() -> None:
    """Render the ingestion manager page."""
    render_ingestion_manager_page_real()


def render_ingestion_tracing_page() -> None:
    """Render the ingestion tracing page."""
    render_ingestion_tracing_page_real()


def render_query_tracing_page() -> None:
    """Render the query tracing page."""
    render_query_tracing_page_real()


def render_ingestion_manager_page() -> None:
    """Render the ingestion manager page (placeholder)."""
    render_placeholder_page("Ingestion Manager")


def render_ingestion_tracing_page() -> None:
    """Render the ingestion tracing page (placeholder)."""
    render_placeholder_page("Ingestion Tracing")


def render_query_tracing_page() -> None:
    """Render the query tracing page (placeholder)."""
    render_placeholder_page("Query Tracing")


def render_evaluation_page() -> None:
    """Render the evaluation page (placeholder)."""
    render_placeholder_page("Evaluation")


def main() -> None:
    """Main entry point for the Dashboard."""
    st.set_page_config(
        page_title="RAG Knowledge Hub Dashboard",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Define pages
    pages = {
        "Overview": render_overview_page,
        "Data Browser": render_data_browser_page,
        "Ingestion Manager": render_ingestion_manager_page,
        "Ingestion Tracing": render_ingestion_tracing_page,
        "Query Tracing": render_query_tracing_page,
        "Evaluation": render_evaluation_page,
    }

    # Sidebar navigation
    with st.sidebar:
        st.title("📚 Knowledge Hub")
        st.caption("RAG Management Dashboard")

        st.divider()

        selected_page = st.radio(
            "Navigate",
            list(pages.keys()),
            label_visibility="collapsed",
        )

        st.divider()

        # Show version info
        st.caption("Version: 0.1.0")
        st.caption("Powered by Streamlit")

    # Render selected page
    pages[selected_page]()


if __name__ == "__main__":
    main()
