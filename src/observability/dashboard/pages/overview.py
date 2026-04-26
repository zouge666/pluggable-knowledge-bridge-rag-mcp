"""
System overview page for Dashboard.

Displays component configurations and data statistics.
"""

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService


def render_component_card(component: dict) -> None:
    """Render a component configuration card.

    Args:
        component: Dict with name, provider, model, and details.
    """
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            st.markdown(f"**{component['name']}**")

        with col2:
            provider = component["provider"]
            if provider == "disabled":
                st.markdown(f"🔸 {provider}")
            else:
                st.markdown(f"✅ {provider}")

        with col3:
            model = component["model"]
            if model and model != "-":
                st.caption(f"Model: {model}")

        # Show details in expander
        details = component.get("details", {})
        if details:
            with st.expander("Details", expanded=False):
                for key, value in details.items():
                    st.caption(f"**{key}**: {value}")


def render_overview_page() -> None:
    """Render the system overview page."""
    st.title("📊 System Overview")

    # Initialize config service
    config_service = ConfigService()

    # Load configuration
    try:
        settings = config_service.load_settings()
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        st.info("Please check that `config/settings.yaml` exists and is valid.")
        return

    # === Component Configuration Section ===
    st.header("🔧 Component Configuration")

    components = config_service.get_component_configs()

    for component in components:
        render_component_card(component)
        st.divider()

    # === Retrieval Settings Section ===
    st.header("🔍 Retrieval Settings")

    retrieval_config = config_service.get_retrieval_config()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dense Top-K", retrieval_config["dense_top_k"])
    with col2:
        st.metric("Sparse Top-K", retrieval_config["sparse_top_k"])
    with col3:
        st.metric("Fusion Top-K", retrieval_config["fusion_top_k"])
    with col4:
        st.metric("RRF K", retrieval_config["rrf_k"])

    # === Ingestion Settings Section ===
    st.header("📝 Ingestion Settings")

    ingestion_config = config_service.get_ingestion_config()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chunk Size", ingestion_config["chunk_size"])
    with col2:
        st.metric("Chunk Overlap", ingestion_config["chunk_overlap"])
    with col3:
        st.metric("Splitter", ingestion_config["splitter"])
    with col4:
        st.metric("Batch Size", ingestion_config["batch_size"])

    # === Observability Settings Section ===
    st.header("📈 Observability Settings")

    observability_config = config_service.get_observability_config()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Log Level", observability_config["log_level"])
        st.caption(f"Trace File: `{observability_config['trace_file']}`")
    with col2:
        trace_enabled = observability_config["trace_enabled"]
        st.metric("Trace Enabled", "✅ Yes" if trace_enabled else "❌ No")
        structured = observability_config["structured_logging"]
        st.metric("Structured Logging", "✅ Yes" if structured else "❌ No")

    # === Data Statistics Section ===
    st.header("📊 Data Statistics")

    try:
        from src.libs.vector_store.chroma_store import ChromaStore

        chroma_store = ChromaStore(
            persist_directory=settings.vector_store.persist_directory,
            collection_name=settings.vector_store.collection_name,
        )

        stats = chroma_store.get_collection_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats["count"])
        with col2:
            st.metric("Collection", stats["collection_name"])

        st.caption(f"Persist Directory: `{stats['persist_directory']}`")

    except Exception as e:
        st.warning(f"Could not load data statistics: {e}")
        st.info("Run ingestion to populate the vector store.")


# Streamlit page entry point
if __name__ == "__main__":
    render_overview_page()
