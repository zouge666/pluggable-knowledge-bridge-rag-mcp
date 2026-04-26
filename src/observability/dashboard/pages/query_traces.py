"""
Query traces page for Dashboard.

Displays query history, Dense/Sparse comparison, and Rerank changes.
"""

import streamlit as st
from typing import Optional

from src.observability.dashboard.services.trace_service import (
    TraceService,
    TraceRecord,
    TraceStage,
)


def render_query_tracing_page() -> None:
    """Render the query tracing page."""
    st.title("🔍 Query Traces")

    # Get trace service
    trace_service = _get_trace_service()

    # Read query traces
    traces = trace_service.get_query_traces(limit=50)

    if not traces:
        st.info("No query traces found. Run queries to generate traces.")
        st.caption("Traces are stored in `logs/trace.jsonl`")
        return

    # === Search ===
    search_query = st.text_input(
        "Search queries",
        placeholder="Enter keywords to search...",
    )

    # Filter traces by search query
    if search_query:
        traces = _filter_traces(traces, search_query)

    # === Summary ===
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Queries", len(traces))

    with col2:
        success_count = sum(1 for t in traces if t.success)
        st.metric("Successful", success_count)

    with col3:
        if traces:
            avg_time = sum(t.total_elapsed_ms for t in traces) / len(traces)
            st.metric("Avg Time", f"{avg_time:.1f}ms")

    st.divider()

    # === Trace List ===
    st.subheader("Query History")

    for trace in traces:
        with st.container():
            # Trace header
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                status_icon = "✅" if trace.success else "❌"
                st.markdown(f"{status_icon} **{trace.trace_id[:8]}...**")

            with col2:
                if trace.timestamp:
                    st.caption(f"Time: {trace.timestamp}")

            with col3:
                st.caption(f"{trace.total_elapsed_ms:.1f}ms")

            # Error message if failed
            if not trace.success and trace.error:
                st.error(f"Error: {trace.error}")

            # Expandable details
            with st.expander("View Details", expanded=False):
                _render_trace_details(trace_service, trace)

            st.divider()


def _filter_traces(traces: list, search_query: str) -> list:
    """Filter traces by search query.

    Args:
        traces: List of traces.
        search_query: Search query string.

    Returns:
        list: Filtered traces.
    """
    filtered = []
    for trace in traces:
        # Search in metadata
        if trace.metadata:
            query_text = trace.metadata.get("query", "")
            if search_query.lower() in query_text.lower():
                filtered.append(trace)
                continue

        # Search in trace_id
        if search_query.lower() in trace.trace_id.lower():
            filtered.append(trace)
            continue

    return filtered


def _render_trace_details(trace_service: TraceService, trace: TraceRecord) -> None:
    """Render trace details with comparison charts.

    Args:
        trace_service: TraceService instance.
        trace: Trace record.
    """
    st.subheader("Stage Timing")

    # Stage summary
    stage_summary = trace_service.get_stage_summary(trace)

    if stage_summary:
        # Create bar chart data
        import pandas as pd

        df = pd.DataFrame([
            {"Stage": stage, "Elapsed (ms)": elapsed}
            for stage, elapsed in stage_summary.items()
        ])

        # Sort by elapsed time
        df = df.sort_values("Elapsed (ms)", ascending=True)

        # Display bar chart
        st.bar_chart(df.set_index("Stage"))

    # Dense vs Sparse comparison
    _render_dense_sparse_comparison(trace)

    # Rerank changes
    _render_rerank_changes(trace)

    # Stage details
    st.subheader("Stage Details")

    for stage in trace.stages:
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                st.markdown(f"**{stage.stage}**")

            with col2:
                st.caption(f"{stage.elapsed_ms:.2f}ms")

            with col3:
                if stage.method:
                    st.caption(f"Method: {stage.method}")
                if stage.provider:
                    st.caption(f"Provider: {stage.provider}")

            # Show details if available
            if stage.details:
                with st.expander("Details", expanded=False):
                    for key, value in stage.details.items():
                        st.caption(f"**{key}**: {value}")

            st.divider()


def _render_dense_sparse_comparison(trace: TraceRecord) -> None:
    """Render Dense vs Sparse retrieval comparison.

    Args:
        trace: Trace record.
    """
    dense_stage = None
    sparse_stage = None

    for stage in trace.stages:
        if stage.stage == "dense_retrieval":
            dense_stage = stage
        elif stage.stage == "sparse_retrieval":
            sparse_stage = stage

    if dense_stage is None and sparse_stage is None:
        return

    st.subheader("Dense vs Sparse Retrieval")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Dense Retrieval**")
        if dense_stage:
            st.metric("Elapsed", f"{dense_stage.elapsed_ms:.2f}ms")
            if dense_stage.details:
                count = dense_stage.details.get("result_count", "-")
                st.metric("Results", count)
        else:
            st.info("Not used")

    with col2:
        st.markdown("**Sparse Retrieval**")
        if sparse_stage:
            st.metric("Elapsed", f"{sparse_stage.elapsed_ms:.2f}ms")
            if sparse_stage.details:
                count = sparse_stage.details.get("result_count", "-")
                st.metric("Results", count)
        else:
            st.info("Not used")


def _render_rerank_changes(trace: TraceRecord) -> None:
    """Render Rerank changes.

    Args:
        trace: Trace record.
    """
    rerank_stage = None

    for stage in trace.stages:
        if stage.stage == "rerank":
            rerank_stage = stage
            break

    if rerank_stage is None:
        return

    st.subheader("Rerank")

    st.metric("Elapsed", f"{rerank_stage.elapsed_ms:.2f}ms")

    if rerank_stage.details:
        col1, col2 = st.columns(2)

        with col1:
            input_count = rerank_stage.details.get("input_count", "-")
            st.metric("Input Count", input_count)

        with col2:
            output_count = rerank_stage.details.get("output_count", "-")
            st.metric("Output Count", output_count)

        # Show ranking changes if available
        ranking_changes = rerank_stage.details.get("ranking_changes")
        if ranking_changes:
            with st.expander("Ranking Changes", expanded=False):
                import pandas as pd
                df = pd.DataFrame(ranking_changes)
                st.dataframe(df, use_container_width=True)


@st.cache_resource
def _get_trace_service() -> TraceService:
    """Get or create TraceService instance.

    Returns:
        TraceService: TraceService instance.
    """
    try:
        from src.core.settings import load_settings

        settings = load_settings()
        trace_file = settings.observability.trace_file

        return TraceService(trace_file=trace_file)

    except Exception:
        return TraceService()


# Streamlit page entry point
if __name__ == "__main__":
    render_query_tracing_page()