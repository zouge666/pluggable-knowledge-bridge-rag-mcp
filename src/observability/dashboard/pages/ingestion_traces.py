"""
Ingestion traces page for Dashboard.

Displays ingestion history and stage timing waterfall.
"""

import streamlit as st
from typing import Optional

from src.observability.dashboard.services.trace_service import (
    TraceService,
    TraceRecord,
    TraceStage,
)


def render_ingestion_tracing_page() -> None:
    """Render the ingestion tracing page."""
    st.title("📊 Ingestion Traces")

    # Get trace service
    trace_service = _get_trace_service()

    # Read ingestion traces
    traces = trace_service.get_ingestion_traces(limit=50)

    if not traces:
        st.info("No ingestion traces found. Run ingestion to generate traces.")
        st.caption("Traces are stored in `logs/traces.jsonl`")
        return

    # === Summary ===
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Traces", len(traces))

    with col2:
        success_count = sum(1 for t in traces if t.success)
        st.metric("Successful", success_count)

    with col3:
        failed_count = len(traces) - success_count
        st.metric("Failed", failed_count)

    st.divider()

    # === Trace List ===
    st.subheader("Trace History")

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


def _render_trace_details(trace_service: TraceService, trace: TraceRecord) -> None:
    """Render trace details with waterfall chart.

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

        # Display table
        st.dataframe(
            df.sort_values("Elapsed (ms)", ascending=False),
            use_container_width=True,
        )

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
        # Default trace file
        return TraceService()


# Streamlit page entry point
if __name__ == "__main__":
    render_ingestion_tracing_page()