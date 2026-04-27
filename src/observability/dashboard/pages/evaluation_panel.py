"""
Evaluation Panel page for Dashboard.

Run evaluations and view metrics.
"""

import streamlit as st
from pathlib import Path

from src.observability.dashboard.services.config_service import ConfigService


def render_evaluation_panel_page() -> None:
    """Render the evaluation panel page."""
    st.title("📊 Evaluation Panel")

    # Initialize config service
    config_service = ConfigService()

    # Load configuration
    try:
        settings = config_service.load_settings()
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        st.info("Please check that `config/settings.yaml` exists and is valid.")
        return

    # === Evaluation Configuration Section ===
    st.header("⚙️ Evaluation Configuration")

    eval_config = settings.evaluation

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Provider", eval_config.provider)
        st.metric("Enabled", "✅ Yes" if eval_config.enabled else "❌ No")
    with col2:
        st.metric("Metrics", ", ".join(eval_config.metrics) if eval_config.metrics else "Default")

    st.divider()

    # === Run Evaluation Section ===
    st.header("🚀 Run Evaluation")

    # Test set selection
    test_sets_dir = Path("tests/fixtures")
    available_test_sets = list(test_sets_dir.glob("golden_test_set*.json")) if test_sets_dir.exists() else []

    if not available_test_sets:
        st.warning("No golden test sets found in `tests/fixtures/`")
        st.info("Create a golden test set file named `golden_test_set.json` to run evaluations.")
        test_set_path = st.text_input("Test Set Path", value="tests/fixtures/golden_test_set.json")
    else:
        test_set_names = [str(p) for p in available_test_sets]
        selected_test_set = st.selectbox(
            "Select Test Set",
            options=test_set_names,
            index=0,
        )
        test_set_path = selected_test_set

    # Evaluation parameters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.number_input("Top-K", min_value=1, max_value=100, value=10)
    with col2:
        collection = st.text_input("Collection Filter (optional)", value="")

    # Run button
    run_button = st.button("▶️ Run Evaluation", type="primary")

    if run_button:
        run_evaluation(test_set_path, top_k, collection if collection else None)

    st.divider()

    # === Results Section ===
    st.header("📈 Results")

    # Check if there are cached results in session state
    if "eval_results" in st.session_state:
        display_results(st.session_state["eval_results"])
    else:
        st.info("Run an evaluation to see results here.")


def run_evaluation(test_set_path: str, top_k: int, collection: str | None) -> None:
    """Run evaluation and display results.

    Args:
        test_set_path: Path to the golden test set.
        top_k: Number of results to retrieve.
        collection: Optional collection filter.
    """
    import time

    with st.spinner("Running evaluation..."):
        progress_bar = st.progress(0, text="Initializing...")

        try:
            # Step 1: Load configuration
            progress_bar.progress(10, text="Loading configuration...")
            from src.core.settings import load_settings
            settings = load_settings()

            # Step 2: Create evaluator
            progress_bar.progress(20, text="Creating evaluator...")
            from src.libs.evaluator.evaluator_factory import EvaluatorFactory
            evaluator = EvaluatorFactory.create(settings)

            # Step 3: Create HybridSearch (or Fake for demo)
            progress_bar.progress(40, text="Initializing search engine...")
            try:
                from src.core.query_engine.hybrid_search import HybridSearch
                from src.core.query_engine.dense_retriever import DenseRetriever
                from src.core.query_engine.sparse_retriever import SparseRetriever
                from src.libs.embedding.embedding_factory import EmbeddingFactory
                from src.libs.vector_store.vector_store_factory import VectorStoreFactory
                from src.ingestion.storage.bm25_indexer import BM25Indexer

                embedding_client = EmbeddingFactory.create(settings)
                vector_store = VectorStoreFactory.create(settings)
                bm25_indexer = BM25Indexer()

                hybrid_search = HybridSearch(
                    settings=settings,
                    dense_retriever=DenseRetriever(
                        embedding_client=embedding_client,
                        vector_store=vector_store,
                    ),
                    sparse_retriever=SparseRetriever(
                        bm25_indexer=bm25_indexer,
                        vector_store=vector_store,
                    ),
                )
            except Exception as e:
                st.warning(f"Search engine initialization failed, using mock mode: {e}")
                from src.core.query_engine.hybrid_search import FakeHybridSearch
                hybrid_search = FakeHybridSearch()

            # Step 4: Run evaluation
            progress_bar.progress(60, text="Running evaluation...")
            from src.observability.evaluation.eval_runner import EvalRunner
            runner = EvalRunner(
                hybrid_search=hybrid_search,
                evaluator=evaluator,
                top_k=top_k,
            )

            result = runner.run_with_details(test_set_path, collection)

            # Complete
            progress_bar.progress(100, text="Done!")

            # Store results in session state
            st.session_state["eval_results"] = result

            st.success(f"✅ Evaluation completed in {result['elapsed_ms']:.2f} ms")

        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Evaluation failed: {e}")
            import traceback
            with st.expander("View error details"):
                traceback.print_exc()


def display_results(results: dict) -> None:
    """Display evaluation results.

    Args:
        results: Evaluation results dictionary.
    """
    # Summary metrics
    st.subheader("Summary Metrics")

    avg_metrics = results.get("avg_metrics", {})
    if avg_metrics:
        cols = st.columns(len(avg_metrics))
        for i, (name, value) in enumerate(avg_metrics.items()):
            with cols[i]:
                st.metric(name, f"{value:.4f}")

    # Test set info
    st.caption(f"Test Set: {results.get('test_set_name', 'N/A')} (v{results.get('test_set_version', 'N/A')})")
    st.caption(f"Total Queries: {results.get('total_queries', 0)}")

    st.divider()

    # Detailed results
    st.subheader("Detailed Results")

    detailed_results = results.get("detailed_results", [])
    if detailed_results:
        for detail in detailed_results:
            with st.expander(f"[{detail['index'] + 1}] {detail['query']}", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Expected IDs:**")
                    expected_ids = detail.get("expected_ids", [])
                    if expected_ids:
                        for eid in expected_ids[:5]:  # Show first 5
                            st.caption(f"- `{eid}`")
                        if len(expected_ids) > 5:
                            st.caption(f"... and {len(expected_ids) - 5} more")
                    else:
                        st.caption("None specified")

                with col2:
                    st.markdown("**Metrics:**")
                    metrics = detail.get("metrics", {})
                    for name, value in metrics.items():
                        st.caption(f"- **{name}**: {value:.4f}")

                # Elapsed time
                elapsed = detail.get("elapsed_ms")
                if elapsed:
                    st.caption(f"⏱️ Elapsed: {elapsed:.2f} ms")

    else:
        st.info("No detailed results available.")


# Streamlit page entry point
if __name__ == "__main__":
    render_evaluation_panel_page()