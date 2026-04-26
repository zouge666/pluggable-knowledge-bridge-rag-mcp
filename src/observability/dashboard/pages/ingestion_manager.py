"""
Ingestion manager page for Dashboard.

Provides file upload, ingestion progress, and document deletion.
"""

import streamlit as st
from typing import Optional, Callable
import tempfile
from pathlib import Path

from src.observability.dashboard.services.data_service import DataService
from src.ingestion.document_manager import DocumentManager


def render_ingestion_manager_page() -> None:
    """Render the ingestion manager page."""
    st.title("📥 Ingestion Manager")

    # Get data service
    data_service = _get_data_service()

    if data_service is None:
        st.warning("Data service not available. Please check configuration.")
        return

    # === Tabs ===
    tab1, tab2 = st.tabs(["Upload & Ingest", "Manage Documents"])

    with tab1:
        _render_upload_tab(data_service)

    with tab2:
        _render_manage_tab(data_service)


def _render_upload_tab(data_service: DataService) -> None:
    """Render the upload and ingest tab.

    Args:
        data_service: DataService instance.
    """
    st.subheader("Upload File")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md"],
        help="Upload a PDF, text, or markdown file for ingestion.",
    )

    # Collection selection
    collection = st.text_input(
        "Collection Name",
        value="default",
        help="The collection to ingest the document into.",
    )

    # Ingest button
    if uploaded_file is not None:
        st.info(f"Selected: {uploaded_file.name} ({uploaded_file.size} bytes)")

        if st.button("Start Ingestion", type="primary"):
            _run_ingestion(uploaded_file, collection)


def _run_ingestion(uploaded_file, collection: str) -> None:
    """Run ingestion on uploaded file.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        collection: Collection name.
    """
    # Create progress bar
    progress_bar = st.progress(0, text="Preparing...")
    status_text = st.empty()

    # Progress callback
    def on_progress(stage: str, current: int, total: int):
        progress = current / total
        progress_bar.progress(progress, text=f"Stage: {stage} ({current}/{total})")
        status_text.info(f"Processing: {stage}")

    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)

        status_text.info("Starting ingestion pipeline...")

        # Get pipeline
        pipeline = _get_ingestion_pipeline(collection, on_progress)

        if pipeline is None:
            st.error("Failed to create ingestion pipeline.")
            return

        # Run ingestion
        result = pipeline.ingest(tmp_path)

        # Clean up temp file
        tmp_path.unlink(missing_ok=True)

        if result.success:
            progress_bar.progress(1.0, text="Complete!")
            status_text.success(f"Ingestion complete! Processed {result.chunks_count} chunks.")
            st.rerun()
        else:
            progress_bar.empty()
            status_text.error(f"Ingestion failed: {result.error}")

    except Exception as e:
        progress_bar.empty()
        st.error(f"Error during ingestion: {e}")


def _render_manage_tab(data_service: DataService) -> None:
    """Render the manage documents tab.

    Args:
        data_service: DataService instance.
    """
    st.subheader("Document Management")

    # Collection filter
    collections = data_service.get_collections()
    if collections:
        selected_collection = st.selectbox(
            "Filter by Collection",
            options=["All"] + collections,
            index=0,
        )
        collection_filter = None if selected_collection == "All" else selected_collection
    else:
        collection_filter = None

    # List documents
    documents = data_service.list_documents(collection=collection_filter, status="success")

    if not documents:
        st.info("No documents found.")
        return

    st.caption(f"Total: {len(documents)} documents")

    for doc in documents:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

            with col1:
                st.markdown(f"**{doc.source_path}**")

            with col2:
                st.caption(f"Collection: {doc.collection or 'default'}")

            with col3:
                st.caption(f"Chunks: {doc.chunk_count}")

            with col4:
                if st.button("🗑️", key=f"del_{doc.file_hash}", help="Delete document"):
                    _delete_document(data_service, doc.source_path, doc.collection)

            st.divider()


def _delete_document(
    data_service: DataService,
    source_path: str,
    collection: Optional[str],
) -> None:
    """Delete a document.

    Args:
        data_service: DataService instance.
        source_path: Source file path.
        collection: Collection name.
    """
    try:
        doc_manager = _get_document_manager()
        if doc_manager is None:
            st.error("Document manager not available.")
            return

        result = doc_manager.delete_document(source_path, collection)

        if result.success:
            st.success(f"Deleted: {source_path}")
            st.rerun()
        else:
            st.error(f"Failed to delete: {result.error}")

    except Exception as e:
        st.error(f"Error deleting document: {e}")


@st.cache_resource
def _get_data_service() -> Optional[DataService]:
    """Get or create DataService instance."""
    try:
        from src.core.settings import load_settings
        from src.libs.vector_store.chroma_store import ChromaStore
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker
        from src.ingestion.storage.image_storage import ImageStorage

        settings = load_settings()
        chroma_store = ChromaStore(settings)
        bm25_indexer = BM25Indexer()
        image_storage = ImageStorage()
        file_integrity = SQLiteIntegrityChecker()

        doc_manager = DocumentManager(
            chroma_store=chroma_store,
            bm25_indexer=bm25_indexer,
            image_storage=image_storage,
            file_integrity=file_integrity,
        )

        return DataService(
            document_manager=doc_manager,
            image_storage=image_storage,
        )

    except Exception as e:
        st.error(f"Failed to initialize data service: {e}")
        return None


def _get_document_manager() -> Optional[DocumentManager]:
    """Get or create DocumentManager instance."""
    try:
        from src.core.settings import load_settings
        from src.libs.vector_store.chroma_store import ChromaStore
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker
        from src.ingestion.storage.image_storage import ImageStorage

        settings = load_settings()
        chroma_store = ChromaStore(settings)
        bm25_indexer = BM25Indexer()
        image_storage = ImageStorage()
        file_integrity = SQLiteIntegrityChecker()

        return DocumentManager(
            chroma_store=chroma_store,
            bm25_indexer=bm25_indexer,
            image_storage=image_storage,
            file_integrity=file_integrity,
        )

    except Exception:
        return None


def _get_ingestion_pipeline(
    collection: str,
    on_progress: Callable,
):
    """Get ingestion pipeline with progress callback.

    Args:
        collection: Collection name.
        on_progress: Progress callback function.

    Returns:
        IngestionPipeline instance or None.
    """
    try:
        from src.ingestion.pipeline import IngestionPipeline
        from src.core.settings import load_settings

        settings = load_settings()

        # Create pipeline with progress callback
        pipeline = IngestionPipeline(
            settings=settings,
            collection=collection,
            on_progress=on_progress,
        )

        return pipeline

    except Exception as e:
        st.error(f"Failed to create pipeline: {e}")
        return None


# Streamlit page entry point
if __name__ == "__main__":
    render_ingestion_manager_page()