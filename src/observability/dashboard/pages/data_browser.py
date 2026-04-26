"""
Data browser page for Dashboard.

Displays document list, chunk details, and image preview.
"""

import streamlit as st
from typing import Optional

from src.observability.dashboard.services.data_service import DataService
from src.ingestion.document_manager import DocumentManager
from src.ingestion.storage.image_storage import ImageStorage


def render_data_browser_page() -> None:
    """Render the data browser page."""
    st.title("📁 Data Browser")

    # Initialize data service
    data_service = _get_data_service()

    if data_service is None:
        st.warning("Data service not available. Please check configuration.")
        st.info("Make sure the vector store and other storage modules are configured.")
        return

    # === Collection Filter ===
    st.sidebar.subheader("Filters")

    collections = data_service.get_collections()
    if collections:
        selected_collection = st.sidebar.selectbox(
            "Collection",
            options=["All"] + collections,
            index=0,
        )
        collection_filter = None if selected_collection == "All" else selected_collection
    else:
        collection_filter = None
        st.sidebar.info("No collections found")

    # === Statistics ===
    stats = data_service.get_collection_stats(collection_filter)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", stats.document_count)
    with col2:
        st.metric("Chunks", stats.chunk_count)
    with col3:
        st.metric("Images", stats.image_count)

    st.divider()

    # === Document List ===
    st.subheader("Documents")

    documents = data_service.list_documents(collection=collection_filter, status="success")

    if not documents:
        st.info("No documents found. Run ingestion to add documents.")
        return

    for doc in documents:
        with st.container():
            # Document header
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

            with col1:
                st.markdown(f"**{doc.source_path}**")

            with col2:
                st.caption(f"Collection: {doc.collection or 'default'}")

            with col3:
                st.caption(f"Chunks: {doc.chunk_count}")

            with col4:
                st.caption(f"Images: {doc.image_count}")

            # Expandable details
            with st.expander("View Details", expanded=False):
                _render_document_details(data_service, doc.source_path)

            st.divider()


def _render_document_details(data_service: DataService, source_path: str) -> None:
    """Render document details including chunks and images.

    Args:
        data_service: DataService instance.
        source_path: Source file path.
    """
    # Get chunks
    chunks = data_service.get_chunks_for_display(source_path)

    if chunks:
        st.subheader("Chunks")
        st.caption(f"Total: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            with st.container():
                # Chunk header
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"**#{i + 1}**")
                    st.caption(f"ID: `{chunk.chunk_id}`")
                    if chunk.page:
                        st.caption(f"Page: {chunk.page}")

                with col2:
                    # Show preview
                    st.text(chunk.text_preview)

                    # Expandable full text
                    with st.expander("Full Text", expanded=False):
                        st.text_area(
                            "Content",
                            chunk.text,
                            height=200,
                            key=f"text_{chunk.chunk_id}",
                            disabled=True,
                        )

                    # Expandable metadata
                    with st.expander("Metadata", expanded=False):
                        for key, value in chunk.metadata.items():
                            st.caption(f"**{key}**: {value}")

                st.divider()

    # Get images
    images = data_service.get_images_for_document(source_path)

    if images:
        st.subheader("Images")
        st.caption(f"Total: {len(images)} images")

        for img in images:
            col1, col2 = st.columns([1, 3])

            with col1:
                st.caption(f"ID: `{img.image_id}`")
                if img.page_num:
                    st.caption(f"Page: {img.page_num}")

            with col2:
                # Try to display image
                try:
                    image_data = data_service.load_image(img.image_id)
                    if image_data:
                        st.image(image_data, width=300)
                    else:
                        st.warning("Image file not found")
                except Exception as e:
                    st.warning(f"Could not load image: {e}")

            st.divider()


@st.cache_resource
def _get_data_service() -> Optional[DataService]:
    """Get or create DataService instance.

    Returns:
        Optional[DataService]: DataService instance or None if not available.
    """
    try:
        from src.core.settings import load_settings
        from src.libs.vector_store.chroma_store import ChromaStore
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker

        # Load settings
        settings = load_settings()

        # Create ChromaStore
        chroma_store = ChromaStore(settings)

        # Create BM25Indexer
        bm25_indexer = BM25Indexer()

        # Create ImageStorage
        image_storage = ImageStorage()

        # Create FileIntegrityChecker
        file_integrity = SQLiteIntegrityChecker()

        # Create DocumentManager
        doc_manager = DocumentManager(
            chroma_store=chroma_store,
            bm25_indexer=bm25_indexer,
            image_storage=image_storage,
            file_integrity=file_integrity,
        )

        # Create DataService
        return DataService(
            document_manager=doc_manager,
            image_storage=image_storage,
        )

    except Exception as e:
        st.error(f"Failed to initialize data service: {e}")
        return None


# Streamlit page entry point
if __name__ == "__main__":
    render_data_browser_page()