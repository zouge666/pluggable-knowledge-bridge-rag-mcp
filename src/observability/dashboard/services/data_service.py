"""
Data service for Dashboard.

Encapsulates ChromaStore and ImageStorage reads for data browsing.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.ingestion.document_manager import (
    DocumentManager,
    DocumentInfo,
    DocumentDetail,
    CollectionStats,
)
from src.ingestion.storage.image_storage import ImageStorage, ImageInfo


@dataclass
class ChunkDisplay:
    """Chunk display data."""

    chunk_id: str
    text: str
    text_preview: str
    metadata: Dict[str, Any]
    page: Optional[int] = None


@dataclass
class ImageDisplay:
    """Image display data."""

    image_id: str
    file_path: str
    page_num: Optional[int] = None


class DataService:
    """Service for data browsing operations."""

    def __init__(
        self,
        document_manager: DocumentManager,
        image_storage: ImageStorage,
    ):
        """
        Initialize DataService.

        Args:
            document_manager: DocumentManager instance.
            image_storage: ImageStorage instance.
        """
        self._doc_manager = document_manager
        self._images = image_storage

    def list_documents(
        self,
        collection: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[DocumentInfo]:
        """
        List all documents.

        Args:
            collection: Collection filter (optional).
            status: Status filter (optional).

        Returns:
            List[DocumentInfo]: Document list.
        """
        return self._doc_manager.list_documents(collection=collection, status=status)

    def get_document_detail(self, source_path: str) -> Optional[DocumentDetail]:
        """
        Get document detail.

        Args:
            source_path: Source file path.

        Returns:
            Optional[DocumentDetail]: Document detail.
        """
        return self._doc_manager.get_document_detail(source_path)

    def get_collection_stats(self, collection: Optional[str] = None) -> CollectionStats:
        """
        Get collection statistics.

        Args:
            collection: Collection name (optional).

        Returns:
            CollectionStats: Collection statistics.
        """
        return self._doc_manager.get_collection_stats(collection=collection)

    def get_chunks_for_display(
        self,
        source_path: str,
        max_preview_length: int = 200,
    ) -> List[ChunkDisplay]:
        """
        Get chunks for display.

        Args:
            source_path: Source file path.
            max_preview_length: Maximum preview text length.

        Returns:
            List[ChunkDisplay]: Chunk display list.
        """
        detail = self.get_document_detail(source_path)
        if detail is None:
            return []

        chunks = []
        for chunk in detail.chunks:
            text = chunk.text
            preview = text[:max_preview_length] + "..." if len(text) > max_preview_length else text

            chunks.append(ChunkDisplay(
                chunk_id=chunk.chunk_id,
                text=text,
                text_preview=preview,
                metadata=chunk.metadata,
                page=chunk.metadata.get("page"),
            ))

        return chunks

    def get_images_for_document(
        self,
        source_path: str,
    ) -> List[ImageDisplay]:
        """
        Get images for a document.

        Args:
            source_path: Source file path.

        Returns:
            List[ImageDisplay]: Image display list.
        """
        detail = self.get_document_detail(source_path)
        if detail is None:
            return []

        images = []
        for image_id in detail.images:
            info = self._images.get(image_id)
            if info:
                images.append(ImageDisplay(
                    image_id=image_id,
                    file_path=info.file_path,
                    page_num=info.page_num,
                ))

        return images

    def load_image(self, image_id: str) -> Optional[bytes]:
        """
        Load image data.

        Args:
            image_id: Image ID.

        Returns:
            Optional[bytes]: Image data.
        """
        return self._images.load(image_id)

    def load_image_base64(self, image_id: str) -> Optional[str]:
        """
        Load image data as base64.

        Args:
            image_id: Image ID.

        Returns:
            Optional[str]: Base64 encoded image data.
        """
        return self._images.load_as_base64(image_id)

    def get_collections(self) -> List[str]:
        """
        Get list of collections.

        Returns:
            List[str]: Collection names.
        """
        documents = self.list_documents()
        collections = set()
        for doc in documents:
            if doc.collection:
                collections.add(doc.collection)
        return sorted(list(collections))
