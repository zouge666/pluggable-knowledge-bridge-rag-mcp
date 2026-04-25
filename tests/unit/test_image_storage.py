"""
Unit tests for ImageStorage.
"""

import base64
import os
import tempfile
import shutil
import pytest

from src.ingestion.storage.image_storage import ImageStorage, FakeImageStorage, ImageInfo
from src.core.trace.trace_context import TraceContext


class TestImageInfo:
    """Tests for ImageInfo dataclass."""

    def test_image_info_creation(self):
        """Test creating ImageInfo."""
        info = ImageInfo(
            image_id="test-123",
            file_path="/path/to/image.png",
            collection="test-collection",
            doc_hash="abc123",
            page_num=1,
            created_at="2024-01-01T00:00:00",
        )
        assert info.image_id == "test-123"
        assert info.file_path == "/path/to/image.png"
        assert info.collection == "test-collection"
        assert info.doc_hash == "abc123"
        assert info.page_num == 1
        assert info.created_at == "2024-01-01T00:00:00"

    def test_image_info_to_dict(self):
        """Test converting ImageInfo to dict."""
        info = ImageInfo(
            image_id="test-123",
            file_path="/path/to/image.png",
            collection="test-collection",
        )
        result = info.to_dict()
        assert result["image_id"] == "test-123"
        assert result["file_path"] == "/path/to/image.png"
        assert result["collection"] == "test-collection"


class TestImageStorage:
    """Tests for ImageStorage."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def image_storage(self, temp_dir):
        """Create ImageStorage instance."""
        image_dir = os.path.join(temp_dir, "images")
        db_path = os.path.join(temp_dir, "db", "image_index.db")
        return ImageStorage(image_dir=image_dir, db_path=db_path)

    @pytest.fixture
    def sample_image_data(self):
        """Sample PNG image data (1x1 red pixel)."""
        # Minimal valid PNG: 1x1 red pixel
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jN0gAAAABJRU5ErkJggg=="
        )

    def test_init_creates_directories(self, temp_dir):
        """Test that init creates necessary directories."""
        image_dir = os.path.join(temp_dir, "images")
        db_path = os.path.join(temp_dir, "db", "image_index.db")

        ImageStorage(image_dir=image_dir, db_path=db_path)

        assert os.path.exists(image_dir)
        assert os.path.exists(os.path.dirname(db_path))

    def test_save_image(self, image_storage, sample_image_data):
        """Test saving an image."""
        info = image_storage.save(
            image_id="test-1",
            image_data=sample_image_data,
            collection="test-collection",
            doc_hash="doc-123",
            page_num=1,
        )

        assert info.image_id == "test-1"
        assert info.collection == "test-collection"
        assert info.doc_hash == "doc-123"
        assert info.page_num == 1
        assert info.created_at is not None
        assert os.path.exists(info.file_path)

    def test_save_image_with_trace(self, image_storage, sample_image_data):
        """Test saving image with trace context."""
        trace = TraceContext(trace_id="test-trace")
        info = image_storage.save(
            image_id="test-2",
            image_data=sample_image_data,
            collection="test-collection",
            trace=trace,
        )

        assert info.image_id == "test-2"
        # Trace should have recorded the stage
        assert len(trace.stages) == 1
        assert trace.stages[0]["stage"] == "image_storage"

    def test_save_from_base64(self, image_storage, sample_image_data):
        """Test saving image from base64 data."""
        base64_data = base64.b64encode(sample_image_data).decode("utf-8")
        info = image_storage.save_from_base64(
            image_id="test-3",
            base64_data=base64_data,
            collection="test-collection",
        )

        assert info.image_id == "test-3"
        assert os.path.exists(info.file_path)

    def test_get_image(self, image_storage, sample_image_data):
        """Test getting image info."""
        image_storage.save(
            image_id="test-4",
            image_data=sample_image_data,
            collection="test-collection",
        )

        info = image_storage.get("test-4")
        assert info is not None
        assert info.image_id == "test-4"

    def test_get_nonexistent_image(self, image_storage):
        """Test getting nonexistent image."""
        info = image_storage.get("nonexistent")
        assert info is None

    def test_get_path(self, image_storage, sample_image_data):
        """Test getting image path."""
        saved_info = image_storage.save(
            image_id="test-5",
            image_data=sample_image_data,
        )

        path = image_storage.get_path("test-5")
        assert path == saved_info.file_path

    def test_load_image(self, image_storage, sample_image_data):
        """Test loading image data."""
        image_storage.save(
            image_id="test-6",
            image_data=sample_image_data,
        )

        loaded_data = image_storage.load("test-6")
        assert loaded_data == sample_image_data

    def test_load_nonexistent_image(self, image_storage):
        """Test loading nonexistent image."""
        data = image_storage.load("nonexistent")
        assert data is None

    def test_load_as_base64(self, image_storage, sample_image_data):
        """Test loading image as base64."""
        image_storage.save(
            image_id="test-7",
            image_data=sample_image_data,
        )

        base64_data = image_storage.load_as_base64("test-7")
        assert base64_data is not None
        decoded = base64.b64decode(base64_data)
        assert decoded == sample_image_data

    def test_list_by_collection(self, image_storage, sample_image_data):
        """Test listing images by collection."""
        image_storage.save("img-1", sample_image_data, collection="collection-a")
        image_storage.save("img-2", sample_image_data, collection="collection-a")
        image_storage.save("img-3", sample_image_data, collection="collection-b")

        images_a = image_storage.list_by_collection("collection-a")
        assert len(images_a) == 2
        assert {img.image_id for img in images_a} == {"img-1", "img-2"}

        images_b = image_storage.list_by_collection("collection-b")
        assert len(images_b) == 1

    def test_list_by_doc_hash(self, image_storage, sample_image_data):
        """Test listing images by doc hash."""
        image_storage.save("img-1", sample_image_data, doc_hash="doc-a")
        image_storage.save("img-2", sample_image_data, doc_hash="doc-a")
        image_storage.save("img-3", sample_image_data, doc_hash="doc-b")

        images_a = image_storage.list_by_doc_hash("doc-a")
        assert len(images_a) == 2

    def test_delete_image(self, image_storage, sample_image_data):
        """Test deleting an image."""
        info = image_storage.save("test-8", sample_image_data)
        assert os.path.exists(info.file_path)

        result = image_storage.delete("test-8")
        assert result is True
        assert not os.path.exists(info.file_path)
        assert image_storage.get("test-8") is None

    def test_delete_nonexistent_image(self, image_storage):
        """Test deleting nonexistent image."""
        result = image_storage.delete("nonexistent")
        assert result is False

    def test_delete_by_collection(self, image_storage, sample_image_data):
        """Test deleting all images in a collection."""
        image_storage.save("img-1", sample_image_data, collection="collection-a")
        image_storage.save("img-2", sample_image_data, collection="collection-a")
        image_storage.save("img-3", sample_image_data, collection="collection-b")

        count = image_storage.delete_by_collection("collection-a")
        assert count == 2
        assert len(image_storage.list_by_collection("collection-a")) == 0
        assert len(image_storage.list_by_collection("collection-b")) == 1

    def test_exists(self, image_storage, sample_image_data):
        """Test checking if image exists."""
        image_storage.save("test-9", sample_image_data)

        assert image_storage.exists("test-9") is True
        assert image_storage.exists("nonexistent") is False

    def test_count(self, image_storage, sample_image_data):
        """Test counting images."""
        image_storage.save("img-1", sample_image_data, collection="collection-a")
        image_storage.save("img-2", sample_image_data, collection="collection-a")
        image_storage.save("img-3", sample_image_data, collection="collection-b")

        assert image_storage.count() == 3
        assert image_storage.count("collection-a") == 2
        assert image_storage.count("collection-b") == 1

    def test_save_overwrites_existing(self, image_storage, sample_image_data):
        """Test that save overwrites existing image with same ID."""
        # Create different image data
        different_data = sample_image_data + b"extra"

        image_storage.save("test-10", sample_image_data)
        image_storage.save("test-10", different_data)

        loaded = image_storage.load("test-10")
        assert loaded == different_data

    def test_default_collection(self, image_storage, sample_image_data):
        """Test saving with default collection."""
        info = image_storage.save("test-11", sample_image_data)
        assert "default" in info.file_path


class TestFakeImageStorage:
    """Tests for FakeImageStorage."""

    @pytest.fixture
    def fake_storage(self):
        """Create FakeImageStorage instance."""
        return FakeImageStorage()

    @pytest.fixture
    def sample_image_data(self):
        """Sample image data."""
        return b"fake image data"

    def test_save_and_load(self, fake_storage, sample_image_data):
        """Test save and load in fake storage."""
        info = fake_storage.save("test-1", sample_image_data, collection="test")
        assert info.image_id == "test-1"

        loaded = fake_storage.load("test-1")
        assert loaded == sample_image_data

    def test_save_from_base64(self, fake_storage, sample_image_data):
        """Test save from base64."""
        base64_data = base64.b64encode(sample_image_data).decode("utf-8")
        info = fake_storage.save_from_base64("test-2", base64_data)
        assert info.image_id == "test-2"

        loaded = fake_storage.load("test-2")
        assert loaded == sample_image_data

    def test_get(self, fake_storage, sample_image_data):
        """Test get method."""
        fake_storage.save("test-3", sample_image_data)
        info = fake_storage.get("test-3")
        assert info is not None
        assert info.image_id == "test-3"

    def test_get_nonexistent(self, fake_storage):
        """Test getting nonexistent image."""
        assert fake_storage.get("nonexistent") is None

    def test_get_path(self, fake_storage, sample_image_data):
        """Test getting path."""
        info = fake_storage.save("test-4", sample_image_data)
        path = fake_storage.get_path("test-4")
        assert path == info.file_path

    def test_load_as_base64(self, fake_storage, sample_image_data):
        """Test loading as base64."""
        fake_storage.save("test-5", sample_image_data)
        base64_data = fake_storage.load_as_base64("test-5")
        assert base64_data is not None
        decoded = base64.b64decode(base64_data)
        assert decoded == sample_image_data

    def test_list_by_collection(self, fake_storage, sample_image_data):
        """Test listing by collection."""
        fake_storage.save("img-1", sample_image_data, collection="collection-a")
        fake_storage.save("img-2", sample_image_data, collection="collection-a")
        fake_storage.save("img-3", sample_image_data, collection="collection-b")

        images = fake_storage.list_by_collection("collection-a")
        assert len(images) == 2

    def test_list_by_doc_hash(self, fake_storage, sample_image_data):
        """Test listing by doc hash."""
        fake_storage.save("img-1", sample_image_data, doc_hash="doc-a")
        fake_storage.save("img-2", sample_image_data, doc_hash="doc-a")
        fake_storage.save("img-3", sample_image_data, doc_hash="doc-b")

        images = fake_storage.list_by_doc_hash("doc-a")
        assert len(images) == 2

    def test_delete(self, fake_storage, sample_image_data):
        """Test deleting image."""
        fake_storage.save("test-6", sample_image_data)
        result = fake_storage.delete("test-6")
        assert result is True
        assert fake_storage.get("test-6") is None

    def test_delete_nonexistent(self, fake_storage):
        """Test deleting nonexistent image."""
        result = fake_storage.delete("nonexistent")
        assert result is False

    def test_delete_by_collection(self, fake_storage, sample_image_data):
        """Test deleting by collection."""
        fake_storage.save("img-1", sample_image_data, collection="collection-a")
        fake_storage.save("img-2", sample_image_data, collection="collection-a")
        fake_storage.save("img-3", sample_image_data, collection="collection-b")

        count = fake_storage.delete_by_collection("collection-a")
        assert count == 2
        assert fake_storage.count("collection-a") == 0

    def test_exists(self, fake_storage, sample_image_data):
        """Test exists method."""
        fake_storage.save("test-7", sample_image_data)
        assert fake_storage.exists("test-7") is True
        assert fake_storage.exists("nonexistent") is False

    def test_count(self, fake_storage, sample_image_data):
        """Test count method."""
        fake_storage.save("img-1", sample_image_data, collection="collection-a")
        fake_storage.save("img-2", sample_image_data, collection="collection-a")
        fake_storage.save("img-3", sample_image_data, collection="collection-b")

        assert fake_storage.count() == 3
        assert fake_storage.count("collection-a") == 2

    def test_clear(self, fake_storage, sample_image_data):
        """Test clearing all images."""
        fake_storage.save("img-1", sample_image_data)
        fake_storage.save("img-2", sample_image_data)

        fake_storage.clear()
        assert fake_storage.count() == 0
