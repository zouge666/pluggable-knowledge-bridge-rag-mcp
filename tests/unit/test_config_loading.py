"""
Tests for configuration loading and validation.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.core.settings import (
    ConfigurationError,
    Settings,
    load_settings,
    validate_settings,
)


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_load_valid_settings(self, tmp_path: Path) -> None:
        """Should load valid settings from YAML file."""
        config_content = """
llm:
  provider: openai
  model: gpt-4o
  api_key: test-key

embedding:
  provider: openai
  model: text-embedding-ada-002

vector_store:
  provider: chroma
  persist_directory: ./data/db/chroma

retrieval:
  dense_top_k: 20
  sparse_top_k: 20
  fusion_top_k: 10
"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config_content)

        settings = load_settings(str(config_file))

        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o"
        assert settings.embedding.provider == "openai"
        assert settings.vector_store.provider == "chroma"

    def test_load_missing_file(self) -> None:
        """Should raise ConfigurationError for missing file."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_settings("nonexistent/path/settings.yaml")

        assert "Configuration file not found" in str(exc_info.value)

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        """Should raise ConfigurationError for invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ConfigurationError) as exc_info:
            load_settings(str(config_file))

        assert "Failed to parse YAML" in str(exc_info.value)

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Should load with default values for empty file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        settings = load_settings(str(config_file))

        # Should have default values
        assert settings.llm.provider == "openai"
        assert settings.embedding.provider == "openai"


class TestValidateSettings:
    """Tests for validate_settings function."""

    def test_validate_valid_settings(self) -> None:
        """Should pass validation for valid settings."""
        settings = Settings()
        # Should not raise
        validate_settings(settings)

    def test_validate_missing_llm_provider(self) -> None:
        """Should fail when llm.provider is missing."""
        settings = Settings()
        settings.llm.provider = ""

        with pytest.raises(ConfigurationError) as exc_info:
            validate_settings(settings)

        assert "llm.provider" in str(exc_info.value)

    def test_validate_invalid_llm_provider(self) -> None:
        """Should fail when llm.provider is invalid."""
        settings = Settings()
        settings.llm.provider = "invalid_provider"

        with pytest.raises(ConfigurationError) as exc_info:
            validate_settings(settings)

        assert "llm.provider" in str(exc_info.value)
        assert "invalid provider" in str(exc_info.value)

    def test_validate_invalid_embedding_provider(self) -> None:
        """Should fail when embedding.provider is invalid."""
        settings = Settings()
        settings.embedding.provider = "invalid_provider"

        with pytest.raises(ConfigurationError) as exc_info:
            validate_settings(settings)

        assert "embedding.provider" in str(exc_info.value)

    def test_validate_invalid_vector_store_provider(self) -> None:
        """Should fail when vector_store.provider is invalid."""
        settings = Settings()
        settings.vector_store.provider = "invalid_provider"

        with pytest.raises(ConfigurationError) as exc_info:
            validate_settings(settings)

        assert "vector_store.provider" in str(exc_info.value)

    def test_validate_invalid_rerank_provider(self) -> None:
        """Should fail when rerank.provider is invalid and enabled."""
        settings = Settings()
        settings.rerank.enabled = True
        settings.rerank.provider = "invalid_provider"

        with pytest.raises(ConfigurationError) as exc_info:
            validate_settings(settings)

        assert "rerank.provider" in str(exc_info.value)

    def test_validate_negative_top_k(self) -> None:
        """Should fail when dense_top_k is not positive."""
        settings = Settings()
        settings.retrieval.dense_top_k = 0

        with pytest.raises(ConfigurationError) as exc_info:
            validate_settings(settings)

        assert "retrieval.dense_top_k" in str(exc_info.value)

    def test_validate_negative_chunk_size(self) -> None:
        """Should fail when chunk_size is not positive."""
        settings = Settings()
        settings.ingestion.chunk_size = -1

        with pytest.raises(ConfigurationError) as exc_info:
            validate_settings(settings)

        assert "ingestion.chunk_size" in str(exc_info.value)

    def test_validate_negative_chunk_overlap(self) -> None:
        """Should fail when chunk_overlap is negative."""
        settings = Settings()
        settings.ingestion.chunk_overlap = -1

        with pytest.raises(ConfigurationError) as exc_info:
            validate_settings(settings)

        assert "ingestion.chunk_overlap" in str(exc_info.value)


class TestSettingsDataclass:
    """Tests for Settings dataclass structure."""

    def test_settings_has_all_sections(self) -> None:
        """Settings should have all configuration sections."""
        settings = Settings()

        assert hasattr(settings, "llm")
        assert hasattr(settings, "embedding")
        assert hasattr(settings, "vision_llm")
        assert hasattr(settings, "vector_store")
        assert hasattr(settings, "retrieval")
        assert hasattr(settings, "rerank")
        assert hasattr(settings, "evaluation")
        assert hasattr(settings, "observability")
        assert hasattr(settings, "ingestion")

    def test_llm_settings_defaults(self) -> None:
        """LLMSettings should have correct defaults."""
        settings = Settings()

        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o"
        assert settings.llm.temperature == 0.0
        assert settings.llm.max_tokens == 4096

    def test_embedding_settings_defaults(self) -> None:
        """EmbeddingSettings should have correct defaults."""
        settings = Settings()

        assert settings.embedding.provider == "openai"
        assert settings.embedding.model == "text-embedding-ada-002"
        assert settings.embedding.dimensions == 1536

    def test_vector_store_settings_defaults(self) -> None:
        """VectorStoreSettings should have correct defaults."""
        settings = Settings()

        assert settings.vector_store.provider == "chroma"
        assert settings.vector_store.persist_directory == "./data/db/chroma"
        assert settings.vector_store.collection_name == "knowledge_hub"


class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_error_message_without_path(self) -> None:
        """Should format message without field path."""
        error = ConfigurationError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.field_path is None

    def test_error_message_with_path(self) -> None:
        """Should format message with field path."""
        error = ConfigurationError("is required", "llm.provider")

        assert "llm.provider" in str(error)
        assert error.field_path == "llm.provider"
