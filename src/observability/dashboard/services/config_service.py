"""
Configuration service for Dashboard.

Provides utilities to read and format Settings for display.
"""

from dataclasses import asdict
from typing import Any, Dict, List

from src.core.settings import (
    Settings,
    LLMSettings,
    EmbeddingSettings,
    VisionLLMSettings,
    VectorStoreSettings,
    RetrievalSettings,
    RerankSettings,
    EvaluationSettings,
    ObservabilitySettings,
    IngestionSettings,
    load_settings,
    ConfigurationError,
)


class ConfigService:
    """Service for reading and formatting configuration."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize ConfigService.

        Args:
            config_path: Path to the settings YAML file.
        """
        self.config_path = config_path
        self._settings: Settings | None = None

    def load_settings(self) -> Settings:
        """Load settings from YAML file.

        Returns:
            Settings object.

        Raises:
            ConfigurationError: If configuration cannot be loaded.
        """
        if self._settings is None:
            self._settings = load_settings(self.config_path)
        return self._settings

    def get_component_configs(self) -> List[Dict[str, Any]]:
        """Get formatted component configurations for display.

        Returns:
            List of component config dicts with name, provider, and details.
        """
        settings = self.load_settings()

        components = [
            {
                "name": "LLM",
                "provider": settings.llm.provider,
                "model": settings.llm.model,
                "details": {
                    "temperature": settings.llm.temperature,
                    "max_tokens": settings.llm.max_tokens,
                },
            },
            {
                "name": "Embedding",
                "provider": settings.embedding.provider,
                "model": settings.embedding.model,
                "details": {
                    "dimensions": settings.embedding.dimensions,
                },
            },
            {
                "name": "Vision LLM",
                "provider": settings.vision_llm.provider if settings.vision_llm.enabled else "disabled",
                "model": settings.vision_llm.model if settings.vision_llm.enabled else "-",
                "details": {
                    "enabled": settings.vision_llm.enabled,
                    "max_image_size": settings.vision_llm.max_image_size if settings.vision_llm.enabled else "-",
                },
            },
            {
                "name": "Vector Store",
                "provider": settings.vector_store.provider,
                "model": settings.vector_store.collection_name,
                "details": {
                    "persist_directory": settings.vector_store.persist_directory,
                },
            },
            {
                "name": "Reranker",
                "provider": settings.rerank.provider if settings.rerank.enabled else "disabled",
                "model": settings.rerank.model if settings.rerank.enabled else "-",
                "details": {
                    "enabled": settings.rerank.enabled,
                    "top_k": settings.rerank.top_k if settings.rerank.enabled else "-",
                },
            },
            {
                "name": "Evaluation",
                "provider": settings.evaluation.provider if settings.evaluation.enabled else "disabled",
                "model": "-",
                "details": {
                    "enabled": settings.evaluation.enabled,
                    "metrics": ", ".join(settings.evaluation.metrics) if settings.evaluation.enabled else "-",
                },
            },
        ]

        return components

    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration.

        Returns:
            Dict with retrieval settings.
        """
        settings = self.load_settings()
        return {
            "dense_top_k": settings.retrieval.dense_top_k,
            "sparse_top_k": settings.retrieval.sparse_top_k,
            "fusion_top_k": settings.retrieval.fusion_top_k,
            "rrf_k": settings.retrieval.rrf_k,
        }

    def get_ingestion_config(self) -> Dict[str, Any]:
        """Get ingestion configuration.

        Returns:
            Dict with ingestion settings.
        """
        settings = self.load_settings()
        return {
            "chunk_size": settings.ingestion.chunk_size,
            "chunk_overlap": settings.ingestion.chunk_overlap,
            "splitter": settings.ingestion.splitter,
            "batch_size": settings.ingestion.batch_size,
        }

    def get_observability_config(self) -> Dict[str, Any]:
        """Get observability configuration.

        Returns:
            Dict with observability settings.
        """
        settings = self.load_settings()
        return {
            "log_level": settings.observability.log_level,
            "trace_enabled": settings.observability.trace_enabled,
            "trace_file": settings.observability.trace_file,
            "structured_logging": settings.observability.structured_logging,
        }

    def get_raw_settings(self) -> Dict[str, Any]:
        """Get raw settings as dictionary.

        Returns:
            Dict representation of all settings.
        """
        settings = self.load_settings()
        return asdict(settings)
