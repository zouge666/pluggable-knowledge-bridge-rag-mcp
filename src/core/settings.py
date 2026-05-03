"""
Configuration loading and validation.

Provides Settings dataclass and utilities to load/validate config from YAML.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required fields."""

    def __init__(self, message: str, field_path: Optional[str] = None):
        self.field_path = field_path
        if field_path:
            message = f"{field_path}: {message}"
        super().__init__(message)


@dataclass
class LLMSettings:
    """LLM provider configuration."""
    provider: str = "openai"
    model: str = "gpt-4o"
    deployment_name: str = ""
    azure_endpoint: str = ""
    api_version: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class EmbeddingSettings:
    """Embedding provider configuration."""
    provider: str = "openai"
    model: str = "text-embedding-ada-002"
    dimensions: int = 1536
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = ""
    api_key: str = ""
    base_url: str = ""


@dataclass
class VisionLLMSettings:
    """Vision LLM provider configuration."""
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o"
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = ""
    api_key: str = ""
    base_url: str = ""
    max_image_size: int = 2048


@dataclass
class VectorStoreSettings:
    """Vector store configuration."""
    provider: str = "chroma"
    persist_directory: str = "./data/db/chroma"
    collection_name: str = "knowledge_hub"


@dataclass
class RetrievalSettings:
    """Retrieval configuration."""
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    rrf_k: int = 60


@dataclass
class RerankSettings:
    """Reranker configuration."""
    enabled: bool = False
    provider: str = "none"
    model: str = ""
    top_k: int = 5


@dataclass
class EvaluationSettings:
    """Evaluation configuration."""
    enabled: bool = False
    provider: str = "custom"
    metrics: List[str] = field(default_factory=lambda: ["hit_rate", "mrr", "faithfulness"])


@dataclass
class ObservabilitySettings:
    """Observability configuration."""
    log_level: str = "INFO"
    trace_enabled: bool = True
    trace_file: str = "./logs/traces.jsonl"
    structured_logging: bool = True


@dataclass
class IngestionSettings:
    """Ingestion pipeline configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter: str = "recursive"
    batch_size: int = 100

    # Chunk refiner settings
    chunk_refiner: Dict[str, Any] = field(default_factory=lambda: {"use_llm": True})

    # Metadata enricher settings
    metadata_enricher: Dict[str, Any] = field(default_factory=lambda: {"use_llm": True})


@dataclass
class Settings:
    """Root configuration container."""
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    vision_llm: VisionLLMSettings = field(default_factory=VisionLLMSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)


def load_settings(path: str = "config/settings.yaml") -> Settings:
    """
    Load settings from YAML file.

    Args:
        path: Path to the settings YAML file.

    Returns:
        Settings object with loaded configuration.

    Raises:
        ConfigurationError: If file not found or required fields missing.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    _load_env_files(config_path)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML: {e}") from e

    settings = _parse_settings(data)
    validate_settings(settings)

    return settings


def _load_env_files(config_path: Path) -> None:
    """
    Load .env files when python-dotenv is available.

    This keeps secrets out of tracked YAML while remaining backward-compatible:
    if python-dotenv is not installed, existing shell env vars still work.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return

    candidates = []
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        candidates.append(cwd_env)

    # Typical project layout: config/settings.yaml -> project root/.env
    project_env = config_path.parent.parent / ".env"
    if project_env.exists():
        candidates.append(project_env)

    loaded = set()
    for env_path in candidates:
        resolved = env_path.resolve()
        if resolved in loaded:
            continue
        load_dotenv(dotenv_path=resolved, override=False)
        loaded.add(resolved)


def _parse_settings(data: Dict[str, Any]) -> Settings:
    """Parse raw dict data into Settings dataclass."""
    llm_data = data.get("llm", {})
    embedding_data = data.get("embedding", {})
    vision_llm_data = data.get("vision_llm", {})
    vector_store_data = data.get("vector_store", {})
    retrieval_data = data.get("retrieval", {})
    rerank_data = data.get("rerank", {})
    evaluation_data = data.get("evaluation", {})
    observability_data = data.get("observability", {})
    ingestion_data = data.get("ingestion", {})

    return Settings(
        llm=LLMSettings(
            provider=llm_data.get("provider", "openai"),
            model=llm_data.get("model", "gpt-4o"),
            deployment_name=llm_data.get("deployment_name", ""),
            azure_endpoint=llm_data.get("azure_endpoint", ""),
            api_version=llm_data.get("api_version", ""),
            api_key=llm_data.get("api_key", ""),
            base_url=llm_data.get("base_url", ""),
            temperature=llm_data.get("temperature", 0.0),
            max_tokens=llm_data.get("max_tokens", 4096),
        ),
        embedding=EmbeddingSettings(
            provider=embedding_data.get("provider", "openai"),
            model=embedding_data.get("model", "text-embedding-ada-002"),
            dimensions=embedding_data.get("dimensions", 1536),
            azure_endpoint=embedding_data.get("azure_endpoint", ""),
            deployment_name=embedding_data.get("deployment_name", ""),
            api_version=embedding_data.get("api_version", ""),
            api_key=embedding_data.get("api_key", ""),
            base_url=embedding_data.get("base_url", ""),
        ),
        vision_llm=VisionLLMSettings(
            enabled=vision_llm_data.get("enabled", False),
            provider=vision_llm_data.get("provider", "openai"),
            model=vision_llm_data.get("model", "gpt-4o"),
            azure_endpoint=vision_llm_data.get("azure_endpoint", ""),
            deployment_name=vision_llm_data.get("deployment_name", ""),
            api_version=vision_llm_data.get("api_version", ""),
            api_key=vision_llm_data.get("api_key", ""),
            base_url=vision_llm_data.get("base_url", ""),
            max_image_size=vision_llm_data.get("max_image_size", 2048),
        ),
        vector_store=VectorStoreSettings(
            provider=vector_store_data.get("provider", "chroma"),
            persist_directory=vector_store_data.get("persist_directory", "./data/db/chroma"),
            collection_name=vector_store_data.get("collection_name", "knowledge_hub"),
        ),
        retrieval=RetrievalSettings(
            dense_top_k=retrieval_data.get("dense_top_k", 20),
            sparse_top_k=retrieval_data.get("sparse_top_k", 20),
            fusion_top_k=retrieval_data.get("fusion_top_k", 10),
            rrf_k=retrieval_data.get("rrf_k", 60),
        ),
        rerank=RerankSettings(
            enabled=rerank_data.get("enabled", False),
            provider=rerank_data.get("provider", "none"),
            model=rerank_data.get("model", ""),
            top_k=rerank_data.get("top_k", 5),
        ),
        evaluation=EvaluationSettings(
            enabled=evaluation_data.get("enabled", False),
            provider=evaluation_data.get("provider", "custom"),
            metrics=evaluation_data.get("metrics", ["hit_rate", "mrr", "faithfulness"]),
        ),
        observability=ObservabilitySettings(
            log_level=observability_data.get("log_level", "INFO"),
            trace_enabled=observability_data.get("trace_enabled", True),
            trace_file=observability_data.get("trace_file", "./logs/traces.jsonl"),
            structured_logging=observability_data.get("structured_logging", True),
        ),
        ingestion=IngestionSettings(
            chunk_size=ingestion_data.get("chunk_size", 1000),
            chunk_overlap=ingestion_data.get("chunk_overlap", 200),
            splitter=ingestion_data.get("splitter", "recursive"),
            batch_size=ingestion_data.get("batch_size", 100),
            chunk_refiner=ingestion_data.get("chunk_refiner", {"use_llm": True}),
            metadata_enricher=ingestion_data.get("metadata_enricher", {"use_llm": True}),
        ),
    )


def validate_settings(settings: Settings) -> None:
    """
    Validate that required configuration fields are present.

    Args:
        settings: Settings object to validate.

    Raises:
        ConfigurationError: If required fields are missing or invalid.
    """
    # Validate LLM provider
    if not settings.llm.provider:
        raise ConfigurationError("provider is required", "llm.provider")

    valid_llm_providers = {"openai", "azure", "ollama", "deepseek"}
    if settings.llm.provider not in valid_llm_providers:
        raise ConfigurationError(
            f"invalid provider '{settings.llm.provider}', must be one of {valid_llm_providers}",
            "llm.provider"
        )

    # Validate Embedding provider
    if not settings.embedding.provider:
        raise ConfigurationError("provider is required", "embedding.provider")

    valid_embedding_providers = {"openai", "azure", "ollama"}
    if settings.embedding.provider not in valid_embedding_providers:
        raise ConfigurationError(
            f"invalid provider '{settings.embedding.provider}', must be one of {valid_embedding_providers}",
            "embedding.provider"
        )

    # Validate Vector Store provider
    if not settings.vector_store.provider:
        raise ConfigurationError("provider is required", "vector_store.provider")

    valid_vector_store_providers = {"chroma"}
    if settings.vector_store.provider not in valid_vector_store_providers:
        raise ConfigurationError(
            f"invalid provider '{settings.vector_store.provider}', must be one of {valid_vector_store_providers}",
            "vector_store.provider"
        )

    # Validate Rerank provider if enabled
    if settings.rerank.enabled:
        valid_rerank_providers = {"none", "cross_encoder", "llm"}
        if settings.rerank.provider not in valid_rerank_providers:
            raise ConfigurationError(
                f"invalid provider '{settings.rerank.provider}', must be one of {valid_rerank_providers}",
                "rerank.provider"
            )

    # Validate positive integers
    if settings.retrieval.dense_top_k <= 0:
        raise ConfigurationError("must be a positive integer", "retrieval.dense_top_k")
    if settings.retrieval.sparse_top_k <= 0:
        raise ConfigurationError("must be a positive integer", "retrieval.sparse_top_k")
    if settings.retrieval.fusion_top_k <= 0:
        raise ConfigurationError("must be a positive integer", "retrieval.fusion_top_k")
    if settings.ingestion.chunk_size <= 0:
        raise ConfigurationError("must be a positive integer", "ingestion.chunk_size")
    if settings.ingestion.chunk_overlap < 0:
        raise ConfigurationError("must be a non-negative integer", "ingestion.chunk_overlap")
