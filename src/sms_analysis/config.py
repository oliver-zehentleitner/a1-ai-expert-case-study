"""Central configuration via environment variables and .env file."""

import logging
from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class EmbeddingProviderType(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE = "azure"


class Settings(BaseSettings):
    """
    All configuration is read from environment variables (prefix: SMS_)
    or a ``.env`` file in the project root.

    Examples::

        SMS_EMBEDDING_PROVIDER=openai
        SMS_OPENAI_API_KEY=sk-...
    """

    model_config = SettingsConfigDict(
        env_prefix="SMS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General -----------------------------------------------------------------
    embedding_provider: EmbeddingProviderType = EmbeddingProviderType.OLLAMA
    log_level: str = "INFO"

    # --- Paths -------------------------------------------------------------------
    data_path: Path = Path("assignment/3D8026C4.zip/SMSSpamCollection")
    outputs_dir: Path = Path("outputs")

    # --- Ollama (local, no API key required) ------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"

    # --- OpenAI ------------------------------------------------------------------
    openai_api_key: str = ""
    openai_model: str = "text-embedding-3-small"

    # --- Azure OpenAI ------------------------------------------------------------
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "text-embedding-3-small"
    azure_openai_api_version: str = "2024-05-01-preview"


def configure_logging(level: str = "INFO") -> None:
    """Set up structured console logging for the whole package."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
