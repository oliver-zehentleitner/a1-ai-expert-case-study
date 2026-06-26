"""Central configuration via environment variables and .env file."""

import logging
from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class EmbeddingProviderType(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE = "azure"


# Resolve paths relative to this file (project root), not CWD — the CLI may
# be invoked from any subdirectory (outputs/, notebooks/, …).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """
    Environment-driven configuration (prefix: ``SMS_``).

    Provider selection is a CLI concern (``--provider`` flag), not an env var.
    Only secrets and system-level settings live here.

    Examples::

        SMS_OPENAI_API_KEY=sk-...
        SMS_AZURE_OPENAI_API_KEY=...
        SMS_AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1
    """

    model_config = SettingsConfigDict(
        env_prefix="SMS_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General -----------------------------------------------------------------
    log_level: str = "INFO"

    # --- Paths -------------------------------------------------------------------
    data_path: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / "assignment/3D8026C4.zip/SMSSpamCollection"
    )
    outputs_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "outputs")

    # --- Ollama (local, no API key required) ------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"

    # --- OpenAI ------------------------------------------------------------------
    openai_api_key: str = ""
    openai_model: str = "text-embedding-3-small"

    # --- Azure AI Foundry --------------------------------------------------------
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""   # full /openai/v1 URL
    azure_openai_model: str = "Cohere-embed-v3-multilingual"


def configure_logging(level: str = "INFO") -> None:
    """Set up structured console logging for the whole package."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)   # routes warnings through logging, no source-snippet noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
