"""Ollama local embedding provider — no API key, runs on your machine."""

import logging

import ollama as _ollama

from ..base import EmbeddingProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "nomic-embed-text"
_DEFAULT_DIMENSIONS = 768  # nomic-embed-text output size


class OllamaProvider(EmbeddingProvider):
    """
    Embed texts using a locally running Ollama instance.

    Ollama must be installed and the model pulled before use::

        scripts/setup_ollama.sh   # installs Ollama + pulls nomic-embed-text

    Args:
        model: Ollama model name (default: ``nomic-embed-text``).
        base_url: Ollama API base URL (default: ``http://localhost:11434``).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self._client = _ollama.Client(host=base_url)
        self._dims: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts one by one (Ollama has no batch endpoint for embeddings)."""
        logger.info("Embedding %d texts via Ollama (%s)", len(texts), self.model)
        vectors = []
        for text in texts:
            response = self._client.embeddings(model=self.model, prompt=text)
            vectors.append(response["embedding"])
        if self._dims is None and vectors:
            self._dims = len(vectors[0])
        return vectors

    @property
    def dimensions(self) -> int:
        return self._dims or _DEFAULT_DIMENSIONS
