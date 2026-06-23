"""OpenAI embedding provider."""

import logging

from openai import OpenAI

from ..base import EmbeddingProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "text-embedding-3-small"
_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_BATCH_SIZE = 256  # OpenAI accepts up to 2048 inputs per request


class OpenAIProvider(EmbeddingProvider):
    """
    Embed texts using the OpenAI Embeddings API.

    Requires the environment variable ``SMS_OPENAI_API_KEY`` (or pass
    ``api_key`` directly).

    Args:
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        model: Embedding model name (default: ``text-embedding-3-small``).
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches to respect API limits."""
        logger.info("Embedding %d texts via OpenAI (%s)", len(texts), self.model)
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = self._client.embeddings.create(model=self.model, input=batch)
            all_vectors.extend(item.embedding for item in response.data)

        return all_vectors

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS.get(self.model, 1536)
