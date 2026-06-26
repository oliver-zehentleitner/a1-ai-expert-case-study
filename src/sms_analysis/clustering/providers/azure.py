"""Azure AI Foundry embedding provider.

Azure AI Foundry exposes an OpenAI-compatible /openai/v1 endpoint where model
IDs are used directly — no separate deployment setup required.  Multilingual
models such as Cohere-embed-v3-multilingual are available out of the box.
"""

import logging

from openai import OpenAI

from ..base import EmbeddingProvider

logger = logging.getLogger(__name__)

_BATCH_SIZE = 96   # Cohere models in Azure AI Foundry cap at 96 texts per request


class AzureOpenAIProvider(EmbeddingProvider):
    """
    Embed texts via Azure AI Foundry's OpenAI-compatible endpoint.

    Required environment variables (prefix ``SMS_``)::

        SMS_AZURE_OPENAI_API_KEY=...
        SMS_AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1

    Args:
        api_key: Azure API key.
        endpoint: Full ``/openai/v1`` base URL of the Azure AI Foundry resource.
        model: Model ID available in Azure AI Foundry catalog.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str = "Cohere-embed-v3-multilingual",
    ) -> None:
        self.model = model
        # Normalise: ensure endpoint ends with /openai/v1
        base = endpoint.rstrip("/")
        if not base.endswith("/openai/v1"):
            base = f"{base}/openai/v1"
        self._client = OpenAI(api_key=api_key, base_url=f"{base}/")
        self._dims: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches via Azure AI Foundry."""
        logger.info("Embedding %d texts via Azure AI Foundry (%s)", len(texts), self.model)
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = self._client.embeddings.create(model=self.model, input=batch)
            all_vectors.extend(item.embedding for item in response.data)

        if self._dims is None and all_vectors:
            self._dims = len(all_vectors[0])

        return all_vectors

    @property
    def dimensions(self) -> int:
        return self._dims or 1024
