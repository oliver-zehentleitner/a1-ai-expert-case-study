"""Azure OpenAI embedding provider.

Azure OpenAI uses the same Python client as OpenAI — only the instantiation
differs (endpoint + api_version instead of a global base URL).  This is a
clean example of how the provider abstraction pays off: identical logic,
different config.
"""

import logging

from openai import AzureOpenAI

from ..base import EmbeddingProvider

logger = logging.getLogger(__name__)

_BATCH_SIZE = 256


class AzureOpenAIProvider(EmbeddingProvider):
    """
    Embed texts via Azure OpenAI Service.

    Required environment variables (prefix ``SMS_``)::

        SMS_AZURE_OPENAI_API_KEY=...
        SMS_AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
        SMS_AZURE_OPENAI_DEPLOYMENT=text-embedding-3-small

    Args:
        api_key: Azure OpenAI API key.
        endpoint: Azure resource endpoint URL.
        deployment: Deployment / model name in your Azure resource.
        api_version: Azure OpenAI API version string.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str = "2024-05-01-preview",
    ) -> None:
        self.deployment = deployment
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._dims: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches via Azure OpenAI."""
        logger.info("Embedding %d texts via Azure OpenAI (%s)", len(texts), self.deployment)
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = self._client.embeddings.create(model=self.deployment, input=batch)
            all_vectors.extend(item.embedding for item in response.data)

        if self._dims is None and all_vectors:
            self._dims = len(all_vectors[0])

        return all_vectors

    @property
    def dimensions(self) -> int:
        return self._dims or 1536
