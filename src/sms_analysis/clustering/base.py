"""Abstract embedding provider interface — the architectural core of the GenAI layer."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Common interface for all embedding backends.

    The clustering layer depends *only* on this interface.
    Swap Ollama → OpenAI → Azure by changing the provider — zero changes
    to clustering logic.

    Implementing a new provider requires exactly two things:
    1. Override :meth:`embed` to return vectors.
    2. Override the :attr:`dimensions` property.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts into dense float vectors.

        Args:
            texts: Non-empty list of raw text strings.

        Returns:
            List of embedding vectors in the same order as ``texts``.
            Each vector has :attr:`dimensions` elements.
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding vector dimensionality (e.g. 768, 1536)."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper — embed a single text."""
        return self.embed([text])[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimensions={self.dimensions})"
