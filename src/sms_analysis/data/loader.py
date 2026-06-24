"""SMS dataset loading and Pydantic domain models."""

import logging
from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, field_validator

from .preprocessing import preprocess

logger = logging.getLogger(__name__)


class Label(str, Enum):
    HAM = "ham"
    SPAM = "spam"


class SMSMessage(BaseModel):
    """A single labelled SMS message."""

    label: Label
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("SMS text must not be empty")
        return v

    @property
    def is_spam(self) -> bool:
        return self.label == Label.SPAM


class SMSDataset(BaseModel):
    """The full SMS Spam Collection dataset with convenience accessors."""

    messages: list[SMSMessage]

    @property
    def ham(self) -> list[SMSMessage]:
        return [m for m in self.messages if m.label == Label.HAM]

    @property
    def spam(self) -> list[SMSMessage]:
        return [m for m in self.messages if m.label == Label.SPAM]

    @property
    def texts(self) -> list[str]:
        return [m.text for m in self.messages]

    @property
    def normalized_texts(self) -> list[str]:
        """Raw texts passed through the normalization pipeline on demand."""
        return [preprocess(m.text) for m in self.messages]

    @property
    def normalized_ham_texts(self) -> list[str]:
        """Normalized texts for HAM messages only."""
        return [preprocess(m.text) for m in self.ham]

    @property
    def normalized_spam_texts(self) -> list[str]:
        """Normalized texts for SPAM messages only."""
        return [preprocess(m.text) for m in self.spam]

    @property
    def deduplicated(self) -> "SMSDataset":
        """Return a new dataset with exact duplicate texts removed (first occurrence kept).

        The raw file contains 403 duplicate rows (309 HAM / 94 SPAM) with no
        conflicting labels. Deduplication prevents the same message from appearing
        in both train and test splits, which would inflate evaluation metrics.
        """
        seen: set[str] = set()
        unique: list[SMSMessage] = []
        for m in self.messages:
            if m.text not in seen:
                seen.add(m.text)
                unique.append(m)
        return SMSDataset(messages=unique)

    @property
    def labels(self) -> list[str]:
        return [m.label.value for m in self.messages]

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with columns ``label`` and ``text``."""
        return pd.DataFrame({"label": self.labels, "text": self.texts})

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"SMSDataset({len(self)} messages: "
            f"{len(self.ham)} ham / {len(self.spam)} spam)"
        )


def load_dataset(path: Path) -> SMSDataset:
    """
    Load the SMS Spam Collection from a tab-separated file.

    The file format is one message per line::

        ham\\tGo until jurong point...
        spam\\tFREE entry in 2 a wkly comp...

    Args:
        path: Path to the ``SMSSpamCollection`` file.

    Returns:
        Validated :class:`SMSDataset`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains unexpected data.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=3)

    messages = [
        SMSMessage(label=row.label, text=row.text)
        for _, row in df.iterrows()
    ]

    dataset = SMSDataset(messages=messages)
    logger.info(
        "Loaded %d messages (%d ham / %d spam)",
        len(dataset),
        len(dataset.ham),
        len(dataset.spam),
    )
    return dataset
