"""SMS Spam/Ham Analysis – A1 AI Expert Case Study by Oliver Zehentleitner."""

from .config import EmbeddingProviderType, Settings
from .data.loader import Label, SMSDataset, SMSMessage, load_dataset

__version__ = "0.1.0"
__all__ = [
    "Settings",
    "EmbeddingProviderType",
    "Label",
    "SMSDataset",
    "SMSMessage",
    "load_dataset",
]
