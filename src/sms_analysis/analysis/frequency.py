"""Word frequency analysis for SMS messages."""

import logging
import string
from collections import Counter
from pathlib import Path

import pandas as pd

from sms_analysis.data.preprocessing import preprocess

logger = logging.getLogger(__name__)

_STOPWORDS_FILE = Path(__file__).parent / "stopwords.txt"


def _load_stopwords() -> frozenset[str]:
    words: set[str] = set()
    for line in _STOPWORDS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line.lower())
    return frozenset(words)


_STOPWORDS: frozenset[str] = _load_stopwords()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Lowercase, strip punctuation, split on whitespace.

    Args:
        text: Raw SMS text.
        remove_stopwords: Drop common English stopwords when True.

    Returns:
        List of clean tokens.
    """
    text = preprocess(text)
    text = text.lower().translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def word_frequencies(
    texts: list[str],
    top_n: int = 20,
    remove_stopwords: bool = True,
) -> pd.DataFrame:
    """
    Count word occurrences across a list of texts.

    Args:
        texts: Raw SMS texts to analyse.
        top_n: How many top words to return.
        remove_stopwords: Whether to exclude English stopwords.

    Returns:
        DataFrame with columns ``word`` and ``count``, sorted descending.
    """
    tokens: list[str] = []
    for text in texts:
        tokens.extend(tokenize(text, remove_stopwords=remove_stopwords))

    counts = Counter(tokens)
    df = pd.DataFrame(counts.most_common(top_n), columns=["word", "count"])
    logger.info(
        "Top %d words from %d texts (%d total tokens)",
        top_n, len(texts), len(tokens),
    )
    return df
