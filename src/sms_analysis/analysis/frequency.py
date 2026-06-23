"""Word frequency analysis for SMS messages."""

import logging
import string
from collections import Counter
from functools import lru_cache

import nltk
import pandas as pd

from sms_analysis.data.preprocessing import preprocess

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    """Load NLTK stopwords, downloading on first call."""
    try:
        from nltk.corpus import stopwords
        return frozenset(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return frozenset(stopwords.words("english"))


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
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    if remove_stopwords:
        sw = _stopwords()
        tokens = [t for t in tokens if t not in sw]
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
