"""N-gram frequency analysis."""

import logging

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


def top_ngrams(
    texts: list[str],
    n: int = 2,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Extract the most frequent n-grams from a list of texts.

    Uses scikit-learn's CountVectorizer so it benefits from the same
    English stopword list used elsewhere.

    Args:
        texts: Raw SMS texts.
        n: N-gram size (2 = bigrams, 3 = trigrams).
        top_k: Number of top n-grams to return.

    Returns:
        DataFrame with columns ``ngram`` and ``count``, sorted descending.
    """
    vec = CountVectorizer(
        ngram_range=(n, n),
        stop_words="english",
        max_features=top_k * 10,  # over-sample, then trim
    )
    X = vec.fit_transform(texts)
    totals = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()

    df = (
        pd.DataFrame({"ngram": vocab, "count": totals})
        .nlargest(top_k, "count")
        .reset_index(drop=True)
    )
    logger.info(
        "Top %d %d-grams from %d texts",
        top_k, n, len(texts),
    )
    return df
