"""HAM vs SPAM word comparison using log-odds ratio."""

import logging
from collections import Counter

import numpy as np
import pandas as pd

from .frequency import tokenize

logger = logging.getLogger(__name__)


def log_odds_ratio(
    ham_texts: list[str],
    spam_texts: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Rank words by how strongly they lean towards SPAM or HAM.

    Uses Laplace (add-one) smoothing so unseen words don't cause divide-by-zero.
    A positive ``log_odds`` means the word leans towards SPAM;
    negative means it leans towards HAM.

    Args:
        ham_texts: List of HAM message texts.
        spam_texts: List of SPAM message texts.
        top_n: Number of top words to return *per class*.

    Returns:
        DataFrame with columns
        ``word``, ``log_odds``, ``ham_count``, ``spam_count``, ``leaning``.
        Contains ``top_n`` HAM-leaning rows and ``top_n`` SPAM-leaning rows.
    """
    ham_counts: Counter[str] = Counter()
    for text in ham_texts:
        ham_counts.update(tokenize(text))

    spam_counts: Counter[str] = Counter()
    for text in spam_texts:
        spam_counts.update(tokenize(text))

    vocab = set(ham_counts) | set(spam_counts)
    n_ham = sum(ham_counts.values())
    n_spam = sum(spam_counts.values())
    v = len(vocab)

    rows = []
    for word in vocab:
        p_ham = (ham_counts[word] + 1) / (n_ham + v)
        p_spam = (spam_counts[word] + 1) / (n_spam + v)
        rows.append(
            {
                "word": word,
                "log_odds": float(np.log(p_spam / p_ham)),
                "ham_count": ham_counts[word],
                "spam_count": spam_counts[word],
            }
        )

    df = pd.DataFrame(rows).sort_values("log_odds")

    top_ham = df.head(top_n).copy()
    top_ham["leaning"] = "ham"

    top_spam = df.tail(top_n).sort_values("log_odds", ascending=False).copy()
    top_spam["leaning"] = "spam"

    result = pd.concat([top_ham, top_spam], ignore_index=True)
    logger.info(
        "Log-odds computed over vocab of %d words (%d HAM / %d SPAM texts)",
        v, len(ham_texts), len(spam_texts),
    )
    return result
