"""
Text preprocessing for the SMS Spam Collection dataset.

The dataset was anonymised by its authors: numbers, decimals, and URLs were
replaced with HTML-encoded placeholders such as ``&lt;#&gt;`` and ``&lt;URL&gt;``.
Raw URLs in non-anonymised messages were left in place.

This module normalises both forms to consistent semantic tokens so that the
model sees the same representation at training time and at inference time.
Skipping this step at inference would cause train-serve skew: the model learns
``__URL__`` as a spam signal but never sees it in production.

Tokens produced
---------------
``__URL__``   – any URL placeholder or real URL
``__NUM__``   – any numeric placeholder or standalone number
``__AMP__``   – HTML-encoded ampersand (``&amp;``)
"""

import re

# Pre-compiled patterns — compiled once at import time
_HTML_URL      = re.compile(r'&lt;URL&gt;',                   re.IGNORECASE)
_HTML_NUM      = re.compile(r'&lt;(?:#|DECIMAL|NUMBER)&gt;',  re.IGNORECASE)
_HTML_AMP      = re.compile(r'&amp;',                         re.IGNORECASE)
_HTML_LEFTOVER = re.compile(r'&lt;.*?&gt;',                   re.IGNORECASE)
_REAL_URL      = re.compile(r'https?://\S+|www\.\S+',         re.IGNORECASE)
_STANDALONE_NR = re.compile(r'\b\d{2,}\b')   # numbers ≥ 2 digits (prices, dates, …)


def preprocess(text: str) -> str:
    """
    Normalise a raw SMS text to a clean, model-ready string.

    Applies the same transformations at training time and at inference time,
    preventing train-serve skew.

    Args:
        text: Raw SMS text, possibly containing HTML entities or real URLs.

    Returns:
        Normalised text with semantic placeholder tokens.

    Examples::

        >>> preprocess("Win &lt;URL&gt; now!")
        'Win __URL__ now!'

        >>> preprocess("I sent you &lt;#&gt; bucks")
        'I sent you __NUM__ bucks'

        >>> preprocess("Check http://win-prize.com today")
        'Check __URL__ today'

        >>> preprocess("Good &amp; bad")
        'Good and bad'
    """
    text = _HTML_URL.sub("__URL__", text)
    text = _HTML_NUM.sub("__NUM__", text)
    text = _HTML_AMP.sub(" and ", text)
    text = _HTML_LEFTOVER.sub("", text)       # strip any remaining &lt;…&gt; noise
    text = _REAL_URL.sub("__URL__", text)
    text = _STANDALONE_NR.sub("__NUM__", text)
    return text.strip()
