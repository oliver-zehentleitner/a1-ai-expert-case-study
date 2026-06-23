"""
Text preprocessing for the SMS Spam Collection dataset.

Several messages contain HTML-encoded tokens such as ``&lt;URL&gt;`` and ``&lt;#&gt;``.
Their origin is unclear (researcher anonymisation or smartphone copy-paste
artefacts) and would need to be verified against the original collection notes.
Some messages also contain raw URLs (``http://…``).

This module normalises all observed forms to consistent semantic tokens so that
the model sees the same representation at training time and at inference time.
Skipping this step at inference would cause train-serve skew.

Tokens produced
---------------
``__USER__``  – ``@ &lt;URL&gt;`` or ``@handle`` (username / social-media mention)
``__URL__``   – URL placeholder or real URL
``__NUM__``   – numeric placeholder or standalone number
"""

import re

# Pre-compiled patterns — compiled once at import time.
# Order matters: __USER__ must be matched before the general __URL__ pass.
_AT_USER_HTML  = re.compile(r'@\s*&lt;URL&gt;',               re.IGNORECASE)
_AT_USER_RAW   = re.compile(r'@\w+')
_HTML_URL      = re.compile(r'&lt;URL&gt;',                   re.IGNORECASE)
_HTML_NUM      = re.compile(r'&lt;(?:#|DECIMAL|NUMBER)&gt;',  re.IGNORECASE)
_HTML_AMP      = re.compile(r'&amp;',                         re.IGNORECASE)
_HTML_LEFTOVER = re.compile(r'&lt;.*?&gt;',                   re.IGNORECASE)
_REAL_URL      = re.compile(r'https?://\S+|www\.\S+',         re.IGNORECASE)
_STANDALONE_NR = re.compile(r'\b\d{2,}\b')


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

        >>> preprocess("follow me @ &lt;URL&gt; ok?")
        'follow me __USER__ ok?'

        >>> preprocess("Win &lt;URL&gt; now!")
        'Win __URL__ now!'

        >>> preprocess("I sent you &lt;#&gt; bucks")
        'I sent you __NUM__ bucks'

        >>> preprocess("Check http://win-prize.com today")
        'Check __URL__ today'

        >>> preprocess("Good &amp; bad")
        'Good and bad'
    """
    text = _AT_USER_HTML.sub("__USER__", text)   # @ <URL> → username, before general URL pass
    text = _AT_USER_RAW.sub("__USER__", text)    # @handle not caught by HTML pass
    text = _HTML_URL.sub("__URL__", text)
    text = _HTML_NUM.sub("__NUM__", text)
    text = _HTML_AMP.sub(" and ", text)
    text = _HTML_LEFTOVER.sub("", text)
    text = _REAL_URL.sub("__URL__", text)
    text = _STANDALONE_NR.sub("__NUM__", text)
    return text.strip()
