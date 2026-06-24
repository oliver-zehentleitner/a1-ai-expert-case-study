"""
Text preprocessing for the SMS Spam Collection dataset.

HTML-encoded tokens such as ``&lt;URL&gt;`` and ``&lt;#&gt;`` appear exclusively in HAM
messages and originate from the NUS SMS Corpus (National University of Singapore).
Before publishing their dataset, the NUS team replaced sensitive values —
phone numbers, amounts, URLs, times, e-mail addresses — with typed HTML-encoded
placeholders to protect the privacy of contributors.  SPAM messages (sourced
from Grumbletext and other collections) were not anonymised in the same way and
retain real phone numbers, prices, and links.

Verified token inventory (all HAM-only):

=================  =====
Token              Count
=================  =====
``&lt;#&gt;``         214
``&lt;DECIMAL&gt;``    21
``&lt;TIME&gt;``        7
``&lt;URL&gt;``         3
``&lt;EMAIL&gt;``       1
=================  =====

This module is the **single source of truth** for text normalisation.
The raw dataset is never modified; normalisation is applied on demand via
:meth:`~sms_analysis.data.loader.SMSDataset.normalized_texts`.
The sklearn classification pipeline also uses ``preprocess()`` as its first
step, guaranteeing train-serve consistency.

Tokens produced
---------------
``__USER__``      – ``@ &lt;URL&gt;`` or ``@handle`` (username / social-media mention)
``__URL__``       – URL placeholder or real URL
``__NUM__``       – numeric / decimal / hash placeholder or standalone number
``__TIME__``      – time placeholder from NUS anonymisation
``__EMAIL__``     – e-mail address placeholder from NUS anonymisation
``__CURRENCY__``  – any currency symbol (£, $, €, ¥, ₹, …), normalised to remove
                    locale bias; amounts following the symbol are caught by ``__NUM__``

Character normalisation
-----------------------
``Ü`` → ``u``  The NUS SMS Corpus contains ~50 HAM messages where one or a few
contributors used ``Ü`` as SMS shorthand for "you" — identical in meaning to the
common ``u``.  The likely cause is a Nokia phone with a European language pack
where the ``U`` key cycled to ``Ü`` on repeat press.  In this dataset ``Ü``
appears 50× in HAM and 0× in SPAM.  Keeping it as a distinct token would create
a spurious correlation: the model would learn ``ü`` → HAM, making it trivially
exploitable — a single ``Ü`` appended to a SPAM message could flip the
classification.  Normalising ``Ü`` → ``u`` is semantically correct and closes
this adversarial attack surface.
"""

import re

# Pre-compiled patterns — compiled once at import time.
# Order matters: __USER__ must be matched before the general __URL__ pass.
_AT_USER_HTML  = re.compile(r'@\s*&lt;URL&gt;',               re.IGNORECASE)
_AT_USER_RAW   = re.compile(r'@\w+')
_HTML_URL      = re.compile(r'&lt;URL&gt;',                   re.IGNORECASE)
_HTML_NUM      = re.compile(r'&lt;(?:#|DECIMAL|NUMBER)&gt;',  re.IGNORECASE)
_HTML_TIME     = re.compile(r'&lt;TIME&gt;',                  re.IGNORECASE)
_HTML_EMAIL    = re.compile(r'&lt;EMAIL&gt;',                 re.IGNORECASE)
_HTML_AMP      = re.compile(r'&amp;',                         re.IGNORECASE)
_HTML_LEFTOVER = re.compile(r'&lt;.*?&gt;',                   re.IGNORECASE)
_REAL_URL      = re.compile(r'https?://\S+|www\.\S+',         re.IGNORECASE)
_CURRENCY      = re.compile(r'[£$€¥₹₽₩₿]')
_STANDALONE_NR = re.compile(r'\b\d{2,}\b')
_U_UMLAUT      = re.compile(r'Ü', re.IGNORECASE)


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

        >>> preprocess("Win £500 cash now!")
        'Win __CURRENCY__ __NUM__ cash now!'

        >>> preprocess("Claim your $1000 prize")
        'Claim your __CURRENCY__ __NUM__ prize'
    """
    text = _U_UMLAUT.sub("u", text)               # Ü/ü → u: Nokia keyboard artifact, closes spurious HAM signal
    text = _AT_USER_HTML.sub("__USER__", text)   # @ <URL> → username, before general URL pass
    text = _AT_USER_RAW.sub("__USER__", text)    # @handle not caught by HTML pass
    text = _HTML_URL.sub("__URL__", text)
    text = _HTML_NUM.sub("__NUM__", text)
    text = _HTML_TIME.sub("__TIME__", text)
    text = _HTML_EMAIL.sub("__EMAIL__", text)
    text = _HTML_AMP.sub(" and ", text)
    text = _HTML_LEFTOVER.sub("", text)
    text = _REAL_URL.sub("__URL__", text)
    text = _CURRENCY.sub("__CURRENCY__ ", text)  # before STANDALONE_NR so £500 → __CURRENCY__ __NUM__
    text = _STANDALONE_NR.sub("__NUM__", text)
    return text.strip()
