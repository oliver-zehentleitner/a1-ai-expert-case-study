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

=====================  =====
Token                  Count
=====================  =====
``&lt;#&gt;``             214
``&lt;DECIMAL&gt;``        21
``&lt;TIME&gt;``            7
``&lt;URL&gt;``             3
``&lt;EMAIL&gt;``           1
=====================  =====

This module is the **single source of truth** for text normalisation.
The raw dataset is never modified; normalisation is applied on demand via
:meth:`~sms_analysis.data.loader.SMSDataset.normalized_texts`.
The sklearn classification pipeline also uses ``preprocess()`` as its first
step, guaranteeing train-serve consistency.

NUS anonymisation tokens — mapped to our own tokens
----------------------------------------------------
``&lt;#&gt;``, ``&lt;DECIMAL&gt;``, ``&lt;TIME&gt;``, ``&lt;URL&gt;``, ``&lt;EMAIL&gt;`` are placeholders
inserted by the NUS team in place of the original content.  The specific value
(which number, which URL) is gone, but the *type* is known and valuable:
the **presence** of a URL or phone number in a message is a meaningful
classification signal.  We map NUS tokens to the same tokens we use for real
user content, so the model learns "message contains a URL" regardless of whether
that URL was typed inline or anonymised by NUS.

For the specific URL or number (reputation, blacklists, IoC matching) a
separate system is needed — that is not a task for text classification.

``&amp;`` is a standard HTML encoding of the literal ``&`` character that *was*
in the original SMS.  It is restored to ``and`` (character normalisation,
not anonymisation).

Tokens produced
---------------
``TOKEN_USER``     – ``@ &lt;URL&gt;`` (NUS) or ``@handle`` (typed) — a mention was present
``TOKEN_URL``      – ``&lt;URL&gt;`` (NUS) or ``http://…`` / ``www.…`` (typed) — a URL was present
``TOKEN_NUM``      – ``&lt;#&gt;`` / ``&lt;DECIMAL&gt;`` (NUS) or digit sequence (typed) — a number was present
``TOKEN_TIME``     – ``&lt;TIME&gt;`` (NUS) — a time value was present
``TOKEN_EMAIL``    – ``&lt;EMAIL&gt;`` (NUS) — an e-mail address was present
``TOKEN_CURRENCY`` – currency symbol (£, $, €, …), ISO code (USD, EUR, GBP, …), or written
                    word (dollar, euro, pounds, …) — locale-normalised

Character normalisation
-----------------------
``Ü`` → ``u``  About 50 HAM messages use ``Ü`` as shorthand for "you" —
semantically identical to the common ``u``.  The exact cause is unknown; one
plausible explanation is a phone with a European keyboard where the U key cycled
to Ü, but this is not verified.  What is clear: ``Ü`` appears 50× in HAM and
0× in SPAM.  Left unnormalised the model would learn ``ü`` → HAM — a spurious
correlation that is trivially exploitable (appending a single ``Ü`` to a SPAM
message could shift the prediction).  Normalising ``Ü`` → ``u`` is semantically
correct and removes this bias.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Pre-compiled patterns — compiled once at import time.
# Order matters: __USER__ must be matched before the general __URL__ pass.
_AT_USER_HTML  = re.compile(r'@\s*&lt;URL&gt;',               re.IGNORECASE)
_AT_USER_RAW   = re.compile(r'@\w+')
_HTML_URL      = re.compile(r'&lt;URL&gt;',                   re.IGNORECASE)
_HTML_NUM      = re.compile(r'&lt;(?:#|DECIMAL|NUMBER)&gt;',  re.IGNORECASE)
_HTML_TIME     = re.compile(r'&lt;TIME&gt;',                  re.IGNORECASE)
_HTML_EMAIL    = re.compile(r'&lt;EMAIL&gt;',                 re.IGNORECASE)
_HTML_AMP      = re.compile(r'&amp;',                         re.IGNORECASE)
_HTML_LEFTOVER = re.compile(r'(&lt;.*?&gt;)',                  re.IGNORECASE)
_REAL_URL      = re.compile(r'https?://\S+|www\.\S+',         re.IGNORECASE)
_CURRENCY      = re.compile(r'[£$€¥₹₽₩₿]')
_CURRENCY_CODE = re.compile(                        # ISO 4217 codes — same signal as a currency symbol
    r'\b(USD|EUR|GBP|CHF|JPY|AUD|CAD|CNY|INR|'
    r'MXN|BRL|KRW|SGD|HKD|NOK|SEK|DKK|NZD|ZAR)\b',
    re.IGNORECASE,
)
_CURRENCY_WORD = re.compile(                        # written-out currency words
    r'\b(dollars?|euros?|pounds?|pence|penny|cents?|quid)\b',
    re.IGNORECASE,
)
# Formatted numbers must be matched BEFORE _STANDALONE_NR to avoid splitting them
# into separate digit groups that the tokeniser would later concatenate into noise.
# e.g. "1,000" → \d+,\d+ → TOKEN_NUM (not TOKEN_NUM,TOKEN_NUM → "token_numtoken_num")
# e.g. "9/4/3"  → would otherwise become "943" after punctuation stripping
_FORMATTED_NR  = re.compile(r'\b\d+(?:[.,]\d+)+\b')   # comma/dot separated: 1,000 / 1.50 / 1,234.56
_SEP_NR        = re.compile(r'\b\d+(?:[/\-]\d+)+\b')  # slash/dash sequences: 9/4/3, 02-09-03
_STANDALONE_NR = re.compile(r'\b\d+\b')  # \b prevents matching digits inside alphanumeric tokens (e.g. "hasi69")
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
        'follow me TOKEN_USER ok?'

        >>> preprocess("Win &lt;URL&gt; now!")
        'Win TOKEN_URL now!'

        >>> preprocess("I sent you &lt;#&gt; bucks")
        'I sent you TOKEN_NUM bucks'

        >>> preprocess("Check http://win-prize.com today")
        'Check TOKEN_URL today'

        >>> preprocess("Good &amp; bad")
        'Good  and  bad'

        >>> preprocess("Win £500 cash now!")
        'Win TOKEN_CURRENCY TOKEN_NUM cash now!'

        >>> preprocess("Claim your $1000 prize")
        'Claim your TOKEN_CURRENCY TOKEN_NUM prize'

        >>> preprocess("c u 2 night 4 free")
        'c u TOKEN_NUM night TOKEN_NUM free'

        >>> preprocess("hasi69 is my username")
        'hasi69 is my username'

        >>> preprocess("Win 1000 EUR now!")
        'Win TOKEN_NUM TOKEN_CURRENCY now!'

        >>> preprocess("Send 500 usd to win")
        'Send TOKEN_NUM TOKEN_CURRENCY to win'

        >>> preprocess("Win £100,000 prize now!")
        'Win TOKEN_CURRENCY TOKEN_NUM prize now!'

        >>> preprocess("awarded on 02/09/03")
        'awarded on TOKEN_NUM'

        >>> preprocess("costs 1.50 per week")
        'costs TOKEN_NUM per week'

        >>> preprocess("win 1000 euros now")
        'win TOKEN_NUM TOKEN_CURRENCY now'
    """
    text = _U_UMLAUT.sub("u", text)              # Ü/ü → u: shorthand for "you", removes spurious HAM bias
    text = _AT_USER_HTML.sub("TOKEN_USER", text)     # @ <URL>: NUS mention placeholder → token (presence matters)
    text = _AT_USER_RAW.sub("TOKEN_USER", text)      # @handle: typed mention → same token
    text = _HTML_URL.sub("TOKEN_URL", text)          # NUS URL placeholder → token (presence matters)
    text = _HTML_NUM.sub("TOKEN_NUM", text)          # NUS number placeholder → token (presence matters)
    text = _HTML_TIME.sub("TOKEN_TIME", text)        # NUS time placeholder → token
    text = _HTML_EMAIL.sub("TOKEN_EMAIL", text)      # NUS email placeholder → token
    text = _HTML_AMP.sub(" and ", text)              # &amp; → and: restores real & character, not a placeholder
    for unknown in _HTML_LEFTOVER.findall(text):
        logger.warning("Unknown HTML-encoded token in preprocessing input: %r", unknown)
    text = _HTML_LEFTOVER.sub("", text)
    text = _REAL_URL.sub("TOKEN_URL", text)
    text = _CURRENCY.sub("TOKEN_CURRENCY ", text)    # before STANDALONE_NR so £500 → TOKEN_CURRENCY TOKEN_NUM
    text = _CURRENCY_CODE.sub("TOKEN_CURRENCY", text)
    text = _CURRENCY_WORD.sub("TOKEN_CURRENCY", text)
    text = _FORMATTED_NR.sub("TOKEN_NUM", text)      # before STANDALONE_NR: 1,000 / 1.50 → single TOKEN_NUM
    text = _SEP_NR.sub("TOKEN_NUM", text)            # before STANDALONE_NR: 9/4/3 → TOKEN_NUM (not "943")
    text = _STANDALONE_NR.sub("TOKEN_NUM", text)
    return text.strip()
