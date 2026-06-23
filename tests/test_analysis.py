"""Tests for the analysis layer."""

import pytest

from sms_analysis.analysis.comparison import log_odds_ratio
from sms_analysis.analysis.frequency import tokenize, word_frequencies
from sms_analysis.analysis.ngrams import top_ngrams


class TestTokenize:
    def test_lowercases(self):
        assert "hello" in tokenize("HELLO world", remove_stopwords=False)

    def test_strips_punctuation(self):
        tokens = tokenize("Hello, world!", remove_stopwords=False)
        assert "hello" in tokens
        assert "," not in tokens

    def test_removes_stopwords_by_default(self):
        tokens = tokenize("the quick brown fox")
        assert "the" not in tokens
        assert "fox" in tokens

    def test_keep_stopwords(self):
        tokens = tokenize("the quick brown fox", remove_stopwords=False)
        assert "the" in tokens


class TestWordFrequencies:
    def test_returns_top_n(self):
        texts = ["hello world", "hello python", "world is great"]
        df = word_frequencies(texts, top_n=2, remove_stopwords=False)
        assert len(df) == 2

    def test_most_common_first(self):
        texts = ["spam spam spam", "ham"]
        df = word_frequencies(texts, top_n=5, remove_stopwords=False)
        assert df.iloc[0]["word"] == "spam"

    def test_columns(self):
        df = word_frequencies(["hello world"])
        assert list(df.columns) == ["word", "count"]


class TestLogOddsRatio:
    def test_spam_words_positive(self):
        ham = ["hello friend how are you"] * 10
        spam = ["win free prize money now"] * 10
        df = log_odds_ratio(ham, spam, top_n=1)
        spam_rows = df[df["leaning"] == "spam"]
        assert all(spam_rows["log_odds"] > 0)

    def test_ham_words_negative(self):
        ham = ["hello friend how are you"] * 10
        spam = ["win free prize money now"] * 10
        df = log_odds_ratio(ham, spam, top_n=1)
        ham_rows = df[df["leaning"] == "ham"]
        assert all(ham_rows["log_odds"] < 0)

    def test_output_columns(self):
        df = log_odds_ratio(["hello"], ["spam"])
        assert "word" in df.columns
        assert "log_odds" in df.columns
        assert "leaning" in df.columns


class TestTopNgrams:
    def test_returns_dataframe_with_correct_columns(self):
        texts = ["free prize win now", "call me back please"]
        df = top_ngrams(texts, n=2, top_k=5)
        assert "ngram" in df.columns
        assert "count" in df.columns

    def test_top_k_respected(self):
        texts = ["free prize win now call back"] * 5
        df = top_ngrams(texts, n=2, top_k=3)
        assert len(df) <= 3
