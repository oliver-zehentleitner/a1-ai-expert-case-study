"""End-to-end tests for the classification pipeline.

Covers training, prediction, serialisation, and evaluation — the code paths
most likely to break when internals change.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sms_analysis.classification import SpamClassifier, cross_validate, evaluate
from sms_analysis.classification.pipeline import _build_pipeline

# Minimal labelled corpus — not meant to be realistic, just enough for a
# pipeline to fit and produce non-trivial outputs.
HAM_TEXTS = [
    "Hey, are you coming to dinner tonight?",
    "Let me know when you arrive.",
    "Can you pick up milk on the way home?",
    "I'll be there in 10 minutes.",
    "See you at the cinema at 8.",
    "Thanks for the birthday wishes!",
    "Call me when you're free.",
    "Good morning, how are you?",
    "Running a bit late, sorry.",
    "Did you get my last message?",
]
SPAM_TEXTS = [
    "WINNER! Claim your FREE £500 prize now. Call 0800123456.",
    "Congratulations! You have won a cash reward. Reply WIN to claim.",
    "FREE entry to win a brand new iPhone. Text APPLE to 83000.",
    "URGENT: Your account will be suspended. Call 09001234567 immediately.",
    "You have been selected for a special prize. Visit www.win-prize.com now.",
]

ALL_TEXTS = HAM_TEXTS + SPAM_TEXTS
ALL_LABELS = ["ham"] * len(HAM_TEXTS) + ["spam"] * len(SPAM_TEXTS)


class TestSpamClassifier:
    def test_fit_predict(self):
        clf = SpamClassifier("logistic_regression").fit(ALL_TEXTS, ALL_LABELS)
        preds = clf.predict(ALL_TEXTS)
        assert len(preds) == len(ALL_TEXTS)
        assert set(preds).issubset({"ham", "spam"})

    def test_training_spam_classified(self):
        # Sanity-check that the pipeline learns the training distribution.
        # These are from ALL_TEXTS, not unseen data — the goal is to verify
        # that fit() actually works, not that the model generalises.
        clf = SpamClassifier("logistic_regression").fit(ALL_TEXTS, ALL_LABELS)
        result = clf.predict([SPAM_TEXTS[0]])
        assert result[0] == "spam"

    def test_training_ham_classified(self):
        clf = SpamClassifier("logistic_regression").fit(ALL_TEXTS, ALL_LABELS)
        result = clf.predict([HAM_TEXTS[0]])
        assert result[0] == "ham"

    def test_predict_proba_shape(self):
        clf = SpamClassifier("logistic_regression").fit(ALL_TEXTS, ALL_LABELS)
        proba = clf.predict_proba(ALL_TEXTS[:3])
        assert proba.shape == (3, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_top_features_returns_spam_and_ham(self):
        clf = SpamClassifier("logistic_regression").fit(ALL_TEXTS, ALL_LABELS)
        features = clf.top_features(n=5)
        assert "spam" in features and "ham" in features
        assert len(features["spam"]) == 5

    def test_save_load_roundtrip(self):
        clf = SpamClassifier("logistic_regression").fit(ALL_TEXTS, ALL_LABELS)
        original_preds = clf.predict(ALL_TEXTS)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            clf.save(model_path)
            loaded = SpamClassifier.load(model_path)

        loaded_preds = loaded.predict(ALL_TEXTS)
        assert list(original_preds) == list(loaded_preds)
        assert loaded.model_type == "logistic_regression"

    def test_predict_before_fit_raises(self):
        clf = SpamClassifier("logistic_regression")
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(["hello"])

    def test_naive_bayes_variant(self):
        clf = SpamClassifier("naive_bayes").fit(ALL_TEXTS, ALL_LABELS)
        preds = clf.predict(ALL_TEXTS)
        assert len(preds) == len(ALL_TEXTS)


class TestEvaluate:
    def _get_preds(self):
        clf = SpamClassifier("logistic_regression").fit(ALL_TEXTS, ALL_LABELS)
        return (
            ALL_LABELS,
            clf.predict(ALL_TEXTS),
            clf.predict_proba(ALL_TEXTS),
        )

    def test_evaluate_returns_result(self):
        y_true, y_pred, y_proba = self._get_preds()
        result = evaluate(y_true, y_pred, y_proba)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.f1_spam <= 1.0
        assert 0.0 <= result.roc_auc <= 1.0

    def test_evaluate_without_proba(self):
        y_true, y_pred, _ = self._get_preds()
        result = evaluate(y_true, y_pred)
        assert np.isnan(result.roc_auc)

    def test_evaluate_confusion_shape(self):
        y_true, y_pred, y_proba = self._get_preds()
        result = evaluate(y_true, y_pred, y_proba)
        assert result.confusion.shape == (2, 2)

    def test_summary_string_format(self):
        y_true, y_pred, y_proba = self._get_preds()
        summary = evaluate(y_true, y_pred, y_proba).summary()
        for key in ("Accuracy", "F1-SPAM", "Precision", "Recall", "ROC-AUC"):
            assert key in summary


class TestCrossValidate:
    def test_cv_returns_expected_keys(self):
        pipeline = _build_pipeline("logistic_regression")
        result = cross_validate(pipeline, ALL_TEXTS, ALL_LABELS, n_splits=3)
        assert "f1_mean" in result and "f1_std" in result

    def test_cv_f1_in_range(self):
        pipeline = _build_pipeline("logistic_regression")
        result = cross_validate(pipeline, ALL_TEXTS, ALL_LABELS, n_splits=3)
        assert 0.0 <= result["f1_mean"] <= 1.0
        assert result["f1_std"] >= 0.0
