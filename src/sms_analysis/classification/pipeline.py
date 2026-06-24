"""SPAM/HAM text classification pipeline."""

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sms_analysis.data.preprocessing import preprocess


def _preprocess_batch(texts: list[str]) -> list[str]:
    """Apply preprocessing to a list of texts (module-level so joblib can pickle it)."""
    return [preprocess(t) for t in texts]

logger = logging.getLogger(__name__)

ModelType = Literal["logistic_regression", "naive_bayes"]


def _build_pipeline(model_type: ModelType) -> Pipeline:
    """
    Assemble a scikit-learn Pipeline: TF-IDF vectoriser → classifier.

    TF-IDF settings:
    - ``sublinear_tf=True``: applies log(1 + tf) to dampen the effect of
      very frequent terms (e.g. the word "call" appears 50× in one spam but
      the raw count overshadows the signal — log-scaling helps).
    - ``ngram_range=(1, 2)``: include both unigrams *and* bigrams so the
      model can capture phrases like "free entry" or "win prize".
    """
    vectoriser = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10_000,
        sublinear_tf=True,
    )

    if model_type == "logistic_regression":
        classifier: LogisticRegression | MultinomialNB = LogisticRegression(
            solver="liblinear",  # fast and stable for sparse high-dimensional TF-IDF
            max_iter=1_000,
            C=1.0,
            random_state=42,
        )
    elif model_type == "naive_bayes":
        classifier = MultinomialNB()
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    return Pipeline([
        ("preprocess", FunctionTransformer(_preprocess_batch)),
        ("tfidf", vectoriser),
        ("clf", classifier),
    ])


class SpamClassifier:
    """
    Train, persist, and serve a SPAM/HAM text classifier.

    Wraps a scikit-learn Pipeline so callers never interact with sklearn
    internals directly.

    Example::

        clf = SpamClassifier("logistic_regression")
        clf.fit(dataset.texts, dataset.labels)
        predictions = clf.predict(["Win a free iPhone now!"])
    """

    def __init__(self, model_type: ModelType = "logistic_regression") -> None:
        self.model_type: ModelType = model_type
        self._pipeline: Pipeline | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "SpamClassifier":
        """Train the classifier on labelled texts."""
        self._pipeline = _build_pipeline(self.model_type)
        self._pipeline.fit(texts, labels)
        logger.info("Trained %s on %d samples", self.model_type, len(texts))
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return predicted class labels (``'ham'`` or ``'spam'``)."""
        return self._require_pipeline().predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return ``[p_ham, p_spam]`` probability pairs per text."""
        return self._require_pipeline().predict_proba(texts)

    def top_features(self, n: int = 20) -> dict[str, list[tuple[str, float]]]:
        """
        Return the ``n`` words with the highest coefficient magnitude per class.

        Only available for ``logistic_regression``.
        These coefficients are what you show in the presentation: "these words
        push the model towards SPAM, these towards HAM".
        """
        if self.model_type != "logistic_regression":
            raise ValueError("top_features() is only available for logistic_regression")

        pipeline = self._require_pipeline()
        vocab = pipeline.named_steps["tfidf"].get_feature_names_out()
        coefs = pipeline.named_steps["clf"].coef_[0]  # positive = spam

        idx = coefs.argsort()
        return {
            "ham": [(vocab[i], float(coefs[i])) for i in idx[:n]],
            "spam": [(vocab[i], float(coefs[i])) for i in idx[-n:][::-1]],
        }

    def save(self, path: Path) -> None:
        """Persist the trained pipeline to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_type": self.model_type, "pipeline": self._require_pipeline()}, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "SpamClassifier":
        """Load a previously saved pipeline from disk."""
        data = joblib.load(path)
        if isinstance(data, dict):
            instance = cls(model_type=data["model_type"])
            instance._pipeline = data["pipeline"]
        else:
            instance = cls(model_type="logistic_regression")
            instance._pipeline = data
        logger.info("Model loaded from %s", path)
        return instance

    def _require_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        return self._pipeline
