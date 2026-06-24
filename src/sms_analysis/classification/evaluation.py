"""Classification evaluation — metrics, cross-validation, reporting."""

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structured container for all classification metrics."""

    accuracy: float
    f1_spam: float
    precision_spam: float
    recall_spam: float
    roc_auc: float
    confusion: np.ndarray = field(repr=False)
    report: str = field(repr=False)

    def summary(self) -> str:
        return (
            f"Accuracy={self.accuracy:.3f} | "
            f"F1-SPAM={self.f1_spam:.3f} | "
            f"Precision={self.precision_spam:.3f} | "
            f"Recall={self.recall_spam:.3f} | "
            f"ROC-AUC={self.roc_auc:.3f}"
        )


def evaluate(
    y_true: list[str],
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> EvaluationResult:
    """
    Compute a full set of binary classification metrics.

    Args:
        y_true: Ground-truth labels (``'ham'`` / ``'spam'``).
        y_pred: Predicted labels from the classifier.
        y_proba: Optional ``[p_ham, p_spam]`` probability matrix — required
                 for ROC-AUC. If omitted, ROC-AUC is set to ``float('nan')``.

    Returns:
        :class:`EvaluationResult` with all metrics.

    Note on metric choice:
        The dataset is *imbalanced* (86% HAM / 14% SPAM).
        Accuracy alone is therefore misleading — a model that always predicts
        HAM scores 86% without learning anything.  F1-SPAM and ROC-AUC are
        the primary metrics to watch.
    """
    # Derive spam column index from the label set rather than hardcoding 1.
    # sklearn sorts classes alphabetically → ['ham', 'spam'] → spam_idx=1,
    # but querying the actual labels is robust if that ever changes.
    spam_idx = sorted(set(y_true)).index("spam")
    auc = (
        roc_auc_score(
            [1 if y == "spam" else 0 for y in y_true],
            y_proba[:, spam_idx],
        )
        if y_proba is not None
        else float("nan")
    )

    result = EvaluationResult(
        accuracy=float(np.mean(np.array(y_true) == y_pred)),
        f1_spam=f1_score(y_true, y_pred, pos_label="spam"),
        precision_spam=precision_score(y_true, y_pred, pos_label="spam"),
        recall_spam=recall_score(y_true, y_pred, pos_label="spam"),
        roc_auc=auc,
        confusion=confusion_matrix(y_true, y_pred, labels=["ham", "spam"]),
        report=classification_report(y_true, y_pred),
    )
    logger.info(result.summary())
    return result


def cross_validate(
    pipeline: Pipeline,
    texts: list[str],
    labels: list[str],
    n_splits: int = 5,
) -> dict[str, float]:
    """
    Run stratified k-fold cross-validation and return mean ± std for key metrics.

    Stratified splits preserve the HAM/SPAM ratio in every fold, which matters
    for imbalanced datasets.

    Args:
        pipeline: An *untrained* scikit-learn Pipeline.
        texts: All message texts.
        labels: Corresponding labels.
        n_splits: Number of CV folds (default 5).

    Returns:
        Dict with ``f1_mean``, ``f1_std``, ``roc_auc_mean``, ``roc_auc_std``.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # F1-SPAM (pos_label='spam'), not F1-macro. At 86/14 imbalance the two differ
    # significantly: F1-macro averages over both classes (giving HAM equal weight),
    # while F1-SPAM tracks only the minority class we actually care about detecting.
    f1_spam_scorer = make_scorer(f1_score, pos_label="spam")
    f1_scores = cross_val_score(
        pipeline, texts, labels, cv=cv, scoring=f1_spam_scorer, n_jobs=1
    )
    logger.info(
        "%d-fold CV F1-SPAM: %.3f ± %.3f",
        n_splits, f1_scores.mean(), f1_scores.std(),
    )
    return {
        "f1_mean": float(f1_scores.mean()),
        "f1_std": float(f1_scores.std()),
    }
