"""Reusable, presentation-ready plot functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

sns.set_theme(style="whitegrid", palette="husl")
_FIG_SIZE = (12, 6)
_DPI = 150


def plot_word_frequencies(
    df: pd.DataFrame,
    title: str = "Top Word Frequencies",
    fig_height: float | None = None,
) -> plt.Figure:
    """Horizontal bar chart of word counts."""
    height = fig_height if fig_height is not None else max(6, len(df) * 0.35)
    fig, ax = plt.subplots(figsize=(12, height))
    sns.barplot(data=df, x="count", y="word", ax=ax, orient="h", color="steelblue")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig


def plot_ham_spam_comparison(df: pd.DataFrame) -> plt.Figure:
    """
    Diverging horizontal bar chart.

    HAM-leaning words extend left (negative log-odds),
    SPAM-leaning words extend right (positive log-odds).
    """
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
    colors = df["leaning"].map({"ham": "steelblue", "spam": "tomato"})
    ax.barh(df["word"], df["log_odds"], color=colors)
    ax.margins(y=0)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Log-Odds  (← HAM   SPAM →)")
    ax.set_title("Most Distinctive Words: HAM vs SPAM", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_word_frequencies_by_class(
    ham_df: pd.DataFrame,
    spam_df: pd.DataFrame,
    top_n: int = 20,
) -> plt.Figure:
    """Side-by-side horizontal bar charts: HAM top words vs SPAM top words."""
    fig, (ax_ham, ax_spam) = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.35)))
    sns.barplot(data=ham_df, x="count", y="word", ax=ax_ham, orient="h", color="steelblue")
    ax_ham.set_title(f"HAM – Top {top_n} Words", fontsize=13, fontweight="bold")
    ax_ham.set_xlabel("Count")
    ax_ham.set_ylabel("")
    sns.barplot(data=spam_df, x="count", y="word", ax=ax_spam, orient="h", color="tomato")
    ax_spam.set_title(f"SPAM – Top {top_n} Words", fontsize=13, fontweight="bold")
    ax_spam.set_xlabel("Count")
    ax_spam.set_ylabel("")
    plt.tight_layout()
    return fig


def plot_ngrams(df: pd.DataFrame, n: int = 2) -> plt.Figure:
    """Horizontal bar chart of n-gram frequencies."""
    label = "Bigrams" if n == 2 else "Trigrams" if n == 3 else f"{n}-grams"
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    sns.barplot(data=df, x="count", y="ngram", ax=ax, orient="h", color="mediumpurple")
    ax.set_title(f"Top {label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig


def plot_ngrams_by_class(
    ham_df: pd.DataFrame,
    spam_df: pd.DataFrame,
    n: int = 2,
    top_k: int = 15,
) -> plt.Figure:
    """Side-by-side horizontal bar charts: HAM n-grams vs SPAM n-grams."""
    label = "Bigrams" if n == 2 else "Trigrams" if n == 3 else f"{n}-grams"
    fig, (ax_ham, ax_spam) = plt.subplots(1, 2, figsize=(14, max(6, top_k * 0.38)))
    sns.barplot(data=ham_df, x="count", y="ngram", ax=ax_ham, orient="h", color="steelblue")
    ax_ham.set_title(f"HAM – Top {top_k} {label}", fontsize=13, fontweight="bold")
    ax_ham.set_xlabel("Count")
    ax_ham.set_ylabel("")
    sns.barplot(data=spam_df, x="count", y="ngram", ax=ax_spam, orient="h", color="tomato")
    ax_spam.set_title(f"SPAM – Top {top_k} {label}", fontsize=13, fontweight="bold")
    ax_spam.set_xlabel("Count")
    ax_spam.set_ylabel("")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list[str] = ("ham", "spam")) -> plt.Figure:
    """Heatmap confusion matrix with raw counts and percentages."""
    fig, ax = plt.subplots(figsize=(6, 5))
    total = cm.sum()
    annot = np.array(
        [[f"{v}\n({v/total:.1%})" for v in row] for row in cm]
    )
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: list[str], y_proba: np.ndarray) -> plt.Figure:
    """ROC curve with AUC annotation."""
    from sklearn.metrics import auc as sk_auc

    y_bin = [1 if y == "spam" else 0 for y in y_true]
    fpr, tpr, _ = roc_curve(y_bin, y_proba[:, 1])
    auc_val = sk_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_clusters_2d(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    texts: list[str] | None = None,
    title: str = "Semantic Clusters (UMAP projection)",
) -> plt.Figure:
    """
    Scatter plot of 2D-reduced embeddings coloured by cluster.

    Args:
        embeddings_2d: Array of shape (n, 2) from UMAP or PCA.
        labels: Cluster index per point.
        texts: Optional texts — shown as hover tooltip in notebooks.
        title: Plot title.
    """
    n_clusters = len(np.unique(labels))
    palette = sns.color_palette("husl", n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            s=18,
            alpha=0.7,
            color=palette[k],
            label=f"Cluster {k} (n={mask.sum()})",
        )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save figure and close it to free memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
