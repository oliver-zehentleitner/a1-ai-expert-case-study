"""Command-line interface: ``sms-analyze <task>``."""

import argparse

from . import __version__
from .config import Settings, configure_logging


def run_frequency(settings: Settings) -> None:
    from .analysis.frequency import word_frequencies
    from .data.loader import load_dataset
    from .visualization.plots import (
        plot_word_frequencies,
        plot_word_frequencies_by_class,
        save_figure,
    )

    dataset = load_dataset(settings.data_path)

    for top_n in (20, 50, 100):
        df = word_frequencies(dataset.normalized_texts, top_n=top_n)
        fig = plot_word_frequencies(df, title=f"Top {top_n} Words – All SMS")
        save_figure(fig, settings.outputs_dir / f"word_frequencies_top{top_n}.png")
        if top_n == 20:
            print(df.to_string(index=False))

    for top_n in (20, 50):
        ham_df  = word_frequencies(dataset.normalized_ham_texts,  top_n=top_n)
        spam_df = word_frequencies(dataset.normalized_spam_texts, top_n=top_n)
        fig = plot_word_frequencies_by_class(ham_df, spam_df, top_n=top_n)
        save_figure(fig, settings.outputs_dir / f"word_frequencies_by_class_top{top_n}.png")


def run_comparison(settings: Settings) -> None:
    from .analysis.comparison import log_odds_ratio
    from .data.loader import load_dataset
    from .visualization.plots import plot_ham_spam_comparison, save_figure

    dataset = load_dataset(settings.data_path)
    df = log_odds_ratio(dataset.normalized_ham_texts, dataset.normalized_spam_texts)
    fig = plot_ham_spam_comparison(df)
    save_figure(fig, settings.outputs_dir / "ham_spam_comparison.png")
    print(df.to_string(index=False))


def run_ngrams(settings: Settings) -> None:
    from .analysis.ngrams import top_ngrams
    from .data.loader import load_dataset
    from .visualization.plots import plot_ngrams, plot_ngrams_by_class, save_figure

    dataset = load_dataset(settings.data_path)
    configs = [
        (2, "bigrams",  20),
        (3, "trigrams", 15),
    ]
    for n, label, top_k in configs:
        df = top_ngrams(dataset.normalized_texts, n=n, top_k=top_k)
        fig = plot_ngrams(df, n=n)
        save_figure(fig, settings.outputs_dir / f"{label}_all_top{top_k}.png")
        print(f"\n--- Top {label} (all SMS) ---")
        print(df.to_string(index=False))

        ham_df  = top_ngrams(dataset.normalized_ham_texts,  n=n, top_k=top_k)
        spam_df = top_ngrams(dataset.normalized_spam_texts, n=n, top_k=top_k)
        fig = plot_ngrams_by_class(ham_df, spam_df, n=n, top_k=top_k)
        save_figure(fig, settings.outputs_dir / f"{label}_by_class_top{top_k}.png")
        print(f"\n--- Top {label} HAM ---")
        print(ham_df.to_string(index=False))
        print(f"\n--- Top {label} SPAM ---")
        print(spam_df.to_string(index=False))


def run_classification(settings: Settings, model_type: str) -> None:
    from sklearn.model_selection import train_test_split

    from .classification.evaluation import evaluate
    from .classification.pipeline import SpamClassifier
    from .data.loader import load_dataset
    from .visualization.plots import plot_confusion_matrix, plot_roc_curve, save_figure

    dataset = load_dataset(settings.data_path).deduplicated
    print(f"Evaluating on {len(dataset)} unique messages (duplicates removed to prevent train/test leakage)")
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.texts, dataset.labels, test_size=0.2, stratify=dataset.labels, random_state=42
    )

    clf = SpamClassifier(model_type=model_type)  # type: ignore[arg-type]
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    result  = evaluate(y_test, y_pred, y_proba)

    print(result.summary())
    print(result.report)

    save_figure(
        plot_confusion_matrix(result.confusion),
        settings.outputs_dir / f"confusion_matrix_{model_type}.png",
    )
    save_figure(
        plot_roc_curve(y_test, y_proba),
        settings.outputs_dir / f"roc_curve_{model_type}.png",
    )
    clf.save(settings.outputs_dir / f"model_{model_type}.joblib")


def run_clustering(settings: Settings, args: argparse.Namespace) -> None:
    import random

    from .clustering.semantic import SemanticClusterer
    from .data.loader import load_dataset
    from .interactive import build_provider
    from .visualization.plots import plot_clusters_2d, save_figure

    dataset  = load_dataset(settings.data_path)
    provider = build_provider(args, settings)

    texts  = dataset.normalized_texts
    labels = dataset.labels
    if args.sample and args.sample < len(texts):
        random.seed(42)
        indices = random.sample(range(len(texts)), args.sample)
        texts  = [texts[i]  for i in indices]
        labels = [labels[i] for i in indices]

    clusterer = SemanticClusterer(provider, n_clusters=args.k or None)
    result    = clusterer.fit(texts)

    print(result.summary())
    _print_cluster_summary(result.labels, texts, labels)

    fig = plot_clusters_2d(result.embeddings_2d, result.labels, texts)
    save_figure(fig, settings.outputs_dir / "semantic_clusters.png")


def _print_cluster_summary(
    cluster_labels: "np.ndarray",  # type: ignore[name-defined]
    texts: list[str],
    class_labels: list[str],
    n_examples: int = 3,
) -> None:
    """Print per-cluster size, spam ratio, and example messages."""
    import numpy as np

    n_clusters = int(cluster_labels.max()) + 1
    print(f"\n{'─' * 70}")
    print(f"{'Cluster':<8} {'Size':>5}  {'SPAM%':>6}  Examples")
    print(f"{'─' * 70}")
    for k in range(n_clusters):
        mask = cluster_labels == k
        cluster_class = [class_labels[i] for i, m in enumerate(mask) if m]
        spam_count = sum(1 for l in cluster_class if l == "spam")
        spam_pct   = 100 * spam_count / len(cluster_class) if cluster_class else 0
        examples   = [texts[i] for i, m in enumerate(mask) if m][:n_examples]
        print(f"C{k:<7} {len(cluster_class):>5}  {spam_pct:>5.1f}%")
        for ex in examples:
            truncated = ex[:80].replace("\n", " ")
            print(f"         └ {truncated}")
    print(f"{'─' * 70}")


def run_docs() -> None:
    import webbrowser
    from pathlib import Path

    # docs/ lives at the project root — two levels above this package file.
    docs_index = Path(__file__).parent.parent.parent / "docs" / "index.html"
    if not docs_index.exists():
        import sys
        sys.exit(
            "docs/index.html not found.\n"
            "Build the docs first:  bash dev/sphinx/scripts/build_docs.sh"
        )
    webbrowser.open(docs_index.as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sms-analyze",
        description=f"SMS Spam/Ham Analysis – A1 AI Expert Case Study (v{__version__})",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "task",
        choices=["frequency", "comparison", "ngrams", "classification", "clustering", "docs", "all"],
        help="Which analysis to run (use 'docs' to open the documentation)",
    )

    # --- classification options -------------------------------------------
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "naive_bayes"],
        default="logistic_regression",
        help="Classifier model (default: logistic_regression)",
    )

    # --- clustering options -----------------------------------------------
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "azure"],
        default="ollama",
        help="Embedding provider for clustering (default: ollama). "
             "API keys are read from env / .env file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        metavar="INT",
        help="Number of clusters (default: auto-select via silhouette score)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="INT",
        help="Embed only a random sample of N messages (default: all)",
    )

    # --- global UX flags -------------------------------------------------
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Auto-confirm all interactive prompts (start Ollama, save to .env, etc.)",
    )

    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "docs":
        run_docs()
        return

    dispatch = {
        "frequency":      lambda: run_frequency(settings),
        "comparison":     lambda: run_comparison(settings),
        "ngrams":         lambda: run_ngrams(settings),
        "classification": lambda: run_classification(settings, args.model),
        "clustering":     lambda: run_clustering(settings, args),
    }

    # "all" runs Part 1 (classic ML) only — clustering requires an embedding
    # provider and is always run explicitly via "sms-analyze clustering".
    _part1 = ["frequency", "comparison", "ngrams", "classification"]

    if args.task == "all":
        for name in _part1:
            dispatch[name]()
    else:
        dispatch[args.task]()


if __name__ == "__main__":
    main()
