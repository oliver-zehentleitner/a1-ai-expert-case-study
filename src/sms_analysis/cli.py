"""Command-line interface: ``sms-analyze <task>``."""

import argparse
import sys
from pathlib import Path

from .config import EmbeddingProviderType, Settings, configure_logging


def _provider_from_settings(settings: Settings):
    """Instantiate the configured embedding provider."""
    from .clustering.providers import AzureOpenAIProvider, OllamaProvider, OpenAIProvider

    if settings.embedding_provider == EmbeddingProviderType.OLLAMA:
        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )
    if settings.embedding_provider == EmbeddingProviderType.OPENAI:
        if not settings.openai_api_key:
            sys.exit("SMS_OPENAI_API_KEY is not set")
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)
    if settings.embedding_provider == EmbeddingProviderType.AZURE:
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            sys.exit("SMS_AZURE_OPENAI_API_KEY and SMS_AZURE_OPENAI_ENDPOINT are required")
        return AzureOpenAIProvider(
            api_key=settings.azure_openai_api_key,
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_openai_deployment,
            api_version=settings.azure_openai_api_version,
        )
    raise ValueError(f"Unknown provider: {settings.embedding_provider}")


def run_frequency(settings: Settings) -> None:
    from .data.loader import load_dataset
    from .analysis.frequency import word_frequencies
    from .visualization.plots import plot_word_frequencies, save_figure

    dataset = load_dataset(settings.data_path)
    df = word_frequencies(dataset.texts, top_n=20)
    fig = plot_word_frequencies(df, title="Top 20 Words – All SMS")
    save_figure(fig, settings.outputs_dir / "word_frequencies.png")
    print(df.to_string(index=False))


def run_comparison(settings: Settings) -> None:
    from .data.loader import load_dataset
    from .analysis.comparison import log_odds_ratio
    from .visualization.plots import plot_ham_spam_comparison, save_figure

    dataset = load_dataset(settings.data_path)
    df = log_odds_ratio([m.text for m in dataset.ham], [m.text for m in dataset.spam])
    fig = plot_ham_spam_comparison(df)
    save_figure(fig, settings.outputs_dir / "ham_spam_comparison.png")
    print(df.to_string(index=False))


def run_ngrams(settings: Settings) -> None:
    from .data.loader import load_dataset
    from .analysis.ngrams import top_ngrams
    from .visualization.plots import plot_ngrams, save_figure

    dataset = load_dataset(settings.data_path)
    for n, label in [(2, "bigrams"), (3, "trigrams")]:
        df = top_ngrams(dataset.texts, n=n, top_k=15)
        fig = plot_ngrams(df, n=n)
        save_figure(fig, settings.outputs_dir / f"{label}.png")
        print(f"\n--- Top {label} ---")
        print(df.to_string(index=False))


def run_classification(settings: Settings, model_type: str = "logistic_regression") -> None:
    from sklearn.model_selection import train_test_split

    from .classification.evaluation import evaluate
    from .classification.pipeline import SpamClassifier
    from .data.loader import load_dataset
    from .visualization.plots import plot_confusion_matrix, plot_roc_curve, save_figure

    dataset = load_dataset(settings.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.texts, dataset.labels, test_size=0.2, stratify=dataset.labels, random_state=42
    )

    clf = SpamClassifier(model_type=model_type)  # type: ignore[arg-type]
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    result = evaluate(y_test, y_pred, y_proba)

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


def run_clustering(settings: Settings) -> None:
    from .clustering.semantic import SemanticClusterer
    from .data.loader import load_dataset
    from .visualization.plots import plot_clusters_2d, save_figure

    dataset = load_dataset(settings.data_path)
    provider = _provider_from_settings(settings)
    clusterer = SemanticClusterer(provider)
    result = clusterer.fit(dataset.texts)

    print(result.summary())
    fig = plot_clusters_2d(result.embeddings_2d, result.labels, dataset.texts)
    save_figure(fig, settings.outputs_dir / "semantic_clusters.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sms-analyze",
        description="SMS Spam/Ham Analysis – A1 AI Expert Case Study",
    )
    parser.add_argument(
        "task",
        choices=["frequency", "comparison", "ngrams", "classification", "clustering", "all"],
        help="Which analysis to run",
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "naive_bayes"],
        default="logistic_regression",
        help="Classifier model (only for 'classification' task)",
    )
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "frequency": lambda: run_frequency(settings),
        "comparison": lambda: run_comparison(settings),
        "ngrams": lambda: run_ngrams(settings),
        "classification": lambda: run_classification(settings, args.model),
        "clustering": lambda: run_clustering(settings),
    }

    if args.task == "all":
        for fn in dispatch.values():
            fn()
    else:
        dispatch[args.task]()
