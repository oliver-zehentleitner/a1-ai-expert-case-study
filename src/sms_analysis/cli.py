"""Command-line interface: ``sms-analyze <task>``."""

import argparse
import sys

from .config import EmbeddingProviderType, Settings, configure_logging


def _build_provider(args: argparse.Namespace, settings: Settings):
    """
    Instantiate the embedding provider.

    CLI flag ``--provider`` takes precedence over the ``SMS_EMBEDDING_PROVIDER``
    env var.  API keys and endpoints are always read from the environment / .env
    file — they are secrets and do not belong on the command line.
    """
    from .clustering.providers import AzureOpenAIProvider, OllamaProvider, OpenAIProvider

    provider_type = EmbeddingProviderType(args.provider)

    if provider_type == EmbeddingProviderType.OLLAMA:
        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    if provider_type == EmbeddingProviderType.OPENAI:
        if not settings.openai_api_key:
            sys.exit("SMS_OPENAI_API_KEY is required for --provider openai")
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)

    if provider_type == EmbeddingProviderType.AZURE:
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            sys.exit(
                "SMS_AZURE_OPENAI_API_KEY and SMS_AZURE_OPENAI_ENDPOINT "
                "are required for --provider azure"
            )
        return AzureOpenAIProvider(
            api_key=settings.azure_openai_api_key,
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_openai_deployment,
            api_version=settings.azure_openai_api_version,
        )

    raise ValueError(f"Unknown provider: {provider_type}")


def run_frequency(settings: Settings) -> None:
    from .analysis.frequency import word_frequencies
    from .data.loader import load_dataset
    from .visualization.plots import plot_word_frequencies, save_figure

    dataset = load_dataset(settings.data_path)
    df = word_frequencies(dataset.texts, top_n=20)
    fig = plot_word_frequencies(df, title="Top 20 Words – All SMS")
    save_figure(fig, settings.outputs_dir / "word_frequencies.png")
    print(df.to_string(index=False))


def run_comparison(settings: Settings) -> None:
    from .analysis.comparison import log_odds_ratio
    from .data.loader import load_dataset
    from .visualization.plots import plot_ham_spam_comparison, save_figure

    dataset = load_dataset(settings.data_path)
    df = log_odds_ratio([m.text for m in dataset.ham], [m.text for m in dataset.spam])
    fig = plot_ham_spam_comparison(df)
    save_figure(fig, settings.outputs_dir / "ham_spam_comparison.png")
    print(df.to_string(index=False))


def run_ngrams(settings: Settings) -> None:
    from .analysis.ngrams import top_ngrams
    from .data.loader import load_dataset
    from .visualization.plots import plot_ngrams, save_figure

    dataset = load_dataset(settings.data_path)
    for n, label in [(2, "bigrams"), (3, "trigrams")]:
        df = top_ngrams(dataset.texts, n=n, top_k=15)
        fig = plot_ngrams(df, n=n)
        save_figure(fig, settings.outputs_dir / f"{label}.png")
        print(f"\n--- Top {label} ---")
        print(df.to_string(index=False))


def run_classification(settings: Settings, model_type: str) -> None:
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
    from .clustering.semantic import SemanticClusterer
    from .data.loader import load_dataset
    from .visualization.plots import plot_clusters_2d, save_figure

    dataset  = load_dataset(settings.data_path)
    provider = _build_provider(args, settings)

    texts = dataset.texts
    if args.sample and args.sample < len(texts):
        import random
        random.seed(42)
        texts = random.sample(texts, args.sample)

    clusterer = SemanticClusterer(provider, n_clusters=args.k or None)
    result    = clusterer.fit(texts)

    print(result.summary())
    fig = plot_clusters_2d(result.embeddings_2d, result.labels, texts)
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

    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "frequency":      lambda: run_frequency(settings),
        "comparison":     lambda: run_comparison(settings),
        "ngrams":         lambda: run_ngrams(settings),
        "classification": lambda: run_classification(settings, args.model),
        "clustering":     lambda: run_clustering(settings, args),
    }

    if args.task == "all":
        for fn in dispatch.values():
            fn()
    else:
        dispatch[args.task]()
