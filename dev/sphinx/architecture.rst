Architecture
============

Package layout
--------------

.. code-block:: text

   src/sms_analysis/
   ├── data/
   │   ├── loader.py          SMSMessage, SMSDataset (Pydantic), load_dataset()
   │   └── preprocessing.py   Text normalisation (HTML entities, URLs → tokens)
   ├── analysis/
   │   ├── frequency.py       Word frequency counting
   │   ├── comparison.py      Log-odds ratio HAM vs. SPAM
   │   └── ngrams.py          Bigram / trigram extraction
   ├── classification/
   │   ├── pipeline.py        SpamClassifier — TF-IDF → LR / NB
   │   └── evaluation.py      EvaluationResult, evaluate()
   ├── clustering/
   │   ├── base.py            EmbeddingProvider (ABC)
   │   ├── semantic.py        SemanticClusterer, ClusteringResult
   │   └── providers/
   │       ├── ollama.py      OllamaProvider
   │       ├── openai.py      OpenAIProvider
   │       └── azure.py       AzureOpenAIProvider
   ├── visualization/
   │   └── plots.py           plot_* functions, save_figure()
   ├── config.py              pydantic-settings, configure_logging()
   ├── interactive.py         Zero-setup UX — prompts, Ollama lifecycle, .env writer
   └── cli.py                 sms-analyze entry point

Domain models
-------------

All domain objects are Pydantic ``BaseModel`` subclasses:

- ``SMSMessage(label: Label, text: str)`` — single SMS, ``is_spam`` property
- ``SMSDataset(messages: list[SMSMessage])`` — with ``.ham``, ``.spam``,
  ``.texts``, ``.labels``, ``.to_dataframe()``
- ``ClusteringResult`` — embeddings, 2-D UMAP projection, labels, inertias

Strategy pattern — embedding providers
---------------------------------------

The clustering layer is decoupled from any concrete embedding backend:

.. code-block:: text

   EmbeddingProvider (ABC)
   │  embed(texts: list[str]) → np.ndarray
   │  dimensions: int
   │
   ├── OllamaProvider      — local, no API key, runs on CPU
   ├── OpenAIProvider      — text-embedding-3-small (or any other model)
   └── AzureOpenAIProvider — same openai client, different initialisation

``SemanticClusterer`` receives an ``EmbeddingProvider`` instance; it never
imports any concrete provider class. Switching backend requires no code change —
only the CLI flag ``--provider`` or the object passed at construction.

Train-serve consistency
-----------------------

The SPAM/HAM classifier is an ``sklearn.pipeline.Pipeline``:

.. code-block:: text

   FunctionTransformer(_preprocess_batch)
     ↓
   TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), max_features=10000)
     ↓
   LogisticRegression / ComplementNB

The preprocessing step (HTML entity normalisation, URL replacement) is part of
the serialised pipeline. When the model is loaded and used for inference the
same preprocessing runs automatically — there is no risk of train-serve skew.

``_preprocess_batch`` is a module-level named function (not a lambda) because
joblib cannot pickle lambda functions.

Configuration
-------------

``Settings`` (``pydantic-settings``) reads all configuration from environment
variables with the ``SMS_`` prefix:

- **Secrets** (API keys, endpoints) — env vars or ``.env`` file
- **Behaviour** (provider choice, model name) — CLI flags

The ``interactive`` module bridges first-run UX: if a required secret is absent
it prompts via ``getpass``, then offers to append the value to ``.env``.

Evaluation metrics
------------------

The dataset is imbalanced (86 % HAM / 14 % SPAM). Accuracy is not meaningful.
Primary metrics:

- **F1-SPAM** — harmonic mean of SPAM precision and recall
- **ROC-AUC** — area under the receiver operating characteristic curve

Stratified ``train_test_split`` (80/20) and ``StratifiedKFold`` cross-validation
preserve the class ratio in every split.
