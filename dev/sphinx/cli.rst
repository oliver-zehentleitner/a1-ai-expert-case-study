CLI Reference
=============

The ``sms-analyze`` command provides access to all five tasks from the CLI.

.. code-block:: text

   usage: sms-analyze [-h] [--model {logistic_regression,naive_bayes}]
                      [--provider {ollama,openai,azure}]
                      [--k INT] [--sample INT] [--yes]
                      {frequency,comparison,ngrams,classification,clustering,all}

Tasks
-----

``frequency``
   Task 1 — Top 20 word frequencies across all SMS messages.
   Outputs: ``outputs/word_frequencies.png``

``comparison``
   Task 2 — HAM vs. SPAM distinctive vocabulary via log-odds ratio with Laplace
   smoothing.
   Outputs: ``outputs/ham_spam_comparison.png``

``ngrams``
   Task 3 — Most frequent bigrams and trigrams per class.
   Outputs: ``outputs/bigrams.png``, ``outputs/trigrams.png``

``classification``
   Task 4 — Train and evaluate a SPAM/HAM classifier (Logistic Regression or
   Naive Bayes). Produces confusion matrix, ROC curve, and feature-importance
   plot.  Saves the fitted model to ``outputs/model_<type>.joblib``.

``clustering``
   Task 5 — Embed all SMS messages and cluster them semantically with K-Means.
   K is selected automatically via silhouette score. Saves a 2-D UMAP scatter
   plot.

``all``
   Run all five tasks in sequence.

Options
-------

``--model {logistic_regression,naive_bayes}``
   Classifier model for the ``classification`` task.  Default: ``logistic_regression``.

``--provider {ollama,openai,azure}``
   Embedding backend for the ``clustering`` task.  Default: ``ollama``.

   API keys and endpoints are read from environment variables / ``.env``.
   If a required secret is missing the CLI will prompt for it interactively.

``--k INT``
   Fix the number of clusters for the ``clustering`` task.  Default: automatic
   via silhouette score.

``--sample INT``
   Embed only a random sample of *N* messages.  Useful for quick tests.
   Default: all messages.

``--yes / -y``
   Auto-confirm all interactive prompts (start Ollama, pull model, save to
   ``.env``).  Intended for CI and demo runs.

Zero-setup first run
--------------------

No prior configuration is required for the Ollama backend:

.. code-block:: bash

   sms-analyze clustering

If Ollama is not running the CLI offers to start it. If the model has not been
pulled yet the CLI offers to pull it. Secrets for cloud providers (OpenAI,
Azure) are collected interactively via hidden input and can optionally be saved
to ``.env``.

.. code-block:: bash

   sms-analyze clustering --provider openai

   [!] SMS_OPENAI_API_KEY not set.
   Enter SMS_OPENAI_API_KEY:
   Save to .env for future use? [y/N/always]:

Environment variables
---------------------

All variables use the ``SMS_`` prefix. Copy ``.env.example`` to ``.env`` and
fill in the values you need.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Description
   * - ``SMS_OPENAI_API_KEY``
     - OpenAI API key (required for ``--provider openai``)
   * - ``SMS_AZURE_OPENAI_API_KEY``
     - Azure OpenAI API key
   * - ``SMS_AZURE_OPENAI_ENDPOINT``
     - Azure resource endpoint (``https://<resource>.openai.azure.com/``)
   * - ``SMS_AZURE_OPENAI_DEPLOYMENT``
     - Deployment name (default: ``text-embedding-3-small``)
   * - ``SMS_OLLAMA_BASE_URL``
     - Ollama server URL (default: ``http://localhost:11434``)
   * - ``SMS_OLLAMA_MODEL``
     - Ollama embedding model (default: ``nomic-embed-text``)
   * - ``SMS_OPENAI_MODEL``
     - OpenAI embedding model (default: ``text-embedding-3-small``)
   * - ``SMS_LOG_LEVEL``
     - Logging level (default: ``INFO``)
