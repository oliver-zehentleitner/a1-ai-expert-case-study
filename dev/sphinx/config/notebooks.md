# Notebooks

The five analysis notebooks cover the full pipeline from raw data to semantic clustering.
They are rendered here as read-only views.

**To run them interactively:** clone the repository, follow the setup instructions in the
[README](README.md), then launch Jupyter from the project root:

```bash
jupyter notebook notebooks/
```

The `.ipynb` source files live in `notebooks/` in the repository root.

---

## Task 1 – Word Frequency Analysis

Tokenisation, stop-word filtering, and top-20 word frequencies across the full corpus.

<a href="notebooks/01_word_frequency.html">Open notebook</a>

---

## Task 2 – HAM vs. SPAM Comparison

Log-odds analysis: which words are statistically associated with spam vs. legitimate
messages.  Shows the geographic and temporal bias of the dataset.

<a href="notebooks/02_ham_vs_spam.html">Open notebook</a>

---

## Task 3 – N-Gram Analysis

Bigrams and trigrams reveal phrase-level patterns that unigrams miss.
SPAM n-grams are structural templates; HAM n-grams are genuine conversation.

<a href="notebooks/03_ngrams.html">Open notebook</a>

---

## Task 4 – Classification

TF-IDF pipeline + Logistic Regression.  Covers train/test split, evaluation metrics
(F1, ROC-AUC, confusion matrix), feature importance, and live prediction on custom
messages including German examples.

<a href="notebooks/04_classification.html">Open notebook</a>

---

## Task 5 – Semantic Clustering

Embedding-based clustering with three providers (Ollama, OpenAI, Azure AI Foundry /
Cohere multilingual).  Auto-detects available API keys, runs all providers on the same
corpus, and compares results side by side.

<a href="notebooks/05_semantic_clustering.html">Open notebook</a>
