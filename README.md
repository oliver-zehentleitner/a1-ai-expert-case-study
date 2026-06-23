# AI Expert Case Study – Oliver Zehentleitner

Solution to the [A1 Telekom Austria AI Expert case study](./assignment/Fallbeispiel.pdf)
([job posting](https://www.isg.com/jobs/ad/AI_Expert-110091)).

The SMS dataset is located at
[`./assignment/3D8026C4.zip/SMSSpamCollection`](./assignment/3D8026C4.zip/SMSSpamCollection),
based on the publicly available
[SMS Spam Collection v.1](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)
by Tiago A. Almeida and José María Gómez Hidalgo. Used here for this case study only.

---

## Summary of the task

The case study covers two focus areas applied to an English SMS dataset
(5,572 messages: 86 % HAM / 14 % SPAM):

**Part 1 – Classic ML & Data Science (fully local, no external services)**

| # | Task |
|---|------|
| 1 | Word frequency analysis across all SMS |
| 2 | HAM vs. SPAM: distinctive vocabulary per class |
| 3 | Most frequent bigrams and trigrams |
| 4 | SPAM/HAM classification model with evaluation |

**Part 2 – GenAI & LLM**

| # | Task |
|---|------|
| 5 | Semantic clustering of SMS using embedding vectors |

---

## How is the implementation envisaged?

The solution is structured as an **installable Python package** (`sms_analysis`)
rather than a collection of standalone scripts. Jupyter notebooks import the
package and serve as the presentation layer — the library is the real deliverable.

```
src/sms_analysis/
├── data/           – Pydantic domain models + dataset loader
├── analysis/       – Word frequencies, HAM/SPAM comparison, n-grams  (Tasks 1–3)
├── classification/ – TF-IDF pipeline + evaluation utilities          (Task 4)
├── clustering/
│   ├── base.py     – Abstract EmbeddingProvider interface            (Task 5)
│   └── providers/  – Ollama · OpenAI · Azure OpenAI implementations
├── visualization/  – Reusable, presentation-ready plot functions
└── cli.py          – sms-analyze CLI entrypoint
```

**Key architectural decision — provider abstraction:**
The clustering logic depends only on the abstract `EmbeddingProvider` interface.
Switching from local (Ollama) to cloud (OpenAI / Azure) is a single config
change — no code change. This follows the Strategy pattern and makes the
solution testable, extensible, and provider-agnostic.

**Configuration** is fully environment-driven via `pydantic-settings`.
All variables use the `SMS_` prefix (see `.env.example`).

---

## Dataset & preprocessing

### Source and class distribution

The dataset contains 5,572 SMS messages labelled by hand:

| Class | Count | Share |
|-------|------:|------:|
| HAM   | 4,825 |  86 % |
| SPAM  |   747 |  14 % |

This imbalance is why accuracy is a meaningless metric here — a model that
always predicts HAM scores 86 % without learning anything useful.

### HTML-encoded tokens in the raw data

Several messages contain HTML-encoded entities such as `&lt;URL&gt;` or `&lt;#&gt;`.
Their exact origin is unclear: they may be researcher-applied anonymisation
placeholders, or they may be copy-paste artefacts from a smartphone browser —
this would need to be verified against the original data collection notes.
Regardless of origin, the observed tokens are:

| Raw token | Observed meaning |
|-----------|-----------------|
| `&lt;#&gt;` | A number (phone, amount, date, …) |
| `&lt;URL&gt;` | A hyperlink **or a username/handle** (e.g. `@ &lt;URL&gt;`) |
| `&lt;DECIMAL&gt;` | A decimal number |
| `&lt;NUMBER&gt;` | A numeric value |
| `&amp;` | Ampersand |
| Other `&lt;…&gt;` | Further tokens of unclear origin |

These are **not** real HTML — they are ASCII representations of `<#>`, `<URL>`,
etc., stored in a plain-text file.

### Real URLs also present

A small number of messages contain actual URLs (`http://…`, `www.…`). In this
dataset they appear exclusively in SPAM — a correlation that is a property of
this sample, not a general truth. Ham messages regularly contain URLs in the
real world (shared links, meeting invites, etc.).

### How we normalise

All of the above are collapsed to three semantic tokens before any vectorisation:

| Input | Normalised to |
|-------|--------------|
| `@ &lt;URL&gt;`, `@handle` | `__USER__` |
| `&lt;URL&gt;`, `http://…`, `www.…` | `__URL__` |
| `&lt;#&gt;`, `&lt;DECIMAL&gt;`, `&lt;NUMBER&gt;`, standalone digits | `__NUM__` |
| `&amp;` | `and` |
| Other `&lt;…&gt;` | removed |

**Why tokens instead of deletion?** The *presence* of a URL, username mention,
or number is a strong statistical signal — spam messages contain them at a much
higher rate in this dataset. Deleting them discards that signal. Replacing them
with a consistent token lets TF-IDF and the embedding model learn the pattern.

**What ML does and does not decide here:**
The model learns that *spam messages tend to contain URLs or phone numbers more
often* — it does not evaluate whether a specific URL or number is malicious.
That distinction matters in production:

- A URL is not inherently spam or ham. A domain that is clean today can be
  compromised tomorrow, and a spammer can register a fresh domain that appears
  on no blacklist yet.
- A phone number is not inherently spam or ham either. Numbers get recycled,
  spoofed, or legitimately used by businesses.

For high-quality spam filtering, URL and phone-number reputation checks belong
in a dedicated layer — URL blacklists / threat-intel feeds (IoC matching) and
carrier/number reputation services — not in the ML text classifier. The ML
signal ("a URL is present") and the reputation check ("this specific URL is
known-bad") complement each other and should not be conflated.

**Train-serve consistency:** the normalisation step is the first stage of the
serialised `sklearn.Pipeline`. When the trained model is loaded for inference,
the same preprocessing runs automatically — there is no risk of divergence
between training and prediction.

---

## Advantages and disadvantages of this implementation, and why it was chosen

### Classic ML (Part 1)

**Text representation — TF-IDF with bigrams**
- *Advantage:* Sparse, interpretable, fast. The IDF term down-weights ubiquitous words
  (no stopword list needed for the core signal). Including bigrams lets the model
  capture phrases like "free entry" or "win prize" that are individually ambiguous.
- *Disadvantage:* No semantic understanding — "automobile" and "car" are unrelated vectors.
  Word order is lost beyond the bigram window.

**Model — Logistic Regression (primary) + Naive Bayes (baseline)**
- *Advantage:* LR coefficients are directly interpretable: each word has a signed weight
  visible in the feature-importance plot. Excellent performance on sparse TF-IDF input.
  NB provides a fast, explainable baseline rooted in probability theory.
- *Disadvantage:* Linear decision boundary — cannot model non-linear feature interactions
  without explicit engineering. A transformer model (e.g. fine-tuned BERT) would likely
  close the remaining error gap on edge cases.
- *Why not SVM / Random Forest / deep learning?* SVM is competitive but less interpretable.
  Ensemble methods are optimised for dense, low-dimensional tabular data. Deep learning
  would be overkill for 5,500 samples and a 99 %+ achievable AUC with linear models.

**Evaluation — F1-SPAM and ROC-AUC (not accuracy)**
The dataset is imbalanced (86 / 14). A model that always predicts HAM scores 86 % accuracy
without learning anything. F1-SPAM and ROC-AUC are the meaningful primary metrics.
Stratified k-fold cross-validation preserves the class ratio in every fold.

### GenAI / Embeddings (Part 2)

**Embeddings — nomic-embed-text (Ollama) / text-embedding-3-small (OpenAI / Azure)**
- *Advantage:* Dense semantic vectors capture meaning, not just vocabulary. Two semantically
  similar spam messages using completely different words land in the same cluster — impossible
  with TF-IDF. Running locally via Ollama requires no API key and works offline.
- *Disadvantage:* Embedding 5,500 messages locally is slow on CPU (~minutes). API-based
  providers are faster but introduce a cost and a network dependency.

**Clustering — K-Means with automatic K selection via silhouette score**
- *Advantage:* Simple, deterministic, efficient. Automatic K selection avoids manual tuning.
  UMAP reduction to 2D produces publication-quality scatter plots where cluster separation
  is visually obvious.
- *Disadvantage:* K-Means assumes spherical clusters and is sensitive to outliers.
  HDBSCAN would be more robust but harder to explain in 15 minutes.

---

## How does the implementation work?

### Prerequisites

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

### Part 1 — Classic ML (fully local, no setup beyond pip install)

**Run all analyses at once:**
```bash
sms-analyze all
```

**Or run individual tasks:**
```bash
sms-analyze frequency        # Task 1 – top word frequencies
sms-analyze comparison       # Task 2 – HAM vs. SPAM vocabulary
sms-analyze ngrams           # Task 3 – bigrams + trigrams
sms-analyze classification   # Task 4 – train, evaluate, save model
sms-analyze classification --model naive_bayes   # baseline comparison
```

Generated plots and models are saved to `outputs/`.

**Or open the Jupyter notebooks:**
```bash
jupyter notebook notebooks/
```

### Part 2 — Semantic Clustering (Task 5)

**Option A — Ollama (local, no API key)**

Linux / macOS:
```bash
bash scripts/setup_ollama.sh   # installs Ollama + pulls nomic-embed-text
sms-analyze clustering
```

Windows (PowerShell):
```powershell
.\scripts\setup_ollama.ps1     # downloads OllamaSetup.exe, then pulls model
sms-analyze clustering
```

**Option B — OpenAI**
```bash
sms-analyze clustering --provider openai
# prompts for SMS_OPENAI_API_KEY on first run and offers to save it to .env
```

**Option C — Azure OpenAI**
```bash
sms-analyze clustering --provider azure
# prompts for SMS_AZURE_OPENAI_API_KEY, endpoint, and deployment on first run
```

### Tests

```bash
pytest tests/ -v
```

### Documentation

```bash
pip install -e ".[docs]"   # first time only — adds Sphinx + furo to the project venv
bash scripts/build_docs.sh
```

Opens at `docs/_build/html/index.html`. Covers CLI reference, full API autodoc,
and architecture overview.

### Use as a library

```python
from sms_analysis import load_dataset
from sms_analysis.classification import SpamClassifier

dataset = load_dataset("assignment/3D8026C4.zip/SMSSpamCollection")
clf = SpamClassifier("logistic_regression").fit(dataset.texts, dataset.labels)
print(clf.predict(["WINNER! Claim your free prize now!"]))
```
