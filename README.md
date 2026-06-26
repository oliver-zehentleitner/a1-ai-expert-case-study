# AI Expert Case Study – Oliver Zehentleitner

Solution to the [A1 Telekom Austria AI Expert case study](./assignment/Fallbeispiel.pdf)
([job posting](https://www.isg.com/jobs/ad/AI_Expert-110091)).

**[Full documentation](docs/index.html)** — CLI reference, API autodoc, architecture overview.

The SMS dataset is located at
[`./assignment/3D8026C4.zip/SMSSpamCollection`](./assignment/3D8026C4.zip/SMSSpamCollection)
(note: `3D8026C4.zip` is the extracted assignment directory, not a ZIP archive),
based on the *SMS Spam Collection v.1* by Tiago A. Almeida and José María Gómez Hidalgo
(`http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/` — canonical citation URL,
kept for attribution; the dataset is also available via the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)).
Used here for this case study only.

---

## Summary of the task

The case study covers two focus areas applied to an English SMS dataset
(5,574 messages: 86 % HAM / 14 % SPAM):

**Part 1 – Classic ML & Data Science (fully local, no external services)**

| # | Task | Notebook |
|---|------|----------|
| 1 | Word frequency analysis across all SMS | [01_word_frequency.ipynb](notebooks/01_word_frequency.ipynb) |
| 2 | HAM vs. SPAM: distinctive vocabulary per class | [02_ham_vs_spam.ipynb](notebooks/02_ham_vs_spam.ipynb) |
| 3 | Most frequent bigrams and trigrams | [03_ngrams.ipynb](notebooks/03_ngrams.ipynb) |
| 4 | SPAM/HAM classification model with evaluation | [04_classification.ipynb](notebooks/04_classification.ipynb) |

**Part 2 – GenAI & LLM**

| # | Task | Notebook |
|---|------|----------|
| 5 | Semantic clustering of SMS using embedding vectors | [05_semantic_clustering.ipynb](notebooks/05_semantic_clustering.ipynb) |

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
│   └── providers/  – Ollama · OpenAI · Azure AI Foundry (Cohere) implementations
├── visualization/  – Reusable, presentation-ready plot functions
└── cli.py          – sms-analyze CLI entrypoint
```

**Key architectural decision — provider abstraction:**
The clustering logic depends only on the abstract `EmbeddingProvider` interface.
Switching providers is a single config change — no code change. This follows the
Strategy pattern and makes the solution testable, extensible, and provider-agnostic.

The three providers are intentionally diverse — different vendors, different model
families, different deployment models — rather than the same model hosted three ways:

| Provider | Model | Dimensions | Focus |
|----------|-------|-----------|-------|
| Ollama (local) | nomic-embed-text | 768 | English, privacy-first, no API key |
| OpenAI | text-embedding-3-small | 1536 | English, state-of-the-art |
| Azure AI Foundry | Cohere-embed-v3-multilingual | 1024 | **Multilingual**, relevant for German SMS |

Azure AI Foundry is used via its `/openai/v1` compatibility endpoint with the Cohere
serverless model — no deployment configuration required. `text-embedding-3-small` would
also be available on Azure but would require creating an explicit deployment and would
duplicate the OpenAI provider without adding value.

**Configuration** is fully environment-driven via `pydantic-settings`.
All variables use the `SMS_` prefix (see `.env.example`).

---

## Dataset & preprocessing

### Source and class distribution

The raw file contains 5,574 messages labelled by hand:

| Class | Count | Share |
|-------|------:|------:|
| HAM   | 4,827 |  86 % |
| SPAM  |   747 |  14 % |

**Duplicate rows:** 403 of the 5,574 rows are exact duplicates (309 HAM / 94 SPAM).
5,171 unique message texts remain after deduplication.
No duplicate pair carries conflicting labels.
For evaluation, the classifier uses the deduplicated set to prevent the same
text from appearing in both train and test splits.

This imbalance is why accuracy is a meaningless metric here — a model that
always predicts HAM scores 86 % without learning anything useful.

The dataset is a composite of four sources (see `assignment/3D8026C4.zip/readme`):

| Source | Type | Count |
|--------|------|------:|
| Grumbletext (UK forum, manually extracted) | SPAM | ~425 |
| NUS SMS Corpus (National University of Singapore) | HAM | ~3,375 |
| Caroline Tag's PhD thesis | HAM | ~450 |
| SMS Spam Corpus v.0.1 Big | HAM + SPAM | ~1,324 |

This multi-source origin is important — the HAM and SPAM messages were collected
and processed differently, which has direct consequences for preprocessing
(see below).

### HTML-encoded tokens — investigation and findings

The raw data contains HTML-encoded strings such as `&lt;URL&gt;` and `&lt;#&gt;`.
Rather than assuming their origin, I verified it by counting how many
messages contain each token type, broken down by class:

| Token | Messages containing token | HAM | SPAM |
|-------|-------------------------:|----:|-----:|
| `&lt;#&gt;` | 214 | 214 | 0 |
| `&lt;DECIMAL&gt;` | 21 | 21 | 0 |
| `&lt;TIME&gt;` | 7 | 7 | 0 |
| `&lt;URL&gt;` | 3 | 3 | 0 |
| `&lt;EMAIL&gt;` | 1 | 1 | 0 |

**Finding: every single token appears exclusively in HAM messages — zero in SPAM.**

This rules out both hypotheses that were on the table:

- **Not spam template variables** (unfilled placeholders from mass-SMS tools) —
  those would appear in SPAM, not HAM.
- **Not scraping artefacts from the Grumbletext forum** — Grumbletext is the
  source of the SPAM messages; these tokens don't appear there at all.

**What they actually are:** privacy-driven anonymisation applied by the NUS SMS
Corpus team before publishing the HAM messages. Volunteers contributed real SMS
messages; the NUS team replaced phone numbers, monetary amounts, decimal values,
times, and e-mail addresses with typed placeholders so the messages could be
shared publicly without exposing personal data.

The SPAM messages (Grumbletext, manually copy-pasted) were not processed the
same way — they retain real phone numbers, prices, and URLs.

**Why this matters for the model:** these tokens are not spam indicators.
A real number in a spam message and an anonymised number in a HAM message both
map to `TOKEN_NUM` after normalisation. The ML signal is not the token itself
but the *density and combination* of such tokens — spam messages are number-heavy
(multiple phone numbers, shortcodes, prices, dates), while HAM messages
containing `TOKEN_NUM` typically have just one or two occurrences.

### Real URLs also present

A small number of messages contain actual raw URLs (`http://…`, `www.…`).
In this dataset they appear exclusively in SPAM — a correlation that is a
property of this sample, not a general truth. HAM messages regularly contain
links in real life (shared content, meeting invites, etc.).

### Normalisation pipeline

**The raw dataset is never modified.** `SMSDataset.texts` always returns the
original strings exactly as loaded from the file.

Normalisation is applied on demand:

```python
dataset.normalized_texts        # all messages, normalised
dataset.normalized_ham_texts    # HAM only, normalised
dataset.normalized_spam_texts   # SPAM only, normalised
```

All three properties call `preprocess()` from
`sms_analysis.data.preprocessing` — the single source of truth for
normalisation. Changing the pipeline means changing one function; all
consumers (analysis tasks, classification, n-gram extraction) pick up the
change automatically without touching the raw data.

The full normalisation mapping:

| Input pattern | Origin | Action |
|--------------|--------|--------|
| `@ &lt;URL&gt;`, `@handle` | NUS placeholder / typed | → `TOKEN_USER` |
| `&lt;URL&gt;`, `http://…`, `www.…` | NUS placeholder / typed | → `TOKEN_URL` |
| `&lt;#&gt;`, `&lt;DECIMAL&gt;`, standalone digits (≥ 2) | NUS placeholder / typed | → `TOKEN_NUM` |
| `&lt;TIME&gt;` | NUS placeholder | → `TOKEN_TIME` |
| `&lt;EMAIL&gt;` | NUS placeholder | → `TOKEN_EMAIL` |
| `£`, `$`, `€`, `¥`, `₹`, `₽`, `₩`, `₿` | typed | → `TOKEN_CURRENCY` |
| `&amp;` | HTML encoding of real `&` | → `and` |
| `Ü` / `ü` | shorthand for "you" | → `u` (avoids spurious HAM bias) |
| Other `&lt;…&gt;` | unknown | removed + WARNING logged |

**Why tokens instead of deletion?** The *presence* of a URL or phone number is
a meaningful classification signal — spam messages are dense with them, HAM
messages are not. The specific URL or number is irrelevant for this task; for
URL/number reputation a separate system (blacklists, IoC feeds) is needed.
NUS anonymisation tokens (`&lt;URL&gt;`, `&lt;#&gt;`, …) map to the same tokens as their
typed equivalents so the model learns the same pattern regardless of source.

**What ML does and does not decide here:**
The model learns that *spam messages tend to be dense with URLs and phone numbers* —
it does not evaluate whether a specific URL or number is malicious.
That distinction matters in production:

- A URL is not inherently spam or ham. A domain that is clean today can be
  compromised tomorrow; a spammer can register a fresh domain that appears
  on no blacklist yet.
- A phone number is not inherently spam or ham. Numbers get recycled,
  spoofed, or legitimately used by businesses.

URL and phone-number reputation belong in dedicated layers — URL blacklists,
threat-intel feeds (IoC matching), carrier reputation services — complementing
the ML classifier rather than being conflated with it.

**Train-serve consistency:** the same `preprocess()` function is also the
first step of the serialised `sklearn.Pipeline` used by the classifier.
When the trained model is loaded for inference, preprocessing runs automatically —
there is no risk of divergence between training and prediction.

### Dataset limitations and what they mean for the model

This dataset is useful for demonstrating the approach but carries biases that
directly affect what the trained model can and cannot do.

**Geographic and linguistic bias**

The corpus is UK/Singaporean English. HAM reflects Singaporean student
vocabulary ("lor", "lah", "da"); SPAM reflects UK premium-rate fraud
("prize", "claim", "150p per msg"). The model is a reasonable UK/Singapore
spam detector — on Austrian, German, or other traffic it would encounter
different patterns and performance would degrade.

**Fit for A1's actual use case**

A1's customers communicate in German. Austrian spam runs in German —
0900 numbers, Paketbenachrichtigungs-Phishing, Kleinanzeigen-Betrug.
None of that is in this dataset. For production: same package, same
architecture, different training corpus.

The live prediction demo in [Notebook 4](notebooks/04_classification.ipynb)
tests both English and German messages and reveals a nuanced picture:

- **German SPAM is reliably detected** (e.g. 98 % confidence on a German prize
  announcement). Structural spam signals — `TOKEN_NUM` for premium-rate numbers,
  `TOKEN_URL`, `TOKEN_CURRENCY` — survive preprocessing unchanged and cross
  language boundaries.
- **German HAM is at higher false-positive risk than English HAM** when the
  message contains those same structural tokens. A German appointment reminder
  with a URL scores 74 % SPAM probability; its English equivalent scores 45 %
  (borderline HAM). A German bank TAN confirmation scores 89 % SPAM; the English
  version 59 %. The reason: English words like "appointment", "reschedule",
  "transaction", "bank" carry small HAM weight from the training corpus; their
  German equivalents ("Termin", "Absagen", "Freigabe") are unknown to the model
  and contribute zero weight — the structural spam tokens dominate unopposed.

**Currency symbols as a concrete example**

In this dataset, `£` (British pound) appears almost exclusively in SPAM.
Without normalisation, the model would learn "£ = spam signal" — which is true
for UK data but useless (or misleading) for traffic where `€`, `$`, or `₹`
are the local currency. The preprocessing step normalises all currency symbols
to `TOKEN_CURRENCY`, so the model learns "a currency symbol before an amount =
spam signal" rather than anything tied to a specific currency.

Extensions worth considering: ISO currency codes (`EUR`, `USD`, `GBP`, …) are
not covered by symbol matching and would benefit from the same treatment.
Cryptocurrency contexts could justify a dedicated `TOKEN_CRYPTO` token —
crypto scams have a distinct signature (`BTC`, `ETH`, wallet addresses) that
may be worth separating from fiat-currency spam. Both are deliberate design
decisions, not oversights.

**Spurious correlations and adversarial robustness — a concrete example**

The NUS corpus contains ~50 HAM messages where one or a few contributors used
`Ü` as SMS shorthand for "you" (identical in meaning to the common `u`). The
exact cause is unknown — one plausible explanation is a phone with a European
keyboard layout, but this is not verified. What is clear from the data: `Ü`
appears 50 times in HAM and exactly zero times in SPAM.

Left unnormalised, the model would learn `ü` → HAM with very high confidence —
a spurious correlation driven entirely by the typing habits of one person's
phone hardware, not by anything meaningful about the content. This creates a
trivially exploitable weakness: a spammer who appended a single `Ü` to their
message could shift the classifier toward a HAM prediction.

The fix is to normalise `Ü` → `u` in preprocessing. Semantically correct
(both mean "you"), eliminates the attack surface, and removes a feature that
would not generalise to any data collected on different devices.

This is a concrete example of why understanding training data matters beyond
accuracy numbers. A 99% AUC model can carry silent exploitable weaknesses that
only surface when someone looks closely at what the features actually represent.

**What diverse, multi-locale training data would change**

Different regions have different spam vocabularies, different common scam types,
different norms for legitimate business SMS. A production-grade multilingual
spam filter would need:

- Multiple corpora covering different languages and regions
- Balanced representation of local HAM patterns (not just Singlish)
- Ongoing labelling as spam patterns evolve

Without this, any model is essentially an in-distribution classifier: it works
well on data that resembles its training set, and its quality guarantees end
where that resemblance ends. This is not a flaw in the approach — it is an
inherent property of supervised ML that needs to be stated explicitly when
setting expectations.

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
The CLI evaluation uses a single stratified 80/20 train/test split on the deduplicated
dataset. The codebase also includes `StratifiedKFold` utilities for cross-validation
where a more robust estimate across multiple folds is needed.

### GenAI / Embeddings (Part 2)

**Embeddings — three diverse providers (Ollama / OpenAI / Azure AI Foundry)**
- *Advantage:* Dense semantic vectors capture meaning, not just vocabulary. Two semantically
  similar spam messages using completely different words land in the same cluster — impossible
  with TF-IDF. Running locally via Ollama requires no API key and works offline.
  The three providers cover different model families and deployment models, showing that the
  abstraction works across genuinely different backends (not just the same model re-hosted).
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
pip install -r requirements.txt
pip install -e .
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### Documentation

Open the docs in your default browser:

```bash
sms-analyze docs
```

The rendered documentation is included under `docs/`. Rebuilding from source
is not part of this submission package.

### Part 1 — Classic ML (fully local, no setup beyond pip install)

**Run all analyses at once (Part 1 only — Tasks 1–4):**
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

**Inspect the preprocessing pipeline:**
```bash
sms-analyze normalized
```
Writes `outputs/normalized_texts.tsv` — a tab-separated file with three columns
(`label`, `original`, `normalized`) covering all 5 574 messages. Open it in
Excel / LibreOffice Calc or load it with `pd.read_csv(..., sep="\t")` to verify
what the normalisation pipeline does to individual messages (e.g. how `£100,000`,
`07-815-296`, or NUS placeholders like `&lt;#&gt;` are handled).
`sms-analyze all` includes this step automatically.

Generated plots, trained models, and the TSV are saved to `outputs/` by default.
`outputs/` is intentionally empty — it is the runtime output directory and
its contents are not included in this deliverable.
A pre-generated copy is available at
[notebooks/outputs/normalized_texts.tsv](notebooks/outputs/normalized_texts.tsv).

**Or open the Jupyter notebooks:**
```bash
jupyter notebook notebooks/
```

### Part 2 — Semantic Clustering (Task 5)

**Option A — Ollama (local, no API key)**

```bash
sms-analyze clustering
```

On first run the CLI checks whether Ollama is installed and running and whether
the model is pulled — and offers to handle each step interactively.
Use `--yes` to confirm all prompts automatically.

**Option B — OpenAI**
```bash
sms-analyze clustering --provider openai
# prompts for SMS_OPENAI_API_KEY on first run and offers to save it to .env
```

**Option C — Azure AI Foundry (Cohere-embed-v3-multilingual)**
```bash
sms-analyze clustering --provider azure
# prompts for SMS_AZURE_OPENAI_API_KEY and endpoint on first run
# endpoint format: https://<resource>.openai.azure.com/openai/v1
```

### Tests

```bash
python -m pytest tests/ -v
```

### Troubleshooting

**`sms-analyze all` hangs or takes very long**

On some Linux / macOS systems, OpenBLAS or OpenMP spawn more threads than the OS scheduler handles well inside a virtualenv, causing the process to stall. Fix:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 sms-analyze all
```

**Windows (PowerShell)**
```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
sms-analyze all
```

### Use as a library

A runnable end-to-end example is in [`scripts/example.py`](scripts/example.py).
It loads the dataset, trains the classifier, evaluates it, predicts new messages,
and prints the top model features.

```python
from pathlib import Path
from sms_analysis import load_dataset
from sms_analysis.classification import SpamClassifier, evaluate
from sklearn.model_selection import train_test_split

dataset = load_dataset(Path("assignment/3D8026C4.zip/SMSSpamCollection")).deduplicated
X_train, X_test, y_train, y_test = train_test_split(
    dataset.texts, dataset.labels, test_size=0.2, stratify=dataset.labels, random_state=42
)
clf = SpamClassifier("logistic_regression").fit(X_train, y_train)
result = evaluate(y_test, clf.predict(X_test), clf.predict_proba(X_test))
print(result.summary())
# Accuracy=0.963 | F1-SPAM=0.840 | Precision=0.935 | Recall=0.763 | ROC-AUC=0.991

predictions = clf.predict(["WINNER! Claim your free prize now!", "See you at 8?"])
# ['spam' 'ham']
```
