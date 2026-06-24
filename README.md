# AI Expert Case Study – Oliver Zehentleitner

Solution to the [A1 Telekom Austria AI Expert case study](./assignment/Fallbeispiel.pdf)
([job posting](https://www.isg.com/jobs/ad/AI_Expert-110091)).

**[Full documentation](docs/index.html)** — CLI reference, API autodoc, architecture overview.

The SMS dataset is located at
[`./assignment/3D8026C4.zip/SMSSpamCollection`](./assignment/3D8026C4.zip/SMSSpamCollection),
based on the publicly available
[SMS Spam Collection v.1](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)
by Tiago A. Almeida and José María Gómez Hidalgo. Used here for this case study only.

---

## Summary of the task

The case study covers two focus areas applied to an English SMS dataset
(5,574 messages: 86 % HAM / 14 % SPAM):

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
Rather than assuming their origin, we verified it by counting how many
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
map to `__NUM__` after normalisation. The ML signal is not the token itself
but the *density and combination* of such tokens — spam messages are number-heavy
(multiple phone numbers, shortcodes, prices, dates), while HAM messages
containing `__NUM__` typically have just one or two occurrences.

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

| Input pattern | Normalised to |
|--------------|--------------|
| `@ &lt;URL&gt;`, `@handle` | `__USER__` |
| `&lt;URL&gt;`, `http://…`, `www.…` | `__URL__` |
| `&lt;#&gt;`, `&lt;DECIMAL&gt;`, `&lt;NUMBER&gt;`, standalone digits (≥2) | `__NUM__` |
| `&lt;TIME&gt;` | `__TIME__` |
| `&lt;EMAIL&gt;` | `__EMAIL__` |
| `£`, `$`, `€`, `¥`, `₹`, `₽`, `₩`, `₿` | `__CURRENCY__` |
| `&amp;` | `and` |
| Other `&lt;…&gt;` | removed |

**Why tokens instead of deletion?** The *presence* of a URL, username mention,
or number is a strong statistical signal — spam messages are token-dense in a
way that HAM messages are not. Deleting them discards that signal. Replacing
them with a consistent placeholder lets TF-IDF and the embedding model learn
the pattern while keeping the vocabulary clean.

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

The corpus is a mix of UK and Singaporean English. The HAM messages from the
NUS corpus reflect the vocabulary and writing style of Singaporean university
students (e.g. "lor", "lah", "da" as discourse particles). The SPAM messages
were collected from a UK forum and reflect UK premium-rate SMS fraud patterns
("prize", "claim", "await collection", "150p per msg", "BT national rate").

A model trained on this data is a reasonable UK/Singapore spam detector.
Applied to Austrian, German, or Brazilian traffic, it would encounter different
vocabulary, different fraud patterns, different SMS conventions — and its
performance would degrade accordingly. The training data does not cover these
cases, so the model cannot be expected to generalise to them.

**Currency symbols as a concrete example**

In this dataset, `£` (British pound) appears almost exclusively in SPAM.
Without normalisation, the model would learn "£ = spam signal" — which is true
for UK data but useless (or misleading) for traffic where `€`, `$`, or `₹`
are the local currency. The preprocessing step normalises all currency symbols
to `__CURRENCY__`, so the model learns "a currency symbol before an amount =
spam signal" rather than anything tied to a specific currency. This is a
deliberate decision to improve generalisability within the limits of the
available data.

**Spurious correlations and adversarial robustness — a concrete example**

The NUS corpus contains ~50 HAM messages where one or a few contributors used
`Ü` as SMS shorthand for "you" (identical in meaning to the common `u`). The
likely cause: a Nokia phone with a European language pack where the `U` key
cycled to `Ü` on repeated press. In this dataset `Ü` appears 50 times in HAM
and exactly zero times in SPAM.

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

Generated plots and models are saved to `outputs/` (gitignored — transient).
A committed snapshot of a clean run is in `solution/`.

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

Open the docs in your default browser:

```bash
sms-analyze docs
```

To rebuild the docs from source (e.g. after code changes):

```bash
pip install -e ".[docs]"                    # first time only
bash dev/sphinx/scripts/build_docs.sh
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
