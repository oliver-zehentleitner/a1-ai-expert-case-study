# AGENTS.md

## Project context

This repository is a job application case study submitted to A1 Telekom Austria
for an "AI Expert" position. It analyses an English SMS spam/ham dataset with
classical ML (TF-IDF + Logistic Regression) and LLM-based embedding clustering.

The primary audience for this code is the hiring committee. The goal is to
demonstrate software design, architecture thinking, and applied ML competence —
not to ship a product.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/sms_analysis/` | Installable Python package — the real deliverable |
| `src/sms_analysis/data/` | Pydantic domain models, dataset loader, preprocessing |
| `src/sms_analysis/analysis/` | Tasks 1–3: word freq, HAM/SPAM comparison, n-grams |
| `src/sms_analysis/classification/` | Task 4: TF-IDF pipeline, LR/NB, evaluation |
| `src/sms_analysis/clustering/` | Task 5: `EmbeddingProvider` ABC + Ollama/OpenAI/Azure |
| `src/sms_analysis/visualization/` | Reusable Matplotlib/Seaborn plot functions |
| `src/sms_analysis/cli.py` | `sms-analyze` CLI entry point |
| `src/sms_analysis/interactive.py` | Zero-setup UX: secret prompts, Ollama lifecycle, `.env` writer |
| `src/sms_analysis/config.py` | pydantic-settings (`SMS_` prefix) — secrets + system settings |
| `src/sms_analysis/scripts/` | `setup_ollama.sh` / `.ps1` — bundled with the package via package-data |
| `notebooks/` | Jupyter presentation layer — imports the package, shows results |
| `tests/` | pytest unit tests |
| `dev/sphinx/config/` | Sphinx source files (`conf.py`, `*.rst`) — edit these |
| `dev/sphinx/scripts/` | `build_docs.sh` (Linux/macOS), `build_docs.ps1` (Windows), `install_docs_deps.sh` |
| `docs/` | Committed generated HTML — built from `dev/sphinx/config/`, shipped as-is |
| `assignment/` | Original case study PDF + dataset ZIP (read-only, do not modify) |
| `outputs/` | Generated plots and saved models — **fully gitignored** (only `.gitkeep` tracked); populated on each run |
| `solution/` | Committed snapshot of one clean run — plots + trained models, the shipped deliverable |
| `learning/` | Personal study notes — **local only** (see `.gitignore`) |

---

## Dataset — what every agent should know immediately

**File:** `assignment/3D8026C4.zip/SMSSpamCollection`
**Format:** Tab-separated, no header. Two columns: `label` (`ham`/`spam`) and `text`.

**Class distribution:**

| Class | Count | Share |
|-------|------:|------:|
| HAM   | 4,827 |  86 % |
| SPAM  |   747 |  14 % |

This imbalance is fundamental: accuracy is a meaningless metric. A model that
always predicts HAM scores 86% without learning anything. Use **F1-SPAM** and
**ROC-AUC** as primary metrics.

### HTML-encoded tokens in the raw data

Some messages contain HTML-encoded entities like `&lt;URL&gt;` or `&lt;#&gt;`.
**Origin verified:** every such token appears exclusively in HAM messages —
zero occurrences in SPAM. They are privacy-driven anonymisation placeholders
inserted by the NUS SMS Corpus team before publishing (phone numbers, amounts,
times, and e-mail addresses were replaced with typed tokens).
The SPAM messages (Grumbletext) were not processed the same way and retain
real phone numbers, prices, and URLs.

| Token | Messages containing | HAM | SPAM |
|-------|--------------------:|----:|-----:|
| `&lt;#&gt;` | 214 | 214 | 0 |
| `&lt;DECIMAL&gt;` | 21 | 21 | 0 |
| `&lt;TIME&gt;` | 7 | 7 | 0 |
| `&lt;URL&gt;` | 3 | 3 | 0 |
| `&lt;EMAIL&gt;` | 1 | 1 | 0 |

These are **not** real HTML — they are ASCII text stored in a plain TSV file.

### A subtle dataset quirk: `&lt;URL&gt;` ≠ always a URL

The `&lt;URL&gt;` token was used by the dataset authors (or the copy-paste process)
to replace **both** actual hyperlinks **and** social-media usernames/handles.
The pattern `@ &lt;URL&gt;` in a message indicates a username mention, not a link.
This is why preprocessing maps `@ &lt;URL&gt;` → `TOKEN_USER` (before the general URL
pass) rather than `TOKEN_URL`.

### Real URLs in the data

A small number of messages contain actual raw URLs (`http://…`, `www.…`).
In this dataset they appear exclusively in SPAM — but that is a property of the
sample, not a general truth. Ham messages regularly contain links in real life.

### URLs, phone numbers, and ML — an important conceptual boundary

The **presence** of a URL or phone number is a strong statistical signal for the
ML model (spam messages contain them far more often in this dataset). Tokenising
them preserves this signal.

However, a URL or phone number is **not inherently spam or ham**, and this can
change at any time:
- A domain that is clean today can be compromised tomorrow.
- A spammer can register a fresh domain that appears on no blacklist yet.
- Phone numbers get recycled, spoofed, or legitimately used by businesses.

**The ML classifier decides:** "messages containing a URL tend to be spam."
**A separate reputation layer decides:** "this specific URL/number is known-bad."

URL and phone-number reputation belong in dedicated systems — URL blacklists,
threat-intel feeds (IoC matching), carrier reputation services — not in text
classification. The two signals are complementary and should not be conflated.

---

## Preprocessing — `src/sms_analysis/data/preprocessing.py`

All normalisation happens in `preprocess(text: str) -> str`.
**Pattern application order matters** — `TOKEN_USER` must be matched before
the general `TOKEN_URL` pass or username placeholders get misclassified as links.

| Input pattern | Regex | Output |
|--------------|-------|--------|
| `Ü` / `ü` | `_U_UMLAUT` | `u` (shorthand for "you"; removes spurious HAM bias) |
| `@ &lt;URL&gt;` | `_AT_USER_HTML` | `TOKEN_USER` |
| `@handle` | `_AT_USER_RAW` | `TOKEN_USER` |
| `&lt;URL&gt;` | `_HTML_URL` | `TOKEN_URL` |
| `http://…`, `www.…` | `_REAL_URL` | `TOKEN_URL` |
| `&lt;#&gt;`, `&lt;DECIMAL&gt;`, `&lt;NUMBER&gt;` | `_HTML_NUM` | `TOKEN_NUM` |
| standalone digits (≥ 2) | `_STANDALONE_NR` | `TOKEN_NUM` |
| `&lt;TIME&gt;` | `_HTML_TIME` | `TOKEN_TIME` |
| `&lt;EMAIL&gt;` | `_HTML_EMAIL` | `TOKEN_EMAIL` |
| `£`, `$`, `€`, `¥`, `₹`, `₽`, `₩`, `₿` | `_CURRENCY` | `TOKEN_CURRENCY` |
| `&amp;` | `_HTML_AMP` | `and` |
| any remaining `&lt;…&gt;` | `_HTML_LEFTOVER` | *(removed + WARNING logged)* |

NUS placeholders (`&lt;URL&gt;`, `&lt;#&gt;`, …) map to the **same tokens** as their
typed equivalents — the *presence* of a URL or number is the signal, not
the specific value. For URL/number reputation (blacklists, IoC feeds) a
separate system is needed.

This function is also the first step of the serialised sklearn `Pipeline`
(see Classification section). Preprocessing at inference is therefore automatic.

---

## Classification — `src/sms_analysis/classification/`

### Pipeline structure

```
FunctionTransformer(_preprocess_batch)
    → TfidfVectorizer(ngram_range=(1,2), max_features=10_000, sublinear_tf=True)
    → LogisticRegression(solver="liblinear", C=1.0) or MultinomialNB()
```

The pipeline is saved as a `.joblib` file. Loading it gives a fully functional
classifier that applies the same preprocessing automatically.

**Critical:** `_preprocess_batch` must remain a **named module-level function**.
joblib cannot pickle lambdas. If you move or rename it, model serialisation breaks.

### TF-IDF choices and why

- `sublinear_tf=True` — log(1 + tf) dampens very frequent terms; raw counts
  would let a word appearing 50× in one spam message dominate the signal.
- `ngram_range=(1,2)` — bigrams capture phrases like "free entry" or "win prize"
  that are individually ambiguous.
- `stop_words="english"` — removes function words that carry no spam signal.
- `max_features=10_000` — caps vocabulary size; avoids overfitting on rare tokens.

### Model choice rationale

- **Logistic Regression (primary):** Coefficients are directly interpretable —
  each token has a signed weight. Excellent on sparse TF-IDF vectors. The
  `top_features()` method returns the top-N spam/ham words by coefficient.
- **Naive Bayes (baseline):** Fast, probabilistic, zero hyperparameter tuning.
  Good sanity check and standalone presentation point.
- **Why not SVM / Random Forest / deep learning?** SVM is competitive but less
  interpretable. Ensembles are optimised for dense tabular data. Deep learning
  is overkill for 5,500 samples where linear models already reach 99%+ AUC.

### Evaluation

Primary metrics: **F1-SPAM** and **ROC-AUC** (accuracy is misleading here).
The CLI uses a single stratified 80/20 train/test split on the **deduplicated**
dataset (`SMSDataset.deduplicated` — 5,171 unique messages; 403 raw duplicates
removed to prevent the same text appearing in both splits).
Results are encapsulated in `EvaluationResult` (dataclass).

Actual metrics on deduplicated data:
F1-SPAM=0.840, Precision=0.935, Recall=0.763, ROC-AUC=0.991

---

## Clustering — `src/sms_analysis/clustering/`

### Architecture

`EmbeddingProvider` (ABC in `base.py`) defines the interface:
- `embed(texts: list[str]) -> list[list[float]]`
- `dimensions: int` (property)

Three concrete providers: `OllamaProvider`, `OpenAIProvider`, `AzureOpenAIProvider`.
`SemanticClusterer` depends **only** on the ABC — swapping providers = one CLI flag.

### K-Means + automatic K selection

If `n_clusters` is not given, `SemanticClusterer` tries every K in `range(3, 16)`
and picks the one with the highest silhouette score. Inertia values are stored
in `ClusteringResult.inertias` for elbow-plot visualisation.

### 2D reduction for plotting

UMAP is used for 2D reduction (better cluster separation than t-SNE for this
use case). Graceful fallback to PCA if `umap-learn` is not installed.

### Default embedding models

| Provider | Default model | Dimensions |
|----------|--------------|-----------|
| Ollama | `nomic-embed-text` | 768 |
| OpenAI | `text-embedding-3-small` | 1536 |
| Azure | `text-embedding-3-small` | 1536 |

---

## Configuration — `src/sms_analysis/config.py`

All settings use `pydantic-settings` with the `SMS_` prefix and `.env` file support.

Key settings:

| Env var | Default | Notes |
|---------|---------|-------|
| `SMS_DATA_PATH` | `assignment/3D8026C4.zip/SMSSpamCollection` | Path to dataset |
| `SMS_OUTPUTS_DIR` | `outputs` | Where plots and models are saved |
| `SMS_OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `SMS_OLLAMA_MODEL` | `nomic-embed-text` | |
| `SMS_OPENAI_API_KEY` | *(empty)* | Prompted interactively if missing |
| `SMS_AZURE_OPENAI_API_KEY` | *(empty)* | Prompted interactively if missing |
| `SMS_AZURE_OPENAI_ENDPOINT` | *(empty)* | |
| `SMS_AZURE_OPENAI_DEPLOYMENT` | `text-embedding-3-small` | |
| `SMS_AZURE_OPENAI_API_VERSION` | `2024-05-01-preview` | |

Provider selection (`ollama` / `openai` / `azure`) is a **CLI flag** (`--provider`),
not an env var. Secrets and system-level settings live in the env / `.env` file.

---

## Interactive / zero-setup UX — `src/sms_analysis/interactive.py`

First-time users run `sms-analyze clustering` with no setup. The module:

1. **Checks if Ollama is installed** — looks in PATH and `~/.local/bin/ollama`.
   If absent, offers to run the bundled setup script (`scripts/setup_ollama.sh`).
2. **Checks if Ollama is running** — pings `/api/version`. If not reachable,
   shows a 3-way menu: Start / Check again / Abort. Waits up to 12 s after start.
3. **Checks if the model is pulled** — queries `/api/tags`. If missing, offers
   to run `ollama pull <model>`.

Secrets (API keys) are collected via `getpass` (not echoed, not in shell history).
The user is offered to persist them to `.env` (append-only — never overwrites
existing keys). Answering "always" sets a session-wide `_AUTO_SAVE` flag.

`--yes` / `-y` CLI flag auto-confirms all interactive prompts (useful for CI
or demos).

The Ollama setup scripts live **inside the package** at
`src/sms_analysis/scripts/` and are bundled via `[tool.setuptools.package-data]`
so they work after `pip install` without knowing the source checkout location.

---

## CLI — `sms-analyze <task>`

```
sms-analyze frequency       # Task 1 – top-20 word frequencies
sms-analyze comparison      # Task 2 – log-odds HAM vs SPAM
sms-analyze ngrams          # Task 3 – top bigrams and trigrams
sms-analyze classification  # Task 4 – LR classifier + evaluation
sms-analyze clustering      # Task 5 – semantic clustering (Ollama default)
sms-analyze docs            # opens docs/index.html in default browser
sms-analyze all             # runs Part 1 only (tasks 1–4); clustering is always explicit
```

Clustering-specific flags:
```
--provider ollama|openai|azure
--k INT          # fixed number of clusters (default: auto via silhouette)
--sample INT     # embed only a random sample of N messages
--yes / -y       # auto-confirm all interactive prompts
```

Classification flag:
```
--model logistic_regression|naive_bayes  (default: logistic_regression)
```

---

## Key design decisions — preserve these

### src layout

The package lives under `src/` so that `import sms_analysis` resolves to the
installed editable package, never to stray files in the project root. Prevents
import-path bugs in tests and notebooks.

### Pydantic everywhere

Domain objects (`SMSMessage`, `SMSDataset`, `ClusteringResult`) are Pydantic
models. Config is `pydantic-settings`. Free validation, IDE auto-complete, and
`.model_dump()` serialisation.

Key `SMSDataset` properties beyond `.texts` / `.labels`:
- `.normalized_texts` / `.normalized_ham_texts` / `.normalized_spam_texts` — on-demand normalisation via `preprocess()`
- `.deduplicated` — new `SMSDataset` with exact duplicate texts removed; use before train/test splits
- `.ham` / `.spam` — filtered sublists

### Strategy pattern for embedding providers

`EmbeddingProvider` (ABC) defines the interface. Concrete providers implement it.
`SemanticClusterer` depends only on the ABC. Switching backend = one flag.

### Preprocessing baked into the sklearn Pipeline

`_preprocess_batch` is the first step of the joblib-serialised sklearn `Pipeline`.
Eliminates train-serve skew. **Must remain a named module-level function.**

### CLI flags for behaviour, env vars for secrets

`--provider` and `--model` are CLI flags. API keys live in `.env`. The
interactive module bridges the gap if a secret is missing at runtime.

### Fail loud

Raise / `sys.exit` with a clear message rather than silently degrading. A
cryptic result is worse than an explicit abort.

---

## Docs workflow

```
dev/sphinx/
├── config/     ← Sphinx sourcedir (conf.py + .rst files) — edit these
└── scripts/    ← build_docs.sh / build_docs.ps1, install_docs_deps.sh

docs/           ← committed generated HTML — do not edit manually
```

Build:
```bash
# Linux / macOS
pip install -e ".[docs]"                      # first time only
bash dev/sphinx/scripts/build_docs.sh         # build into docs/

# Windows (PowerShell)
pip install -e ".[docs]"
.\dev\sphinx\scripts\build_docs.ps1
```

`build_docs.sh` / `build_docs.ps1` also copies `assignment/` into `docs/assignment/` so that
relative links in `README.md` resolve correctly when `docs/index.html` is
opened directly (local file, GitHub Pages, or `sms-analyze docs`).

To open the docs without rebuilding:
```bash
sms-analyze docs
```

---

## Development workflow

### Install

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### Test

```bash
pytest tests/ -v
```

### Run all analyses

```bash
sms-analyze all
```

---

## Local-only files (in `.gitignore`)

The following files exist on each developer's machine but are **never committed**:

| File / pattern | Purpose |
|----------------|---------|
| `.env` | API keys, endpoints, optional overrides |
| `CLAUDE.md` | Claude-specific project instructions |
| `AGENTS.override.md` | Local instruction overrides for any AI agent |
| `CLAUDE.local.md` | Alternative local Claude config |
| `learning/` | Personal study notes and case-study prep material |
| `outputs/*` | All generated files (plots, models, cache) — only `.gitkeep` committed |
| `dev/sphinx/_build/` | Sphinx build cache |

**AI agents:** if `AGENTS.override.md` is present, read it — it contains
project- and user-specific instruction overrides.

---

## What is always committed

| File | Notes |
|------|-------|
| `AGENTS.md` (this file) | Committed |
| `TASKS.md` | Committed when present |
| `dev/sphinx/` | Sphinx sources and scripts |
| `docs/` | Generated HTML (`_build/` excluded) |
| `.env.example` | Template — no real secrets |
| `notebooks/` | With cleared outputs |
| `solution/` | Committed plots + trained models from one clean run |

---

## PR workflow

| Remote | Repository |
|--------|-----------|
| `origin` | `oliver-zehentleitner-aigent/a1-ai-expert-case-study` (AI agent fork) |
| `upstream` | `oliver-zehentleitner/a1-ai-expert-case-study` (reviewed and merged here) |

- Always branch from `upstream/main` (run `git fetch upstream` first).
- PRs go from `oliver-zehentleitner-aigent:<branch>` → `oliver-zehentleitner:main`.
- Never commit to a branch whose PR has already been merged — open a new branch.
- Check `gh pr list --repo oliver-zehentleitner/a1-ai-expert-case-study --state all`
  before starting work to understand what has already landed.

---

## Coding conventions

- Type hints on all public functions and class attributes.
- Google-style docstrings for public API.
- Modules use `logging.getLogger(__name__)` — `basicConfig` only in `cli.py`.
- No inline credits, author attributions, or tool references in code or commits.
- No "Generated with …" anywhere — not in code, not in commit messages, not in
  PR titles or bodies.
