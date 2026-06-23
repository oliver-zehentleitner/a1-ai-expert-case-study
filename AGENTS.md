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
| `src/sms_analysis/data/` | Pydantic domain models + dataset loader |
| `src/sms_analysis/analysis/` | Tasks 1–3: word freq, HAM/SPAM comparison, n-grams |
| `src/sms_analysis/classification/` | Task 4: TF-IDF pipeline, LR/NB, evaluation |
| `src/sms_analysis/clustering/` | Task 5: `EmbeddingProvider` ABC + Ollama/OpenAI/Azure |
| `src/sms_analysis/visualization/` | Reusable Matplotlib/Seaborn plot functions |
| `src/sms_analysis/cli.py` | `sms-analyze` CLI entry point |
| `src/sms_analysis/interactive.py` | Zero-setup UX: secret prompts, Ollama lifecycle, `.env` writer |
| `src/sms_analysis/config.py` | pydantic-settings (`SMS_` prefix) — secrets + system settings |
| `notebooks/` | Jupyter presentation layer — import package, show results |
| `tests/` | pytest unit tests |
| `scripts/` | Shell utilities (Ollama setup, docs build) |
| `docs/` | Sphinx documentation source |
| `learning/` | Personal study notes — **local only** (see `.gitignore`) |
| `outputs/` | Generated plots and saved models (tracked only as `.gitkeep`) |
| `assignment/` | Original case study PDF + dataset (read-only, do not modify) |

---

## Key design decisions — preserve these

### src layout

The package lives under `src/` so that `import sms_analysis` in tests and
notebooks resolves to the installed editable package, never to stray files in the
project root. This prevents import-path bugs.

### Pydantic everywhere

Domain objects (`SMSMessage`, `SMSDataset`, `ClusteringResult`) are Pydantic
models. Config is `pydantic-settings`. This gives free validation, IDE
auto-complete, and `.model_dump()` serialisation.

### Strategy pattern for embedding providers

`EmbeddingProvider` (ABC in `clustering/base.py`) defines the interface.
`OllamaProvider`, `OpenAIProvider`, and `AzureOpenAIProvider` each implement it.
`SemanticClusterer` depends on the abstract interface, not on any concrete
provider. Switching backend = changing one CLI flag (`--provider`).

### Preprocessing baked into the sklearn Pipeline

`_preprocess_batch()` is the first step of the joblib-serialised sklearn
`Pipeline`. This eliminates train-serve skew: the same preprocessing applied at
training time is applied automatically at inference time.

**The function must remain a named module-level function — not a lambda.**
joblib cannot pickle lambdas when saving models.

### CLI flags for behaviour, env vars for secrets

`--provider` and `--model` are CLI flags. API keys and endpoints live in `.env`
/ environment variables. The interactive module bridges the gap: if a secret is
missing at runtime it prompts and offers to persist it.

### Fail loud

Raise / `sys.exit` with a clear message rather than silently degrading. A
cryptic result is worse than an explicit abort.

---

## Local-only files (in `.gitignore`)

The following files exist on each developer's machine but are **never committed**:

| File / pattern | Purpose |
|----------------|---------|
| `.env` | API keys, endpoints, optional overrides |
| `CLAUDE.md` | Claude-specific project instructions |
| `AGENTS.override.md` | Local instruction overrides for any AI agent |
| `CLAUDE.local.md` | Alternative local Claude config |
| `learning/` | Personal study notes and hearing prep material |
| `outputs/*.joblib` | Trained model artefacts |
| `outputs/*.pkl` | Pickled intermediate data |
| `outputs/embeddings_cache/` | Cached embedding vectors |
| `docs/_build/` | Compiled Sphinx output |

**AI agents:** use these files when they are present — they contain
project-specific and user-specific instructions. Do not recreate or commit them.

---

## What is always committed

| File | Notes |
|------|-------|
| `AGENTS.md` (this file) | Committed |
| `TASKS.md` | Committed when present |
| `scripts/` | All scripts |
| `docs/` | Source files; `_build/` excluded |
| `.env.example` | Template — no real secrets |
| `notebooks/` | With cleared outputs |

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

### Build documentation

```bash
bash scripts/install_docs_deps.sh   # first time only
bash scripts/build_docs.sh
# result: docs/_build/html/index.html
```

### Run all analyses

```bash
sms-analyze all
```

### Semantic clustering (guided wizard on first run)

```bash
sms-analyze clustering              # prompts for Ollama, model pull
sms-analyze clustering --yes        # auto-confirm all prompts
sms-analyze clustering --provider openai
sms-analyze clustering --provider azure
```

---

## PR workflow

| Remote | Repository |
|--------|-----------|
| `origin` | `oliver-zehentleitner-aigent/a1-ai-expert-case-study` (AI agent fork) |
| `upstream` | `oliver-zehentleitner/a1-ai-expert-case-study` (reviewed and merged here) |

- Always branch from `upstream/main`.
- PRs go from `oliver-zehentleitner-aigent:<branch>` → `oliver-zehentleitner:main`.
- Never commit to a branch whose PR has already been merged.

---

## Coding conventions

- Type hints on all public functions and class attributes.
- Google-style docstrings for public API.
- Modules use `logging.getLogger(__name__)` — `basicConfig` only in `cli.py`.
- No inline credits, author attributions, or tool references in code or commits.
