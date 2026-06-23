#!/usr/bin/env bash
# Install Sphinx documentation build dependencies into the active virtualenv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi

pip install \
    "sphinx>=7.3" \
    "furo>=2024.1" \
    "sphinx-autodoc-typehints>=2.2" \
    "myst-parser>=3.0"

echo "Done. Run  bash scripts/build_docs.sh  to build."
