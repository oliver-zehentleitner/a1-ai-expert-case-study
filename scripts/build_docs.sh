#!/usr/bin/env bash
# Build the Sphinx HTML documentation.
# Usage:  bash scripts/build_docs.sh [-W]   (-W turns warnings into errors)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi

sphinx-build -b html "$ROOT/docs" "$ROOT/docs/_build/html" "$@"

echo ""
echo "Docs ready: docs/_build/html/index.html"
