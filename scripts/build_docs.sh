#!/usr/bin/env bash
# Build the Sphinx HTML documentation.
# Usage:  bash scripts/build_docs.sh [-W]   (-W turns warnings into errors)
# Run with your project venv already activated.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sphinx-build -b html "$ROOT/docs" "$ROOT/docs/_build/html" "$@"

echo ""
echo "Docs ready: docs/_build/html/index.html"
