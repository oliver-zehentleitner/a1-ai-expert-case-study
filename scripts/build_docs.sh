#!/usr/bin/env bash
# Build the Sphinx HTML documentation.
# Source: dev/sphinx/   Output: docs/   (committed and shipped)
# Usage:  bash scripts/build_docs.sh [-W]   (-W turns warnings into errors)
# Run with your project venv already activated.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sphinx-build -b html "$ROOT/dev/sphinx" "$ROOT/docs" "$@"

# Required for GitHub Pages to serve Sphinx's _static/ directory correctly.
touch "$ROOT/docs/.nojekyll"

echo ""
echo "Docs ready: docs/index.html"
