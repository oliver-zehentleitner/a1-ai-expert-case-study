#!/usr/bin/env bash
# Build the Sphinx HTML documentation.
# Source: dev/sphinx/   Output: docs/   (committed and shipped)
#
# Usage:
#   bash dev/sphinx/scripts/build_docs.sh           # build only
#   bash dev/sphinx/scripts/build_docs.sh --open    # build + open in default browser
#   bash dev/sphinx/scripts/build_docs.sh -W        # treat warnings as errors
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPEN=0
SPHINX_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --open|-o) OPEN=1 ;;
        *)         SPHINX_ARGS+=("$arg") ;;
    esac
done

sphinx-build -b html "$ROOT/dev/sphinx/config" "$ROOT/docs" "${SPHINX_ARGS[@]+"${SPHINX_ARGS[@]}"}"

# Required for GitHub Pages to serve Sphinx's _static/ directory correctly.
touch "$ROOT/docs/.nojekyll"

# Copy assignment files so relative links in README work when viewed from docs/.
if [[ -d "$ROOT/assignment" ]]; then
    rm -rf "$ROOT/docs/assignment"
    cp -r "$ROOT/assignment" "$ROOT/docs/assignment"
fi

echo ""
echo "Docs ready: docs/index.html"

if [[ $OPEN -eq 1 ]]; then
    case "$(uname -s)" in
        Darwin) open "$ROOT/docs/index.html" ;;
        MINGW*|MSYS*|CYGWIN*) start "$ROOT/docs/index.html" ;;
        *) xdg-open "$ROOT/docs/index.html" ;;
    esac
fi
