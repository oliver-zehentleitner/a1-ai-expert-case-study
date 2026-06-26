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

# Stage README, assignment/ and notebooks/ into the Sphinx source dir so
# myst-parser resolves relative links correctly during the build.
cp "$ROOT/README.md" "$ROOT/dev/sphinx/config/README.md"
if [[ -d "$ROOT/assignment" ]]; then
    cp -r "$ROOT/assignment" "$ROOT/dev/sphinx/config/assignment"
fi
if [[ -d "$ROOT/notebooks" ]]; then
    cp -r "$ROOT/notebooks" "$ROOT/dev/sphinx/config/notebooks"
fi

sphinx-build -b html "$ROOT/dev/sphinx/config" "$ROOT/docs" "${SPHINX_ARGS[@]+"${SPHINX_ARGS[@]}"}"

# Clean up staged files — they are build-time only, not source-controlled.
rm -f "$ROOT/dev/sphinx/config/README.md"
rm -rf "$ROOT/dev/sphinx/config/assignment"
rm -rf "$ROOT/dev/sphinx/config/notebooks"

# Required for GitHub Pages to serve Sphinx's _static/ directory correctly.
touch "$ROOT/docs/.nojekyll"

# Copy assignment files into docs/ so links work when browsing docs/index.html.
if [[ -d "$ROOT/assignment" ]]; then
    rm -rf "$ROOT/docs/assignment"
    cp -r "$ROOT/assignment" "$ROOT/docs/assignment"
fi

# docs/notebooks/ only needs the nbconvert HTML output; the .ipynb files are
# served via Sphinx's _downloads/ mechanism and don't need to live here too.
mkdir -p "$ROOT/docs/notebooks"

# Convert notebooks to HTML for read-only viewing in the docs.
# Uses committed cell outputs — notebooks are not re-executed during the build.
echo "Converting notebooks to HTML…"
VENV_JUPYTER="$ROOT/.venv/bin/jupyter"
if [[ -x "$VENV_JUPYTER" ]]; then
    for nb in "$ROOT/notebooks"/*.ipynb; do
        "$VENV_JUPYTER" nbconvert \
            --to html \
            --output-dir "$ROOT/docs/notebooks/" \
            --no-prompt \
            "$nb" 2>/dev/null
        echo "  converted: $(basename "$nb")"
    done
else
    echo "  WARNING: .venv/bin/jupyter not found — skipping notebook HTML conversion"
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
