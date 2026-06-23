#!/usr/bin/env bash
# Install Ollama and pull the embedding model used for semantic clustering.
# Safe to run multiple times (idempotent).
#
# Installs to ~/.local/bin (no sudo required).
# Add ~/.local/bin to PATH if not already present.
set -euo pipefail

MODEL="${SMS_OLLAMA_MODEL:-nomic-embed-text}"
INSTALL_DIR="$HOME/.local/bin"
OLLAMA="$INSTALL_DIR/ollama"

mkdir -p "$INSTALL_DIR"

echo "==> Checking Ollama installation..."
if command -v ollama &>/dev/null; then
    OLLAMA="$(command -v ollama)"
    echo "==> Ollama already installed: $($OLLAMA --version 2>&1 | head -1)"
elif [ -x "$OLLAMA" ]; then
    echo "==> Ollama found at $OLLAMA"
else
    echo "==> Installing Ollama to $INSTALL_DIR (no sudo required)..."

    LATEST=$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest \
        | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['tag_name'])")
    URL="https://github.com/ollama/ollama/releases/download/${LATEST}/ollama-linux-amd64.tar.zst"

    echo "==> Downloading Ollama ${LATEST}..."
    curl -fsSL "$URL" -o /tmp/ollama.tar.zst

    echo "==> Extracting..."
    tar --use-compress-program=unzstd -xf /tmp/ollama.tar.zst -C "$INSTALL_DIR"
    # The archive extracts to bin/ollama — move to flat location
    [ -f "$INSTALL_DIR/bin/ollama" ] && mv "$INSTALL_DIR/bin/ollama" "$OLLAMA"
    rm -f /tmp/ollama.tar.zst

    chmod +x "$OLLAMA"
    echo "==> Installed: $($OLLAMA --version 2>&1 | head -1)"
fi

echo "==> Starting Ollama service (if not running)..."
if ! pgrep -x ollama &>/dev/null; then
    "$OLLAMA" serve &>/dev/null &
    sleep 3
fi

echo "==> Pulling model: ${MODEL}"
"$OLLAMA" pull "${MODEL}"

echo ""
echo "Setup complete."
echo ""
echo "Add to PATH if needed:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
echo "Run the clustering analysis:"
echo "  sms-analyze clustering"
