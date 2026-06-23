"""
Interactive CLI helpers — zero-setup UX for first-time users.

Handles:
- Prompting for missing secrets (API keys, endpoints)
- Offering to persist them to .env
- Checking / starting the Ollama server
- Pulling missing Ollama models

Design rules:
- Secrets are collected via getpass (not echoed, not in shell history).
- Nothing is written to .env without explicit consent.
- ``--yes`` skips all prompts by answering "yes" to everything.
"""

import getpass
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ENV_FILE = Path(".env")

# Set to True when the user answers "always" — persists for the session lifetime.
_AUTO_SAVE = False


# ---------------------------------------------------------------------------
# Low-level prompt helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: bool = True, yes: bool = False) -> bool:
    """Yes/no prompt.  Returns ``True`` for yes."""
    if yes or _AUTO_SAVE:
        hint = "Y/n" if default else "y/N"
        print(f"{prompt} [{hint}] → yes (auto)")
        return True
    hint = "Y/n" if default else "y/N"
    try:
        raw = input(f"{prompt} [{hint}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit("Aborted.")
    return (raw in ("y", "yes")) if raw else default


def _ask_secret(prompt: str) -> str:
    """Hidden input for passwords / API keys."""
    try:
        value = getpass.getpass(f"{prompt}: ")
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit("Aborted.")
    if not value.strip():
        sys.exit("Aborted: empty value not accepted.")
    return value.strip()


def _ask_input(prompt: str, default: str = "") -> str:
    """Plain text input with optional default."""
    display = f"{prompt} [{default}]: " if default else f"{prompt}: "
    try:
        raw = input(display).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit("Aborted.")
    return raw or default


def _offer_save(pairs: dict[str, str], yes: bool) -> None:
    """Offer to write key=value pairs to .env, respecting AUTO_SAVE."""
    global _AUTO_SAVE

    if _AUTO_SAVE:
        _write_env(pairs)
        return

    if yes:
        _write_env(pairs)
        return

    try:
        raw = input("Save to .env for future use? [y/N/always]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return  # no save on interrupt

    if raw in ("always", "a"):
        _AUTO_SAVE = True
        _write_env(pairs)
    elif raw in ("y", "yes"):
        _write_env(pairs)
    # else: discard — used only for this session


def _write_env(pairs: dict[str, str]) -> None:
    """Append missing keys to .env (never overwrites existing values)."""
    existing: set[str] = set()
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                existing.add(line.split("=", 1)[0].strip())

    new_lines = [f"{k}={v}" for k, v in pairs.items() if k not in existing]
    if new_lines:
        with _ENV_FILE.open("a") as f:
            f.write("\n" + "\n".join(new_lines) + "\n")
        print(f"Saved {list(pairs)} to {_ENV_FILE}")


# ---------------------------------------------------------------------------
# Ollama lifecycle
# ---------------------------------------------------------------------------

def _find_ollama_binary() -> str | None:
    """Locate the ollama binary in PATH or the user install location."""
    if found := shutil.which("ollama"):
        return found
    candidate = Path.home() / ".local" / "bin" / "ollama"
    if candidate.is_file():
        return str(candidate)
    return None


def _ollama_reachable(base_url: str) -> bool:
    import httpx
    try:
        httpx.get(f"{base_url}/api/version", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


def _start_ollama(base_url: str) -> None:
    binary = _find_ollama_binary()
    if not binary:
        print(
            "[!] Ollama binary not found.\n"
            "    Install it with:  bash scripts/setup_ollama.sh"
        )
        sys.exit("Aborted: Ollama not installed.")

    print(f"Starting Ollama ({binary})…")
    subprocess.Popen(
        [binary, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for i in range(12):
        time.sleep(1)
        if _ollama_reachable(base_url):
            print("Ollama is ready.")
            return
        print(f"  waiting… ({i + 1}s)", end="\r")
    sys.exit("\nOllama did not start within 12 seconds.")


def _ensure_ollama_model(base_url: str, model: str, yes: bool) -> None:
    import httpx
    resp = httpx.get(f"{base_url}/api/tags", timeout=5)
    pulled = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
    if model in pulled:
        return

    if not _ask(f"[!] Model '{model}' is not pulled yet. Pull it now?", default=True, yes=yes):
        sys.exit(f"Aborted: model '{model}' not available.")

    binary = _find_ollama_binary()
    print(f"Pulling {model}… (first run only, ~274 MB)")
    subprocess.run([binary, "pull", model], check=True)


def ensure_ollama(base_url: str, model: str, yes: bool) -> None:
    """Make sure Ollama is running and the requested model is available."""
    if not _ollama_reachable(base_url):
        if not _ask("[!] Ollama is not running. Start it now?", default=True, yes=yes):
            sys.exit("Aborted: Ollama not available.")
        _start_ollama(base_url)

    _ensure_ollama_model(base_url, model, yes)


# ---------------------------------------------------------------------------
# Provider factory — the single entry point for cli.py
# ---------------------------------------------------------------------------

def build_provider(args, settings):
    """
    Return a fully configured :class:`EmbeddingProvider`.

    Prompts interactively for any missing secrets and offers to persist them
    to ``.env``.  Pass ``--yes`` to accept all prompts automatically.
    """
    from .clustering.providers import AzureOpenAIProvider, OllamaProvider, OpenAIProvider

    yes = getattr(args, "yes", False)
    provider_name: str = args.provider

    # ── Ollama ──────────────────────────────────────────────────────────────
    if provider_name == "ollama":
        ensure_ollama(settings.ollama_base_url, settings.ollama_model, yes)
        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    # ── OpenAI ──────────────────────────────────────────────────────────────
    if provider_name == "openai":
        api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[!] SMS_OPENAI_API_KEY not set.")
            api_key = _ask_secret("Enter SMS_OPENAI_API_KEY")
            _offer_save({"SMS_OPENAI_API_KEY": api_key}, yes)
        return OpenAIProvider(api_key=api_key, model=settings.openai_model)

    # ── Azure OpenAI ─────────────────────────────────────────────────────────
    if provider_name == "azure":
        missing: dict[str, str] = {}

        api_key = settings.azure_openai_api_key
        if not api_key:
            print("[!] SMS_AZURE_OPENAI_API_KEY not set.")
            api_key = _ask_secret("Enter SMS_AZURE_OPENAI_API_KEY")
            missing["SMS_AZURE_OPENAI_API_KEY"] = api_key

        endpoint = settings.azure_openai_endpoint
        if not endpoint:
            print("[!] SMS_AZURE_OPENAI_ENDPOINT not set.")
            endpoint = _ask_input(
                "Enter SMS_AZURE_OPENAI_ENDPOINT",
                "https://<resource>.openai.azure.com/",
            )
            missing["SMS_AZURE_OPENAI_ENDPOINT"] = endpoint

        deployment = settings.azure_openai_deployment
        if not deployment:
            deployment = _ask_input(
                "Enter SMS_AZURE_OPENAI_DEPLOYMENT",
                "text-embedding-3-small",
            )
            missing["SMS_AZURE_OPENAI_DEPLOYMENT"] = deployment

        if missing:
            _offer_save(missing, yes)

        return AzureOpenAIProvider(
            api_key=api_key,
            endpoint=endpoint,
            deployment=deployment,
            api_version=settings.azure_openai_api_version,
        )

    raise ValueError(f"Unknown provider: {provider_name!r}")
