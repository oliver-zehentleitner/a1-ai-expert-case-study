"""
Interactive CLI helpers — zero-setup UX for first-time users.

Handles:
- Prompting for missing secrets (API keys, endpoints)
- Offering to persist them to .env
- Installing, starting, and pulling models for Ollama

Design rules:
- Secrets are collected via getpass (not echoed, not in shell history).
- Nothing is written to .env without explicit consent.
- ``--yes`` skips all prompts by answering "yes" to everything.
"""

import getpass
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Shell scripts bundled with the package (setup_ollama.sh / .ps1).
_SCRIPTS_DIR = Path(__file__).parent / "scripts"

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"

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


def _ask_choice(options: list[str], yes_index: int | None = None, yes: bool = False) -> int:
    """Numbered menu — returns the 0-based index of the chosen option.

    Args:
        options:   List of option labels to display.
        yes_index: Index to select automatically when ``yes=True``.
        yes:       If ``True`` and ``yes_index`` is set, skip the prompt.
    """
    if yes and yes_index is not None:
        print(f"  → {options[yes_index]} (auto)")
        return yes_index
    for i, label in enumerate(options, 1):
        print(f"  {i}) {label}")
    while True:
        try:
            raw = input("Choice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit("Aborted.")
        raw = raw or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  Please enter a number between 1 and {len(options)}.")


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
    """Locate the ollama binary in PATH or known install locations."""
    if found := shutil.which("ollama"):
        return found
    candidates = [
        Path.home() / ".local" / "bin" / "ollama",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    # Last resort on Windows: refresh PATH from registry (handles custom install locations).
    if platform.system() == "Windows":
        _refresh_path_windows()
        if found := shutil.which("ollama"):
            return found
    return None


def _ollama_reachable(base_url: str) -> bool:
    import httpx
    try:
        httpx.get(f"{base_url}/api/version", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


def _run_install_script() -> None:
    """Run the bundled Ollama setup script for the current platform."""
    if platform.system() == "Windows":
        script = _SCRIPTS_DIR / "setup_ollama.ps1"
        if not script.exists():
            sys.exit("Setup script not found. Install Ollama manually (see scripts/setup_ollama.ps1).")
        subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            check=True,
        )
    else:
        script = _SCRIPTS_DIR / "setup_ollama.sh"
        if not script.exists():
            sys.exit("Setup script not found. Install Ollama manually (see scripts/setup_ollama.sh).")
        subprocess.run(["bash", str(script)], check=True)


def _refresh_path_windows() -> None:
    """Re-read machine + user PATH from the registry into the current process."""
    result = subprocess.run(
        ["powershell", "-Command",
         "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' +"
         "[Environment]::GetEnvironmentVariable('Path','User')"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        os.environ["PATH"] = result.stdout.strip()


def _install_ollama(yes: bool) -> str:
    """Run the setup script and return the path to the installed binary."""
    print("Running Ollama installer…")
    _run_install_script()
    if platform.system() == "Windows":
        _refresh_path_windows()
    binary = _find_ollama_binary()
    if not binary:
        sys.exit(
            "Ollama installation completed but binary not found.\n"
            "Re-open your terminal so PATH is refreshed, then try again."
        )
    print(f"Ollama installed: {binary}")
    return binary


def _start_ollama_server(base_url: str, binary: str) -> None:
    """Launch `ollama serve` in the background and wait until it is reachable."""
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
    sys.exit("\nOllama did not respond within 12 seconds.")


def _ensure_ollama_model(base_url: str, model: str, binary: str, yes: bool) -> None:
    import httpx
    resp = httpx.get(f"{base_url}/api/tags", timeout=5)
    pulled = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
    if model in pulled:
        return

    if not _ask(f"[!] Model '{model}' is not pulled yet. Pull it now?", default=True, yes=yes):
        sys.exit(f"Aborted: model '{model}' not available.")

    print(f"Pulling {model}… (first run only, ~274 MB)")
    subprocess.run([binary, "pull", model], check=True)


def ensure_ollama(base_url: str, model: str, yes: bool) -> None:
    """Ensure Ollama is installed, running, and the requested model is available.

    Steps:

    1. **Installed?** — locate the binary; offer to run the setup script if absent.
    2. **Running?** — ping the API; offer Start / Try again / Abort in a loop.
    3. **Model pulled?** — check ``/api/tags``; offer to pull if missing.
    """
    # ── Step 1: installed? ───────────────────────────────────────────────────
    binary = _find_ollama_binary()
    if not binary:
        print("[!] Ollama is not installed.")
        if not _ask("Install it now using the bundled setup script?", default=True, yes=yes):
            sys.exit("Aborted: Ollama not installed.")
        binary = _install_ollama(yes)

    # ── Step 2: running? (loop with 3-way choice) ────────────────────────────
    while not _ollama_reachable(base_url):
        print("[!] Ollama is not running.")
        choice = _ask_choice(
            ["Start Ollama", "Check again", "Abort"],
            yes_index=0,
            yes=yes,
        )
        if choice == 0:
            _start_ollama_server(base_url, binary)
        elif choice == 1:
            continue   # re-evaluate the while condition
        else:
            sys.exit("Aborted: Ollama not running.")

    # ── Step 3: model available? ─────────────────────────────────────────────
    _ensure_ollama_model(base_url, model, binary, yes)


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

    # ── Azure AI Foundry ─────────────────────────────────────────────────────
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
                "https://<resource>.openai.azure.com/openai/v1",
            )
            missing["SMS_AZURE_OPENAI_ENDPOINT"] = endpoint

        if missing:
            _offer_save(missing, yes)

        return AzureOpenAIProvider(
            api_key=api_key,
            endpoint=endpoint,
            model=settings.azure_openai_model,
        )

    raise ValueError(f"Unknown provider: {provider_name!r}")
