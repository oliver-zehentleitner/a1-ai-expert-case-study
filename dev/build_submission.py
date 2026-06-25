#!/usr/bin/env python3
"""
Build a submission ZIP for A1.

Usage (from any directory):
    python dev/build_submission.py

Output: dist/a1-submission-YYYY-MM-DD.zip
"""

import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

# Any path component matching one of these → skip the whole subtree
SKIP_DIR_COMPONENTS = {
    ".git",
    "_build",
    "dev",
    ".idea",
    "learning",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "sms_analysis.egg-info",
    "dist",
}

# Subtrees relative to ROOT whose files are skipped entirely
SKIP_SUBTREES = [
    Path("docs/_sources"),
    Path("docs/_modules"),
    Path("docs/_downloads"),
    Path("docs/.doctrees"),
    Path("outputs"),         # included as empty dir — evaluators run sms-analyze themselves
    Path("notebooks/outputs"),
]

# Empty directories to add explicitly (ZIP doesn't store dirs without files)
EMPTY_DIRS = ["outputs/"]

# Filenames (basename only) to skip wherever they appear
SKIP_NAMES = {
    ".gitignore",
    ".gitkeep",
    "AGENTS.md",
    "AGENTS.override.md",
}

SKIP_SUFFIXES = {".pyc"}


def should_skip(rel: Path) -> bool:
    for part in rel.parts:
        if part in SKIP_DIR_COMPONENTS:
            return True
    for subtree in SKIP_SUBTREES:
        try:
            rel.relative_to(subtree)
            return True
        except ValueError:
            pass
    if rel.name in SKIP_NAMES:
        return True
    if rel.suffix in SKIP_SUFFIXES:
        return True
    return False


def main() -> None:
    DIST.mkdir(exist_ok=True)
    zipname = f"a1-submission-{date.today()}.zip"
    zippath = DIST / zipname

    included: list[Path] = []

    with zipfile.ZipFile(zippath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for empty_dir in EMPTY_DIRS:
            zf.writestr(zipfile.ZipInfo(empty_dir), "")
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if should_skip(rel):
                continue
            zf.write(path, rel)
            included.append(rel)

    size_kb = zippath.stat().st_size // 1024
    print(f"Created: {zippath}")
    print(f"Size:    {size_kb} KB  |  {len(included)} files")


if __name__ == "__main__":
    main()
