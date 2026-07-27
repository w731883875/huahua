from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_INCLUDE_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".txt",
    ".ps1",
    ".spec",
}
DEFAULT_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "runtime",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check project text files for UTF-8 consistency.")
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to check.")
    parser.add_argument("--allow-bom", action="store_true", help="Allow UTF-8 BOM.")
    parser.add_argument("--allow-crlf", action="store_true", help="Allow CRLF line endings.")
    args = parser.parse_args()

    files = list(_iter_files([Path(item) for item in args.paths]))
    failed = False
    checked = 0
    for path in files:
        issues = _encoding_issues(path, allow_bom=args.allow_bom, allow_crlf=args.allow_crlf)
        checked += 1
        if issues:
            failed = True
            print(f"FAIL {path}: {', '.join(issues)}")
    if failed:
        print(f"Checked {checked} text files; encoding issues found.")
        return 1
    print(f"Checked {checked} text files; all OK.")
    return 0


def _iter_files(paths: list[Path]):
    for path in paths:
        if path.is_file():
            if _should_check(path):
                yield path
            continue
        if not path.is_dir():
            continue
        for item in path.rglob("*"):
            if item.is_dir():
                continue
            if any(part in DEFAULT_SKIP_DIRS for part in item.parts):
                continue
            if _should_check(item):
                yield item


def _should_check(path: Path) -> bool:
    return path.suffix.lower() in DEFAULT_INCLUDE_SUFFIXES or path.name in {".editorconfig", ".gitattributes"}


def _encoding_issues(path: Path, *, allow_bom: bool, allow_crlf: bool) -> list[str]:
    issues: list[str] = []
    data = path.read_bytes()
    if b"\x00" in data:
        issues.append("contains NUL byte")
    if data.startswith(b"\xef\xbb\xbf") and not allow_bom:
        issues.append("has UTF-8 BOM")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        issues.append(f"not valid UTF-8 at byte {exc.start}")
    if b"\r\n" in data and not allow_crlf:
        issues.append("uses CRLF line endings")
    return issues


if __name__ == "__main__":
    raise SystemExit(main())
