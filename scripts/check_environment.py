from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from importlib import metadata
from pathlib import Path


SUPPORTED_PYTHON_MIN = (3, 12)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 14)

RUNTIME_PACKAGES = (
    ("numpy", "numpy", "2.1"),
    ("opencv-python", "cv2", "4.10"),
    ("mss", "mss", "10.0"),
    ("pillow", "PIL", "12.0"),
    ("pyinstaller", "PyInstaller", "6.11"),
)

REQUIRED_PROJECT_FILES = (
    "run_app.py",
    "requirements.txt",
    "huahua_assistant/app.py",
    "huahua_assistant/game_model.py",
    "huahua_assistant/vision.py",
)


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = [int(part) for part in re.findall(r"\d+", value)]
    return tuple(numbers[:4])


def _version_at_least(value: str, minimum: str) -> bool:
    current = _version_tuple(value)
    required = _version_tuple(minimum)
    width = max(len(current), len(required))
    return current + (0,) * (width - len(current)) >= required + (0,) * (width - len(required))


def environment_report(project_root: Path) -> dict:
    python_version = sys.version_info[:3]
    python_supported = SUPPORTED_PYTHON_MIN <= python_version[:2] < SUPPORTED_PYTHON_MAX_EXCLUSIVE
    packages = []
    for distribution, module, minimum in RUNTIME_PACKAGES:
        module_found = importlib.util.find_spec(module) is not None
        try:
            installed_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            installed_version = None
        version_ok = bool(installed_version and _version_at_least(installed_version, minimum))
        packages.append(
            {
                "distribution": distribution,
                "module": module,
                "minimum": minimum,
                "installed": installed_version,
                "module_found": module_found,
                "ok": bool(module_found and version_ok),
            }
        )

    tkinter_found = importlib.util.find_spec("tkinter") is not None
    project_files = {
        relative: (project_root / relative).is_file()
        for relative in REQUIRED_PROJECT_FILES
    }
    ok = (
        python_supported
        and tkinter_found
        and all(item["ok"] for item in packages)
        and all(project_files.values())
    )
    return {
        "ok": ok,
        "python": {
            "executable": sys.executable,
            "version": ".".join(str(part) for part in python_version),
            "supported": python_supported,
            "supported_range": ">=3.12,<3.14",
        },
        "tkinter_found": tkinter_found,
        "packages": packages,
        "project_files": project_files,
    }


def _print_text(report: dict) -> None:
    python = report["python"]
    print(f"Python: {python['version']} ({python['executable']})")
    print(f"Python supported: {python['supported']} ({python['supported_range']})")
    print(f"tkinter found: {report['tkinter_found']}")
    print("Packages:")
    for package in report["packages"]:
        installed = package["installed"] or "missing"
        print(
            f"- {package['distribution']}: {installed} "
            f"(required >={package['minimum']}, module={package['module_found']}, ok={package['ok']})"
        )
    missing_files = [path for path, exists in report["project_files"].items() if not exists]
    print(f"Project files: {'ok' if not missing_files else 'missing ' + ', '.join(missing_files)}")
    print(f"Environment ready: {report['ok']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the Huahua development/runtime environment.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root to validate.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON report.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Return success even when runtime dependencies are missing.",
    )
    args = parser.parse_args()

    report = environment_report(args.project_root.resolve())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report["ok"] or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
