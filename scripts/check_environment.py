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
    # Windows.Graphics.Capture 的 Python 绑定。少了它源码版可能仍能跑（退回截屏），
    # 但打包产物一定会退回截屏——所以必须在这里出现，哪怕只是"包存在"。
    ("windows-capture", "windows_capture", "2.0"),
)

REQUIRED_PROJECT_FILES = (
    ".editorconfig",
    ".gitattributes",
    "run_app.py",
    "requirements.txt",
    "requirements-lock.txt",
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


def _parse_lock_lines(lines: list[str]) -> dict[str, str]:
    locked: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        distribution, version = line.split("==", 1)
        distribution = distribution.strip()
        version = version.strip()
        if distribution and version:
            locked[distribution] = version
    return locked


def _try_import(module: str) -> tuple[bool, str | None]:
    """真的 import 一次。

    ``find_spec`` 只说明"有这个模块文件"，文件在但 DLL 缺依赖时照样导入失败——
    那正是打包后 exe 悄悄退回截屏的原因。所以两个信号分开报。
    """
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - 任何导入失败都要如实记下来
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def capture_probe() -> dict:
    """实窗采集探针：找**真实**窗口，用 WGC 读一帧。

    只读不改：不激活窗口、不改尺寸、不写任何文件。没找到窗口算"未验证"，
    不算环境坏——这时候本来就没法验证。
    """
    try:
        from huahua_assistant.capture import capture_window_content, find_capture_window
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "ok": False, "reason": "import_failed", "detail": str(exc)}
    window = find_capture_window()
    if window is None:
        return {
            "attempted": True,
            "ok": False,
            "reason": "no_window",
            "detail": "没找到可抓的游戏窗口（游戏没开会是这样，不代表环境有问题）",
        }
    image, method = capture_window_content(window)
    if image is None:
        return {"attempted": True, "ok": False, "reason": "content_failed", "detail": str(method)}
    return {
        "attempted": True,
        "ok": True,
        "reason": "ok",
        "detail": f"{method} {getattr(image, 'shape', '')}",
    }


def environment_report(project_root: Path, *, probe_capture: bool = False) -> dict:
    python_version = sys.version_info[:3]
    python_supported = SUPPORTED_PYTHON_MIN <= python_version[:2] < SUPPORTED_PYTHON_MAX_EXCLUSIVE
    packages = []
    for distribution, module, minimum in RUNTIME_PACKAGES:
        module_found = importlib.util.find_spec(module) is not None
        import_ok, import_error = _try_import(module) if module_found else (False, "module_not_found")
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
                # 三件事分开报：包存在 / 模块可导入 / 版本够
                "installed": installed_version,
                "module_found": module_found,
                "import_ok": import_ok,
                "import_error": import_error,
                "version_ok": version_ok,
                "ok": bool(module_found and import_ok and version_ok),
            }
        )

    tkinter_found = importlib.util.find_spec("tkinter") is not None
    project_files = {
        relative: (project_root / relative).is_file()
        for relative in REQUIRED_PROJECT_FILES
    }
    lock_path = project_root / "requirements-lock.txt"
    lock_entries = _parse_lock_lines(lock_path.read_text(encoding="utf-8").splitlines()) if lock_path.is_file() else {}
    locked_packages = []
    for distribution, expected in lock_entries.items():
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            installed = None
        locked_packages.append(
            {
                "distribution": distribution,
                "expected": expected,
                "installed": installed,
                "ok": installed == expected,
            }
        )
    lock_ok = bool(locked_packages) and all(item["ok"] for item in locked_packages)
    capture_package = next(
        (item for item in packages if item["module"] == "windows_capture"), None
    )
    capture = {
        "package_ok": bool(capture_package and capture_package["installed"]),
        "module_ok": bool(capture_package and capture_package["import_ok"]),
        "package_version": capture_package["installed"] if capture_package else None,
        "probe": (
            capture_probe()
            if probe_capture
            else {
                "attempted": False,
                "ok": None,
                "reason": "not_requested",
                "detail": "需要真实窗口；加 --probe-capture 才会跑",
            }
        ),
    }
    ok = (
        python_supported
        and tkinter_found
        and all(item["ok"] for item in packages)
        and all(project_files.values())
        and lock_ok
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
        "capture": capture,
        "lock": {
            "path": str(lock_path),
            "packages": locked_packages,
            "ok": lock_ok,
        },
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
            f"(required >={package['minimum']}, package={bool(package['installed'])}, "
            f"module={package['module_found']}, import={package['import_ok']}, "
            f"ok={package['ok']})"
        )
        if not package["import_ok"] and package["import_error"]:
            print(f"    import error: {package['import_error']}")
    capture = report.get("capture") or {}
    probe = capture.get("probe") or {}
    probe_state = (
        "skipped"
        if not probe.get("attempted")
        else ("ok" if probe.get("ok") else f"failed ({probe.get('reason')})")
    )
    print(
        "Capture (Windows.Graphics.Capture): "
        f"package={capture.get('package_ok')} "
        f"module={capture.get('module_ok')} "
        f"live={probe_state}"
    )
    if probe.get("detail"):
        print(f"    {probe['detail']}")
    lock = report["lock"]
    print(f"Exact lock: {lock['ok']} ({len(lock['packages'])} packages)")
    for package in lock["packages"]:
        if not package["ok"]:
            print(
                f"- lock mismatch {package['distribution']}: "
                f"installed={package['installed'] or 'missing'}, expected={package['expected']}"
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
        "--probe-capture",
        action="store_true",
        help="额外验证一次**实窗**采集（找一个真实窗口用 WGC 读一帧；没开游戏时会是 no_window）",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Return success even when runtime dependencies are missing.",
    )
    args = parser.parse_args()

    report = environment_report(args.project_root.resolve(), probe_capture=args.probe_capture)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report["ok"] or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
