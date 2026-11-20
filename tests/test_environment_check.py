from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_environment import (
    REQUIRED_PROJECT_FILES,
    RUNTIME_PACKAGES,
    _parse_lock_lines,
    _try_import,
    _version_at_least,
    capture_probe,
    environment_report,
)


class EnvironmentCheckTests(unittest.TestCase):
    def test_minimum_version_comparison_handles_short_versions(self) -> None:
        self.assertTrue(_version_at_least("4.10", "4.10"))
        self.assertTrue(_version_at_least("12.0.1", "12.0"))
        self.assertFalse(_version_at_least("4.9.9", "4.10"))

    def test_lock_parser_ignores_comments_and_keeps_exact_versions(self) -> None:
        locked = _parse_lock_lines(
            [
                "# generated lock",
                "",
                "numpy==2.5.1",
                "opencv-python==5.0.0.93",
                "invalid-range>=1.0",
            ]
        )

        self.assertEqual(
            {"numpy": "2.5.1", "opencv-python": "5.0.0.93"},
            locked,
        )

    def test_project_check_does_not_create_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in REQUIRED_PROJECT_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            report = environment_report(root)

            self.assertTrue(all(report["project_files"].values()))
            self.assertFalse((root / "runtime").exists())


class CaptureDependencyTest(unittest.TestCase):
    """WGC 依赖必须同时出现在检查器和锁文件里。

    漏了它的后果不是"报错"，而是**静默退回截屏幕**：源码版还能凑合，打包产物
    一定退，工具窗口一盖就截到自己。所以这里锁的是"有没有被检查"，不是版本号。
    """

    def test_checker_requires_windows_capture(self) -> None:
        modules = {module for _, module, _ in RUNTIME_PACKAGES}
        self.assertIn("windows_capture", modules)
        distributions = {name for name, _, _ in RUNTIME_PACKAGES}
        self.assertIn("windows-capture", distributions)

    def test_lock_file_pins_windows_capture_and_its_deps(self) -> None:
        root = Path(__file__).resolve().parents[1]
        locked = _parse_lock_lines(
            (root / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
        )
        self.assertIn("windows-capture", locked, "照锁安装必须能装出 WGC")
        # windows-capture 自己的运行期依赖：缺了它就是"包在、导入失败"
        self.assertIn("protobuf", locked)
        self.assertIn("pycryptodome", locked)

    def test_report_separates_package_module_and_live_probe(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = environment_report(root)
        capture = report["capture"]
        self.assertIn("package_ok", capture)
        self.assertIn("module_ok", capture)
        self.assertFalse(capture["probe"]["attempted"], "默认不去摸真实窗口")
        self.assertEqual(capture["probe"]["reason"], "not_requested")

    def test_import_failure_is_reported_with_reason(self) -> None:
        ok, error = _try_import("definitely_not_a_real_module_xyz")
        self.assertFalse(ok)
        self.assertTrue(error)

    def test_live_probe_is_honest_about_missing_window(self) -> None:
        probe = capture_probe()
        self.assertTrue(probe["attempted"])
        self.assertIn("reason", probe)
        self.assertIsInstance(probe["ok"], bool)
        if not probe["ok"]:
            # 失败必须给出可读原因，不能只留一个 False
            self.assertTrue(probe["reason"])
            self.assertTrue(probe["detail"])


class BuildScriptContractTest(unittest.TestCase):
    """打包脚本的四条硬要求（静态检查，不真的跑 PyInstaller）。

    这些是"看不出来"的坑：用 PATH 上的 python、装范围版依赖、不查退出码、
    产物直接盖 dist/ ——每一条都能让 exe 悄悄退回截屏幕，而日志只有一句 Built。
    """

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.text = (root / "scripts" / "build_exe.ps1").read_text(encoding="utf-8")

    def test_uses_project_interpreter(self) -> None:
        self.assertIn(r".venv\Scripts\python.exe", self.text)
        self.assertIn("$Python", self.text)

    def test_installs_from_lock_file(self) -> None:
        self.assertIn("pip install -r requirements-lock.txt", self.text)
        # 不能真的去装范围版（注释里提旧写法不算）
        self.assertNotIn("-m pip install -r requirements.txt", self.text)

    def test_checks_every_exit_code(self) -> None:
        self.assertIn("LASTEXITCODE", self.text)
        self.assertIn("throw", self.text)

    def test_builds_into_its_own_artifact_dir(self) -> None:
        self.assertIn("build_artifacts", self.text)
        self.assertIn("--distpath", self.text)

    def test_report_records_version_and_hashes(self) -> None:
        for token in ("git rev-parse", "Get-FileHash", "build_report.json", "dirty_source"):
            with self.subTest(token=token):
                self.assertIn(token, self.text)


if __name__ == "__main__":
    unittest.main()
