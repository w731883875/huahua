from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_environment import (
    REQUIRED_PROJECT_FILES,
    _version_at_least,
    environment_report,
)


class EnvironmentCheckTests(unittest.TestCase):
    def test_minimum_version_comparison_handles_short_versions(self) -> None:
        self.assertTrue(_version_at_least("4.10", "4.10"))
        self.assertTrue(_version_at_least("12.0.1", "12.0"))
        self.assertFalse(_version_at_least("4.9.9", "4.10"))

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


if __name__ == "__main__":
    unittest.main()
