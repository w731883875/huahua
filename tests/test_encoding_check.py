from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_encoding import _iter_files


class EncodingCheckTests(unittest.TestCase):
    def test_virtual_environment_and_test_caches_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_file = root / "project.py"
            dependency_file = root / ".venv" / "Lib" / "dependency.py"
            cache_file = root / ".pytest_cache" / "cached.txt"
            dependency_file.parent.mkdir(parents=True)
            cache_file.parent.mkdir(parents=True)
            project_file.write_text("value = 1\n", encoding="utf-8")
            dependency_file.write_bytes(b"value = 2\r\n")
            cache_file.write_bytes(b"cached\r\n")

            files = set(_iter_files([root]))

            self.assertIn(project_file, files)
            self.assertNotIn(dependency_file, files)
            self.assertNotIn(cache_file, files)


if __name__ == "__main__":
    unittest.main()
