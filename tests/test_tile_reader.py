"""棋盘 LvN 角标读取的回归测试。

真值来自 ``tests/fixtures/board_level_labels.json``（人工逐格核对过 11 张实机截图中的 3 张）。
这里锁定两条底线：

1. **存在性判定必须零误报**——没有角标的格子不能被当成颜料，
   否则空白地块会污染规则进度。
2. **等级读取要有下限**——数字读得不准可以返回低置信度，但不能比随机猜更差。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from huahua_assistant import tile_reader

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "board_level_labels.json"


def _imread(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _cells(entry: dict) -> dict[tuple[int, int], np.ndarray]:
    image = _imread(ROOT / entry["image"])
    x0, y0 = entry["origin"]
    cell = int(entry["cell"])
    return {
        (row, col): image[y0 + row * cell : y0 + (row + 1) * cell, x0 + col * cell : x0 + (col + 1) * cell]
        for row in range(6)
        for col in range(6)
    }


class TileReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.samples: list[tuple[str, tuple[int, int], int, np.ndarray]] = []
        cls.empty: list[tuple[str, tuple[int, int], np.ndarray]] = []
        for entry in payload["screenshots"]:
            cells = _cells(entry)
            for key, level in entry["labels"].items():
                row, col = (int(part) for part in key.split(","))
                cls.samples.append((entry["image"], (row, col), int(level), cells[(row, col)]))
            for key in entry.get("empty", []):
                row, col = (int(part) for part in key.split(","))
                cls.empty.append((entry["image"], (row, col), cells[(row, col)]))

    def test_template_bank_is_shipped(self) -> None:
        bank = tile_reader.bank()
        self.assertTrue(bank.templates, "assets/layout/level_templates.npz 缺失或为空")
        self.assertGreaterEqual(len(bank.levels()), 5)

    def test_label_window_shape_is_stable(self) -> None:
        window = tile_reader.label_window(self.samples[0][3])
        self.assertEqual(window.shape, (tile_reader.LABEL_WINDOW[1], tile_reader.LABEL_WINDOW[0]))

    def test_empty_tile_never_reports_a_label(self) -> None:
        false_positives = []
        for image, position, tile in self.empty:
            reading = tile_reader.read_label(tile)
            if reading.present:
                false_positives.append((Path(image).name, position, round(reading.density, 1)))
        self.assertEqual(false_positives, [], f"空白/不可放置地块被误判成有角标: {false_positives}")

    def test_painted_tile_always_reports_a_label(self) -> None:
        misses = []
        for image, position, level, tile in self.samples:
            reading = tile_reader.read_label(tile)
            if not reading.present:
                misses.append((Path(image).name, position, round(reading.density, 1)))
        self.assertEqual(misses, [], f"有颜料的格子没读出角标: {misses}")

    def test_level_accuracy_has_a_floor(self) -> None:
        bank = tile_reader.bank()
        correct = 0
        for _image, _position, level, tile in self.samples:
            reading = bank.read(tile)
            if reading.level == level:
                correct += 1
        accuracy = correct / len(self.samples)
        # 当前实测 ~85%（部分样本在模板库里，属于上限）；留出余量防止退化。
        self.assertGreaterEqual(accuracy, 0.70, f"等级读取准确率退化到 {accuracy:.1%}")

    def test_trusted_level_only_returns_high_confidence_reads(self) -> None:
        for _image, _position, _level, tile in self.samples:
            reading = tile_reader.read_label(tile)
            if reading.trusted_level is not None:
                self.assertGreaterEqual(reading.level_score, tile_reader.LEVEL_TRUST_SCORE)

    def test_missing_template_file_yields_empty_bank(self) -> None:
        bank = tile_reader.LevelTemplateBank.load(ROOT / "runtime" / "__no_such_templates__.npz")
        self.assertEqual(len(bank), 0)
        reading = bank.read(self.samples[0][3])
        self.assertFalse(reading.present)
        self.assertIsNone(reading.level)


if __name__ == "__main__":
    unittest.main()
