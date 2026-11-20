"""标定文件的合理性闸门。

背景（2026-11-19 实机事故）：本局开局后 4 秒，自愈在转场画面上把「窗口左上角」
当成了棋盘，往 ``runtime/layout.json`` 写进一组 ``1922x1030 -> (0.0, 0.0, 0.532)``。
之后整局每一帧都按那个框裁格子：真盘面 空18/阻挡6/虚空11/颜料1，
程序读到 空3/阻挡10/虚空17/颜料6。

这里锁住两件事：
1. 这种一眼假的条目**读不进来**（历史文件已经被污染也不怕）；
2. 这种框**写不出去**（不能让自愈的一次误判变成永久标定）。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from huahua_assistant.layout import (
    ACTIVE_LAYOUTS,
    calibration_is_plausible,
    load_layout,
    save_layout,
    LayoutRect,
)
from huahua_assistant.models import Rect


def _write(path: Path, calibrations: list[dict]) -> None:
    path.write_text(
        json.dumps({"canvas": [1600, 900], "calibrations": calibrations}, ensure_ascii=False),
        encoding="utf-8",
    )


class CalibrationPlausibilityTests(unittest.TestCase):
    def test_corner_rect_is_rejected(self) -> None:
        # 事故现场的那组数
        self.assertFalse(calibration_is_plausible(LayoutRect(0.0, 0.0, 0.532039)))

    def test_reasonable_rect_is_accepted(self) -> None:
        self.assertTrue(calibration_is_plausible(LayoutRect(0.187, 0.256, 0.594)))
        self.assertTrue(calibration_is_plausible(LayoutRect(0.174, 0.244, 0.607)))

    def test_extreme_size_is_rejected(self) -> None:
        self.assertFalse(calibration_is_plausible(LayoutRect(0.19, 0.25, 0.05)))
        self.assertFalse(calibration_is_plausible(LayoutRect(0.19, 0.25, 0.97)))

    def test_no_nan_gets_through(self) -> None:
        self.assertFalse(calibration_is_plausible(LayoutRect(float("nan"), 0.25, 0.6)))
        self.assertFalse(calibration_is_plausible(LayoutRect(0.19, float("inf"), 0.6)))


class PoisonedFileTests(unittest.TestCase):
    def test_poisoned_entry_is_ignored_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "layout.json"
            _write(path, [{"width": 1922, "height": 1030, "x_ratio": 0.0, "y_ratio": 0.0, "size_ratio": 0.532039}])

            layout = load_layout(1922, 1030, path=path)

            # 坏条目被丢掉 → 退回内置实测比例，而不是按窗口左上角裁格子
            self.assertEqual(layout.active[0].x_ratio, ACTIVE_LAYOUTS[0].x_ratio)
            self.assertEqual(layout.active[0].y_ratio, ACTIVE_LAYOUTS[0].y_ratio)

    def test_good_entry_next_to_poisoned_one_still_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "layout.json"
            _write(
                path,
                [
                    {"width": 1922, "height": 1030, "x_ratio": 0.0, "y_ratio": 0.0, "size_ratio": 0.532039},
                    {"width": 1922, "height": 1030, "x_ratio": 0.175, "y_ratio": 0.217, "size_ratio": 0.643},
                ],
            )

            layout = load_layout(1922, 1030, path=path)

            self.assertAlmostEqual(layout.active[0].x_ratio, 0.175)
            self.assertAlmostEqual(layout.active[0].size_ratio, 0.643)

    def test_implausible_rect_is_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "layout.json"
            _write(path, [{"width": 1522, "height": 885, "x_ratio": 0.141919, "y_ratio": 0.248588, "size_ratio": 0.60565}])
            before = path.read_text(encoding="utf-8")

            save_layout(1922, 1030, Rect(0, 0, 548, 548), path=path)

            self.assertEqual(path.read_text(encoding="utf-8"), before, "坏框不该改动标定文件")

    def test_plausible_rect_is_written_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "layout.json"
            _write(path, [{"width": 1522, "height": 885, "x_ratio": 0.141919, "y_ratio": 0.248588, "size_ratio": 0.60565}])

            save_layout(1922, 1030, Rect(336, 210, 621, 621), path=path)

            layout = load_layout(1922, 1030, path=path)
            self.assertAlmostEqual(layout.active[0].x_ratio, round(336 / 1922, 6))
            self.assertAlmostEqual(layout.active[0].y_ratio, round(210 / 1030, 6))
            self.assertAlmostEqual(layout.active[0].size_ratio, round(621 / 1030, 6))
            # 不同尺寸的标定不能被这次写入挤掉
            other = load_layout(1522, 885, path=path)
            self.assertAlmostEqual(other.active[0].x_ratio, 0.141919)


if __name__ == "__main__":
    unittest.main()
