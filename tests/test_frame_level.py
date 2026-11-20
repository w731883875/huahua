"""棋盘边框色等级读取的回归测试。

边框色是等级的第一来源：银=Lv1、绿=Lv2、青蓝=Lv3、紫=Lv4、橙=Lv5、红=Lv6。
这里锁定三条：

1. **32 个带真值的棋盘格子必须全对**（实测 100%）。
2. **淡金色生效范围地块不能被读成 Lv5**（色相 27~30 压在橙色 15~17 旁边，
   靠「低饱和」这一刀分开）。
3. **空白/冰块/漩涡地块不产生等级**，避免凭空造出颜料。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from huahua_assistant import frame_level

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


class FrameLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.screenshots: list[tuple[str, dict[tuple[int, int], np.ndarray]]] = []
        cls.labeled: list[tuple[str, tuple[int, int], int, np.ndarray]] = []
        cls.empty: list[tuple[str, tuple[int, int], np.ndarray]] = []
        for entry in payload["screenshots"]:
            cells = _cells(entry)
            cls.screenshots.append((entry["image"], cells))
            for key, level in entry["labels"].items():
                row, col = (int(part) for part in key.split(","))
                cls.labeled.append((entry["image"], (row, col), int(level), cells[(row, col)]))
            for key in entry.get("empty", []):
                row, col = (int(part) for part in key.split(","))
                cls.empty.append((entry["image"], (row, col), cells[(row, col)]))

    def test_level_reads_are_all_correct(self) -> None:
        failures = []
        for image, position, level, tile in self.labeled:
            reading = frame_level.read_frame_level(tile)
            if reading.level != level:
                failures.append(
                    (
                        Path(image).name,
                        position,
                        level,
                        reading.level,
                        round(reading.mean_saturation, 1),
                        round(reading.sat_ratio, 2),
                        reading.hue,
                    )
                )
        self.assertEqual(failures, [], f"边框色等级读错: {failures}")

    def test_level_seven_is_read_from_the_frame(self) -> None:
        """彩框是圆形彩虹环：菱形带里低饱和但非零、且没有主色相。

        历史上这里会把 Lv7 读成 Lv1（彩虹环落进银框判据），现在靠
        ``RAINBOW_SAT_RATIO_MIN`` + ``RAINBOW_HUE_SHARE_MAX`` 直接认出来。
        """

        seen = False
        for image, position, level, tile in self.labeled:
            if level != 7:
                continue
            seen = True
            reading = frame_level.read_frame_level(tile)
            self.assertEqual(7, reading.level)
            self.assertGreaterEqual(reading.sat_ratio, frame_level.RAINBOW_SAT_RATIO_MIN)
        self.assertTrue(seen, "夹具里应当有一枚 Lv7 样本")

    def test_hue_windows_do_not_overlap(self) -> None:
        windows = sorted(frame_level.HUE_WINDOWS.items(), key=lambda item: item[1][0])
        for index in range(len(windows) - 1):
            self.assertLess(
                windows[index][1][1],
                windows[index + 1][1][0],
                f"Lv{windows[index][0]} 和 Lv{windows[index + 1][0]} 的色相窗口重叠了",
            )

    def test_pale_gold_scope_highlight_is_not_level_five(self) -> None:
        """初始颜料的作用范围会给整格盖一层淡金，色相 27~30，不能读成 Lv5。

        注意 ``初始和背包宝箱.png`` 第 1 行是 ``空白×4 + 颜料 + 空白``，
        第 5 列（索引 4）是真正的 Lv3 颜料，必须排除。
        """
        target = "jietu/初始和背包宝箱.png"
        cells = next(c for name, c in self.screenshots if name == target)
        for col in (0, 1, 2, 3, 5):
            reading = frame_level.read_frame_level(cells[(1, col)])
            self.assertNotEqual(reading.level, 5, f"淡金地块 (1,{col}) 被读成了 Lv5")
            self.assertLess(
                reading.mean_saturation,
                frame_level.SILVER_MEAN_SAT_MAX,
                f"淡金地块 (1,{col}) 的带内平均饱和度 {reading.mean_saturation:.0f} 不像低饱和覆盖层",
            )

    def test_empty_tiles_do_not_become_paint_in_the_pipeline(self) -> None:
        """空白/冰块/漩涡地块没有 LvN 角标，整条链路不能把它们变成带等级的颜料。

        单看边框颜色做不到这一点（空白地块的菱形带里也会有图标/背景的彩色像素），
        所以这条约束由 ``vision._classify_tile_observation`` 里的角标仲裁负责。
        """
        from huahua_assistant import vision

        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        offenders = []
        for entry in payload["screenshots"]:
            image = _imread(ROOT / entry["image"])
            observation = vision.analyze_scene_realtime(image)
            for key in entry.get("empty", []):
                row, col = (int(part) for part in key.split(","))
                tile = observation.board.tiles[row * 6 + col]
                if tile.state.value == "paint" and tile.paint_level is not None:
                    offenders.append((Path(entry["image"]).name, (row, col), tile.paint_level))
        self.assertEqual(offenders, [], f"无颜料地块在链路里变成了颜料: {offenders}")

    def test_degenerate_input_is_safe(self) -> None:
        reading = frame_level.read_frame_level(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertIsNone(reading.level)
        self.assertEqual(reading.confidence, 0.0)
        reading = frame_level.read_frame_level(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertIsNone(reading.level)


if __name__ == "__main__":
    unittest.main()
