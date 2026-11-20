"""Lv7 宝箱的等级读取。

棋盘上最值钱的一枚就是 Lv7，而它同时踩中两个坑：

1. 模板库里原本只有 Lv1~Lv6，``Lv7`` 角标只能匹配到最像的 ``Lv1``，
   得分 0.38~0.46，读出来是「1 级宝箱」。
2. 它的边框是**白色细线菱形 + 一圈彩色辉光**（不像 Lv1~Lv6 的彩色菱形描边），
   整圈落在菱形带内侧，带内平均饱和度 81、``sat_ratio`` 0.22，正好落进「银框 = Lv1」
   的判据里；彩虹转起来时主色相偏橙，又会判成 Lv5。

修法四件：往模板库补一枚 Lv7 角标样本；``frame_level`` 先用**几何**
（菱形外圈几乎没有亮饱和像素）认 Lv7，再用「带内低饱和但非零 + 没有主色相」兜一层；
角标也读出 7 时改采信角标的分数；``run_recorder`` 只做双向去抖，不再写死「只升不降」。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from huahua_assistant import frame_level, tile_reader
from huahua_assistant.vision import _detect_board_tile_level

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "board_lv7_chest.png"
CELL = 103
#: 夹具截图里几枚已知等级的格子，用来确认修 Lv7 没有把别人带跑。
#: （5,3 那格角标能量密度 60，刚好压在 PRESENCE_DENSITY_MIN=64 下面，
#:  裁剪挪 1px 就会掉出阈值，不适合当稳定样本。）
KNOWN = {
    (1, 0): 3,
    (1, 3): 4,
    (3, 2): 4,
    (4, 0): 4,
    (4, 2): 3,
    (4, 5): 3,
}


def _imread(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _tile(row: int, col: int) -> np.ndarray:
    image = _imread(FIXTURE)
    return image[row * CELL : (row + 1) * CELL, col * CELL : (col + 1) * CELL]


class LevelSevenTests(unittest.TestCase):
    def test_template_bank_covers_level_seven(self) -> None:
        self.assertIn(7, tile_reader.bank().levels())

    def test_chest_label_reads_as_seven(self) -> None:
        reading = tile_reader.read_label(_tile(4, 3))
        self.assertTrue(reading.present)
        self.assertEqual(7, reading.level)
        self.assertGreaterEqual(reading.level_score, tile_reader.LEVEL_TRUST_SCORE)

    def test_chest_frame_reads_as_seven(self) -> None:
        """彩虹环是圆形的，整圈落在菱形带内侧，只漏一点饱和且没有主色相。"""

        frame = frame_level.read_frame_level(_tile(4, 3))
        self.assertEqual(7, frame.level)
        # 两个边界都必须满足，否则会滑回「银框 = Lv1」或误伤 Lv1 的图案漏色。
        self.assertGreaterEqual(frame.sat_ratio, frame_level.RAINBOW_SAT_RATIO_MIN)
        self.assertLess(frame.sat_ratio, frame_level.SILVER_SAT_RATIO_MAX)
        self.assertLess(frame.sat_ratio, 0.30)
        stats = frame_level.frame_stats(_tile(4, 3))
        self.assertLess(stats[3], frame_level.RAINBOW_HUE_SHARE_MAX)

    def test_chest_level_is_seven_despite_the_frame(self) -> None:
        level, confidence = _detect_board_tile_level(_tile(4, 3))
        self.assertEqual(7, level)
        self.assertGreater(confidence, 0.9)

    def test_other_levels_are_untouched(self) -> None:
        for position, expected in KNOWN.items():
            with self.subTest(position=position):
                level, _confidence = _detect_board_tile_level(_tile(*position))
                self.assertEqual(expected, level)

    def test_a_real_level_one_stays_level_one(self) -> None:
        """银框翻案的边界：真 Lv1 的 Lv1 分是 1.00，翻案余量必须是负的。"""

        image = _imread(ROOT / "jietu" / "结束--地块全部无空白和入画按钮.png")
        tile = image[222 : 222 + 87, 313 + 2 * 87 : 313 + 3 * 87]  # (0, 2) 真值 Lv1
        reading = tile_reader.read_label(tile)
        self.assertEqual(1, reading.level)
        # 真 Lv1：Lv1 的分数要明显高于其它等级，翻案的余量才是负的。
        best_other = max(value for level, value in reading.scores.items() if level != 1)
        self.assertGreater(reading.scores[1] - best_other, 0.10)
        level, _confidence = _detect_board_tile_level(tile)
        self.assertEqual(1, level)


    def test_a_clear_seven_label_beats_an_orange_frame(self) -> None:
        """角标清晰读出 7 时，边框就算判成 Lv5（橙）也必须让位。

        2026-11-18 实测：r2c3 黑帆真实 Lv7，彩虹环转起来时边框读成 Lv5/Lv1，
        只有一帧读到 7。角标是像素级模板匹配，7 的分数碰到 0.95 就是硬证据。
        这里拿一枚**真 Lv5**（橙框）的格子，伪造「角标读出 7」来锁这条优先级。
        """

        image = _imread(ROOT / "jietu" / "结束--地块全部无空白和入画按钮.png")
        tile = image[222 : 222 + 87, 313 + 0 * 87 : 313 + 1 * 87]  # (0, 0) 真值 Lv5
        self.assertEqual(5, frame_level.read_frame_level(tile).level)
        label = tile_reader.LabelReading(
            level=7,
            level_score=1.0,
            presence_score=1.0,
            density=90.0,
            present=True,
            scores={5: 0.41, 7: 1.0},
        )
        level, confidence = _detect_board_tile_level(tile, label=label)
        self.assertEqual(7, level)
        self.assertGreaterEqual(confidence, tile_reader.LEVEL_TRUST_SCORE)

    def test_a_weak_seven_label_does_not_beat_the_frame(self) -> None:
        """负例：角标只有 0.90（低于信任线）时不许翻案，仍用边框的 Lv5。"""

        image = _imread(ROOT / "jietu" / "结束--地块全部无空白和入画按钮.png")
        tile = image[222 : 222 + 87, 313 + 0 * 87 : 313 + 1 * 87]
        label = tile_reader.LabelReading(
            level=7,
            level_score=0.90,
            presence_score=0.95,
            density=90.0,
            present=True,
            scores={5: 0.41, 7: 0.90},
        )
        level, _confidence = _detect_board_tile_level(tile, label=label)
        self.assertEqual(5, level)

    def test_the_geometric_rule_is_what_finds_seven(self) -> None:
        """Lv7 的菱形描边是**白色细线**，外圈几乎没有亮且饱和的像素。

        彩虹环会转：主色相偏橙就判 Lv5、饱和掉下去就判银框 Lv1。**形状不会变**，
        所以现在第一判据是几何（``diamond_lit_ratio``），颜色只做第二道。
        """

        frame = frame_level.read_frame_level(_tile(4, 3))
        self.assertEqual(7, frame.level)
        self.assertLess(frame.diamond_lit_ratio, frame_level.DIAMOND_LIT_RATIO_MIN)
        self.assertGreaterEqual(frame.sat_ratio, frame_level.RING_SAT_RATIO_MIN)

    def test_a_tile_clipped_by_the_panel_edge_is_not_seven(self) -> None:
        """负例：底排被面板下沿裁掉菱形边框的格子，dlit 0.356/0.378，不能当 Lv7。"""

        cases = [
            (ROOT / "jietu" / "结束--地块全部无空白和入画按钮.png", 5, 4),
            (ROOT / "jietu" / "斜对角.png", 5, 4),
        ]
        for path, row, col in cases:
            with self.subTest(path=path.name, row=row, col=col):
                image = _imread(path)
                tile = image[222 + row * 87 : 222 + (row + 1) * 87, 313 + col * 87 : 313 + (col + 1) * 87]
                frame = frame_level.read_frame_level(tile)
                self.assertNotEqual(7, frame.level)
                level, _confidence = _detect_board_tile_level(tile)
                self.assertNotEqual(7, level)


if __name__ == "__main__":
    unittest.main()
