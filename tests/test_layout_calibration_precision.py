"""自愈落盘的标定框必须精修到格线上，否则实时路径会整片认不出颜料。

2026-11-15 22:50 实机事故：窗口 1239x780、画面上有 5 枚颜料，软件只认出 2 枚且
等级全丢。定界结果不是分类器、不是模板库、也不是「等待画面缓存」——
``runtime/layout.json`` 里那组自愈标定是 ``(181,203,439)``，而真框是
``(177,211,442)``：y 偏 8px ≈ 6x6 棋盘 11% 格高。实时路径不做 refine，
36 格裁剪整体错位；完整路径会 refine，所以一直是对的。

固件是事故当时的实机帧（冻结在 ``tests/fixtures``，不读会漂移的 runtime dump）。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from huahua_assistant import vision
from huahua_assistant.layout import ScreenLayout
from huahua_assistant.models import Rect, TileState

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "board_windowed_1239x780_paints.png"
WIDTH, HEIGHT = 1239, 780
COARSE = Rect(181, 203, 439, 439)   # 事故里被写进 layout.json 的框
GOOD = Rect(177, 211, 442, 442)     # 真框（早上实机验证过的标定）
EXPECTED = {
    (2, 3, 4, "邪影"),
    (3, 2, 4, "英灵"),
    (3, 3, 4, "赛季"),
    (3, 4, 6, "鼠人"),
    (4, 3, 4, "幽灵"),
}


def _image() -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(FIXTURE), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise unittest.SkipTest(f"读不到固件 {FIXTURE}")
    return image


def _paints(image: np.ndarray, rect: Rect) -> set[tuple[int, int, int | None, str | None]]:
    tiles = vision._classify_tiles(image, rect)
    return {
        (t.row + 1, t.col + 1, t.paint_level, t.paint_subtype)
        for t in tiles
        if t.state == TileState.PAINT
    }


class CalibrationPrecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        vision.reset_vision_caches()
        vision._RECOVER_ATTEMPTS.clear()

    def test_coarse_frame_loses_three_paints(self) -> None:
        """事故复现：这个框只认出 2 枚，等级全丢 —— 断言不是恒真的前提。"""
        image = _image()
        paints = _paints(image, COARSE)
        self.assertEqual(2, len(paints))
        self.assertTrue(all(level is None for _, _, level, _ in paints))

    def test_true_frame_recognises_all_five_paints_with_levels(self) -> None:
        self.assertEqual(EXPECTED, _paints(_image(), GOOD))

    def test_recovery_refines_the_rect_before_it_is_saved(self) -> None:
        """自愈找到粗框后必须精修：落盘的标定要保证实时路径认得出全部颜料。"""
        image = _image()
        with patch.object(vision, "_locate_board_rect", return_value=(COARSE, 0.86)):
            rect = vision._recover_board_rect(image, ScreenLayout(), WIDTH, HEIGHT, drawing=True)
        self.assertIsNotNone(rect)
        self.assertEqual(EXPECTED, _paints(image, rect))
        self.assertGreaterEqual(
            vision._score_board_rect(image, rect),
            vision._score_board_rect(image, COARSE),
        )


if __name__ == "__main__":
    unittest.main()
