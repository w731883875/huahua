"""宝箱误判的回归测试。

2026-11-12 的实机复盘暴露了三件事，全部固化成负例：

1. ``_detect_treasure_icon`` 只看「金色占比」，而 Lv5 的橙色菱形外框本身
   就占 0.232（一个空框都能判成宝箱），英灵 0.288 比真宝箱 0.266 还金。
2. 棋盘识别把颜色兜底排在「官方 27 图标」之前，导致认不出的金色家族
   一律被当成「玩法·宝箱」，还会把怪物格子硬翻成玩法（r3c4 怪物 Lv4 →
   玩法·宝箱 Lv5）。
3. ``attached_mid_run`` 用 ``reason == "active_scene"`` 当判据，而游戏开局
   自带 1 块初始颜料，于是每局都被标成中途接入。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from huahua_assistant import paint_icon, vision

ICON_DIR = Path(__file__).resolve().parent.parent / "runtime" / "s8_paint_icons"
FRAME_DIR = Path(__file__).resolve().parent.parent / "runtime" / "s8_frames"


def _load(path: Path) -> np.ndarray | None:
    """cv2.imread 不支持非 ASCII 路径。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3 and raw.shape[2] == 4:
        bgr = raw[:, :, :3].astype(float)
        alpha = raw[:, :, 3:4].astype(float) / 255.0
        bgr = bgr * alpha + np.array([26, 36, 48]) * (1 - alpha)
        return bgr.astype(np.uint8)
    return raw


class ColorTreasureGuardTests(unittest.TestCase):
    """颜色兜底的三道闸，全部是纯函数，不依赖任何图片。"""

    def test_lv5_frame_is_never_color_treasure(self) -> None:
        """Lv5 橙框自己就过线，这一级颜色判据没有分辨力。"""
        self.assertFalse(
            vision._color_treasure_allowed(5, vision.PaintCategory.GAMEPLAY, 0.9)
        )

    def test_monster_with_high_confidence_cannot_become_treasure(self) -> None:
        """升级不会把怪物变成宝箱，与背包面板那条路径的守卫保持一致。"""
        self.assertFalse(
            vision._color_treasure_allowed(
                4, vision.PaintCategory.MONSTER, vision.TREASURE_MONSTER_GUARD_CONFIDENCE
            )
        )

    def test_weak_monster_verdict_still_allows_color(self) -> None:
        """怪物判定本身不确定时，不能拿它当借口否掉宝箱。"""
        self.assertTrue(
            vision._color_treasure_allowed(4, vision.PaintCategory.MONSTER, 0.30)
        )

    def test_unknown_level_rejects_color_fallback(self) -> None:
        """等级读不出来就没法扣外框基线，颜色说什么都不能信。"""
        self.assertFalse(
            vision._color_treasure_allowed(None, vision.PaintCategory.GAMEPLAY, 0.9)
        )

    def test_normal_gameplay_level_allows_color(self) -> None:
        self.assertTrue(
            vision._color_treasure_allowed(4, vision.PaintCategory.GAMEPLAY, 0.6)
        )


@unittest.skipUnless(ICON_DIR.exists(), "缺少官方图标目录")
class TreasureIconThresholdTests(unittest.TestCase):
    """颜色阈值本身：真宝箱要过，金色家族不能过。"""

    def test_official_treasure_icon_is_detected(self) -> None:
        image = _load(ICON_DIR / "UI_S8Gameplay_ColorPlay1_128.png")
        self.assertIsNotNone(image)
        verdict, confidence = vision._detect_treasure_icon(image)
        self.assertTrue(verdict, f"官方宝箱图标被漏判（conf={confidence:.2f}）")

    def test_mechanical_god_is_not_treasure(self) -> None:
        """金色占比 0.133 —— 旧阈值 0.13 会把它判成宝箱。"""
        image = _load(ICON_DIR / "UI_S8Gameplay_ColorPlay4_128.png")
        self.assertIsNotNone(image)
        verdict, _ = vision._detect_treasure_icon(image)
        self.assertFalse(verdict, "机械之神被颜色判成宝箱，阈值又松回去了")

    def test_yingling_is_golden_but_must_not_be_treasure(self) -> None:
        """英灵（怪物）金色占比 0.288，比真宝箱还高——颜色单靠自己必须否掉它。"""
        image = _load(ICON_DIR / "UI_S8Gameplay_ColorMonster13_128.png")
        self.assertIsNotNone(image)
        cat, _, _, _, subtype, treasure = vision._analyze_board_paint_tile(
            image, None, icon_fallback=True
        )
        self.assertEqual(cat, vision.PaintCategory.MONSTER)
        self.assertEqual(subtype, "英灵")
        self.assertFalse(treasure)


@unittest.skipUnless(ICON_DIR.exists(), "缺少官方图标目录")
class BoardTileOrderingTests(unittest.TestCase):
    """图案优先于颜色：官方图标兜底必须先于颜色兜底执行。"""

    def setUp(self) -> None:
        # 真实链路每帧都会 reset_budget，单独调用要自己开。
        paint_icon.reset_budget(allow=True)

    def test_mechanical_god_identified_by_icon_not_color(self) -> None:
        image = _load(ICON_DIR / "UI_S8Gameplay_ColorPlay4_128.png")
        self.assertIsNotNone(image)
        cat, _, _, _, subtype, treasure = vision._analyze_board_paint_tile(
            image, None, icon_fallback=True
        )
        self.assertEqual(subtype, "机械之神")
        self.assertEqual(cat, vision.PaintCategory.GAMEPLAY)
        self.assertFalse(treasure, "机械之神被兜底成了宝箱")

    def test_official_treasure_still_identified(self) -> None:
        image = _load(ICON_DIR / "UI_S8Gameplay_ColorPlay1_128.png")
        self.assertIsNotNone(image)
        cat, _, _, _, subtype, treasure = vision._analyze_board_paint_tile(
            image, None, icon_fallback=True
        )
        self.assertEqual(subtype, "宝箱")
        self.assertEqual(cat, vision.PaintCategory.GAMEPLAY)
        self.assertTrue(treasure)


@unittest.skipUnless(FRAME_DIR.exists(), "缺少外框素材目录")
class EmptyFrameTests(unittest.TestCase):
    """没有任何图案的纯外框，绝不能被判成颜料、更不能判成宝箱。"""

    def test_empty_lv5_frame_is_not_treasure(self) -> None:
        image = _load(FRAME_DIR / "diamond_Lv5_橙.png")
        if image is None:
            self.skipTest("外框素材缺失")
        paint_icon.reset_budget(allow=True)
        _, _, level, _, _, treasure = vision._analyze_board_paint_tile(
            image, None, icon_fallback=True
        )
        self.assertFalse(treasure, f"空 Lv5 框被判成宝箱（读到的等级={level}）")

    def test_lv5_frame_gold_ratio_exceeds_threshold(self) -> None:
        """这条记录事实本身：Lv5 橙框就该过旧阈值，证明问题不是偶发。"""
        image = _load(FRAME_DIR / "diamond_Lv5_橙.png")
        if image is None:
            self.skipTest("外框素材缺失")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ratio = float(
            np.mean(
                (hsv[:, :, 0] >= 13)
                & (hsv[:, :, 0] <= 38)
                & (hsv[:, :, 1] > 70)
                & (hsv[:, :, 2] > 100)
            )
        )
        self.assertGreater(ratio, 0.20, f"Lv5 橙框金色占比只有 {ratio:.3f}，前提变了")


if __name__ == "__main__":
    unittest.main()
