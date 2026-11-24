#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「虚空上的等级光环」识别回归测试（2026-11-24 实机 c5r6）。

用户口径（原话）：

> 还有个放到虚空 c5r6 的也识别不了（需要显示出来只剩底色光圈无怪物/玩法图标）
> 放到虚空的颜料会留一圈**等级光环**（一般 3-5 级） + 中心空

也就是颜料落到虚空后**失去自身内容、退化成笔触效果**：画面上只剩底色 + 一圈
等级光环，没有怪物/玩法图标，也就没有 LvN 角标。视觉层原来直接判成纯 VOID，
那枚颜料的笔触效果就凭空消失了。

判据只有两处可用，缺一不可：
  1. 环带（半径 0.55~0.88）里落进某个**等级色相窗口**的占比要高；
  2. 环带相对**底色**（最外圈，半径 > 0.95）要有对比——纯色虚空在环带里
     当然一片同色，只靠第 1 条会直接得满分（自测 (110,90,70) 的纯色块
     曾拿到 score=1.0、level=3 的假阳性）。
  3. 中心要「空」：中心颜色与底色接近。用颜色差而不是边缘密度——图标常是
     实心大色块，内部平整、边缘密度极低（实测 0.069，会被误判成空）。
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from huahua_assistant import vision


def _flat(bgr: tuple[int, int, int], size: int = 96) -> np.ndarray:
    image = np.zeros((size, size, 3), np.uint8)
    image[:, :] = bgr
    return image


def _ring(base: tuple[int, int, int], ring_bgr: tuple[int, int, int], thickness: int = 6) -> np.ndarray:
    image = _flat(base)
    center = image.shape[0] // 2
    cv2.circle(image, (center, center), 40, ring_bgr, thickness)
    return image


class RetainedAuraDetectionTest(unittest.TestCase):
    """纯虚空必须判「没有光环」，等级光环必须连等级一起读出来。"""

    def test_plain_void_has_no_aura(self) -> None:
        for bgr in ((110, 90, 70), (80, 76, 72), (150, 146, 142)):
            score, _ring, _edge, level = vision._tile_retained_aura_score(_flat(bgr))
            self.assertEqual(score, 0.0, f'纯色 {bgr} 不该判成有光环')
            self.assertIsNone(level, f'纯色 {bgr} 不该读出等级')

    def test_level_ring_reads_level(self) -> None:
        cases = {
            3: (200, 150, 40),   # 青蓝
            4: (180, 60, 150),   # 紫
            5: (30, 120, 240),   # 橙
        }
        for expected_level, ring_bgr in cases.items():
            score, ring_ratio, _edge, level = vision._tile_retained_aura_score(_ring((95, 88, 84), ring_bgr))
            self.assertGreaterEqual(score, vision.RETAINED_AURA_RING_MIN)
            self.assertEqual(level, expected_level, f'{ring_bgr} 应读出 Lv{expected_level}')
            self.assertGreater(ring_ratio, 0.0)

    def test_solid_icon_centre_is_not_aura(self) -> None:
        """环 + 实心图标 = 有内容的颜料残影，不是「只剩底色」。"""
        image = _ring((95, 88, 84), (30, 120, 240))
        cv2.rectangle(image, (32, 32), (64, 64), (255, 255, 255), -1)
        score, _ring_ratio, _edge, level = vision._tile_retained_aura_score(image)
        self.assertLess(score, vision.RETAINED_AURA_RING_MIN)
        self.assertIsNone(level)

    def test_colored_icon_centre_is_not_aura(self) -> None:
        image = _ring((95, 88, 84), (30, 120, 240))
        center = image.shape[0] // 2
        cv2.circle(image, (center, center), 18, (60, 60, 200), -1)
        score, _ring_ratio, _edge, level = vision._tile_retained_aura_score(image)
        self.assertLess(score, vision.RETAINED_AURA_RING_MIN)
        self.assertIsNone(level)

    def test_empty_crop_is_safe(self) -> None:
        score, ring_ratio, center_edge, level = vision._tile_retained_aura_score(np.zeros((0, 0, 3), np.uint8))
        self.assertEqual(score, 0.0)
        self.assertEqual(ring_ratio, 0.0)
        self.assertEqual(center_edge, 1.0)
        self.assertIsNone(level)

    def test_aura_level_excludes_unsupported_levels(self) -> None:
        """Lv1/Lv2/Lv7 不在光环等级名单里：银、绿在虚空上不出现，Lv7 是动画彩虹环。"""
        self.assertEqual(set(vision.RETAINED_AURA_LEVELS), {3, 4, 5, 6})


class RetainedAuraModelTest(unittest.TestCase):
    """识别到的光环要能进模型、进记录，不能停在视觉层。"""

    def test_tile_carries_retained_aura(self) -> None:
        from huahua_assistant import game_model as gm

        tile = gm.Tile(state=gm.TileState.VOID, retained_aura=0.42, retained_aura_level=4)
        self.assertEqual(tile.state, gm.TileState.VOID)
        self.assertAlmostEqual(tile.retained_aura, 0.42)
        self.assertEqual(tile.retained_aura_level, 4)

    def test_compact_record_shows_retained_aura(self) -> None:
        """记录里必须能看出「这格还有光环」，否则复盘时等同于丢了。"""
        from huahua_assistant import app
        from huahua_assistant import game_model as gm

        tiles = []
        for index in range(36):
            if index == 34:
                tiles.append(gm.Tile(state=gm.TileState.VOID, retained_aura=0.42, retained_aura_level=4))
            else:
                tiles.append(gm.Tile(state=gm.TileState.BLANK))
        payload = app._compact_tile_paints(gm.PuzzleState(tiles=tuple(tiles)))
        self.assertEqual(payload[34], {'ra': 0.42, 'ral': 4})
        self.assertIsNone(payload[0], '没有光环的空地仍然记 None')

    def test_observation_has_retained_aura_fields(self) -> None:
        observation = vision.TileObservation(
            row=0,
            col=0,
            rect=vision.Rect(0, 0, 10, 10),
            state=vision.TileState.VOID,
            confidence=0.8,
            features={},
        )
        self.assertEqual(observation.retained_aura, 0.0)
        self.assertIsNone(observation.retained_aura_level)


if __name__ == '__main__':
    unittest.main()
