#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「已确认的颜料格不会被单帧漏读打回虚空/空白」的回归测试。

2026-11-24 实机 run-20261124-221208（29 帧、同一块静止棋盘）实测到状态抖动：

    c4r5  B×9  -> P×2  -> **B×1**  -> P×17
    c3r6  B×17 -> **V×1** -> P×11
    c5r3  B×3  -> P×18 -> V×1 -> P×1 -> V×2 -> P×1 -> V×1 -> P×2

等级（Lv）本来就有 ``stable_frames`` 投票保护，但状态那一路直接放行：
一帧读不出就把颜料降成虚空/空白，模型里那格的颜料、地块加成、光环全归零重算。
"""

from __future__ import annotations

import unittest

from huahua_assistant import level_smoothing
from huahua_assistant.models import (
    BoardObservation,
    PaintCategory,
    TileObservation,
    TileState,
)


def paint(row: int = 0, col: int = 0, level: int = 4) -> TileObservation:
    return TileObservation(
        row=row,
        col=col,
        rect=(0, 0, 1, 1),
        state=TileState.PAINT,
        confidence=0.8,
        features={},
        paint_category=PaintCategory.MONSTER,
        paint_category_confidence=0.8,
        paint_level=level,
        paint_level_confidence=0.8,
    )


def blank(row: int = 0, col: int = 0) -> TileObservation:
    return TileObservation(
        row=row,
        col=col,
        rect=(0, 0, 1, 1),
        state=TileState.BLANK,
        confidence=0.8,
        features={},
    )


def void(row: int = 0, col: int = 0) -> TileObservation:
    return TileObservation(
        row=row,
        col=col,
        rect=(0, 0, 1, 1),
        state=TileState.VOID,
        confidence=0.8,
        features={},
    )


def board(tile: TileObservation) -> BoardObservation:
    return BoardObservation(rect=(0, 0, 1, 1), score=0.8, tiles=(tile,))


class DropDebounceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stabilizer = level_smoothing.LevelStabilizer()

    def _feed(self, *tiles: TileObservation) -> TileObservation:
        result = None
        for tile in tiles:
            result = self.stabilizer.observe(board(tile)).tiles[0]
        assert result is not None
        return result

    def test_single_blank_frame_keeps_paint(self) -> None:
        """P, B 这种单帧漏读，不能把颜料打回空白。"""
        result = self._feed(paint(), blank())
        self.assertIs(result.state, TileState.PAINT)
        self.assertEqual(result.paint_level, 4)

    def test_single_void_frame_keeps_paint(self) -> None:
        """P, V 同理——实机 c3r6 就掉成过一次虚空。"""
        result = self._feed(paint(), void())
        self.assertIs(result.state, TileState.PAINT)

    def test_alternating_frames_keep_paint(self) -> None:
        """实机 c5r3 那种反复跳：P,V,P,V,P 也要一直是颜料。"""
        result = self._feed(paint(), void(), paint(), void(), paint())
        self.assertIs(result.state, TileState.PAINT)

    def test_persistent_absence_eventually_drops(self) -> None:
        """真的没了（连续 stable_frames 帧读不出）就该降级，不能一直抱着不放。"""
        frames = level_smoothing.DEFAULT_STABLE_FRAMES
        tiles = [paint()] + [blank()] * frames
        result = self._feed(*tiles)
        self.assertIs(result.state, TileState.BLANK)
        self.assertIsNone(result.paint_level)

    def test_absence_counter_resets_after_paint_frame(self) -> None:
        """中间读回颜料一次，掉格计数要清零，否则零散的漏读会攒成降级。"""
        frames = level_smoothing.DEFAULT_STABLE_FRAMES
        tiles = [paint()] + [blank()] * (frames - 1) + [paint()] + [blank()] * (frames - 1)
        result = self._feed(*tiles)
        self.assertIs(result.state, TileState.PAINT)

    def test_never_paint_stays_absent(self) -> None:
        """从没确认过颜料的格子，读成空白就还是空白，不该被拉成颜料。"""
        result = self._feed(blank(), blank())
        self.assertIs(result.state, TileState.BLANK)

    def test_reset_clears_absence_state(self) -> None:
        """新一局 reset 之后不能残留上一局的掉格计数。"""
        for _ in range(level_smoothing.DEFAULT_STABLE_FRAMES):
            self.stabilizer.observe(board(void()))
        self.stabilizer.reset()
        self.assertEqual(self.stabilizer._absent_counts, {})


if __name__ == "__main__":
    unittest.main()