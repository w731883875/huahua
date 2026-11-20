"""棋盘「整体偏一格」的回归测试。

6x6 棋盘正好 7 条网格线，所以棋盘内部任意 6x6 子区域也能凑出 7 条线、周期还完全
一致，网格分数几乎分不出来。实测在 1024x576 一档，候选框整体右偏了整整一格；
结束态那张更严重，框右偏一格后最右列整列读不出东西（一直挂着的「+(0,4) 六格认
不出」就是这个）。这里用合成网格和实拍图各钉一遍。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from huahua_assistant import vision
from huahua_assistant.capture import read_image
from huahua_assistant.models import Rect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_SCREENSHOT_ROOT = PROJECT_ROOT / "jietu"

CELL = 60
ORIGIN = 40
# 8 条线 = 7 格：这样「偏一格的框」也能凑够 7 条线，和真实棋盘内部的情况一致。
LINES = 8


def _grid_edges(size: int = 480) -> np.ndarray:
    edges = np.zeros((size, size), dtype=np.uint8)
    for index in range(LINES):
        pos = ORIGIN + index * CELL
        edges[:, pos - 1 : pos + 1] = 255
        edges[pos - 1 : pos + 1, :] = 255
    return edges


def _screenshot(name: str) -> np.ndarray | None:
    path = USER_SCREENSHOT_ROOT / name
    return read_image(path) if path.is_file() else None


class OuterEdgePenaltyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = _grid_edges()
        self.aligned = Rect(ORIGIN, ORIGIN, CELL * 6, CELL * 6)
        self.shifted = Rect(ORIGIN + CELL, ORIGIN, CELL * 6, CELL * 6)

    def test_grid_score_cannot_tell_one_cell_shift(self) -> None:
        # 这就是 bug 的根因：两个框的网格分数一样高，光看网格分永远选不对。
        aligned = vision._score_edges_grid(self.edges, self.aligned)
        shifted = vision._score_edges_grid(self.edges, self.shifted)
        self.assertAlmostEqual(aligned, shifted, places=6)
        self.assertGreater(aligned, 0.9)

    def test_penalty_keeps_aligned_board(self) -> None:
        self.assertEqual(vision._outer_edge_penalty(self.edges, self.aligned), 1.0)

    def test_penalty_punishes_one_cell_shift(self) -> None:
        self.assertLess(vision._outer_edge_penalty(self.edges, self.shifted), 0.5)

    def test_penalty_prefers_aligned_board(self) -> None:
        aligned = vision._score_edges_grid(self.edges, self.aligned) * vision._outer_edge_penalty(
            self.edges, self.aligned
        )
        shifted = vision._score_edges_grid(self.edges, self.shifted) * vision._outer_edge_penalty(
            self.edges, self.shifted
        )
        self.assertGreater(aligned, shifted)

    def test_penalty_ignores_blank_background(self) -> None:
        blank = np.zeros((480, 480), dtype=np.uint8)
        # 一条线都没有时无话可说，不能凭空扣分（否则纯色帧会被判成偏移）。
        self.assertEqual(vision._outer_edge_penalty(blank, self.aligned), 1.0)


class RealScreenshotBoardPositionTest(unittest.TestCase):
    """实拍 11 张：棋盘左边界都该落在 x≈312，结束态以前会跑到 399。"""

    def test_end_state_board_is_not_shifted(self) -> None:
        image = _screenshot("结束--地块全部无空白和入画按钮.png")
        if image is None:
            self.skipTest("缺少实拍截图")
        scene = vision.analyze_scene(image)
        self.assertIsNotNone(scene)
        self.assertLessEqual(abs(scene.board.rect.x - 312), 6, "结束态棋盘整体偏了一格")
        self.assertLessEqual(abs(scene.board.rect.w - 521), 6)

    def test_active_screens_share_board_position(self) -> None:
        names = (
            "初始和背包宝箱.png",
            "扩散后选择前.png",
            "转锋后选择前.png",
            "堆砌选择前.png",
        )
        for name in names:
            image = _screenshot(name)
            if image is None:
                self.skipTest("缺少实拍截图")
            scene = vision.analyze_scene(image)
            self.assertIsNotNone(scene, name)
            self.assertLessEqual(abs(scene.board.rect.x - 312), 6, name)
            self.assertLessEqual(abs(scene.board.rect.y - 225), 8, name)


class LayoutCandidatePoolTest(unittest.TestCase):
    def test_pool_contains_calibrated_layout(self) -> None:
        # 兜底比例在 1037x544 会整体右偏一格，标定值必须自己也在候选池里。
        image = np.zeros((878, 1673, 3), dtype=np.uint8)
        # 必须带尺寸。_screen_layout() 不带尺寸时会退回「文件里第一组」，
        # 而 runtime/layout.json 存的是**用户当前窗口**的标定（2026-11-15 是
        # 1239x780 的 0.1429/0.2705/0.5667）——拿它当 1673x878 的先验，会和
        # 候选池（走带尺寸的 _screen_layout）口径不一致，这条测试因此变红。
        expected = vision._screen_layout(1673, 878).active[0].resolve(1673, 878)
        rects = vision._estimated_board_layout_candidates(image)
        self.assertTrue(
            any(
                abs(rect.x - expected.x) <= 2 and abs(rect.y - expected.y) <= 2 and abs(rect.w - expected.w) <= 2
                for rect in rects
            ),
            rects,
        )


if __name__ == "__main__":
    unittest.main()
