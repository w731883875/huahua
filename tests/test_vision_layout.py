from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from huahua_assistant import presence, vision
from huahua_assistant.capture import read_image
from huahua_assistant.layout import (
    ACTIVE_LAYOUTS,
    ScreenLayout,
    aspect_key,
    calibrated_layout,
    load_layout,
    save_layout,
)
from huahua_assistant.models import Rect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_SCREENSHOT_ROOT = PROJECT_ROOT / "jietu"


def _screenshot(name: str) -> np.ndarray | None:
    path = USER_SCREENSHOT_ROOT / name
    if not path.is_file():
        return None
    return read_image(path)


class LayoutGeometryTest(unittest.TestCase):
    def test_board_rect_matches_measured_calibration(self) -> None:
        rect = ScreenLayout().board_rect(1673, 878)
        self.assertEqual(rect, Rect(313, 225, 522, 522))

    def test_board_rect_scales_with_short_side(self) -> None:
        small = ScreenLayout().board_rect(1673, 878)
        large = ScreenLayout().board_rect(3346, 1756)
        self.assertLessEqual(abs(large.w - small.w * 2), 1)
        self.assertLessEqual(abs(large.x - small.x * 2), 1)

    def test_cell_rects_tile_the_board_exactly(self) -> None:
        layout = ScreenLayout()
        board = layout.board_rect(1673, 878)
        cells = layout.cell_rects(board)
        self.assertEqual(len(cells), 36)
        self.assertEqual((cells[0].x, cells[0].y), (board.x, board.y))
        self.assertEqual((cells[-1].right, cells[-1].bottom), (board.right, board.bottom))
        # 相邻格子首尾相接，没有空隙也不重叠
        for row in range(6):
            for col in range(6):
                index = row * 6 + col
                if col < 5:
                    self.assertEqual(cells[index].right, cells[index + 1].x)
                if row < 5:
                    self.assertEqual(cells[index].bottom, cells[index + 6].y)

    def test_presence_rect_is_inside_board(self) -> None:
        layout = ScreenLayout()
        board = layout.board_rect(1673, 878)
        inner = layout.presence_rect(board)
        self.assertGreater(inner.x, board.x)
        self.assertLess(inner.right, board.right)
        self.assertLess(inner.w, board.w)

    def test_candidates_are_deduplicated_and_ordered(self) -> None:
        layout = ScreenLayout()
        candidates = layout.candidates(1673, 878, drawing=True)
        self.assertEqual(len(candidates), len(ACTIVE_LAYOUTS))
        self.assertEqual(candidates[0], layout.board_rect(1673, 878))
        self.assertEqual(len(set(candidates)), len(candidates))


class LayoutCalibrationTest(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layout.json"
            measured = Rect(336, 256, 636, 636)
            save_layout(1929, 1049, measured, path)
            loaded = load_layout(1929, 1049, path)
            self.assertEqual(loaded.board_rect(1929, 1049), measured)

    def test_missing_file_falls_back_to_builtin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_layout(1673, 878, Path(tmp) / "absent.json")
            self.assertEqual(loaded.board_rect(1673, 878), Rect(313, 225, 522, 522))

    def test_calibrated_layout_reproduces_measured_rect(self) -> None:
        measured = Rect(336, 256, 636, 636)
        layout = calibrated_layout(1929, 1049, measured)
        self.assertEqual(layout.board_rect(1929, 1049), measured)

    def test_aspect_key_is_stable(self) -> None:
        self.assertEqual(aspect_key(1673, 878), aspect_key(3346, 1756))


class PresenceTest(unittest.TestCase):
    def test_gradient_signature_is_unit_length(self) -> None:
        image = _screenshot("初始和背包宝箱.png")
        if image is None:
            self.skipTest("缺少实机截图")
        signature = presence.gradient_signature(image[:120, :120])
        self.assertAlmostEqual(float(np.linalg.norm(signature)), 1.0, places=4)

    def test_template_separates_board_from_flat_region(self) -> None:
        image = _screenshot("初始和背包宝箱.png")
        if image is None:
            self.skipTest("缺少实机截图")
        layout = ScreenLayout()
        height, width = image.shape[:2]
        board = layout.board_rect(width, height)
        template = presence.build_template([image], layout)
        board_score = presence.similarity(image, template, layout.presence_rect(board))
        flat = Rect(int(width * 0.02), int(height * 0.02), 200, 200)
        flat_score = presence.similarity(image, template, flat)
        self.assertGreater(board_score, flat_score + 0.1)

    def test_is_drawing_screen_uses_threshold(self) -> None:
        image = _screenshot("初始和背包宝箱.png")
        if image is None:
            self.skipTest("缺少实机截图")
        layout = ScreenLayout()
        height, width = image.shape[:2]
        board = layout.board_rect(width, height)
        template = presence.build_template([image], layout)
        ok, score = presence.is_drawing_screen(
            image, layout.presence_rect(board), template=template, threshold=0.1
        )
        self.assertTrue(ok)
        self.assertGreater(score, 0.1)


class RealtimePathTest(unittest.TestCase):
    def test_realtime_recognizes_real_screenshot_quickly(self) -> None:
        image = _screenshot("初始和背包宝箱.png")
        if image is None:
            self.skipTest("缺少实机截图")
        vision.reset_vision_caches()
        self.assertIsNotNone(vision.analyze_scene_realtime(image))

        started = time.perf_counter()
        for _ in range(3):
            scene = vision.analyze_scene_realtime(image)
        elapsed = (time.perf_counter() - started) / 3

        self.assertIsNotNone(scene)
        self.assertGreaterEqual(scene.board.score, 0.70)
        # 布局算坐标 + 网格存在性判断后，一帧应明显低于旧的 0.28s。
        self.assertLess(elapsed, 0.60)

    def test_blank_frame_is_rejected(self) -> None:
        vision.reset_vision_caches()
        blank = np.zeros((878, 1673, 3), dtype=np.uint8)
        self.assertIsNone(vision.analyze_scene_realtime(blank))

    def test_grid_presence_prefers_the_board_rect(self) -> None:
        image = _screenshot("初始和背包宝箱.png")
        if image is None:
            self.skipTest("缺少实机截图")
        layout = ScreenLayout()
        height, width = image.shape[:2]
        board = layout.board_rect(width, height)
        picked = vision._pick_board_rect(image, layout, width, height, drawing=True)
        self.assertIsNotNone(picked)
        self.assertLessEqual(abs(picked.x - board.x), 40)


if __name__ == "__main__":
    unittest.main()
