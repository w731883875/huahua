"""Continuous resize and stale-frame regression tests with real game fixtures."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from huahua_assistant import geometry, vision, paint_panel
from huahua_assistant.capture import CaptureResult, WindowInfo
from huahua_assistant.frame_pump import FramePump
from huahua_assistant.models import Rect, TileState

FIXTURES = Path(__file__).parent / "fixtures"


def image(name):
    return cv2.imdecode(np.fromfile(str(FIXTURES / name), dtype=np.uint8), cv2.IMREAD_COLOR)


class AdaptiveGeometryTests(unittest.TestCase):
    def test_same_tracker_changes_size_without_calibration(self):
        tracker = vision.VisionTracker()
        with patch.object(vision, "load_layout", side_effect=AssertionError("unexpected calibration")):
            for name in ("board_resized_1536x877_paints.png", "board_resized_1765x965_paints.png",
                         "board_windowed_1239x780.png", "board_resized_1536x877_paints.png"):
                with self.subTest(name=name):
                    frame = image(name)
                    scene = tracker.analyze_quick(frame)
                    self.assertIsNotNone(scene)
                    self.assertEqual(5, sum(t.state == TileState.PAINT for t in scene.board.tiles))
                    self.assertEqual(frame.shape[1::-1], (scene.layout["width"], scene.layout["height"]))
        self.assertEqual(3, tracker.layout_generation)

    def test_same_size_content_moves_and_rescales(self):
        frame = image("board_resized_1765x965_paints.png")
        tracker = vision.VisionTracker()
        first = tracker.analyze_quick(frame)
        transform = np.array([[.85, 0, 80], [0, .85, 35]], dtype=np.float32)
        moved = cv2.warpAffine(frame, transform, frame.shape[1::-1])
        second = tracker.analyze_quick(moved)
        self.assertIsNotNone(second)
        self.assertGreater(tracker.layout_generation, 0)
        self.assertAlmostEqual(second.board.rect.x, first.board.rect.x*.85+80, delta=5)
        self.assertAlmostEqual(second.board.rect.y, first.board.rect.y*.85+35, delta=5)
        self.assertAlmostEqual(second.board.rect.w, first.board.rect.w*.85, delta=5)

    def test_realtime_and_full_entries_discard_old_geometry(self):
        for method, target in (("analyze", "analyze_scene"), ("analyze_realtime", "analyze_scene_realtime")):
            tracker = vision.VisionTracker()
            tracker.frame_size = (1200, 700)
            tracker.board_rect = Rect(150, 150, 400, 400)
            tracker.last_scene = MagicMock()
            with patch.object(vision, target, return_value=None) as analyze:
                getattr(tracker, method)(np.zeros((900, 1600, 3), np.uint8))
            self.assertIsNone(analyze.call_args.kwargs["previous_board_rect"])
            self.assertIsNone(tracker.last_scene)

    def test_two_cards_follow_real_boundaries_at_several_scales(self):
        # Measured borders in the supplied screenshots, independent of layout ratios.
        labels = {
            "brush_panel_user_stage1.jpg": ((897, 313, 572, 109), (897, 435, 572, 108)),
            "brush_panel_user_stage2.jpg": ((838, 256, 571, 108), (838, 377, 571, 109)),
        }
        for name, boxes in labels.items():
            for scale in (.5, .75, 1., 1.25):
                with self.subTest(name=name, scale=scale):
                    frame = cv2.resize(image("brush_panel_layout/"+name), None, fx=scale, fy=scale)
                    grid = geometry.locate_grid(frame)
                    self.assertIsNotNone(grid)
                    detected = geometry.locate_brush_panel(frame, grid.rect)
                    self.assertIsNotNone(detected)
                    _, cards = detected
                    self.assertEqual(2, len(cards))
                    for actual, expected in zip(cards, boxes):
                        for a, b in zip((actual.x, actual.y, actual.w, actual.h), expected):
                            self.assertAlmostEqual(a, b*scale, delta=max(4, 5*scale))

    def test_non_game_pixels_do_not_form_a_grid(self):
        rng = np.random.default_rng(42)
        for frame in (np.zeros((720, 1280, 3), np.uint8), rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)):
            self.assertIsNone(geometry.locate_grid(frame))

    def test_tall_and_wide_letterboxed_windows_reacquire_the_board(self):
        original = image("board_resized_1765x965_paints.png")
        expected = geometry.locate_grid(original).rect
        tracker = vision.VisionTracker()
        for width, height in ((900, 1350), (2400, 700), (1400, 900)):
            scale = min(width / original.shape[1], height / original.shape[0])
            resized = cv2.resize(original, None, fx=scale, fy=scale)
            x, y = (width-resized.shape[1])//2, (height-resized.shape[0])//2
            frame = np.zeros((height, width, 3), np.uint8)
            frame[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
            with self.subTest(size=(width, height)), \
                 patch.object(vision, "load_layout", side_effect=AssertionError("unexpected calibration")):
                scene = tracker.analyze_quick(frame)
                self.assertIsNotNone(scene)
                self.assertAlmostEqual(scene.board.rect.x, x+expected.x*scale, delta=5)
                self.assertAlmostEqual(scene.board.rect.y, y+expected.y*scale, delta=5)
                self.assertAlmostEqual(scene.board.rect.w, expected.w*scale, delta=5)

    def test_viewport_keeps_titlebar_fixed_and_centers_content(self):
        window = WindowInfo(1, "game", 2, "game.exe", 100, 100, 1702, 1039, False,
                            client_bounds=(101, 138, 1701, 1038))
        first = CaptureResult(np.zeros((939, 1602, 3), np.uint8), "window-content:game", window)
        smaller = replace(window, right=1370, bottom=907, client_bounds=(101,138,1369,906))
        second = CaptureResult(np.zeros((807,1270,3),np.uint8), "window-content:game", smaller)
        predicted = geometry.map_rect(Rect(231,235,571,570), geometry.capture_viewport(first), geometry.capture_viewport(second))
        self.assertAlmostEqual(184, predicted.x, delta=2)
        self.assertAlmostEqual(221, predicted.y, delta=2)
        self.assertAlmostEqual(451, predicted.w, delta=2)


class PumpGeometryTests(unittest.TestCase):
    def test_changed_window_during_analysis_is_not_published_or_recorded(self):
        window = WindowInfo(1, "game", 2, "game.exe", 0, 0, 1000, 600, False)
        capture = CaptureResult(np.zeros((600,1000,3),np.uint8), "window-content:game", window)
        record = MagicMock()
        pump = FramePump(capture_func=lambda:capture, analyze_func=lambda _:MagicMock(),
                         needs_full_refresh=lambda:False, full_analyze_func=lambda _:None,
                         geometry_probe=lambda:replace(window,right=1200),record_func=record)
        result = pump._capture_and_analyze()
        self.assertIsNone(result.scene)
        self.assertIsNone(result.error)
        self.assertTrue(result.perf["geometry_pending"])
        record.assert_not_called()

    def test_dragging_defers_expensive_analysis(self):
        window = WindowInfo(1, "game", 2, "game.exe", 0, 0, 1000, 600, False)
        capture = CaptureResult(np.zeros((600,1000,3),np.uint8), "window-content:game", window)
        analyze = MagicMock(return_value=None)
        pump = FramePump(capture_func=lambda:capture, analyze_func=analyze,
                         needs_full_refresh=lambda:False, full_analyze_func=lambda _:None,
                         geometry_probe=lambda:capture.window)
        pump._capture_and_analyze()
        analyze.reset_mock()
        capture = replace(capture, image=np.zeros((650,1100,3),np.uint8),window=replace(window,right=1100,bottom=650))
        result = pump._capture_and_analyze()
        self.assertTrue(result.perf["geometry_pending"])
        analyze.assert_not_called()


class AsyncGeometryTests(unittest.TestCase):
    def test_old_paint_ocr_cannot_publish_after_resize_or_release_new_request(self):
        crop = np.zeros((40, 120, 3), np.uint8)
        paint_panel.reset_async_state()
        try:
            with patch.object(paint_panel.threading, "Thread") as thread:
                paint_panel.read_panel_async(crop)
                old_args = thread.call_args.kwargs["args"]
                paint_panel.reset_async_state()
                paint_panel.read_panel_async(crop)
                new_args = thread.call_args.kwargs["args"]
            reading = MagicMock(is_empty=False)
            with patch.object(paint_panel.windows_ocr, "recognize_chinese_lines", return_value=["text"]), \
                 patch.object(paint_panel, "parse_lines", return_value=reading):
                paint_panel._refresh_reading(*old_args)
                self.assertIsNone(paint_panel.last_reading())
                self.assertTrue(paint_panel._read_inflight)
                paint_panel._refresh_reading(*new_args)
                self.assertIs(reading, paint_panel.last_reading())
                self.assertFalse(paint_panel._read_inflight)
        finally:
            paint_panel.reset_async_state()


if __name__ == "__main__":
    unittest.main()
