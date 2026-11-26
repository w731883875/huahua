"""Current paint must follow the visible banner across rerolls and resizes."""
import io
import os
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from huahua_assistant import brush_panel, geometry, paint_panel, vision, windows_ocr
from huahua_assistant.models import BoardObservation, PaintCategory, PaintDirection, PaintObject
from tests.test_anchor_tile_sync import _board, _panel

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/current_paint_war_god.png"
LINES = (
    "玩法颜料．征战之神LV3",
    "调色目标：在横向地块上放置玩法颜料（0/3）",
    "调色效果：使横向的玩法颜料提升1级",
)


def banner(image):
    grid = geometry.locate_grid(image)
    assert grid is not None
    board = BoardObservation(rect=grid.rect, tiles=(), score=.9)
    regions = tuple(vision._estimate_regions_fast(image, board))
    return regions, vision._current_paint_text_crop(image, regions[0].rect)


class InitialPaintTests(unittest.TestCase):
    def tearDown(self):
        paint_panel.reset_async_state()

    def test_header_punctuation_variants_and_partial_title(self):
        for separator in ("．", ".", "·", "-", "・", "", ":"):
            with self.subTest(separator=separator):
                reading = paint_panel.parse_lines([f"玩法颜料{separator}征战之神LV3"])
                overrides = vision._panel_text_overrides(reading)
                self.assertEqual(overrides["category"], PaintCategory.GAMEPLAY)
                self.assertEqual(overrides["subtype"], "征战之神")
                self.assertEqual(overrides["level"], 3)
                self.assertNotIn("direction", overrides)

    def test_cold_frame_sees_outer_diamond_before_ocr_arrives(self):
        original = cv2.imread(str(FIXTURE))
        for scale in (.65, 1.0, 1.3):
            with self.subTest(scale=scale), patch.object(paint_panel, "read_panel_async", return_value=None):
                image = cv2.resize(original, None, fx=scale, fy=scale)
                regions, _ = banner(image)
                current = vision._analyze_current_paint_fast(image, regions)
                self.assertEqual(current.category, PaintCategory.GAMEPLAY)
                self.assertEqual(current.level, 3)

    def test_old_board_marker_and_scope_cannot_override_panel_text(self):
        current = replace(_panel(category=PaintCategory.GAMEPLAY, subtype="征战之神"),
                          direction=PaintDirection.HORIZONTAL,
                          keyword_scores={"panel_text:identity": 1., "panel_text:direction": 1.})
        tiles = _board(subtype="黑帮", level=6, category=PaintCategory.MONSTER, anchor=True).tiles
        got = vision._correct_current_paint_from_board_anchor(np.zeros((600, 600, 3), np.uint8), current, tiles)
        self.assertIs(got, current)
        highlighted = tuple(replace(tile, is_scope_highlight=tile.col == 2,
                                    scope_highlight_confidence=.98) for tile in tiles)
        self.assertIs(vision._correct_current_paint_from_scope(current, highlighted), current)

    def test_reroll_discards_cached_and_inflight_old_panel(self):
        image = cv2.imread(str(FIXTURE))
        _, crop = banner(image)
        different = np.full_like(crop, 210)
        cv2.putText(different, "NEW PAINT", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 20), 4)
        with patch.object(paint_panel.threading, "Thread") as thread:
            self.assertIsNone(paint_panel.read_panel_async(crop))
            old_args = thread.call_args.kwargs["args"]
        with patch.object(windows_ocr, "recognize_chinese_lines", return_value=LINES):
            paint_panel._refresh_reading(*old_args)
        self.assertEqual(paint_panel.last_reading().subtype, "征战之神")
        with patch.object(paint_panel.threading, "Thread") as thread:
            self.assertIsNone(paint_panel.read_panel_async(different))
            new_args = thread.call_args.kwargs["args"]
        with patch.object(windows_ocr, "recognize_chinese_lines", return_value=LINES):
            paint_panel._refresh_reading(*old_args)
        self.assertIsNone(paint_panel.last_reading())
        self.assertTrue(paint_panel._read_inflight)
        self.assertGreater(new_args[1], old_args[1])

    def test_expired_result_is_not_reused_after_ocr_failures(self):
        _, crop = banner(cv2.imread(str(FIXTURE)))
        with patch.object(paint_panel.threading, "Thread") as thread:
            paint_panel.read_panel_async(crop)
            args = thread.call_args.kwargs["args"]
        with patch.object(windows_ocr, "recognize_chinese_lines", return_value=LINES):
            paint_panel._refresh_reading(*args)
        with patch.object(paint_panel.time, "monotonic", return_value=time.monotonic()+10), \
             patch.object(paint_panel.threading, "Thread"):
            self.assertIsNone(paint_panel.read_panel_async(crop))

    def test_banner_animation_keeps_cache_but_different_paint_does_not(self):
        def signature(name):
            im = cv2.imdecode(np.fromfile(str(ROOT / "jietu" / name), np.uint8), 1)
            return paint_panel._panel_signature(banner(im)[1])
        same_before = signature("堆砌选择前.png")
        same_after = signature("转锋后选择前.png")
        different = signature("堆砌后选择前.png")
        self.assertTrue(paint_panel._same_panel(same_before, same_after))
        self.assertFalse(paint_panel._same_panel(same_before, different))

    def test_brush_text_change_cancels_old_inflight_result(self):
        brush_panel.reset_cache()
        first = np.full((180, 400, 3), 70, np.uint8)
        second = first.copy()
        cv2.putText(first, "AAAA AAAA", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, .8, (235, 235, 235), 2)
        cv2.putText(second, "XXXX XXXX", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, .8, (235, 235, 235), 2)
        # Average brightness is virtually identical, as with real card pages.
        self.assertLess(np.mean(abs(brush_panel.panel_signature_vector(first)-brush_panel.panel_signature_vector(second))), 14)
        try:
            with patch.object(brush_panel.threading, "Thread") as thread:
                brush_panel.read_brush_panel_async(first)
                old_args = thread.call_args.kwargs["args"]
                brush_panel.read_brush_panel_async(second)
                new_args = thread.call_args.kwargs["args"]
            self.assertGreater(new_args[2], old_args[2])
            reading = brush_panel.BrushPanelReading(stage="effect", affixes=("old",), confidence=.9)
            with patch.object(brush_panel, "read_brush_panel", return_value=reading):
                brush_panel._async_read_worker(*old_args)
            self.assertIsNone(brush_panel._ASYNC_STATE["reading"])
            self.assertTrue(brush_panel._ASYNC_STATE["pending"])
        finally:
            brush_panel.reset_cache()


@unittest.skipUnless(os.name == "nt", "Windows OCR integration")
class WindowsOcrTests(unittest.TestCase):
    def tearDown(self):
        windows_ocr.reset_cache()
        paint_panel.reset_async_state()

    def test_failed_request_is_retried_for_identical_pixels(self):
        image = np.full((40, 200, 3), 127, np.uint8)
        worker = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO())
        for response in ('__ERROR__"file open failed"', '', '__RESULT__null', '__RESULT__bad-json'):
            with self.subTest(response=response):
                windows_ocr.reset_cache()
                with patch.object(windows_ocr, "_ensure_ocr_process_locked", return_value=worker), \
                     patch.object(windows_ocr, "_readline_with_timeout", side_effect=[response, '__RESULT__["recovered"]']), \
                     patch.object(windows_ocr._logger, "warning"):
                    self.assertEqual(windows_ocr.recognize_chinese_lines(image), ())
                    self.assertIsNotNone(windows_ocr.last_error())
                    self.assertEqual(windows_ocr.recognize_chinese_lines(image), ("recovered",))
                    self.assertIsNone(windows_ocr.last_error())

    def test_actual_ocr_and_same_tracker_follow_resizes(self):
        original = cv2.imread(str(FIXTURE))
        tracker = vision.VisionTracker()
        for scale, padding in ((1., 0), (.65, 45), (1.3, 0), (.8, 20)):
            image = cv2.resize(original, None, fx=scale, fy=scale)
            if padding:
                image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT)
            with self.subTest(scale=scale, padding=padding):
                deadline = time.monotonic()+10
                actual = None
                while time.monotonic() < deadline:
                    scene = tracker.analyze_realtime(image)
                    self.assertIsNotNone(scene)
                    current = scene.current_paint
                    actual = (current.category, current.subtype, current.level, current.target_object,
                              current.direction, current.match_current, current.match_required, current.upgrade_amount)
                    if actual == (PaintCategory.GAMEPLAY, "征战之神", 3, PaintObject.GAMEPLAY,
                                  PaintDirection.HORIZONTAL, 0, 3, 1):
                        break
                    time.sleep(.1)
                self.assertEqual(actual, (PaintCategory.GAMEPLAY, "征战之神", 3, PaintObject.GAMEPLAY,
                                         PaintDirection.HORIZONTAL, 0, 3, 1))


if __name__ == "__main__":
    unittest.main()
