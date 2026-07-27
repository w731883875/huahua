from __future__ import annotations

import unittest
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from huahua_assistant.capture import read_image
from huahua_assistant.models import Rect, RegionKind, RegionObservation, TileState
from huahua_assistant.vision import (
    VisionTracker,
    _analyze_brush_options,
    _classify_backpack_frame_shape,
    _classify_tile_for_board_score,
    _find_finish_drawing_signal,
    analyze_scene,
    analyze_scene_quick,
    analyze_scene_realtime,
    classify_tile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures"
USER_SCREENSHOT_ROOT = PROJECT_ROOT / "jietu"


def _fixture(name: str) -> np.ndarray:
    image = cv2.imread(str(PROJECT_ROOT / "runtime" / name))
    if image is None:
        raise AssertionError(f"missing tracked vision fixture: {name}")
    return image


def _user_screenshot(name: str) -> np.ndarray:
    path = USER_SCREENSHOT_ROOT / name
    if not path.is_file():
        raise AssertionError(f"missing tracked user screenshot: {name}")
    return read_image(path)


def _test_fixture(name: str) -> np.ndarray:
    path = TEST_FIXTURE_ROOT / name
    if not path.is_file():
        raise AssertionError(f"missing tracked test fixture: {name}")
    return read_image(path)


def _synthetic_start_drawing_scene() -> np.ndarray:
    source = _user_screenshot("初始和背包宝箱.png")
    height, width = source.shape[:2]
    side = int(round(min(width, height) * 0.594))
    active = Rect(int(round(width * 0.187)), int(round(height * 0.256)), side, side)
    centered = Rect(int(round(width * 0.354)), int(round(height * 0.256)), side, side)
    image = np.zeros_like(source)
    image[:, :] = (34, 29, 18)
    image[centered.y : centered.bottom, centered.x : centered.right] = source[
        active.y : active.bottom,
        active.x : active.right,
    ]

    button = Rect(
        int(round(width * 0.72)),
        int(round(height * 0.86)),
        int(round(width * 0.15)),
        int(round(height * 0.10)),
    )
    cv2.rectangle(image, (button.x, button.y), (button.right - 1, button.bottom - 1), (230, 210, 120), -1)
    cv2.rectangle(
        image,
        (button.x + 20, button.y + 20),
        (button.right - 20, button.bottom - 20),
        (20, 120, 240),
        -1,
    )
    return image


class OpenCvCompatibilityTests(unittest.TestCase):
    def test_hough_line_shapes_from_opencv_4_and_5_are_equivalent(self) -> None:
        crop = np.zeros((80, 80, 3), dtype=np.uint8)
        flat = np.array([[0, 0, 40, 40], [5, 40, 45, 0]], dtype=np.int32)
        nested = flat.reshape(-1, 1, 4)

        with patch("huahua_assistant.vision.cv2.HoughLinesP", return_value=flat):
            flat_result = _classify_backpack_frame_shape(crop)
        with patch("huahua_assistant.vision.cv2.HoughLinesP", return_value=nested):
            nested_result = _classify_backpack_frame_shape(crop)

        self.assertEqual(nested_result, flat_result)

    def test_one_pixel_board_candidate_crop_is_safe(self) -> None:
        crop = np.zeros((1, 1, 3), dtype=np.uint8)

        scoring_result = _classify_tile_for_board_score(crop)
        detailed_result = classify_tile(crop)

        self.assertIsInstance(scoring_result, TileState)
        self.assertIsInstance(detailed_result[0], TileState)

    def test_partial_panels_and_invalid_channels_are_not_scenes(self) -> None:
        partial_panel = np.zeros((200, 900, 3), dtype=np.uint8)
        grayscale = np.zeros((600, 900), dtype=np.uint8)

        self.assertIsNone(analyze_scene(partial_panel))
        self.assertIsNone(analyze_scene_quick(partial_panel))
        self.assertIsNone(analyze_scene(grayscale))

    def test_low_scoring_full_frame_is_rejected(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)

        with (
            patch("huahua_assistant.vision._best_estimated_board_layout", return_value=None),
            patch("huahua_assistant.vision._candidate_board_rects", return_value=[Rect(100, 100, 400, 400)]),
            patch("huahua_assistant.vision._score_board_rect", return_value=0.69),
        ):
            self.assertIsNone(analyze_scene(image))

    def test_stage1_cards_do_not_emit_false_stage2_affixes(self) -> None:
        image = _fixture("brush_options_panel.png")
        height, width = image.shape[:2]
        region = RegionObservation(
            kind=RegionKind.BRUSH_OPTIONS,
            rect=Rect(0, 0, width, height),
            confidence=1.0,
            label="fixture",
        )

        observation = _analyze_brush_options(image, (region,))

        self.assertIsNotNone(observation)
        self.assertEqual(("transform", "spread"), observation.stage1_options)
        self.assertEqual("spread", observation.chosen_stage1)
        self.assertEqual((), observation.stage2_options)

    def test_tracked_full_frames_reject_desktop_and_keep_game_states(self) -> None:
        desktop_image = _test_fixture("non_game_desktop.png")
        desktop = analyze_scene(desktop_image)
        realtime_desktop = analyze_scene_realtime(desktop_image)
        active_game = analyze_scene(_fixture("capture_perf_write.png"))
        terminal_game_image = _fixture("capture_auto.png")
        terminal_game = analyze_scene(terminal_game_image)

        self.assertIsNone(desktop)
        self.assertIsNone(realtime_desktop)
        self.assertIsNotNone(active_game)
        self.assertGreaterEqual(active_game.board.score, 0.70)
        self.assertEqual(36, len(active_game.board.tiles))
        self.assertIsNotNone(terminal_game)
        self.assertIsNotNone(_find_finish_drawing_signal(terminal_game_image))

    def test_user_screenshots_cover_real_current_paint_directions(self) -> None:
        cases = {
            "初始和背包宝箱.png": "horizontal",
            "其他.png": "adjacent",
            "纵向.png": "vertical",
            "斜对角.png": "diagonal",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                scene = analyze_scene(_user_screenshot(filename))
                self.assertIsNotNone(scene)
                self.assertIsNotNone(scene.current_paint)
                self.assertEqual(expected, scene.current_paint.direction.value)

    def test_user_stage1_screenshots_do_not_leak_stage2_affixes(self) -> None:
        cases = {
            "堆砌选择前.png": {"transform", "stack"},
            "斜对角.png": {"stack", "spread"},
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                scene = analyze_scene(_user_screenshot(filename))
                self.assertIsNotNone(scene)
                self.assertIsNotNone(scene.brush_options)
                self.assertEqual(expected, set(scene.brush_options.stage1_options))
                self.assertEqual((), scene.brush_options.stage2_options)

    def test_user_stage2_screenshots_keep_selected_brush_family(self) -> None:
        cases = {
            "转锋后选择前.png": "transform",
            "堆砌后选择前.png": "stack",
            "扩散后选择前.png": "spread",
            "重新获取随机调色颜料之前.png": "transform",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                scene = analyze_scene(_user_screenshot(filename))
                self.assertIsNotNone(scene)
                self.assertIsNotNone(scene.brush_options)
                self.assertEqual(expected, scene.brush_options.chosen_stage1)
                self.assertEqual(2, len(scene.brush_options.stage2_options))

    def test_realtime_stage2_options_match_visible_cards(self) -> None:
        cases = {
            "转锋后选择前.png": (
                "随机改变自身的内容，等级不变",
                "当前颜料等级随机改变",
            ),
            "堆砌后选择前.png": (
                "自身提升2级",
                "范围内随机1个颜料提升1级",
            ),
            "扩散后选择前.png": (
                "将距离最近的1个阻挡地块转变为空白地块",
                "当前颜料可放置在阻挡地块",
            ),
            "重新获取随机调色颜料之前.png": (
                "作用方向改为横向",
                "重新获得一个随机初始颜料",
            ),
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                scene = analyze_scene_realtime(_user_screenshot(filename))
                self.assertIsNotNone(scene)
                self.assertIsNotNone(scene.brush_options)
                self.assertEqual(expected, scene.brush_options.stage2_options)

    def test_realtime_stage1_cards_do_not_emit_stage2_templates(self) -> None:
        cases = {
            "其他.png": ("transform", "stack"),
            "堆砌选择前.png": ("transform", "stack"),
            "斜对角.png": ("stack", "spread"),
            "纵向.png": ("transform", "stack"),
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                scene = analyze_scene_realtime(_user_screenshot(filename))
                self.assertIsNotNone(scene)
                self.assertIsNotNone(scene.brush_options)
                self.assertEqual(expected, scene.brush_options.stage1_options)
                self.assertEqual((), scene.brush_options.stage2_options)

    def test_realtime_analysis_meets_interactive_latency_budget(self) -> None:
        images = [_user_screenshot(path.name) for path in sorted(USER_SCREENSHOT_ROOT.glob("*.png"))]
        self.assertEqual(11, len(images))
        for image in images:
            tracker = VisionTracker()
            started = time.perf_counter()
            scene = tracker.analyze_realtime(image)
            initial_elapsed = time.perf_counter() - started
            self.assertIsNotNone(scene)
            self.assertLess(initial_elapsed, 0.75)

            started = time.perf_counter()
            scene = tracker.analyze_quick(image)
            update_elapsed = time.perf_counter() - started
            self.assertIsNotNone(scene)
            self.assertLess(update_elapsed, 0.45)

    def test_gold_scope_cells_stay_blank_and_anchor_marker_is_explicit(self) -> None:
        scene = analyze_scene_realtime(_user_screenshot("初始和背包宝箱.png"))

        self.assertIsNotNone(scene)
        scope = [(tile.row, tile.col) for tile in scene.board.tiles if tile.is_scope_highlight]
        anchors = [(tile.row, tile.col) for tile in scene.board.tiles if tile.is_current_anchor]
        self.assertEqual([(1, 0), (1, 1), (1, 2), (1, 3), (1, 5)], scope)
        self.assertEqual([(1, 4)], anchors)
        self.assertTrue(
            all(scene.board.tile(row, col).state == TileState.BLANK for row, col in scope)
        )

    def test_centered_start_scene_transitions_to_active_layout_within_budget(self) -> None:
        tracker = VisionTracker()
        waiting = _synthetic_start_drawing_scene()

        started = time.perf_counter()
        waiting_scene = tracker.analyze_realtime(waiting)
        initial_elapsed = time.perf_counter() - started

        self.assertIsNotNone(waiting_scene)
        self.assertIsNotNone(waiting_scene.region(RegionKind.START_DRAWING))
        self.assertIsNone(waiting_scene.current_paint)
        self.assertIsNone(waiting_scene.backpack)
        self.assertIsNone(waiting_scene.brush_options)
        self.assertGreaterEqual(waiting_scene.board.score, 0.75)
        self.assertGreater(waiting_scene.board.rect.x, waiting.shape[1] * 0.30)
        self.assertLess(initial_elapsed, 1.0)

        started = time.perf_counter()
        active_scene = tracker.analyze_quick(_user_screenshot("初始和背包宝箱.png"))
        transition_elapsed = time.perf_counter() - started

        self.assertIsNotNone(active_scene)
        self.assertIsNone(active_scene.region(RegionKind.START_DRAWING))
        self.assertIsNotNone(active_scene.current_paint)
        self.assertIsNotNone(active_scene.backpack)
        self.assertLess(active_scene.board.rect.x, waiting.shape[1] * 0.25)
        self.assertLess(transition_elapsed, 0.5)

        started = time.perf_counter()
        waiting_again = tracker.analyze_quick(waiting)
        repaint_elapsed = time.perf_counter() - started

        self.assertIsNotNone(waiting_again)
        self.assertIsNotNone(waiting_again.region(RegionKind.START_DRAWING))
        self.assertGreater(waiting_again.board.rect.x, waiting.shape[1] * 0.30)
        self.assertLess(repaint_elapsed, 0.5)

    def test_pending_paint_panel_is_not_a_brush_page(self) -> None:
        scene = analyze_scene(_user_screenshot("重新获取随机调色颜料之后.png"))

        self.assertIsNotNone(scene)
        self.assertIsNone(scene.brush_options)

    def test_realtime_clears_stale_brush_and_current_rule_after_reroll(self) -> None:
        tracker = VisionTracker()
        before = tracker.analyze_realtime(_user_screenshot("重新获取随机调色颜料之前.png"))
        after = tracker.analyze_quick(_user_screenshot("重新获取随机调色颜料之后.png"))

        self.assertIsNotNone(before)
        self.assertIsNotNone(before.brush_options)
        self.assertIsNotNone(after)
        self.assertIsNone(after.brush_options)
        self.assertIsNotNone(after.current_paint)
        self.assertEqual("unknown", after.current_paint.direction.value)
        self.assertEqual("unknown", after.current_paint.target_object.value)

    def test_enter_painting_page_is_a_terminal_action(self) -> None:
        image = _user_screenshot("结束--地块全部无空白和入画按钮.png")

        signal = _find_finish_drawing_signal(image)
        realtime_scene = analyze_scene_realtime(image)

        self.assertIsNotNone(signal)
        self.assertEqual("enter painting button", signal[2])
        self.assertIsNotNone(realtime_scene)
        self.assertIsNotNone(realtime_scene.region(RegionKind.FINISH_DRAWING))


if __name__ == "__main__":
    unittest.main()
