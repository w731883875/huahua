from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from huahua_assistant.capture import read_image
from huahua_assistant.models import Rect, RegionKind, RegionObservation, TileState
from huahua_assistant.vision import (
    _analyze_brush_options,
    _classify_backpack_frame_shape,
    _classify_tile_for_board_score,
    _find_finish_drawing_signal,
    analyze_scene,
    analyze_scene_quick,
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
        desktop = analyze_scene(_test_fixture("non_game_desktop.png"))
        active_game = analyze_scene(_fixture("capture_perf_write.png"))
        terminal_game_image = _fixture("capture_auto.png")
        terminal_game = analyze_scene(terminal_game_image)

        self.assertIsNone(desktop)
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

    def test_pending_paint_panel_is_not_a_brush_page(self) -> None:
        scene = analyze_scene(_user_screenshot("重新获取随机调色颜料之后.png"))

        self.assertIsNotNone(scene)
        self.assertIsNone(scene.brush_options)

    def test_enter_painting_page_is_a_terminal_action(self) -> None:
        image = _user_screenshot("结束--地块全部无空白和入画按钮.png")

        signal = _find_finish_drawing_signal(image)

        self.assertIsNotNone(signal)
        self.assertEqual("enter painting button", signal[2])


if __name__ == "__main__":
    unittest.main()
