"""拖动游戏窗口改尺寸后，实时路径必须自己把棋盘标定找回来。

2026-11-15 23:14 实机事故：用户把窗口拖成 1536x877，``layout.json`` 里只有
1239x780 / 1920x1080 两组标定，实时路径退回内置比例，算出候选 (287,225,521)。
这个框网格分 0.416、棋盘综合分 0.146，却因为 presence 模板阈值只有 0.15
（相似度 0.249）被当成真棋盘；随后 ``analyze_scene_realtime`` 判分不达标直接
返回 None，自愈分支永远进不去、标定永远落不了盘 —— 自动识别全错，点
「完整识别」也拿不到棋盘（它同样走实时路径）。

固件是事故当时的实机帧（冻结在 tests/fixtures，不读会漂移的 runtime dump）。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from huahua_assistant import vision
from huahua_assistant.layout import ScreenLayout, calibrated_layout
from huahua_assistant.models import Rect, TileState

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "board_resized_1536x877_paints.png"
WIDTH, HEIGHT = 1536, 877
TRUE_RECT = Rect(241, 225, 523, 523)    # 全搜索 + 精修出来的真框
WRONG_RECT = Rect(287, 225, 521, 521)   # 内置比例算出来的错框


def _image() -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(FIXTURE), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise unittest.SkipTest(f"读不到固件 {FIXTURE}")
    return image


def _paint_count(image: np.ndarray, rect: Rect) -> int:
    return sum(1 for t in vision._classify_tiles(image, rect) if t.state == TileState.PAINT)


class ResizedWindowLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        vision.set_layout_autosave(False)   # 测试绝不写 runtime/layout.json
        vision.reset_vision_caches()
        vision._RECOVER_ATTEMPTS.clear()

    def tearDown(self) -> None:
        vision.reset_vision_caches()
        vision._RECOVER_ATTEMPTS.clear()

    def test_wrong_candidate_is_rejected_despite_presence(self) -> None:
        """presence 只证明「是画画界面」，不能替棋盘框背书。"""
        image = _image()
        # 错框确实过 presence，但棋盘自身的两条证据都立不住。
        self.assertLess(vision._score_board_rect(image, WRONG_RECT), 0.30)
        self.assertLess(
            vision._board_grid_score(image, WRONG_RECT, outer_penalty=False),
            vision.GRID_PRESENCE_SCORE,
        )
        self.assertIsNone(
            vision._pick_board_rect(image, ScreenLayout(), WIDTH, HEIGHT, drawing=True)
        )

    def test_calibrated_layout_recognises_all_five_paints(self) -> None:
        image = _image()
        layout = calibrated_layout(WIDTH, HEIGHT, TRUE_RECT)
        with patch.object(vision, "load_layout", return_value=layout):
            vision.reset_vision_caches()
            rect = vision._screen_layout(WIDTH, HEIGHT).board_rect(WIDTH, HEIGHT)
            self.assertEqual(TRUE_RECT, rect)
            self.assertEqual(5, _paint_count(image, rect))
            scene = vision.analyze_scene_realtime(image)
        self.assertIsNotNone(scene)
        self.assertEqual(TRUE_RECT, scene.board.rect)

    def test_realtime_self_heals_without_a_calibration(self) -> None:
        """没有这组尺寸的标定时，实时路径应自愈一次并交回真框。"""
        image = _image()
        with patch.object(vision, "load_layout", return_value=ScreenLayout()):
            vision.reset_vision_caches()
            scene = vision.analyze_scene_realtime(image)
        self.assertIsNotNone(scene)
        self.assertLessEqual(abs(scene.board.rect.x - TRUE_RECT.x), 6)
        self.assertLessEqual(abs(scene.board.rect.y - TRUE_RECT.y), 6)
        self.assertEqual(5, sum(1 for t in scene.board.tiles if t.state == TileState.PAINT))

    def test_realtime_self_heals_even_if_a_wrong_candidate_slips_through(self) -> None:
        """就算 presence 放过错框，低分也必须触发自愈，不能直接判「没棋盘」。"""
        image = _image()
        with patch.object(vision, "load_layout", return_value=ScreenLayout()), patch.object(
            vision, "_pick_board_rect", return_value=WRONG_RECT
        ):
            vision.reset_vision_caches()
            scene = vision.analyze_scene_realtime(image)
        self.assertIsNotNone(scene)
        self.assertLessEqual(abs(scene.board.rect.x - TRUE_RECT.x), 6)
        self.assertLessEqual(abs(scene.board.rect.y - TRUE_RECT.y), 6)
        self.assertEqual(5, sum(1 for t in scene.board.tiles if t.state == TileState.PAINT))

    def test_capture_watchdog_restores_a_hidden_window(self) -> None:
        """完整识别 withdraw 了主窗口，超时必须把它还回来。"""
        from huahua_assistant.app import HuahuaApp

        class FakeApp:
            _capture_inflight = True
            status_var = MagicMock()
            _restore_app_window = MagicMock()
            _set_notice = MagicMock()

        fake = FakeApp()
        HuahuaApp._capture_watchdog(fake)
        fake._restore_app_window.assert_called_once_with()
        fake._set_notice.assert_called_once()
        self.assertIn("超时", fake.status_var.set.call_args[0][0])

    def test_capture_watchdog_is_noop_after_a_normal_result(self) -> None:
        from huahua_assistant.app import HuahuaApp

        class FakeApp:
            _capture_inflight = False
            status_var = MagicMock()
            _restore_app_window = MagicMock()
            _set_notice = MagicMock()

        fake = FakeApp()
        HuahuaApp._capture_watchdog(fake)
        fake._restore_app_window.assert_not_called()
        fake.status_var.set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
