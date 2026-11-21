"""自动跟随不能抢焦点，棋盘尺寸先验要跟着分辨率走。

这两条都是实机反馈出来的：

1. 开着自动跟随切不到别的软件、游戏画面一黑一闪。根因是每 100ms 一次的截图
   带 ``activate=True``，等于每帧把游戏拽回前台；置顶的箭头/桌宠窗口还在每帧
   ``deiconify + lift``，又是一次激活。
2. 1920x1080 下棋盘框只套住中间 4x4，连带「当前颜料」区域往下滑，面板文字读不到，
   方向和作用颜料全成 UNKNOWN。
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from huahua_assistant import frame_pump, vision
from huahua_assistant.capture import CaptureResult, WindowInfo
from huahua_assistant.models import Rect


def _window(pid: int = 4242) -> WindowInfo:
    return WindowInfo(
        hwnd=1001,
        title="Torchlight: Infinite",
        process_id=pid,
        process_name="torchlight_infinite.exe",
        left=0,
        top=0,
        right=1920,
        bottom=1080,
        minimized=False,
    )


class AutoFollowDoesNotStealFocusTests(unittest.TestCase):
    def test_capture_never_activates_the_game_window(self) -> None:
        """自动跟随截图必须 activate=False，否则每帧抢一次焦点。"""
        with (
            patch.object(frame_pump, "find_capture_window", return_value=_window()),
            patch.object(frame_pump, "_foreground_allows_capture", return_value=True),
            patch.object(frame_pump, "capture_window_content", return_value=(None, "test")),
            patch.object(frame_pump, "capture_target") as capture_target,
        ):
            capture_target.return_value = CaptureResult(
                image=np.zeros((4, 4, 3), dtype=np.uint8), source="test", window=_window()
            )
            frame_pump.default_auto_capture()

        self.assertTrue(capture_target.called)
        kwargs = capture_target.call_args.kwargs
        self.assertFalse(kwargs["activate"], "自动跟随不能激活游戏窗口")
        self.assertEqual(0.0, kwargs["activation_delay"])

    def test_window_content_capture_skips_the_foreground_gate(self) -> None:
        """窗口内容抓得到时，前台是别的软件也照样读——遮挡/后台都不再是问题。"""
        frame = np.full((60, 80, 3), 128, dtype=np.uint8)
        with (
            patch.object(frame_pump, "find_capture_window", return_value=_window()),
            patch.object(frame_pump, "_foreground_allows_capture", return_value=False),
            patch.object(frame_pump, "capture_window_content", return_value=(frame, "wgc")),
            patch.object(frame_pump, "capture_target") as capture_target,
        ):
            result = frame_pump.default_auto_capture()

        capture_target.assert_not_called()
        self.assertNotEqual(frame_pump.PAUSED_SOURCE, result.source)
        self.assertIn("window-content", result.source)

    def test_other_app_in_foreground_pauses_the_capture(self) -> None:
        """窗口内容抓不到、前台又是别的软件时返回 paused，不去截图、更不去激活。"""
        with (
            patch.object(frame_pump, "find_capture_window", return_value=_window()),
            patch.object(frame_pump, "_foreground_allows_capture", return_value=False),
            patch.object(frame_pump, "capture_window_content", return_value=(None, "wgc_unavailable")),
            patch.object(frame_pump, "capture_target") as capture_target,
        ):
            result = frame_pump.default_auto_capture()

        capture_target.assert_not_called()
        self.assertEqual(frame_pump.PAUSED_SOURCE, result.source)

    def test_foreground_check_accepts_the_tool_itself(self) -> None:
        """工具自己在前台是常态（用户在看面板），不能因此停掉识别。"""
        window = _window(pid=999999)

        class _User32:
            @staticmethod
            def GetForegroundWindow() -> int:
                return 777

            @staticmethod
            def GetWindowThreadProcessId(hwnd, out) -> int:  # noqa: ANN001
                out._obj.value = os.getpid()  # 前台是本工具进程
                return 1

        with patch.object(frame_pump.ctypes, "windll") as windll:
            windll.user32 = _User32()
            allowed = frame_pump._foreground_allows_capture(window)

        self.assertTrue(allowed, "前台是本工具时仍要继续识别")

    def test_foreground_check_rejects_an_unrelated_app(self) -> None:
        """前台是浏览器之类的别的软件 → 暂停，不去截图。"""
        window = _window(pid=999999)

        class _User32:
            @staticmethod
            def GetForegroundWindow() -> int:
                return 777

            @staticmethod
            def GetWindowThreadProcessId(hwnd, out) -> int:  # noqa: ANN001
                out._obj.value = 424242  # 既不是游戏也不是本工具
                return 1

        with patch.object(frame_pump.ctypes, "windll") as windll:
            windll.user32 = _User32()
            allowed = frame_pump._foreground_allows_capture(window)

        self.assertFalse(allowed)


class BoardSizePriorTests(unittest.TestCase):
    def test_an_inner_slice_of_the_board_scores_low(self) -> None:
        """529/1080≈0.49 的「棋盘内部一圈」必须明显吃亏，不能再压过真棋盘。"""
        shape = (1080, 1920, 3)
        inner = vision._board_size_score(shape, Rect(429, 333, 529, 529))
        real = vision._board_size_score(shape, Rect(354, 241, 632, 632))
        self.assertLess(inner, 0.85)
        self.assertGreaterEqual(real, 0.99)
        self.assertGreater(real - inner, 0.15)

    def test_prior_scales_with_the_window(self) -> None:
        """同一块棋盘缩放到不同分辨率，尺寸先验都应接近满分。"""
        for height, width, side in (
            (1080, 1920, 626),
            (900, 1600, 522),
            (1440, 2560, 831),
            (2160, 3840, 1268),
        ):
            score = vision._board_size_score((height, width, 3), Rect(0, 0, side, side))
            self.assertGreaterEqual(score, 0.99, f"{width}x{height} 尺寸先验异常")


if __name__ == "__main__":
    unittest.main()
