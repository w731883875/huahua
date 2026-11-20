"""窗口内容抓取（Windows.Graphics.Capture）的接线与降级。

实机反馈出来的问题：DirectX 游戏的窗口用 ``PrintWindow`` 读不出内容，只能退回
「截屏幕」。一旦工具窗口盖在游戏上，截到的就是工具自己——棋盘 36 格全是 ``?``，
自动识别等于没开。WGC 走的是窗口自己渲染的表面，盖在上面的窗口不会进帧。

这里只测接线、降级和顺序，不依赖真实窗口。
"""

from __future__ import annotations

import ctypes
import unittest
from unittest.mock import patch

import numpy as np

from huahua_assistant import capture as capture_module
from huahua_assistant.capture import (
    LegacyCaptureBackend,
    WindowInfo,
    WindowsGraphicsCaptureBackend,
    _looks_like_valid_capture,
    _visible_bounds,
    capture_target,
    resolve_capture_backend,
    window_content_capture_ready,
)


def _window(hwnd: int = 4321, *, minimized: bool = False) -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd,
        title="Torchlight: Infinite",
        process_id=987,
        process_name="torchlight_infinite.exe",
        left=0,
        top=0,
        right=1550,
        bottom=830,
        minimized=minimized,
    )


class VisibleBoundsTests(unittest.TestCase):
    """GetWindowRect 含 9px 隐形边框，WGC 的帧不含——两边必须对齐。"""

    def test_falls_back_when_dwm_refuses(self) -> None:
        fallback = ctypes.wintypes.RECT(0, 0, 100, 50)

        with patch.object(capture_module.ctypes, "windll") as windll:
            windll.dwmapi.DwmGetWindowAttribute.return_value = -1
            result = _visible_bounds(123, fallback)

        self.assertEqual((0, 0, 100, 50), (result.left, result.top, result.right, result.bottom))

    def test_uses_dwm_bounds_when_available(self) -> None:
        fallback = ctypes.wintypes.RECT(0, 0, 1938, 1038)

        def fake_attribute(_hwnd, _attr, out, _size):  # noqa: ANN001
            target = out._obj
            target.left, target.top, target.right, target.bottom = 9, 9, 1929, 1029
            return 0

        with patch.object(capture_module.ctypes, "windll") as windll:
            windll.dwmapi.DwmGetWindowAttribute.side_effect = fake_attribute
            result = _visible_bounds(123, fallback)

        self.assertEqual((9, 9, 1929, 1029), (result.left, result.top, result.right, result.bottom))

    def test_empty_dwm_rectangle_is_rejected(self) -> None:
        fallback = ctypes.wintypes.RECT(0, 0, 100, 50)

        def fake_attribute(_hwnd, _attr, out, _size):  # noqa: ANN001
            return 0  # 成功但什么都没写：矩形是空的

        with patch.object(capture_module.ctypes, "windll") as windll:
            windll.dwmapi.DwmGetWindowAttribute.side_effect = fake_attribute
            result = _visible_bounds(123, fallback)

        self.assertEqual((0, 0, 100, 50), (result.left, result.top, result.right, result.bottom))


class BackendPlanTests(unittest.TestCase):
    def test_auto_puts_window_content_first(self) -> None:
        plan = resolve_capture_backend("auto")

        self.assertIsInstance(plan.content_backends[0], WindowsGraphicsCaptureBackend)
        self.assertIn(LegacyCaptureBackend, [type(item) for item in plan.content_backends])

    def test_auto_can_be_turned_off(self) -> None:
        with patch.object(capture_module, "_capture_config_bool", side_effect=lambda path, default: (
            False if path == "capture.wgc_enabled" else default
        )):
            plan = resolve_capture_backend("auto")

        self.assertNotIsInstance(plan.content_backends[0], WindowsGraphicsCaptureBackend)

    def test_explicit_wgc_keeps_the_legacy_fallback(self) -> None:
        plan = resolve_capture_backend("wgc")

        self.assertIsInstance(plan.content_backends[0], WindowsGraphicsCaptureBackend)
        self.assertIn(LegacyCaptureBackend, [type(item) for item in plan.content_backends])


class WgcBackendTests(unittest.TestCase):
    def test_minimized_window_is_not_attempted(self) -> None:
        backend = WindowsGraphicsCaptureBackend()
        image, method = backend.capture_window_content(_window(minimized=True))

        self.assertIsNone(image)
        self.assertEqual("wgc_minimized", method)

    def test_missing_module_is_reported_once(self) -> None:
        backend = WindowsGraphicsCaptureBackend()
        with (
            patch.object(capture_module, "integrity_allows_window_capture", return_value=True),
            patch.object(capture_module, "_WgcSession", side_effect=ImportError("no module")),
        ):
            _image, first = backend.capture_window_content(_window())
            _image, second = backend.capture_window_content(_window())

        self.assertIn("wgc_unavailable", first)
        self.assertEqual(first, second, "缺模块之后不该每次重试")

    def test_refused_window_falls_through_without_caching_failure(self) -> None:
        backend = WindowsGraphicsCaptureBackend()
        with (
            patch.object(capture_module, "integrity_allows_window_capture", return_value=True),
            patch.object(capture_module, "_WgcSession", side_effect=RuntimeError("capture item refused")),
        ):
            image, method = backend.capture_window_content(_window())

        self.assertIsNone(image)
        self.assertIn("wgc_start_error", method)
        self.assertFalse(window_content_capture_ready(_window()), "没建立会话就不能当成可用")

    def test_low_integrity_is_not_retried_every_frame(self) -> None:
        """LOW 完整性（自动化环境拉起的进程）对任何目标都是 E_ACCESSDENIED。

        这条必须先判：否则每帧都构造一次会话对象、抛一次异常，还把状态报成
        「可用」，上层据此认为「工具窗口挡住游戏不要紧」，脏帧直接进识别。
        """

        backend = WindowsGraphicsCaptureBackend()
        with (
            patch.object(capture_module, "integrity_allows_window_capture", return_value=False),
            patch.object(capture_module, "_WgcSession") as session,
        ):
            _image, first = backend.capture_window_content(_window())
            _image, second = backend.capture_window_content(_window(hwnd=777))
            status = backend.unavailable_reason

        session.assert_not_called()
        self.assertEqual("wgc_blocked_low_integrity", first)
        self.assertEqual(first, second, "判定之后不该再按句柄重试")
        self.assertEqual("wgc_blocked_low_integrity", status)
        with patch.object(capture_module, "_WGC_BACKEND", backend):
            self.assertFalse(
                capture_module.window_content_capture_available(),
                "判定不可用之后不能再对外声称可用，否则遮挡判断会失效",
            )

    def test_repeated_start_failures_stop_pretending_to_be_ready(self) -> None:
        backend = WindowsGraphicsCaptureBackend()
        with (
            patch.object(capture_module, "integrity_allows_window_capture", return_value=True),
            patch.object(
                capture_module,
                "_WgcSession",
                side_effect=Exception("Failed to convert item to `GraphicsCaptureItem`"),
            ) as session,
        ):
            for _ in range(backend.MAX_START_FAILURES):
                backend.capture_window_content(_window())

        self.assertTrue(backend.unavailable_reason, "连续失败到阈值必须判定不可用")
        self.assertIn("wgc_start_failed", backend.unavailable_reason or "")
        attempts = session.call_count
        self.assertEqual(backend.MAX_START_FAILURES, attempts)

        with patch.object(capture_module, "_WgcSession", side_effect=AssertionError("不该再试")):
            _image, method = backend.capture_window_content(_window(hwnd=555))
        self.assertIn("wgc_start_failed", method)

    def test_reset_lets_a_restarted_game_try_again(self) -> None:
        backend = WindowsGraphicsCaptureBackend()
        with (
            patch.object(capture_module, "integrity_allows_window_capture", return_value=True),
            patch.object(capture_module, "_WgcSession", side_effect=Exception("boom")),
        ):
            for _ in range(backend.MAX_START_FAILURES):
                backend.capture_window_content(_window())
            self.assertIsNotNone(backend.unavailable_reason)

        backend.reset()
        self.assertIsNone(backend.unavailable_reason)

    def test_no_session_means_not_ready(self) -> None:
        self.assertFalse(window_content_capture_ready(None))
        self.assertFalse(window_content_capture_ready(_window(hwnd=999999)))


class CaptureOrderTests(unittest.TestCase):
    """窗口内容优先：读得到就不去激活窗口、也不去截屏幕。"""

    def test_content_is_tried_before_activation(self) -> None:
        class ContentBackend:
            name = "content"

            def capture_window_content(self, _window):
                return np.full((80, 120, 3), 90, dtype=np.uint8), "fake"

            def capture_window_region(self, _window):
                raise AssertionError("不该走到截屏")

        with (
            patch.object(capture_module, "find_matching_windows", return_value=[_window()]),
            patch.object(capture_module, "activate_window") as activate_window,
        ):
            result = capture_target(activate=True, backend=ContentBackend())

        activate_window.assert_not_called()
        self.assertIn("window-content", result.source)
        self.assertEqual((80, 120), result.image.shape[:2])

    def test_falls_back_to_screen_when_content_is_unavailable(self) -> None:
        class EmptyBackend:
            name = "empty"

            def capture_window_content(self, _window):
                return None, "content_unavailable"

            def capture_window_region(self, _window):
                return np.full((80, 120, 3), 70, dtype=np.uint8), "region"

        with (
            patch.object(capture_module, "find_matching_windows", return_value=[_window()]),
            patch.object(capture_module, "activate_window"),
            patch.object(capture_module, "is_foreground_window", return_value=True),
        ):
            result = capture_target(activate=True, backend=EmptyBackend())

        self.assertIn("content_failed", result.source)
        self.assertIn("region", result.source)


class ValidCaptureTests(unittest.TestCase):
    def test_blank_frame_is_rejected(self) -> None:
        self.assertFalse(_looks_like_valid_capture(np.zeros((600, 800, 3), dtype=np.uint8)))

    def test_flat_grey_frame_is_rejected(self) -> None:
        self.assertFalse(_looks_like_valid_capture(np.full((600, 800, 3), 40, dtype=np.uint8)))

    def test_textured_frame_is_accepted(self) -> None:
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
        self.assertTrue(_looks_like_valid_capture(frame))

    def test_validation_samples_instead_of_scanning_every_pixel(self) -> None:
        """1920x1020 全分辨率转灰度比抓屏本身还慢，必须走抽样。"""

        rng = np.random.default_rng(11)
        frame = rng.integers(0, 255, (1020, 1920, 3), dtype=np.uint8)
        sampled_shapes: list[tuple[int, ...]] = []
        original = capture_module.cv2.cvtColor

        def spy(source, code):  # noqa: ANN001
            sampled_shapes.append(source.shape)
            return original(source, code)

        with patch.object(capture_module.cv2, "cvtColor", side_effect=spy):
            _looks_like_valid_capture(frame)

        self.assertTrue(sampled_shapes, "应当发生一次灰度转换")
        self.assertLess(sampled_shapes[0][0], 1020, "不该对整帧做灰度转换")


if __name__ == "__main__":
    unittest.main()
