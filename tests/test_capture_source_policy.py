#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""截图来源分级 + 自动外发/自动落盘闸门的回归测试。

要钉住的事只有一件：**不是游戏窗口自己渲染的像素，不能自动落下、不能自动外发。**
容易出错的地方是「带 window 对象就算抓到游戏」——窗口存在但各后端失败时回退的
整屏图同样带 window 对象，只看这个字段会放行整个桌面。
"""
from __future__ import annotations

import unittest

import numpy as np

from huahua_assistant import app as app_mod
from huahua_assistant import capture as cap
from huahua_assistant import frame_pump as fp


def _window(**kwargs) -> cap.WindowInfo:
    base = dict(
        hwnd=1001,
        title='Torchlight: Infinite',
        left=0,
        top=0,
        right=1280,
        bottom=720,
        minimized=False,
        process_id=4242,
        process_name='TLS.exe',
    )
    base.update(kwargs)
    return cap.WindowInfo(**base)


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


class CaptureKindTest(unittest.TestCase):
    def test_kind_mapping(self) -> None:
        cases = [
            ('window-content:T process:TLS.exe method:wgc', cap.CAPTURE_KIND_WINDOW_CONTENT),
            ('window-foreground-screen:T process:TLS.exe', cap.CAPTURE_KIND_WINDOW_REGION),
            ('window-screen:T process:TLS.exe', cap.CAPTURE_KIND_WINDOW_REGION),
            ('screen:fallback window_capture_failed:T', cap.CAPTURE_KIND_SCREEN),
            ('screen:fallback window_not_found:torchlight_infinite', cap.CAPTURE_KIND_SCREEN),
            ('paused:game-not-foreground', cap.CAPTURE_KIND_PAUSED),
            (fp.PAUSED_WINDOW_MISSING, cap.CAPTURE_KIND_PAUSED),
            ('frame-pump:error', cap.CAPTURE_KIND_ERROR),
            ('', cap.CAPTURE_KIND_UNKNOWN),
            (None, cap.CAPTURE_KIND_UNKNOWN),
            ('something-new', cap.CAPTURE_KIND_UNKNOWN),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(cap.capture_kind(source), expected)

    def test_only_window_content_is_trusted(self) -> None:
        trusted = {
            'window-content:T process:TLS.exe method:wgc',
        }
        untrusted = [
            'window-foreground-screen:T process:TLS.exe',
            'window-screen:T process:TLS.exe',
            'screen:fallback window_capture_failed:T',
            'paused:game-window-missing',
            'frame-pump:error',
            '',
        ]
        for source in trusted:
            self.assertTrue(cap.CaptureResult(image=_frame(), source=source).trusted_for_automation)
        for source in untrusted:
            self.assertFalse(
                cap.CaptureResult(image=_frame(), source=source, window=_window()).trusted_for_automation,
                f'{source} 不该被当成可信来源',
            )

    def test_window_object_alone_is_not_enough(self) -> None:
        """负例：带 window 对象的整屏回退仍然不可信。"""
        result = cap.CaptureResult(
            image=_frame(),
            source='screen:fallback window_capture_failed:Torchlight attempts:wgc:failed',
            window=_window(),
        )
        self.assertIsNotNone(result.window, '这条来源确实带 window 对象')
        self.assertFalse(result.trusted_for_automation)


class DefaultAutoCaptureTest(unittest.TestCase):
    """``default_auto_capture`` 的五种情形。"""

    def setUp(self) -> None:
        self._orig = {
            'find_capture_window': fp.find_capture_window,
            'capture_window_content': fp.capture_window_content,
            'capture_window_region': fp.capture_window_region,
            'foreground': fp._foreground_allows_capture,
        }
        self.calls: dict[str, int] = {'target': 0, 'content': 0, 'screen': 0}

    def tearDown(self) -> None:
        fp.find_capture_window = self._orig['find_capture_window']
        fp.capture_window_content = self._orig['capture_window_content']
        fp.capture_window_region = self._orig['capture_window_region']
        fp._foreground_allows_capture = self._orig['foreground']

    def _fake_target(self, window, **kwargs):
        self.calls['target'] += 1
        return cap.CaptureResult(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            source='window-screen:Torchlight method:test',
            window=window,
        )

    # ---- 1. 游戏没开 / 最小化 ----------------------------------------
    def test_window_missing_waits_instead_of_capturing_desktop(self) -> None:
        fp.find_capture_window = lambda *a, **k: None
        result = fp.default_auto_capture()
        self.assertEqual(result.source, fp.PAUSED_WINDOW_MISSING)
        self.assertEqual(result.kind, cap.CAPTURE_KIND_PAUSED)
        self.assertIsNone(result.window)
        self.assertEqual(self.calls['target'], 0, '找不到窗口时不该再走整屏回退')
        self.assertFalse(result.trusted_for_automation)

    # ---- 2. 正常内容帧 -------------------------------------------------
    def test_window_content_frame_is_trusted(self) -> None:
        fp.find_capture_window = lambda *a, **k: _window()
        fp.capture_window_content = lambda w: (_frame(), 'wgc')
        result = fp.default_auto_capture()
        self.assertTrue(result.source.startswith('window-content'))
        self.assertTrue(result.trusted_for_automation)
        self.assertEqual(self.calls['target'], 0)

    # ---- 3. 游戏在但不在前台 ------------------------------------------
    def test_background_game_pauses(self) -> None:
        fp.find_capture_window = lambda *a, **k: _window()
        fp.capture_window_content = lambda w: (None, 'failed')
        fp._foreground_allows_capture = lambda w: False
        result = fp.default_auto_capture()
        self.assertEqual(result.source, fp.PAUSED_SOURCE)
        self.assertEqual(result.kind, cap.CAPTURE_KIND_PAUSED)
        self.assertFalse(result.trusted_for_automation)

    # ---- 4. 窗口内容失败但游戏在前台：允许降级 -------------------------
    def test_foreground_fallback_captures_only_the_selected_window_region(self) -> None:
        fp.find_capture_window = lambda *a, **k: _window()
        fp.capture_window_content = lambda w: (None, 'failed')
        fp._foreground_allows_capture = lambda w: True
        fp.capture_window_region = self._fake_target
        result = fp.default_auto_capture()
        self.assertEqual(self.calls['target'], 1, '前台降级路径要保留')
        self.assertEqual(result.kind, cap.CAPTURE_KIND_WINDOW_REGION)
        self.assertFalse(result.trusted_for_automation, '降级出来的合成画面不能自动外发')


class FramePumpWriteGateTest(unittest.TestCase):
    def _pump(self, capture_func, saved: list):
        return fp.FramePump(
            capture_func=capture_func,
            analyze_func=lambda frame: None,
            needs_full_refresh=lambda: False,
            full_analyze_func=lambda frame: None,
            maybe_save_frame=lambda frame: saved.append(frame) or 0.0,
        )

    def test_trusted_frame_is_saved(self) -> None:
        saved: list = []
        pump = self._pump(
            lambda: cap.CaptureResult(
                image=_frame(), source='window-content:T process:TLS.exe method:wgc'
            ),
            saved,
        )
        result = pump._capture_and_analyze()
        self.assertEqual(len(saved), 1)
        self.assertIsNone(result.perf['write_gate'])

    def test_screen_fallback_is_not_saved(self) -> None:
        saved: list = []
        pump = self._pump(
            lambda: cap.CaptureResult(
                image=_frame(),
                source='screen:fallback window_capture_failed:T',
                window=_window(),
            ),
            saved,
        )
        result = pump._capture_and_analyze()
        self.assertEqual(saved, [], '整屏回退不该自动写调试图')
        self.assertEqual(result.perf['write_gate'], 'untrusted_source:screen')

    def test_window_region_is_not_saved(self) -> None:
        saved: list = []
        pump = self._pump(
            lambda: cap.CaptureResult(
                image=_frame(),
                source='window-screen:T process:TLS.exe',
                window=_window(),
            ),
            saved,
        )
        pump._capture_and_analyze()
        self.assertEqual(saved, [], '桌面区域里可能是别人的窗口')

    def test_capture_error_reports_error_kind(self) -> None:
        def boom():
            raise RuntimeError('no screen')

        pump = fp.FramePump(
            capture_func=boom,
            analyze_func=lambda frame: None,
            needs_full_refresh=lambda: False,
            full_analyze_func=lambda frame: None,
        )
        result = pump._capture_and_analyze()
        self.assertEqual(result.capture.kind, cap.CAPTURE_KIND_ERROR)
        self.assertIsNone(result.snapshot)
        self.assertFalse(result.capture.trusted_for_automation)


class LlmGateTest(unittest.TestCase):
    """外发闸门：只看窗口内容。这里是 ``_llm_may_receive_frame`` 的负例。"""

    def test_trusted_only_for_window_content(self) -> None:
        ok = cap.CaptureResult(image=_frame(), source='window-content:T method:wgc')
        self.assertTrue(app_mod._llm_may_receive_frame(ok))
        for source in (
            'window-screen:T process:TLS.exe',
            'screen:fallback window_capture_failed:T',
            'paused:game-window-missing',
            'frame-pump:error',
        ):
            with self.subTest(source=source):
                self.assertFalse(
                    app_mod._llm_may_receive_frame(
                        cap.CaptureResult(image=_frame(), source=source, window=_window())
                    )
                )
        self.assertFalse(app_mod._llm_may_receive_frame(None))

    def test_send_gate_uses_capture_kind(self) -> None:
        self.assertTrue(app_mod._llm_may_send_source(cap.CAPTURE_KIND_WINDOW_CONTENT))
        for kind in (
            cap.CAPTURE_KIND_WINDOW_REGION,
            cap.CAPTURE_KIND_SCREEN,
            cap.CAPTURE_KIND_PAUSED,
            cap.CAPTURE_KIND_ERROR,
            cap.CAPTURE_KIND_UNKNOWN,
            '',
            None,
        ):
            with self.subTest(kind=kind):
                self.assertFalse(app_mod._llm_may_send_source(kind))


if __name__ == '__main__':
    unittest.main()
