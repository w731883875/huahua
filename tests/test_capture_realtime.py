from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from huahua_assistant.capture import (
    CaptureResult,
    LegacyCaptureBackend,
    WindowInfo,
    activate_window,
    capture_target,
    physical_pixel_coordinates,
)


def _game_window() -> WindowInfo:
    return WindowInfo(
        hwnd=123,
        title="Torchlight: Infinite",
        process_id=456,
        process_name="torchlight_infinite.exe",
        left=10,
        top=20,
        right=1699,
        bottom=906,
        minimized=False,
    )


class CaptureRealtimeTests(unittest.TestCase):
    def test_confirmed_foreground_region_capture_is_marked_trusted(self) -> None:
        class RegionBackend:
            name = "test"

            def capture_window_content(self, _window):
                return None, "content_unavailable"

            def capture_window_region(self, _window):
                return np.zeros((886, 1689, 3), dtype=np.uint8), "region"

        with (
            patch("huahua_assistant.capture.find_matching_windows", return_value=[_game_window()]),
            patch("huahua_assistant.capture.activate_window"),
            patch("huahua_assistant.capture.is_foreground_window", return_value=True),
        ):
            result = capture_target(
                activate=True,
                activation_delay=0.0,
                fallback_to_screen=False,
                backend=RegionBackend(),
            )

        self.assertTrue(result.source.startswith("window-foreground-screen:"))

    def test_directx_game_skips_printwindow_retry(self) -> None:
        backend = LegacyCaptureBackend()

        with patch("huahua_assistant.capture._capture_window_print") as print_window:
            image, reason = backend.capture_window_content(_game_window())

        self.assertIsNone(image)
        self.assertEqual("printwindow_skipped_directx", reason)
        print_window.assert_not_called()

    def test_capture_target_runs_inside_physical_pixel_context(self) -> None:
        expected = CaptureResult(image=np.zeros((4, 5, 3), dtype=np.uint8), source="test")
        events: list[str] = []

        class RecordingContext:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, exc_type, exc, traceback):
                events.append("exit")

        with (
            patch("huahua_assistant.capture.physical_pixel_coordinates", return_value=RecordingContext()),
            patch("huahua_assistant.capture._capture_target_physical", return_value=expected) as physical,
        ):
            actual = capture_target(activate=False, activation_delay=0.0)

        self.assertIs(actual, expected)
        self.assertEqual(["enter", "exit"], events)
        physical.assert_called_once()

    def test_thread_dpi_context_is_restored(self) -> None:
        setter = MagicMock(side_effect=[321, 654])
        fake_windll = SimpleNamespace(user32=SimpleNamespace(SetThreadDpiAwarenessContext=setter))

        with patch("huahua_assistant.capture.ctypes.windll", fake_windll):
            with physical_pixel_coordinates():
                pass

        self.assertEqual(2, setter.call_count)
        self.assertEqual(321, setter.call_args_list[1].args[0])

    def test_activation_raises_window_without_leaving_it_topmost(self) -> None:
        user32 = SimpleNamespace(
            SetWindowPos=MagicMock(),
            BringWindowToTop=MagicMock(),
            SetForegroundWindow=MagicMock(),
        )
        with (
            patch("huahua_assistant.capture.restore_window") as restore,
            patch("huahua_assistant.capture.ctypes.windll", SimpleNamespace(user32=user32)),
        ):
            activate_window(123)

        restore.assert_called_once_with(123)
        self.assertEqual(2, user32.SetWindowPos.call_count)
        self.assertEqual(-1, user32.SetWindowPos.call_args_list[0].args[1])
        self.assertEqual(-2, user32.SetWindowPos.call_args_list[1].args[1])


if __name__ == "__main__":
    unittest.main()
