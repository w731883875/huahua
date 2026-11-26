"""Capture discovery must stay independent of a busy Tk callback at finish."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from huahua_assistant import app, capture, frame_pump
from huahua_assistant.level_smoothing import LevelStabilizer
from tests.test_capture_realtime import _game_window


class WindowLookupTests(unittest.TestCase):
    def test_skip_own_process_before_any_synchronous_title_read(self):
        titles = {101: "Huahua", 102: "Torchlight: Infinite"}

        def process_id(hwnd, output):
            output._obj.value = 777 if hwnd == 101 else 888
            return 1

        def text(hwnd, output, size):
            output.value = titles[hwnd]
            return len(output.value)

        def rect(hwnd, output):
            output._obj.right, output._obj.bottom = 1280, 720
            return 1

        user32 = SimpleNamespace(
            EnumWindows=lambda callback, param: [callback(h, param) for h in titles],
            IsWindowVisible=lambda hwnd: True,
            GetWindowThreadProcessId=process_id,
            GetWindowTextLengthW=Mock(side_effect=lambda hwnd: len(titles[hwnd])),
            GetWindowTextW=Mock(side_effect=text), GetWindowRect=rect,
            IsIconic=lambda hwnd: False, GetClientRect=rect,
            ClientToScreen=lambda hwnd, output: True,
        )
        with (
            patch.object(capture.ctypes, "windll", SimpleNamespace(user32=user32)),
            patch.object(capture, "getpid", return_value=777),
            patch.object(capture, "_process_name", return_value="torchlight_infinite.exe"),
            patch.object(capture, "_visible_bounds", side_effect=lambda hwnd, bounds: bounds),
        ):
            window = capture.find_capture_window()

        self.assertEqual(window.hwnd, 102)
        self.assertEqual(window.client_bounds, (0, 0, 1280, 720))
        user32.GetWindowTextLengthW.assert_called_once_with(102)
        self.assertEqual(user32.GetWindowTextW.call_args.args[0], 102)

    def test_region_failure_never_falls_back_to_whole_desktop(self):
        backend = Mock()
        backend.name = "test"
        backend.capture_window_region.return_value = (None, "failed")
        with (
            patch.object(capture, "resolve_capture_backend", return_value=capture.CaptureBackendPlan("test", (), (backend,))),
            patch.object(capture, "capture_screen") as screen,
        ):
            with self.assertRaisesRegex(RuntimeError, "test:failed"):
                capture.capture_window_region(_game_window())
        screen.assert_not_called()

    def test_minimized_window_does_not_use_stale_bounds(self):
        with patch.object(capture, "resolve_capture_backend") as resolve:
            with self.assertRaises(RuntimeError):
                capture.capture_window_region(replace(_game_window(), minimized=True))
        resolve.assert_not_called()

    def test_capture_stage_timings_reach_slow_frame_diagnostics(self):
        reports = []
        stages = {"find_window": 0.01, "window_region": 7.7}
        pump = frame_pump.FramePump(
            capture_func=lambda: capture.CaptureResult(
                image=np.zeros((8, 8, 3), dtype=np.uint8), source="test", timings=stages),
            analyze_func=lambda frame: None, needs_full_refresh=lambda: False,
            full_analyze_func=lambda frame: None, slow_frame_func=reports.append,
        )
        result = pump._capture_and_analyze()
        pump._report_slow_frame(result, 8.0)
        self.assertEqual(reports[0]["capture_stages"], stages)


class FinishStatsRefreshTests(unittest.TestCase):
    def test_successful_finish_wakes_confirmation_capture_after_saving(self):
        for reliable in (True, False):
            with self.subTest(reliable=reliable):
                events = []
                fake = SimpleNamespace(
                    current_state=object(), current_scene=None, last_capture_quality=None,
                    real_trajectory_steps=[], real_session_id="test-run", real_session_started_at=1.0,
                    real_session_finished=False, level_stabilizer=LevelStabilizer(),
                    _initial_snapshot_from_steps=Mock(return_value=None),
                    _set_finish_diagnostic=Mock(), _schedule_learning_stats_refresh=Mock(),
                    _wake_auto_follow_pump=lambda: events.append("wake"),
                )
                with (
                    patch.object(app, "_best_trusted_terminal_summary", return_value={"known": {"total": 20}}),
                    patch.object(app, "_terminal_reliability_for_finish", return_value=(reliable, "test")),
                    patch.object(app, "_high_score_gate", return_value={"passed": True}),
                    patch.object(app, "record_high_score_trajectory", side_effect=lambda *a, **k: events.append("save") or True),
                    patch.object(app, "record_observation"),
                ):
                    saved, _ = app.HuahuaApp._finish_real_session_internal(fake, reason="board_full_autosave")
                self.assertEqual(saved, reliable)
                self.assertEqual(events, ["save", "wake"] if reliable else [])
                self.assertEqual(fake.level_stabilizer.frozen, reliable)
                self.assertEqual(fake.real_session_finished, reliable)

    def test_stats_refresh_does_not_start_recommendations_after_finish_or_in_record_mode(self):
        for finished, recording in ((True, False), (False, True), (True, True), (False, False)):
            with self.subTest(finished=finished, recording=recording):
                fake = SimpleNamespace(
                    current_state=object(), real_session_finished=finished,
                    _write_learning_stats=Mock(), _write_current_terrain_bucket_hint=Mock(),
                    _refresh_after_correction=Mock(),
                )
                with patch.object(app, "recording_mode_enabled", return_value=recording):
                    app.HuahuaApp._apply_learning_stats_to_ui(fake)
                fake._write_learning_stats.assert_called_once_with()
                self.assertEqual(fake._refresh_after_correction.call_count, int(not finished and not recording))

    def test_manual_correction_still_redraws_without_recommendations_in_record_mode(self):
        fake = SimpleNamespace(
            current_state=object(), current_scene=None, real_session_finished=False,
            canvas=Mock(), _draw_grid=Mock(), _write_simulation=Mock(), _refresh_overlay=Mock(),
        )
        with (
            patch.object(app, "recording_mode_enabled", return_value=True),
            patch.object(app, "recommend_placements") as recommend,
            patch.object(app, "recommend_install_plan") as plan,
        ):
            app.HuahuaApp._refresh_after_correction(fake)
        recommend.assert_not_called()
        plan.assert_not_called()
        self.assertEqual(fake.recommendations, [])
        fake._draw_grid.assert_called_once_with(None, fake.current_state)


if __name__ == "__main__":
    unittest.main()
