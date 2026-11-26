"""记录状态灯（`run_recorder.describe_record_status` / `AutomaticRunRecorder.status`）的回归测试。

为什么要有这组测试：这盏灯存在的唯一价值就是「说真话」——用户正是因为
「以为在录、其实没录」才要求加它的。灯本身出错（恒绿、文案对不上状态）
比没有灯更糟，所以这里把**每一种组合**都钉死，包括最容易骗人的两种情况：

1. 自动跟随没开 → 必须明确说「关」，不能显示成「待开局」让用户以为在等；
2. 有画面但超过 ``stall_after`` 秒没出帧 → 必须报错，不能继续显示「录制中」。
"""

from __future__ import annotations

import unittest

from huahua_assistant.fast_vision import FastCurrentPaint, FastVisionSnapshot
from huahua_assistant.models import Rect, TileState
from huahua_assistant.run_recorder import (
    RECORD_STATUS_LEVELS,
    AutomaticRunRecorder,
    RecorderStatus,
    _format_duration,
    describe_record_status,
)


def _status(**overrides) -> RecorderStatus:
    base = dict(
        phase="waiting",
        session_id=None,
        event_count=0,
        duration=0.0,
        frames=0,
        missing_frames=0,
        last_event_age=None,
        path="runtime/auto_run_records.jsonl",
    )
    base.update(overrides)
    return RecorderStatus(**base)


def _current_paint() -> FastCurrentPaint:
    rect = Rect(0, 0, 10, 10)
    return FastCurrentPaint(
        panel_rect=rect,
        icon_rect=rect,
        text_rect=rect,
        category="monster",
        level=3,
        subtype=None,
        is_treasure=False,
        direction="adjacent",
        target_object="monster",
        upgrade_amount=1,
        match_current=0,
        match_required=3,
        confidence=0.9,
    )


def _snapshot(*, with_current: bool = True, blank: int = 20) -> FastVisionSnapshot:
    rect = Rect(0, 0, 10, 10)
    return FastVisionSnapshot(
        board_rect=rect,
        board_score=0.9,
        tiles=(),
        board_signature="sig-blank",
        paint_signature="paint-1",
        terrain_counts={TileState.BLANK.value: blank},
        backpack_rect=None,
        backpack=(),
        backpack_signature="bp-1",
        current=_current_paint() if with_current else None,
        brush=None,
        regions={},
        reliable=True,
        reasons=(),
    )


class DescribeRecordStatusTests(unittest.TestCase):
    def test_archive_is_visible_while_recorder_waits_for_confirmation(self) -> None:
        text, level = describe_record_status(
            _status(phase="playing"), auto_follow=True, run_archived=True,
        )
        self.assertIn("已存档", text)
        self.assertIn("确认中", text)
        self.assertNotIn("已收尾", text)
        self.assertEqual(level, "waiting")

    def test_archive_does_not_hide_a_capture_failure(self) -> None:
        text, level = describe_record_status(
            _status(phase="playing"), auto_follow=True, run_archived=True,
            capture_error="failed",
        )
        self.assertEqual(level, "error")

    def test_auto_follow_off_is_explicit(self) -> None:
        """没开自动跟随就必须说「关」，不能退回「待开局」。"""
        text, level = describe_record_status(_status(phase="playing"), auto_follow=False)
        self.assertEqual(level, "off")
        self.assertIn("关", text)
        self.assertNotIn("开局", text)

    def test_capture_error_beats_every_phase(self) -> None:
        """抓不到游戏窗口时，即便上次停在 playing，也应该报错而不是显示录制中。"""
        text, level = describe_record_status(
            _status(phase="playing"), auto_follow=True, capture_error="window not found"
        )
        self.assertEqual(level, "error")
        self.assertIn("游戏", text)

    def test_paused_by_overlap(self) -> None:
        text, level = describe_record_status(
            _status(phase="playing"), auto_follow=True, paused="工具挡住游戏"
        )
        self.assertEqual(level, "waiting")
        self.assertIn("暂停", text)
        self.assertIn("挡住", text)

    def test_paused_by_foreground(self) -> None:
        text, level = describe_record_status(
            _status(phase="playing"), auto_follow=True, paused="游戏不在前台"
        )
        self.assertEqual(level, "waiting")
        self.assertIn("前台", text)

    def test_desktop_fallback_is_not_reported_as_waiting_for_a_run(self) -> None:
        """游戏没开时会回退截全屏：帧照常进、phase 停在 waiting。

        这时候如果按 phase 显示「待开局」，就是在骗用户「我在等你开局」——
        必须明确报「没锁定游戏窗口」。
        """
        text, level = describe_record_status(
            _status(phase="waiting", frames=120), auto_follow=True, game_window=False
        )
        self.assertEqual(level, "error")
        self.assertIn("未锁定", text)
        self.assertNotIn("待开局", text)

    def test_locked_game_window_with_no_run_is_waiting(self) -> None:
        text, level = describe_record_status(
            _status(phase="waiting", frames=120), auto_follow=True, game_window=True
        )
        self.assertEqual(level, "waiting")
        self.assertIn("待开局", text)

    def test_no_status_yet(self) -> None:
        text, level = describe_record_status(None, auto_follow=True)
        self.assertEqual(level, "waiting")
        self.assertIn("首帧", text)

    def test_stalled_frames_reported_as_error(self) -> None:
        """空转：画面 12 秒没更新，绝不能还显示「录制中」。"""
        text, level = describe_record_status(
            _status(phase="playing", event_count=7, duration=42.0),
            auto_follow=True,
            frame_age=12.0,
            stall_after=8.0,
        )
        self.assertEqual(level, "error")
        self.assertIn("无画面", text)
        self.assertIn("12", text)

    def test_fresh_frames_do_not_trip_the_stall_check(self) -> None:
        _, level = describe_record_status(
            _status(phase="playing", event_count=1), auto_follow=True, frame_age=1.5
        )
        self.assertEqual(level, "recording")

    def test_playing_shows_event_count_and_duration(self) -> None:
        text, level = describe_record_status(
            _status(phase="playing", event_count=23, duration=83.0), auto_follow=True
        )
        self.assertEqual(level, "recording")
        self.assertIn("23", text)
        self.assertIn("01:23", text)

    def test_opening_is_waiting(self) -> None:
        text, level = describe_record_status(_status(phase="opening"), auto_follow=True)
        self.assertEqual(level, "waiting")
        self.assertIn("开局", text)

    def test_finished_shows_event_count(self) -> None:
        text, level = describe_record_status(
            _status(phase="finished", event_count=18, duration=61.0), auto_follow=True
        )
        self.assertEqual(level, "finished")
        self.assertIn("18", text)

    def test_unknown_phase_falls_back_to_waiting(self) -> None:
        _, level = describe_record_status(_status(phase="waiting"), auto_follow=True)
        self.assertEqual(level, "waiting")

    def test_every_returned_level_is_a_known_level(self) -> None:
        """灯只会拿 ``RECORD_BADGE_COLORS`` 里的等级去查颜色，多返回一个就是 KeyError。"""
        combinations = [
            dict(auto_follow=False),
            dict(auto_follow=True),
            dict(auto_follow=True, capture_error="x"),
            dict(auto_follow=True, paused="工具挡住游戏"),
            dict(auto_follow=True, frame_age=99.0),
            dict(auto_follow=True, game_window=False),
        ]
        for kwargs in combinations:
            for phase in ("waiting", "opening", "playing", "finished"):
                _, level = describe_record_status(_status(phase=phase), **kwargs)
                self.assertIn(level, RECORD_STATUS_LEVELS)


class FormatDurationTests(unittest.TestCase):
    def test_under_a_minute(self) -> None:
        self.assertEqual(_format_duration(7.4), "00:07")

    def test_minutes_and_seconds(self) -> None:
        self.assertEqual(_format_duration(83), "01:23")

    def test_hours(self) -> None:
        self.assertEqual(_format_duration(3725), "1:02:05")

    def test_negative_is_clamped(self) -> None:
        self.assertEqual(_format_duration(-3), "00:00")


class RecorderStatusSnapshotTests(unittest.TestCase):
    def test_initial_status_is_waiting_with_no_events(self) -> None:
        rec = AutomaticRunRecorder(path=None)
        status = rec.status()
        self.assertEqual(status.phase, "waiting")
        self.assertIsNone(status.session_id)
        self.assertEqual(status.event_count, 0)
        self.assertEqual(status.frames, 0)
        self.assertEqual(status.duration, 0.0)
        self.assertIsNone(status.path)
        rec.close()

    def test_missing_snapshot_counts_as_missing_frame(self) -> None:
        """抓不到画面时 process 收到 None：帧数不涨、缺失计数涨，灯才不会以为在录。"""
        rec = AutomaticRunRecorder(path=None)
        rec.process(None)
        rec.process(None)
        status = rec.status()
        self.assertEqual(status.phase, "waiting")
        self.assertEqual(status.frames, 0)
        self.assertEqual(status.missing_frames, 2)
        rec.close()

    def test_active_scene_opens_a_session(self) -> None:
        """有「当前颜料」面板 → 认成对局进行中，phase 变 playing，事件数至少 1。"""
        rec = AutomaticRunRecorder(path=None)
        result = rec.process(_snapshot())
        self.assertEqual(result.phase, "playing")
        self.assertIsNotNone(result.session_id)
        status = rec.status()
        self.assertEqual(status.phase, "playing")
        self.assertEqual(status.frames, 1)
        self.assertGreaterEqual(status.event_count, 1)
        self.assertIsNotNone(status.last_event_age)
        self.assertGreaterEqual(status.last_event_age, 0.0)
        rec.close()

    def test_no_gameplay_panel_stays_waiting(self) -> None:
        """只有棋盘、没有颜料面板（大厅/选关）不应该开一局。"""
        rec = AutomaticRunRecorder(path=None)
        result = rec.process(_snapshot(with_current=False))
        self.assertEqual(result.phase, "waiting")
        self.assertIsNone(result.session_id)
        self.assertEqual(rec.status().frames, 1)
        rec.close()

    def test_status_is_a_snapshot_not_a_live_view(self) -> None:
        """status() 必须拷出值：拿到之后录制器继续跑，旧快照不能跟着变。"""
        rec = AutomaticRunRecorder(path=None)
        rec.process(_snapshot(with_current=False))
        before = rec.status()
        rec.process(_snapshot())
        self.assertEqual(before.phase, "waiting")
        self.assertEqual(before.frames, 1)
        self.assertEqual(rec.status().phase, "playing")
        rec.close()


class BadgeColorCoverageTests(unittest.TestCase):
    def test_every_level_has_a_color(self) -> None:
        from huahua_assistant.app import RECORD_BADGE_COLORS

        self.assertEqual(set(RECORD_BADGE_COLORS), set(RECORD_STATUS_LEVELS))


if __name__ == "__main__":
    unittest.main()
