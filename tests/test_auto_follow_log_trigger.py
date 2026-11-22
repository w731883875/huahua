"""日志事件 → 采集节拍的接线。

不在这里测 Tk，只测「拿到事件之后该做什么」：立刻转连拍、等动画走完再叫醒、
没有事件就退回慢节拍、日志不可用就保持用户设的原节拍。
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from huahua_assistant import app as app_module
from huahua_assistant.app import (
    LOG_BURST_INTERVAL,
    LOG_IDLE_INTERVAL,
    LOG_SETTLE_MS,
    HuahuaApp,
)
from huahua_assistant.game_log import GameLogEvent


class _FakePump:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.wakes = 0
        self.intervals: list[float] = []

    def set_interval(self, interval: float) -> None:
        self.interval = interval
        self.intervals.append(interval)

    def wake(self) -> None:
        self.wakes += 1


class _FakeWatcher:
    def __init__(self, events: list[GameLogEvent], *, running: bool = True) -> None:
        self._events = list(events)
        self.running = running
        self.revision = 1 if events else 0
        # 静默判定要用：日志最近一次写入距今多久、可识别事件距今多久。
        # 默认都当成「刚刚」——也就是不静默，和加这两个判定之前的行为一致。
        self.line_age = 0.0
        self.idle_seconds = 0.0 if events else float("inf")
        # 「日志在写却没有可识别事件」已经持续多久。0 表示不静默。
        self.silent_seconds = 0.0
        self.lines_seen = 0

    def drain(self) -> list[GameLogEvent]:
        events, self._events = self._events, []
        return events


class _FakeVar:
    def __init__(self, value: bool = True) -> None:
        self._value = value

    def get(self) -> bool:
        return self._value


class _FakeStringVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


def _event(kind: str) -> GameLogEvent:
    return GameLogEvent(name=f"S8_HZXY_DrawPaper_{kind}", kind=kind, at=time.monotonic(), game_time=None)


#: 事件落盘是排查用的旁路，测试里不该真写 runtime/observations.jsonl。
_RECORDER: object = None


def setUpModule() -> None:
    global _RECORDER
    _RECORDER = patch.object(app_module, "record_observation").start()


def tearDownModule() -> None:
    patch.stopall()


def _app(
    *,
    watcher: _FakeWatcher | None,
    pump: _FakePump | None,
    base_interval_ms: int = 100,
    auto_follow: bool = True,
) -> SimpleNamespace:
    scheduled: list[tuple[int, object]] = []
    _RECORDER.reset_mock()
    stub = SimpleNamespace(
        game_log_watcher=watcher,
        auto_follow_pump=pump,
        auto_follow_var=_FakeVar(auto_follow),
        auto_follow_interval_var=SimpleNamespace(get=lambda: base_interval_ms),
        auto_follow_burst_until=0.0,
        auto_follow_last_event_at=0.0,
        auto_follow_log_revision=0,
        last_game_log_events=[],
        game_log_event_count=0,
        game_log_last_event=None,
        game_log_nudge_at=0.0,
        status_var=_FakeStringVar(),
        game_log_var=_FakeStringVar(),
        notice="",
        # ``_consume_game_log_events`` 会先过一遍等级动画窗口；真实 ``__init__``
        # 默认就是 None，这里补齐以免属性缺失。
        level_animation_window=None,
        level_reconcile=SimpleNamespace(ran=False),
        # 日志「在写但认不出」的静默状态（见 _note_log_silence）。默认为未静默。
        auto_follow_log_silent=False,
        auto_follow_silence_notice_key=False,
        after=lambda delay, callback: scheduled.append((delay, callback)),
    )
    # 被调用的那几个方法本身也挂在 self 上，这里手动绑一下。
    stub._wake_auto_follow_pump = lambda: HuahuaApp._wake_auto_follow_pump(stub)
    stub._auto_follow_base_interval = lambda: HuahuaApp._auto_follow_base_interval(stub)
    stub._auto_follow_idle_interval = lambda: HuahuaApp._auto_follow_idle_interval(stub)
    stub._auto_follow_fallback_interval = lambda: HuahuaApp._auto_follow_fallback_interval(stub)
    stub._log_is_silent = lambda watcher: HuahuaApp._log_is_silent(stub, watcher)
    stub._note_log_silence = lambda watcher, now: HuahuaApp._note_log_silence(stub, watcher, now)
    stub._report_log_silence = lambda watcher: HuahuaApp._report_log_silence(stub, watcher)
    stub._report_log_resumed = lambda watcher: HuahuaApp._report_log_resumed(stub, watcher)
    stub._sync_game_log_status = lambda watcher=None: HuahuaApp._sync_game_log_status(stub, watcher)
    stub._nudge_auto_follow_after_event = lambda event, now: HuahuaApp._nudge_auto_follow_after_event(
        stub, event, now
    )
    stub._note_level_animation_events = lambda events: HuahuaApp._note_level_animation_events(stub, events)
    stub._set_notice = lambda message, level="info": setattr(stub, "notice", message)
    stub.scheduled = scheduled
    return stub


class ConsumeEventTests(unittest.TestCase):
    def test_capture_event_switches_to_burst_and_schedules_a_wake(self) -> None:
        watcher = _FakeWatcher([_event("pigment")])
        pump = _FakePump()
        stub = _app(watcher=watcher, pump=pump)

        HuahuaApp._consume_game_log_events(stub)

        self.assertEqual(LOG_BURST_INTERVAL, pump.interval)
        self.assertEqual([LOG_SETTLE_MS], [delay for delay, _ in stub.scheduled])
        self.assertGreater(stub.auto_follow_burst_until, time.monotonic() - 0.1)
        self.assertEqual(["pigment"], [item.kind for item in stub.last_game_log_events])

    def test_settle_callback_wakes_the_pump(self) -> None:
        watcher = _FakeWatcher([_event("pigment")])
        pump = _FakePump()
        stub = _app(watcher=watcher, pump=pump)

        HuahuaApp._consume_game_log_events(stub)
        _delay, callback = stub.scheduled[0]
        callback()

        self.assertEqual(1, pump.wakes)

    def test_score_animation_also_triggers_a_capture(self) -> None:
        """结算数字滚动说明一局刚结束，正是最该读一帧的时候。"""

        watcher = _FakeWatcher([_event("score")])
        pump = _FakePump(interval=LOG_IDLE_INTERVAL)
        stub = _app(watcher=watcher, pump=pump)

        HuahuaApp._consume_game_log_events(stub)

        self.assertEqual(LOG_BURST_INTERVAL, pump.interval)
        self.assertEqual(1, len(stub.scheduled))

    def test_no_events_returns_to_idle_interval(self) -> None:
        watcher = _FakeWatcher([])
        pump = _FakePump(interval=LOG_BURST_INTERVAL)
        stub = _app(watcher=watcher, pump=pump)

        HuahuaApp._consume_game_log_events(stub)

        self.assertEqual(LOG_IDLE_INTERVAL, pump.interval)

    def test_burst_is_held_while_the_window_is_open(self) -> None:
        """连拍窗口内没有新事件也别急着降速，动画可能还没走完。"""

        watcher = _FakeWatcher([])
        pump = _FakePump(interval=LOG_BURST_INTERVAL)
        stub = _app(watcher=watcher, pump=pump)
        stub.auto_follow_burst_until = time.monotonic() + 5.0

        HuahuaApp._consume_game_log_events(stub)

        self.assertEqual(LOG_BURST_INTERVAL, pump.interval)

    def test_without_a_watcher_the_user_interval_is_kept(self) -> None:
        pump = _FakePump(interval=LOG_IDLE_INTERVAL)
        stub = _app(watcher=None, pump=pump, base_interval_ms=100)

        HuahuaApp._consume_game_log_events(stub)

        self.assertAlmostEqual(0.1, pump.interval)

    def test_a_dead_watcher_also_falls_back(self) -> None:
        pump = _FakePump(interval=LOG_IDLE_INTERVAL)
        stub = _app(watcher=_FakeWatcher([], running=False), pump=pump, base_interval_ms=150)

        HuahuaApp._consume_game_log_events(stub)

        self.assertAlmostEqual(0.15, pump.interval)

    def test_idle_interval_is_configurable(self) -> None:
        watcher = _FakeWatcher([])
        pump = _FakePump(interval=LOG_BURST_INTERVAL)
        stub = _app(watcher=watcher, pump=pump)

        with patch.object(app_module, "config_value", return_value=2.5):
            HuahuaApp._consume_game_log_events(stub)

        self.assertEqual(2.5, pump.interval)

    def test_no_pump_is_a_no_op(self) -> None:
        watcher = _FakeWatcher([_event("pigment")])
        stub = _app(watcher=watcher, pump=None)

        HuahuaApp._consume_game_log_events(stub)  # 不该抛

    def test_events_are_counted_and_recorded(self) -> None:
        """事件要留下痕迹：界面数得出来，runtime/observations.jsonl 也查得到。"""

        watcher = _FakeWatcher([_event("pigment"), _event("into")])
        stub = _app(watcher=watcher, pump=_FakePump())

        HuahuaApp._consume_game_log_events(stub)

        self.assertEqual(2, stub.game_log_event_count)
        self.assertEqual("into", stub.game_log_last_event.kind)
        # 两条事件各记一条 game_log_event，外加「入画」触发的新局信号。
        self.assertEqual(3, _RECORDER.call_count)
        self.assertEqual("game_log_event", _RECORDER.call_args_list[0].args[0])
        self.assertEqual("game_log_event", _RECORDER.call_args_list[1].args[0])
        self.assertEqual("new_run_pending", _RECORDER.call_args_list[2].args[0])
        self.assertEqual("日志事件 2", stub.game_log_var.value)

    def test_auto_follow_off_still_counts_and_nudges(self) -> None:
        """自动跟随没开时画面不该动，但「我看见了」这件事必须说出口。"""

        watcher = _FakeWatcher([_event("pigment")])
        stub = _app(watcher=watcher, pump=None, auto_follow=False)

        HuahuaApp._consume_game_log_events(stub)

        self.assertEqual(1, stub.game_log_event_count)
        self.assertIn("pigment", stub.notice)
        self.assertEqual([], stub.scheduled)  # 不该去叫醒不存在的采集线程

    def test_auto_follow_on_does_not_nudge(self) -> None:
        watcher = _FakeWatcher([_event("pigment")])
        stub = _app(watcher=watcher, pump=_FakePump())

        HuahuaApp._consume_game_log_events(stub)

        self.assertEqual("", stub.notice)

    def test_nudge_is_rate_limited(self) -> None:
        """连着落三颗颜料不该把提示刷三遍。"""

        watcher = _FakeWatcher([_event("pigment")])
        stub = _app(watcher=watcher, pump=None, auto_follow=False)
        HuahuaApp._consume_game_log_events(stub)
        first_notice = stub.notice

        stub.notice = ""
        HuahuaApp._consume_game_log_events(stub)

        self.assertNotEqual("", first_notice)
        self.assertEqual("", stub.notice)


class LogSilenceTests(unittest.TestCase):
    """日志在写、却解析不出事件时的兜底。

    这是 2026-11-22 实测出来的真实缺陷：第二局用户大量安装颜料，但那些颜料落在
    初始颜料生效范围外、任务没完成，游戏只播通用按钮音 ``ECommonBtnType``，
    一条 ``S8_*`` 都没有。于是日志驱动整段沉默，64 秒里零帧。
    现存证据：那段窗口日志写了 147 行，``parse_line`` 命中 0 条。
    """

    def test_silent_when_log_is_written_but_nothing_parses(self) -> None:
        watcher = _FakeWatcher([], running=True)
        watcher.line_age = 0.5           # 日志刚写过
        watcher.idle_seconds = 60.0      # 但 60 秒没有可识别事件
        watcher.silent_seconds = 60.0
        stub = _app(watcher=watcher, pump=_FakePump())

        self.assertTrue(HuahuaApp._log_is_silent(stub, watcher))

    def test_not_silent_when_the_game_stopped_writing(self) -> None:
        """游戏关掉/停写时不能算静默，否则会在没人玩的时候白截。"""

        watcher = _FakeWatcher([], running=True)
        watcher.line_age = 300.0
        watcher.idle_seconds = 300.0
        watcher.silent_seconds = 0.0
        stub = _app(watcher=watcher, pump=_FakePump())

        self.assertFalse(HuahuaApp._log_is_silent(stub, watcher))

    def test_not_silent_right_after_an_event(self) -> None:
        watcher = _FakeWatcher([], running=True)
        watcher.line_age = 0.5
        watcher.idle_seconds = 1.0
        watcher.silent_seconds = 0.0
        stub = _app(watcher=watcher, pump=_FakePump())

        self.assertFalse(HuahuaApp._log_is_silent(stub, watcher))

    def test_pump_falls_back_to_slower_cadence_when_silent(self) -> None:
        watcher = _FakeWatcher([], running=True)
        watcher.line_age = 0.5
        watcher.idle_seconds = 60.0
        watcher.silent_seconds = 60.0
        pump = _FakePump()
        stub = _app(watcher=watcher, pump=pump)

        HuahuaApp._consume_game_log_events(stub)

        # 静默时用兜底节拍；不静默时是 idle 节拍（1.5s 默认）。
        self.assertEqual(app_module.LOG_FALLBACK_INTERVAL, pump.intervals[-1])
        self.assertTrue(stub.auto_follow_log_silent)

    def test_silence_is_surfaced_to_the_user_not_only_logged(self) -> None:
        """静默必须能在界面上看见——以前这种情况界面上一切正常。"""

        watcher = _FakeWatcher([], running=True)
        watcher.line_age = 0.5
        watcher.idle_seconds = 60.0
        watcher.silent_seconds = 60.0
        stub = _app(watcher=watcher, pump=_FakePump())

        HuahuaApp._consume_game_log_events(stub)

        self.assertIn("兜底节拍", stub.notice)
        self.assertIn("兜底节拍", stub.status_var.value)

    def test_silence_notice_is_not_repeated_every_tick(self) -> None:
        watcher = _FakeWatcher([], running=True)
        watcher.line_age = 0.5
        watcher.idle_seconds = 60.0
        watcher.silent_seconds = 60.0
        stub = _app(watcher=watcher, pump=_FakePump())

        HuahuaApp._consume_game_log_events(stub)
        first = stub.notice
        stub.notice = ""
        HuahuaApp._consume_game_log_events(stub)

        self.assertNotEqual("", first)
        self.assertEqual("", stub.notice)

    def test_silence_flag_clears_when_events_resume(self) -> None:
        watcher = _FakeWatcher([], running=True)
        watcher.line_age = 0.5
        watcher.idle_seconds = 60.0
        watcher.silent_seconds = 60.0
        stub = _app(watcher=watcher, pump=_FakePump())
        HuahuaApp._consume_game_log_events(stub)
        self.assertTrue(stub.auto_follow_log_silent)

        # 事件恢复：换一个有事件的 watcher。
        resumed = _FakeWatcher([_event("pigment")], running=True)
        stub.game_log_watcher = resumed
        HuahuaApp._consume_game_log_events(stub)

        self.assertFalse(stub.auto_follow_log_silent)


class WakeTests(unittest.TestCase):
    def test_wake_only_when_auto_follow_is_on(self) -> None:
        pump = _FakePump()
        stub = _app(watcher=None, pump=pump)
        stub.auto_follow_var = _FakeVar(False)

        HuahuaApp._wake_auto_follow_pump(stub)

        self.assertEqual(0, pump.wakes)


if __name__ == "__main__":
    unittest.main()
