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

    def set_interval(self, interval: float) -> None:
        self.interval = interval

    def wake(self) -> None:
        self.wakes += 1


class _FakeWatcher:
    def __init__(self, events: list[GameLogEvent], *, running: bool = True) -> None:
        self._events = list(events)
        self.running = running
        self.revision = 1 if events else 0

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
        after=lambda delay, callback: scheduled.append((delay, callback)),
    )
    # 被调用的那几个方法本身也挂在 self 上，这里手动绑一下。
    stub._wake_auto_follow_pump = lambda: HuahuaApp._wake_auto_follow_pump(stub)
    stub._auto_follow_base_interval = lambda: HuahuaApp._auto_follow_base_interval(stub)
    stub._auto_follow_idle_interval = lambda: HuahuaApp._auto_follow_idle_interval(stub)
    stub._sync_game_log_status = lambda watcher=None: HuahuaApp._sync_game_log_status(stub, watcher)
    stub._nudge_auto_follow_after_event = lambda event, now: HuahuaApp._nudge_auto_follow_after_event(
        stub, event, now
    )
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


class WakeTests(unittest.TestCase):
    def test_wake_only_when_auto_follow_is_on(self) -> None:
        pump = _FakePump()
        stub = _app(watcher=None, pump=pump)
        stub.auto_follow_var = _FakeVar(False)

        HuahuaApp._wake_auto_follow_pump(stub)

        self.assertEqual(0, pump.wakes)


if __name__ == "__main__":
    unittest.main()
