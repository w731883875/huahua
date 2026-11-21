"""采集线程的「按需唤醒」。

日志说画面变了，采集线程要立刻跑下一帧，而不是把剩下的节拍等完。原来它只
``stop_event.wait(remaining)``，只能被停止唤醒；现在多了一个 wake 事件。
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import numpy as np

from huahua_assistant import frame_pump
from huahua_assistant.capture import CaptureResult
from huahua_assistant.frame_pump import FramePump


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> CaptureResult:
        self.calls += 1
        return CaptureResult(image=np.zeros((8, 8, 3), dtype=np.uint8), source="test")


def _pump(counter: _Counter, interval: float) -> FramePump:
    pump = FramePump(
        capture_func=counter,
        analyze_func=lambda _frame: None,
        needs_full_refresh=lambda: False,
        full_analyze_func=lambda _frame: None,
    )
    pump.start(interval=interval)
    return pump


class WakeTests(unittest.TestCase):
    def test_wake_cuts_the_interval_short(self) -> None:
        counter = _Counter()
        pump = _pump(counter, 5.0)
        try:
            self._wait(counter, 1)
            before = counter.calls
            started = time.perf_counter()
            pump.wake()
            self._wait(counter, before + 1, timeout=1.0)
            self.assertLess(time.perf_counter() - started, 1.0, "wake 之后不该等满 5 秒")
        finally:
            pump.stop(wait=True)

    def test_interval_change_alone_does_not_trigger(self) -> None:
        """改节拍不代表画面变了，不该白跑一帧。"""

        counter = _Counter()
        pump = _pump(counter, 5.0)
        try:
            self._wait(counter, 1)
            before = counter.calls
            pump.set_interval(0.1)
            time.sleep(0.4)
            self.assertEqual(before, counter.calls)
        finally:
            pump.stop(wait=True)

    def test_interval_is_floored(self) -> None:
        counter = _Counter()
        pump = _pump(counter, 1.0)
        try:
            pump.set_interval(0.0)
            self.assertGreaterEqual(pump.interval, 0.1)
        finally:
            pump.stop(wait=True)

    def test_stop_releases_a_sleeping_pump(self) -> None:
        counter = _Counter()
        pump = _pump(counter, 30.0)
        self._wait(counter, 1)
        started = time.perf_counter()
        pump.stop(wait=True)
        self.assertLess(time.perf_counter() - started, 1.0, "stop 必须能把睡着的那一轮叫醒")

    @staticmethod
    def _wait(counter: _Counter, target: int, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and counter.calls < target:
            time.sleep(0.01)


class SlowFrameReportTests(unittest.TestCase):
    """慢帧要把分段耗时交出去。

    用户报「开局识别卡十秒」时，日志里只有 tick 侧那句 ``auto_follow_restart``
    （泵 12 秒没出帧），分不清是抓图慢还是识别慢。这个回调就是补这条证据的。
    """

    def test_slow_frame_is_reported_with_breakdown(self) -> None:
        reports: list[dict] = []

        def slow_analyze(_frame):
            time.sleep(0.05)
            return None

        pump = FramePump(
            capture_func=_Counter(),
            analyze_func=slow_analyze,
            needs_full_refresh=lambda: False,
            full_analyze_func=lambda _frame: None,
            slow_frame_func=reports.append,
        )
        try:
            # 阈值打桩到 10ms，否则要真跑 3 秒。
            with patch.object(frame_pump, "SLOW_FRAME_SECONDS", 0.01):
                pump.start(interval=0.1)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and not reports:
                    time.sleep(0.01)
                pump.stop(wait=True)
        finally:
            pump.stop(wait=True)
        self.assertTrue(reports, "帧耗时超过阈值时必须上报")
        payload = reports[0]
        self.assertLessEqual({"kind", "elapsed", "capture", "analysis"}, set(payload))
        self.assertEqual("slow_frame", payload["kind"])

    def test_fast_frames_are_not_reported(self) -> None:
        reports: list[dict] = []
        pump = FramePump(
            capture_func=_Counter(),
            analyze_func=lambda _frame: None,
            needs_full_refresh=lambda: False,
            full_analyze_func=lambda _frame: None,
            slow_frame_func=reports.append,
        )
        try:
            pump.start(interval=0.1)
            time.sleep(0.3)
        finally:
            pump.stop(wait=True)
        self.assertEqual([], reports, "正常帧不该刷诊断日志")


if __name__ == "__main__":
    unittest.main()
