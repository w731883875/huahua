"""采集线程的「按需唤醒」。

日志说画面变了，采集线程要立刻跑下一帧，而不是把剩下的节拍等完。原来它只
``stop_event.wait(remaining)``，只能被停止唤醒；现在多了一个 wake 事件。
"""

from __future__ import annotations

import time
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
