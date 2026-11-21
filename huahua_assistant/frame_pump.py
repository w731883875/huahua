from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .capture import (
    CaptureResult,
    WindowInfo,
    capture_target,
    capture_window_content,
    find_capture_window,
)
from .fast_vision import FastVisionSnapshot, snapshot_from_scene
from .models import SceneObservation


# 自动跟随暂停时用的来源标记。上层看到它就不刷新任何状态。
PAUSED_SOURCE = "paused:game-not-foreground"
#: 找不到可抓的游戏窗口（没开、最小化、尺寸异常）。和上面的区别只是原因：
#: 前者是"游戏在但不在前台"，这个是"根本没有可抓的游戏"。
PAUSED_WINDOW_MISSING = "paused:game-window-missing"


@dataclass(frozen=True)
class FramePumpResult:
    sequence: int
    capture: CaptureResult
    frame_shape: tuple[int, int, int] | None
    scene: SceneObservation | None
    snapshot: FastVisionSnapshot | None
    elapsed: float
    perf: dict
    error: str | None = None
    #: 这一帧**从游戏里取下来的时刻**（``time.monotonic()``）。建议的有效期、
    #: 日志对齐、"这一帧到底多旧"都从这里算——不能用界面重绘的时刻，
    #: 否则同一帧每重绘一次就被续命一次。
    captured_at: float = 0.0


#: 一帧跑超过这个秒数就记一条 ``slow_frame``。
#: 自动跟随的节拍是 1.5s，正常帧 0.3~0.8s；超过 3s 才会被 tick 判成「很久没出帧」。
SLOW_FRAME_SECONDS = 3.0


class FramePump:
    """Background realtime loop that keeps only the newest analyzed frame."""

    def __init__(
        self,
        *,
        capture_func: Callable[[], CaptureResult],
        analyze_func: Callable[[np.ndarray], SceneObservation | None],
        needs_full_refresh: Callable[[], bool],
        full_analyze_func: Callable[[np.ndarray], SceneObservation | None],
        maybe_save_frame: Callable[[np.ndarray], float] | None = None,
        record_func: Callable[[FastVisionSnapshot | None, CaptureResult], object] | None = None,
        model_state_func: Callable[[], object] | None = None,
        slow_frame_func: Callable[[dict], None] | None = None,
    ) -> None:
        self._capture_func = capture_func
        self._analyze_func = analyze_func
        self._needs_full_refresh = needs_full_refresh
        self._full_analyze_func = full_analyze_func
        self._maybe_save_frame = maybe_save_frame
        self._record_func = record_func
        #: 规则推演出的模型状态（``game_model.PuzzleState``）。快照里的
        #: ``bonus_level`` / ``aura_level`` 只能从这里回填，视觉读不到。
        self._model_state_func = model_state_func
        #: 慢帧回调：``elapsed`` 超过 :data:`SLOW_FRAME_SECONDS` 时收到分段耗时。
        self._slow_frame_func = slow_frame_func
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        #: 提前唤醒正在等间隔的循环。日志说画面变了就置位，不必等满一个节拍。
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: FramePumpResult | None = None
        self._sequence = 0
        self._interval = 0.5

    def start(self, *, interval: float = 0.5) -> None:
        self._interval = max(0.1, float(interval))
        self._stop_event.clear()
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="HuahuaFramePump", daemon=True)
        self._thread.start()

    def stop(self, *, wait: bool = False, timeout: float = 0.6) -> None:
        self._stop_event.set()
        self._wake.set()
        if wait and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))

    @property
    def interval(self) -> float:
        return self._interval

    def set_interval(self, interval: float) -> None:
        """改节拍。不主动唤醒——节拍变化本身不代表画面变了。"""

        self._interval = max(0.1, float(interval))

    def wake(self) -> None:
        """让循环立刻跑下一帧，而不是等完剩下的间隔。"""

        self._wake.set()

    def pop_latest(self, last_sequence: int = 0) -> FramePumpResult | None:
        with self._lock:
            latest = self._latest
        if latest is None or latest.sequence <= last_sequence:
            return None
        return latest

    def _model_state(self) -> object | None:
        if self._model_state_func is None:
            return None
        try:
            return self._model_state_func()
        except Exception:
            # 模型状态只是快照的附加信息，读不到不该让整帧失败。
            return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._wake.clear()
            loop_started_at = time.perf_counter()
            result = self._capture_and_analyze()
            with self._lock:
                self._latest = result
            elapsed = time.perf_counter() - loop_started_at
            if elapsed >= SLOW_FRAME_SECONDS:
                self._report_slow_frame(result, elapsed)
            remaining = self._interval - elapsed
            if remaining > 0:
                # 等间隔，或者被 wake() 提前叫醒（日志说画面变了）。
                self._wake.wait(remaining)

    def _report_slow_frame(self, result: FramePumpResult, elapsed: float) -> None:
        """把慢帧的分段耗时交出去。

        没有这条，「开局识别卡十秒」只能看到 tick 侧那句 ``auto_follow_restart``：
        到底是抓图慢（WGC 在游戏切场景时不来帧）还是识别慢，日志里分不清。
        """

        if self._slow_frame_func is None:
            return
        try:
            self._slow_frame_func(
                {
                    "kind": "slow_frame",
                    "elapsed": round(elapsed, 2),
                    "sequence": result.sequence,
                    "frame_shape": None if result.frame_shape is None else list(result.frame_shape),
                    "error": result.error,
                    **result.perf,
                }
            )
        except Exception:  # noqa: BLE001 - 诊断失败不能影响采集
            pass

    def _capture_and_analyze(self) -> FramePumpResult:
        self._sequence += 1
        started_at = time.perf_counter()
        capture_elapsed = 0.0
        analysis_elapsed = 0.0
        write_elapsed = 0.0
        full_refresh_elapsed = 0.0
        record_elapsed = 0.0
        record_phase: str | None = None
        record_events: tuple[str, ...] = ()
        #: 写盘闸门的判定结果。``None`` 表示没拦；否则是拦截原因（进 perf 便于排查）。
        write_gate: str | None = None
        try:
            capture_started_at = time.perf_counter()
            capture = self._capture_func()
            captured_at = time.monotonic()
            capture_elapsed = time.perf_counter() - capture_started_at
            frame = capture.image
            # 只有窗口自己渲染的像素才允许自动落盘。整屏/桌面区域里可能是玩家的
            # 浏览器、聊天窗口、桌面文件——自动写进 runtime/ 就是把别人的画面
            # 留在磁盘上，而这一帧根本不该被当成游戏画面。
            if self._maybe_save_frame is not None:
                if capture.trusted_for_automation:
                    write_elapsed = self._maybe_save_frame(frame)
                else:
                    write_gate = f"untrusted_source:{capture.kind}"
            analysis_started_at = time.perf_counter()
            scene = self._analyze_func(frame)
            analysis_elapsed = time.perf_counter() - analysis_started_at
            if self._needs_full_refresh():
                full_started_at = time.perf_counter()
                scene = self._full_analyze_func(frame)
                full_refresh_elapsed = time.perf_counter() - full_started_at
            snapshot = snapshot_from_scene(scene, self._model_state())
            if self._record_func is not None:
                record_started_at = time.perf_counter()
                record_result = self._record_func(snapshot, capture)
                record_elapsed = time.perf_counter() - record_started_at
                record_phase = getattr(record_result, "phase", None)
                record_events = tuple(getattr(record_result, "events", ()) or ())
            return FramePumpResult(
                sequence=self._sequence,
                capture=capture,
                frame_shape=tuple(int(item) for item in frame.shape),
                scene=scene,
                snapshot=snapshot,
                elapsed=time.perf_counter() - started_at,
                perf={
                    "capture": round(capture_elapsed, 3),
                    "analysis": round(analysis_elapsed, 3),
                    "write": round(write_elapsed, 3),
                    "write_gate": write_gate,
                    "full_refresh": round(full_refresh_elapsed, 3),
                    "record": round(record_elapsed, 4),
                    "record_phase": record_phase,
                    "record_events": record_events,
                    "under_200ms": (time.perf_counter() - started_at) <= 0.2,
                    "pump_interval": round(self._interval, 3),
                },
                captured_at=captured_at,
            )
        except Exception as exc:
            return FramePumpResult(
                sequence=self._sequence,
                capture=CaptureResult(image=np.zeros((1, 1, 3), dtype=np.uint8), source="frame-pump:error"),
                frame_shape=None,
                scene=None,
                snapshot=None,
                elapsed=time.perf_counter() - started_at,
                perf={
                    "capture": round(capture_elapsed, 3),
                    "analysis": round(analysis_elapsed, 3),
                    "write": round(write_elapsed, 3),
                    "write_gate": write_gate,
                    "full_refresh": round(full_refresh_elapsed, 3),
                    "record": round(record_elapsed, 4),
                    "record_phase": record_phase,
                    "record_events": record_events,
                    "under_200ms": False,
                    "pump_interval": round(self._interval, 3),
                },
                error=str(exc),
                captured_at=time.monotonic(),
            )


def default_auto_capture() -> CaptureResult:
    """自动跟随的截图入口。

    这里**绝不能** ``activate=True``。自动跟随每 100ms 跑一次，每帧都把游戏窗口
    拽到前台：用户一切到浏览器/聊天窗口就被游戏抢回焦点，根本切不过去；
    游戏被反复「后台 → 前台」激活，画面还会一黑一闪，像是每帧都在截图。

    优先走窗口内容抓取（Windows.Graphics.Capture）：它读的是窗口自己渲染的表面，
    别的窗口盖在上面**不会**进帧，所以工具窗口挡着游戏也照样能读，不需要暂停。
    只有窗口内容抓不到时，才退回「只截本来就在前台的游戏」这条老路。

    **找不到可抓的窗口就等，不截整屏。** 以前这里会一路回退到
    ``screen:fallback``：游戏关掉或最小化之后，自动跟随仍然每 100ms 抓一张
    桌面图像交给识别和托管——玩家看到的是"箭头在指它自己的浏览器"。
    """
    window = find_capture_window()
    if window is None:
        return CaptureResult(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            source=PAUSED_WINDOW_MISSING,
            window=None,
        )

    content, method = capture_window_content(window)
    if content is not None:
        return CaptureResult(
            image=content,
            source=(
                f"window-content:{window.title} process:{window.process_name} "
                f"method:{method}"
            ),
            window=window,
        )

    if not _foreground_allows_capture(window):
        return CaptureResult(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            source=PAUSED_SOURCE,
            window=window,
        )
    return capture_target(
        fallback_to_screen=True,
        fallback_if_target_uncaptureable=False,
        activate=False,
        activation_delay=0.0,
    )


def _foreground_allows_capture(window: WindowInfo) -> bool:
    """前台窗口是游戏、或本工具自己时，才值得截图。

    工具自己在前台是常态（用户在看面板），不能因此停掉识别；
    只有前台是「别的软件」才说明用户走开了，这时候截图既没意义又会抢焦点。
    """
    try:
        user32 = ctypes.windll.user32
        hwnd = int(user32.GetForegroundWindow())
    except Exception:
        return True
    if hwnd == 0 or hwnd == int(window.hwnd):
        return True
    try:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        foreground_pid = int(pid.value)
    except Exception:
        return True
    return foreground_pid in (int(window.process_id), os.getpid())
