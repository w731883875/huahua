from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .capture import CaptureResult, capture_target
from .fast_vision import FastVisionSnapshot, snapshot_from_scene
from .models import SceneObservation


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
    ) -> None:
        self._capture_func = capture_func
        self._analyze_func = analyze_func
        self._needs_full_refresh = needs_full_refresh
        self._full_analyze_func = full_analyze_func
        self._maybe_save_frame = maybe_save_frame
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
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
        if wait and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))

    def pop_latest(self, last_sequence: int = 0) -> FramePumpResult | None:
        with self._lock:
            latest = self._latest
        if latest is None or latest.sequence <= last_sequence:
            return None
        return latest

    def _run(self) -> None:
        while not self._stop_event.is_set():
            loop_started_at = time.perf_counter()
            result = self._capture_and_analyze()
            with self._lock:
                self._latest = result
            elapsed = time.perf_counter() - loop_started_at
            remaining = self._interval - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _capture_and_analyze(self) -> FramePumpResult:
        self._sequence += 1
        started_at = time.perf_counter()
        capture_elapsed = 0.0
        analysis_elapsed = 0.0
        write_elapsed = 0.0
        full_refresh_elapsed = 0.0
        try:
            capture_started_at = time.perf_counter()
            capture = self._capture_func()
            capture_elapsed = time.perf_counter() - capture_started_at
            frame = capture.image
            if self._maybe_save_frame is not None:
                write_elapsed = self._maybe_save_frame(frame)
            analysis_started_at = time.perf_counter()
            scene = self._analyze_func(frame)
            analysis_elapsed = time.perf_counter() - analysis_started_at
            if self._needs_full_refresh():
                full_started_at = time.perf_counter()
                scene = self._full_analyze_func(frame)
                full_refresh_elapsed = time.perf_counter() - full_started_at
            return FramePumpResult(
                sequence=self._sequence,
                capture=capture,
                frame_shape=tuple(int(item) for item in frame.shape),
                scene=scene,
                snapshot=snapshot_from_scene(scene),
                elapsed=time.perf_counter() - started_at,
                perf={
                    "capture": round(capture_elapsed, 3),
                    "analysis": round(analysis_elapsed, 3),
                    "write": round(write_elapsed, 3),
                    "full_refresh": round(full_refresh_elapsed, 3),
                    "pump_interval": round(self._interval, 3),
                },
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
                    "full_refresh": round(full_refresh_elapsed, 3),
                    "pump_interval": round(self._interval, 3),
                },
                error=str(exc),
            )


def default_auto_capture() -> CaptureResult:
    return capture_target(
        fallback_to_screen=True,
        fallback_if_target_uncaptureable=False,
        activate=True,
        activation_delay=0.02,
    )
