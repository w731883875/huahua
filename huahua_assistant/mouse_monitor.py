from __future__ import annotations

import ctypes
import os
import threading
import time
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass


WM_LBUTTONDOWN = 0x0201
WM_QUIT = 0x0012
WH_MOUSE_LL = 14


@dataclass(frozen=True)
class MouseClick:
    ts: float
    x: int
    y: int
    window_geometry: tuple[int, int, int, int, int] | None = None
    source: str = "hook"


class _Point(ctypes.Structure):
    _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))


class _MouseHookData(ctypes.Structure):
    _fields_ = (
        ("pt", _Point),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    )


def _click_window_geometry(user32):
    from .capture import physical_pixel_coordinates, _visible_bounds
    hwnd = int(user32.GetForegroundWindow() or 0)
    if not hwnd:
        return None
    rect = wintypes.RECT()
    with physical_pixel_coordinates():
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        # Capture uses DWM visible bounds; GetWindowRect alone includes the
        # invisible resize grip and would reject every click in windowed mode.
        rect = _visible_bounds(hwnd, rect)
        return (hwnd, rect.left, rect.top, rect.right-rect.left, rect.bottom-rect.top)


class MouseClickMonitor:
    """Capture click coordinates; consumers discard clicks outside game controls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clicks: deque[MouseClick] = deque(maxlen=128)
        self._history: deque[MouseClick] = deque(maxlen=256)
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._hook = None
        self._callback = None
        self._ready = threading.Event()
        self.last_error = 0
        self._stop = threading.Event()
        self._poll_thread = None
        self._counts = {"hook": 0, "poll": 0, "duplicate": 0, "error": 0}
        self.integrity = None

    def start(self) -> None:
        if os.name != "nt" or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._ready.clear()
        self.last_error = 0
        from .wgc_probe import process_integrity
        self.integrity = process_integrity()
        self._thread = threading.Thread(target=self._run, name="HuahuaMouseClickMonitor", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=1.0)
        if self._poll_thread is None or not self._poll_thread.is_alive():
            self._poll_thread = threading.Thread(target=self._poll, name="HuahuaMouseFallback", daemon=True)
            self._poll_thread.start()

    def _record(self, click):
        with self._lock:
            self._counts[click.source] += 1
            previous = self._history[-1] if self._history else None
            # Two independent input feeds report the same physical press. Never
            # collapse separate presses from the same feed (e.g. double-click).
            if (previous and previous.source != click.source and abs(click.ts-previous.ts) < .06
                    and abs(click.x-previous.x) <= 3 and abs(click.y-previous.y) <= 3
                    and click.window_geometry == previous.window_geometry):
                self._counts["duplicate"] += 1
                return
            self._clicks.append(click)
            self._history.append(click)

    def diagnostics(self):
        with self._lock:
            return {**self._counts, "hook_installed": bool(self._hook), "hook_error": self.last_error,
                    "poll_running": bool(self._poll_thread and self._poll_thread.is_alive()),
                    "integrity": self.integrity, "warning": self.warning,
                    "last_click_at": self._history[-1].ts if self._history else None}

    @property
    def warning(self):
        rid = (self.integrity or {}).get("rid")
        if rid is not None and rid < 0x2000:
            return "input_low_integrity"
        if self.last_error:
            return "input_hook_failed"
        return None

    def _poll(self):
        # Windows can silently remove a low-level hook after a slow callback.
        # The high (currently-down) bit provides an independent fallback; the
        # unreliable process-shared 'pressed since last call' bit is not used.
        from .capture import physical_pixel_coordinates
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
        user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
        user32.GetAsyncKeyState.restype = ctypes.c_short
        down = bool(user32.GetAsyncKeyState(1) & 0x8000)
        while not self._stop.wait(.008):
            try:
                pressed = bool(user32.GetAsyncKeyState(1) & 0x8000)
                if pressed and not down:
                    ts = time.time()
                    point = wintypes.POINT()
                    with physical_pixel_coordinates():
                        if user32.GetCursorPos(ctypes.byref(point)):
                            geometry = _click_window_geometry(user32)
                            self._record(MouseClick(ts, point.x, point.y, geometry, "poll"))
                down = pressed
            except Exception:
                self._counts["error"] += 1

    def drain(self, *, max_age: float = 2.0) -> tuple[MouseClick, ...]:
        cutoff = time.time() - max(0.1, float(max_age))
        with self._lock:
            while self._clicks and self._clicks[0].ts < cutoff:
                self._clicks.popleft()
            result = tuple(self._clicks)
            self._clicks.clear()
        return result

    def close(self) -> None:
        self._stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=.5)
        if os.name != "nt" or not self._thread_id:
            return
        try:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        except Exception:
            return
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def since(self, timestamp: float) -> tuple[MouseClick, ...]:
        """Independent read cursor for card recording; never drains board clicks."""
        with self._lock:
            return tuple(click for click in self._history if click.ts > timestamp)

    def _run(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._thread_id = int(kernel32.GetCurrentThreadId())
        result_type = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
        callback_type = ctypes.WINFUNCTYPE(result_type, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

        def hook_callback(code, message, data):
            if code >= 0 and int(message) == WM_LBUTTONDOWN:
                try:
                    ts = time.time()
                    payload = ctypes.cast(data, ctypes.POINTER(_MouseHookData)).contents
                    geometry = _click_window_geometry(user32)
                    self._record(MouseClick(ts, int(payload.pt.x), int(payload.pt.y), geometry))
                except Exception:
                    self._counts["error"] += 1
            return user32.CallNextHookEx(self._hook, code, message, data)

        self._callback = callback_type(hook_callback)
        kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        user32.SetWindowsHookExW.argtypes = (ctypes.c_int, callback_type, wintypes.HMODULE, wintypes.DWORD)
        user32.SetWindowsHookExW.restype = ctypes.c_void_p
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
        # 不声明 argtypes 的话 ctypes 按 32 位 int 传参，64 位下 lParam 一超范围就抛
        # OverflowError，钩子链里每个鼠标事件都刷一条异常。
        user32.CallNextHookEx.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        user32.CallNextHookEx.restype = result_type
        user32.UnhookWindowsHookEx.argtypes = (ctypes.c_void_p,)
        user32.UnhookWindowsHookEx.restype = wintypes.BOOL
        self._hook = user32.SetWindowsHookExW(WH_MOUSE_LL, self._callback, kernel32.GetModuleHandleW(None), 0)
        self.last_error = 0 if self._hook else ctypes.get_last_error()
        self._ready.set()
        if not self._hook:
            return
        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
        user32.UnhookWindowsHookEx(self._hook)
        self._hook = None
