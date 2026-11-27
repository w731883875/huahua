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

    def start(self) -> None:
        if os.name != "nt" or (self._thread is not None and self._thread.is_alive()):
            return
        self._thread = threading.Thread(target=self._run, name="HuahuaMouseClickMonitor", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=1.0)

    def drain(self, *, max_age: float = 2.0) -> tuple[MouseClick, ...]:
        cutoff = time.time() - max(0.1, float(max_age))
        with self._lock:
            while self._clicks and self._clicks[0].ts < cutoff:
                self._clicks.popleft()
            result = tuple(self._clicks)
            self._clicks.clear()
        return result

    def close(self) -> None:
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
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = int(kernel32.GetCurrentThreadId())
        result_type = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
        callback_type = ctypes.WINFUNCTYPE(result_type, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

        def hook_callback(code, message, data):
            if code >= 0 and int(message) == WM_LBUTTONDOWN:
                payload = ctypes.cast(data, ctypes.POINTER(_MouseHookData)).contents
                geometry = _click_window_geometry(user32)
                click = MouseClick(time.time(), int(payload.pt.x), int(payload.pt.y), geometry)
                with self._lock:
                    self._clicks.append(click)
                    self._history.append(click)
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
        self._hook = user32.SetWindowsHookExW(WH_MOUSE_LL, self._callback, kernel32.GetModuleHandleW(None), 0)
        self._ready.set()
        if not self._hook:
            self.last_error = int(kernel32.GetLastError())
            return
        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
        user32.UnhookWindowsHookEx(self._hook)
        self._hook = None
