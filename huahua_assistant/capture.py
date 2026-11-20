from __future__ import annotations

import ctypes
import threading
import time
from contextlib import contextmanager
from ctypes import wintypes
from dataclasses import dataclass
from os.path import basename
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import mss
import numpy as np

from . import wgc_probe
from .wgc_probe import integrity_allows_window_capture


DEFAULT_WINDOW_TITLE = "torchlight_infinite"
DEFAULT_WINDOW_CANDIDATES = ("torchlight_infinite", "Torchlight: Infinite", "Torchlight")
PER_MONITOR_AWARE_V2 = -4


def enable_process_dpi_awareness() -> bool:
    """Enable physical-pixel coordinates before Tk creates any windows."""

    try:
        user32 = ctypes.windll.user32
        setter = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if setter is not None:
            setter.argtypes = (ctypes.c_void_p,)
            setter.restype = wintypes.BOOL
            if setter(ctypes.c_void_p(PER_MONITOR_AWARE_V2)):
                return True
    except Exception:
        pass
    try:
        shcore = ctypes.windll.shcore
        if shcore.SetProcessDpiAwareness(2) in (0, -2147024891):
            return True
    except Exception:
        pass
    try:
        return bool(ctypes.windll.user32.SetProcessDPIAware())
    except Exception:
        return False


@contextmanager
def physical_pixel_coordinates():
    """Temporarily disable DPI virtualization for capture worker threads."""

    previous = None
    setter = None
    try:
        user32 = ctypes.windll.user32
        setter = getattr(user32, "SetThreadDpiAwarenessContext", None)
        if setter is not None:
            setter.argtypes = (ctypes.c_void_p,)
            setter.restype = ctypes.c_void_p
            previous = setter(ctypes.c_void_p(PER_MONITOR_AWARE_V2))
    except Exception:
        previous = None
        setter = None
    try:
        yield
    finally:
        if setter is not None and previous:
            try:
                setter(previous)
            except Exception:
                pass


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    process_id: int
    process_name: str
    left: int
    top: int
    right: int
    bottom: int
    minimized: bool

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def captureable(self) -> bool:
        return (
            not self.minimized
            and self.width > 50
            and self.height > 50
            and self.left > -20000
            and self.top > -20000
        )


@dataclass(frozen=True)
class CaptureResult:
    image: np.ndarray
    source: str
    window: WindowInfo | None = None

    @property
    def kind(self) -> str:
        """结构化来源分级，见 :func:`capture_kind`。"""
        return capture_kind(self.source)

    @property
    def trusted_for_automation(self) -> bool:
        """这一帧的像素能不能自动落盘/自动交给跨进程托管。

        **只认窗口自己渲染的内容**。``window-screen`` 虽然带 ``window`` 对象，
        读的却是合成后的桌面——窗口被别的程序盖住时它拿到的是别人的像素，
        ``screen:fallback`` 更是整屏。仅凭"window 对象存在"放行是错的。
        """
        return self.kind in TRUSTED_CAPTURE_KINDS


#: 采集来源分级。UI、落盘策略、外发策略都只看这个，不再各自去猜 source 字符串。
CAPTURE_KIND_WINDOW_CONTENT = "window_content"   # 窗口自己渲染的表面（WGC/PrintWindow）
CAPTURE_KIND_WINDOW_REGION = "window_region"     # 桌面合成后按窗口矩形裁的影
CAPTURE_KIND_SCREEN = "screen"                   # 整屏兜底
CAPTURE_KIND_PAUSED = "paused"                   # 主动等待（游戏不在前台/窗口不在）
CAPTURE_KIND_ERROR = "error"                     # 采集/识别链路出错
CAPTURE_KIND_UNKNOWN = "unknown"

#: 允许自动落盘与自动外发的来源。**只放窗口内容**，加白名单要同时改验收用例。
TRUSTED_CAPTURE_KINDS = frozenset({CAPTURE_KIND_WINDOW_CONTENT})


def capture_kind(source: str) -> str:
    """把 ``source`` 字符串归到上面的分级。

    ``source`` 是唯一事实来源（早就写进了日志、记录和界面），这里只做归类，
    不引入第二套标记，避免"来源字段说是窗口内容、字符串说不是"的自相矛盾。
    """
    text = str(source or "").strip().casefold()
    if not text:
        return CAPTURE_KIND_UNKNOWN
    if text.startswith("window-content"):
        return CAPTURE_KIND_WINDOW_CONTENT
    if text.startswith("window-foreground-screen") or text.startswith("window-screen"):
        return CAPTURE_KIND_WINDOW_REGION
    if text.startswith("screen:"):
        return CAPTURE_KIND_SCREEN
    if text.startswith("paused:"):
        return CAPTURE_KIND_PAUSED
    if text.startswith("frame-pump:error") or text.startswith("error:"):
        return CAPTURE_KIND_ERROR
    return CAPTURE_KIND_UNKNOWN


class CaptureBackend(Protocol):
    name: str

    def capture_window_content(self, window: WindowInfo) -> tuple[np.ndarray | None, str]:
        """Capture pixels owned by the target window, not composed desktop pixels."""

    def capture_window_region(self, window: WindowInfo) -> tuple[np.ndarray | None, str]:
        """Capture the target window rectangle from the composed desktop."""


@dataclass(frozen=True)
class CaptureBackendPlan:
    name: str
    content_backends: tuple[CaptureBackend, ...]
    region_backends: tuple[CaptureBackend, ...]


class LegacyCaptureBackend:
    name = "legacy"

    def capture_window_content(self, window: WindowInfo) -> tuple[np.ndarray | None, str]:
        if window.process_name.casefold() == "torchlight_infinite.exe":
            return None, "printwindow_skipped_directx"
        try:
            image = _capture_window_print(window)
        except Exception as exc:
            return None, f"printwindow_error:{type(exc).__name__}"
        if image is None:
            return None, "printwindow_failed"
        if not _looks_like_valid_capture(image):
            return None, "printwindow_invalid"
        return image, "printwindow"

    def capture_window_region(self, window: WindowInfo) -> tuple[np.ndarray | None, str]:
        region = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
        }
        try:
            image = capture_screen(region)
        except Exception as exc:
            return None, f"mss_region_error:{type(exc).__name__}"
        return image, "mss-region"


class _WgcSession:
    """One live Windows.Graphics.Capture session for a single window handle.

    ``CreateForWindow`` captures the window's own composition surface, so other
    windows drawn on top of it never enter the frame. That is exactly what
    ``PrintWindow`` cannot do for DirectX windows, and what plain screen copying
    gets wrong.

    Frames are copied on demand: the callback runs for every presented frame but
    only touches ``frame_buffer`` when a grab is waiting. The buffer is a mapped
    view owned by the native side and is released once the callback returns, so
    it must be copied before returning.
    """

    def __init__(self, hwnd: int, update_interval_ms: int | None = None) -> None:
        from windows_capture import WindowsCapture  # type: ignore[import-not-found]

        self.hwnd = int(hwnd)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._frame: np.ndarray | None = None
        self._frame_at = 0.0
        self._copy_at = 0.0
        self._sequence = 0
        self._want = False
        self._closed = False
        self._error: str | None = None
        self._size = (0, 0)

        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            secondary_window=False,
            minimum_update_interval=update_interval_ms,
            window_hwnd=self.hwnd,
        )

        @capture.event
        def on_frame_arrived(frame, capture_control) -> None:  # noqa: ANN001
            buffer = frame.frame_buffer
            with self._lock:
                self._sequence += 1
                self._frame_at = time.monotonic()
                self._size = (int(frame.width), int(frame.height))
                if not self._want:
                    return
                # BGRA -> BGR, contiguous copy out of the native mapping.
                self._frame = np.ascontiguousarray(buffer[:, :, :3])
                # Freshness must track the copy, not the presentation: a window
                # that keeps presenting frames we skip would otherwise make a
                # stale copy look new.
                self._copy_at = self._frame_at
                self._want = False
                self._ready.set()

        @capture.event
        def on_closed() -> None:
            with self._lock:
                self._closed = True
                self._ready.set()

        self._capture = capture
        self._control = capture.start_free_threaded()

    @property
    def size(self) -> tuple[int, int]:
        with self._lock:
            return self._size

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    def grab(
        self,
        timeout: float = 0.06,
        cold_timeout: float = 0.30,
        max_stale: float = 1.0,
        immediate_window: float = 0.04,
    ) -> tuple[np.ndarray | None, str]:
        """Return a picture of the window.

        Windows only delivers a frame when the window presents a new one. A
        paused game or an idle window presents nothing, so waiting for a fresh
        frame would cost the full timeout on every call. A frame younger than
        ``immediate_window`` is therefore handed back straight away, and a
        window that has never produced a frame gets a longer cold timeout.
        """

        with self._lock:
            if self._closed:
                return None, "wgc_closed"
            if self._frame is not None:
                age = time.monotonic() - self._copy_at
                if age <= immediate_window:
                    return self._frame, "wgc-cached"
                wait_for = timeout
            else:
                wait_for = cold_timeout
            started_at_sequence = self._sequence
            self._ready.clear()
            self._want = True
        self._ready.wait(wait_for)
        with self._lock:
            if self._closed:
                return None, "wgc_closed"
            if self._frame is None:
                return None, "wgc_no_frame"
            if self._sequence <= started_at_sequence:
                # Nothing new arrived in time. A recent copy is still a true
                # picture of the window; an old one is not.
                age = time.monotonic() - self._copy_at
                if age > max_stale:
                    return None, f"wgc_stale:{age:.2f}s"
                return self._frame, "wgc-reused"
            return self._frame, "wgc"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._ready.set()
        try:
            self._control.stop()
        except Exception:
            pass
        try:
            self._control.wait()
        except Exception:
            pass


class WindowsGraphicsCaptureBackend:
    """Capture window-owned pixels through Windows.Graphics.Capture.

    Keeps one session per window handle: creating and dropping a capture session
    for every grab leaks Direct3D devices (the reference implementation needed a
    release fix for that), and a persistent session is also much cheaper.
    """

    name = "wgc"

    #: 同一个句柄连续失败几次就判定「这台机器上 WGC 用不了」。
    #: 偶尔被拒绝（窗口刚创建、会话刚失效）还能重试；每次抓帧都重新构造一遍
    #: 会话对象则纯亏——本机实测每条 proto 都会抛
    #: ``Failed to convert item to GraphicsCaptureItem``，一帧一次异常。
    MAX_START_FAILURES = 3

    def __init__(self) -> None:
        self._sessions: dict[int, _WgcSession] = {}
        self._lock = threading.Lock()
        self._unavailable: str | None = None
        self._failures: dict[int, int] = {}

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable

    def reset(self) -> None:
        """把「不可用」判定清掉（换了游戏窗口、用户重开游戏时用）。"""

        with self._lock:
            self._unavailable = None
            self._failures.clear()

    def capture_window_content(self, window: WindowInfo) -> tuple[np.ndarray | None, str]:
        if self._unavailable is not None:
            return None, self._unavailable
        if window.minimized:
            return None, "wgc_minimized"
        session, reason = self._session_for(window.hwnd)
        if session is None:
            return None, reason
        image, method = session.grab()
        if image is None:
            if method == "wgc_closed":
                self._drop(window.hwnd)
            return None, method
        if not _looks_like_valid_capture(image):
            return None, "wgc_invalid"
        return image, method

    def capture_window_region(self, window: WindowInfo) -> tuple[np.ndarray | None, str]:
        return None, "window_region_unavailable"

    def prewarm(self, window: WindowInfo) -> str:
        """Open the session and wait for a first frame.

        Building the Direct3D device and the capture item takes several hundred
        milliseconds. Doing it here keeps that cost off the first real frame.
        """

        if window.minimized:
            return "wgc_minimized"
        session, reason = self._session_for(window.hwnd)
        if session is None:
            return reason
        session.grab(timeout=0.05, cold_timeout=0.6)
        return "wgc_ready"

    def has_session(self, hwnd: int) -> bool:
        with self._lock:
            return int(hwnd) in self._sessions

    def _session_for(self, hwnd: int) -> tuple[_WgcSession | None, str]:
        with self._lock:
            session = self._sessions.get(int(hwnd))
        if session is not None:
            return session, "wgc"
        # 进程完整性是 WGC 的第一道门：LOW 完整性（自动化环境拉起的进程）对
        # **任何**目标都是 E_ACCESSDENIED。这条不先判，就会变成每帧抛一次异常、
        # 还把状态报成「可用」（见 wgc_probe 模块头）。判定很便宜，做一次就够。
        if self._unavailable is None and not integrity_allows_window_capture():
            self._unavailable = "wgc_blocked_low_integrity"
            return None, self._unavailable
        try:
            session = _WgcSession(int(hwnd))
        except ImportError:
            self._unavailable = "wgc_unavailable:ModuleNotFoundError"
            return None, self._unavailable
        except Exception as exc:  # capture item refused, device lost, old Windows
            message = f"wgc_start_error:{type(exc).__name__}:{_short_error(exc)}"
            # A refused window is worth retrying later; a missing API is not.
            if "not supported" in str(exc).lower() or "classnotavailable" in str(exc).lower():
                self._unavailable = message
                return None, message
            with self._lock:
                self._failures[int(hwnd)] = self._failures.get(int(hwnd), 0) + 1
                failures = self._failures[int(hwnd)]
                if failures >= self.MAX_START_FAILURES and self._unavailable is None:
                    # 连续失败到阈值：这不是「这个窗口暂时不行」，是这套 API
                    # 在这台机器上根本用不了。继续每帧重试只会白烧时间，
                    # 更要紧的是**别再对外报告「可用」**——上层据此判断
                    # 「工具窗口挡住游戏要不要紧」，谎报会直接让脏帧进识别。
                    self._unavailable = f"wgc_start_failed:{failures}x:{_short_error(exc)}"
            return None, message
        with self._lock:
            self._sessions[int(hwnd)] = session
            self._failures.pop(int(hwnd), None)
        return session, "wgc"

    def _drop(self, hwnd: int) -> None:
        with self._lock:
            session = self._sessions.pop(int(hwnd), None)
        if session is not None:
            session.close()

    def close(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def prune(self, live_hwnds: Iterable[int]) -> None:
        keep = {int(item) for item in live_hwnds}
        with self._lock:
            stale = [hwnd for hwnd in self._sessions if hwnd not in keep]
        for hwnd in stale:
            self._drop(hwnd)


def _short_error(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", " ")
    return text[:120] if text else "no message"


# ---------------------------------------------------------------- 截屏免疫
#
# 这台机器上 Windows.Graphics.Capture 整体起不来（连本进程的 Tk 窗口都报
# ``Failed to convert item to GraphicsCaptureItem``），所以只能退回「屏幕复制」。
# 屏幕复制拍到的是合成后的桌面像素，工具窗口压上去就进帧——这是用户看到的
# 「软件窗口挡棋盘」的真正原因，不是布局问题。
#
# ``SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`` 让窗口对**所有**截屏
# API（GDI BitBlt / DXGI 桌面复制 / WGC）都不可见，屏幕上照常显示。官方文档
# 的用途就是「共享会话里不让别人看到这个窗口」。Win10 2004（19041）+。
#
# 实测（本机 Win11 26200）：洋红满窗覆盖区，截屏洋红占比 0.775 → 0.000，
# 能看到窗口后面的真实画面；还原 WDA_NONE 后立刻回到 0.775。
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011
GA_ROOT = 2

#: 窗口免疫状态最多每隔这么久复查一次。Tk 轮询很密（80ms 级），
#: 每次走一遍窗口树 + 三个 user32 调用没必要。
_EXCLUSION_REFRESH_SECONDS = 1.0
_exclusion_cache: dict[str, object] = {"at": -1e9, "ok": False, "hwnds": (), "error": None}


def set_capture_exclusion(hwnd: int, enabled: bool = True) -> bool:
    """把单个顶层窗口标成「对截屏不可见」（或还原）。"""

    try:
        user32 = ctypes.windll.user32
        user32.SetWindowDisplayAffinity.argtypes = (wintypes.HWND, wintypes.DWORD)
        user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
        value = WDA_EXCLUDEFROMCAPTURE if enabled else WDA_NONE
        return bool(user32.SetWindowDisplayAffinity(wintypes.HWND(int(hwnd)), value))
    except Exception:
        return False


def capture_exclusion(hwnd: int) -> int | None:
    """读窗口当前的 display affinity；读不到返回 None。"""

    try:
        user32 = ctypes.windll.user32
        user32.GetWindowDisplayAffinity.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
        user32.GetWindowDisplayAffinity.restype = wintypes.BOOL
        value = wintypes.DWORD()
        if not user32.GetWindowDisplayAffinity(wintypes.HWND(int(hwnd)), ctypes.byref(value)):
            return None
        return int(value.value)
    except Exception:
        return None


def window_is_excluded_from_capture(hwnd: int) -> bool:
    return capture_exclusion(hwnd) == WDA_EXCLUDEFROMCAPTURE


def _top_level_hwnd(widget: object) -> int | None:
    """Tk 的 ``winfo_id()`` 给的是内层窗口，affinity 只认顶层窗口。"""

    try:
        inner = int(widget.winfo_id())  # type: ignore[attr-defined]
    except Exception:
        return None
    if inner <= 0:
        return None
    try:
        user32 = ctypes.windll.user32
        user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
        user32.GetAncestor.restype = wintypes.HWND
        root = user32.GetAncestor(wintypes.HWND(inner), GA_ROOT)
        return int(root) if root else inner
    except Exception:
        return inner


def tool_window_hwnds(widget: object) -> list[int]:
    """本体窗口 + 全部 Toplevel（浮层、桌宠、对话框）的顶层句柄。"""

    handles: list[int] = []
    try:
        children = list(widget.winfo_children())  # type: ignore[attr-defined]
    except Exception:
        children = []
    for item in [widget, *children]:
        hwnd = _top_level_hwnd(item)
        if hwnd and hwnd not in handles:
            handles.append(hwnd)
    return handles


#: 远程/串流环境在显示设备名里留下的痕迹。ToDesk、向日葵、AnyDesk、Parsec
#: 都会注册虚拟显示适配器；RDP 由 SM_REMOTESESSION 覆盖，不需要出现在这里。
_REMOTE_DISPLAY_HINTS = (
    "todesk",
    "oray",
    "sunlogin",
    "anydesk",
    "parsec",
    "rustdesk",
    "teamviewer",
    "mirror driver",
    "virtual display",
)

_remote_reason_cache: str | None = None
_remote_reason_checked = False


def remote_session_reason() -> str | None:
    """判断「这台机器现在是不是被远程看」，返回原因字符串或 ``None``。

    两条互相独立的证据：

    1. ``SM_REMOTESESSION`` —— 只有微软自己的 RDP 会置位；
    2. 显示适配器名字里带串流软件痕迹 —— ToDesk / 向日葵 / AnyDesk / Parsec
       这类会装虚拟显示驱动，这条能覆盖 RDP 之外的情况。

    为什么要判这个：工具窗口开了「对截屏免疫」(``WDA_EXCLUDEFROMCAPTURE``)
    之后，**任何**抓屏通道（含远程串流、录屏、PrintWindow）看到的都是纯黑。
    本地盯着屏幕看没问题，远程用户就只剩一块黑。2026-11-18 实机：用户远程
    连接时表现为「有边框、无标题栏、不能拖动、任务栏预览点不开」。

    结果缓存一次（驱动和会话不会中途变化，不必每 2 秒重扫）。
    """

    global _remote_reason_cache, _remote_reason_checked
    if _remote_reason_checked:
        return _remote_reason_cache
    _remote_reason_checked = True
    reason: str | None = None
    try:
        if ctypes.windll.user32.GetSystemMetrics(0x1000):  # SM_REMOTESESSION
            reason = "rdp"
    except Exception:
        reason = None
    if reason is None:
        reason = _virtual_display_hint()
    _remote_reason_cache = reason
    return reason


def _virtual_display_hint() -> str | None:
    """扫显示适配器名字，找出串流软件留下的虚拟显示器。"""

    try:
        user32 = ctypes.windll.user32
        size = ctypes.sizeof(_DISPLAY_DEVICEW())
        index = 0
        while index < 16:
            device = _DISPLAY_DEVICEW()
            device.cb = size
            if not user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
                break
            index += 1
            name = str(device.DeviceString or "").lower()
            for hint in _REMOTE_DISPLAY_HINTS:
                if hint in name:
                    return f"display:{hint}"
    except Exception:
        return None
    return None


class _DISPLAY_DEVICEW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("DeviceName", wintypes.WCHAR * 32),
        ("DeviceString", wintypes.WCHAR * 128),
        ("StateFlags", wintypes.DWORD),
        ("DeviceID", wintypes.WCHAR * 128),
        ("DeviceKey", wintypes.WCHAR * 128),
    ]


def is_remote_session() -> bool:
    """``remote_session_reason()`` 的布尔版本。"""

    return remote_session_reason() is not None


def exclude_tool_windows_from_capture(widget: object) -> list[int]:
    """让工具自己的每个顶层窗口都对截屏不可见。

    返回「确认已经免疫」的句柄。任何一步失败都不抛：免疫只是优化，
    失败时上层会退回原来的「遮挡就暂停」逻辑。
    """

    applied: list[int] = []
    for hwnd in tool_window_hwnds(widget):
        if window_is_excluded_from_capture(hwnd):
            applied.append(hwnd)
            continue
        if set_capture_exclusion(hwnd):
            applied.append(hwnd)
    return applied


def windows_excluded_from_capture(widget: object, *, refresh_seconds: float | None = None) -> bool:
    """工具窗口是否已经整体对截屏免疫。

    真时「工具窗口盖住游戏」不再需要暂停：屏幕复制根本拍不到它。
    带一秒级缓存，因为 Tk 轮询比这件事需要的频率高得多。
    """

    key = f"windows_excluded:{id(widget)}"
    now = time.monotonic()
    interval = _EXCLUSION_REFRESH_SECONDS if refresh_seconds is None else float(refresh_seconds)
    if (
        _exclusion_cache.get("key") == key
        and now - float(_exclusion_cache["at"]) < interval
    ):
        return bool(_exclusion_cache["ok"])

    handles = tool_window_hwnds(widget)
    # 必须**每一个**顶层窗口都免疫才算数：只盖住主窗口、桌宠还漏在外面，
    # 那张桌宠照样会把画面弄脏，而这里一旦谎报，遮挡判断就再也不会拦。
    applied = exclude_tool_windows_from_capture(widget)
    ok = bool(handles) and set(applied) == set(handles)
    _exclusion_cache.update({"key": key, "at": now, "ok": ok, "hwnds": tuple(handles)})
    return ok


def capture_exclusion_status(widget: object | None = None) -> dict[str, object]:
    """诊断用：每个窗口的 affinity 值，以及整体是否免疫。"""

    payload: dict[str, object] = {
        "supported": True,
        "excluded": False,
        "windows": {},
        "last_error": _exclusion_cache.get("error"),
    }
    if widget is None:
        return payload
    windows: dict[int, int | None] = {}
    for hwnd in tool_window_hwnds(widget):
        windows[hwnd] = capture_exclusion(hwnd)
    payload["windows"] = windows
    payload["excluded"] = bool(windows) and all(
        value == WDA_EXCLUDEFROMCAPTURE for value in windows.values()
    )
    return payload


class DxcamRegionCaptureBackend:
    name = "dxcam"

    def __init__(self) -> None:
        self._camera = None
        self._camera_error: str | None = None

    def capture_window_content(self, window: WindowInfo) -> tuple[np.ndarray | None, str]:
        return None, "window_content_unavailable"

    def capture_window_region(self, window: WindowInfo) -> tuple[np.ndarray | None, str]:
        camera, reason = self._get_camera()
        if camera is None:
            return None, reason

        try:
            image = camera.grab(region=(window.left, window.top, window.right, window.bottom))
        except Exception as exc:
            return None, f"dxcam_region_error:{type(exc).__name__}"
        if image is None:
            return None, "dxcam_region_failed"
        image = np.ascontiguousarray(image)
        if not _looks_like_valid_capture(image):
            return None, "dxcam_region_invalid"
        return image, "dxcam-region"

    def _get_camera(self):
        if self._camera is not None:
            return self._camera, "dxcam"
        if self._camera_error is not None:
            return None, self._camera_error
        try:
            import dxcam  # type: ignore[import-not-found]
        except Exception as exc:
            self._camera_error = f"dxcam_unavailable:{type(exc).__name__}"
            return None, self._camera_error
        try:
            self._camera = dxcam.create(output_color="BGR")
        except Exception as exc:
            self._camera_error = f"dxcam_create_error:{type(exc).__name__}"
            return None, self._camera_error
        if self._camera is None:
            self._camera_error = "dxcam_create_failed"
            return None, self._camera_error
        return self._camera, "dxcam"


_LEGACY_BACKEND = LegacyCaptureBackend()
_DXCAM_BACKEND = DxcamRegionCaptureBackend()
_WGC_BACKEND = WindowsGraphicsCaptureBackend()
_CAPTURE_THREAD_LOCAL = threading.local()


def capture_screen(region: dict[str, int] | None = None) -> np.ndarray:
    sct = getattr(_CAPTURE_THREAD_LOCAL, "mss", None)
    if sct is None:
        sct = mss.mss()
        _CAPTURE_THREAD_LOCAL.mss = sct
    target = region if region is not None else sct.monitors[1]
    shot = sct.grab(target)
    frame = np.asarray(shot)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def capture_window_content(window: WindowInfo) -> tuple[np.ndarray | None, str]:
    """Capture window-owned pixels first so desktop overlays do not enter recognition."""

    image, method = _WGC_BACKEND.capture_window_content(window)
    if image is not None:
        return image, f"wgc:{method}"
    return _LEGACY_BACKEND.capture_window_content(window)


def close_capture_backends() -> None:
    """Release persistent capture sessions (call on shutdown)."""

    _WGC_BACKEND.close()


def wgc_status() -> str:
    reason = _WGC_BACKEND.unavailable_reason
    return reason or "wgc_ready"


def reset_wgc_state() -> None:
    """清掉「WGC 不可用」的判定，让下次抓帧重新试探一遍。

    用户重开游戏、换显示器、升级驱动之后值得再试；连续失败的那套判定
    只是「这次别浪费帧时间」，不该变成永久结论。
    """

    _WGC_BACKEND.reset()


def capture_diagnostics(hwnds: dict[str, int] | None = None) -> dict[str, object]:
    """采集链路自检：完整性级别 + 逐个目标的 CreateForWindow HRESULT。"""

    return wgc_probe.diagnose(hwnds)


def format_capture_diagnostics(payload: dict[str, object]) -> str:
    return wgc_probe.format_report(payload)


def window_content_capture_ready(window: WindowInfo | None) -> bool:
    """True once WGC is reading this window directly.

    From that moment the frame no longer depends on what is drawn on top of the
    game, so overlap with the tool's own window stops being a reason to pause.
    """

    if window is None:
        return False
    return _WGC_BACKEND.has_session(window.hwnd)


def window_content_capture_available() -> bool:
    """True when window-content capture could work at all, session or not.

    Overlap only matters when we are reduced to copying the screen. If WGC is
    usable, the correct answer for "is the tool window in the way" is always no
    -- and saying yes because the session is not up yet would stop the capture
    thread, which is the only thing that can ever create that session.
    """

    return _WGC_BACKEND.unavailable_reason is None


def prewarm_window_capture(window: WindowInfo | None) -> str:
    """Open the capture session ahead of time so the first frame is not slow."""

    if window is None:
        return "no_window"
    return _WGC_BACKEND.prewarm(window)


def resolve_capture_backend(backend: str | CaptureBackend | None = None) -> CaptureBackendPlan:
    if backend is None:
        backend = _capture_config_value("capture.backend", "auto")
    if not isinstance(backend, str):
        return CaptureBackendPlan(
            name=getattr(backend, "name", backend.__class__.__name__),
            content_backends=(backend,),
            region_backends=(backend, _LEGACY_BACKEND),
        )

    requested = backend.strip().casefold() or "auto"
    if requested in {"legacy", "printwindow", "mss"}:
        return CaptureBackendPlan("legacy", (_LEGACY_BACKEND,), (_LEGACY_BACKEND,))
    if requested in {"wgc", "graphicscapture"}:
        return CaptureBackendPlan("wgc", (_WGC_BACKEND, _LEGACY_BACKEND), (_LEGACY_BACKEND,))
    if requested in {"dxcam", "highspeed"}:
        return CaptureBackendPlan(requested, (_DXCAM_BACKEND,), (_DXCAM_BACKEND, _LEGACY_BACKEND))
    if requested == "auto":
        highspeed_enabled = _capture_config_bool("capture.highspeed_enabled", False)
        allow_composited = _capture_config_bool("capture.allow_composited_highspeed", False)
        # WGC goes first: it reads the window's own surface, so a window drawn
        # over the game no longer ruins the frame. PrintWindow is useless for
        # DirectX and screen copying returns whatever is on top, so both stay
        # behind it as fallbacks.
        content: tuple[CaptureBackend, ...] = (
            (_WGC_BACKEND, _LEGACY_BACKEND)
            if _capture_config_bool("capture.wgc_enabled", True)
            else (_LEGACY_BACKEND,)
        )
        if highspeed_enabled and allow_composited:
            return CaptureBackendPlan("auto", content, (_DXCAM_BACKEND, _LEGACY_BACKEND))
        return CaptureBackendPlan("auto", content, (_LEGACY_BACKEND,))
    raise ValueError(f"Unknown capture backend: {backend}")


def capture_target(
    window_title: str = DEFAULT_WINDOW_TITLE,
    fallback_to_screen: bool = True,
    fallback_if_target_uncaptureable: bool = False,
    activate: bool = True,
    activation_delay: float = 0.35,
    backend: str | CaptureBackend | None = None,
) -> CaptureResult:
    with physical_pixel_coordinates():
        return _capture_target_physical(
            window_title=window_title,
            fallback_to_screen=fallback_to_screen,
            fallback_if_target_uncaptureable=fallback_if_target_uncaptureable,
            activate=activate,
            activation_delay=activation_delay,
            backend=backend,
        )


def _capture_target_physical(
    window_title: str,
    fallback_to_screen: bool,
    fallback_if_target_uncaptureable: bool,
    activate: bool,
    activation_delay: float,
    backend: str | CaptureBackend | None,
) -> CaptureResult:
    backend_plan = resolve_capture_backend(backend)
    candidates = _target_candidates(window_title)
    matches = find_matching_windows(candidates)
    for matched in matches:
        if not matched.captureable:
            restore_window(matched.hwnd)

    if matches:
        matches = find_matching_windows(candidates)

    captureable = [window for window in matches if window.captureable]
    window = max(captureable, key=lambda item: item.width * item.height) if captureable else None
    if window is not None:
        attempts: list[str] = []

        # Window content first: Windows.Graphics.Capture reads the window's own
        # surface, so it neither needs the window in front nor a screen copy.
        # Trying it before activating keeps a manual capture from stealing focus
        # from the game (and from waiting out the activation delay).
        for capture_backend in backend_plan.content_backends:
            internal_image, internal_method = capture_backend.capture_window_content(window)
            attempts.append(f"{capture_backend.name}:{internal_method}")
            if internal_image is not None:
                return CaptureResult(
                    image=internal_image,
                    source=(
                        f"window-content:{window.title} process:{window.process_name} "
                        f"backend:{capture_backend.name} method:{internal_method}"
                    ),
                    window=window,
                )

        # Nothing owns the pixels: fall back to copying the screen, which does
        # need the window in front.
        foreground_confirmed = False
        if activate:
            activate_window(window.hwnd)
            if activation_delay > 0:
                time.sleep(activation_delay)
            refreshed_matches = find_matching_windows(candidates)
            refreshed_captureable = [item for item in refreshed_matches if item.captureable]
            if refreshed_captureable:
                window = max(refreshed_captureable, key=lambda item: item.width * item.height)
            foreground_confirmed = is_foreground_window(window.hwnd)

        for capture_backend in backend_plan.region_backends:
            region_image, region_method = capture_backend.capture_window_region(window)
            attempts.append(f"{capture_backend.name}:{region_method}")
            if region_image is not None:
                return CaptureResult(
                    image=region_image,
                    source=(
                        f"{'window-foreground-screen' if foreground_confirmed else 'window-screen'}:"
                        f"{window.title} process:{window.process_name} "
                        f"backend:{capture_backend.name} method:{region_method} "
                        f"content_failed:{_format_capture_attempts(attempts)}"
                    ),
                    window=window,
                )

        failed = _format_capture_attempts(attempts)
        if fallback_to_screen:
            return CaptureResult(
                image=capture_screen(),
                source=(
                    f"screen:fallback window_capture_failed:{window.title} "
                    f"process:{window.process_name} attempts:{failed}"
                ),
                window=window,
            )
        raise RuntimeError(f"Window capture failed: {failed}")

    if matches:
        found = ", ".join(_window_summary(item) for item in matches)
        if not fallback_if_target_uncaptureable:
            raise RuntimeError(
                "Matched game window is not captureable. "
                "Restore the game window to the foreground, then capture again. "
                f"Matched: {found}"
            )
        source = f"screen:fallback target_not_captureable:{found}"
    elif not fallback_to_screen:
        raise RuntimeError(f"Window not found: {window_title}")
    else:
        source = f"screen:fallback window_not_found:{window_title}"

    return CaptureResult(
        image=capture_screen(),
        source=source,
    )


def _capture_config_value(path: str, default):
    try:
        from .config import config_value
    except Exception:
        return default
    return config_value(path, default)


def _capture_config_bool(path: str, default: bool) -> bool:
    value = _capture_config_value(path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _format_capture_attempts(attempts: Iterable[str]) -> str:
    return ",".join(item for item in attempts if item) or "none"


def find_window(title_contains: str) -> WindowInfo | None:
    matches = find_matching_windows([title_contains])
    if not matches:
        return None
    captureable = [window for window in matches if window.captureable]
    pool = captureable if captureable else matches
    return max(pool, key=lambda window: window.width * window.height)


def find_capture_window(window_title: str = DEFAULT_WINDOW_TITLE) -> WindowInfo | None:
    """找**可抓取**的游戏窗口；最小化、尺寸异常的一律不收。

    与 :func:`find_window` 的差别有两点，都是为了跟 ``capture_target`` 的判定
    对齐：用完整候选标题（否则会先判"没窗口"、下一行又抓到窗口），
    并且**只返回可抓的**——``find_window`` 在全都不可抓时也会返回一个对象，
    调用方拿它去截图只会得到一张黑图或桌面。
    """
    matches = find_matching_windows(_target_candidates(window_title))
    captureable = [window for window in matches if window.captureable]
    if not captureable:
        return None
    return max(captureable, key=lambda window: window.width * window.height)


def find_matching_windows(targets: Iterable[str]) -> list[WindowInfo]:
    needles = [target.casefold() for target in targets if target.strip()]
    matches: list[WindowInfo] = []
    for window in list_visible_windows():
        haystacks = (window.title.casefold(), window.process_name.casefold())
        if any(needle in haystack for needle in needles for haystack in haystacks):
            matches.append(window)
    return matches


def list_visible_windows() -> list[WindowInfo]:
    user32 = ctypes.windll.user32
    windows: list[WindowInfo] = []

    enum_windows_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.strip()
        if not title:
            return True

        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True

        minimized = bool(user32.IsIconic(hwnd))
        # GetWindowRect includes the invisible resize border (about 9 px per
        # side on Windows 10/11). Windows.Graphics.Capture does not, so the two
        # would disagree and every derived screen coordinate would be off by
        # that margin. The DWM's extended frame bounds are the visible ones.
        # A minimized window has no meaningful visible bounds, so keep the
        # rectangle Windows reports for it.
        if not minimized:
            rect = _visible_bounds(hwnd, rect)

        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return True

        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        process_name = _process_name(process_id.value)

        windows.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=title,
                process_id=int(process_id.value),
                process_name=process_name,
                left=int(rect.left),
                top=int(rect.top),
                right=int(rect.right),
                bottom=int(rect.bottom),
                minimized=minimized,
            )
        )
        return True

    user32.EnumWindows(enum_windows_proc(callback), 0)
    return windows


DWMWA_EXTENDED_FRAME_BOUNDS = 9


def _visible_bounds(hwnd: int, fallback: wintypes.RECT) -> wintypes.RECT:
    """Window rectangle without the invisible resize border.

    ``GetWindowRect`` reserves ~9 px per side for the resize grip that is not
    drawn. Windows.Graphics.Capture frames start at the visible edge, so keeping
    the inflated rectangle would shift every screen coordinate derived from a
    frame. Falls back to the given rectangle when the DWM cannot answer.
    """

    try:
        rect = wintypes.RECT()
        result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if result == 0 and rect.right > rect.left and rect.bottom > rect.top:
            return rect
    except Exception:
        pass
    return fallback


def restore_window(hwnd: int) -> None:
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    user32.ShowWindow(hwnd, SW_RESTORE)


def activate_window(hwnd: int) -> None:
    restore_window(hwnd)
    user32 = ctypes.windll.user32
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_SHOWWINDOW = 0x0040
    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2
    flags = SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW
    # SetForegroundWindow can be denied when capture runs from a background
    # worker. A brief topmost toggle reliably raises the game without leaving
    # it permanently topmost.
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, flags)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)


def is_foreground_window(hwnd: int) -> bool:
    try:
        return int(ctypes.windll.user32.GetForegroundWindow()) == int(hwnd)
    except Exception:
        return False


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 1),
    ]


def _capture_window_print(window: WindowInfo) -> np.ndarray | None:
    width = int(window.width)
    height = int(window.height)
    if width <= 0 or height <= 0:
        return None

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    user32.GetWindowDC.argtypes = (wintypes.HWND,)
    user32.GetWindowDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
    user32.ReleaseDC.restype = ctypes.c_int
    gdi32.CreateCompatibleDC.argtypes = (wintypes.HDC,)
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = (wintypes.HDC, ctypes.c_int, ctypes.c_int)
    gdi32.CreateCompatibleBitmap.restype = wintypes.HANDLE
    gdi32.SelectObject.argtypes = (wintypes.HDC, wintypes.HANDLE)
    gdi32.SelectObject.restype = wintypes.HANDLE
    gdi32.DeleteObject.argtypes = (wintypes.HANDLE,)
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = (wintypes.HDC,)
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.GetDIBits.argtypes = (
        wintypes.HDC,
        wintypes.HANDLE,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.LPVOID,
        ctypes.POINTER(BITMAPINFO),
        wintypes.UINT,
    )
    gdi32.GetDIBits.restype = ctypes.c_int
    hwnd = wintypes.HWND(window.hwnd)
    window_dc = user32.GetWindowDC(hwnd)
    if not window_dc:
        return None

    memory_dc = None
    bitmap = None
    old_bitmap = None
    try:
        memory_dc = gdi32.CreateCompatibleDC(window_dc)
        if not memory_dc:
            return None
        bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
        if not bitmap:
            return None
        old_bitmap = gdi32.SelectObject(memory_dc, bitmap)

        user32.PrintWindow.argtypes = (wintypes.HWND, wintypes.HDC, wintypes.UINT)
        user32.PrintWindow.restype = wintypes.BOOL
        PW_RENDERFULLCONTENT = 0x00000002
        ok = user32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT)
        if not ok:
            ok = user32.PrintWindow(hwnd, memory_dc, 0)
        if not ok:
            return None

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0
        buffer_size = width * height * 4
        buffer = ctypes.create_string_buffer(buffer_size)
        rows = gdi32.GetDIBits(memory_dc, bitmap, 0, height, buffer, ctypes.byref(bmi), 0)
        if rows != height:
            return None
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR).copy()
    finally:
        if old_bitmap:
            try:
                gdi32.SelectObject(memory_dc, old_bitmap)
            except Exception:
                pass
        if bitmap:
            try:
                gdi32.DeleteObject(bitmap)
            except Exception:
                pass
        if memory_dc:
            try:
                gdi32.DeleteDC(memory_dc)
            except Exception:
                pass
        user32.ReleaseDC(hwnd, window_dc)


def _looks_like_valid_capture(image: np.ndarray) -> bool:
    if image is None or image.ndim != 3 or image.shape[0] < 50 or image.shape[1] < 50:
        return False
    # Sample a strided view: a full-resolution grayscale conversion of a
    # 1920x1020 frame costs more than the capture itself.
    step = max(1, min(image.shape[0], image.shape[1]) // 240)
    sample = image[::step, ::step]
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    nonzero_ratio = float(np.count_nonzero(gray > 3)) / float(gray.size)
    if nonzero_ratio < 0.02:
        return False
    if float(gray.std()) < 1.0:
        return False
    return True


def _window_summary(window: WindowInfo) -> str:
    return (
        f"{window.title}({window.process_name}) "
        f"x={window.left} y={window.top} "
        f"w={window.width} h={window.height} "
        f"minimized={window.minimized}"
    )


def _target_candidates(window_title: str) -> tuple[str, ...]:
    if window_title == DEFAULT_WINDOW_TITLE:
        return DEFAULT_WINDOW_CANDIDATES
    return (window_title, *DEFAULT_WINDOW_CANDIDATES)


def _process_name(process_id: int) -> str:
    if process_id <= 0:
        return ""

    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not handle:
        return ""

    try:
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return basename(buffer.value)
        return ""
    finally:
        kernel32.CloseHandle(handle)


def read_image(path: str | Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ext = output.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Cannot encode image as {ext}")
    encoded.tofile(str(output))
    return output
