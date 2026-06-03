from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass
from os.path import basename
from pathlib import Path
from typing import Iterable

import cv2
import mss
import numpy as np


DEFAULT_WINDOW_TITLE = "torchlight_infinite"
DEFAULT_WINDOW_CANDIDATES = ("torchlight_infinite", "Torchlight: Infinite", "Torchlight")


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


def capture_screen(region: dict[str, int] | None = None) -> np.ndarray:
    with mss.mss() as sct:
        target = region if region is not None else sct.monitors[1]
        shot = sct.grab(target)
    frame = np.asarray(shot)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def capture_window_content(window: WindowInfo) -> tuple[np.ndarray | None, str]:
    """Capture window-owned pixels first so desktop overlays do not enter recognition."""

    try:
        image = _capture_window_print(window)
    except Exception as exc:
        return None, f"printwindow_error:{type(exc).__name__}"
    if image is None:
        return None, "printwindow_failed"
    if not _looks_like_valid_capture(image):
        return None, "printwindow_invalid"
    return image, "printwindow"


def capture_target(
    window_title: str = DEFAULT_WINDOW_TITLE,
    fallback_to_screen: bool = True,
    fallback_if_target_uncaptureable: bool = False,
    activate: bool = True,
    activation_delay: float = 0.35,
) -> CaptureResult:
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
        if activate:
            activate_window(window.hwnd)
            if activation_delay > 0:
                time.sleep(activation_delay)
            refreshed_matches = find_matching_windows(candidates)
            refreshed_captureable = [item for item in refreshed_matches if item.captureable]
            if refreshed_captureable:
                window = max(refreshed_captureable, key=lambda item: item.width * item.height)
        internal_image, internal_method = capture_window_content(window)
        if internal_image is not None:
            return CaptureResult(
                image=internal_image,
                source=f"window-content:{window.title} process:{window.process_name} method:{internal_method}",
                window=window,
            )
        region = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
        }
        return CaptureResult(
            image=capture_screen(region),
            source=f"window-screen:{window.title} process:{window.process_name} internal_failed:{internal_method}",
            window=window,
        )

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


def find_window(title_contains: str) -> WindowInfo | None:
    matches = find_matching_windows([title_contains])
    if not matches:
        return None
    captureable = [window for window in matches if window.captureable]
    pool = captureable if captureable else matches
    return max(pool, key=lambda window: window.width * window.height)


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
                minimized=bool(user32.IsIconic(hwnd)),
            )
        )
        return True

    user32.EnumWindows(enum_windows_proc(callback), 0)
    return windows


def restore_window(hwnd: int) -> None:
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    user32.ShowWindow(hwnd, SW_RESTORE)


def activate_window(hwnd: int) -> None:
    restore_window(hwnd)
    user32 = ctypes.windll.user32
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)


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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
