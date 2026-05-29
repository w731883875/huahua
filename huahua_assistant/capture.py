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


def capture_target(
    window_title: str = DEFAULT_WINDOW_TITLE,
    fallback_to_screen: bool = True,
    fallback_if_target_uncaptureable: bool = False,
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
        activate_window(window.hwnd)
        time.sleep(0.35)
        refreshed_matches = find_matching_windows(candidates)
        refreshed_captureable = [item for item in refreshed_matches if item.captureable]
        if refreshed_captureable:
            window = max(refreshed_captureable, key=lambda item: item.width * item.height)
        region = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
        }
        return CaptureResult(
            image=capture_screen(region),
            source=f"window:{window.title} process:{window.process_name}",
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
