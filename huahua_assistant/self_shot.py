"""诊断工具：在**进程内**把自己的窗口拍成图片。

为什么必须同进程：``SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`` 生效后，
这个窗口对所有抓取路径都是纯黑（``PrintWindow`` 实测 mean=0.0、uniq=1），
而该接口按文档只能在窗口所属进程调用（跨进程实测返回 False）。
于是「界面黑屏」到底是①内容没画出来 还是②只是被免疫排除掉，从进程外无法区分。

用法：``HUAHUA_SELF_SHOT=输出目录`` 启动工具，启动 12 秒后自动拍一组图。
拍完立刻把亲和度恢复成 17，不会破坏「对截屏免疫」的既有行为。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import time
from pathlib import Path

WDA_NONE = 0x0
WDA_EXCLUDEFROMCAPTURE = 0x11

GA_ROOT = 2

__all__ = ["capture_self_windows"]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def _print_window(hwnd: int) -> "tuple[bytes, int, int] | None":
    """窗口自绘内容 → BGRA 字节串。失败返回 None。"""

    import numpy as np

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None
    hdc = user32.GetWindowDC(hwnd)
    if not hdc:
        return None
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, width, height)
    gdi32.SelectObject(mem, bmp)
    user32.PrintWindow(hwnd, mem, 2)  # PW_RENDERFULLCONTENT
    header = _BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    header.biWidth = width
    header.biHeight = -height
    header.biPlanes = 1
    header.biBitCount = 32
    buffer = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(mem, bmp, 0, height, buffer, ctypes.byref(header), 0)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)
    raw = buffer.raw
    array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return array, width, height


def _top_level_windows(root) -> list[tuple[int, str]]:
    """本进程所有可见顶层窗口（主窗 + 桌宠 + 浮层 + 对话框）。

    不用 ``root.winfo_children()`` 递归：桌宠窗口是 ``overrideredirect`` + layered，
    实测在 Tk 树里取到的句柄经 ``GetAncestor(GA_ROOT)`` 后拍不出内容，而且运行中
    还会被销毁重建。直接按进程枚举系统顶层窗口，漏不掉也不会拿到子窗口句柄。
    """

    import os

    import ctypes
    import ctypes.wintypes as wintypes

    user32 = ctypes.windll.user32
    pid = os.getpid()
    handles: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _callback(hwnd, _lparam):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != pid:
            return True
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        if not title:
            class_name = ctypes.create_unicode_buffer(128)
            user32.GetClassNameW(hwnd, class_name, 128)
            title = class_name.value or "untitled"
        handles.append((int(hwnd), title))
        return True

    user32.EnumWindows(_callback, 0)
    # 主窗口排第一，便于人工先看它
    main = user32.GetAncestor(root.winfo_id(), GA_ROOT)
    handles.sort(key=lambda item: 0 if item[0] == main else 1)
    return handles


def capture_self_windows(root, out_dir: str | Path) -> list[dict]:
    """把每个顶层窗口拍一张存盘，返回每张的统计信息。

    顺序很重要：先清掉免疫 → 抓 → 再恢复。中途异常也要恢复，否则窗口会永远
    暴露在截屏里（用户开着录屏就会把界面录进去）。
    """

    import cv2
    import numpy as np

    user32 = ctypes.windll.user32
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.c_uint]
    user32.SetWindowDisplayAffinity.restype = ctypes.c_bool

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for handle, title in _top_level_windows(root):
        affinity = ctypes.c_uint(0)
        user32.GetWindowDisplayAffinity(handle, ctypes.byref(affinity))
        cleared = False
        try:
            if affinity.value != WDA_NONE:
                cleared = bool(user32.SetWindowDisplayAffinity(handle, WDA_NONE))
                time.sleep(0.25)  # 让 DWM 重新合成一帧
            shot = _print_window(handle)
            if shot is None:
                results.append({"title": title, "hwnd": hex(handle), "error": "print_window_failed"})
                continue
            array, width, height = shot
            bgr = array[:, :, :3].copy()
            safe = "".join(ch if ch.isalnum() else "_" for ch in title)[:32] or "window"
            path = target / f"{safe}.png"
            cv2.imwrite(str(path), bgr)
            results.append(
                {
                    "title": title,
                    "hwnd": hex(handle),
                    "path": str(path),
                    "size": [width, height],
                    "mean": round(float(bgr.mean()), 1),
                    "std": round(float(bgr.std()), 1),
                    "unique_colors": int(len(np.unique(bgr.reshape(-1, 3), axis=0))),
                    "wda_before": affinity.value,
                    "wda_cleared": cleared,
                }
            )
        except Exception as error:  # 单个窗口拍失败不能连累其它窗口，但必须留证据
            results.append(
                {"title": title, "hwnd": hex(handle), "error": f"{type(error).__name__}: {error}"}
            )
        finally:
            if cleared:
                user32.SetWindowDisplayAffinity(handle, WDA_EXCLUDEFROMCAPTURE)
    return results
