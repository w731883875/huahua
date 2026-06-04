from __future__ import annotations

import ctypes
import json
import sys
import time
import tkinter as tk
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageTk

from .game_model import (
    BrushType,
    Paint,
    PuzzleState,
    current_rule_is_pending,
    recommend_affixes,
    recommend_brush_types,
)
from .models import Rect, SceneObservation
from .pet_brain import evaluate_pet_state


TRANSPARENT_COLOR = "#010203"
TRANSPARENT_RGB = (1, 2, 3)
CAPTURE_HIDE_SETTLE_MS = 80
PET_IDLE_ASSET = "qnb.png"
PET_ACTIVE_ASSET = "maonang.png"
PET_TARGET_HEIGHT = 190
PET_SCREEN_MARGIN = 18
PET_POSITION_PATH = Path("runtime/desktop_pet_position.json")


@dataclass(frozen=True)
class PetMarker:
    key: str
    x: int
    y: int
    label: str
    kind: str
    priority: int = 1
    size: int = 58
    caption: str = ""


class DesktopPetMarkerLayer:
    """Small click-through coordinate markers anchored to game-window detections."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.windows: dict[str, tk.Toplevel] = {}
        self.canvases: dict[str, tk.Canvas] = {}
        self.wndprocs: dict[int, tuple[int, object]] = {}
        self.phase = False

    def render(self, markers: Iterable[PetMarker]) -> None:
        marker_list = list(markers)
        active = {marker.key for marker in marker_list}
        for key in list(self.windows):
            if key not in active:
                self._destroy_marker(key)
        for marker in marker_list:
            self._render_marker(marker)

    def hide(self) -> int:
        hidden = 0
        for window in self.windows.values():
            try:
                window.withdraw()
                hidden += 1
            except tk.TclError:
                pass
        return hidden

    def hide_for_capture(self, settle_ms: int = CAPTURE_HIDE_SETTLE_MS) -> None:
        hidden = self.hide()
        if hidden > 0:
            _settle_tk_compositor(self.root, settle_ms)

    def destroy(self) -> None:
        for key in list(self.windows):
            self._destroy_marker(key)
        self._restore_window_procs()

    def _render_marker(self, marker: PetMarker) -> None:
        window, canvas = self._ensure_marker(marker.key, marker.size)
        left = int(marker.x - marker.size / 2)
        top = int(marker.y - marker.size / 2)
        try:
            window.deiconify()
            window.geometry(f"{marker.size}x{marker.size}+{left}+{top}")
            window.attributes("-topmost", True)
        except tk.TclError:
            return
        self._draw_marker(canvas, marker)
        self._make_clickthrough(window, canvas)

    def _ensure_marker(self, key: str, size: int) -> tuple[tk.Toplevel, tk.Canvas]:
        window = self.windows.get(key)
        canvas = self.canvases.get(key)
        if window is not None and canvas is not None and window.winfo_exists():
            return window, canvas
        window = tk.Toplevel(self.root)
        window.title(f"Huahua Pet Marker {key}")
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.attributes("-alpha", 0.92)
        window.configure(bg=TRANSPARENT_COLOR)
        try:
            window.attributes("-transparentcolor", TRANSPARENT_COLOR)
        except tk.TclError:
            pass
        canvas = tk.Canvas(window, bg=TRANSPARENT_COLOR, width=size, height=size, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        self.windows[key] = window
        self.canvases[key] = canvas
        return window, canvas

    def _destroy_marker(self, key: str) -> None:
        window = self.windows.pop(key, None)
        self.canvases.pop(key, None)
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass

    def _draw_marker(self, canvas: tk.Canvas, marker: PetMarker) -> None:
        size = marker.size
        canvas.configure(width=size, height=size)
        canvas.delete("all")
        pulse = 3 if self.phase else 0
        colors = _marker_colors(marker.kind, marker.priority)
        pad = 6 - min(3, pulse)
        canvas.create_oval(
            pad,
            pad,
            size - pad,
            size - pad,
            fill=colors[0],
            outline=colors[1],
            width=4 + pulse,
        )
        canvas.create_oval(
            size * 0.22,
            size * 0.18,
            size * 0.78,
            size * 0.76,
            fill=colors[2],
            outline="",
        )
        canvas.create_text(
            size / 2,
            size / 2,
            text=marker.label,
            fill=colors[3],
            font=("Microsoft YaHei UI", max(10, int(size * 0.27)), "bold"),
        )
        caption = marker.caption or _marker_kind_caption(marker.kind)
        canvas.create_text(
            size / 2,
            size - 10,
            text=_fit_marker_caption(caption),
            fill=colors[3],
            font=("Microsoft YaHei UI", 7, "bold"),
        )

    def _make_clickthrough(self, window: tk.Toplevel, canvas: tk.Canvas) -> None:
        _make_clickthrough_widgets((window, canvas), self.wndprocs)

    def _restore_window_procs(self) -> None:
        _restore_window_procs(self.wndprocs)


class DesktopPetActor:
    """Persistent click-through desktop pet shown independently from markers."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.window: tk.Toplevel | None = None
        self.label: tk.Label | None = None
        self.bubble_label: tk.Label | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.photo_asset_name: str | None = None
        self.wndprocs: dict[int, tuple[int, object]] = {}
        self.active = False
        self.visible = False
        self.draggable = True
        self.manual_position: tuple[int, int] | None = _load_pet_position()
        self.drag_offset: tuple[int, int] | None = None
        self.last_render_key: tuple | None = None

    def show(self, *, active: bool = False, window_info=None, pet_state=None) -> None:
        self.active = active
        if not self._ensure_window():
            return
        self._load_image(active)
        if self.window is None or self.label is None or self.bubble_label is None or self.photo is None:
            return
        message = _pet_state_message(pet_state, active)
        bubble_bg, bubble_fg = _pet_state_palette(pet_state)
        self.bubble_label.configure(text=message, bg=bubble_bg, fg=bubble_fg)
        self.bubble_label.update_idletasks()
        image_width = int(self.photo.width())
        image_height = int(self.photo.height())
        bubble_width = max(84, int(self.bubble_label.winfo_reqwidth()))
        bubble_height = int(self.bubble_label.winfo_reqheight())
        width = max(image_width, bubble_width)
        height = image_height + bubble_height + 4
        if self.manual_position is None:
            left, top = _pet_position(self.root, width, height, window_info)
        else:
            left, top = _clamp_pet_position(self.root, self.manual_position[0], self.manual_position[1], width, height)
        render_key = (
            bool(active),
            message,
            bubble_bg,
            bubble_fg,
            self.photo_asset_name,
            left,
            top,
            width,
            height,
            bool(self.visible),
        )
        if self.last_render_key == render_key:
            self._apply_input_mode()
            return
        try:
            self.label.configure(image=self.photo)
            self.window.geometry(f"{width}x{height}+{left}+{top}")
            self.window.deiconify()
            self.window.lift()
            self.window.attributes("-topmost", True)
            self.visible = True
            self.last_render_key = render_key
        except tk.TclError:
            return
        self._apply_input_mode()

    def set_draggable(self, enabled: bool) -> None:
        self.draggable = bool(enabled)
        self._apply_input_mode()

    def hide(self) -> int:
        if self.window is None:
            return 0
        try:
            if self.window.state() != "withdrawn":
                self.window.withdraw()
                self.visible = False
                self.last_render_key = None
                return 1
        except tk.TclError:
            pass
        return 0

    def hide_for_capture(self, settle_ms: int = CAPTURE_HIDE_SETTLE_MS) -> None:
        hidden = self.hide()
        if hidden > 0:
            _settle_tk_compositor(self.root, settle_ms)

    def destroy(self) -> None:
        _restore_window_procs(self.wndprocs)
        if self.window is not None:
            try:
                self.window.destroy()
            except tk.TclError:
                pass
        self.window = None
        self.label = None
        self.bubble_label = None
        self.photo = None
        self.photo_asset_name = None
        self.drag_offset = None
        self.last_render_key = None
        self.visible = False

    def _ensure_window(self) -> bool:
        if self.window is not None and self.window.winfo_exists():
            return True
        try:
            window = tk.Toplevel(self.root)
            window.title("Huahua Desktop Pet")
            window.overrideredirect(True)
            window.attributes("-topmost", True)
            window.configure(bg=TRANSPARENT_COLOR)
            try:
                window.attributes("-transparentcolor", TRANSPARENT_COLOR)
            except tk.TclError:
                pass
            bubble_label = tk.Label(
                window,
                bg="#10202a",
                fg="#f8fdff",
                bd=0,
                highlightthickness=0,
                padx=10,
                pady=4,
                font=("Microsoft YaHei UI", 10, "bold"),
            )
            label = tk.Label(window, bg=TRANSPARENT_COLOR, bd=0, highlightthickness=0)
            bubble_label.pack(side=tk.TOP, anchor=tk.CENTER, pady=(0, 2))
            label.pack(side=tk.TOP, anchor=tk.CENTER)
            for widget in (window, bubble_label, label):
                widget.bind("<ButtonPress-1>", self._start_drag, add="+")
                widget.bind("<B1-Motion>", self._drag_to, add="+")
                widget.bind("<ButtonRelease-1>", self._finish_drag, add="+")
            self.window = window
            self.label = label
            self.bubble_label = bubble_label
            return True
        except tk.TclError:
            return False

    def _load_image(self, active: bool) -> None:
        asset_name = PET_ACTIVE_ASSET if active else PET_IDLE_ASSET
        if self.photo is not None and self.photo_asset_name == asset_name:
            return
        image = _load_pet_image(asset_name)
        self.photo = ImageTk.PhotoImage(image)
        self.photo_asset_name = asset_name

    def _apply_input_mode(self) -> None:
        if self.window is None or not self.window.winfo_exists():
            return
        if self.draggable:
            _make_desktop_pet_window_interactive(self.window, self.wndprocs)
        else:
            _make_desktop_pet_window_clickthrough(self.window, self.wndprocs)

    def _start_drag(self, event: tk.Event) -> str | None:
        if not self.draggable or self.window is None:
            return None
        try:
            self.drag_offset = (int(event.x_root) - self.window.winfo_x(), int(event.y_root) - self.window.winfo_y())
            self.window.lift()
        except tk.TclError:
            self.drag_offset = None
        return "break"

    def _drag_to(self, event: tk.Event) -> str | None:
        if not self.draggable or self.window is None or self.drag_offset is None:
            return None
        try:
            width = max(1, self.window.winfo_width())
            height = max(1, self.window.winfo_height())
            left = int(event.x_root) - self.drag_offset[0]
            top = int(event.y_root) - self.drag_offset[1]
            left, top = _clamp_pet_position(self.root, left, top, width, height)
            self.window.geometry(f"+{left}+{top}")
            self.manual_position = (left, top)
        except tk.TclError:
            return None
        return "break"

    def _finish_drag(self, _event: tk.Event) -> str | None:
        if self.draggable and self.window is not None:
            try:
                self.manual_position = (self.window.winfo_x(), self.window.winfo_y())
                _save_pet_position(self.manual_position)
            except tk.TclError:
                pass
        self.drag_offset = None
        return "break" if self.draggable else None


def _legacy_install_desktop_pet_markers(app_cls: type) -> None:
    """Patch HuahuaApp to use small coordinate markers instead of a full overlay."""

    if getattr(app_cls, "_desktop_pet_marker_patch_installed", False):
        return

    original_init = app_cls.__init__
    original_on_close = app_cls._on_close
    original_restore_app_window = app_cls._restore_app_window
    original_render_auto_follow_result = app_cls._render_auto_follow_result
    original_toggle_overlay = app_cls.toggle_overlay
    original_destroy_overlay = app_cls._destroy_overlay_window
    original_capture_async = app_cls.capture_async
    original_quick_capture_async = app_cls.quick_capture_async
    original_auto_follow_tick = app_cls._auto_follow_tick

    def layer(self) -> DesktopPetMarkerLayer:
        existing = getattr(self, "_desktop_pet_layer", None)
        if existing is None:
            existing = DesktopPetMarkerLayer(self)
            setattr(self, "_desktop_pet_layer", existing)
        return existing

    def actor(self) -> DesktopPetActor:
        existing = getattr(self, "_desktop_pet_actor", None)
        if existing is None:
            existing = DesktopPetActor(self)
            setattr(self, "_desktop_pet_actor", existing)
        return existing

    def show_actor(self) -> None:
        try:
            active = bool(getattr(self, "recommendations", None) or getattr(self, "install_plan", None))
            actor(self).show(active=active, window_info=getattr(self, "last_capture_window", None))
        except tk.TclError:
            return

    def hide_markers(self) -> None:
        existing = getattr(self, "_desktop_pet_layer", None)
        if existing is not None:
            existing.hide_for_capture()

    def hide_marker_surfaces(self) -> None:
        hidden = 0
        existing_layer = getattr(self, "_desktop_pet_layer", None)
        if existing_layer is not None:
            hidden += existing_layer.hide()
        if hidden > 0:
            _settle_tk_compositor(self, CAPTURE_HIDE_SETTLE_MS)

    def patched_toggle_overlay(self) -> None:
        if self.overlay_var.get():
            self._rerender_side_panel()
            self.status_var.set("桌宠坐标提示已开启：小标记会贴到推荐位置，鼠标穿透，不接管点击。")
            self._set_notice("桌宠坐标提示已开启：请用窗口化/无边框全屏；识别前会隐藏标记避免污染截图。", "info")
            self._refresh_overlay()
        else:
            self._destroy_overlay_window()
            self._rerender_side_panel()
            self.status_var.set("桌宠坐标提示已关闭。")
            self._set_notice("桌宠坐标提示已关闭。", "info")

    def patched_destroy_overlay(self) -> None:
        existing = getattr(self, "_desktop_pet_layer", None)
        if existing is not None:
            existing.destroy()
            setattr(self, "_desktop_pet_layer", None)
        original_destroy_overlay(self)

    def patched_refresh_overlay(self) -> None:
        if not self.overlay_var.get():
            self._destroy_overlay_window()
            return
        markers = _build_markers(self)
        if not markers:
            layer(self).hide()
        else:
            pet_layer = layer(self)
            pet_layer.phase = bool(getattr(self, "overlay_phase", False))
            pet_layer.render(markers)
        if getattr(self, "overlay_after_id", None) is None:
            self.overlay_after_id = self.after(560, self._overlay_tick)

    def patched_overlay_tick(self) -> None:
        self.overlay_after_id = None
        if not self.overlay_var.get():
            self._destroy_overlay_window()
            return
        self.overlay_phase = not bool(getattr(self, "overlay_phase", False))
        self._refresh_overlay()

    def patched_capture_async(self) -> None:
        hide_markers(self)
        original_capture_async(self)

    def patched_quick_capture_async(self) -> None:
        hide_markers(self)
        original_quick_capture_async(self)

    def patched_auto_follow_tick(self) -> None:
        hide_markers(self)
        original_auto_follow_tick(self)

    app_cls.toggle_overlay = patched_toggle_overlay
    app_cls._destroy_overlay_window = patched_destroy_overlay
    app_cls._refresh_overlay = patched_refresh_overlay
    app_cls._overlay_tick = patched_overlay_tick
    app_cls.capture_async = patched_capture_async
    app_cls.quick_capture_async = patched_quick_capture_async
    app_cls._auto_follow_tick = patched_auto_follow_tick

    # Keep a reference for debugging and to make it obvious that this patch was installed.
    app_cls._desktop_pet_marker_patch_installed = True
    app_cls._desktop_pet_original_toggle_overlay = original_toggle_overlay


def _build_markers(app) -> list[PetMarker]:
    scene: SceneObservation | None = getattr(app, "current_scene", None)
    state: PuzzleState | None = getattr(app, "current_state", None)
    window = getattr(app, "last_capture_window", None)
    if scene is None or state is None or window is None:
        return []
    if _coordinate_markers_are_unsafe(app):
        return []

    markers: list[PetMarker] = []
    markers.extend(_board_markers(app, scene))
    markers.extend(_backpack_markers(app, scene))
    markers.extend(_brush_markers(app, scene, state))
    return markers[:5]


def _coordinate_markers_are_unsafe(app) -> bool:
    quality = getattr(app, "last_capture_quality", None)
    if not isinstance(quality, dict) or quality.get("ok", True):
        return False
    return "capture_source_composited" in str(quality.get("reason") or "")


def _settle_tk_compositor(root: tk.Misc, settle_ms: int) -> None:
    _flush_tk(root)
    _flush_dwm()
    if settle_ms > 0:
        time.sleep(settle_ms / 1000.0)
    _flush_tk(root)
    _flush_dwm()


def _flush_tk(root: tk.Misc) -> None:
    try:
        root.update_idletasks()
    except tk.TclError:
        pass


def _flush_dwm() -> None:
    try:
        ctypes.windll.dwmapi.DwmFlush()
    except Exception:
        pass


def _make_clickthrough_widgets(widgets: Iterable[tk.Misc], wndprocs: dict[int, tuple[int, object]]) -> None:
    try:
        user32 = ctypes.windll.user32
        GWL_EXSTYLE = -20
        GWLP_WNDPROC = -4
        WM_NCHITTEST = 0x0084
        HTTRANSPARENT = -1
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOZORDER = 0x0004
        SWP_FRAMECHANGED = 0x0020

        get_window_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_window_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        get_window_long.restype = ctypes.c_ssize_t
        set_window_long.restype = ctypes.c_ssize_t
        call_window_proc = user32.CallWindowProcW
        call_window_proc.restype = ctypes.c_ssize_t
        wndproc_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )

        for widget in widgets:
            hwnd = int(widget.winfo_id())
            style = get_window_long(hwnd, GWL_EXSTYLE)
            set_window_long(
                hwnd,
                GWL_EXSTYLE,
                style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            )
            if hwnd not in wndprocs:
                original_proc = get_window_long(hwnd, GWLP_WNDPROC)

                def transparent_proc(window_hwnd, msg, wparam, lparam, *, original=original_proc):
                    if msg == WM_NCHITTEST:
                        return HTTRANSPARENT
                    return call_window_proc(original, window_hwnd, msg, wparam, lparam)

                callback = wndproc_type(transparent_proc)
                set_window_long(hwnd, GWLP_WNDPROC, ctypes.cast(callback, ctypes.c_void_p).value)
                wndprocs[hwnd] = (int(original_proc), callback)
            user32.SetWindowPos(
                hwnd,
                0,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED,
            )
    except Exception:
        return


def _make_desktop_pet_window_clickthrough(window: tk.Toplevel, wndprocs: dict[int, tuple[int, object]]) -> None:
    try:
        user32 = ctypes.windll.user32
        GWL_EXSTYLE = -20
        GWLP_WNDPROC = -4
        WM_NCHITTEST = 0x0084
        HTTRANSPARENT = -1
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOZORDER = 0x0004
        SWP_FRAMECHANGED = 0x0020
        LWA_COLORKEY = 0x00000001

        hwnd = int(window.winfo_id())
        get_window_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_window_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        get_window_long.restype = ctypes.c_ssize_t
        set_window_long.restype = ctypes.c_ssize_t
        style = get_window_long(hwnd, GWL_EXSTYLE)
        set_window_long(
            hwnd,
            GWL_EXSTYLE,
            style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
        )

        colorref = TRANSPARENT_RGB[0] | (TRANSPARENT_RGB[1] << 8) | (TRANSPARENT_RGB[2] << 16)
        user32.SetLayeredWindowAttributes.argtypes = (
            wintypes.HWND,
            wintypes.COLORREF,
            wintypes.BYTE,
            wintypes.DWORD,
        )
        user32.SetLayeredWindowAttributes.restype = wintypes.BOOL
        user32.SetLayeredWindowAttributes(hwnd, colorref, 255, LWA_COLORKEY)

        call_window_proc = user32.CallWindowProcW
        call_window_proc.restype = ctypes.c_ssize_t
        wndproc_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        if hwnd not in wndprocs:
            original_proc = get_window_long(hwnd, GWLP_WNDPROC)

            def transparent_proc(window_hwnd, msg, wparam, lparam, *, original=original_proc):
                if msg == WM_NCHITTEST:
                    return HTTRANSPARENT
                return call_window_proc(original, window_hwnd, msg, wparam, lparam)

            callback = wndproc_type(transparent_proc)
            set_window_long(hwnd, GWLP_WNDPROC, ctypes.cast(callback, ctypes.c_void_p).value)
            wndprocs[hwnd] = (int(original_proc), callback)

        user32.SetWindowPos(
            hwnd,
            0,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED,
        )
    except Exception:
        return


def _make_desktop_pet_window_interactive(window: tk.Toplevel, wndprocs: dict[int, tuple[int, object]]) -> None:
    _restore_window_procs(wndprocs)
    try:
        user32 = ctypes.windll.user32
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOZORDER = 0x0004
        SWP_FRAMECHANGED = 0x0020
        LWA_COLORKEY = 0x00000001

        hwnd = int(window.winfo_id())
        get_window_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_window_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        get_window_long.restype = ctypes.c_ssize_t
        set_window_long.restype = ctypes.c_ssize_t
        style = get_window_long(hwnd, GWL_EXSTYLE)
        style |= WS_EX_LAYERED | WS_EX_TOOLWINDOW
        style &= ~WS_EX_TRANSPARENT
        style &= ~WS_EX_NOACTIVATE
        set_window_long(hwnd, GWL_EXSTYLE, style)

        colorref = TRANSPARENT_RGB[0] | (TRANSPARENT_RGB[1] << 8) | (TRANSPARENT_RGB[2] << 16)
        user32.SetLayeredWindowAttributes.argtypes = (
            wintypes.HWND,
            wintypes.COLORREF,
            wintypes.BYTE,
            wintypes.DWORD,
        )
        user32.SetLayeredWindowAttributes.restype = wintypes.BOOL
        user32.SetLayeredWindowAttributes(hwnd, colorref, 255, LWA_COLORKEY)
        user32.SetWindowPos(
            hwnd,
            0,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED,
        )
    except Exception:
        return


def _restore_window_procs(wndprocs: dict[int, tuple[int, object]]) -> None:
    if not wndprocs:
        return
    try:
        user32 = ctypes.windll.user32
        GWLP_WNDPROC = -4
        set_window_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        set_window_long.restype = ctypes.c_ssize_t
        for hwnd, (original_proc, _callback) in list(wndprocs.items()):
            try:
                set_window_long(hwnd, GWLP_WNDPROC, original_proc)
            except Exception:
                pass
    finally:
        wndprocs.clear()


def _load_pet_image(asset_name: str) -> Image.Image:
    path = _pet_asset_path(asset_name)
    if path.exists():
        image = Image.open(path).convert("RGBA")
    else:
        image = _fallback_pet_image()
    if image.height <= 0:
        return image
    ratio = PET_TARGET_HEIGHT / float(image.height)
    width = max(1, int(image.width * ratio))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = image.resize((width, PET_TARGET_HEIGHT), resampling)
    return _rgba_to_transparent_color_key(resized)


def _rgba_to_transparent_color_key(image: Image.Image) -> Image.Image:
    """Tk on Windows can render PNG alpha as a black box; use color-key transparency instead."""

    rgba = image.convert("RGBA")
    keyed = Image.new("RGBA", rgba.size, (*TRANSPARENT_RGB, 255))
    opaque = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    opaque.alpha_composite(rgba)
    alpha = rgba.getchannel("A")
    mask = alpha.point(lambda value: 255 if value >= 96 else 0)
    keyed.paste(opaque, mask=mask)
    return keyed.convert("RGB")


def _pet_asset_path(asset_name: str) -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / "huahua_assistant" / "assets" / "pets" / asset_name
    return Path(__file__).resolve().parent / "assets" / "pets" / asset_name


def _fallback_pet_image() -> Image.Image:
    image = Image.new("RGBA", (160, 190), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((26, 46, 134, 154), fill=(235, 247, 255, 245), outline=(100, 230, 255, 255), width=5)
    draw.polygon(((42, 58), (58, 10), (78, 62)), fill=(235, 247, 255, 245), outline=(100, 230, 255, 255))
    draw.polygon(((82, 62), (104, 10), (120, 58)), fill=(235, 247, 255, 245), outline=(100, 230, 255, 255))
    draw.ellipse((55, 90, 68, 103), fill=(40, 56, 68, 255))
    draw.ellipse((92, 90, 105, 103), fill=(40, 56, 68, 255))
    draw.arc((61, 95, 99, 130), 20, 160, fill=(40, 56, 68, 255), width=3)
    draw.text((42, 162), "Huahua", fill=(235, 247, 255, 245))
    return image


def _pet_position(root: tk.Tk, width: int, height: int, window_info) -> tuple[int, int]:
    screen_left, screen_top, screen_width, screen_height = _virtual_screen(root)
    screen_right = screen_left + screen_width
    screen_bottom = screen_top + screen_height
    left = screen_right - width - PET_SCREEN_MARGIN
    top = screen_bottom - height - PET_SCREEN_MARGIN

    if window_info is not None and getattr(window_info, "captureable", False):
        right_space = screen_right - int(window_info.right)
        left_space = int(window_info.left) - screen_left
        if right_space >= width + PET_SCREEN_MARGIN * 2:
            left = int(window_info.right) + PET_SCREEN_MARGIN
        elif left_space >= width + PET_SCREEN_MARGIN * 2:
            left = int(window_info.left) - width - PET_SCREEN_MARGIN
        top = int(window_info.top + window_info.height * 0.66 - height / 2)

    left = _clamp(left, screen_left + PET_SCREEN_MARGIN, screen_right - width - PET_SCREEN_MARGIN)
    top = _clamp(top, screen_top + PET_SCREEN_MARGIN, screen_bottom - height - PET_SCREEN_MARGIN)
    return left, top


def _clamp_pet_position(root: tk.Tk, left: int, top: int, width: int, height: int) -> tuple[int, int]:
    screen_left, screen_top, screen_width, screen_height = _virtual_screen(root)
    screen_right = screen_left + screen_width
    screen_bottom = screen_top + screen_height
    return (
        _clamp(left, screen_left, max(screen_left, screen_right - width)),
        _clamp(top, screen_top, max(screen_top, screen_bottom - height)),
    )


def _load_pet_position() -> tuple[int, int] | None:
    try:
        data = json.loads(PET_POSITION_PATH.read_text(encoding="utf-8"))
        left = int(data.get("left"))
        top = int(data.get("top"))
    except Exception:
        return None
    return left, top


def _save_pet_position(position: tuple[int, int]) -> None:
    try:
        PET_POSITION_PATH.parent.mkdir(parents=True, exist_ok=True)
        PET_POSITION_PATH.write_text(
            json.dumps({"left": int(position[0]), "top": int(position[1])}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return


def _find_toolbar(root: tk.Misc) -> tk.Frame | None:
    for child in root.winfo_children():
        if isinstance(child, tk.Frame):
            return child
    return None


def _virtual_screen(root: tk.Tk) -> tuple[int, int, int, int]:
    try:
        return (
            int(root.winfo_vrootx()),
            int(root.winfo_vrooty()),
            int(root.winfo_vrootwidth()),
            int(root.winfo_vrootheight()),
        )
    except tk.TclError:
        return 0, 0, int(root.winfo_screenwidth()), int(root.winfo_screenheight())


def _clamp(value: int, low: int, high: int) -> int:
    if high < low:
        return low
    return max(low, min(high, int(value)))


def _board_markers(app, scene: SceneObservation) -> list[PetMarker]:
    markers: list[PetMarker] = []
    recommendations = list(getattr(app, "recommendations", []) or [])
    for index, rec in enumerate(recommendations[:2], start=1):
        point = app._screen_point_for_board_cell(rec.row, rec.col)
        if point is None:
            continue
        markers.append(
            PetMarker(
                key=f"board-{index}",
                x=point[0],
                y=point[1],
                label=str(index),
                kind="board",
                priority=index,
                size=_cell_marker_size(scene.board.tile(rec.row, rec.col).rect),
                caption=_recommendation_caption(rec),
            )
        )
    return markers


def _backpack_markers(app, _scene: SceneObservation) -> list[PetMarker]:
    markers: list[PetMarker] = []
    desired_paint = _desired_backpack_paint(app)
    if desired_paint is None:
        return markers
    index = app._visible_backpack_index_for_paint(desired_paint)
    if index is None:
        return markers
    point = app._screen_point_for_backpack_slot(index)
    if point is None:
        return markers
    markers.append(
        PetMarker(
            key="backpack-1",
            x=point[0],
            y=point[1],
            label="B",
            kind="backpack",
            priority=1,
            size=50,
            caption=_backpack_caption(app),
        )
    )
    return markers


def _brush_markers(app, scene: SceneObservation, state: PuzzleState) -> list[PetMarker]:
    if scene.brush_options is None or not current_rule_is_pending(state):
        return []
    scenario = getattr(app, "brush_scenario", None)
    if scenario is None:
        return []
    panel = scene.brush_options.panel_rect
    window = app.last_capture_window
    if window is None:
        return []

    selected_type = getattr(app, "selected_brush_type", None)
    selected_affix = getattr(app, "selected_affix", None)
    if selected_type is None:
        options = tuple(scenario.stage1_options)
        if not options:
            return []
        recs = recommend_brush_types(state, options)
        best_rec = recs[0] if recs else None
        best = best_rec.brush_type if best_rec else options[0]
        index = _index_or_zero(options, best)
        point = _stage1_point(app, panel, index, len(options))
        if point is None:
            return []
        return [
            PetMarker(
                key="brush-stage1",
                x=point[0],
                y=point[1],
                label=_brush_short_label(best),
                kind="brush",
                priority=1,
                size=54,
                caption=_short_caption(best_rec.reason if best_rec else "推荐大类"),
            )
        ]
    if selected_affix is None and scenario.stage2_options:
        affixes = tuple(scenario.stage2_options)
        recs = recommend_affixes(state, affixes)
        best_rec = recs[0] if recs else None
        best_affix = best_rec.affix if best_rec else affixes[0]
        index = _index_or_zero(affixes, best_affix)
        point = _stage2_point(app, panel, index, len(affixes))
        if point is None:
            return []
        return [
            PetMarker(
                key="brush-stage2",
                x=point[0],
                y=point[1],
                label="F",
                kind="brush",
                priority=1,
                size=54,
                caption=_short_caption(best_rec.reason if best_rec else "推荐词条"),
            )
        ]
    return []


def _desired_backpack_paint(app) -> Paint | None:
    install_plan = getattr(app, "install_plan", None)
    if install_plan is not None and install_plan.steps:
        return install_plan.steps[0].paint
    recommendations = list(getattr(app, "recommendations", []) or [])
    if recommendations and recommendations[0].paint is not None:
        return recommendations[0].paint
    return None


def _backpack_caption(app) -> str:
    recommendations = list(getattr(app, "recommendations", []) or [])
    if recommendations:
        return _recommendation_caption(recommendations[0]) or "先点背包"
    install_plan = getattr(app, "install_plan", None)
    if install_plan is not None and install_plan.steps:
        return "先点背包"
    return "背包"


def _recommendation_caption(rec) -> str:
    explanation = getattr(rec, "explanation", None)
    short = getattr(explanation, "short", "") if explanation is not None else ""
    if short:
        return str(short)
    if getattr(rec, "completes_match", False):
        return "完成匹配"
    return "推荐"


def _marker_kind_caption(kind: str) -> str:
    if kind == "board":
        return "地块"
    if kind == "backpack":
        return "背包"
    if kind == "brush":
        return "词条"
    return "推荐"


def _short_caption(text: str, max_chars: int = 5) -> str:
    normalized = "".join(str(text or "").replace("，", ",").replace("；", ",").split())
    if not normalized:
        return "推荐"
    first = normalized.split(",", 1)[0].split("。", 1)[0].split("|", 1)[0]
    return first[:max_chars] or "推荐"


def _fit_marker_caption(text: str) -> str:
    return _short_caption(text, max_chars=5)


def _stage1_point(app, panel: Rect, index: int, count: int) -> tuple[int, int] | None:
    if count <= 1:
        x_ratio = 0.50
    else:
        x_ratio = (index + 0.5) / count
    return app._screen_point_for_capture_rect(panel, x_ratio, 0.34)


def _stage2_point(app, panel: Rect, index: int, count: int) -> tuple[int, int] | None:
    if count <= 1:
        y_ratio = 0.45
    else:
        y_ratio = 0.35 + min(index, count - 1) * 0.25
    return app._screen_point_for_capture_rect(panel, 0.58, y_ratio)


def _index_or_zero(items: tuple, target) -> int:
    try:
        return items.index(target)
    except ValueError:
        return 0


def _brush_short_label(brush_type: BrushType) -> str:
    if brush_type == BrushType.TRANSFORM:
        return "T"
    if brush_type == BrushType.STACK:
        return "S"
    if brush_type == BrushType.SPREAD:
        return "D"
    return "?"


def _cell_marker_size(rect: Rect) -> int:
    return max(42, min(62, int(min(rect.w, rect.h) * 0.52)))


def _marker_colors(kind: str, priority: int) -> tuple[str, str, str, str]:
    if kind == "backpack":
        return "#143b3e", "#64e6ff", "#0f5c64", "#e9fdff"
    if kind == "brush":
        return "#3a1648", "#ff8cff", "#5b1f76", "#fff2ff"
    if priority == 1:
        return "#473714", "#ffd84d", "#7c5c16", "#fff8d7"
    return "#162f43", "#64e6ff", "#214e68", "#e8fbff"


def _pet_state_is_active(pet_state) -> bool:
    mode = getattr(pet_state, "mode", "observing") if pet_state is not None else "observing"
    return mode not in {"observing"}


def _pet_state_message(pet_state, active: bool) -> str:
    message = str(getattr(pet_state, "message", "") or "").strip()
    if message:
        return message[:12]
    return "推荐中" if active else "观察中"


def _pet_state_palette(pet_state) -> tuple[str, str]:
    mode = getattr(pet_state, "mode", "observing") if pet_state is not None else "observing"
    if mode == "recommend_backpack":
        return "#10353a", "#dffcff"
    if mode == "recommend_board_cell":
        return "#44330f", "#fff4c5"
    if mode == "recommend_brush":
        return "#3a1745", "#fff0ff"
    if mode == "wait_for_drop":
        return "#123047", "#e6f6ff"
    if mode == "finish_hint":
        return "#143817", "#e7ffe4"
    return "#101820", "#eef7ff"


def install_desktop_pet_markers(app_cls: type) -> None:
    """Install the persistent desktop pet plus optional coordinate markers."""

    if getattr(app_cls, "_desktop_pet_patch_installed", False):
        return

    original_init = app_cls.__init__
    original_on_close = app_cls._on_close
    original_restore_app_window = app_cls._restore_app_window
    original_toggle_overlay = app_cls.toggle_overlay
    original_destroy_overlay = app_cls._destroy_overlay_window
    original_capture_async = app_cls.capture_async
    original_quick_capture_async = app_cls.quick_capture_async
    original_auto_follow_tick = app_cls._auto_follow_tick
    original_auto_follow_error = app_cls._auto_follow_error
    original_render_auto_follow_result = app_cls._render_auto_follow_result

    def actor(self) -> DesktopPetActor:
        existing = getattr(self, "_desktop_pet_actor", None)
        if existing is None:
            existing = DesktopPetActor(self)
            setattr(self, "_desktop_pet_actor", existing)
        return existing

    def layer(self) -> DesktopPetMarkerLayer:
        existing = getattr(self, "_desktop_pet_layer", None)
        if existing is None:
            existing = DesktopPetMarkerLayer(self)
            setattr(self, "_desktop_pet_layer", existing)
        return existing

    def show_actor(self) -> None:
        try:
            pet_state = evaluate_pet_state(self)
            setattr(self, "last_pet_state", pet_state)
            active = _pet_state_is_active(pet_state) or bool(getattr(self, "recommendations", None) or getattr(self, "install_plan", None))
            if hasattr(self, "desktop_pet_drag_var"):
                actor(self).set_draggable(bool(self.desktop_pet_drag_var.get()))
            actor(self).show(active=active, window_info=getattr(self, "last_capture_window", None), pet_state=pet_state)
        except tk.TclError:
            return

    def toggle_pet_drag(self) -> None:
        enabled = bool(getattr(self, "desktop_pet_drag_var", tk.BooleanVar(value=True)).get())
        actor(self).set_draggable(enabled)
        show_actor(self)
        if enabled:
            self.status_var.set("桌宠拖动已开启：可直接拖动桌宠到任意位置。")
            self._set_notice("桌宠拖动已开启：拖到合适位置后，可取消勾选恢复鼠标穿透。", "info")
        else:
            self.status_var.set("桌宠拖动已关闭：桌宠恢复鼠标穿透，不挡游戏点击。")
            self._set_notice("桌宠拖动已关闭：当前桌宠位置已保留，鼠标点击会穿透到游戏。", "success")

    def install_drag_toggle(self) -> None:
        if hasattr(self, "desktop_pet_drag_var"):
            return
        self.desktop_pet_drag_var = tk.BooleanVar(value=True)
        toolbar = _find_toolbar(self)
        if toolbar is None:
            return
        check = tk.Checkbutton(
            toolbar,
            text="拖动桌宠",
            variable=self.desktop_pet_drag_var,
            fg="#d7e3ee",
            bg="#101820",
            activeforeground="#ffffff",
            activebackground="#101820",
            selectcolor="#0d141b",
            command=lambda: toggle_pet_drag(self),
        )
        check.pack(side=tk.LEFT, padx=(8, 0))

    def hide_marker_surfaces(self) -> None:
        hidden = 0
        existing_layer = getattr(self, "_desktop_pet_layer", None)
        if existing_layer is not None:
            hidden += existing_layer.hide()
        existing_actor = getattr(self, "_desktop_pet_actor", None)
        if existing_actor is not None:
            hidden += existing_actor.hide()
        if hidden > 0:
            _settle_tk_compositor(self, CAPTURE_HIDE_SETTLE_MS)

    def patched_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        install_drag_toggle(self)
        self.after(250, lambda: show_actor(self))

    def patched_on_close(self) -> None:
        existing_actor = getattr(self, "_desktop_pet_actor", None)
        if existing_actor is not None:
            existing_actor.destroy()
            setattr(self, "_desktop_pet_actor", None)
        existing_layer = getattr(self, "_desktop_pet_layer", None)
        if existing_layer is not None:
            existing_layer.destroy()
            setattr(self, "_desktop_pet_layer", None)
        original_on_close(self)

    def patched_restore_app_window(self) -> None:
        original_restore_app_window(self)
        show_actor(self)

    def patched_toggle_overlay(self) -> None:
        original_toggle_overlay(self)
        if self.overlay_var.get():
            self._refresh_overlay()
        else:
            existing_layer = getattr(self, "_desktop_pet_layer", None)
            if existing_layer is not None:
                existing_layer.hide()
        show_actor(self)

    def patched_destroy_overlay(self) -> None:
        existing_layer = getattr(self, "_desktop_pet_layer", None)
        if existing_layer is not None:
            existing_layer.destroy()
            setattr(self, "_desktop_pet_layer", None)
        original_destroy_overlay(self)

    def patched_refresh_overlay(self) -> None:
        if not self.overlay_var.get():
            existing_layer = getattr(self, "_desktop_pet_layer", None)
            if existing_layer is not None:
                existing_layer.hide()
            show_actor(self)
            return
        markers = _build_markers(self)
        if markers:
            pet_layer = layer(self)
            pet_layer.phase = bool(getattr(self, "overlay_phase", False))
            pet_layer.render(markers)
        else:
            layer(self).hide()
        show_actor(self)
        if getattr(self, "overlay_after_id", None) is None:
            self.overlay_after_id = self.after(560, self._overlay_tick)

    def patched_overlay_tick(self) -> None:
        self.overlay_after_id = None
        if not self.overlay_var.get():
            existing_layer = getattr(self, "_desktop_pet_layer", None)
            if existing_layer is not None:
                existing_layer.hide()
            show_actor(self)
            return
        self.overlay_phase = not bool(getattr(self, "overlay_phase", False))
        self._refresh_overlay()

    def patched_capture_async(self) -> None:
        hide_marker_surfaces(self)
        original_capture_async(self)

    def patched_quick_capture_async(self) -> None:
        hide_marker_surfaces(self)
        original_quick_capture_async(self)

    def patched_auto_follow_tick(self) -> None:
        if not self.auto_follow_var.get() or self.auto_follow_busy or self._app_overlaps_game_window():
            original_auto_follow_tick(self)
            show_actor(self)
            return
        hide_marker_surfaces(self)
        original_auto_follow_tick(self)

    def patched_auto_follow_error(self, *args, **kwargs) -> None:
        original_auto_follow_error(self, *args, **kwargs)
        show_actor(self)

    def patched_render_auto_follow_result(self, *args, **kwargs) -> None:
        original_render_auto_follow_result(self, *args, **kwargs)
        show_actor(self)

    app_cls.__init__ = patched_init
    app_cls._on_close = patched_on_close
    app_cls._restore_app_window = patched_restore_app_window
    app_cls.toggle_overlay = patched_toggle_overlay
    app_cls._destroy_overlay_window = patched_destroy_overlay
    app_cls._refresh_overlay = patched_refresh_overlay
    app_cls._overlay_tick = patched_overlay_tick
    app_cls.capture_async = patched_capture_async
    app_cls.quick_capture_async = patched_quick_capture_async
    app_cls._auto_follow_tick = patched_auto_follow_tick
    app_cls._auto_follow_error = patched_auto_follow_error
    app_cls._render_auto_follow_result = patched_render_auto_follow_result

    app_cls._desktop_pet_patch_installed = True
    app_cls._desktop_pet_marker_patch_installed = True
    app_cls._desktop_pet_original_toggle_overlay = original_toggle_overlay
