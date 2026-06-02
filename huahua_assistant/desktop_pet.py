from __future__ import annotations

import ctypes
import tkinter as tk
from ctypes import wintypes
from dataclasses import dataclass
from typing import Iterable

from .game_model import (
    BrushType,
    Paint,
    PuzzleState,
    current_rule_is_pending,
    recommend_affixes,
    recommend_brush_types,
)
from .models import Rect, SceneObservation


TRANSPARENT_COLOR = "#010203"


@dataclass(frozen=True)
class PetMarker:
    key: str
    x: int
    y: int
    label: str
    kind: str
    priority: int = 1
    size: int = 58


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

    def hide(self) -> None:
        for window in self.windows.values():
            try:
                window.withdraw()
            except tk.TclError:
                pass

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
        if marker.kind == "board":
            canvas.create_text(size / 2, size - 10, text="cell", fill=colors[3], font=("Segoe UI", 7, "bold"))
        elif marker.kind == "backpack":
            canvas.create_text(size / 2, size - 10, text="bag", fill=colors[3], font=("Segoe UI", 7, "bold"))
        elif marker.kind == "brush":
            canvas.create_text(size / 2, size - 10, text="affix", fill=colors[3], font=("Segoe UI", 7, "bold"))

    def _make_clickthrough(self, window: tk.Toplevel, canvas: tk.Canvas) -> None:
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

            for hwnd in (window.winfo_id(), canvas.winfo_id()):
                style = get_window_long(hwnd, GWL_EXSTYLE)
                set_window_long(
                    hwnd,
                    GWL_EXSTYLE,
                    style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                )
                if int(hwnd) not in self.wndprocs:
                    original_proc = get_window_long(hwnd, GWLP_WNDPROC)

                    def transparent_proc(window_hwnd, msg, wparam, lparam, *, original=original_proc):
                        if msg == WM_NCHITTEST:
                            return HTTRANSPARENT
                        return call_window_proc(original, window_hwnd, msg, wparam, lparam)

                    callback = wndproc_type(transparent_proc)
                    set_window_long(hwnd, GWLP_WNDPROC, ctypes.cast(callback, ctypes.c_void_p).value)
                    self.wndprocs[int(hwnd)] = (int(original_proc), callback)
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

    def _restore_window_procs(self) -> None:
        if not self.wndprocs:
            return
        try:
            user32 = ctypes.windll.user32
            GWLP_WNDPROC = -4
            set_window_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            set_window_long.restype = ctypes.c_ssize_t
            for hwnd, (original_proc, _callback) in list(self.wndprocs.items()):
                try:
                    set_window_long(hwnd, GWLP_WNDPROC, original_proc)
                except Exception:
                    pass
        finally:
            self.wndprocs.clear()


def install_desktop_pet_markers(app_cls: type) -> None:
    """Patch HuahuaApp to use small coordinate markers instead of a full overlay."""

    if getattr(app_cls, "_desktop_pet_marker_patch_installed", False):
        return

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

    def hide_markers(self) -> None:
        existing = getattr(self, "_desktop_pet_layer", None)
        if existing is not None:
            existing.hide()
            try:
                self.update_idletasks()
            except tk.TclError:
                pass

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

    markers: list[PetMarker] = []
    markers.extend(_board_markers(app, scene))
    markers.extend(_backpack_markers(app, scene))
    markers.extend(_brush_markers(app, scene, state))
    return markers[:5]


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
        best = recs[0].brush_type if recs else options[0]
        index = _index_or_zero(options, best)
        point = _stage1_point(window.left, window.top, panel, index, len(options))
        return [
            PetMarker(
                key="brush-stage1",
                x=point[0],
                y=point[1],
                label=_brush_short_label(best),
                kind="brush",
                priority=1,
                size=54,
            )
        ]
    if selected_affix is None and scenario.stage2_options:
        affixes = tuple(scenario.stage2_options)
        recs = recommend_affixes(state, affixes)
        best_affix = recs[0].affix if recs else affixes[0]
        index = _index_or_zero(affixes, best_affix)
        point = _stage2_point(window.left, window.top, panel, index, len(affixes))
        return [
            PetMarker(
                key="brush-stage2",
                x=point[0],
                y=point[1],
                label="F",
                kind="brush",
                priority=1,
                size=54,
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


def _stage1_point(window_left: int, window_top: int, panel: Rect, index: int, count: int) -> tuple[int, int]:
    if count <= 1:
        x_ratio = 0.50
    else:
        x_ratio = (index + 0.5) / count
    return int(window_left + panel.x + panel.w * x_ratio), int(window_top + panel.y + panel.h * 0.34)


def _stage2_point(window_left: int, window_top: int, panel: Rect, index: int, count: int) -> tuple[int, int]:
    if count <= 1:
        y_ratio = 0.45
    else:
        y_ratio = 0.35 + min(index, count - 1) * 0.25
    return int(window_left + panel.x + panel.w * 0.58), int(window_top + panel.y + panel.h * y_ratio)


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
