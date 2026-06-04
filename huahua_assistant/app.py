from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
import tkinter as tk
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox

from .capture import DEFAULT_WINDOW_TITLE, WindowInfo, capture_target, find_window, write_image
from .config import config_value
from .codes import affix_code, affix_display_from_code, brush_code, brush_payload_codes, compact_paint_with_code, compact_rule_with_code
from .fast_events import FastVisionEvent, detect_fast_events
from .fast_vision import FastVisionSnapshot, snapshot_from_scene
from .frame_pump import FramePump, default_auto_capture
from .game_model import (
    AffixRule,
    BrushChoiceScenario,
    BrushType,
    CurrentPaintRule,
    InstallPlan,
    Paint,
    PlacementRecommendation,
    PuzzleState,
    ResourceState,
    apply_affix_to_state,
    build_affix_catalog,
    current_scope_cells,
    current_rule_is_pending,
    force_complete_current_rule,
    install_backpack_paint,
    normalize_brush_options,
    place_current_rule_anchor,
    random_brush_choice_scenario,
    random_state,
    recommend_affixes,
    recommend_brush_types,
    recommend_install_plan,
    recommend_placements,
    recommend_special_actions,
    sort_backpack_for_rule,
    state_from_observation,
)
from .learning import (
    export_learning_bundle,
    load_learning_stats,
    merge_learning_bundle,
    record_correction,
    record_high_score_trajectory,
    record_observation,
)
from .models import BoardObservation, BrushOptionsObservation, PaintCategory, PaintDirection, PaintObject, Rect, RegionKind, SceneObservation, TileState
from .pet_brain import PetBrainState, evaluate_pet_state
from .vision import VisionTracker, analyze_and_save_debug


STATE_LABELS = {
    TileState.BLANK: "空",
    TileState.BLOCKED: "阻",
    TileState.VOID: "虚",
    TileState.FADED: "失",
    TileState.PAINT: "颜",
    TileState.UNKNOWN: "?",
}

STATE_COLORS = {
    TileState.BLANK: "#eef3fb",
    TileState.BLOCKED: "#00a8d8",
    TileState.VOID: "#151b22",
    TileState.FADED: "#78808a",
    TileState.PAINT: "#ffd84d",
    TileState.UNKNOWN: "#b33030",
}

NOTICE_STYLES = {
    "info": ("#18324a", "#d7e3ee"),
    "success": ("#123d30", "#c8ffe8"),
    "warning": ("#4a3513", "#ffe8a3"),
    "error": ("#4a1f1f", "#ffd6d6"),
}

class HuahuaApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Huahua Puzzle Assistant v0.1")
        self.geometry("1180x720")
        self.minsize(1080, 640)
        self.configure(bg="#101820")

        self.status_var = tk.StringVar(
            value=f"就绪。优先捕捉窗口：{DEFAULT_WINDOW_TITLE}"
        )
        self.notice_var = tk.StringVar(value="提示：关键风险和高价值记录会显示在这里。")
        self.notice_label: tk.Label | None = None
        self.perf_var = tk.StringVar(value="性能：等待识别")
        self.perf_label: tk.Label | None = None
        self.board_var = tk.StringVar(value="局面：尚未识别")
        self.debug_path = Path("runtime/debug.png")
        self.current_scene: SceneObservation | None = None
        self.current_state: PuzzleState | None = None
        self.recommendations: list[PlacementRecommendation] = []
        self.install_plan: InstallPlan | None = None
        self.selected_backpack_index: int | None = None
        self.board_cell_rects: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        self.backpack_slot_rects: dict[int, tuple[float, float, float, float]] = {}
        self.brush_stage1_rects: dict[BrushType, tuple[float, float, float, float]] = {}
        self.brush_affix_rects: dict[int, tuple[float, float, float, float]] = {}
        self.brush_scenario: BrushChoiceScenario | None = None
        self.selected_brush_type: BrushType | None = None
        self.selected_affix: AffixRule | None = None
        self.affix_catalog = build_affix_catalog()
        self.refresh_available_var = tk.BooleanVar(value=False)
        self.force_used_var = tk.IntVar(value=0)
        self.refresh_used_var = tk.IntVar(value=0)
        self.auto_follow_var = tk.BooleanVar(value=False)
        self.auto_follow_interval_var = tk.IntVar(value=1000)
        self.overlay_var = tk.BooleanVar(value=False)
        self.overlay_window: tk.Toplevel | None = None
        self.overlay_canvas: tk.Canvas | None = None
        self.overlay_after_id: str | None = None
        self.overlay_phase = False
        self.overlay_wndprocs: dict[int, tuple[int, object]] = {}
        self.last_capture_window: WindowInfo | None = None
        self.last_capture_size: tuple[int, int] | None = None
        self.last_fast_snapshot: FastVisionSnapshot | None = None
        self.previous_fast_snapshot: FastVisionSnapshot | None = None
        self.last_fast_events: tuple[FastVisionEvent, ...] = ()
        self.last_pet_state: PetBrainState | None = None
        self.tracker = VisionTracker()
        self.detected_backpack_cache: dict[str, tuple[Paint, float]] = {}
        self.backpack_session_key: str | None = None
        self.manual_correction_applied = False
        self.real_trajectory_steps: list[dict] = []
        self.last_real_step_key: str | None = None
        self.last_auto_trajectory_key: str | None = None
        self.last_real_capture_step: dict | None = None
        self.last_capture_quality: dict | None = None
        self.last_rule_anomalies: list[dict] = []
        self.last_quick_step_key: str | None = None
        self.pending_tentative_step: dict | None = None
        self.pending_tentative_signature: str | None = None
        self.quick_capture_record_count = 0
        self.real_session_id = _new_session_id()
        self.real_session_started_at = time.time()
        self.real_session_step_index = 0
        self.last_settlement_key: str | None = None
        self.real_session_finished = False
        self.auto_follow_after_id: str | None = None
        self.auto_follow_busy = False
        self.auto_follow_overlap_paused = False
        self.auto_follow_generation = 0
        self.auto_follow_pump: FramePump | None = None
        self.auto_follow_last_sequence = 0
        self.last_high_value_key: str | None = None
        self.last_finish_skip_reason: str | None = None
        self.last_finish_diagnostic: dict | None = None
        self.last_perf: dict = {}
        self.last_desktop_pet_perf: dict = {}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self, bg="#101820")
        toolbar.pack(fill=tk.X, padx=14, pady=(14, 8))

        capture_btn = tk.Button(toolbar, text="完整识别", command=self.capture_async, width=14)
        capture_btn.pack(side=tk.LEFT)

        quick_capture_btn = tk.Button(toolbar, text="重新快速获取", command=self.quick_capture_async, width=14)
        quick_capture_btn.pack(side=tk.LEFT, padx=(8, 0))

        overlay_check = tk.Checkbutton(
            toolbar,
            text="坐标提示",
            variable=self.overlay_var,
            fg="#d7e3ee",
            bg="#101820",
            activeforeground="#ffffff",
            activebackground="#101820",
            selectcolor="#0d141b",
            command=self.toggle_overlay,
        )
        overlay_check.pack(side=tk.LEFT, padx=(8, 0))

        auto_follow_check = tk.Checkbutton(
            toolbar,
            text="自动跟随",
            variable=self.auto_follow_var,
            fg="#d7e3ee",
            bg="#101820",
            activeforeground="#ffffff",
            activebackground="#101820",
            selectcolor="#0d141b",
            command=self.toggle_auto_follow,
        )
        auto_follow_check.pack(side=tk.LEFT, padx=(8, 0))

        tk.Spinbox(
            toolbar,
            from_=300,
            to=5000,
            increment=100,
            width=5,
            textvariable=self.auto_follow_interval_var,
        ).pack(side=tk.LEFT, padx=(3, 0))

        new_run_btn = tk.Button(toolbar, text="新开一局", command=self.start_new_real_session, width=10)
        new_run_btn.pack(side=tk.LEFT, padx=(8, 0))

        finish_run_btn = tk.Button(toolbar, text="结束本局", command=self.finish_real_session, width=10)
        finish_run_btn.pack(side=tk.LEFT, padx=(8, 0))

        simulate_btn = tk.Button(toolbar, text="模拟模式", command=self.simulate_random, width=14)
        simulate_btn.pack(side=tk.LEFT, padx=(8, 0))

        manual_brush_btn = tk.Button(toolbar, text="手动笔触", command=self._open_manual_brush_dialog, width=14)
        manual_brush_btn.pack(side=tk.LEFT, padx=(8, 0))

        verify_btn = tk.Button(toolbar, text="纠错识别", command=self._open_capture_verify_dialog, width=14)
        verify_btn.pack(side=tk.LEFT, padx=(8, 0))

        stats_btn = tk.Button(toolbar, text="识别统计", command=self._open_correction_stats_dialog, width=15)
        stats_btn.pack(side=tk.LEFT, padx=(8, 0))

        export_btn = tk.Button(toolbar, text="导出数据", command=self._export_learning_data, width=11)
        export_btn.pack(side=tk.LEFT, padx=(8, 0))

        merge_btn = tk.Button(toolbar, text="合并数据", command=self._merge_learning_data, width=11)
        merge_btn.pack(side=tk.LEFT, padx=(8, 0))

        force_btn = tk.Button(toolbar, text="强制完成", command=self.force_complete_current, width=10)
        force_btn.pack(side=tk.LEFT, padx=(8, 0))

        refresh_check = tk.Checkbutton(
            toolbar,
            text="5级宠物刷新",
            variable=self.refresh_available_var,
            fg="#d7e3ee",
            bg="#101820",
            activeforeground="#ffffff",
            activebackground="#101820",
            selectcolor="#0d141b",
            command=self._rerender_side_panel,
        )
        refresh_check.pack(side=tk.LEFT, padx=(10, 0))

        tk.Label(toolbar, text="强制已用", fg="#d7e3ee", bg="#101820").pack(side=tk.LEFT, padx=(10, 3))
        tk.Spinbox(
            toolbar,
            from_=0,
            to=3,
            width=3,
            textvariable=self.force_used_var,
            command=self._rerender_side_panel,
        ).pack(side=tk.LEFT)

        tk.Label(toolbar, text="刷新已用", fg="#d7e3ee", bg="#101820").pack(side=tk.LEFT, padx=(10, 3))
        tk.Spinbox(
            toolbar,
            from_=0,
            to=2,
            width=3,
            textvariable=self.refresh_used_var,
            command=self._rerender_side_panel,
        ).pack(side=tk.LEFT)

        self.board_label = tk.Label(toolbar, textvariable=self.board_var, fg="#d7e3ee", bg="#101820")
        self.board_label.pack(side=tk.LEFT, padx=14)

        self.notice_label = tk.Label(
            self,
            textvariable=self.notice_var,
            anchor="w",
            justify=tk.LEFT,
            padx=10,
            pady=7,
            font=("Microsoft YaHei UI", 10, "bold"),
            relief=tk.FLAT,
            wraplength=1120,
        )
        self.notice_label.pack(fill=tk.X, padx=14, pady=(0, 6))
        self._set_notice("提示：完整识别建立地图；快速获取/自动跟随用于实战节奏。低置信度不会写入高分轨迹。", "info")

        self.perf_label = tk.Label(
            self,
            textvariable=self.perf_var,
            anchor="w",
            justify=tk.LEFT,
            padx=10,
            pady=5,
            fg="#d7e3ee",
            bg="#162232",
            font=("Microsoft YaHei UI", 9),
            wraplength=1120,
        )
        self.perf_label.pack(fill=tk.X, padx=14, pady=(0, 6))

        body = tk.Frame(self, bg="#101820")
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)

        self.canvas = tk.Canvas(body, bg="#0d141b", highlightthickness=0, width=560)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.region_text = tk.Text(
            body,
            width=28,
            bg="#0d141b",
            fg="#d7e3ee",
            insertbackground="#d7e3ee",
            relief=tk.FLAT,
            font=("Microsoft YaHei UI", 10),
            wrap=tk.WORD,
        )
        self.region_text.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        self.region_text.configure(state=tk.DISABLED)

        status = tk.Label(
            self,
            textvariable=self.status_var,
            anchor="w",
            fg="#d7e3ee",
            bg="#101820",
        )
        status.pack(fill=tk.X, padx=14, pady=(0, 14))

        self._draw_empty_grid()

    def _set_notice(self, message: str, level: str = "info") -> None:
        bg, fg = NOTICE_STYLES.get(level, NOTICE_STYLES["info"])
        self.notice_var.set(message)
        if self.notice_label is not None:
            self.notice_label.configure(bg=bg, fg=fg)

    def _update_perf_panel(
        self,
        *,
        mode: str | None = None,
        elapsed: float | None = None,
        capture_source: str | None = None,
        capture_perf: dict | None = None,
        recommendation_elapsed: float | None = None,
    ) -> None:
        perf = dict(self.last_perf)
        if mode is not None:
            perf["mode"] = mode
        if elapsed is not None:
            perf["elapsed"] = round(float(elapsed), 3)
        if capture_source is not None:
            perf["source"] = capture_source
        if capture_perf is not None:
            perf["capture_perf"] = dict(capture_perf)
        if recommendation_elapsed is not None:
            perf["recommendation"] = round(float(recommendation_elapsed), 3)
        quality = self.last_capture_quality
        if quality is not None:
            perf["quality_ok"] = bool(quality.get("ok"))
            perf["quality_score"] = quality.get("score")
            perf["quality_reason"] = quality.get("reason")
        pet_state = self.last_pet_state
        if pet_state is not None:
            perf["pet_state"] = pet_state.mode
            perf["pet_message"] = pet_state.message
        pet_perf = getattr(self, "last_desktop_pet_perf", None)
        if isinstance(pet_perf, dict):
            perf["pet_render"] = pet_perf.get("render")
        self.last_perf = perf
        self.perf_var.set(_format_perf_panel(perf))

    def capture_async(self) -> None:
        self.status_var.set("正在完整识别游戏窗口...")
        self._set_notice("正在完整识别：会短暂隐藏工具窗口，避免识别到工具自身。", "info")
        self.auto_follow_generation += 1
        self._stop_auto_follow_pump(wait=True)
        self.tracker.reset()
        self.withdraw()
        self.update_idletasks()
        thread = threading.Thread(target=self._capture_worker, daemon=True)
        thread.start()

    def toggle_auto_follow(self) -> None:
        if self.auto_follow_var.get():
            self.auto_follow_generation += 1
            self.auto_follow_last_sequence = 0
            self.auto_follow_busy = False
            self.status_var.set("自动跟随已开启：请把工具窗口移到游戏窗口外，避免识别到工具自身。")
            self._set_notice("自动跟随已开启：工具窗口如果盖住游戏，会暂停识别，不会遮挡你的点击。", "success")
            self._schedule_auto_follow(delay_ms=50)
        else:
            self.auto_follow_generation += 1
            if self.auto_follow_after_id is not None:
                self.after_cancel(self.auto_follow_after_id)
                self.auto_follow_after_id = None
            self._stop_auto_follow_pump()
            self.auto_follow_busy = False
            self.auto_follow_overlap_paused = False
            self.status_var.set("自动跟随已关闭。")
            self._set_notice("自动跟随已关闭：可以手动点“完整识别”或“重新快速获取”。", "info")

    def toggle_overlay(self) -> None:
        self._destroy_overlay_window()
        if self.overlay_var.get():
            self._rerender_side_panel()
            self.status_var.set("坐标提示已开启：不覆盖游戏窗口，只在工具内显示推荐坐标。")
            self._set_notice("坐标提示已开启：只在工具内显示坐标，不再创建覆盖游戏的蒙版。", "info")
        else:
            self._rerender_side_panel()
            self.status_var.set("坐标提示已关闭。")
            self._set_notice("坐标提示已关闭。", "info")

    def _ensure_overlay_window(self) -> None:
        if self.overlay_window is not None and self.overlay_window.winfo_exists():
            return
        overlay = tk.Toplevel(self)
        overlay.title("Huahua Highlight Overlay")
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.42)
        overlay.configure(bg="#010203")
        try:
            overlay.attributes("-transparentcolor", "#010203")
        except tk.TclError:
            pass
        canvas = tk.Canvas(overlay, bg="#010203", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        self.overlay_window = overlay
        self.overlay_canvas = canvas
        self._make_overlay_clickthrough()
        self.after(80, self._make_overlay_clickthrough)

    def _destroy_overlay_window(self) -> None:
        if self.overlay_after_id is not None:
            self.after_cancel(self.overlay_after_id)
            self.overlay_after_id = None
        self._restore_overlay_window_procs()
        if self.overlay_window is not None and self.overlay_window.winfo_exists():
            self.overlay_window.destroy()
        self.overlay_window = None
        self.overlay_canvas = None

    def _make_overlay_clickthrough(self) -> None:
        if self.overlay_window is None:
            return
        try:
            import ctypes
            from ctypes import wintypes

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

            hwnds = [self.overlay_window.winfo_id()]
            if self.overlay_canvas is not None:
                hwnds.append(self.overlay_canvas.winfo_id())

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

            for hwnd in hwnds:
                style = get_window_long(hwnd, GWL_EXSTYLE)
                set_window_long(
                    hwnd,
                    GWL_EXSTYLE,
                    style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                )
                if hwnd not in self.overlay_wndprocs:
                    original_proc = get_window_long(hwnd, GWLP_WNDPROC)

                    def _transparent_proc(window_hwnd, msg, wparam, lparam, *, _original_proc=original_proc):
                        if msg == WM_NCHITTEST:
                            return HTTRANSPARENT
                        return call_window_proc(_original_proc, window_hwnd, msg, wparam, lparam)

                    callback = wndproc_type(_transparent_proc)
                    set_window_long(hwnd, GWLP_WNDPROC, ctypes.cast(callback, ctypes.c_void_p).value)
                    self.overlay_wndprocs[int(hwnd)] = (int(original_proc), callback)
                user32.SetWindowPos(
                    hwnd,
                    0,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED,
                )
            try:
                self.overlay_window.attributes("-disabled", True)
            except tk.TclError:
                pass
        except Exception:
            return

    def _restore_overlay_window_procs(self) -> None:
        if not self.overlay_wndprocs:
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            GWLP_WNDPROC = -4
            set_window_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            set_window_long.restype = ctypes.c_ssize_t
            for hwnd, (original_proc, _callback) in list(self.overlay_wndprocs.items()):
                try:
                    set_window_long(hwnd, GWLP_WNDPROC, original_proc)
                except Exception:
                    pass
        finally:
            self.overlay_wndprocs.clear()

    def _refresh_overlay(self) -> None:
        if not self.overlay_var.get():
            return
        # Full-screen transparent overlays can still consume input in some
        # games/Windows configurations. Keep this mode input-safe by rendering
        # hints inside the assistant UI instead of covering the game window.
        return

    def _overlay_tick(self) -> None:
        self.overlay_after_id = None
        if not self.overlay_var.get():
            return
        self.overlay_phase = not self.overlay_phase
        self._draw_overlay()
        self.overlay_after_id = self.after(520, self._overlay_tick)

    def _draw_overlay(self) -> None:
        if self.overlay_window is None or self.overlay_canvas is None:
            return
        window = self.last_capture_window
        scene = self.current_scene
        state = self.current_state
        if window is None or scene is None or state is None:
            self.overlay_window.withdraw()
            if self.overlay_var.get():
                self.status_var.set("高亮蒙版等待真实窗口坐标：请先完整识别游戏窗口。")
            return
        self.overlay_window.deiconify()
        self.overlay_window.geometry(f"{window.width}x{window.height}+{window.left}+{window.top}")
        self._make_overlay_clickthrough()
        canvas = self.overlay_canvas
        canvas.configure(width=window.width, height=window.height)
        canvas.delete("all")
        color = "#ffd84d" if self.overlay_phase else "#64e6ff"
        self._draw_overlay_board_hints(canvas, scene, color)
        self._draw_overlay_backpack_hints(canvas, scene, state, color)

    def _set_last_capture_context(
        self,
        window: WindowInfo | None,
        capture_size: tuple[int, int] | None,
    ) -> None:
        self.last_capture_window = window
        self.last_capture_size = capture_size if _valid_capture_size(capture_size) else None

    def _update_fast_snapshot(self, snapshot: FastVisionSnapshot | None) -> None:
        if snapshot is None:
            self.previous_fast_snapshot = self.last_fast_snapshot
            self.last_fast_snapshot = None
            self.last_fast_events = ()
            self.last_pet_state = PetBrainState(mode="observing", message="等待识别", confidence=0.2)
            return
        previous = self.last_fast_snapshot
        self.previous_fast_snapshot = previous
        self.last_fast_snapshot = snapshot
        self.last_fast_events = detect_fast_events(previous, snapshot)
        self.last_pet_state = evaluate_pet_state(self)

    def _screen_point_from_capture_point(self, x: float, y: float) -> tuple[int, int] | None:
        local = self._window_local_point_from_capture_point(x, y)
        if local is None:
            return None
        window = self.last_capture_window
        if window is None:
            return int(round(local[0])), int(round(local[1]))
        return int(round(window.left + local[0])), int(round(window.top + local[1]))

    def _screen_point_for_capture_rect(
        self,
        rect: Rect,
        x_ratio: float = 0.5,
        y_ratio: float = 0.5,
    ) -> tuple[int, int] | None:
        x = float(rect.x) + float(rect.w) * x_ratio
        y = float(rect.y) + float(rect.h) * y_ratio
        return self._screen_point_from_capture_point(x, y)

    def _window_local_point_from_capture_point(self, x: float, y: float) -> tuple[float, float] | None:
        window = self.last_capture_window
        capture_size = self.last_capture_size
        if window is not None and capture_size is not None and window.width > 0 and window.height > 0:
            capture_w, capture_h = capture_size
            return float(x) / capture_w * window.width, float(y) / capture_h * window.height
        return float(x), float(y)

    def _window_local_rect_from_capture_rect(self, rect: Rect, pad: float = 0.0) -> tuple[float, float, float, float]:
        x1 = float(rect.x) + pad
        y1 = float(rect.y) + pad
        x2 = float(rect.right) - pad
        y2 = float(rect.bottom) - pad
        p1 = self._window_local_point_from_capture_point(x1, y1) or (x1, y1)
        p2 = self._window_local_point_from_capture_point(x2, y2) or (x2, y2)
        return p1[0], p1[1], p2[0], p2[1]

    def _screen_rect_from_capture_rect(self, rect: Rect, pad: float = 0.0) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = self._window_local_rect_from_capture_rect(rect, pad=pad)
        window = self.last_capture_window
        if window is None:
            return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        return (
            int(round(window.left + x1)),
            int(round(window.top + y1)),
            int(round(window.left + x2)),
            int(round(window.top + y2)),
        )

    def _draw_overlay_board_hints(self, canvas: tk.Canvas, scene: SceneObservation, color: str) -> None:
        if not self.recommendations:
            return
        best = self.recommendations[0]
        for index, rec in enumerate(self.recommendations[:3], start=1):
            tile = scene.board.tile(rec.row, rec.col)
            width = 7 if rec == best else 4
            outline = color if rec == best else "#9ff5ff"
            x1, y1, x2, y2 = self._window_local_rect_from_capture_rect(tile.rect, pad=5)
            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=outline,
                width=width,
            )
            tx, ty = self._window_local_point_from_capture_point(tile.rect.x + 12, tile.rect.y + 14) or (
                tile.rect.x + 12,
                tile.rect.y + 14,
            )
            canvas.create_text(
                tx,
                ty,
                text=f"{index}",
                fill=outline,
                font=("Microsoft YaHei UI", 16, "bold"),
                anchor=tk.W,
            )

    def _draw_overlay_backpack_hints(self, canvas: tk.Canvas, scene: SceneObservation, state: PuzzleState, color: str) -> None:
        if scene.backpack is None or not state.backpack:
            return
        desired = Counter(step.paint for step in self.install_plan.steps) if self.install_plan else Counter()
        if not desired and self.recommendations and self.recommendations[0].paint is not None:
            desired[self.recommendations[0].paint] += 1
        if not desired:
            return
        for observed in scene.backpack.paints:
            paint = Paint(
                category=observed.category,
                level=observed.level,
                subtype=observed.subtype,
                is_treasure=observed.is_treasure,
            )
            if desired[paint] <= 0:
                continue
            desired[paint] -= 1
            rect = observed.slot_rect
            x1, y1, x2, y2 = self._window_local_rect_from_capture_rect(rect, pad=3)
            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=color,
                width=6,
            )
            tx, ty = self._window_local_point_from_capture_point((rect.x + rect.right) / 2, rect.y - 10) or (
                (rect.x + rect.right) / 2,
                rect.y - 10,
            )
            canvas.create_text(
                tx,
                ty,
                text="先点",
                fill=color,
                font=("Microsoft YaHei UI", 13, "bold"),
            )

    def _schedule_auto_follow(self, delay_ms: int | None = None) -> None:
        if not self.auto_follow_var.get():
            return
        if self.auto_follow_after_id is not None:
            self.after_cancel(self.auto_follow_after_id)
        interval = delay_ms if delay_ms is not None else max(300, int(self.auto_follow_interval_var.get() or 1000))
        self.auto_follow_after_id = self.after(interval, self._auto_follow_tick)

    def _start_auto_follow_pump(self) -> None:
        if self.auto_follow_pump is None:
            self.auto_follow_pump = FramePump(
                capture_func=default_auto_capture,
                analyze_func=self.tracker.analyze_quick,
                needs_full_refresh=self._needs_stage2_brush_refresh,
                full_analyze_func=self.tracker.analyze,
                maybe_save_frame=self._maybe_save_auto_frame,
            )
        interval_ms = max(200, int(self.auto_follow_interval_var.get() or 500))
        self.auto_follow_pump.start(interval=interval_ms / 1000.0)

    def _stop_auto_follow_pump(self, *, wait: bool = False) -> None:
        if self.auto_follow_pump is not None:
            self.auto_follow_pump.stop(wait=wait)
        self.auto_follow_pump = None
        self.auto_follow_last_sequence = 0

    def _maybe_save_auto_frame(self, frame) -> float:
        if not bool(config_value("debug.save_fast_captures", False)):
            return 0.0
        write_started_at = time.perf_counter()
        write_image("runtime/capture_auto.png", frame)
        return time.perf_counter() - write_started_at

    def _auto_follow_tick(self) -> None:
        self.auto_follow_after_id = None
        if not self.auto_follow_var.get():
            return
        if self.auto_follow_busy:
            self._schedule_auto_follow()
            return
        if self._app_overlaps_game_window():
            self._stop_auto_follow_pump()
            self.auto_follow_overlap_paused = True
            self.status_var.set("自动跟随暂停：工具窗口覆盖了游戏画面，请把工具移到游戏窗口外后会自动继续。")
            self._set_notice("自动跟随暂停：工具窗口覆盖了游戏画面。把工具移到游戏窗口外后会自动继续。", "warning")
            self._schedule_auto_follow(delay_ms=1000)
            return
        self.auto_follow_overlap_paused = False
        if self.auto_follow_pump is None:
            self._start_auto_follow_pump()
        self._poll_auto_follow_pump(self.auto_follow_generation)

    def _poll_auto_follow_pump(self, generation: int) -> None:
        if not self.auto_follow_var.get() or generation != self.auto_follow_generation:
            return
        pump = self.auto_follow_pump
        if pump is None:
            self._schedule_auto_follow(delay_ms=500)
            return
        result = pump.pop_latest(self.auto_follow_last_sequence)
        if result is None:
            self._schedule_auto_follow(delay_ms=80)
            return
        self.auto_follow_last_sequence = result.sequence
        self._update_fast_snapshot(result.snapshot)
        if result.error:
            self._auto_follow_error(RuntimeError(result.error), generation)
            return
        capture_size = _capture_shape_size(result.frame_shape)
        self._render_auto_follow_result(
            result.scene,
            result.capture.source,
            result.elapsed,
            result.capture.window,
            capture_size,
            generation,
            result.perf,
        )

    def _auto_follow_error(self, exc: Exception, generation: int) -> None:
        if generation != self.auto_follow_generation or not self.auto_follow_var.get():
            return
        self.auto_follow_busy = False
        self.status_var.set(f"自动跟随失败：{exc}")
        self._set_notice(f"自动跟随失败：{exc}", "error")
        self._schedule_auto_follow(delay_ms=1500)

    def quick_capture_async(self) -> None:
        self.status_var.set("正在快速获取棋盘状态...")
        self._set_notice("正在快速获取：跳过 OCR，优先给出 1 秒级局面建议。", "info")
        self.auto_follow_generation += 1
        self._stop_auto_follow_pump(wait=True)
        self.withdraw()
        self.update_idletasks()
        thread = threading.Thread(target=self._quick_capture_worker, daemon=True)
        thread.start()

    def _capture_worker(self) -> None:
        try:
            time.sleep(0.35)
            started_at = time.perf_counter()
            capture = capture_target(
                DEFAULT_WINDOW_TITLE,
                fallback_to_screen=True,
                fallback_if_target_uncaptureable=False,
                activate=False,
            )
            frame = capture.image
            write_image("runtime/capture.png", frame)
            scene, debug_path = analyze_and_save_debug(frame, self.debug_path, tracker=self.tracker, save_crops=False)
            elapsed = time.perf_counter() - started_at
        except Exception as exc:
            self.after(0, lambda: self._show_error(exc))
            return

        capture_size = _capture_image_size(frame)
        self.after(0, lambda: self._render_result(scene, debug_path, capture.source, elapsed, capture.window, capture_size))

    def _quick_capture_worker(self) -> None:
        try:
            time.sleep(0.18)
            started_at = time.perf_counter()
            capture_started_at = time.perf_counter()
            capture = capture_target(
                DEFAULT_WINDOW_TITLE,
                fallback_to_screen=True,
                fallback_if_target_uncaptureable=False,
                activate=False,
            )
            capture_elapsed = time.perf_counter() - capture_started_at
            frame = capture.image
            write_elapsed = 0.0
            if bool(config_value("debug.save_fast_captures", False)):
                write_started_at = time.perf_counter()
                write_image("runtime/capture_quick.png", frame)
                write_elapsed = time.perf_counter() - write_started_at
            analysis_started_at = time.perf_counter()
            scene = self.tracker.analyze_quick(frame)
            analysis_elapsed = time.perf_counter() - analysis_started_at
            full_refresh_elapsed = 0.0
            if self._needs_stage2_brush_refresh():
                full_started_at = time.perf_counter()
                scene = self.tracker.analyze(frame)
                full_refresh_elapsed = time.perf_counter() - full_started_at
            elapsed = time.perf_counter() - started_at
            perf = _fast_capture_perf(capture_elapsed, analysis_elapsed, write_elapsed, full_refresh_elapsed)
        except Exception as exc:
            self.after(0, lambda: self._show_error(exc))
            return

        capture_size = _capture_image_size(frame)
        self.after(0, lambda: self._render_quick_result(scene, capture.source, elapsed, capture.window, capture_size, perf))

    def _needs_stage2_brush_refresh(self) -> bool:
        return (
            self.current_state is not None
            and current_rule_is_pending(self.current_state)
            and self.brush_scenario is not None
            and self.selected_brush_type is not None
            and self.selected_affix is None
        )

    def _show_error(self, exc: Exception) -> None:
        self._restore_app_window()
        self.status_var.set("识别失败。")
        messagebox.showerror("Huahua Assistant", str(exc))

    def _render_result(
        self,
        scene: SceneObservation | None,
        debug_path: Path | None,
        capture_source: str,
        elapsed: float,
        capture_window: WindowInfo | None,
        capture_size: tuple[int, int] | None,
    ) -> None:
        self._restore_app_window()
        self._set_last_capture_context(capture_window, capture_size)
        self.current_scene = scene
        if scene is None:
            self.current_state = None
            self.recommendations = []
            self.install_plan = None
            self._update_fast_snapshot(None)
            self._clear_brush_choice()
            self.last_capture_quality = None
            self._update_perf_panel(mode="完整识别", elapsed=elapsed, capture_source=capture_source)
            self.board_var.set("局面：未识别到棋盘")
            self.status_var.set(f"来源：{capture_source}。未找到棋盘。{elapsed:.2f}s。调试图：{debug_path}")
            self._draw_empty_grid()
            self._write_regions(None)
            return

        board = scene.board
        backpack = self._merge_detected_backpack(scene)
        self.manual_correction_applied = False
        self.current_state = state_from_observation(scene.board, scene.current_paint, backpack)
        quality = _capture_quality_gate(scene, backpack)
        self.last_capture_quality = quality
        self.last_rule_anomalies = (
            _detect_capture_rule_anomalies(self.last_real_capture_step, scene, self.current_state)
            if quality["ok"]
            else []
        )
        self._record_scene_observation(scene, backpack, quality, self.last_rule_anomalies)
        if scene.brush_options is not None and self.current_state.current_rule is not None:
            self.current_state = replace(
                self.current_state,
                current_anchor=None,
                current_rule_pending=True,
            )
        self._clear_brush_choice()
        self._apply_detected_brush_options(scene.brush_options)
        recommendation_started_at = time.perf_counter()
        self.recommendations = recommend_placements(self.current_state, learning_stats=load_learning_stats())
        self.install_plan = recommend_install_plan(self.current_state)
        recommendation_elapsed = time.perf_counter() - recommendation_started_at
        self._update_fast_snapshot(snapshot_from_scene(scene))
        self._update_perf_panel(
            mode="完整识别",
            elapsed=elapsed,
            capture_source=capture_source,
            recommendation_elapsed=recommendation_elapsed,
        )
        high_value_notice = self._record_high_value_seen(
            scene,
            self.current_state,
            backpack,
            quality,
            mode="完整识别",
            recommendations=self.recommendations,
        )
        auto_saved = self._auto_record_real_trajectory(scene, backpack)
        rect = board.rect
        self.board_var.set(f"棋盘：x={rect.x} y={rect.y} size={rect.w} 置信={board.score:.2f}")
        status = f"来源：{capture_source}。{elapsed:.2f}s。调试图已保存：{debug_path}"
        if not quality["ok"]:
            status += f" 低置信度：{quality['reason']}。"
        if self.last_rule_anomalies:
            status += f" 规则异常 x{len(self.last_rule_anomalies)}。"
        if auto_saved:
            status += " 已自动保存高价值轨迹。"
        settlement_note = self._detect_settlement_candidate(scene, self.current_state) if quality["ok"] else None
        if settlement_note:
            status += f" {settlement_note}"
        self.status_var.set(status)
        if not quality["ok"]:
            self._set_notice(f"本次完整识别低置信度：{quality['reason']}。已记录观察，但不会进入高分轨迹。", "warning")
        elif high_value_notice:
            self._set_notice(high_value_notice, "success")
        elif self.last_rule_anomalies:
            self._set_notice(f"发现规则异常 x{len(self.last_rule_anomalies)}：右侧面板可查看细节。", "warning")
        else:
            self._set_notice(f"完整识别完成：质量 {quality['score']:.2f}，可用于学习记录。", "success")
        self._draw_scene(scene)
        self._write_regions(scene)
        self._refresh_overlay()

    def _render_quick_result(
        self,
        scene: SceneObservation | None,
        capture_source: str,
        elapsed: float,
        capture_window: WindowInfo | None,
        capture_size: tuple[int, int] | None,
        capture_perf: dict | None = None,
    ) -> None:
        self._restore_app_window()
        self._set_last_capture_context(capture_window, capture_size)
        self._update_fast_snapshot(snapshot_from_scene(scene))
        self._render_fast_scene_result(scene, capture_source, elapsed, auto=False, capture_perf=capture_perf)

    def _render_auto_follow_result(
        self,
        scene: SceneObservation | None,
        capture_source: str,
        elapsed: float,
        capture_window: WindowInfo | None,
        capture_size: tuple[int, int] | None,
        generation: int,
        capture_perf: dict | None = None,
    ) -> None:
        if generation != self.auto_follow_generation or not self.auto_follow_var.get():
            self.auto_follow_busy = False
            return
        try:
            self._update_fast_snapshot(snapshot_from_scene(scene))
            self._set_last_capture_context(capture_window, capture_size)
            self._render_fast_scene_result(scene, capture_source, elapsed, auto=True, capture_perf=capture_perf)
        finally:
            self.auto_follow_busy = False
            self._schedule_auto_follow()

    def _render_fast_scene_result(
        self,
        scene: SceneObservation | None,
        capture_source: str,
        elapsed: float,
        *,
        auto: bool,
        capture_perf: dict | None = None,
    ) -> None:
        if scene is None:
            prefix = "自动跟随" if auto else "快速获取"
            perf_text = _format_fast_capture_perf(capture_perf)
            self.last_capture_quality = None
            self.last_pet_state = PetBrainState(mode="observing", message="等待识别", confidence=0.2)
            self._update_perf_panel(mode=prefix, elapsed=elapsed, capture_source=capture_source, capture_perf=capture_perf)
            self.status_var.set(f"{prefix}：未找到棋盘。来源={capture_source}，耗时={elapsed:.2f}s。{perf_text}")
            return

        if self.real_session_finished and _looks_like_new_run(scene, self.last_real_capture_step):
            self.start_new_real_session(record_event=True)

        previous_scene = self.current_scene
        self.current_scene = scene
        previous_state = self.current_state
        previous_backpack_visible = bool(previous_scene and previous_scene.backpack and previous_scene.backpack.paints)
        backpack = self._merge_detected_backpack(scene)
        backpack_visible = bool(scene.backpack and scene.backpack.paints)
        backpack_just_appeared = backpack_visible and not previous_backpack_visible
        state = state_from_observation(scene.board, scene.current_paint, backpack)
        fast_current_changed = _fast_current_paint_changed(scene, previous_state)
        if previous_state is not None:
            if fast_current_changed:
                state = replace(
                    state,
                    current_rule=None,
                    active_affix=None,
                    backpack=sort_backpack_for_rule(backpack or previous_state.backpack, None),
                    current_anchor=None,
                    current_buff_id=previous_state.current_buff_id + 1,
                    current_rule_pending=False,
                )
            else:
                state = replace(
                    state,
                    current_rule=previous_state.current_rule,
                    active_affix=previous_state.active_affix,
                    backpack=sort_backpack_for_rule(backpack or previous_state.backpack, previous_state.current_rule),
                    current_anchor=previous_state.current_anchor,
                    current_buff_id=previous_state.current_buff_id,
                    current_rule_pending=previous_state.current_rule_pending,
                )
        if backpack_just_appeared and previous_state is not None:
            state = self._confirm_pending_anchor_from_backpack_appearance(previous_state, state, scene)
        self.current_state = state
        self.manual_correction_applied = False
        self.last_capture_quality = None
        self.last_rule_anomalies = []
        self._merge_quick_brush_stage2(scene.brush_options)
        recommendation_started_at = time.perf_counter()
        self.recommendations = recommend_placements(self.current_state, learning_stats=load_learning_stats())
        self.install_plan = recommend_install_plan(self.current_state)
        recommendation_elapsed = time.perf_counter() - recommendation_started_at
        quick_quality = _quick_capture_quality(scene, elapsed, capture_perf, capture_source=capture_source)
        self.last_capture_quality = quick_quality
        self.last_pet_state = evaluate_pet_state(self)
        self._update_perf_panel(
            mode="自动跟随" if auto else "快速获取",
            elapsed=elapsed,
            capture_source=capture_source,
            capture_perf=capture_perf,
            recommendation_elapsed=recommendation_elapsed,
        )
        quick_saved = self._record_quick_capture_step(
            scene,
            self.current_state,
            elapsed,
            capture_perf=capture_perf,
            capture_source=capture_source,
        )
        high_value_notice = self._record_high_value_seen(
            scene,
            self.current_state,
            self.current_state.backpack,
            quick_quality,
            mode=("自动跟随" if auto else "快速获取"),
            recommendations=self.recommendations,
        )
        rect = scene.board.rect
        self.board_var.set(f"棋盘：x={rect.x} y={rect.y} size={rect.w} 置信={scene.board.score:.2f}")
        prefix = "自动跟随" if auto else "快速获取"
        status = (
            f"{prefix}：{elapsed:.2f}s。"
            "棋盘快路径，已跳过 OCR。"
        )
        perf_text = _format_fast_capture_perf(capture_perf)
        if perf_text:
            status += f" {perf_text}"
        if fast_current_changed:
            status += " 检测到新初始颜料图标变化，已暂停旧规则推荐，请点完整识别或完成笔触选择后快速获取。"
        if backpack_just_appeared and self.current_state is not None and not current_rule_is_pending(self.current_state):
            status += " 已检测到背包出现，确认落笔完成。"
        if quick_saved:
            status += f" 已记录快速样本 #{self.quick_capture_record_count}。"
            if self.quick_capture_record_count in {50, 100, 200, 500} or self.quick_capture_record_count % 1000 == 0:
                status += " 数据量达到分析节点，请查看识别统计。"
        elif self.pending_tentative_step is not None:
            status += " 检测到初始颜料疑似试放，等待下一次截图确认后再写入轨迹。"
        settlement_note = self._detect_settlement_candidate(scene, self.current_state) if quick_quality["ok"] else None
        if settlement_note:
            status += f" {settlement_note}"
            if auto and not self.real_session_finished:
                saved, score = self._finish_real_session_internal(reason="auto_settlement")
                if saved:
                    status += f" 自动结束本局，内部评分={score:.0f}，已保存。"
                else:
                    suffix = self.last_finish_skip_reason or "重复/未保存"
                    status += f" 自动结束本局，内部评分={score:.0f}，{suffix}。"
        self.status_var.set(status)
        if not quick_quality["ok"]:
            self._set_notice(f"{prefix}低置信度：棋盘={quick_quality['score']:.2f}，本次不写入真实轨迹。", "warning")
        elif high_value_notice:
            self._set_notice(high_value_notice, "success")
        elif settlement_note:
            self._set_notice(f"检测到可能结算：{settlement_note}", "warning")
        elif quick_saved:
            self._set_notice(f"{prefix}已记录真实样本 #{self.quick_capture_record_count}。", "success")
        elif self.pending_tentative_step is not None:
            self._set_notice("检测到初始颜料疑似试放：等待下一次截图确认，暂不写入轨迹。", "warning")
        self._draw_scene(scene)
        self._write_regions(scene)
        self._refresh_overlay()

    def _detect_settlement_candidate(self, scene: SceneObservation, state: PuzzleState | None) -> str | None:
        if state is None:
            return None
        finish_region = scene.region(RegionKind.FINISH_DRAWING)
        if finish_region is not None and finish_region.confidence >= 0.58:
            key = _stable_payload_key(
                {
                    "session": self.real_session_id,
                    "finish_button": [
                        finish_region.rect.x,
                        finish_region.rect.y,
                        finish_region.rect.w,
                        finish_region.rect.h,
                    ],
                    "terminal": _best_available_terminal_summary(state, self.real_trajectory_steps),
                }
            )
            if key == self.last_settlement_key:
                return "已识别到“完成绘制”按钮，等待点击“结束本局”保存内部评分。"
            self.last_settlement_key = key
            record_observation(
                "settlement_candidate",
                {
                    "session": self.real_session_id,
                    "board": _compact_board_summary(scene),
                    "terminal": _best_available_terminal_summary(state, self.real_trajectory_steps),
                    "steps": len(self.real_trajectory_steps),
                    "reason": "finish_drawing_button",
                    "button_confidence": round(float(finish_region.confidence), 3),
                },
                source="real_capture",
            )
            return "识别到“完成绘制”按钮，已进入可结算状态；可点“结束本局”保存内部评分。"
        counts = Counter(tile.state for tile in state.tiles)
        blank_cells = counts.get(TileState.BLANK, 0)
        open_cells = blank_cells + counts.get(TileState.VOID, 0)
        if blank_cells > 0:
            return None
        key = _stable_payload_key({"session": self.real_session_id, "board": _compact_board_summary(scene)})
        reason = "no_blank_cells" if open_cells > 0 else "no_blank_or_void_cells"
        note = "无空白地块" if open_cells > 0 else "棋盘已满"
        if key == self.last_settlement_key:
            return f"{note}，等待点击“结束本局”保存内部评分。"
        self.last_settlement_key = key
        record_observation(
            "settlement_candidate",
            {
                "session": self.real_session_id,
                "board": _compact_board_summary(scene),
                "terminal": _trajectory_terminal_summary(state, state.backpack),
                "steps": len(self.real_trajectory_steps),
                "reason": reason,
            },
            source="real_capture",
        )
        return f"识别到{note}，可能已可完成绘制；请点“结束本局”保存内部评分。"

    def _merge_quick_brush_stage2(self, observation: BrushOptionsObservation | None) -> None:
        if observation is None or self.brush_scenario is None or self.selected_brush_type is None:
            return
        affix_by_name = {affix.name: affix for affix in self.affix_catalog}
        stage2 = tuple(
            affix_by_name[name]
            for name in observation.stage2_options
            if name in affix_by_name and affix_by_name[name].brush_type == self.selected_brush_type
        )
        if not stage2:
            return
        self.brush_scenario = BrushChoiceScenario(
            stage1_options=self.brush_scenario.stage1_options,
            chosen_stage1=self.selected_brush_type,
            stage2_options=stage2[:2],
        )
        self.status_var.set("已读取二阶段词缀，请选择其中一条后再放置初始颜料")

    def start_new_real_session(self, record_event: bool = True) -> None:
        self.real_session_id = _new_session_id()
        self.real_session_started_at = time.time()
        self.real_session_step_index = 0
        self.real_trajectory_steps = []
        self.last_real_step_key = None
        self.last_quick_step_key = None
        self.last_auto_trajectory_key = None
        self.last_real_capture_step = None
        self.last_high_value_key = None
        self.last_finish_skip_reason = None
        self.last_finish_diagnostic = None
        self.pending_tentative_step = None
        self.pending_tentative_signature = None
        self.quick_capture_record_count = 0
        self.last_settlement_key = None
        self.real_session_finished = False
        self.current_scene = None
        self.current_state = None
        self.previous_fast_snapshot = None
        self.last_fast_snapshot = None
        self.last_fast_events = ()
        self.last_pet_state = None
        self.last_capture_quality = None
        self.last_perf = {}
        self._update_perf_panel(mode="新局", elapsed=0.0)
        self.recommendations = []
        self.install_plan = None
        self.selected_backpack_index = None
        self.detected_backpack_cache = {}
        self.backpack_session_key = None
        self._clear_brush_choice()
        self.tracker.reset()
        if record_event:
            record_observation(
                "run_session_started",
                {"session": self.real_session_id, "started_at": self.real_session_started_at},
                source="real_capture",
            )
        self.status_var.set(f"已开始新的一局：{self.real_session_id}")

    def finish_real_session(self) -> None:
        if not self.real_trajectory_steps:
            reason = "本局还没有真实步骤记录，请先完整识别或重新快速获取。"
            self._set_finish_diagnostic(saved=False, score=0.0, terminal={}, reliable_steps=[], finish_seen=False, reason_text=reason)
            self.status_var.set(reason)
            self._set_notice(_format_finish_diagnostic(self.last_finish_diagnostic), "warning")
            return
        if self.current_state is None:
            reason = "当前没有可评分的局面，请先识别一次终局画面。"
            reliable_steps = _trusted_trajectory_steps(self.real_trajectory_steps)
            self._set_finish_diagnostic(saved=False, score=0.0, terminal={}, reliable_steps=reliable_steps, finish_seen=False, reason_text=reason)
            self.status_var.set(reason)
            self._set_notice(_format_finish_diagnostic(self.last_finish_diagnostic), "warning")
            return
        if self.real_session_finished:
            reliable_steps = _trusted_trajectory_steps(self.real_trajectory_steps)
            finish_seen = self.current_scene is not None and _scene_has_finish_evidence(self.current_scene, self.current_state)
            terminal = _best_trusted_terminal_summary(
                self.current_state,
                self.real_trajectory_steps,
                current_quality=self.last_capture_quality,
            )
            score = _internal_run_score(terminal)
            reason = "本局已经自动/手动结束处理过，不重复保存；要记录下一局请点“新开一局”。"
            self.last_finish_skip_reason = reason
            self._set_finish_diagnostic(
                saved=False,
                score=score,
                terminal=terminal,
                reliable_steps=reliable_steps,
                finish_seen=finish_seen,
                reason_text=reason,
            )
            self.status_var.set(f"本局已结束 {self.real_session_id}：不重复保存。")
            self._set_notice(_format_finish_diagnostic(self.last_finish_diagnostic), "warning")
            return
        saved, score = self._finish_real_session_internal(reason="manual_finish")
        suffix = "已保存" if saved else (self.last_finish_skip_reason or "重复/未保存")
        self.status_var.set(
            f"本局结束 {self.real_session_id}：内部评分={score:.0f}，步骤={len(self.real_trajectory_steps)}，{suffix}。"
        )
        self._set_notice(
            _format_finish_diagnostic(self.last_finish_diagnostic),
            "success" if saved else "warning",
        )

    def _set_finish_diagnostic(
        self,
        *,
        saved: bool,
        score: float,
        terminal: dict,
        reliable_steps: list[dict],
        finish_seen: bool,
        reason_text: str,
        quality: dict | None = None,
    ) -> None:
        self.last_finish_diagnostic = {
            "saved": saved,
            "score": score,
            "reason": reason_text,
            "steps": len(self.real_trajectory_steps),
            "reliable_steps": len(reliable_steps),
            "finish_seen": finish_seen,
            "last_quality": quality if quality is not None else self.last_capture_quality,
            "terminal": terminal,
        }

    def _finish_real_session_internal(self, *, reason: str) -> tuple[bool, float]:
        self.last_finish_skip_reason = None
        if self.current_state is None:
            self.last_finish_skip_reason = "当前没有可评分局面"
            self._set_finish_diagnostic(
                saved=False,
                score=0.0,
                terminal={},
                reliable_steps=[],
                finish_seen=False,
                reason_text=self.last_finish_skip_reason,
            )
            return False, 0.0
        raw_finish_seen = self.current_scene is not None and _scene_has_finish_evidence(self.current_scene, self.current_state)
        current_trusted = _quality_is_trusted_for_trajectory(self.last_capture_quality)
        finish_seen = raw_finish_seen and current_trusted
        reliable_steps = _trusted_trajectory_steps(self.real_trajectory_steps)
        terminal = _best_trusted_terminal_summary(
            self.current_state,
            reliable_steps,
            current_quality=self.last_capture_quality,
        )
        score = _internal_run_score(terminal)
        score_breakdown = _internal_score_breakdown(terminal)
        terminal_has_high_value = _summary_has_high_value(terminal.get("known") or {})
        if self.last_capture_quality is not None and not current_trusted and not finish_seen and not terminal_has_high_value:
            self.last_finish_skip_reason = _finish_reliability_reason(
                terminal,
                reliable_steps,
                finish_seen=finish_seen,
                last_quality=self.last_capture_quality,
            )
            self._set_finish_diagnostic(
                saved=False,
                score=score,
                terminal=terminal,
                reliable_steps=reliable_steps,
                finish_seen=finish_seen,
                reason_text=self.last_finish_skip_reason,
                quality=self.last_capture_quality,
            )
            record_observation(
                "run_session_finish_skipped",
                {
                    "session": self.real_session_id,
                    "reason": reason,
                    "skip_reason": self.last_finish_skip_reason,
                    "quality": self.last_capture_quality,
                    "internal_score": score,
                    "score_breakdown": score_breakdown,
                    "steps": len(self.real_trajectory_steps),
                    "reliable_steps": len(reliable_steps),
                    "raw_finish_seen": raw_finish_seen,
                    "terminal": terminal,
                },
                source="real_capture",
            )
            return False, score
        reliable, reason_text = _terminal_reliability_for_finish(terminal, reliable_steps, finish_seen=finish_seen)
        if not reliable:
            self.last_finish_skip_reason = reason_text
            self._set_finish_diagnostic(
                saved=False,
                score=score,
                terminal=terminal,
                reliable_steps=reliable_steps,
                finish_seen=finish_seen,
                reason_text=self.last_finish_skip_reason,
            )
            record_observation(
                "run_session_finish_skipped",
                {
                    "session": self.real_session_id,
                    "reason": reason,
                    "skip_reason": self.last_finish_skip_reason,
                    "internal_score": score,
                    "score_breakdown": score_breakdown,
                    "steps": len(self.real_trajectory_steps),
                    "reliable_steps": len(reliable_steps),
                    "raw_finish_seen": raw_finish_seen,
                    "terminal": terminal,
                },
                source="real_capture",
            )
            return False, score
        payload = {
            "version": 1,
            "kind": "internal_scored_run",
            "session": self.real_session_id,
            "reason": reason,
            "started_at": self.real_session_started_at,
            "finished_at": time.time(),
            "internal_score": score,
            "score_model": "huahua-v2-treasure-equivalence",
            "score_breakdown": score_breakdown,
            "terminal": terminal,
            "finish_seen": finish_seen,
            "steps": reliable_steps[-240:],
        }
        saved = record_high_score_trajectory(payload, source="real_capture")
        self._set_finish_diagnostic(
            saved=saved,
            score=score,
            terminal=terminal,
            reliable_steps=reliable_steps,
            finish_seen=finish_seen,
            reason_text=("结算轨迹已保存" if saved else "终局可信，但轨迹重复或已保存过，未重复写入"),
        )
        record_observation(
            "run_session_finished",
            {
                "session": self.real_session_id,
                "reason": reason,
                "internal_score": score,
                "score_model": "huahua-v2-treasure-equivalence",
                "score_breakdown": score_breakdown,
                "steps": len(self.real_trajectory_steps),
                "saved": saved,
                "terminal": terminal,
                "raw_finish_seen": raw_finish_seen,
                "finish_seen": finish_seen,
            },
            source="real_capture",
        )
        self.real_session_finished = True
        return saved, score

    def _record_quick_capture_step(
        self,
        scene: SceneObservation,
        state: PuzzleState,
        elapsed: float,
        *,
        capture_perf: dict | None = None,
        capture_source: str | None = None,
    ) -> bool:
        quality = _quick_capture_quality(scene, elapsed, capture_perf, capture_source=capture_source)
        step = _compact_real_capture_step(
            scene=scene,
            state=state,
            backpack=state.backpack,
            recommendations=self.recommendations,
            brush_scenario=self.brush_scenario,
            resources=self._resource_state(),
            quality=quality,
        )
        step["type"] = "quick_capture"
        step["mode"] = "quick_capture"
        step["ocr"] = False
        step["inferred_action"] = _infer_real_capture_action(self.last_real_capture_step, step)
        if not _quality_is_trusted_for_trajectory(quality):
            record_observation(
                "quick_capture_ignored",
                {
                    "session": self.real_session_id,
                    "quality": quality,
                    "board": step.get("board"),
                    "backpack": step.get("backpack"),
                    "current": step.get("current"),
                    "inferred_action": step.get("inferred_action"),
                    "note": "低置信度快速截图只保留观察，不进入真实轨迹/结算",
                },
                source="real_capture",
            )
            return False
        tentative_saved = self._handle_tentative_step(step, observation_kind="quick_capture_step")
        if tentative_saved is not None:
            if tentative_saved:
                self.quick_capture_record_count += 1
            return tentative_saved
        if _is_noop_capture_update(step["inferred_action"]):
            return False

        step_key = _stable_payload_key(
            {
                "board": step.get("board"),
                "current": step.get("current"),
                "anchor": step.get("anchor"),
                "pending": step.get("pending"),
                "backpack": step.get("backpack"),
                "brush": step.get("brush"),
                "resources": step.get("resources"),
                "tiles": step.get("tiles"),
                "action": step.get("inferred_action"),
            }
        )
        if step_key == self.last_quick_step_key:
            return False
        step = self._stamp_real_step(step)
        self.last_quick_step_key = step_key
        self.last_real_capture_step = step
        self.real_trajectory_steps.append(step)
        self.real_trajectory_steps = self.real_trajectory_steps[-80:]
        self.quick_capture_record_count += 1
        record_observation(
            "quick_capture_step",
            step,
            source="real_capture",
        )
        return True

    def _stamp_real_step(self, step: dict) -> dict:
        stamped = dict(step)
        stamped["session"] = self.real_session_id
        stamped["step_index"] = self.real_session_step_index
        stamped["t"] = round(time.time() - self.real_session_started_at, 3)
        self.real_session_step_index += 1
        return stamped

    def _handle_tentative_step(self, step: dict, *, observation_kind: str) -> bool | None:
        if not _quality_is_trusted_for_trajectory(step.get("quality")):
            record_observation(
                "trajectory_step_ignored",
                {
                    "session": self.real_session_id,
                    "kind": "untrusted_tentative_or_confirm_step",
                    "quality": step.get("quality"),
                    "action": step.get("inferred_action"),
                    "note": "Untrusted capture kept as observation only; not added to trajectory.",
                },
                source="real_capture",
            )
            return False
        action = step.get("inferred_action") or {}
        if _is_tentative_place_current_action(action, step):
            signature = _tentative_place_signature(step)
            if _backpack_visible_in_step(step):
                pending_trusted = _quality_is_trusted_for_trajectory(
                    (self.pending_tentative_step or {}).get("quality")
                )
                confirmed_from_pending = (
                    signature
                    and signature == self.pending_tentative_signature
                    and self.pending_tentative_step is not None
                    and pending_trusted
                )
                confirmed = dict(self.pending_tentative_step) if confirmed_from_pending else dict(step)
                confirmed["tentative"] = False
                confirmed["confirmed_by_next_capture"] = confirmed_from_pending
                confirmed["confirmed_by_backpack"] = True
                confirmed["inferred_action"] = dict(confirmed.get("inferred_action") or {})
                confirmed["inferred_action"]["tentative"] = False
                confirmed["inferred_action"]["confirmed"] = True
                confirmed["inferred_action"]["confirmed_by_backpack"] = True
                stamped = self._stamp_real_step(confirmed)
                self.last_real_capture_step = stamped
                self.real_trajectory_steps.append(stamped)
                self.real_trajectory_steps = self.real_trajectory_steps[-80:]
                self.pending_tentative_step = None
                self.pending_tentative_signature = None
                record_observation(observation_kind, stamped, source="real_capture")
                return True

            tentative = dict(step)
            tentative["tentative"] = True
            tentative["session"] = self.real_session_id
            tentative["inferred_action"] = dict(action)
            tentative["inferred_action"]["tentative"] = True
            self.pending_tentative_step = tentative
            self.pending_tentative_signature = signature
            record_observation(
                "tentative_place_current_paint",
                {
                    "session": self.real_session_id,
                    "signature": signature,
                    "action": tentative.get("inferred_action"),
                    "note": "初始颜料疑似试放，等待下一次截图确认后再写入轨迹",
                },
                source="real_capture",
            )
            return False

        if self.pending_tentative_step is not None:
            record_observation(
                "tentative_place_cancelled",
                {
                    "session": self.real_session_id,
                    "signature": self.pending_tentative_signature,
                    "next_action": action,
                    "note": "初始颜料试放未保持，已丢弃，不写入轨迹",
                },
                source="real_capture",
            )
            self.pending_tentative_step = None
            self.pending_tentative_signature = None
        return None

    def _confirm_pending_anchor_from_backpack_appearance(
        self,
        previous_state: PuzzleState,
        current_state: PuzzleState,
        scene: SceneObservation,
    ) -> PuzzleState:
        if not current_rule_is_pending(previous_state) or previous_state.current_rule is None:
            return current_state
        anchor = _find_new_anchor_from_scene(scene, previous_state)
        if anchor is None:
            return current_state
        result = place_current_rule_anchor(previous_state, anchor[0], anchor[1])
        if not result.success:
            return current_state
        return replace(
            result.state,
            backpack=current_state.backpack,
            current_buff_id=result.state.current_buff_id,
        )

    def _restore_app_window(self) -> None:
        self.deiconify()
        self.lift()

    def _app_overlaps_game_window(self) -> bool:
        try:
            if not self.winfo_viewable():
                return False
            game_window = find_window(DEFAULT_WINDOW_TITLE)
            if game_window is None or not game_window.captureable:
                return False
            self.update_idletasks()
            app_rect = (
                self.winfo_rootx(),
                self.winfo_rooty(),
                self.winfo_rootx() + self.winfo_width(),
                self.winfo_rooty() + self.winfo_height(),
            )
            game_rect = (game_window.left, game_window.top, game_window.right, game_window.bottom)
            return _rects_intersect(app_rect, game_rect)
        except tk.TclError:
            return False

    def _auto_record_real_trajectory(self, scene: SceneObservation, backpack: tuple[Paint, ...]) -> bool:
        if self.current_state is None:
            return False
        quality = self.last_capture_quality or _capture_quality_gate(scene, backpack)
        finish_seen = _scene_has_finish_evidence(scene, self.current_state)
        if not _quality_is_trusted_for_trajectory(quality):
            record_observation(
                "capture_step_ignored",
                {
                    "session": self.real_session_id,
                    "quality": quality,
                    "finish_seen": finish_seen,
                    "reason": "untrusted_capture",
                    "board": _compact_board_summary(scene),
                    "note": "Untrusted full/auto capture was not appended to the real trajectory.",
                },
                source="real_capture",
            )
            return False
        if any(item.get("severity") == "block" for item in self.last_rule_anomalies):
            return False
        step = _compact_real_capture_step(
            scene=scene,
            state=self.current_state,
            backpack=backpack,
            recommendations=self.recommendations,
            brush_scenario=self.brush_scenario,
            resources=self._resource_state(),
            quality=quality,
        )
        step["inferred_action"] = _infer_real_capture_action(self.last_real_capture_step, step)
        tentative_saved = self._handle_tentative_step(step, observation_kind="capture_step")
        if tentative_saved is not None:
            return tentative_saved
        step = self._stamp_real_step(step)
        self.last_real_capture_step = step
        step_key = _stable_payload_key(step)
        if step_key != self.last_real_step_key:
            self.real_trajectory_steps.append(step)
            self.real_trajectory_steps = self.real_trajectory_steps[-60:]
            self.last_real_step_key = step_key

        reliable_steps = _trusted_trajectory_steps(self.real_trajectory_steps)
        terminal = (
            _best_trusted_terminal_summary(self.current_state, reliable_steps, current_quality=quality)
            if finish_seen
            else _trajectory_terminal_summary(self.current_state, backpack)
        )
        reliable, _reason_text = _terminal_reliability_for_finish(terminal, reliable_steps, finish_seen=finish_seen)
        if not reliable:
            return False
        should_save, reason = _should_save_auto_trajectory(terminal)
        if not should_save:
            return False
        payload = {
            "version": 1,
            "kind": "auto_capture_trajectory",
            "session": self.backpack_session_key,
            "reason": reason,
            "terminal": terminal,
            "finish_seen": finish_seen,
            "steps": reliable_steps[-40:],
        }
        payload_key = _stable_payload_key(payload)
        if payload_key == self.last_auto_trajectory_key:
            return False
        saved = record_high_score_trajectory(payload, source="real_capture")
        if saved:
            self.last_auto_trajectory_key = payload_key
        return saved

    def _export_learning_data(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Learning Data",
            defaultextension=".json",
            filetypes=[("Huahua learning bundle", "*.json"), ("All files", "*.*")],
            initialfile="huahua_learning_bundle.json",
        )
        if not path:
            return
        try:
            result = export_learning_bundle(path)
        except OSError as exc:
            messagebox.showerror("Export Data", f"导出失败：{exc}")
            return
        self.status_var.set(
            f"已导出学习数据：观察{result['observations']}条，轨迹{result['trajectories']}条，纠错{result['corrections']}条 -> {result['path']}"
        )

    def _merge_learning_data(self) -> None:
        path = filedialog.askopenfilename(
            title="Merge Learning Data",
            filetypes=[("Huahua learning bundle", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            result = merge_learning_bundle(path)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Merge Data", f"合并失败：{exc}")
            return
        self.status_var.set(
            f"已合并学习数据：新增观察{result['added_observations']}条，新增轨迹{result['added_trajectories']}条，新增纠错{result['added_corrections']}条"
        )
        self._rerender_side_panel()

    def _draw_empty_grid(self) -> None:
        self.canvas.delete("all")
        self._draw_grid(None)

    def _draw_scene(self, scene: SceneObservation) -> None:
        self.canvas.delete("all")
        self._draw_grid(scene.board, self.current_state)

    def simulate_random(self) -> None:
        self.current_scene = None
        self.detected_backpack_cache = {}
        self.backpack_session_key = None
        self.manual_correction_applied = False
        self.current_state = random_state()
        self.selected_backpack_index = None
        self.recommendations = recommend_placements(self.current_state)
        self.install_plan = recommend_install_plan(self.current_state)
        self._clear_brush_choice()
        self.board_var.set("模式：模拟拼图")
        self.status_var.set("模拟模式：本地随机生成棋盘和初始颜料，不写入真实学习记录。")
        self.canvas.delete("all")
        self._draw_grid(None, self.current_state)
        self._write_simulation()

    def _merge_detected_backpack(self, scene: SceneObservation) -> tuple[Paint, ...]:
        session_key = _scene_backpack_session_key(scene)
        if session_key != self.backpack_session_key:
            self.detected_backpack_cache = {}
            self.backpack_session_key = session_key
        if scene.backpack is None:
            return tuple(item[0] for item in self.detected_backpack_cache.values())
        for observed in scene.backpack.paints:
            key = _observed_paint_key(observed)
            paint = Paint(
                category=observed.category,
                level=observed.level,
                subtype=observed.subtype,
                is_treasure=observed.is_treasure,
            )
            confidence = observed.category_confidence * 0.55 + observed.level_confidence * 0.45
            previous = self.detected_backpack_cache.get(key)
            if previous is None or confidence >= previous[1]:
                self.detected_backpack_cache[key] = (paint, confidence)
        return tuple(item[0] for item in self.detected_backpack_cache.values())

    def _record_scene_observation(
        self,
        scene: SceneObservation,
        backpack: tuple[Paint, ...],
        quality: dict | None = None,
        anomalies: list[dict] | None = None,
    ) -> None:
        quality = quality or _capture_quality_gate(scene, backpack)
        record_observation(
            "capture_quality",
            quality,
            source="real_capture",
        )
        if scene.current_paint is not None:
            paint = scene.current_paint
            observed_rule = (
                CurrentPaintRule(
                    paint=Paint(
                        category=paint.category,
                        level=paint.level,
                        subtype=paint.subtype,
                        is_treasure=paint.is_treasure,
                    ),
                    direction=paint.direction,
                    target_object=paint.target_object,
                    match_required=paint.match_required,
                    upgrade_amount=paint.upgrade_amount,
                    match_current=paint.match_current or 0,
                )
                if paint.level is not None
                and paint.match_required is not None
                and paint.upgrade_amount is not None
                else None
            )
            record_observation(
                "current_paint",
                {
                    "paint_code": (
                        compact_paint_with_code(
                            Paint(
                                category=paint.category,
                                level=paint.level,
                                subtype=paint.subtype,
                                is_treasure=paint.is_treasure,
                            )
                        ).get("code")
                        if paint.level is not None
                        else None
                    ),
                    "rule_code": compact_rule_with_code(observed_rule).get("code") if observed_rule is not None else None,
                    "category": paint.category.value,
                    "level": paint.level,
                    "direction": paint.direction.value,
                    "target": paint.target_object.value,
                    "match_current": paint.match_current,
                    "match_required": paint.match_required,
                    "upgrade_amount": paint.upgrade_amount,
                    "raw_text_hint": paint.raw_text_hint,
                },
                source="real_capture",
            )
            self._record_current_paint_rule_warnings(paint)
        for anomaly in anomalies or []:
            record_observation(
                "rule_anomaly",
                anomaly,
                source="real_capture",
            )
        if scene.brush_options is not None:
            brush = scene.brush_options
            record_observation(
                "brush_options",
                {
                    "stage1_options": brush.stage1_options,
                    "stage1_codes": [code for item in brush.stage1_options if (code := brush_code(item))],
                    "chosen_stage1": brush.chosen_stage1,
                    "chosen_stage1_code": brush_code(brush.chosen_stage1),
                    "stage2_options": brush.stage2_options,
                    "stage2_codes": [code for item in brush.stage2_options if (code := affix_code(item))],
                    "confidence": brush.confidence,
                    "stage1_scores": brush.stage1_scores,
                    "stage2_scores": brush.stage2_scores,
                },
                source="real_capture",
            )
            for name in brush.stage2_options:
                if (
                    "进度要求减少1" in name
                    or "进度要求减少2" in name
                    or "所需数量-2" in name
                    or ("调色目标" in name and "减少" in name)
                ):
                    record_observation(
                        "rule_conflict",
                        {
                            "kind": "match_required_delta",
                            "affix_name": name,
                            "affix_code": affix_code(name),
                            "note": "已记录转锋减进度 -1/-2 实战文案，按识别词条 delta 保存",
                        },
                        source="real_capture",
                    )
        if scene.backpack is not None:
            record_observation(
                "backpack_capture",
                {
                    "visible": len(scene.backpack.paints),
                    "cumulative": len(backpack),
                    "visible_paints": [
                        {
                            "category": observed.category.value,
                            "level": observed.level,
                            "subtype": observed.subtype,
                            "is_treasure": observed.is_treasure,
                            "category_confidence": round(observed.category_confidence, 3),
                            "level_confidence": round(observed.level_confidence, 3),
                        }
                        for observed in scene.backpack.paints
                    ],
                    "slots": [
                        {
                            "slot": [observed.slot_rect.x, observed.slot_rect.y, observed.slot_rect.w, observed.slot_rect.h],
                            "category": observed.category.value,
                            "category_confidence": round(observed.category_confidence, 3),
                            "level": observed.level,
                            "level_confidence": round(observed.level_confidence, 3),
                            "subtype": observed.subtype,
                            "is_treasure": observed.is_treasure,
                            "fingerprint": observed.fingerprint,
                            "merge_key": _observed_paint_key(observed),
                        }
                        for observed in scene.backpack.paints
                    ],
                    "paints": [
                        {
                            "code": compact_paint_with_code(paint).get("code"),
                            "category": paint.category.value,
                            "level": paint.level,
                            "subtype": paint.subtype,
                            "is_treasure": paint.is_treasure,
                        }
                        for paint in backpack
                    ],
                },
                source="real_capture",
            )

    def _record_high_value_seen(
        self,
        scene: SceneObservation,
        state: PuzzleState | None,
        backpack: tuple[Paint, ...],
        quality: dict | None,
        *,
        mode: str,
        recommendations: list[PlacementRecommendation] | None = None,
    ) -> str | None:
        quality = quality or _capture_quality_gate(scene, backpack)
        if not quality.get("ok"):
            return None
        board_paints = tuple(tile.paint for tile in state.tiles if tile.paint is not None) if state is not None else ()
        board_summary = _paint_collection_summary(board_paints)
        backpack_summary = _paint_collection_summary(backpack)
        combined_summary = _paint_collection_summary(tuple(paint for paint in (*board_paints, *backpack) if paint is not None))
        if not _summary_has_high_value(combined_summary):
            return None
        payload = {
            "session": self.real_session_id,
            "mode": mode,
            "quality": quality,
            "board": _compact_board_summary(scene),
            "board_known": board_summary,
            "backpack": backpack_summary,
            "known": combined_summary,
            "current": _compact_rule(state.current_rule) if state is not None else None,
            "score": _quality_proxy(combined_summary),
            "score_breakdown": _internal_score_breakdown({"known": combined_summary}),
            "recommendations": [
                _compact_recommendation(rec)
                for rec in (recommendations or [])[:3]
            ],
        }
        event_key = _stable_payload_key(
            {
                "session": self.real_session_id,
                "mode": mode,
                "board_states": payload["board"].get("states"),
                "known": combined_summary,
                "current": payload.get("current"),
            }
        )
        if event_key != self.last_high_value_key:
            self.last_high_value_key = event_key
            record_observation("high_value_seen", payload, source="real_capture")
        return f"已记录高价值识别：{_format_high_value_summary(combined_summary)}。"

    def _record_current_paint_rule_warnings(self, paint) -> None:
        if paint.level is None:
            return
        expected_match, expected_upgrade = _empirical_current_paint_defaults(paint.level)
        if paint.match_required is not None and paint.match_required != expected_match:
            record_observation(
                "rule_mismatch",
                {
                    "kind": "current_match_required",
                    "level": paint.level,
                    "observed": paint.match_required,
                    "expected": expected_match,
                    "raw_text_hint": paint.raw_text_hint,
                },
                source="real_capture",
            )
        if paint.upgrade_amount is not None and not _empirical_upgrade_is_allowed(paint.level, paint.upgrade_amount):
            record_observation(
                "rule_mismatch",
                {
                    "kind": "current_upgrade_amount",
                    "level": paint.level,
                    "observed": paint.upgrade_amount,
                    "expected": expected_upgrade,
                    "allowed": _empirical_allowed_upgrades(paint.level),
                    "raw_text_hint": paint.raw_text_hint,
                },
                source="real_capture",
            )

    def force_complete_current(self) -> None:
        if self.current_state is None:
            self.status_var.set("当前没有可强制完成的模拟棋局")
            return
        force_used = _clamp_int_var(self.force_used_var, 0, 3)
        if force_used >= 3:
            self.status_var.set("强制完成次数已用完")
            return
        result = force_complete_current_rule(self.current_state, force_used)
        self.current_state = result.state
        if result.success:
            self.force_used_var.set(force_used + 1)
            self.selected_backpack_index = None
            self._begin_pending_brush_choice()
        self.status_var.set(result.message)
        self._refresh_simulation_state_views()

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        self.canvas.delete("all")
        if self.current_scene is None:
            self._draw_grid(None, self.current_state)
            return
        self._draw_grid(self.current_scene.board, self.current_state)

    def _rerender_side_panel(self) -> None:
        if self.current_scene is not None:
            self._write_regions(self.current_scene)
        elif self.current_state is not None:
            self._write_simulation()

    def _refresh_simulation_state_views(self) -> None:
        if self.current_state is None:
            return
        self.recommendations = recommend_placements(self.current_state)
        self.install_plan = recommend_install_plan(self.current_state)
        self.canvas.delete("all")
        self._draw_grid(self.current_scene.board if self.current_scene else None, self.current_state)
        self._rerender_side_panel()
        self._refresh_overlay()

    def _clear_brush_choice(self) -> None:
        self.brush_scenario = None
        self.selected_brush_type = None
        self.selected_affix = None
        self.brush_stage1_rects = {}
        self.brush_affix_rects = {}

    def _apply_detected_brush_options(self, observation: BrushOptionsObservation | None) -> None:
        if observation is None:
            return
        stage1 = _brush_types_from_values(observation.stage1_options)
        affix_by_name = {affix.name: affix for affix in self.affix_catalog}
        stage2 = tuple(
            affix_by_name[name]
            for name in observation.stage2_options
            if name in affix_by_name
        )
        chosen = _brush_type_from_value(observation.chosen_stage1)
        if chosen is None and stage2:
            chosen = stage2[0].brush_type
        if not stage1 and chosen is not None:
            stage1 = (chosen,)
        if not stage1 and not stage2:
            return
        if chosen is None:
            chosen = stage1[0]
        if chosen not in stage1:
            stage1 = normalize_brush_options((*stage1, chosen))

        # Real flow: full recognition sees stage-1 cards first. Stage-2 affixes
        # should appear in the tool only after the player chooses a category and
        # uses quick refresh/capture to read the resulting two affix cards.
        if current_rule_is_pending(self.current_state) and self.selected_brush_type is None:
            stage2 = ()
            chosen = stage1[0]
        self.brush_scenario = BrushChoiceScenario(
            stage1_options=stage1,
            chosen_stage1=chosen,
            stage2_options=stage2,
        )
        self.selected_brush_type = chosen if stage2 else None
        self.selected_affix = None
        self.selected_backpack_index = None

    def _begin_pending_brush_choice(self) -> None:
        if self.current_state is None or not current_rule_is_pending(self.current_state):
            self._clear_brush_choice()
            return
        stage1 = normalize_brush_options(tuple(random.sample(list(BrushType), 2)))
        self.brush_scenario = BrushChoiceScenario(
            stage1_options=stage1,
            chosen_stage1=stage1[0],
            stage2_options=(),
        )
        self.selected_brush_type = None
        self.selected_affix = None
        self.selected_backpack_index = None

    def _pending_brush_choice_required(self) -> bool:
        return (
            self.current_state is not None
            and current_rule_is_pending(self.current_state)
            and self.brush_scenario is not None
            and self.selected_affix is None
        )

    def _choose_brush_stage1(self, brush_type: BrushType) -> None:
        if self.current_state is None or self.brush_scenario is None:
            return
        if self.selected_affix is not None:
            self.status_var.set("词缀已经选择，当前初始颜料放置前不能重新选择大类")
            return
        if brush_type not in self.brush_scenario.stage1_options:
            return
        options = [affix for affix in self.affix_catalog if affix.brush_type == brush_type]
        stage2 = tuple(random.sample(options, min(2, len(options)))) if options else ()
        self.brush_scenario = BrushChoiceScenario(
            stage1_options=self.brush_scenario.stage1_options,
            chosen_stage1=brush_type,
            stage2_options=stage2,
        )
        self.selected_brush_type = brush_type
        self.selected_affix = None
        self.status_var.set(f"已选择{_brush_label(brush_type)}，请从右侧2个词缀中再选1个")
        self._refresh_simulation_state_views()

    def _choose_brush_affix(self, affix_index: int) -> None:
        if self.current_state is None or self.brush_scenario is None:
            return
        if self.selected_affix is not None:
            self.status_var.set("词缀已经选择，请直接在棋盘上放置新的初始颜料")
            return
        if not 0 <= affix_index < len(self.brush_scenario.stage2_options):
            return
        affix = self.brush_scenario.stage2_options[affix_index]
        result = apply_affix_to_state(self.current_state, affix)
        self.current_state = result.state
        if result.success:
            self.selected_affix = affix
        self.status_var.set(result.message)
        self._refresh_simulation_state_views()

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self.current_state is None:
            return
        point = (float(event.x), float(event.y))
        for brush_type, rect in self.brush_stage1_rects.items():
            if _point_in_rect(point, rect):
                self._choose_brush_stage1(brush_type)
                return
        for affix_index, rect in self.brush_affix_rects.items():
            if _point_in_rect(point, rect):
                self._choose_brush_affix(affix_index)
                return
        for index, rect in self.backpack_slot_rects.items():
            if _point_in_rect(point, rect):
                if current_rule_is_pending(self.current_state):
                    self.status_var.set("请先在棋盘空白地块放置新的初始颜料")
                    return
                self.selected_backpack_index = index
                paint = self.current_state.backpack[index]
                self.status_var.set(f"已选中背包第{index + 1}格：{_paint_display_label(paint)}")
                self.canvas.delete("all")
                self._draw_grid(self.current_scene.board if self.current_scene else None, self.current_state)
                self._refresh_overlay()
                return
        if current_rule_is_pending(self.current_state):
            for (row, col), rect in self.board_cell_rects.items():
                if not _point_in_rect(point, rect):
                    continue
                if self._pending_brush_choice_required():
                    self.status_var.set("请先在右侧完成转锋/堆砌/扩散二阶段选择，再放置新的初始颜料")
                    return
                result = place_current_rule_anchor(self.current_state, row, col)
                self.current_state = result.state
                self.selected_backpack_index = None
                if result.success:
                    self._clear_brush_choice()
                self.status_var.set(result.message)
                self._refresh_simulation_state_views()
                return
        if self.selected_backpack_index is None:
            return
        for (row, col), rect in self.board_cell_rects.items():
            if not _point_in_rect(point, rect):
                continue
            result = install_backpack_paint(self.current_state, self.selected_backpack_index, row, col)
            self.current_state = result.state
            if result.success:
                self.selected_backpack_index = None
                if result.completed_match:
                    self._begin_pending_brush_choice()
            elif self.selected_backpack_index is not None and self.selected_backpack_index >= len(self.current_state.backpack):
                self.selected_backpack_index = None
            self.status_var.set(result.message)
            self._refresh_simulation_state_views()
            return

    def _open_capture_verify_dialog(self) -> None:
        if self.current_scene is None:
            self.status_var.set("请先完整识别真实游戏画面；模拟模式不参与学习纠错")
            return

        scene = self.current_scene
        window = tk.Toplevel(self)
        window.title("纠错真实识别")
        window.configure(bg="#101820")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()

        content = tk.Frame(window, bg="#101820")
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(
            content,
            text="确认/纠错本次真实识别",
            fg="#ffffff",
            bg="#101820",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        risk_hint = _risk_hint_text(load_learning_stats())
        tk.Label(
            content,
            text=risk_hint,
            fg="#f2c14e" if "高风险" in risk_hint else "#95abc1",
            bg="#101820",
            font=("Microsoft YaHei UI", 9),
            wraplength=430,
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 8))

        paint = scene.current_paint
        brush = scene.brush_options
        backpack_count = len(self.detected_backpack_cache)

        category_var = tk.StringVar(value=paint.category.value if paint else PaintCategory.UNKNOWN.value)
        level_var = tk.StringVar(value="" if paint is None or paint.level is None else str(paint.level))
        direction_var = tk.StringVar(value=paint.direction.value if paint else PaintDirection.UNKNOWN.value)
        target_var = tk.StringVar(value=paint.target_object.value if paint else PaintObject.UNKNOWN.value)
        match_required_var = tk.StringVar(value="" if paint is None or paint.match_required is None else str(paint.match_required))
        upgrade_var = tk.StringVar(value="" if paint is None or paint.upgrade_amount is None else str(paint.upgrade_amount))
        stage1_var = tk.StringVar(value=", ".join(brush.stage1_options) if brush else "")
        stage2_var = tk.StringVar(value="; ".join(brush.stage2_options) if brush else "")
        backpack_var = tk.StringVar(value=str(backpack_count))
        note_var = tk.StringVar()
        apply_var = tk.BooleanVar(value=True)

        fields = [
            ("类型", category_var, [item.value for item in PaintCategory]),
            ("等级", level_var, None),
            ("方向", direction_var, [item.value for item in PaintDirection]),
            ("目标", target_var, [item.value for item in PaintObject]),
            ("需求数", match_required_var, None),
            ("升级数", upgrade_var, None),
            ("一阶段", stage1_var, None),
            ("二阶段词缀", stage2_var, None),
            ("背包累计数", backpack_var, None),
            ("备注", note_var, None),
        ]

        for row, (label, variable, options) in enumerate(fields, start=2):
            tk.Label(content, text=label, fg="#95abc1", bg="#101820").grid(row=row, column=0, sticky="w", pady=3)
            if options:
                widget = tk.OptionMenu(content, variable, *options)
                widget.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=22)
            else:
                widget = tk.Entry(content, textvariable=variable, bg="#172331", fg="#d7e3ee", insertbackground="#d7e3ee", width=30)
            widget.grid(row=row, column=1, columnspan=3, sticky="ew", pady=3)

        tk.Checkbutton(
            content,
            text="保存后应用到当前局面（重算金框/推荐；背包数量只记录不造假）",
            variable=apply_var,
            fg="#d7e3ee",
            bg="#101820",
            activeforeground="#ffffff",
            activebackground="#101820",
            selectcolor="#0d141b",
        ).grid(row=len(fields) + 2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        original = {
            "category": paint.category.value if paint else None,
            "level": paint.level if paint else None,
            "direction": paint.direction.value if paint else None,
            "target": paint.target_object.value if paint else None,
            "match_required": paint.match_required if paint else None,
            "upgrade_amount": paint.upgrade_amount if paint else None,
            "stage1_options": list(brush.stage1_options) if brush else [],
            "stage2_options": list(brush.stage2_options) if brush else [],
            "backpack_count": backpack_count,
        }

        def save() -> None:
            corrected = {
                "category": category_var.get(),
                "level": _optional_int(level_var.get()),
                "direction": direction_var.get(),
                "target": target_var.get(),
                "match_required": _optional_int(match_required_var.get()),
                "upgrade_amount": _optional_int(upgrade_var.get()),
                "stage1_options": _split_manual_list(stage1_var.get(), separator=","),
                "stage2_options": _split_manual_list(stage2_var.get(), separator=";"),
                "backpack_count": _optional_int(backpack_var.get()),
                "note": note_var.get().strip(),
            }
            changed = [
                key
                for key, value in corrected.items()
                if key != "note" and value != original.get(key)
            ]
            applied = self._apply_capture_correction(corrected) if apply_var.get() else False
            record_correction(
                {
                    "original": original,
                    "corrected": corrected,
                    "changed_fields": changed,
                    "applied_requested": apply_var.get(),
                    "applied_to_current_state": applied,
                    "board_rect": scene.board.rect,
                    "board_score": scene.board.score,
                },
                source="real_capture",
            )
            suffix = "，已应用到当前局面" if applied else "，仅记录未应用"
            self.status_var.set(f"已保存真实识别纠错：{len(changed)} 个字段变化{suffix}")
            self._refresh_after_correction()
            window.destroy()

        button_row = len(fields) + 4
        tk.Button(content, text="保存纠错", command=save, width=18).grid(row=button_row, column=1, sticky="e", pady=(12, 0))
        tk.Button(content, text="取消", command=window.destroy, width=12).grid(row=button_row, column=2, sticky="w", padx=(8, 0), pady=(12, 0))

    def _apply_capture_correction(self, corrected: dict) -> bool:
        if self.current_state is None:
            return False
        rule = self.current_state.current_rule
        if rule is None:
            return False

        category = _paint_category_from_value(corrected.get("category"), rule.paint.category)
        direction = _paint_direction_from_value(corrected.get("direction"), rule.direction)
        target = _paint_object_from_value(corrected.get("target"), rule.target_object)
        level = corrected.get("level") if corrected.get("level") is not None else rule.paint.level
        match_required = corrected.get("match_required") if corrected.get("match_required") is not None else rule.match_required
        upgrade_amount = corrected.get("upgrade_amount") if corrected.get("upgrade_amount") is not None else rule.upgrade_amount
        paint = replace(rule.paint, category=category, level=max(1, min(7, int(level))))
        next_rule = CurrentPaintRule(
            paint=paint,
            direction=direction,
            target_object=target,
            match_required=max(1, int(match_required)),
            upgrade_amount=max(1, int(upgrade_amount)),
            match_current=rule.match_current,
            extra_directions=rule.extra_directions,
        )
        if self.current_state.current_anchor is not None:
            match_current = min(
                _safe_current_match_count(self.current_state, next_rule),
                next_rule.match_required,
            )
            next_rule = replace(next_rule, match_current=match_current)
        self.current_state = replace(
            self.current_state,
            current_rule=next_rule,
            backpack=sort_backpack_for_rule(self.current_state.backpack, next_rule),
        )
        self._apply_corrected_brush_options(corrected)
        self.manual_correction_applied = True
        return True

    def _apply_corrected_brush_options(self, corrected: dict) -> None:
        stage1 = normalize_brush_options(
            [
                brush_type
                for value in corrected.get("stage1_options", [])
                if (brush_type := _brush_type_from_value(value)) is not None
            ]
        )
        stage2_names = corrected.get("stage2_options", [])
        affix_by_name = {affix.name: affix for affix in self.affix_catalog}
        stage2 = tuple(affix_by_name[name] for name in stage2_names if name in affix_by_name)
        if not stage1 and not stage2:
            return
        chosen = stage2[0].brush_type if stage2 else stage1[0]
        if chosen not in stage1:
            stage1 = normalize_brush_options((*stage1, chosen))
        self.brush_scenario = BrushChoiceScenario(
            stage1_options=stage1,
            chosen_stage1=chosen,
            stage2_options=stage2,
        )
        self.selected_brush_type = chosen if stage2 else None
        self.selected_affix = None

    def _refresh_after_correction(self) -> None:
        if self.current_state is None:
            return
        if self.current_scene is not None:
            self.last_capture_quality = _capture_quality_gate(self.current_scene, self.current_state.backpack)
            self.last_rule_anomalies = (
                _detect_capture_rule_anomalies(self.last_real_capture_step, self.current_scene, self.current_state)
                if self.last_capture_quality["ok"]
                else []
            )
        self.recommendations = recommend_placements(
            self.current_state,
            learning_stats=load_learning_stats() if self.current_scene is not None else None,
        )
        self.install_plan = recommend_install_plan(self.current_state)
        self.selected_backpack_index = None
        self.canvas.delete("all")
        self._draw_grid(self.current_scene.board if self.current_scene else None, self.current_state)
        if self.current_scene is not None:
            self._write_regions(self.current_scene)
        else:
            self._write_simulation()
        self._refresh_overlay()

    def _open_correction_stats_dialog(self) -> None:
        stats = load_learning_stats()
        window = tk.Toplevel(self)
        window.title("Correction Stats")
        window.configure(bg="#101820")
        window.geometry("560x460")
        window.transient(self)

        content = tk.Frame(window, bg="#101820")
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        tk.Label(
            content,
            text="真实纠错统计",
            fg="#ffffff",
            bg="#101820",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        text = tk.Text(
            content,
            bg="#0d141b",
            fg="#d7e3ee",
            insertbackground="#d7e3ee",
            relief=tk.FLAT,
            font=("Microsoft YaHei UI", 10),
            wrap=tk.WORD,
        )
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, _format_correction_stats(stats))
        text.configure(state=tk.DISABLED)

    def _open_manual_brush_dialog(self) -> None:
        window = tk.Toplevel(self)
        window.title("手动笔触选项")
        window.configure(bg="#101820")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()

        stage1_default = self.brush_scenario.stage1_options if self.brush_scenario else (BrushType.TRANSFORM, BrushType.STACK)
        chosen_default = self.brush_scenario.chosen_stage1 if self.brush_scenario else stage1_default[0]
        stage_vars = {
            brush_type: tk.BooleanVar(value=brush_type in stage1_default)
            for brush_type in BrushType
        }
        chosen_var = tk.StringVar(value=chosen_default.value)
        affix1_var = tk.StringVar()
        affix2_var = tk.StringVar()

        content = tk.Frame(window, bg="#101820")
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(content, text="一阶段：出现的两个大类", fg="#d7e3ee", bg="#101820", font=("Microsoft YaHei UI", 10, "bold")).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 6),
        )
        for col, brush_type in enumerate(BrushType):
            tk.Checkbutton(
                content,
                text=_brush_label(brush_type),
                variable=stage_vars[brush_type],
                fg="#d7e3ee",
                bg="#101820",
                activeforeground="#ffffff",
                activebackground="#101820",
                selectcolor="#0d141b",
            ).grid(row=1, column=col, sticky="w", padx=(0, 12))

        tk.Label(content, text="二阶段：已选择的大类", fg="#d7e3ee", bg="#101820").grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 4))
        chosen_menu = tk.OptionMenu(content, chosen_var, *[brush.value for brush in BrushType])
        chosen_menu.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=18)
        chosen_menu.grid(row=3, column=0, columnspan=3, sticky="w")

        tk.Label(content, text="二阶段词缀 A", fg="#d7e3ee", bg="#101820").grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 4))
        affix1_menu = tk.OptionMenu(content, affix1_var, "")
        affix1_menu.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=42)
        affix1_menu.grid(row=5, column=0, columnspan=3, sticky="ew")

        tk.Label(content, text="二阶段词缀 B", fg="#d7e3ee", bg="#101820").grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 4))
        affix2_menu = tk.OptionMenu(content, affix2_var, "")
        affix2_menu.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=42)
        affix2_menu.grid(row=7, column=0, columnspan=3, sticky="ew")

        affix_display: dict[str, AffixRule] = {}

        def refresh_affix_menus(*_args: object) -> None:
            brush_type = BrushType(chosen_var.get())
            affixes = [affix for affix in self.affix_catalog if affix.brush_type == brush_type]
            affix_display.clear()
            labels = []
            for affix in affixes:
                code = affix_code(affix) or "?"
                label = f"{code}. {affix.name}"
                affix_display[label] = affix
                labels.append(label)
            if not labels:
                labels = ["-"]
            _replace_option_menu(affix1_menu, affix1_var, labels)
            _replace_option_menu(affix2_menu, affix2_var, labels)
            if affix1_var.get() not in labels:
                affix1_var.set(labels[0])
            if affix2_var.get() not in labels:
                affix2_var.set(labels[min(1, len(labels) - 1)])

        chosen_var.trace_add("write", refresh_affix_menus)
        refresh_affix_menus()

        buttons = tk.Frame(content, bg="#101820")
        buttons.grid(row=8, column=0, columnspan=3, sticky="e", pady=(16, 0))

        def apply_manual() -> None:
            selected = normalize_brush_options([brush for brush in BrushType if stage_vars[brush].get()])
            if len(selected) != 2:
                messagebox.showwarning("Huahua Assistant", "一阶段必须正好选择两个笔触大类。")
                return
            chosen = BrushType(chosen_var.get())
            if chosen not in selected:
                messagebox.showwarning("Huahua Assistant", "二阶段大类必须来自一阶段出现的两个大类。")
                return
            affix_a = affix_display.get(affix1_var.get())
            affix_b = affix_display.get(affix2_var.get())
            if affix_a is None or affix_b is None:
                messagebox.showwarning("Huahua Assistant", "请选择两个二阶段词缀。")
                return
            self.brush_scenario = BrushChoiceScenario(stage1_options=selected, chosen_stage1=chosen, stage2_options=(affix_a, affix_b))
            self._rerender_side_panel()
            window.destroy()

        def clear_manual() -> None:
            self.brush_scenario = None
            self._rerender_side_panel()
            window.destroy()

        tk.Button(buttons, text="应用", command=apply_manual, width=10).pack(side=tk.RIGHT)
        tk.Button(buttons, text="清空", command=clear_manual, width=10).pack(side=tk.RIGHT, padx=(0, 8))
        tk.Button(buttons, text="取消", command=window.destroy, width=10).pack(side=tk.RIGHT, padx=(0, 8))

    def _draw_grid(self, board: BoardObservation | None, model_state: PuzzleState | None = None) -> None:
        self.board_cell_rects = {}
        self.backpack_slot_rects = {}
        self.brush_stage1_rects = {}
        self.brush_affix_rects = {}
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        rule_panel_w = 260 if model_state and model_state.current_rule and width >= 760 else 0
        rule_panel_gap = 18 if rule_panel_w else 0
        backpack_width = max(320, width - 48)
        _, backpack_rows, _, backpack_slot_h, backpack_gap = _backpack_layout(
            len(model_state.backpack) if model_state else 0,
            backpack_width,
        )
        summary_reserve = 30 if model_state and model_state.current_rule else 0
        backpack_reserve = 0
        if backpack_rows:
            backpack_reserve = 52 + backpack_rows * backpack_slot_h + max(0, backpack_rows - 1) * backpack_gap
        board_width = max(300, width - 48 - rule_panel_w - rule_panel_gap)
        size = min(board_width, height - backpack_reserve - summary_reserve - 48)
        size = max(260, min(size, 520))
        used_width = size + rule_panel_w + rule_panel_gap
        left = max(24, (width - used_width) // 2)
        top = 42 + summary_reserve
        cell = size / 6
        rec_by_cell = {(rec.row, rec.col): rec for rec in self.recommendations}
        plan_by_cell = {(step.row, step.col): index for index, step in enumerate(self.install_plan.steps, start=1)} if self.install_plan else {}
        best_cell = (self.recommendations[0].row, self.recommendations[0].col) if self.recommendations else None
        scope_cells = set(current_scope_cells(model_state, include_anchor=True)) if model_state else set()
        anchor_cell = model_state.current_anchor if model_state else None

        if model_state and model_state.current_rule:
            self._draw_rule_summary(model_state, width / 2, 18)
            if rule_panel_w:
                self._draw_current_rule_card(
                    model_state,
                    left + size + rule_panel_gap,
                    top,
                    rule_panel_w,
                    min(max(size, 360), height - top - 24),
                )

        for col in range(6):
            x = left + (col + 0.5) * cell
            self.canvas.create_text(
                x,
                top - 15,
                text=f"c{col + 1}",
                fill="#9fb4c8",
                font=("Microsoft YaHei UI", 9, "bold"),
            )
        for row in range(6):
            y = top + (row + 0.5) * cell
            self.canvas.create_text(
                left - 14,
                y,
                text=f"r{row + 1}",
                fill="#9fb4c8",
                font=("Microsoft YaHei UI", 9, "bold"),
                anchor=tk.E,
            )
            for col in range(6):
                tile = board.tile(row, col) if board else None
                model_tile = model_state.tile(row, col) if model_state else None
                state = tile.state if tile else (model_tile.state if model_tile else TileState.UNKNOWN)
                x1 = left + col * cell
                y1 = top + row * cell
                x2 = left + (col + 1) * cell
                y2 = top + (row + 1) * cell
                self.board_cell_rects[(row, col)] = (x1, y1, x2, y2)
                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=STATE_COLORS[state],
                    outline="#314152",
                    width=2,
                )
                if (row, col) in scope_cells:
                    self.canvas.create_rectangle(
                        x1 + 2,
                        y1 + 2,
                        x2 - 2,
                        y2 - 2,
                        outline="#f2c14e",
                        width=5 if (row, col) == anchor_cell else 3,
                    )
                if (row, col) in plan_by_cell:
                    self.canvas.create_oval(
                        x1 + 9,
                        y1 + 9,
                        x1 + 32,
                        y1 + 32,
                        fill="#f2c14e",
                        outline="#101820",
                        width=2,
                    )
                    self.canvas.create_text(
                        x1 + 20,
                        y1 + 20,
                        text=str(plan_by_cell[(row, col)]),
                        fill="#101820",
                        font=("Microsoft YaHei UI", 10, "bold"),
                    )
                if (row, col) in rec_by_cell:
                    rec = rec_by_cell[(row, col)]
                    outline = "#ffd84d" if (row, col) == best_cell else "#64e6ff"
                    self.canvas.create_rectangle(
                        x1 + 4,
                        y1 + 4,
                        x2 - 4,
                        y2 - 4,
                        outline=outline,
                        width=5 if (row, col) == best_cell else 3,
                    )
                    self.canvas.create_text(
                        x2 - 10,
                        y1 + 14,
                        text=f"r{row + 1}c{col + 1}\n{int(rec.score)}",
                        fill=outline,
                        font=("Microsoft YaHei UI", 8, "bold"),
                        anchor=tk.E,
                        justify=tk.RIGHT,
                    )
                if model_state and current_rule_is_pending(model_state) and (row, col) in rec_by_cell:
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        y2 - 13,
                        text="放初始",
                        fill="#101820" if state not in {TileState.VOID, TileState.UNKNOWN} else "#f2c14e",
                        font=("Microsoft YaHei UI", 8, "bold"),
                    )
                label = _tile_display_label(state, model_tile.paint if model_tile else None)
                if tile:
                    label = f"{label}\n{tile.confidence:.2f}"
                elif model_tile and model_tile.paint:
                    prefix = "初" if (row, col) == anchor_cell else ""
                    label = f"{prefix}{label}\nLv{model_tile.paint.level}"
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text=label,
                    fill="#ffffff" if state in {TileState.VOID, TileState.UNKNOWN} else "#101820",
                    font=("Microsoft YaHei UI", 14, "bold"),
                    justify=tk.CENTER,
                )

        self._draw_backpack_bar(model_state, 24, top + size + 18, width - 48)

    def _draw_current_rule_card(
        self,
        model_state: PuzzleState,
        left: float,
        top: float,
        width: float,
        height: float,
    ) -> None:
        rule = model_state.current_rule
        if rule is None:
            return
        pending = current_rule_is_pending(model_state)
        title = "待放置初始颜料" if pending else "当前初始颜料"
        outline = "#ff5d5d" if pending else "#f2c14e"
        fill = "#111b24"
        self.canvas.create_rectangle(
            left,
            top,
            left + width,
            top + height,
            fill=fill,
            outline=outline,
            width=4 if pending else 3,
        )
        self.canvas.create_text(
            left + 12,
            top + 14,
            text=title,
            fill="#ffffff",
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor=tk.W,
        )
        if self.manual_correction_applied:
            self.canvas.create_text(
                left + width - 12,
                top + 14,
                text="人工修正",
                fill="#f2c14e",
                font=("Microsoft YaHei UI", 9, "bold"),
                anchor=tk.E,
            )

        icon_x1 = left + 16
        icon_y1 = top + 42
        icon_x2 = left + 112
        icon_y2 = icon_y1 + 86
        icon_fill = "#ffd84d" if rule.paint.category.value == "gameplay" else "#b983ff"
        self.canvas.create_rectangle(icon_x1, icon_y1, icon_x2, icon_y2, fill=icon_fill, outline="#101820", width=3)
        self.canvas.create_text(
            (icon_x1 + icon_x2) / 2,
            icon_y1 + 30,
            text=_paint_short_label(rule.paint),
            fill="#101820",
            font=("Microsoft YaHei UI", 15, "bold"),
        )
        self.canvas.create_text(
            (icon_x1 + icon_x2) / 2,
            icon_y1 + 62,
            text=f"Lv{rule.paint.level}",
            fill="#101820",
            font=("Microsoft YaHei UI", 17, "bold"),
        )

        lines = [
            f"方向: {_direction_display_label(rule.direction)}",
            f"目标: {_target_display_label(rule.target_object)}",
            f"匹配: {rule.match_current}/{rule.match_required}",
            f"效果: 提升{rule.upgrade_amount}级",
        ]
        y = icon_y1 + 8
        for line in lines:
            self.canvas.create_text(
                icon_x2 + 12,
                y,
                text=line,
                fill="#d7e3ee",
                font=("Microsoft YaHei UI", 9),
                anchor=tk.W,
            )
            y += 21

        body_top = icon_y2 + 18
        if pending and self.brush_scenario is not None:
            body_top = self._draw_brush_choice_panel(left + 12, body_top, width - 24)

        hint = self._current_rule_hint(pending)
        self.canvas.create_rectangle(
            left + 12,
            top + height - 44,
            left + width - 12,
            top + height - 14,
            fill="#203044",
            outline="#314152",
            width=1,
        )
        self.canvas.create_text(
            left + width / 2,
            top + height - 29,
            text=hint,
            fill="#f5f7fa",
            font=("Microsoft YaHei UI", 9, "bold"),
        )

    def _draw_brush_choice_panel(self, left: float, top: float, width: float) -> float:
        scenario = self.brush_scenario
        if scenario is None:
            return top

        self.canvas.create_text(
            left,
            top,
            text="二阶段选择",
            fill="#ffffff",
            font=("Microsoft YaHei UI", 10, "bold"),
            anchor=tk.W,
        )
        y = top + 24
        self.canvas.create_text(
            left,
            y,
            text="1. 选择大类",
            fill="#95abc1",
            font=("Microsoft YaHei UI", 9),
            anchor=tk.W,
        )
        y += 16
        button_gap = 8
        button_w = (width - button_gap) / 2
        button_h = 56
        learning_stats = load_learning_stats() if self.current_scene is not None else None
        stage1_recs = (
            recommend_brush_types(self.current_state, scenario.stage1_options, learning_stats=learning_stats)
            if self.current_state
            else []
        )
        score_by_type = {rec.brush_type: rec.score for rec in stage1_recs}
        reason_by_type = {rec.brush_type: rec.reason for rec in stage1_recs}
        best_type = stage1_recs[0].brush_type if stage1_recs else None
        for index, brush_type in enumerate(scenario.stage1_options):
            x1 = left + index * (button_w + button_gap)
            x2 = x1 + button_w
            selected = brush_type == self.selected_brush_type
            self.brush_stage1_rects[brush_type] = (x1, y, x2, y + button_h)
            rec_mark = "推荐 " if brush_type == best_type else ""
            label = (
                f"{rec_mark}{brush_code(brush_type)} {_brush_label(brush_type)}\n"
                f"评分 {score_by_type.get(brush_type, 0):.0f}\n"
                f"{reason_by_type.get(brush_type, '按当前局面估算')}"
            )
            self._draw_canvas_button(
                x1,
                y,
                x2,
                y + button_h,
                label,
                selected=selected,
                disabled=self.selected_affix is not None,
                font_size=8,
            )

        y += button_h + 14
        if self.selected_brush_type is None:
            self.canvas.create_text(
                left,
                y,
                text="先点上方一个大类，才会出现2个词缀",
                fill="#f2c14e",
                font=("Microsoft YaHei UI", 9),
                anchor=tk.W,
            )
            return y + 26

        self.canvas.create_text(
            left,
            y,
            text="2. 选择词缀",
            fill="#95abc1",
            font=("Microsoft YaHei UI", 9),
            anchor=tk.W,
        )
        y += 18
        affix_recs = (
            recommend_affixes(self.current_state, scenario.stage2_options, learning_stats=learning_stats)
            if self.current_state
            else []
        )
        score_by_affix = {rec.affix: rec.score for rec in affix_recs}
        reason_by_affix = {rec.affix: rec.reason for rec in affix_recs}
        for index, affix in enumerate(scenario.stage2_options):
            selected = affix == self.selected_affix
            x1 = left
            x2 = left + width
            y2 = y + 52
            self.brush_affix_rects[index] = (x1, y, x2, y2)
            score_text = f"评分 {score_by_affix.get(affix, 0):.0f}"
            label = f"{affix_code(affix) or '?'} {affix.name}\n{score_text} | {reason_by_affix.get(affix, '按当前局面估算收益')}"
            self._draw_canvas_button(
                x1,
                y,
                x2,
                y2,
                label,
                selected=selected,
                disabled=self.selected_affix is not None and not selected,
                font_size=8,
            )
            y += 60

        return y

    def _draw_canvas_button(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        text: str,
        *,
        selected: bool = False,
        disabled: bool = False,
        font_size: int = 9,
    ) -> None:
        fill = "#f2c14e" if selected else "#223248" if not disabled else "#18222e"
        outline = "#ffffff" if selected else "#3e536b"
        text_fill = "#101820" if selected else "#d7e3ee" if not disabled else "#738497"
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=2)
        self.canvas.create_text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            text=text,
            fill=text_fill,
            font=("Microsoft YaHei UI", font_size, "bold"),
            justify=tk.CENTER,
            width=max(20, int(x2 - x1 - 12)),
        )

    def _current_rule_hint(self, pending: bool) -> str:
        if not pending:
            return "金框范围内安装背包颜料"
        if self.brush_scenario is None:
            return "点击左侧推荐格放置初始颜料"
        if self.selected_brush_type is None:
            return "先选一阶段大类，再点重新快速获取"
        if self.selected_affix is None:
            return "二阶段词条出现后选1条"
        return "已选词条，点击左侧推荐格放置初始颜料"

    def _draw_rule_summary(self, model_state: PuzzleState, center_x: float, top: float) -> None:
        rule = model_state.current_rule
        if rule is None:
            return
        if current_rule_is_pending(model_state):
            summary = (
                f"待放置初始颜料 Lv{rule.paint.level} {_paint_short_label(rule.paint)} | "
                f"目标: {_direction_display_label(rule.direction)} {_target_display_label(rule.target_object)} "
                f"(0/{rule.match_required}) | 请选择空白地块"
            )
        else:
            summary = (
                f"初始颜料 Lv{rule.paint.level} {_paint_short_label(rule.paint)} | "
                f"目标: {_direction_display_label(rule.direction)} {_target_display_label(rule.target_object)} "
                f"({rule.match_current}/{rule.match_required}) | "
                f"效果: 提升{rule.upgrade_amount}级"
            )
        self.canvas.create_text(
            center_x,
            top,
            text=summary,
            fill="#f5f7fa",
            font=("Microsoft YaHei UI", 10, "bold"),
            anchor=tk.N,
        )

    def _draw_backpack_bar(self, model_state: PuzzleState | None, left: float, top: float, width: float) -> None:
        if model_state is None or not model_state.backpack:
            return
        per_row, rows, slot_w, slot_h, gap = _backpack_layout(len(model_state.backpack), width)
        row_width = per_row * slot_w + max(0, per_row - 1) * gap
        start_x = left + max(0, (width - row_width) / 2)
        self.canvas.create_text(
            left,
            top,
            text=f"背包 {len(model_state.backpack)} 格",
            fill="#d7e3ee",
            font=("Microsoft YaHei UI", 10, "bold"),
            anchor=tk.W,
        )
        if model_state.current_rule is not None:
            action_hint = (
                "新的初始颜料待放置：先点棋盘空白格"
                if current_rule_is_pending(model_state)
                else f"已按 {_target_display_label(model_state.current_rule.target_object)} 目标优先排序，先点背包再点金框内格子安装"
            )
            self.canvas.create_text(
                left,
                top + 18,
                text=action_hint,
                fill="#95abc1",
                font=("Microsoft YaHei UI", 9),
                anchor=tk.W,
            )
        planned = Counter(step.paint for step in self.install_plan.steps) if self.install_plan else Counter()
        for index, paint in enumerate(model_state.backpack):
            row = index // per_row
            col = index % per_row
            x1 = start_x + col * (slot_w + gap)
            y1 = top + 38 + row * (slot_h + gap)
            x2 = x1 + slot_w
            y2 = y1 + slot_h
            self.backpack_slot_rects[index] = (x1, y1, x2, y2)
            planned_here = planned[paint] > 0
            if planned_here:
                planned[paint] -= 1
            fill = "#ffe08a" if planned_here else "#172331"
            selected_here = self.selected_backpack_index == index
            outline = "#64e6ff" if selected_here else "#f2c14e" if planned_here else "#314152"
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=4 if selected_here else 3 if planned_here else 2)
            self.canvas.create_text(
                (x1 + x2) / 2,
                y1 + 12,
                text=_paint_short_label(paint),
                fill="#101820" if planned_here else "#d7e3ee",
                font=("Microsoft YaHei UI", 8, "bold"),
            )
            self.canvas.create_text(
                (x1 + x2) / 2,
                y1 + 28,
                text=f"#{index + 1}  Lv{paint.level}",
                fill="#101820" if planned_here else "#d7e3ee",
                font=("Microsoft YaHei UI", 8, "bold"),
            )

    def _write_regions(self, scene: SceneObservation | None) -> None:
        self.region_text.configure(state=tk.NORMAL)
        self.region_text.delete("1.0", tk.END)
        if scene is None:
            self.region_text.insert(tk.END, "识别区域\n\n")
            self._write_perf_debug()
            self.region_text.insert(tk.END, "无")
            self.region_text.configure(state=tk.DISABLED)
            return

        region_names = {
            RegionKind.CURRENT_PAINT: "当前初始颜料",
            RegionKind.BACKPACK: "背包",
            RegionKind.BRUSH_OPTIONS: "笔触选项",
            RegionKind.CONFIRM_BUTTON: "确认按钮",
            RegionKind.FINISH_DRAWING: "完成绘制",
            RegionKind.FORCE_COMPLETE: "强制完成",
            RegionKind.REFRESH_AFFIX: "刷新词缀",
        }
        self.region_text.insert(tk.END, "识别区域\n\n")
        self._write_perf_debug()
        if self.manual_correction_applied:
            self.region_text.insert(tk.END, "已应用人工纠错\n当前局面已使用纠错值重算；上方识别区域仍保留原始截图识别值。\n\n")
        self.region_text.insert(
            tk.END,
            f"棋盘\nx={scene.board.rect.x} y={scene.board.rect.y} "
            f"w={scene.board.rect.w} h={scene.board.rect.h}\n"
            f"置信={scene.board.score:.2f}\n\n",
        )
        self._write_coordinate_debug(scene)
        for region in scene.regions:
            name = region_names[region.kind]
            text = (
                f"{name}\n"
                f"x={region.rect.x} y={region.rect.y} "
                f"w={region.rect.w} h={region.rect.h}\n"
                f"置信={region.confidence:.2f}\n\n"
            )
            self.region_text.insert(tk.END, text)
        if scene.current_paint is not None:
            paint = scene.current_paint
            level = paint.level if paint.level is not None else "?"
            self.region_text.insert(tk.END, "当前初始颜料\n")
            self.region_text.insert(
                tk.END,
                f"类型={paint.category.value} ({paint.category_confidence:.2f})\n"
                f"小类={paint.subtype}\n"
                f"宝箱={paint.is_treasure}\n"
                f"等级={level} ({paint.level_confidence:.2f})\n"
                f"目标={paint.target_object.value}\n"
                f"方向={paint.direction.value}\n"
                f"升级={paint.upgrade_amount}\n"
                f"匹配={paint.match_current}/{paint.match_required}\n"
                f"提示={paint.raw_text_hint}\n"
                f"关键词={_format_keyword_scores(paint.keyword_scores)}\n"
                "截图=runtime/current_paint_*.png\n\n",
            )
        if scene.brush_options is not None:
            brush = scene.brush_options
            self.region_text.insert(tk.END, "笔触选项识别\n")
            self.region_text.insert(
                tk.END,
                f"一阶段={', '.join(brush.stage1_options) or '-'}\n"
                f"已选={brush.chosen_stage1 or '-'}\n"
                f"二阶段={'; '.join(brush.stage2_options) or '-'}\n"
                f"置信={brush.confidence:.2f}\n"
                f"一阶段评分={_format_keyword_scores(brush.stage1_scores)}\n"
                f"二阶段评分={_format_keyword_scores(brush.stage2_scores)}\n"
                "截图=runtime/brush_options_panel.png\n\n",
            )
        if scene.backpack is not None:
            self.region_text.insert(
                tk.END,
                f"背包识别\n"
                f"可见={len(scene.backpack.paints)} 累计={len(self.detected_backpack_cache)}\n"
                f"置信={scene.backpack.confidence:.2f}\n"
                "支持手动滚动：你在游戏里滚动背包后，再点完整识别/快速获取\n\n",
            )
        if self.last_capture_quality is not None:
            quality = self.last_capture_quality
            status = "通过" if quality.get("ok") else "暂缓"
            self.region_text.insert(
                tk.END,
                f"识别质量：{status} 评分={quality.get('score', 0.0):.2f}\n"
                f"{quality.get('reason', '-')}\n\n",
            )
        if self.last_rule_anomalies:
            self.region_text.insert(tk.END, "规则异常提醒\n")
            for anomaly in self.last_rule_anomalies[:4]:
                self.region_text.insert(tk.END, f"- {_format_rule_anomaly(anomaly)}\n")
            blocked = any(item.get("severity") == "block" for item in self.last_rule_anomalies)
            note = "warn/info 仅记录提醒；block 会阻止本次进入高分轨迹。"
            if blocked:
                note = "存在 block 级规则异常，本次截图不会进入高分轨迹；建议复查识别或补充规则。"
            self.region_text.insert(tk.END, f"{note}\n\n")
        self._write_learning_stats()
        self._write_backpack()
        self._write_install_plan()
        self._write_recommendations()
        self._write_brush_recommendations()
        self._write_special_actions()
        self.region_text.configure(state=tk.DISABLED)

    def _write_perf_debug(self) -> None:
        self.region_text.insert(tk.END, "性能面板\n")
        for line in _format_perf_details(self.last_perf):
            self.region_text.insert(tk.END, f"{line}\n")
        self.region_text.insert(tk.END, "\n")

    def _write_coordinate_debug(self, scene: SceneObservation) -> None:
        window = self.last_capture_window
        capture_size = self.last_capture_size
        board_rect = scene.board.rect
        board_screen_rect = self._screen_rect_from_capture_rect(board_rect)
        board_center = self._screen_point_for_capture_rect(board_rect)

        if window is None:
            window_text = "无窗口信息"
        else:
            window_text = f"({window.left},{window.top}) {window.width}x{window.height}"
        if capture_size is None:
            capture_text = "未知"
            scale_text = "未知"
        else:
            capture_w, capture_h = capture_size
            capture_text = f"{capture_w}x{capture_h}"
            if window is not None and capture_w > 0 and capture_h > 0:
                scale_text = f"x={window.width / capture_w:.4f} y={window.height / capture_h:.4f}"
            else:
                scale_text = "无窗口，使用截图坐标"

        self.region_text.insert(
            tk.END,
            "坐标调试\n"
            f"窗口={window_text}\n"
            f"截图={capture_text}\n"
            f"缩放={scale_text}\n"
            f"棋盘截图=x{board_rect.x} y{board_rect.y} w{board_rect.w} h{board_rect.h}\n",
        )
        if board_screen_rect is not None:
            x1, y1, x2, y2 = board_screen_rect
            self.region_text.insert(tk.END, f"棋盘屏幕=({x1},{y1})-({x2},{y2})\n")
        if board_center is not None:
            self.region_text.insert(tk.END, f"棋盘中心=({board_center[0]},{board_center[1]})\n")
        for index, rec in enumerate(self.recommendations[:2], start=1):
            point = self._screen_point_for_board_cell(rec.row, rec.col)
            if point is None:
                continue
            self.region_text.insert(
                tk.END,
                f"推荐{index}=r{rec.row + 1}c{rec.col + 1} -> 屏幕({point[0]},{point[1]})\n",
            )
        if scene.backpack is not None and scene.backpack.paints:
            slot_point = self._screen_point_for_backpack_slot(0)
            if slot_point is not None:
                self.region_text.insert(tk.END, f"背包#1=屏幕({slot_point[0]},{slot_point[1]})\n")
        if scene.brush_options is not None:
            brush_point = self._screen_point_for_capture_rect(scene.brush_options.panel_rect)
            if brush_point is not None:
                self.region_text.insert(tk.END, f"词条面板中心=屏幕({brush_point[0]},{brush_point[1]})\n")
        self.region_text.insert(tk.END, "\n")

    def _write_simulation(self) -> None:
        self.region_text.configure(state=tk.NORMAL)
        self.region_text.delete("1.0", tk.END)
        self.region_text.insert(tk.END, "模拟棋局\n\n")
        if self.current_state and self.current_state.current_rule:
            rule = self.current_state.current_rule
            treasure = "是" if rule.paint.is_treasure else "否"
            pending = current_rule_is_pending(self.current_state)
            anchor_text = "待选择" if pending or self.current_state.current_anchor is None else f"({self.current_state.current_anchor[0] + 1}, {self.current_state.current_anchor[1] + 1})"
            scope_count = 0 if pending else len(current_scope_cells(self.current_state, include_anchor=True))
            note = "请选择一个空白地块放置新的初始颜料" if pending else f"每安装1次都会立即结算，不用等{rule.match_required}/{rule.match_required}"
            self.region_text.insert(
                tk.END,
                f"当前初始颜料\n"
                f"类型={_paint_display_label(rule.paint)}\n"
                f"宝箱={treasure}\n"
                f"方向={_direction_display_label(rule.direction)}\n"
                f"目标={_target_display_label(rule.target_object)}\n"
                f"效果=范围内目标颜料提升{rule.upgrade_amount}级\n"
                f"进度={rule.match_current}/{rule.match_required}\n"
                f"锚点={anchor_text}\n"
                f"金框覆盖={scope_count}格\n"
                f"说明={note}\n\n",
            )
        self._write_backpack()
        self._write_install_plan()
        self._write_recommendations()
        self._write_brush_recommendations()
        self._write_special_actions()
        self.region_text.configure(state=tk.DISABLED)

    def _write_recommendations(self) -> None:
        if not self.recommendations:
            reason = "背包为空 / 真实模式尚未识别背包" if self.current_state and not self.current_state.backpack else "暂无"
            self.region_text.insert(tk.END, f"单步预览\n{reason}\n\n")
            return
        pending = self.current_state is not None and current_rule_is_pending(self.current_state)
        self.region_text.insert(tk.END, "初始颜料放置推荐\n" if pending else "单步预览（安装即生效）\n")
        for index, rec in enumerate(self.recommendations, start=1):
            mark = "最佳 " if index == 1 else ""
            action = "放初始" if pending else "安装"
            self.region_text.insert(
                tk.END,
                f"{mark}{index}. {action} r{rec.row + 1} c{rec.col + 1} "
                f"评分={rec.score:.1f} {rec.reason}\n",
            )
            for line in _recommendation_explanation_lines(rec):
                self.region_text.insert(tk.END, f"   {line}\n")
            if self.overlay_var.get():
                screen = self._screen_point_for_board_cell(rec.row, rec.col)
                if screen is not None:
                    self.region_text.insert(tk.END, f"   坐标：棋盘({rec.row + 1},{rec.col + 1}) 屏幕({screen[0]},{screen[1]})\n")
                backpack_index = self._visible_backpack_index_for_paint(rec.paint)
                if backpack_index is not None:
                    slot_point = self._screen_point_for_backpack_slot(backpack_index)
                    if slot_point is not None:
                        self.region_text.insert(tk.END, f"   先点背包#{backpack_index + 1} 屏幕({slot_point[0]},{slot_point[1]})\n")
        self.region_text.insert(tk.END, "\n")

    def _screen_point_for_board_cell(self, row: int, col: int) -> tuple[int, int] | None:
        scene = self.current_scene
        if scene is None:
            return None
        tile = scene.board.tile(row, col)
        return self._screen_point_for_capture_rect(tile.rect)

    def _visible_backpack_index_for_paint(self, paint: Paint | None) -> int | None:
        scene = self.current_scene
        if paint is None or scene is None or scene.backpack is None:
            return None
        for index, observed in enumerate(scene.backpack.paints):
            observed_paint = Paint(
                category=observed.category,
                level=observed.level,
                subtype=observed.subtype,
                is_treasure=observed.is_treasure,
            )
            if observed_paint == paint:
                return index
        return None

    def _screen_point_for_backpack_slot(self, index: int) -> tuple[int, int] | None:
        scene = self.current_scene
        if scene is None or scene.backpack is None or not 0 <= index < len(scene.backpack.paints):
            return None
        rect = scene.backpack.paints[index].slot_rect
        return self._screen_point_for_capture_rect(rect)

    def _write_backpack(self) -> None:
        self.region_text.insert(tk.END, "背包（已按当前目标优先排序；滚动背包后再次完整识别可累计）\n")
        if not self.current_state or not self.current_state.backpack:
            self.region_text.insert(tk.END, "空 / 等待识别\n\n")
            return
        for index, paint in enumerate(self.current_state.backpack, start=1):
            self.region_text.insert(tk.END, f"{index}. {_paint_display_label(paint)}\n")
        self.region_text.insert(tk.END, "\n")

    def _write_learning_stats(self) -> None:
        stats = load_learning_stats()
        if not stats:
            return
        conflicts = stats.get("event_counts", {}).get("rule_conflict", 0)
        mismatches = sum((stats.get("rule_mismatches") or {}).values())
        anomalies = stats.get("rule_anomalies") or {}
        anomaly_severity = stats.get("rule_anomaly_severity") or {}
        affix_seen = stats.get("affix_seen") or {}
        affix_code_seen = stats.get("affix_code_seen") or {}
        correction_fields = stats.get("correction_fields") or {}
        trajectories = stats.get("trajectories", 0)
        best_score = stats.get("trajectory_best_score")
        quick_steps = stats.get("quick_capture_steps", 0)
        ignored_quick = stats.get("quick_capture_ignored", 0)
        ignored_trajectory = stats.get("trajectory_step_ignored", 0)
        quick_avg = stats.get("quick_capture_avg_elapsed")
        finished_runs = stats.get("finished_runs", 0)
        finished_skipped = stats.get("finished_skipped", 0)
        finished_best = stats.get("finished_best_score")
        high_value_seen = stats.get("high_value_seen", 0)
        high_value_counts = stats.get("high_value_counts") or {}
        inferred_actions = stats.get("inferred_actions") or {}
        inferred_codes = stats.get("inferred_action_codes") or {}
        refresh_events = stats.get("refresh_events") or {}
        prior_samples = stats.get("policy_prior_samples") or {}
        recommendation_shorts = stats.get("recommendation_shorts") or {}
        recommendation_tags = stats.get("recommendation_tags") or {}
        high_value_recommendation_tags = stats.get("high_value_recommendation_tags") or {}
        top_affixes = _top_affix_display_items(affix_seen, affix_code_seen, limit=3)
        top_corrections = sorted(correction_fields.items(), key=lambda item: item[1], reverse=True)[:3]
        top_actions = sorted(inferred_actions.items(), key=lambda item: item[1], reverse=True)[:3]
        top_codes = sorted(inferred_codes.items(), key=lambda item: item[1], reverse=True)[:5]
        top_rec_shorts = sorted(recommendation_shorts.items(), key=lambda item: item[1], reverse=True)[:3]
        top_rec_tags = sorted(recommendation_tags.items(), key=lambda item: item[1], reverse=True)[:4]
        top_high_value_tags = sorted(high_value_recommendation_tags.items(), key=lambda item: item[1], reverse=True)[:4]
        affix_text = ", ".join(f"{name}x{count}" for name, count in top_affixes) if top_affixes else "-"
        correction_text = ", ".join(f"{_field_display_name(name)}x{count}" for name, count in top_corrections) if top_corrections else "-"
        action_text = ", ".join(f"{name}x{count}" for name, count in top_actions) if top_actions else "-"
        code_text = ", ".join(f"{name}x{count}" for name, count in top_codes) if top_codes else "-"
        rec_short_text = ", ".join(f"{name}x{count}" for name, count in top_rec_shorts) if top_rec_shorts else "-"
        rec_tag_text = ", ".join(f"{name}x{count}" for name, count in top_rec_tags) if top_rec_tags else "-"
        high_value_tag_text = ", ".join(f"{name}x{count}" for name, count in top_high_value_tags) if top_high_value_tags else "-"
        best_score_text = f"{best_score:.0f}" if isinstance(best_score, (int, float)) else "-"
        quick_avg_text = f"{quick_avg:.2f}s" if isinstance(quick_avg, (int, float)) else "-"
        finished_best_text = f"{finished_best:.0f}" if isinstance(finished_best, (int, float)) else "-"
        self.region_text.insert(
            tk.END,
            "学习记录\n"
            f"事件={stats.get('total_events', 0)} 纠错={stats.get('corrections', 0)} 词条冲突={conflicts} 规则异常={mismatches}\n"
            f"异常校验={anomalies} 严重度={anomaly_severity}\n"
            f"快速获取={quick_steps} 平均={quick_avg_text} 忽略低质={ignored_quick} 信任拦截={ignored_trajectory} 已结束局数={finished_runs} 跳过结算={finished_skipped} 最佳内部评分={finished_best_text}\n"
            f"高价值识别={high_value_seen} 明细={high_value_counts}\n"
            f"{_risk_hint_text(stats)}\n"
            f"高频词条={affix_text}\n"
            f"高频纠错={correction_text}\n"
            f"高价值轨迹={trajectories} 最佳={best_score_text}\n"
            f"推断动作={action_text}\n"
            f"动作短码={code_text} 强制={stats.get('forced_events', 0)} 刷新={refresh_events} 升级事件={stats.get('upgrade_events', 0)}\n"
            f"推荐语义={rec_short_text} 标签={rec_tag_text} 高价值标签={high_value_tag_text}\n"
            f"经验样本：落点={prior_samples.get('placements', 0)} 大类={prior_samples.get('brushes', 0)} 词条={prior_samples.get('affixes', 0)}\n"
            f"宝箱占比={stats.get('treasure_ratio', 0.0):.3f}\n"
            "文件=runtime/observations.jsonl, runtime/corrected_observations.jsonl, runtime/high_score_trajectories.jsonl, runtime/stats.json\n\n",
        )

    def _write_install_plan(self) -> None:
        self.region_text.insert(tk.END, "安装计划（逐步模拟，每步立即结算）\n")
        if self.install_plan is None:
            self.region_text.insert(tk.END, "等待当前初始颜料和背包识别\n\n")
            return
        self.region_text.insert(tk.END, f"{'可完成' if self.install_plan.completes_match else '未完成'}: {self.install_plan.reason}\n")
        for index, step in enumerate(self.install_plan.steps, start=1):
            self.region_text.insert(
                tk.END,
                f"{index}. r{step.row + 1} c{step.col + 1} <- {_paint_display_label(step.paint)}\n",
            )
        self.region_text.insert(tk.END, "\n")

    def _write_brush_recommendations(self) -> None:
        if self.current_state is None:
            return
        if current_rule_is_pending(self.current_state) and self.brush_scenario is not None:
            self.region_text.insert(tk.END, "笔触选择流程\n")
            if self.selected_brush_type is None:
                self.region_text.insert(tk.END, "1. 先在右侧选择转锋/堆砌/扩散大类\n2. 点“重新快速获取”读取二阶段词条\n\n")
                return
            if self.selected_affix is None:
                self.region_text.insert(tk.END, f"已选大类：{_brush_label(self.selected_brush_type)}；请点“重新快速获取”后选择二阶段词条\n\n")
                return
        stage1 = (
            self.brush_scenario.stage1_options
            if self.brush_scenario
            else (BrushType.TRANSFORM, BrushType.STACK, BrushType.SPREAD)
        )
        learning_stats = load_learning_stats() if self.current_scene is not None else None
        self.region_text.insert(tk.END, "笔触一阶段\n")
        for rec in recommend_brush_types(self.current_state, stage1, learning_stats=learning_stats):
            self.region_text.insert(
                tk.END,
                f"- {brush_code(rec.brush_type)} {_brush_label(rec.brush_type)} 评分={rec.score:.1f} {rec.reason}\n",
            )
        self.region_text.insert(tk.END, "\n")

        if not self.brush_scenario:
            self.region_text.insert(tk.END, "笔触二阶段\n等待真实选项识别\n\n")
            return
        self.region_text.insert(tk.END, f"笔触二阶段（{_brush_label(self.brush_scenario.chosen_stage1)}）\n")
        for rec in recommend_affixes(
            self.current_state,
            self.brush_scenario.stage2_options,
            learning_stats=learning_stats,
        ):
            self.region_text.insert(
                tk.END,
                f"- {affix_code(rec.affix) or '?'} {rec.affix.name} 评分={rec.score:.1f} {rec.reason}\n",
            )
        self.region_text.insert(tk.END, "\n")

    def _write_special_actions(self) -> None:
        if self.current_state is None:
            return
        resources = self._resource_state()
        actions = recommend_special_actions(self.current_state, resources, self.brush_scenario)
        self.region_text.insert(tk.END, "特殊动作\n")
        self.region_text.insert(
            tk.END,
            f"资源：强制={resources.force_remaining}/3 "
            f"刷新={resources.refresh_remaining}/2 "
            f"5级宠物={'有' if resources.refresh_available else '无'}\n",
        )
        if not actions:
            refresh_state = "可用" if resources.refresh_available else "未启用"
            self.region_text.insert(tk.END, f"- 刷新={refresh_state}；等待可操作局面\n\n")
            return
        for action in actions:
            mark = "建议用" if action.recommended else "暂缓"
            self.region_text.insert(
                tk.END,
                f"- {mark} {action.action.value} 评分={action.score:.1f} {action.reason}\n",
            )
        self.region_text.insert(tk.END, "\n")

    def _resource_state(self) -> ResourceState:
        force_used = _clamp_int_var(self.force_used_var, 0, 3)
        refresh_used = _clamp_int_var(self.refresh_used_var, 0, 2)
        refresh_available = self.refresh_available_var.get()
        return ResourceState(
            force_remaining=max(0, 3 - force_used),
            force_used=force_used,
            refresh_available=refresh_available,
            refresh_remaining=max(0, 2 - refresh_used) if refresh_available else 0,
        )

    def _on_close(self) -> None:
        self._destroy_overlay_window()
        if self.auto_follow_after_id is not None:
            self.after_cancel(self.auto_follow_after_id)
            self.auto_follow_after_id = None
        self._stop_auto_follow_pump()
        self.destroy()


def main() -> None:
    app = HuahuaApp()
    app.mainloop()


def _format_keyword_scores(scores: dict[str, float]) -> str:
    if not scores:
        return "-"
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:4]
    return ", ".join(f"{key}:{value:.2f}" for key, value in top)


def _risk_hint_text(stats: dict) -> str:
    risky = stats.get("risky_fields") or []
    corrections = stats.get("corrections", 0)
    if risky:
        labels = "、".join(_field_display_name(field) for field in risky[:4])
        return f"高风险字段：{labels}。本次建议重点复查这些项。"
    if corrections < 3:
        return "纠错样本不足：先积累几次真实纠错，工具再判断高风险字段。"
    return "暂无高风险字段：按当前纠错样本看识别较稳定。"


def _format_correction_stats(stats: dict) -> str:
    corrections = stats.get("corrections", 0)
    quick_steps = stats.get("quick_capture_steps", 0)
    ignored_quick = stats.get("quick_capture_ignored", 0)
    ignored_trajectory = stats.get("trajectory_step_ignored", 0)
    quick_avg = stats.get("quick_capture_avg_elapsed")
    quick_avg_text = f"{quick_avg:.2f}s" if isinstance(quick_avg, (int, float)) else "-"
    finished_best = stats.get("finished_best_score")
    finished_best_text = f"{finished_best:.0f}" if isinstance(finished_best, (int, float)) else "-"
    lines = [
        f"真实纠错样本：{corrections}",
        _risk_hint_text(stats),
        f"快速获取样本：{quick_steps} 平均耗时={quick_avg_text} 低质忽略={ignored_quick} 信任拦截={ignored_trajectory}",
        f"已结束局数：{stats.get('finished_runs', 0)} 跳过结算={stats.get('finished_skipped', 0)} 最佳内部评分={finished_best_text}",
        f"高价值识别：{stats.get('high_value_seen', 0)} 明细={stats.get('high_value_counts', {})}",
        "",
        "字段准确率",
    ]
    field_accuracy = stats.get("field_accuracy") or {}
    if not field_accuracy:
        lines.append("暂无纠错样本。请真实完整识别后用纠错识别保存几次。")
    else:
        ranked = sorted(
            field_accuracy.items(),
            key=lambda item: (item[1].get("error_rate", 0.0), item[1].get("errors", 0)),
            reverse=True,
        )
        for field, data in ranked:
            samples = data.get("samples", 0)
            errors = data.get("errors", 0)
            accuracy = data.get("accuracy", 0.0)
            error_rate = data.get("error_rate", 0.0)
            risk = " 高风险" if field in (stats.get("risky_fields") or []) else ""
            lines.append(
                f"- {_field_display_name(field)}: 样本{samples} 错{errors} "
                f"准确率{accuracy:.0%} 错误率{error_rate:.0%}{risk}"
            )
    lines.extend(
        [
            "",
            f"词缀冲突记录：{stats.get('event_counts', {}).get('rule_conflict', 0)}",
            f"经验规则不一致：{sum((stats.get('rule_mismatches') or {}).values())}",
            f"宝箱识别占比：{stats.get('treasure_ratio', 0.0):.1%}",
            f"真实高分轨迹：{stats.get('trajectories', 0)} 条",
            f"快速获取动作：{stats.get('quick_capture_actions', {})}",
            "",
            "说明：这里只统计真实 Capture 后保存的纠错；模拟模式不会进入统计。",
        ]
    )
    return "\n".join(lines)


def _top_affix_display_items(affix_seen: dict, affix_code_seen: dict, *, limit: int) -> list[tuple[str, int]]:
    if affix_code_seen:
        return sorted(
            ((affix_display_from_code(str(code)) or str(code), int(count)) for code, count in affix_code_seen.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
    items: list[tuple[str, int]] = []
    for name, count in affix_seen.items():
        code = affix_code(str(name))
        label = f"{code} {name}" if code else str(name)
        try:
            parsed_count = int(count)
        except (TypeError, ValueError):
            parsed_count = 0
        items.append((label, parsed_count))
    return sorted(items, key=lambda item: item[1], reverse=True)[:limit]


def _field_display_name(field: str) -> str:
    return {
        "category": "类型",
        "level": "等级",
        "direction": "方向",
        "target": "目标",
        "match_required": "需求数",
        "upgrade_amount": "升级数",
        "stage1_options": "一阶段",
        "stage2_options": "二阶段词缀",
        "backpack_count": "背包数量",
    }.get(field, field)


def _paint_category_from_value(value, fallback):
    try:
        return PaintCategory(value)
    except (TypeError, ValueError):
        return fallback


def _paint_direction_from_value(value, fallback):
    try:
        return PaintDirection(value)
    except (TypeError, ValueError):
        return fallback


def _paint_object_from_value(value, fallback):
    try:
        return PaintObject(value)
    except (TypeError, ValueError):
        return fallback


def _safe_current_match_count(state: PuzzleState, rule: CurrentPaintRule) -> int:
    if state.current_anchor is None:
        return 0
    row, col = state.current_anchor
    cells = _scope_cells_for_display(row, col, rule.direction)
    for extra in rule.extra_directions:
        cells.extend(_scope_cells_for_display(row, col, extra))
    seen = set()
    count = 0
    for r, c in cells:
        if (r, c) == state.current_anchor or (r, c) in seen:
            continue
        seen.add((r, c))
        tile = state.tile(r, c)
        if tile.paint is None:
            continue
        if _paint_matches_target_for_display(tile.paint, rule.target_object):
            count += 1
    return count


def _scope_cells_for_display(row: int, col: int, direction: PaintDirection) -> list[tuple[int, int]]:
    if direction == PaintDirection.HORIZONTAL:
        return [(row, c) for c in range(6)]
    if direction == PaintDirection.VERTICAL:
        return [(r, col) for r in range(6)]
    if direction == PaintDirection.DIAGONAL:
        cells = {(row, col)}
        for delta in range(-6, 7):
            for r, c in ((row + delta, col + delta), (row + delta, col - delta)):
                if 0 <= r < 6 and 0 <= c < 6:
                    cells.add((r, c))
        return list(cells)
    if direction == PaintDirection.ADJACENT:
        return [(r, c) for r, c in ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)) if 0 <= r < 6 and 0 <= c < 6]
    if direction == PaintDirection.GLOBAL:
        return [(r, c) for r in range(6) for c in range(6)]
    return [(row, col)]


def _paint_matches_target_for_display(paint: Paint, target: PaintObject) -> bool:
    if target == PaintObject.MONSTER:
        return paint.category == PaintCategory.MONSTER
    if target == PaintObject.GAMEPLAY:
        return paint.category == PaintCategory.GAMEPLAY
    return True


def _optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _split_manual_list(value: str, *, separator: str) -> list[str]:
    normalized = value.replace("，", ",").replace("；", ";")
    if separator == ",":
        parts = normalized.split(",")
    else:
        parts = normalized.split(";")
    return [part.strip() for part in parts if part.strip()]


def _compact_real_capture_step(
    *,
    scene: SceneObservation,
    state: PuzzleState,
    backpack: tuple[Paint, ...],
    recommendations: list[PlacementRecommendation],
    brush_scenario: BrushChoiceScenario | None,
    resources: ResourceState,
    quality: dict | None = None,
) -> dict:
    rule = state.current_rule
    brush = scene.brush_options
    if brush_scenario:
        stage1 = [item.value for item in brush_scenario.stage1_options]
        chosen = brush_scenario.chosen_stage1.value
        stage2 = [affix.name for affix in brush_scenario.stage2_options]
    else:
        stage1 = list(brush.stage1_options) if brush else []
        chosen = brush.chosen_stage1 if brush else None
        stage2 = list(brush.stage2_options) if brush else []
    brush_payload = {
        "stage1": stage1,
        "chosen": chosen,
        "stage2": stage2,
    }
    brush_payload.update(brush_payload_codes(stage1, chosen, stage2))
    return {
        "type": "capture",
        "board": _compact_board_summary(scene),
        "current": _compact_rule(rule),
        "anchor": list(state.current_anchor) if state.current_anchor is not None else None,
        "pending": current_rule_is_pending(state),
        "backpack": _paint_collection_summary(backpack),
        "brush": brush_payload,
        "resources": {
            "force_used": resources.force_used,
            "force_remaining": resources.force_remaining,
            "refresh_available": resources.refresh_available,
            "refresh_remaining": resources.refresh_remaining,
        },
        "tiles": _compact_tile_paints(state),
        "quality": quality or {},
        "top": [_compact_recommendation(rec) for rec in recommendations[:3]],
    }


def _compact_board_summary(scene: SceneObservation) -> dict:
    state_counts = Counter(tile.state.value for tile in scene.board.tiles)
    states = "".join(_tile_state_code(tile.state) for tile in scene.board.tiles)
    return {
        "states": states,
        "blank": state_counts.get(TileState.BLANK.value, 0),
        "blocked": state_counts.get(TileState.BLOCKED.value, 0),
        "void": state_counts.get(TileState.VOID.value, 0),
        "faded": state_counts.get(TileState.FADED.value, 0),
        "paint": state_counts.get(TileState.PAINT.value, 0),
        "score": round(scene.board.score, 3),
    }


def _compact_tile_paints(state: PuzzleState) -> list[dict | None]:
    result: list[dict | None] = []
    for tile in state.tiles:
        result.append(_compact_paint(tile.paint) if tile.paint is not None else None)
    return result


def _tile_state_code(state: TileState) -> str:
    return {
        TileState.BLANK: "B",
        TileState.BLOCKED: "K",
        TileState.VOID: "V",
        TileState.FADED: "F",
        TileState.PAINT: "P",
        TileState.UNKNOWN: "?",
    }[state]


def _compact_rule(rule: CurrentPaintRule | None) -> dict | None:
    if rule is None:
        return None
    return compact_rule_with_code(rule)


def _compact_recommendation(rec: PlacementRecommendation) -> dict:
    payload = {
        "r": rec.row,
        "c": rec.col,
        "paint": _compact_paint(rec.paint) if rec.paint is not None else None,
        "score": round(rec.score, 1),
        "complete": rec.completes_match,
    }
    explanation = _compact_recommendation_explanation(rec)
    if explanation:
        payload["explanation"] = explanation
    return payload


def _compact_recommendation_explanation(rec: PlacementRecommendation) -> dict:
    explanation = getattr(rec, "explanation", None)
    if explanation is None:
        return {}
    payload: dict = {}
    if explanation.short:
        payload["short"] = explanation.short
    if explanation.tags:
        payload["tags"] = list(explanation.tags)
    layers = {}
    for name in ("immediate", "value", "terrain", "risk", "resource", "expected"):
        value = float(getattr(explanation, name, 0.0) or 0.0)
        if abs(value) >= 0.5:
            layers[name] = round(value, 1)
    if layers:
        payload["layers"] = layers
    return payload


def _recommendation_explanation_lines(rec: PlacementRecommendation) -> list[str]:
    explanation = getattr(rec, "explanation", None)
    if explanation is None:
        return []

    lines: list[str] = []
    if explanation.short:
        lines.append(f"解释：{explanation.short}")

    parts: list[str] = []
    for label, attr in (
        ("即时", "immediate"),
        ("价值", "value"),
        ("地形", "terrain"),
        ("风险", "risk"),
        ("资源", "resource"),
        ("期望", "expected"),
    ):
        value = float(getattr(explanation, attr, 0.0) or 0.0)
        if abs(value) >= 0.5:
            parts.append(f"{label}{value:+.0f}")
    if parts:
        lines.append("分层：" + " ".join(parts))

    tags = tuple(getattr(explanation, "tags", ()) or ())
    if tags:
        labels = [_recommendation_tag_label(tag) for tag in tags[:6]]
        lines.append("标签：" + "、".join(labels))
    return lines


def _recommendation_tag_label(tag: str) -> str:
    return {
        "anchor": "初始落点",
        "can_complete": "可完成",
        "complete_match": "完成匹配",
        "treasure": "宝箱",
        "placed_to_7": "本格到7",
        "treasure_to_7": "宝箱到7",
        "gameplay_to_7": "玩法到7",
        "monster_to_7": "怪物到7",
        "high_value_gain": "高价值",
        "overwrite_risk": "覆盖风险",
        "terrain_gain": "地形收益",
        "terrain_loss": "地形损失",
        "lookahead_gain": "后续更好",
        "lookahead_risk": "后续风险",
        "learned_prior": "历史偏好",
        "special_tile": "特殊地块",
        "scope_value_gain": "范围收益",
    }.get(tag, tag)


def _infer_real_capture_action(previous: dict | None, current: dict) -> dict:
    if previous is None:
        return {"type": "session_start", "confidence": 0.5}

    previous_board = previous.get("board") or {}
    current_board = current.get("board") or {}
    previous_current = previous.get("current")
    current_rule = current.get("current")
    previous_brush = previous.get("brush") or {}
    current_brush = current.get("brush") or {}
    previous_resources = previous.get("resources") or {}
    current_resources = current.get("resources") or {}
    previous_backpack = previous.get("backpack") or {}
    current_backpack = current.get("backpack") or {}

    board_delta = _state_count_delta(previous_board, current_board)
    changed_cells = _changed_capture_cells(previous, current)
    current_changed = previous_current != current_rule
    brush_changed = previous_brush != current_brush
    backpack_delta = int(current_backpack.get("total", 0)) - int(previous_backpack.get("total", 0))
    force_delta = int(previous_resources.get("force_remaining", 0)) - int(current_resources.get("force_remaining", 0))
    refresh_delta = int(previous_resources.get("refresh_remaining", 0)) - int(current_resources.get("refresh_remaining", 0))

    if force_delta > 0 or board_delta.get("faded", 0) > 0:
        return _action_payload(
            "force_complete",
            confidence=0.86,
            code="F",
            force_used_delta=force_delta,
            board_delta=board_delta,
            changed_cells=changed_cells,
            detail=_action_detail("force_complete", changed_cells, previous, current),
            summary=_action_summary("force_complete", changed_cells, previous, current),
        )
    if refresh_delta > 0:
        stage = "stage2" if previous_brush.get("chosen") == current_brush.get("chosen") else "stage1"
        return _action_payload(
            "refresh_affix",
            confidence=0.82,
            code="R2" if stage == "stage2" else "R1",
            stage=stage,
            refresh_used_delta=refresh_delta,
            detail=_action_detail("refresh_affix", changed_cells, previous, current),
            summary=_action_summary("refresh_affix", changed_cells, previous, current),
        )
    if current_changed and current_rule is not None:
        return _action_payload(
            "new_current_paint",
            confidence=0.78,
            code="N",
            previous=previous_current,
            current=current_rule,
            changed_cells=changed_cells,
            detail=_action_detail("new_current_paint", changed_cells, previous, current),
            summary=_action_summary("new_current_paint", changed_cells, previous, current),
        )
    if brush_changed and current_brush.get("stage2"):
        return _action_payload(
            "brush_options_changed",
            confidence=0.70,
            code="B",
            brush=current_brush,
            detail=_action_detail("brush_options_changed", changed_cells, previous, current),
            summary=_action_summary("brush_options_changed", changed_cells, previous, current),
        )
    if board_delta.get("paint", 0) > 0 or backpack_delta < 0:
        action_type = "place_current_paint" if previous.get("pending") else "install_paint"
        return _action_payload(
            action_type,
            confidence=0.76 if action_type == "place_current_paint" else 0.74,
            code="C" if action_type == "place_current_paint" else "P",
            backpack_delta=backpack_delta,
            board_delta=board_delta,
            changed_cells=changed_cells,
            detail=_action_detail(action_type, changed_cells, previous, current),
            summary=_action_summary(action_type, changed_cells, previous, current),
        )
    if board_delta.get("blank", 0) > 0 or board_delta.get("blocked", 0) < 0 or board_delta.get("void", 0) < 0:
        return _action_payload(
            "terrain_changed",
            confidence=0.68,
            code="T",
            board_delta=board_delta,
            changed_cells=changed_cells,
            detail=_action_detail("terrain_changed", changed_cells, previous, current),
            summary=_action_summary("terrain_changed", changed_cells, previous, current),
        )
    return _action_payload(
        "capture_update",
        confidence=0.45,
        code="U",
        backpack_delta=backpack_delta,
        board_delta=board_delta,
        changed_cells=changed_cells,
        detail=_action_detail("capture_update", changed_cells, previous, current),
        summary=_action_summary("capture_update", changed_cells, previous, current),
    )


def _action_payload(action_type: str, *, confidence: float, code: str, **fields) -> dict:
    payload = {
        "type": action_type,
        "code": code,
        "confidence": confidence,
    }
    payload.update(fields)
    return payload


def _action_detail(action_type: str, changed_cells: list[dict], previous: dict, current: dict) -> dict:
    summary = _action_summary(action_type, changed_cells, previous, current)
    current_rule = current.get("current") if isinstance(current.get("current"), dict) else None
    previous_rule = previous.get("current") if isinstance(previous.get("current"), dict) else None
    detail = {
        "code": _action_code_from_type(action_type),
        "placed": summary.get("placed_cells", []),
        "upgraded": summary.get("upgraded_cells", []),
        "faded": summary.get("faded_cells", []),
        "terrain": summary.get("terrain_cells", []),
        "match": summary.get("current_match"),
        "current_code": current_rule.get("code") if current_rule else None,
        "previous_code": previous_rule.get("code") if previous_rule else None,
    }
    if action_type in {"install_paint", "place_current_paint"}:
        detail["placed_paints"] = summary.get("placed_paints", [])
    if action_type == "refresh_affix":
        previous_brush = previous.get("brush") or {}
        current_brush = current.get("brush") or {}
        detail["from_stage2"] = previous_brush.get("stage2_codes") or previous_brush.get("stage2") or []
        detail["to_stage2"] = current_brush.get("stage2_codes") or current_brush.get("stage2") or []
    return detail


def _action_code_from_type(action_type: str) -> str:
    return {
        "session_start": "S",
        "force_complete": "F",
        "refresh_affix": "R",
        "new_current_paint": "N",
        "brush_options_changed": "B",
        "place_current_paint": "C",
        "install_paint": "P",
        "terrain_changed": "T",
        "capture_update": "U",
    }.get(action_type, "?")


def _action_summary(action_type: str, changed_cells: list[dict], previous: dict, current: dict) -> dict:
    placed = [cell for cell in changed_cells if cell.get("to_state") == "P" and cell.get("from_state") != "P"]
    upgraded = [
        cell
        for cell in changed_cells
        if isinstance(cell.get("from_paint"), dict)
        and isinstance(cell.get("to_paint"), dict)
        and (cell["to_paint"].get("l") or 0) > (cell["from_paint"].get("l") or 0)
    ]
    faded = [cell for cell in changed_cells if cell.get("to_state") == "F"]
    terrain = [cell for cell in changed_cells if cell.get("from_state") != cell.get("to_state") and cell.get("to_state") != "P"]
    return {
        "action": action_type,
        "placed_cells": _cell_refs(placed),
        "upgraded_cells": _cell_refs(upgraded),
        "faded_cells": _cell_refs(faded),
        "terrain_cells": _cell_refs(terrain),
        "placed_paints": [cell.get("to_paint") for cell in placed if cell.get("to_paint") is not None],
        "upgrade_count": len(upgraded),
        "changed_count": len(changed_cells),
        "current_match": ((current.get("current") or {}).get("match") if isinstance(current.get("current"), dict) else None),
    }


def _cell_refs(cells: list[dict]) -> list[list[int]]:
    refs: list[list[int]] = []
    for cell in cells[:12]:
        row = cell.get("r")
        col = cell.get("c")
        if isinstance(row, int) and isinstance(col, int):
            refs.append([row, col])
    return refs


def _state_count_delta(previous_board: dict, current_board: dict) -> dict:
    keys = ("blank", "blocked", "void", "faded", "paint")
    return {
        key: int(current_board.get(key, 0)) - int(previous_board.get(key, 0))
        for key in keys
    }


def _is_noop_capture_update(inferred_action: dict) -> bool:
    if inferred_action.get("type") != "capture_update":
        return False
    if inferred_action.get("changed_cells"):
        return False
    summary = inferred_action.get("summary") or {}
    return int(summary.get("changed_count", 0) or 0) == 0


def _is_tentative_place_current_action(action: dict, step: dict) -> bool:
    if action.get("type") != "place_current_paint":
        return False
    if step.get("ocr") is not False:
        return False
    summary = action.get("summary") or {}
    placed_cells = summary.get("placed_cells") or []
    if len(placed_cells) != 1:
        return False
    if (step.get("quality") or {}).get("mode") != "quick_capture":
        return False
    return True


def _backpack_visible_in_step(step: dict) -> bool:
    backpack = step.get("backpack") or {}
    return int(backpack.get("total", 0) or 0) > 0


def _find_new_anchor_from_scene(scene: SceneObservation, previous_state: PuzzleState) -> tuple[int, int] | None:
    candidates: list[tuple[float, int, int]] = []
    recommendations = recommend_placements(previous_state, limit=36, include_lookahead=False)
    recommended_cells = {(rec.row, rec.col): rec.score for rec in recommendations}
    for tile in scene.board.tiles:
        row, col = tile.row, tile.col
        before = previous_state.tile(row, col)
        if tile.state != TileState.PAINT or before.state == TileState.PAINT:
            continue
        score = float(tile.confidence)
        if (row, col) in recommended_cells:
            score += 2.0 + recommended_cells[(row, col)] / 1000.0
        if before.state == TileState.BLANK:
            score += 0.6
        candidates.append((score, row, col))
    if not candidates:
        return None
    _score, row, col = max(candidates, key=lambda item: item[0])
    return row, col


def _tentative_place_signature(step: dict) -> str | None:
    action = step.get("inferred_action") or {}
    summary = action.get("summary") or {}
    placed_cells = summary.get("placed_cells") or []
    if not placed_cells:
        return None
    return _stable_payload_key(
        {
            "session": step.get("session"),
            "current": step.get("current"),
            "anchor": step.get("anchor"),
            "placed": placed_cells[:1],
        }
    )


def _changed_capture_cells(previous: dict, current: dict) -> list[dict]:
    previous_states = str((previous.get("board") or {}).get("states") or "")
    current_states = str((current.get("board") or {}).get("states") or "")
    previous_tiles = previous.get("tiles") or []
    current_tiles = current.get("tiles") or []
    changed: list[dict] = []
    for index in range(36):
        prev_state = previous_states[index] if index < len(previous_states) else "?"
        curr_state = current_states[index] if index < len(current_states) else "?"
        prev_paint = previous_tiles[index] if index < len(previous_tiles) else None
        curr_paint = current_tiles[index] if index < len(current_tiles) else None
        if prev_state == curr_state and prev_paint == curr_paint:
            continue
        changed.append(
            {
                "r": index // 6,
                "c": index % 6,
                "from_state": prev_state,
                "to_state": curr_state,
                "from_paint": prev_paint,
                "to_paint": curr_paint,
            }
        )
    return changed[:12]


def _trajectory_terminal_summary(state: PuzzleState, backpack: tuple[Paint, ...]) -> dict:
    board_paints = tuple(tile.paint for tile in state.tiles if tile.paint is not None)
    known_paints = tuple(paint for paint in (*board_paints, *backpack) if paint is not None)
    if not known_paints and state.current_rule is not None:
        known_paints = (state.current_rule.paint,)
    known = _paint_collection_summary(known_paints)
    return {
        "known": known,
        "board_known": _paint_collection_summary(board_paints),
        "backpack": _paint_collection_summary(backpack),
        "quality": _quality_proxy(known),
    }


def _quality_is_trusted_for_trajectory(quality: dict | None) -> bool:
    return isinstance(quality, dict) and quality.get("ok") is True


def _trusted_trajectory_steps(steps: list[dict]) -> list[dict]:
    return [step for step in steps if _quality_is_trusted_for_trajectory(step.get("quality"))]


def _empty_terminal_summary() -> dict:
    empty = {
        "total": 0,
        "levels": {},
        "lv7": 0,
        "gameplay7": 0,
        "monster7": 0,
        "treasure": 0,
        "treasure5": 0,
        "treasure6": 0,
        "treasure7": 0,
        "max_treasure": 0,
    }
    return {
        "known": dict(empty),
        "board_known": dict(empty),
        "backpack": dict(empty),
        "quality": 0.0,
    }


def _best_trusted_terminal_summary(
    state: PuzzleState,
    steps: list[dict],
    *,
    current_quality: dict | None = None,
) -> dict:
    reliable_steps = _trusted_trajectory_steps(steps)
    if _quality_is_trusted_for_trajectory(current_quality):
        best = _trajectory_terminal_summary(state, state.backpack)
        best_score = _internal_run_score(best)
    elif reliable_steps:
        best = _terminal_summary_from_step(reliable_steps[0])
        best_score = _internal_run_score(best)
    else:
        return _empty_terminal_summary()

    for step in reliable_steps:
        step_terminal = _terminal_summary_from_step(step)
        step_score = _internal_run_score(step_terminal)
        if step_score > best_score:
            best = step_terminal
            best_score = step_score
    return best


def _best_available_terminal_summary(state: PuzzleState, steps: list[dict]) -> dict:
    terminal = _trajectory_terminal_summary(state, state.backpack)
    best = terminal
    best_score = _internal_run_score(terminal)
    for step in steps:
        quality = step.get("quality") or {}
        if quality.get("ok") is not True:
            continue
        step_terminal = _terminal_summary_from_step(step)
        step_score = _internal_run_score(step_terminal)
        if step_score > best_score:
            best = step_terminal
            best_score = step_score
    return best


def _terminal_summary_from_step(step: dict) -> dict:
    backpack = step.get("backpack") or {}
    board_known = _paint_collection_summary_from_compact(step.get("tiles") or [])
    known = _merge_paint_summaries(board_known, backpack)
    return {
        "known": known,
        "board_known": board_known,
        "backpack": backpack,
        "quality": _quality_proxy(known),
    }


def _paint_collection_summary_from_compact(items: list) -> dict:
    paints = [item for item in items if isinstance(item, dict)]
    total = len(paints)
    levels = Counter(str(item.get("l")) for item in paints if item.get("l") is not None)
    treasure_levels = [int(item.get("l") or 0) for item in paints if item.get("t")]
    gameplay7 = sum(1 for item in paints if item.get("c") == PaintCategory.GAMEPLAY.value and not item.get("t") and int(item.get("l") or 0) >= 7)
    monster7 = sum(1 for item in paints if item.get("c") == PaintCategory.MONSTER.value and int(item.get("l") or 0) >= 7)
    return {
        "total": total,
        "levels": dict(levels),
        "lv7": sum(1 for item in paints if int(item.get("l") or 0) >= 7),
        "gameplay7": gameplay7,
        "monster7": monster7,
        "treasure": sum(1 for item in paints if item.get("t")),
        "treasure5": sum(1 for level in treasure_levels if level >= 5),
        "treasure6": sum(1 for level in treasure_levels if level >= 6),
        "treasure7": sum(1 for level in treasure_levels if level >= 7),
        "max_treasure": max(treasure_levels) if treasure_levels else 0,
    }


def _merge_paint_summaries(*summaries: dict) -> dict:
    levels = Counter()
    merged = {
        "total": 0,
        "lv7": 0,
        "gameplay7": 0,
        "monster7": 0,
        "treasure": 0,
        "treasure5": 0,
        "treasure6": 0,
        "treasure7": 0,
        "max_treasure": 0,
    }
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        merged["total"] += int(summary.get("total", 0) or 0)
        merged["lv7"] += int(summary.get("lv7", 0) or 0)
        merged["gameplay7"] += int(summary.get("gameplay7", 0) or 0)
        merged["monster7"] += int(summary.get("monster7", 0) or 0)
        merged["treasure"] += int(summary.get("treasure", 0) or 0)
        merged["treasure5"] += int(summary.get("treasure5", 0) or 0)
        merged["treasure6"] += int(summary.get("treasure6", 0) or 0)
        merged["treasure7"] += int(summary.get("treasure7", 0) or 0)
        merged["max_treasure"] = max(merged["max_treasure"], int(summary.get("max_treasure", 0) or 0))
        for level, count in (summary.get("levels") or {}).items():
            levels[str(level)] += int(count or 0)
    merged["levels"] = dict(levels)
    return merged


def _internal_run_score(terminal: dict) -> float:
    return round(_internal_score_breakdown(terminal)["total"], 3)


def _internal_score_breakdown(terminal: dict) -> dict:
    known = terminal.get("known") or {}
    main = _quality_proxy(known)

    # Keep the main score on the explicit treasure equivalence scale:
    # Lv7 box=5000, Lv6 box=3000, Lv5 box=1000, gameplay Lv7=1000,
    # monster Lv7≈714. Low-level pieces are only tie-breakers.
    treasure7 = int(known.get("treasure7", 0) or 0)
    treasure6_only = max(0, int(known.get("treasure6", 0) or 0) - treasure7)
    treasure5_only = max(0, int(known.get("treasure5", 0) or 0) - int(known.get("treasure6", 0) or 0))
    treasure_tie = treasure7 * 10.0 + treasure6_only * 6.0 + treasure5_only * 3.0
    lv7_tie = max(0, int(known.get("lv7", 0) or 0) - treasure7) * 0.05
    low_level_tie = 0.0

    levels = known.get("levels") or {}
    for level_text, count in levels.items():
        try:
            level = int(level_text)
        except (TypeError, ValueError):
            continue
        if level <= 4:
            low_level_tie += count * level * 0.01
        elif level == 5:
            low_level_tie += count * 0.08
        elif level == 6:
            low_level_tie += count * 0.16
    total = main + treasure_tie + lv7_tie + low_level_tie
    return {
        "main": round(main, 3),
        "treasure_tie": round(treasure_tie, 3),
        "lv7_tie": round(lv7_tie, 3),
        "low_level_tie": round(low_level_tie, 3),
        "total": round(total, 3),
        "model": "huahua-v2-treasure-equivalence",
    }


def _looks_like_new_run(scene: SceneObservation, previous_step: dict | None) -> bool:
    if previous_step is None:
        return True
    board = _compact_board_summary(scene)
    previous_board = previous_step.get("board") or {}
    if board.get("states") == previous_board.get("states"):
        return False
    open_cells = int(board.get("blank", 0)) + int(board.get("void", 0))
    if open_cells <= 0:
        return False
    paint_count = int(board.get("paint", 0))
    blocked_count = int(board.get("blocked", 0))
    # Opening layouts usually have plenty of open cells and only a small number
    # of placed paints. Keep this permissive so it can recover after a finished
    # run without user clicks, but avoid restarting during a full/terminal board.
    return open_cells >= 8 and paint_count <= 8 and blocked_count <= 18


def _paint_collection_summary(paints: tuple[Paint, ...] | list[Paint]) -> dict:
    total = len(paints)
    level_counts = Counter(str(paint.level) for paint in paints)
    treasure_levels = [paint.level for paint in paints if paint.is_treasure]
    gameplay7 = sum(1 for paint in paints if paint.category == PaintCategory.GAMEPLAY and not paint.is_treasure and paint.level >= 7)
    monster7 = sum(1 for paint in paints if paint.category == PaintCategory.MONSTER and paint.level >= 7)
    treasure5 = sum(1 for level in treasure_levels if level >= 5)
    treasure6 = sum(1 for level in treasure_levels if level >= 6)
    treasure7 = sum(1 for level in treasure_levels if level >= 7)
    return {
        "total": total,
        "levels": dict(level_counts),
        "lv7": sum(1 for paint in paints if paint.level >= 7),
        "gameplay7": gameplay7,
        "monster7": monster7,
        "treasure": sum(1 for paint in paints if paint.is_treasure),
        "treasure5": treasure5,
        "treasure6": treasure6,
        "treasure7": treasure7,
        "max_treasure": max(treasure_levels) if treasure_levels else 0,
    }


def _capture_quality_gate(scene: SceneObservation, backpack: tuple[Paint, ...]) -> dict:
    min_board = float(config_value("quality.min_board", 0.48))
    min_current = float(config_value("quality.min_current", 0.45))
    min_backpack = float(config_value("quality.min_backpack", 0.35))
    min_brush = float(config_value("quality.min_brush", 0.35))
    min_overall = float(config_value("quality.min_overall", 0.58))
    max_unknown_tiles = int(config_value("quality.max_unknown_tiles", 1))
    board_score = float(scene.board.score)
    current_score = _current_paint_quality(scene.current_paint)
    backpack_score = float(scene.backpack.confidence) if scene.backpack is not None else (0.72 if backpack else 0.30)
    brush_score = float(scene.brush_options.confidence) if scene.brush_options is not None else 0.72
    tile_scores = [tile.confidence for tile in scene.board.tiles if tile.state != TileState.UNKNOWN]
    tile_score = sum(tile_scores) / len(tile_scores) if tile_scores else 0.0
    unknown_tiles = sum(1 for tile in scene.board.tiles if tile.state == TileState.UNKNOWN)
    score = (
        board_score * 0.34
        + current_score * 0.26
        + backpack_score * 0.20
        + brush_score * 0.10
        + tile_score * 0.10
    )
    reasons: list[str] = []
    if board_score < min_board:
        reasons.append(f"board={board_score:.2f}")
    if current_score < min_current:
        reasons.append(f"current={current_score:.2f}")
    if backpack_score < min_backpack:
        reasons.append(f"backpack={backpack_score:.2f}")
    if brush_score < min_brush:
        reasons.append(f"brush={brush_score:.2f}")
    if unknown_tiles > max_unknown_tiles:
        reasons.append(f"unknown_tiles={unknown_tiles}")
    if score < min_overall:
        reasons.append(f"overall={score:.2f}")
    ok = not reasons
    return {
        "ok": ok,
        "score": round(score, 3),
        "board": round(board_score, 3),
        "current": round(current_score, 3),
        "backpack": round(backpack_score, 3),
        "brush": round(brush_score, 3),
        "tile": round(tile_score, 3),
        "unknown_tiles": unknown_tiles,
        "reason": "ok" if ok else ", ".join(reasons),
    }


def _quick_capture_quality(
    scene: SceneObservation,
    elapsed: float,
    capture_perf: dict | None = None,
    *,
    capture_source: str | None = None,
) -> dict:
    board_score = round(float(scene.board.score), 3)
    unknown_tiles = sum(1 for tile in scene.board.tiles if tile.state == TileState.UNKNOWN)
    min_board = float(config_value("quality.quick_min_board", 0.66))
    max_unknown_tiles = int(config_value("quality.quick_max_unknown_tiles", 1))
    reasons: list[str] = []
    source = capture_source or "unknown"
    if _capture_source_can_include_overlays(source):
        reasons.append("capture_source_composited")
    if board_score < min_board:
        reasons.append(f"board={board_score:.2f}")
    if unknown_tiles > max_unknown_tiles:
        reasons.append(f"unknown_tiles={unknown_tiles}")
    reason = "quick_board_reuse"
    if not reasons and unknown_tiles:
        reason = f"quick_board_reuse_with_{unknown_tiles}_unknown"
    return {
        "ok": not reasons,
        "score": board_score,
        "reason": reason if not reasons else ", ".join(reasons),
        "mode": "quick_capture",
        "ocr": False,
        "elapsed": round(elapsed, 3),
        "unknown_tiles": unknown_tiles,
        "perf": capture_perf or {},
        "capture_source": source,
    }


def _capture_source_can_include_overlays(source: str) -> bool:
    normalized = str(source or "").casefold()
    return normalized.startswith("screen:fallback") or normalized.startswith("window-screen")


def _current_paint_quality(paint) -> float:
    if paint is None:
        return 0.0
    parts = [
        float(paint.category_confidence),
        float(paint.level_confidence),
        max(paint.keyword_scores.values(), default=0.0),
    ]
    if paint.match_required is None:
        parts.append(0.35)
    else:
        parts.append(0.75)
    if paint.upgrade_amount is None:
        parts.append(0.35)
    else:
        parts.append(0.75)
    return max(0.0, min(1.0, sum(parts) / len(parts)))


def _detect_capture_rule_anomalies(
    previous: dict | None,
    scene: SceneObservation,
    state: PuzzleState,
) -> list[dict]:
    anomalies: list[dict] = []
    current = scene.current_paint
    if current is not None and current.level is not None:
        expected_match, expected_upgrade = _empirical_current_paint_defaults(current.level)
        if current.match_required is not None and current.match_required != expected_match:
            anomalies.append(
                {
                    "kind": "current_match_required",
                    "severity": "warn",
                    "level": current.level,
                    "observed": current.match_required,
                    "expected": expected_match,
                    "hint": current.raw_text_hint,
                }
            )
        if current.upgrade_amount is not None and not _empirical_upgrade_is_allowed(current.level, current.upgrade_amount):
            anomalies.append(
                {
                    "kind": "current_upgrade_amount",
                    "severity": "warn",
                    "level": current.level,
                    "observed": current.upgrade_amount,
                    "expected": expected_upgrade,
                    "allowed": _empirical_allowed_upgrades(current.level),
                    "hint": current.raw_text_hint,
                }
            )
    if previous is not None:
        current_step = {
            "board": _compact_board_summary(scene),
            "tiles": _compact_tile_paints(state),
            "current": _compact_rule(state.current_rule),
        }
        changed_cells = _changed_capture_cells(previous, current_step)
        unexpected_levels = _unexpected_level_changes(previous, current_step, changed_cells)
        if unexpected_levels:
            anomalies.append(
                {
                    "kind": "unexpected_level_change",
                    "severity": "block",
                    "cells": unexpected_levels[:8],
                    "note": "相邻截图出现无法用当前规则直接解释的等级变化，可能是漏识别词缀、强制完成后状态变化或视觉误判",
                }
            )
        unexpected_transforms = _unexpected_paint_transforms(previous, current_step, changed_cells)
        if unexpected_transforms:
            anomalies.append(
                {
                    "kind": "unexpected_paint_transform",
                    "severity": "warn",
                    "cells": unexpected_transforms[:8],
                    "note": "颜料类型/宝箱属性发生变化；若非转锋/堆砌变化词条，可能是识别错误或漏规则",
                }
            )
        terrain_changes = _unexpected_terrain_changes(previous, current_step, changed_cells)
        if terrain_changes:
            anomalies.append(
                {
                    "kind": "unexpected_terrain_change",
                    "severity": "warn",
                    "cells": terrain_changes[:8],
                    "note": "地形发生变化；若非扩散/强制完成，应复查识别或补充词条规则",
                }
            )
        faded_cells = _forced_faded_cells(previous, current_step, changed_cells)
        if faded_cells:
            anomalies.append(
                {
                    "kind": "force_faded_seen",
                    "severity": "info",
                    "cells": faded_cells[:8],
                    "note": "检测到颜料变失色，通常来自强制完成破坏",
                }
            )
    return anomalies


def _unexpected_level_changes(previous: dict, current: dict, changed_cells: list[dict]) -> list[dict]:
    previous_rule = previous.get("current") or {}
    current_rule = current.get("current") or {}
    if previous_rule != current_rule:
        return []
    rule_paint = (current_rule.get("paint") or {})
    upgrade = int(current_rule.get("upgrade") or 0)
    if upgrade <= 0:
        return []
    target = current_rule.get("target")
    unexpected: list[dict] = []
    for cell in changed_cells:
        before = cell.get("from_paint")
        after = cell.get("to_paint")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        before_level = before.get("l")
        after_level = after.get("l")
        if not isinstance(before_level, int) or not isinstance(after_level, int):
            continue
        if after_level <= before_level:
            continue
        if _compact_paint_matches_target(after, target):
            expected_max = min(7, before_level + upgrade)
            if after_level <= expected_max:
                continue
        if before == rule_paint or after == rule_paint:
            continue
        unexpected.append(
            {
                "r": cell.get("r"),
                "c": cell.get("c"),
                "from": before,
                "to": after,
                "upgrade": upgrade,
                "target": target,
            }
        )
    return unexpected


def _unexpected_paint_transforms(previous: dict, current: dict, changed_cells: list[dict]) -> list[dict]:
    previous_rule = previous.get("current") or {}
    current_rule = current.get("current") or {}
    if previous_rule != current_rule:
        return []
    unexpected: list[dict] = []
    for cell in changed_cells:
        before = cell.get("from_paint")
        after = cell.get("to_paint")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        before_kind = (before.get("c"), before.get("t"), before.get("s"))
        after_kind = (after.get("c"), after.get("t"), after.get("s"))
        if before_kind == after_kind:
            continue
        # A pure level change is handled by _unexpected_level_changes.
        if before.get("code") != after.get("code") or before_kind != after_kind:
            unexpected.append(
                {
                    "r": cell.get("r"),
                    "c": cell.get("c"),
                    "from": before,
                    "to": after,
                }
            )
    return unexpected


def _unexpected_terrain_changes(previous: dict, current: dict, changed_cells: list[dict]) -> list[dict]:
    previous_rule = previous.get("current") or {}
    current_rule = current.get("current") or {}
    if previous_rule != current_rule:
        return []
    unexpected: list[dict] = []
    for cell in changed_cells:
        from_state = cell.get("from_state")
        to_state = cell.get("to_state")
        if from_state == to_state:
            continue
        if to_state == "P" or from_state == "P":
            continue
        if to_state == "F":
            continue
        if from_state == "?" or to_state == "?":
            continue
        unexpected.append(
            {
                "r": cell.get("r"),
                "c": cell.get("c"),
                "from_state": from_state,
                "to_state": to_state,
            }
        )
    return unexpected


def _forced_faded_cells(previous: dict, current: dict, changed_cells: list[dict]) -> list[dict]:
    result: list[dict] = []
    for cell in changed_cells:
        if cell.get("to_state") != "F":
            continue
        result.append(
            {
                "r": cell.get("r"),
                "c": cell.get("c"),
                "from_state": cell.get("from_state"),
                "from_paint": cell.get("from_paint"),
            }
        )
    return result


def _compact_paint_matches_target(paint: dict, target: str | None) -> bool:
    if target == PaintObject.MONSTER.value:
        return paint.get("c") == PaintCategory.MONSTER.value
    if target == PaintObject.GAMEPLAY.value:
        return paint.get("c") == PaintCategory.GAMEPLAY.value
    if target == PaintObject.ALL.value:
        return True
    return False


def _format_rule_anomaly(anomaly: dict) -> str:
    kind = anomaly.get("kind")
    severity = anomaly.get("severity", "warn")
    if kind in {"current_match_required", "current_upgrade_amount"}:
        label = "进度要求" if kind == "current_match_required" else "升级数"
        return (
            f"[{severity}] {label}异常：Lv{anomaly.get('level')} 识别={anomaly.get('observed')} "
            f"经验={anomaly.get('expected')}"
        )
    if kind == "unexpected_level_change":
        return f"[{severity}] 等级变化异常：{len(anomaly.get('cells') or [])}格变化暂不能用本地规则解释"
    if kind == "unexpected_paint_transform":
        return f"[{severity}] 颜料变化异常：{len(anomaly.get('cells') or [])}格类型/宝箱属性变化"
    if kind == "unexpected_terrain_change":
        return f"[{severity}] 地形变化提醒：{len(anomaly.get('cells') or [])}格地形变化"
    if kind == "force_faded_seen":
        return f"[{severity}] 强制破坏痕迹：{len(anomaly.get('cells') or [])}格变失色"
    return str(anomaly)


def _compact_paint(paint: Paint) -> dict:
    return compact_paint_with_code(paint)


def _quality_proxy(summary: dict) -> float:
    return (
        summary.get("treasure7", 0) * float(config_value("values.treasure7", 5000.0))
        + max(0, summary.get("treasure6", 0) - summary.get("treasure7", 0)) * float(config_value("values.treasure6", 3000.0))
        + max(0, summary.get("treasure5", 0) - summary.get("treasure6", 0)) * float(config_value("values.treasure5", 1000.0))
        + summary.get("gameplay7", 0) * float(config_value("values.gameplay7", 1000.0))
        + summary.get("monster7", 0) * float(config_value("values.monster7", 714.0))
    )


def _should_save_auto_trajectory(terminal: dict) -> tuple[bool, str]:
    known = terminal.get("known") or {}
    score = _internal_run_score(terminal)
    if score >= float(config_value("autosave.min_internal_score", 3000.0)):
        return True, f"score_{score:.0f}"
    if known.get("treasure7", 0) >= int(config_value("autosave.min_treasure7", 1)):
        return True, "treasure7"
    if known.get("treasure6", 0) >= int(config_value("autosave.min_treasure6", 1)):
        return True, "treasure6_plus"
    if known.get("lv7", 0) >= int(config_value("autosave.min_lv7", 3)):
        return True, "many_level7"
    if (
        known.get("gameplay7", 0) >= int(config_value("autosave.min_gameplay7_mixed", 2))
        and known.get("monster7", 0) >= int(config_value("autosave.min_monster7_mixed", 1))
    ):
        return True, "mixed_level7"
    if (
        known.get("treasure5", 0) >= int(config_value("autosave.min_treasure5_with_lv7", 1))
        and known.get("lv7", 0) >= int(config_value("autosave.min_lv7_with_treasure5", 2))
    ):
        return True, "treasure5_with_level7"
    return False, "low_value"


def _summary_has_high_value(summary: dict) -> bool:
    return (
        int(summary.get("treasure5", 0) or 0) > 0
        or int(summary.get("gameplay7", 0) or 0) > 0
        or int(summary.get("monster7", 0) or 0) > 0
        or int(summary.get("lv7", 0) or 0) >= 2
    )


def _format_high_value_summary(summary: dict) -> str:
    parts: list[str] = []
    treasure7 = int(summary.get("treasure7", 0) or 0)
    treasure6 = int(summary.get("treasure6", 0) or 0)
    treasure5 = int(summary.get("treasure5", 0) or 0)
    gameplay7 = int(summary.get("gameplay7", 0) or 0)
    monster7 = int(summary.get("monster7", 0) or 0)
    if treasure7:
        parts.append(f"7级宝箱x{treasure7}")
    if treasure6:
        parts.append(f"6级以上宝箱x{treasure6}")
    elif treasure5:
        parts.append(f"5级以上宝箱x{treasure5}")
    if gameplay7:
        parts.append(f"7级玩法x{gameplay7}")
    if monster7:
        parts.append(f"7级怪物x{monster7}")
    if not parts:
        parts.append(f"7级颜料x{int(summary.get('lv7', 0) or 0)}")
    return "，".join(parts)


def _format_finish_diagnostic(diagnostic: dict | None) -> str:
    if not diagnostic:
        return "本局结束：暂无诊断信息。"
    terminal = diagnostic.get("terminal") or {}
    known = terminal.get("known") or {}
    board_known = terminal.get("board_known") or {}
    backpack = terminal.get("backpack") or {}
    quality = diagnostic.get("last_quality") or {}
    saved = bool(diagnostic.get("saved"))
    score = float(diagnostic.get("score", 0.0) or 0.0)
    reason = diagnostic.get("reason") or "-"
    finish_text = "有" if diagnostic.get("finish_seen") else "无"
    quality_text = _format_finish_quality(quality)
    high_value = _format_high_value_summary(known) if _summary_has_high_value(known) else "无高价值终局"
    board_total = int(board_known.get("total", 0) or 0)
    backpack_total = int(backpack.get("total", 0) or 0)
    result = "已保存结算轨迹" if saved else "未保存结算轨迹"
    return (
        f"结束诊断：{result}；原因：{reason}；内部评分={score:.0f}；"
        f"可靠步骤={int(diagnostic.get('reliable_steps', 0) or 0)}/{int(diagnostic.get('steps', 0) or 0)}；"
        f"结算证据={finish_text}；最后识别={quality_text}；"
        f"终局价值={high_value}；棋盘已识别颜料={board_total}，背包={backpack_total}。"
    )


def _format_finish_quality(quality: dict) -> str:
    if not quality:
        return "无"
    ok_text = "合格" if quality.get("ok") else "低置信"
    score = quality.get("score")
    score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "-"
    reason = quality.get("reason") or "-"
    return f"{ok_text}({score_text}, {reason})"


def _capture_image_size(image) -> tuple[int, int] | None:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _capture_shape_size(shape: tuple[int, ...] | None) -> tuple[int, int] | None:
    if shape is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _valid_capture_size(size: tuple[int, int] | None) -> bool:
    if size is None:
        return False
    width, height = size
    return width > 0 and height > 0


def _fast_capture_perf(
    capture_elapsed: float,
    analysis_elapsed: float,
    write_elapsed: float = 0.0,
    full_refresh_elapsed: float = 0.0,
) -> dict:
    return {
        "capture": round(capture_elapsed, 3),
        "analysis": round(analysis_elapsed, 3),
        "write": round(write_elapsed, 3),
        "full_refresh": round(full_refresh_elapsed, 3),
    }


def _format_fast_capture_perf(perf: dict | None) -> str:
    if not perf:
        return ""
    parts = [
        f"截图={float(perf.get('capture', 0.0) or 0.0):.2f}s",
        f"分析={float(perf.get('analysis', 0.0) or 0.0):.2f}s",
    ]
    write_elapsed = float(perf.get("write", 0.0) or 0.0)
    full_refresh = float(perf.get("full_refresh", 0.0) or 0.0)
    if write_elapsed > 0.01:
        parts.append(f"写图={write_elapsed:.2f}s")
    if full_refresh > 0.01:
        parts.append(f"完整补读={full_refresh:.2f}s")
    return "耗时拆分：" + "，".join(parts) + "。"


def _format_perf_panel(perf: dict) -> str:
    if not perf:
        return "性能：等待识别"
    mode = str(perf.get("mode") or "-")
    elapsed = _format_seconds(perf.get("elapsed"))
    capture_perf = perf.get("capture_perf") if isinstance(perf.get("capture_perf"), dict) else {}
    capture = _format_seconds(capture_perf.get("capture"))
    analysis = _format_seconds(capture_perf.get("analysis"))
    recommendation = _format_seconds(perf.get("recommendation"))
    pet = _format_seconds(perf.get("pet_render"))
    quality = "通过" if perf.get("quality_ok") is True else "暂缓" if perf.get("quality_ok") is False else "-"
    pet_message = str(perf.get("pet_message") or "-")
    source = _short_source(perf.get("source"))
    return (
        f"性能：{mode} 总={elapsed} 截图={capture} 分析={analysis} "
        f"推荐={recommendation} 桌宠={pet} 质量={quality} 桌宠状态={pet_message} 来源={source}"
    )


def _format_perf_details(perf: dict) -> list[str]:
    if not perf:
        return ["等待识别"]
    capture_perf = perf.get("capture_perf") if isinstance(perf.get("capture_perf"), dict) else {}
    lines = [
        f"模式={perf.get('mode', '-')}",
        f"总耗时={_format_seconds(perf.get('elapsed'))}",
        f"截图={_format_seconds(capture_perf.get('capture'))} 分析={_format_seconds(capture_perf.get('analysis'))}",
        f"写图={_format_seconds(capture_perf.get('write'))} 完整补读={_format_seconds(capture_perf.get('full_refresh'))}",
        f"推荐={_format_seconds(perf.get('recommendation'))} 桌宠渲染={_format_seconds(perf.get('pet_render'))}",
        f"质量={'通过' if perf.get('quality_ok') is True else '暂缓' if perf.get('quality_ok') is False else '-'} 评分={_format_score(perf.get('quality_score'))}",
        f"原因={perf.get('quality_reason', '-')}",
        f"桌宠={perf.get('pet_state', '-')} / {perf.get('pet_message', '-')}",
        f"来源={perf.get('source', '-')}",
    ]
    if capture_perf.get("pump_interval") is not None:
        lines.insert(4, f"帧泵间隔={_format_seconds(capture_perf.get('pump_interval'))}")
    return lines


def _format_seconds(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}s"
    return "-"


def _format_score(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "-"


def _short_source(source) -> str:
    text = str(source or "-")
    if text.startswith("window-content"):
        return "窗口内容"
    if text.startswith("window-screen"):
        return "桌面区域"
    if text.startswith("screen:fallback"):
        return "桌面兜底"
    return text[:24]


def _scene_has_finish_evidence(scene: SceneObservation | None, state: PuzzleState | None) -> bool:
    if scene is None:
        return False
    finish_region = scene.region(RegionKind.FINISH_DRAWING)
    if finish_region is not None and finish_region.confidence >= 0.58:
        return True
    if state is None:
        return False
    counts = Counter(tile.state for tile in state.tiles)
    # User-observed rule: completion can be available once all placeable blank
    # cells are gone. Void cells may remain because they are not placeable unless
    # a special spread affix converts/allows them.
    return counts.get(TileState.BLANK, 0) == 0


def _finish_reliability_reason(
    terminal: dict,
    steps: list[dict],
    *,
    finish_seen: bool,
    last_quality: dict | None = None,
) -> str:
    if last_quality is not None and not last_quality.get("ok") and not finish_seen:
        return f"最后一次识别低置信度（{last_quality.get('reason', '-') }），已跳过高分轨迹保存"
    reliable, reason = _terminal_reliability_for_finish(terminal, steps, finish_seen=finish_seen)
    return reason if not reliable else "可保存"


def _terminal_reliability_for_finish(terminal: dict, steps: list[dict], *, finish_seen: bool = False) -> tuple[bool, str]:
    if not steps:
        return False, "没有真实步骤记录，请先完整识别或重新快速获取一次"
    reliable_steps = [step for step in steps if (step.get("quality") or {}).get("ok") is True]
    if not reliable_steps:
        return False, "没有通过质量门槛的真实步骤，已跳过高分轨迹保存"
    latest = reliable_steps[-1]
    latest_quality = latest.get("quality") or {}
    if not finish_seen and float(latest_quality.get("score", 0.0) or 0.0) < float(config_value("quality.finish_min_board", 0.66)):
        return False, f"最后可靠步骤棋盘分数过低（{float(latest_quality.get('score', 0.0) or 0.0):.2f}），已跳过高分轨迹保存"
    known = terminal.get("known") or {}
    board_known = terminal.get("board_known") or {}
    total_known = int(known.get("total", 0) or 0)
    board_total = int(board_known.get("total", 0) or 0)
    if total_known <= 0:
        return False, "终局没有识别到任何颜料/背包内容，已跳过高分轨迹保存"
    if _summary_has_high_value(known):
        return True, "高价值终局可保存"
    # Low-value terminal runs are still useful only when the board paint
    # recognition has meaningful content. Avoid saving empty/garbled endings.
    min_board_paints = int(config_value("quality.finish_min_board_paints", 8))
    if finish_seen and board_total < min_board_paints:
        return False, f"已识别结算，但棋盘颜料等级识别少于{min_board_paints}个；为避免假成功，已跳过高分轨迹保存"
    if finish_seen and total_known > 0:
        return True, "有完成绘制/无空白地块证据，且棋盘颜料识别足够，可保存"
    if board_total >= min_board_paints:
        return True, "终局棋盘颜料数量足够，可保存"
    return False, f"终局价值较低且棋盘颜料识别少于{min_board_paints}个，已跳过高分轨迹保存"


def _terminal_is_reliable_for_finish(terminal: dict, steps: list[dict], *, finish_seen: bool = False) -> bool:
    reliable, _reason = _terminal_reliability_for_finish(terminal, steps, finish_seen=finish_seen)
    return reliable


def _stable_payload_key(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_session_id() -> str:
    return time.strftime("run-%Y%m%d-%H%M%S")


def _empirical_current_paint_defaults(level: int) -> tuple[int, int]:
    if level >= 5:
        return 4, 2
    if level == 4:
        return 4, 1
    return 3, 1


def _empirical_allowed_upgrades(level: int) -> list[int]:
    if level >= 5:
        return [2]
    if level == 4:
        return [1, 2]
    return [1]


def _empirical_upgrade_is_allowed(level: int, upgrade_amount: int) -> bool:
    return upgrade_amount in _empirical_allowed_upgrades(level)


def _tile_display_label(state: TileState, paint: Paint | None) -> str:
    if paint is None:
        return STATE_LABELS[state]
    if paint.is_treasure:
        return f"T{paint.level}"
    if paint.category == PaintCategory.MONSTER or paint.category.value == "monster":
        return f"M{paint.level}"
    if paint.category == PaintCategory.GAMEPLAY or paint.category.value == "gameplay":
        return f"G{paint.level}"
    if paint.is_treasure:
        return "箱"
    if paint.category.value == "monster":
        return "怪"
    if paint.category.value == "gameplay":
        return "玩"
    return STATE_LABELS[state]


def _paint_display_label(paint: Paint) -> str:
    if paint.is_treasure:
        return f"宝箱 Lv{paint.level}"
    category = "怪物" if paint.category.value == "monster" else "玩法" if paint.category.value == "gameplay" else "未知"
    subtype = f"-{paint.subtype}" if paint.subtype else ""
    return f"{category}{subtype} Lv{paint.level}"


def _paint_short_label(paint: Paint) -> str:
    if paint.is_treasure:
        return "宝箱"
    if paint.category.value == "monster":
        return "怪物"
    if paint.category.value == "gameplay":
        return "玩法"
    return "未知"


def _brush_label(brush_type: BrushType) -> str:
    return {
        BrushType.TRANSFORM: "转锋",
        BrushType.STACK: "堆砌",
        BrushType.SPREAD: "扩散",
    }[brush_type]


def _brush_type_from_value(value: str | None) -> BrushType | None:
    if value is None:
        return None
    try:
        return BrushType(value)
    except ValueError:
        return None


def _brush_types_from_values(values: tuple[str, ...]) -> tuple[BrushType, ...]:
    parsed = [_brush_type_from_value(value) for value in values]
    return normalize_brush_options([brush_type for brush_type in parsed if brush_type is not None])


def _scene_backpack_session_key(scene: SceneObservation) -> str:
    paint_hint = scene.current_paint.raw_text_hint if scene.current_paint else "no_paint"
    tile_counts = Counter(tile.state for tile in scene.board.tiles)
    board_key = ",".join(f"{state.value}:{tile_counts[state]}" for state in TileState)
    anchor = next(((tile.row, tile.col) for tile in scene.board.tiles if tile.state == TileState.PAINT), None)
    return f"{paint_hint}|{board_key}|anchor={anchor}"


def _observed_paint_key(observed) -> str:
    treasure = "treasure" if observed.is_treasure else "normal"
    return f"{observed.category.value}:{observed.level}:{observed.subtype}:{treasure}:{observed.slot_rect.x}:{observed.slot_rect.y}"


def _fast_current_paint_changed(scene: SceneObservation, previous_state: PuzzleState | None) -> bool:
    current = scene.current_paint
    if current is None or previous_state is None or previous_state.current_rule is None:
        return False
    previous = previous_state.current_rule.paint
    if current.category != PaintCategory.UNKNOWN and current.category_confidence >= 0.62 and current.category != previous.category:
        return True
    if current.level is not None and current.level_confidence >= 0.62 and abs(current.level - previous.level) >= 2:
        return True
    if current.is_treasure != previous.is_treasure and current.category_confidence >= 0.70:
        return True
    return False


def _direction_display_label(direction) -> str:
    return {
        "horizontal": "横向",
        "vertical": "纵向",
        "diagonal": "对角斜向",
        "adjacent": "相邻",
        "global": "全图",
        "unknown": "未知方向",
    }.get(direction.value, direction.value)


def _target_display_label(target) -> str:
    return {
        "monster": "怪物",
        "gameplay": "玩法",
        "all": "所有",
        "blocked_tile": "阻挡地块",
        "blank_tile": "空白地块",
        "void_tile": "虚空地块",
        "unknown": "颜料",
    }.get(target.value, target.value)


def _backpack_layout(count: int, width: float) -> tuple[int, int, int, int, int]:
    if count <= 0:
        return 1, 0, 0, 0, 0
    gap = 6
    per_row = min(count, max(6, min(10, int(width // 56))))
    slot_w = max(46, min(60, int((width - max(0, per_row - 1) * gap) // per_row)))
    slot_h = 38
    rows = math.ceil(count / per_row)
    return per_row, rows, slot_w, slot_h, gap


def _replace_option_menu(menu_widget: tk.OptionMenu, variable: tk.StringVar, labels: list[str]) -> None:
    menu = menu_widget["menu"]
    menu.delete(0, "end")
    for label in labels:
        menu.add_command(label=label, command=tk._setit(variable, label))


def _point_in_rect(point: tuple[float, float], rect: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _rects_intersect(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    left1, top1, right1, bottom1 = first
    left2, top2, right2, bottom2 = second
    return left1 < right2 and right1 > left2 and top1 < bottom2 and bottom1 > top2


def _clamp_int_var(value: tk.IntVar, minimum: int, maximum: int) -> int:
    try:
        raw = int(value.get())
    except (tk.TclError, ValueError):
        raw = minimum
    clamped = max(minimum, min(maximum, raw))
    if clamped != raw:
        value.set(clamped)
    return clamped
