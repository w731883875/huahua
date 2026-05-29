from __future__ import annotations

import math
import random
import threading
import time
import tkinter as tk
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox

from .capture import DEFAULT_WINDOW_TITLE, capture_target, write_image
from .game_model import (
    AffixRule,
    BrushChoiceScenario,
    BrushType,
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
    state_from_observation,
)
from .learning import load_learning_stats, record_observation
from .models import BoardObservation, BrushOptionsObservation, RegionKind, SceneObservation, TileState
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


class HuahuaApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Huahua Puzzle Assistant v0.1")
        self.geometry("1180x720")
        self.minsize(1080, 640)
        self.configure(bg="#101820")

        self.status_var = tk.StringVar(
            value=f"Ready. Capture prefers window title: {DEFAULT_WINDOW_TITLE}"
        )
        self.board_var = tk.StringVar(value="Scene: not analyzed")
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
        self.tracker = VisionTracker()
        self.detected_backpack_cache: dict[str, Paint] = {}
        self.backpack_session_key: str | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self, bg="#101820")
        toolbar.pack(fill=tk.X, padx=14, pady=(14, 8))

        capture_btn = tk.Button(toolbar, text="Capture", command=self.capture_async, width=14)
        capture_btn.pack(side=tk.LEFT)

        simulate_btn = tk.Button(toolbar, text="Simulate", command=self.simulate_random, width=14)
        simulate_btn.pack(side=tk.LEFT, padx=(8, 0))

        manual_brush_btn = tk.Button(toolbar, text="Manual Brush", command=self._open_manual_brush_dialog, width=14)
        manual_brush_btn.pack(side=tk.LEFT, padx=(8, 0))

        force_btn = tk.Button(toolbar, text="强制完成", command=self.force_complete_current, width=10)
        force_btn.pack(side=tk.LEFT, padx=(8, 0))

        refresh_check = tk.Checkbutton(
            toolbar,
            text="Lv5 Pet Refresh",
            variable=self.refresh_available_var,
            fg="#d7e3ee",
            bg="#101820",
            activeforeground="#ffffff",
            activebackground="#101820",
            selectcolor="#0d141b",
            command=self._rerender_side_panel,
        )
        refresh_check.pack(side=tk.LEFT, padx=(10, 0))

        tk.Label(toolbar, text="Force Used", fg="#d7e3ee", bg="#101820").pack(side=tk.LEFT, padx=(10, 3))
        tk.Spinbox(
            toolbar,
            from_=0,
            to=3,
            width=3,
            textvariable=self.force_used_var,
            command=self._rerender_side_panel,
        ).pack(side=tk.LEFT)

        tk.Label(toolbar, text="Refresh Used", fg="#d7e3ee", bg="#101820").pack(side=tk.LEFT, padx=(10, 3))
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

    def capture_async(self) -> None:
        self.status_var.set("Capturing and analyzing screen...")
        self.withdraw()
        thread = threading.Thread(target=self._capture_worker, daemon=True)
        thread.start()

    def _capture_worker(self) -> None:
        try:
            time.sleep(0.25)
            started_at = time.perf_counter()
            capture = capture_target(
                DEFAULT_WINDOW_TITLE,
                fallback_to_screen=True,
                fallback_if_target_uncaptureable=False,
            )
            frame = capture.image
            write_image("runtime/capture.png", frame)
            scene, debug_path = analyze_and_save_debug(frame, self.debug_path, tracker=self.tracker)
            elapsed = time.perf_counter() - started_at
        except Exception as exc:
            self.after(0, lambda: self._show_error(exc))
            return

        self.after(0, lambda: self._render_result(scene, debug_path, capture.source, elapsed))

    def _show_error(self, exc: Exception) -> None:
        self._restore_app_window()
        self.status_var.set("Capture failed.")
        messagebox.showerror("Huahua Assistant", str(exc))

    def _render_result(
        self,
        scene: SceneObservation | None,
        debug_path: Path | None,
        capture_source: str,
        elapsed: float,
    ) -> None:
        self._restore_app_window()
        self.current_scene = scene
        if scene is None:
            self.current_state = None
            self.recommendations = []
            self.install_plan = None
            self._clear_brush_choice()
            self.board_var.set("Scene: board not detected")
            self.status_var.set(f"Source: {capture_source}. No board found. {elapsed:.2f}s. Debug image: {debug_path}")
            self._draw_empty_grid()
            self._write_regions(None)
            return

        board = scene.board
        backpack = self._merge_detected_backpack(scene)
        self.current_state = state_from_observation(scene.board, scene.current_paint, backpack)
        self._record_scene_observation(scene, backpack)
        if scene.brush_options is not None and self.current_state.current_rule is not None:
            self.current_state = replace(
                self.current_state,
                current_anchor=None,
                current_rule_pending=True,
            )
        self.recommendations = recommend_placements(self.current_state)
        self.install_plan = recommend_install_plan(self.current_state)
        self._clear_brush_choice()
        self._apply_detected_brush_options(scene.brush_options)
        rect = board.rect
        self.board_var.set(f"Board: x={rect.x} y={rect.y} size={rect.w} score={board.score:.2f}")
        self.status_var.set(f"Source: {capture_source}. {elapsed:.2f}s. Debug image saved: {debug_path}")
        self._draw_scene(scene)
        self._write_regions(scene)

    def _restore_app_window(self) -> None:
        self.deiconify()
        self.lift()

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
        self.current_state = random_state()
        self.selected_backpack_index = None
        self.recommendations = recommend_placements(self.current_state)
        self.install_plan = recommend_install_plan(self.current_state)
        self._clear_brush_choice()
        self.board_var.set("Mode: simulated puzzle")
        self.status_var.set("Simulated mode. Random board/current paint generated locally; no screen capture used.")
        self.canvas.delete("all")
        self._draw_grid(None, self.current_state)
        self._write_simulation()

    def _merge_detected_backpack(self, scene: SceneObservation) -> tuple[Paint, ...]:
        session_key = _scene_backpack_session_key(scene)
        if session_key != self.backpack_session_key:
            self.detected_backpack_cache = {}
            self.backpack_session_key = session_key
        if scene.backpack is None:
            return tuple(self.detected_backpack_cache.values())
        for observed in scene.backpack.paints:
            key = observed.fingerprint or _observed_paint_key(observed)
            self.detected_backpack_cache[key] = Paint(
                category=observed.category,
                level=observed.level,
                subtype=observed.subtype,
                is_treasure=observed.is_treasure,
            )
        return tuple(self.detected_backpack_cache.values())

    def _record_scene_observation(self, scene: SceneObservation, backpack: tuple[Paint, ...]) -> None:
        if scene.current_paint is not None:
            paint = scene.current_paint
            record_observation(
                "current_paint",
                {
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
        if scene.brush_options is not None:
            brush = scene.brush_options
            record_observation(
                "brush_options",
                {
                    "stage1_options": brush.stage1_options,
                    "chosen_stage1": brush.chosen_stage1,
                    "stage2_options": brush.stage2_options,
                    "confidence": brush.confidence,
                    "stage1_scores": brush.stage1_scores,
                    "stage2_scores": brush.stage2_scores,
                },
                source="real_capture",
            )
            for name in brush.stage2_options:
                if "进度要求减少1" in name or "所需数量-2" in name:
                    record_observation(
                        "rule_conflict",
                        {
                            "kind": "match_required_delta",
                            "affix_name": name,
                            "note": "转锋减进度存在 -1/-2 版本，需继续用实战截图确认",
                        },
                        source="real_capture",
                    )
        if scene.backpack is not None:
            record_observation(
                "backpack_capture",
                {
                    "visible": len(scene.backpack.paints),
                    "cumulative": len(backpack),
                    "paints": [
                        {
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
        if paint.upgrade_amount is not None and paint.upgrade_amount != expected_upgrade:
            record_observation(
                "rule_mismatch",
                {
                    "kind": "current_upgrade_amount",
                    "level": paint.level,
                    "observed": paint.upgrade_amount,
                    "expected": expected_upgrade,
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

    def _open_manual_brush_dialog(self) -> None:
        window = tk.Toplevel(self)
        window.title("Manual Brush Options")
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

        tk.Label(content, text="Stage 1 options", fg="#d7e3ee", bg="#101820", font=("Microsoft YaHei UI", 10, "bold")).grid(
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

        tk.Label(content, text="Chosen type for Stage 2", fg="#d7e3ee", bg="#101820").grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 4))
        chosen_menu = tk.OptionMenu(content, chosen_var, *[brush.value for brush in BrushType])
        chosen_menu.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=18)
        chosen_menu.grid(row=3, column=0, columnspan=3, sticky="w")

        tk.Label(content, text="Stage 2 affix A", fg="#d7e3ee", bg="#101820").grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 4))
        affix1_menu = tk.OptionMenu(content, affix1_var, "")
        affix1_menu.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=42)
        affix1_menu.grid(row=5, column=0, columnspan=3, sticky="ew")

        tk.Label(content, text="Stage 2 affix B", fg="#d7e3ee", bg="#101820").grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 4))
        affix2_menu = tk.OptionMenu(content, affix2_var, "")
        affix2_menu.configure(bg="#172331", fg="#d7e3ee", activebackground="#233449", activeforeground="#ffffff", width=42)
        affix2_menu.grid(row=7, column=0, columnspan=3, sticky="ew")

        affix_display: dict[str, AffixRule] = {}

        def refresh_affix_menus(*_args: object) -> None:
            brush_type = BrushType(chosen_var.get())
            affixes = [affix for affix in self.affix_catalog if affix.brush_type == brush_type]
            affix_display.clear()
            labels = []
            for index, affix in enumerate(affixes, start=1):
                label = f"{index}. {affix.name}"
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
                messagebox.showwarning("Huahua Assistant", "Stage 1 must select exactly two brush types.")
                return
            chosen = BrushType(chosen_var.get())
            if chosen not in selected:
                messagebox.showwarning("Huahua Assistant", "Chosen Stage 2 type must be one of the Stage 1 options.")
                return
            affix_a = affix_display.get(affix1_var.get())
            affix_b = affix_display.get(affix2_var.get())
            if affix_a is None or affix_b is None:
                messagebox.showwarning("Huahua Assistant", "Please select two Stage 2 affixes.")
                return
            self.brush_scenario = BrushChoiceScenario(stage1_options=selected, chosen_stage1=chosen, stage2_options=(affix_a, affix_b))
            self._rerender_side_panel()
            window.destroy()

        def clear_manual() -> None:
            self.brush_scenario = None
            self._rerender_side_panel()
            window.destroy()

        tk.Button(buttons, text="Apply", command=apply_manual, width=10).pack(side=tk.RIGHT)
        tk.Button(buttons, text="Clear", command=clear_manual, width=10).pack(side=tk.RIGHT, padx=(0, 8))
        tk.Button(buttons, text="Cancel", command=window.destroy, width=10).pack(side=tk.RIGHT, padx=(0, 8))

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
        top = 28 + summary_reserve
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

        for row in range(6):
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
                        text=f"{int(rec.score)}",
                        fill=outline,
                        font=("Microsoft YaHei UI", 9, "bold"),
                        anchor=tk.E,
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
        stage1_recs = recommend_brush_types(self.current_state, scenario.stage1_options) if self.current_state else []
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
                f"{rec_mark}{_brush_label(brush_type)}\n"
                f"score {score_by_type.get(brush_type, 0):.0f}\n"
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
        affix_recs = recommend_affixes(self.current_state, scenario.stage2_options) if self.current_state else []
        score_by_affix = {rec.affix: rec.score for rec in affix_recs}
        reason_by_affix = {rec.affix: rec.reason for rec in affix_recs}
        for index, affix in enumerate(scenario.stage2_options):
            selected = affix == self.selected_affix
            x1 = left
            x2 = left + width
            y2 = y + 52
            self.brush_affix_rects[index] = (x1, y, x2, y2)
            score_text = f"score {score_by_affix.get(affix, 0):.0f}"
            label = f"{affix.name}\n{score_text} | {reason_by_affix.get(affix, '按当前局面估算收益')}"
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
            return "点击左侧空白地块放置"
        if self.selected_brush_type is None:
            return "先选转锋/堆砌/扩散"
        if self.selected_affix is None:
            return "再选1条词缀"
        return "已选词缀，点击左侧放置"

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
            self.region_text.insert(tk.END, "Detected Regions\n\nNone")
            self.region_text.configure(state=tk.DISABLED)
            return

        region_names = {
            RegionKind.CURRENT_PAINT: "当前初始颜料",
            RegionKind.BACKPACK: "背包",
            RegionKind.BRUSH_OPTIONS: "笔触选项",
            RegionKind.CONFIRM_BUTTON: "确认按钮",
            RegionKind.FORCE_COMPLETE: "强制完成",
            RegionKind.REFRESH_AFFIX: "刷新词缀",
        }
        self.region_text.insert(tk.END, "Detected Regions\n\n")
        self.region_text.insert(
            tk.END,
            f"Board\nx={scene.board.rect.x} y={scene.board.rect.y} "
            f"w={scene.board.rect.w} h={scene.board.rect.h}\n"
            f"score={scene.board.score:.2f}\n\n",
        )
        for region in scene.regions:
            name = region_names[region.kind]
            text = (
                f"{name}\n"
                f"x={region.rect.x} y={region.rect.y} "
                f"w={region.rect.w} h={region.rect.h}\n"
                f"confidence={region.confidence:.2f}\n\n"
            )
            self.region_text.insert(tk.END, text)
        if scene.current_paint is not None:
            paint = scene.current_paint
            level = paint.level if paint.level is not None else "?"
            self.region_text.insert(tk.END, "Current Paint\n")
            self.region_text.insert(
                tk.END,
                f"category={paint.category.value} ({paint.category_confidence:.2f})\n"
                f"subtype={paint.subtype}\n"
                f"is_treasure={paint.is_treasure}\n"
                f"level={level} ({paint.level_confidence:.2f})\n"
                f"target={paint.target_object.value}\n"
                f"direction={paint.direction.value}\n"
                f"upgrade={paint.upgrade_amount}\n"
                f"match={paint.match_current}/{paint.match_required}\n"
                f"hint={paint.raw_text_hint}\n"
                f"keywords={_format_keyword_scores(paint.keyword_scores)}\n"
                "crops=runtime/current_paint_*.png\n\n",
            )
        if scene.brush_options is not None:
            brush = scene.brush_options
            self.region_text.insert(tk.END, "Brush Options Detected\n")
            self.region_text.insert(
                tk.END,
                f"stage1={', '.join(brush.stage1_options) or '-'}\n"
                f"chosen={brush.chosen_stage1 or '-'}\n"
                f"stage2={'; '.join(brush.stage2_options) or '-'}\n"
                f"confidence={brush.confidence:.2f}\n"
                f"stage1_scores={_format_keyword_scores(brush.stage1_scores)}\n"
                f"stage2_scores={_format_keyword_scores(brush.stage2_scores)}\n"
                "crop=runtime/brush_options_panel.png\n\n",
            )
        if scene.backpack is not None:
            self.region_text.insert(
                tk.END,
                f"Backpack Capture\n"
                f"visible={len(scene.backpack.paints)} cumulative={len(self.detected_backpack_cache)}\n"
                f"confidence={scene.backpack.confidence:.2f}\n"
                "manual scroll supported: scroll in game, then Capture again\n\n",
            )
        self._write_learning_stats()
        self._write_backpack()
        self._write_install_plan()
        self._write_recommendations()
        self._write_brush_recommendations()
        self._write_special_actions()
        self.region_text.configure(state=tk.DISABLED)

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
        self.region_text.insert(tk.END, "单步预览（安装即生效）\n")
        for index, rec in enumerate(self.recommendations, start=1):
            mark = "BEST " if index == 1 else ""
            self.region_text.insert(
                tk.END,
                f"{mark}{index}. r{rec.row + 1} c{rec.col + 1} "
                f"score={rec.score:.1f} {rec.reason}\n",
            )
        self.region_text.insert(tk.END, "\n")

    def _write_backpack(self) -> None:
        self.region_text.insert(tk.END, "背包（已按当前目标优先排序）\n")
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
        affix_seen = stats.get("affix_seen") or {}
        top_affixes = sorted(affix_seen.items(), key=lambda item: item[1], reverse=True)[:3]
        affix_text = ", ".join(f"{name}x{count}" for name, count in top_affixes) if top_affixes else "-"
        self.region_text.insert(
            tk.END,
            "Learning Log\n"
            f"events={stats.get('total_events', 0)} conflicts={conflicts} mismatches={mismatches}\n"
            f"top affixes={affix_text}\n"
            f"treasure_ratio={stats.get('treasure_ratio', 0.0):.3f}\n"
            "files=runtime/observations.jsonl, runtime/stats.json\n\n",
        )

    def _write_install_plan(self) -> None:
        self.region_text.insert(tk.END, "安装计划（逐步模拟，每步立即结算）\n")
        if self.install_plan is None:
            self.region_text.insert(tk.END, "等待当前初始颜料和背包识别\n\n")
            return
        self.region_text.insert(tk.END, f"{'COMPLETE' if self.install_plan.completes_match else 'INCOMPLETE'}: {self.install_plan.reason}\n")
        for index, step in enumerate(self.install_plan.steps, start=1):
            self.region_text.insert(
                tk.END,
                f"{index}. r{step.row + 1} c{step.col + 1} <- {_paint_display_label(step.paint)}\n",
            )
        self.region_text.insert(tk.END, "\n")

    def _write_brush_recommendations(self) -> None:
        if self.current_state is None:
            return
        stage1 = (
            self.brush_scenario.stage1_options
            if self.brush_scenario
            else (BrushType.TRANSFORM, BrushType.STACK, BrushType.SPREAD)
        )
        self.region_text.insert(tk.END, "Brush Stage 1\n")
        for rec in recommend_brush_types(self.current_state, stage1):
            self.region_text.insert(
                tk.END,
                f"- {rec.brush_type.value} score={rec.score:.1f} {rec.reason}\n",
            )
        self.region_text.insert(tk.END, "\n")

        if not self.brush_scenario:
            self.region_text.insert(tk.END, "Brush Stage 2\nWaiting for real option recognition\n\n")
            return
        self.region_text.insert(tk.END, f"Brush Stage 2 ({self.brush_scenario.chosen_stage1.value})\n")
        for rec in recommend_affixes(self.current_state, self.brush_scenario.stage2_options):
            self.region_text.insert(
                tk.END,
                f"- {rec.affix.name} score={rec.score:.1f} {rec.reason}\n",
            )
        self.region_text.insert(tk.END, "\n")

    def _write_special_actions(self) -> None:
        if self.current_state is None:
            return
        resources = self._resource_state()
        actions = recommend_special_actions(self.current_state, resources, self.brush_scenario)
        self.region_text.insert(tk.END, "Special Actions\n")
        self.region_text.insert(
            tk.END,
            f"resources: force={resources.force_remaining}/3 "
            f"refresh={resources.refresh_remaining}/2 "
            f"pet={'yes' if resources.refresh_available else 'no'}\n",
        )
        if not actions:
            refresh_state = "enabled" if resources.refresh_available else "disabled"
            self.region_text.insert(tk.END, f"- refresh={refresh_state}; waiting for actionable state\n\n")
            return
        for action in actions:
            mark = "USE" if action.recommended else "HOLD"
            self.region_text.insert(
                tk.END,
                f"- {mark} {action.action.value} score={action.score:.1f} {action.reason}\n",
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


def main() -> None:
    app = HuahuaApp()
    app.mainloop()


def _format_keyword_scores(scores: dict[str, float]) -> str:
    if not scores:
        return "-"
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:4]
    return ", ".join(f"{key}:{value:.2f}" for key, value in top)


def _empirical_current_paint_defaults(level: int) -> tuple[int, int]:
    if level >= 5:
        return 4, 2
    if level == 4:
        return 4, 1
    return 3, 1


def _tile_display_label(state: TileState, paint: Paint | None) -> str:
    if paint is None:
        return STATE_LABELS[state]
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


def _clamp_int_var(value: tk.IntVar, minimum: int, maximum: int) -> int:
    try:
        raw = int(value.get())
    except (tk.TclError, ValueError):
        raw = minimum
    clamped = max(minimum, min(maximum, raw))
    if clamped != raw:
        value.set(clamped)
    return clamped
