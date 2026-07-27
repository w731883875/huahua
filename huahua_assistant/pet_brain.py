from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

from .fast_events import FastVisionEvent
from .game_model import (
    BrushType,
    current_rule_is_pending,
    recommend_affixes,
    recommend_brush_types,
)
from .guidance import (
    GamePhase,
    GuidanceController,
    GuidanceState,
    PhaseEvidence,
    RecommendationTarget,
    TargetKind,
    TargetProposal,
)
from .models import RegionKind


@dataclass(frozen=True)
class PetBrainState:
    mode: str
    message: str
    target: str | None = None
    confidence: float = 0.0
    event_kind: str | None = None
    phase: GamePhase = GamePhase.WAITING
    recommendation_target: RecommendationTarget | None = None


def evaluate_pet_state(app) -> PetBrainState:
    guidance = _evaluate_guidance(app)
    state = _evaluate_immediate_pet_state(app)
    return replace(
        state,
        phase=guidance.phase.phase,
        recommendation_target=guidance.target,
    )


def _evaluate_immediate_pet_state(app) -> PetBrainState:
    events = tuple(getattr(app, "last_fast_events", ()) or ())
    if _latest_snapshot_is_missing(app):
        waiting_state = getattr(app, "last_pet_state", None)
        if isinstance(waiting_state, PetBrainState) and waiting_state.mode == "observing":
            return waiting_state

    unstable_state = _unstable_capture_state(app)
    if unstable_state is not None:
        return unstable_state

    finish_event = _first_event(events, "finish_candidate")
    if finish_event is not None:
        return PetBrainState(
            mode="finish_hint",
            message="可结算",
            target="finish",
            confidence=finish_event.confidence,
            event_kind=finish_event.kind,
        )

    state = getattr(app, "current_state", None)
    scene = getattr(app, "current_scene", None)
    if state is not None and scene is not None and current_rule_is_pending(state):
        brush_state = _brush_state(app)
        if brush_state is not None:
            return brush_state

    selection_event = _first_event(events, "selection_state_changed")
    if selection_event is not None:
        return PetBrainState(
            mode="wait_for_drop",
            message="等落笔",
            target="board",
            confidence=selection_event.confidence,
            event_kind=selection_event.kind,
        )

    if _has_backpack_target(app):
        return PetBrainState(
            mode="recommend_backpack",
            message="先点背包",
            target="backpack",
            confidence=0.85,
        )

    if getattr(app, "recommendations", None):
        return PetBrainState(
            mode="recommend_board_cell",
            message="推荐地块",
            target="board",
            confidence=0.8,
        )

    if _first_event(events, "snapshot_ready") is not None:
        return PetBrainState(mode="observing", message="观察中", confidence=0.55, event_kind="snapshot_ready")
    if scene is None:
        return PetBrainState(mode="observing", message="等待识别", confidence=0.25)
    return PetBrainState(mode="observing", message="观察中", confidence=0.45)


def _brush_state(app) -> PetBrainState | None:
    scenario = getattr(app, "brush_scenario", None)
    if scenario is None:
        return None
    selected_type = getattr(app, "selected_brush_type", None)
    selected_affix = getattr(app, "selected_affix", None)
    if selected_type is None:
        return PetBrainState(
            mode="recommend_brush",
            message="选笔触",
            target="brush_stage1",
            confidence=0.78,
        )
    if selected_affix is None:
        return PetBrainState(
            mode="recommend_brush",
            message=f"选{_brush_label(selected_type)}词条",
            target="brush_stage2",
            confidence=0.78,
        )
    return None


def _has_backpack_target(app) -> bool:
    install_plan = getattr(app, "install_plan", None)
    if install_plan is not None and getattr(install_plan, "steps", None):
        return True
    recommendations = list(getattr(app, "recommendations", []) or [])
    return bool(recommendations and getattr(recommendations[0], "paint", None) is not None)


def _latest_snapshot_is_missing(app) -> bool:
    return getattr(app, "last_fast_snapshot", None) is None and not tuple(getattr(app, "last_fast_events", ()) or ())


def _unstable_capture_state(app) -> PetBrainState | None:
    quality = getattr(app, "last_capture_quality", None)
    if not isinstance(quality, dict) or quality.get("ok", True):
        return None
    reason = str(quality.get("reason") or "")
    if "capture_source_composited" in reason:
        return PetBrainState(
            mode="observing",
            message="截图不稳",
            confidence=0.2,
            event_kind="capture_source_composited",
        )
    return PetBrainState(
        mode="observing",
        message="识别不稳",
        confidence=0.25,
        event_kind="quality_low",
    )


def _first_event(events: tuple[FastVisionEvent, ...], kind: str) -> FastVisionEvent | None:
    for event in events:
        if event.kind == kind:
            return event
    return None


def _brush_label(brush_type: BrushType) -> str:
    if brush_type == BrushType.TRANSFORM:
        return "转锋"
    if brush_type == BrushType.STACK:
        return "堆砌"
    if brush_type == BrushType.SPREAD:
        return "扩散"
    return "笔触"


def _evaluate_guidance(app) -> GuidanceState:
    controller = getattr(app, "_guidance_controller", None)
    if not isinstance(controller, GuidanceController):
        controller = GuidanceController()
        setattr(app, "_guidance_controller", controller)
    frame_id = _guidance_frame_id(app)
    evidence = _phase_evidence(app, frame_id)
    proposal = _target_proposal(app, evidence)
    guidance = controller.update(
        evidence,
        proposal,
        context_signature=_guidance_context_signature(app),
        now=time.monotonic(),
    )
    setattr(app, "guidance_state", guidance)
    setattr(app, "recommendation_target", guidance.target)
    return guidance


def _guidance_frame_id(app) -> str:
    snapshot = getattr(app, "last_fast_snapshot", None)
    identity = id(snapshot) if snapshot is not None else None
    if getattr(app, "_guidance_snapshot_identity", object()) != identity:
        setattr(app, "_guidance_snapshot_identity", identity)
        serial = int(getattr(app, "_guidance_frame_serial", 0) or 0) + 1
        setattr(app, "_guidance_frame_serial", serial)
    else:
        serial = int(getattr(app, "_guidance_frame_serial", 0) or 0)
    signature = str(getattr(snapshot, "board_signature", "missing") or "missing")[:12]
    return f"frame-{serial}:{signature}"


def _phase_evidence(app, frame_id: str) -> PhaseEvidence:
    snapshot = getattr(app, "last_fast_snapshot", None)
    scene = getattr(app, "current_scene", None)
    state = getattr(app, "current_state", None)
    events = tuple(getattr(app, "last_fast_events", ()) or ())
    quality = getattr(app, "last_capture_quality", None)
    quality_ok = bool(quality.get("ok", True)) if isinstance(quality, dict) else bool(
        getattr(snapshot, "reliable", True)
    )
    quality_reason = str(quality.get("reason") or "") if isinstance(quality, dict) else ""
    board_confidence = _guidance_confidence(app)
    brush = getattr(snapshot, "brush", None)
    brush_visible = brush is not None or getattr(scene, "brush_options", None) is not None
    scenario = getattr(app, "brush_scenario", None)
    selected_type = getattr(app, "selected_brush_type", None)
    selected_affix = getattr(app, "selected_affix", None)
    stage2_visible = bool(
        getattr(brush, "stage2_options", ())
        or getattr(scenario, "stage2_options", ())
    )
    selection_changed = _first_event(events, "selection_state_changed") is not None
    finish_signals = _finish_signals(app, events)
    current_rule = getattr(state, "current_rule", None)
    rule_pending = bool(state is not None and current_rule_is_pending(state))
    return PhaseEvidence(
        frame_id=frame_id,
        board_detected=snapshot is not None or scene is not None,
        board_signature=str(getattr(snapshot, "board_signature", "") or ""),
        board_confidence=board_confidence,
        quality_ok=quality_ok,
        conflicting="conflict" in quality_reason,
        current_rule_present=current_rule is not None,
        current_rule_pending=rule_pending,
        brush_visible=brush_visible,
        brush_stage2_visible=stage2_visible,
        brush_type_selected=selected_type is not None or bool(getattr(brush, "chosen_stage1", None)),
        affix_selected=selected_affix is not None,
        backpack_target=_has_backpack_target(app) and not selection_changed,
        board_target=selection_changed or bool(getattr(app, "recommendations", None)),
        finish_signals=frozenset(finish_signals),
    )


def _finish_signals(app, events: tuple[FastVisionEvent, ...]) -> set[str]:
    signals: set[str] = set()
    scene = getattr(app, "current_scene", None)
    snapshot = getattr(app, "last_fast_snapshot", None)
    if scene is not None and scene.region(RegionKind.FINISH_DRAWING) is not None:
        signals.add("finish_button")
    terrain_counts = getattr(snapshot, "terrain_counts", None)
    if _terrain_counts_show_terminal_map(terrain_counts):
        signals.add("terminal_map")
    for event in events:
        if event.kind != "finish_candidate":
            continue
        reason = str(event.detail.get("reason") or "")
        if reason == "finish_region":
            signals.add("finish_button")
        elif reason == "no_blank_tiles":
            if _terrain_counts_show_terminal_map(event.detail.get("terrain_counts")):
                signals.add("terminal_map")
        else:
            signals.add("finish_event")
    state = getattr(app, "current_state", None)
    rule = getattr(state, "current_rule", None)
    if rule is not None and int(getattr(rule, "match_current", 0) or 0) >= int(
        getattr(rule, "match_required", 1) or 1
    ):
        signals.add("rule_complete")
    return signals


def _terrain_counts_show_terminal_map(terrain_counts) -> bool:
    if not isinstance(terrain_counts, dict) or "blank" not in terrain_counts:
        return False
    blank_count = terrain_counts.get("blank")
    if not isinstance(blank_count, (int, float)) or not math.isfinite(float(blank_count)):
        return False
    known_cells = 0
    for value in terrain_counts.values():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            continue
        known_cells += int(value)
    return float(blank_count) == 0.0 and known_cells >= 36


def _target_proposal(app, evidence: PhaseEvidence) -> TargetProposal | None:
    if not evidence.quality_ok or evidence.conflicting:
        return None
    confidence = _guidance_confidence(app)
    scene = getattr(app, "current_scene", None)
    state = getattr(app, "current_state", None)

    if len(evidence.finish_signals) >= 2 and scene is not None:
        region = scene.region(RegionKind.FINISH_DRAWING)
        if region is not None:
            point = _capture_rect_point(app, region.rect, 0.5, 0.5)
            if point is not None:
                return TargetProposal(
                    key="finish",
                    kind=TargetKind.FINISH,
                    priority=0,
                    screen_point=point,
                    confidence=min(confidence, float(region.confidence)),
                    reason_short="结束绘制",
                    source_frame_id=evidence.frame_id,
                )

    if state is not None and evidence.current_rule_pending:
        brush_target = _brush_target_proposal(app, evidence, confidence)
        if brush_target is not None:
            return brush_target

    if evidence.backpack_target:
        desired = _desired_backpack_paint(app)
        visible_index = _call_optional(app, "_visible_backpack_index_for_paint", desired)
        point = _call_optional(app, "_screen_point_for_backpack_slot", visible_index)
        if desired is not None and isinstance(visible_index, int) and _target_point_is_safe(app, point):
            backpack_confidence = float(getattr(getattr(scene, "backpack", None), "confidence", confidence))
            return TargetProposal(
                key=f"backpack:{visible_index}",
                kind=TargetKind.BACKPACK,
                priority=30,
                screen_point=(int(point[0]), int(point[1])),
                confidence=min(confidence, backpack_confidence),
                reason_short="先点背包",
                source_frame_id=evidence.frame_id,
            )

    recommendations = list(getattr(app, "recommendations", ()) or ())
    if evidence.board_target and recommendations:
        rec = recommendations[0]
        row = getattr(rec, "row", None)
        col = getattr(rec, "col", None)
        point = _call_optional(app, "_screen_point_for_board_cell", row, col)
        if isinstance(row, int) and isinstance(col, int) and _target_point_is_safe(app, point):
            return TargetProposal(
                key=f"board:{row},{col}",
                kind=TargetKind.BOARD,
                priority=40,
                screen_point=(int(point[0]), int(point[1])),
                confidence=confidence,
                reason_short=_recommendation_reason(rec),
                source_frame_id=evidence.frame_id,
            )
    return None


def _brush_target_proposal(
    app,
    evidence: PhaseEvidence,
    confidence: float,
) -> TargetProposal | None:
    scene = getattr(app, "current_scene", None)
    state = getattr(app, "current_state", None)
    scenario = getattr(app, "brush_scenario", None)
    brush_observation = getattr(scene, "brush_options", None)
    panel = getattr(brush_observation, "panel_rect", None)
    if state is None or scenario is None or panel is None:
        return None
    confidence = min(confidence, float(getattr(brush_observation, "confidence", confidence)))
    selected_type = getattr(app, "selected_brush_type", None)
    selected_affix = getattr(app, "selected_affix", None)
    if selected_type is None:
        options = tuple(getattr(scenario, "stage1_options", ()) or ())
        recommendations = recommend_brush_types(state, options) if options else []
        if not recommendations:
            return None
        best = recommendations[0]
        index = options.index(best.brush_type)
        point = _capture_rect_point(app, panel, (index + 0.5) / len(options), 0.34)
        if point is None:
            return None
        return TargetProposal(
            key=f"brush_stage1:{best.brush_type.value}",
            kind=TargetKind.BRUSH_STAGE1,
            priority=20,
            screen_point=point,
            confidence=confidence,
            reason_short=_short_reason(best.reason),
            source_frame_id=evidence.frame_id,
        )
    if selected_affix is None:
        affixes = tuple(getattr(scenario, "stage2_options", ()) or ())
        recommendations = recommend_affixes(state, affixes) if affixes else []
        if not recommendations:
            return None
        best = recommendations[0]
        index = affixes.index(best.affix)
        y_ratio = 0.45 if len(affixes) <= 1 else 0.35 + index * 0.25
        point = _capture_rect_point(app, panel, 0.58, y_ratio)
        if point is None:
            return None
        identity = best.affix.canonical_affix_id or best.affix.name
        return TargetProposal(
            key=f"brush_stage2:{identity}",
            kind=TargetKind.BRUSH_STAGE2,
            priority=10,
            screen_point=point,
            confidence=confidence,
            reason_short=_short_reason(best.reason),
            source_frame_id=evidence.frame_id,
        )
    return None


def _guidance_confidence(app) -> float:
    quality = getattr(app, "last_capture_quality", None)
    snapshot = getattr(app, "last_fast_snapshot", None)
    value = quality.get("score") if isinstance(quality, dict) else None
    if not isinstance(value, (int, float)):
        value = getattr(snapshot, "board_score", 0.0)
    return max(0.0, min(1.0, float(value or 0.0)))


def _guidance_context_signature(app) -> tuple[object, ...]:
    window = getattr(app, "last_capture_window", None)
    snapshot = getattr(app, "last_fast_snapshot", None)
    board_rect = getattr(snapshot, "board_rect", None)
    return (
        getattr(window, "left", None),
        getattr(window, "top", None),
        getattr(window, "width", None),
        getattr(window, "height", None),
        tuple(getattr(app, "last_capture_size", ()) or ()),
        getattr(board_rect, "x", None),
        getattr(board_rect, "y", None),
        getattr(board_rect, "w", None),
        getattr(board_rect, "h", None),
    )


def _desired_backpack_paint(app):
    install_plan = getattr(app, "install_plan", None)
    steps = tuple(getattr(install_plan, "steps", ()) or ())
    if steps:
        return steps[0].paint
    recommendations = list(getattr(app, "recommendations", ()) or ())
    return getattr(recommendations[0], "paint", None) if recommendations else None


def _capture_rect_point(app, rect, x_ratio: float, y_ratio: float) -> tuple[int, int] | None:
    point = _call_optional(app, "_screen_point_for_capture_rect", rect, x_ratio, y_ratio)
    if not _target_point_is_safe(app, point):
        return None
    return int(point[0]), int(point[1])


def _call_optional(app, name: str, *args):
    function = getattr(app, name, None)
    if not callable(function) or any(arg is None for arg in args):
        return None
    return function(*args)


def _valid_point(point) -> bool:
    return (
        isinstance(point, (tuple, list))
        and len(point) == 2
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in point
        )
    )


def _target_point_is_safe(app, point) -> bool:
    if not _valid_point(point):
        return False
    x, y = float(point[0]), float(point[1])
    window = getattr(app, "last_capture_window", None)
    if window is not None:
        left = getattr(window, "left", None)
        top = getattr(window, "top", None)
        width = getattr(window, "width", None)
        height = getattr(window, "height", None)
        bounds = (left, top, width, height)
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in bounds
        ):
            return False
        if width <= 0 or height <= 0:
            return False
        if not (left <= x < left + width and top <= y < top + height):
            return False
    point_is_on_monitor = (
        _windows_point_is_on_monitor(x, y)
        if callable(getattr(app, "winfo_exists", None))
        else None
    )
    if point_is_on_monitor is False:
        return False
    virtual_bounds = _virtual_screen_bounds(app) if point_is_on_monitor is None else None
    if virtual_bounds is not None:
        left, top, width, height = virtual_bounds
        if width <= 0 or height <= 0:
            return False
        if not (left <= x < left + width and top <= y < top + height):
            return False
    return True


def _windows_point_is_on_monitor(x: float, y: float) -> bool | None:
    try:
        import ctypes
        from ctypes import wintypes

        monitor_from_point = ctypes.windll.user32.MonitorFromPoint
        monitor_from_point.argtypes = (wintypes.POINT, wintypes.DWORD)
        monitor_from_point.restype = wintypes.HMONITOR
        monitor = monitor_from_point(wintypes.POINT(int(x), int(y)), 0)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return None
    return bool(monitor)


def _virtual_screen_bounds(app) -> tuple[int, int, int, int] | None:
    getters = tuple(
        getattr(app, name, None)
        for name in ("winfo_vrootx", "winfo_vrooty", "winfo_vrootwidth", "winfo_vrootheight")
    )
    if not all(callable(getter) for getter in getters):
        return None
    try:
        return tuple(int(getter()) for getter in getters)
    except Exception:
        return None


def _recommendation_reason(recommendation) -> str:
    explanation = getattr(recommendation, "explanation", None)
    short = str(getattr(explanation, "short", "") or "")
    return _short_reason(short or getattr(recommendation, "reason", "") or "推荐地块")


def _short_reason(value: str, limit: int = 12) -> str:
    text = "".join(str(value or "").replace("；", "，").split())
    return (text.split("，", 1)[0] or "推荐")[:limit]
