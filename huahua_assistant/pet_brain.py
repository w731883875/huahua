from __future__ import annotations

from dataclasses import dataclass

from .fast_events import FastVisionEvent
from .game_model import BrushType, current_rule_is_pending


@dataclass(frozen=True)
class PetBrainState:
    mode: str
    message: str
    target: str | None = None
    confidence: float = 0.0
    event_kind: str | None = None


def evaluate_pet_state(app) -> PetBrainState:
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
    return None


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
