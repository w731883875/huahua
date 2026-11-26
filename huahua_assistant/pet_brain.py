from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

from .fast_events import FastVisionEvent
from .game_model import (
    BrushType,
    assess_finish_mode,
    current_scope_cells,
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
from .llm.binding import ADVICE_TTL_SECONDS
from .models import RegionKind, is_enter_painting_region


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
    result = replace(
        state,
        phase=guidance.phase.phase,
        recommendation_target=guidance.target,
    )
    if guidance.phase.phase == GamePhase.FINISH_READY and guidance.target is not None:
        # 阶段已经稳定判成「可结算」、目标也锁到入画按钮了，提示语要跟上，
        # 否则会出现「箭头指着入画、文字还写观察中」的割裂。
        result = replace(result, mode="finish_hint", message="可结算", target="finish")

    # 大模型托管：它给了有效建议就以它为准。本地链路仍在跑，只是这一帧让位——
    # 模型的建议慢半拍但看得更整体，这正是「各做各的事情」的分工。
    llm_target = _llm_recommendation(app)
    if llm_target is not None:
        result = replace(
            result,
            mode="llm_hint",
            message=llm_target.reason_short or "大模型建议",
            target="llm",
            confidence=max(result.confidence, llm_target.confidence),
            recommendation_target=llm_target,
        )
    return result


_LLM_KIND = {
    "place": TargetKind.BOARD,
    "backpack": TargetKind.BACKPACK,
    "brush_stage1": TargetKind.BRUSH_STAGE1,
    "brush_stage2": TargetKind.BRUSH_STAGE2,
    "finish": TargetKind.FINISH,
}


def _llm_recommendation(app):
    """把大模型的建议包成和本地链路同构的推荐目标。"""
    advice = getattr(app, "llm_advice", None)
    if advice is None or not getattr(advice, "ok", False):
        return None
    to_point = getattr(app, "_llm_target_point", None)
    point = to_point() if callable(to_point) else None
    if point is None:
        return None
    kind = _LLM_KIND.get(advice.action, TargetKind.BOARD)
    now = time.monotonic()
    reason = (advice.reason or advice.action_label)[:24]
    # 截止时刻由**原始帧的采集时刻**决定，不从这次渲染算起：同一张建议无论
    # 重绘多少次都还是同一个截止点，否则箭头能被"每帧重算 +6 秒"无限续命。
    deadline_of = getattr(app, "llm_advice_deadline", None)
    deadline = deadline_of() if callable(deadline_of) else None
    if deadline is None:
        deadline = now + ADVICE_TTL_SECONDS
    return RecommendationTarget(
        key=f"llm:{advice.action}:{advice.row},{advice.col}:{advice.option or ''}",
        kind=kind,
        priority=0,
        screen_point=(int(point[0]), int(point[1])),
        confidence=0.9,
        reason_short=f"大模型·{reason}",
        source_frame_id="llm",
        stable_since=now,
        expires_at=float(deadline),
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

    # 调色目标已经凑不满：这时候再推「选背包颜料 / 推荐地块」只会把人往
    # 范围外的空格上带。直接亮「收尾」，提示保住贵重颜料、点完成绘制。
    if state is not None and scene is not None:
        finish = assess_finish_mode(state)
        if finish is not None and not finish.reachable:
            return PetBrainState(
                mode="finish_hint",
                message="凑不满，收尾",
                target="finish",
                confidence=0.7,
            )

    if _backpack_target_needed(app):
        return PetBrainState(
            mode="recommend_backpack",
            message="选背包颜料",
            target="backpack",
            confidence=0.8,
        )

    if _board_target_ready(app):
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


def _desired_backpack_paint(app):
    install_plan = getattr(app, "install_plan", None)
    steps = tuple(getattr(install_plan, "steps", ()) or ())
    if steps:
        return getattr(steps[0], "paint", None)
    recommendations = list(getattr(app, "recommendations", ()) or ())
    if recommendations:
        return getattr(recommendations[0], "paint", None)
    return None


def _visible_backpack_entries(app) -> tuple[object, ...]:
    snapshot = getattr(app, "last_fast_snapshot", None)
    paints = tuple(getattr(snapshot, "backpack", ()) or ())
    if paints:
        return paints
    scene = getattr(app, "current_scene", None)
    backpack = getattr(scene, "backpack", None)
    return tuple(getattr(backpack, "paints", ()) or ())


def _visible_backpack_index(app, paint) -> int | None:
    if paint is None:
        return None
    function = getattr(app, "_visible_backpack_index_for_paint", None)
    if not callable(function):
        return None
    try:
        index = function(paint)
    except (AttributeError, TypeError, ValueError):
        return None
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return None
    return index


def _backpack_paint_is_selected(entry) -> bool:
    if not bool(getattr(entry, "is_selected", False)):
        return False
    value = getattr(entry, "selection_confidence", 0.0)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    return float(value) >= 0.3


def _backpack_target_needed(app) -> bool:
    state = getattr(app, "current_state", None)
    if state is not None and current_rule_is_pending(state):
        return False
    desired = _desired_backpack_paint(app)
    if desired is None:
        return False
    entries = _visible_backpack_entries(app)
    if not entries:
        return False
    index = _visible_backpack_index(app, desired)
    if index is None or index >= len(entries):
        return False
    return not _backpack_paint_is_selected(entries[index])


def _board_target_ready(app) -> bool:
    state = getattr(app, "current_state", None)
    if state is not None and current_rule_is_pending(state):
        # 待落初始颜料：用的是手里那枚（「当前颜料」面板），不用先从背包挑，
        # 所以只要算出了落点就直接提示，别卡在「背包选中」这一步上。
        return bool(getattr(app, "recommendations", ()) or ())
    has_request = _has_backpack_target(app) or bool(getattr(app, "recommendations", ()) or ())
    if not has_request or _backpack_target_needed(app):
        return False
    entries = _visible_backpack_entries(app)
    if not entries:
        return True
    desired = _desired_backpack_paint(app)
    if desired is None:
        return True
    index = _visible_backpack_index(app, desired)
    if index is not None and index < len(entries):
        return _backpack_paint_is_selected(entries[index])
    return any(_backpack_paint_is_selected(entry) for entry in entries)


def _backpack_reason(paint) -> str:
    return f"选{_paint_hint_label(paint)}"


def _paint_hint_label(paint) -> str:
    if paint is None:
        return "背包颜料"
    level = getattr(paint, "level", None)
    suffix = f"Lv{level}" if isinstance(level, int) else ""
    if bool(getattr(paint, "is_treasure", False)):
        return f"宝箱{suffix}" if suffix else "宝箱"
    category = getattr(getattr(paint, "category", None), "value", "")
    prefix = "怪物" if category == "monster" else "玩法" if category == "gameplay" else "颜料"
    return f"{prefix}{suffix}" if suffix else prefix


def _latest_snapshot_is_missing(app) -> bool:
    return getattr(app, "last_fast_snapshot", None) is None and not tuple(getattr(app, "last_fast_events", ()) or ())


def _unstable_capture_state(app) -> PetBrainState | None:
    quality = getattr(app, "last_capture_quality", None)
    if not isinstance(quality, dict) or quality.get("ok", True):
        return None
    if _high_confidence_start_visible(app):
        return None
    scene = getattr(app, "current_scene", None)
    if scene is not None and scene.region(RegionKind.FINISH_DRAWING) is not None:
        # 结束态地块全是颜料/全空，格子状态本来就难分，会被 unknown_tiles 判低。
        # 但「入画」按钮很明确，这时候不该报「识别不稳」。
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
    if _high_confidence_start_visible(app):
        quality_ok = True
    if scene is not None and scene.region(RegionKind.FINISH_DRAWING) is not None:
        # 结束态格子状态难分（全空/全颜料），但入画按钮是硬信号，别判成识别不稳。
        quality_ok = True
    quality_reason = str(quality.get("reason") or "") if isinstance(quality, dict) else ""
    board_confidence = _guidance_confidence(app)
    scenario = getattr(app, "brush_scenario", None)
    brush = getattr(snapshot, "brush", None)
    brush_visible = (
        brush is not None
        or getattr(scene, "brush_options", None) is not None
        or scenario is not None
    )
    selected_type = getattr(app, "selected_brush_type", None)
    selected_affix = getattr(app, "selected_affix", None)
    stage2_visible = bool(
        getattr(brush, "stage2_options", ())
        or getattr(scenario, "stage2_options", ())
    )
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
        brush_type_selected=selected_type is not None,
        affix_selected=selected_affix is not None,
        backpack_target=_backpack_target_needed(app),
        board_target=_board_target_ready(app),
        start_visible=bool(scene is not None and scene.region(RegionKind.START_DRAWING) is not None),
        finish_signals=frozenset(finish_signals),
    )


def _finish_signals(app, events: tuple[FastVisionEvent, ...]) -> set[str]:
    signals: set[str] = set()
    scene = getattr(app, "current_scene", None)
    snapshot = getattr(app, "last_fast_snapshot", None)
    # 「入画」按钮在作画界面常驻（每次都能检出，置信 0.9+），它表示「可以进下一张」，
    # 不是「本局已画完」。把它算成 finish_button 会让桌宠一直报「可结算」。
    finish_region = scene.region(RegionKind.FINISH_DRAWING) if scene is not None else None
    if finish_region is not None and not is_enter_painting_region(finish_region):
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

    if evidence.start_visible and scene is not None:
        region = scene.region(RegionKind.START_DRAWING)
        if region is not None:
            point = _capture_rect_point(app, region.rect, 0.5, 0.5)
            if point is not None:
                return TargetProposal(
                    key="start_drawing",
                    kind=TargetKind.START,
                    priority=0,
                    screen_point=point,
                    confidence=max(confidence, min(0.90, float(region.confidence))),
                    reason_short="开始绘制",
                    source_frame_id=evidence.frame_id,
                )

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
                    # 入画按钮是独立的高精度检测，不该被综合质量分（结束态格子本来就难分）拉低。
                    confidence=max(confidence, min(0.90, float(region.confidence))),
                    reason_short="结束绘制",
                    source_frame_id=evidence.frame_id,
                )

    if state is not None and evidence.current_rule_pending:
        brush_target = _brush_target_proposal(app, evidence, confidence)
        if brush_target is not None:
            return brush_target

    if evidence.backpack_target:
        paint = _desired_backpack_paint(app)
        index = _visible_backpack_index(app, paint)
        point = _call_optional(app, "_screen_point_for_backpack_slot", index)
        if index is not None and _target_point_is_safe(app, point):
            return TargetProposal(
                key=f"backpack:{index}:{_paint_hint_label(paint)}",
                kind=TargetKind.BACKPACK,
                priority=30,
                screen_point=(int(point[0]), int(point[1])),
                confidence=confidence,
                reason_short=_backpack_reason(paint),
                source_frame_id=evidence.frame_id,
            )

    if evidence.board_target:
        row, col, rec = _desired_board_placement(app)
        if not _board_target_is_in_effective_scope(state, row, col):
            return None
        point = _call_optional(app, "_screen_point_for_board_cell", row, col)
        if isinstance(row, int) and isinstance(col, int) and _target_point_is_safe(app, point):
            return TargetProposal(
                key=f"board:{row},{col}",
                kind=TargetKind.BOARD,
                priority=40,
                screen_point=(int(point[0]), int(point[1])),
                confidence=confidence,
                reason_short=_recommendation_reason(rec) if rec is not None else "按方案放置",
                source_frame_id=evidence.frame_id,
            )
    return None


def _brush_card_center(index: int, count: int) -> float:
    """卡片中心在面板高度上的比例。

    笔锋面板固定两张卡片，8 张实拍量出来中心在 0.30 / 0.71（卡片带 0.108~0.900），
    比按张数均分的 0.25 / 0.75 更贴。数量不是 2 时退回均分。
    """
    if count <= 1:
        return 0.5
    if count == 2:
        return 0.71 if index > 0 else 0.30
    return (index + 0.5) / count


def _brush_target_proposal(
    app,
    evidence: PhaseEvidence,
    confidence: float,
) -> TargetProposal | None:
    scene = getattr(app, "current_scene", None)
    state = getattr(app, "current_state", None)
    scenario = getattr(app, "brush_scenario", None)
    brush_observation = getattr(scene, "brush_options", None) or getattr(
        app,
        "_last_brush_observation",
        None,
    )
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
        locked_type = getattr(app, "_locked_brush_stage1", None)
        best = next(
            (item for item in recommendations if item.brush_type == locked_type),
            recommendations[0],
        )
        setattr(app, "_locked_brush_stage1", best.brush_type)
        index = options.index(best.brush_type)
        cards = getattr(brush_observation, "card_rects", ())
        point = (_capture_rect_point(app, cards[index], .5, .5) if index < len(cards)
                 else _capture_rect_point(app, panel, 0.55, _brush_card_center(index, len(options))))
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
        locked_affix = getattr(app, "_locked_brush_stage2", None)
        best = next(
            (item for item in recommendations if item.affix == locked_affix),
            recommendations[0],
        )
        setattr(app, "_locked_brush_stage2", best.affix)
        index = affixes.index(best.affix)
        y_ratio = _brush_card_center(index, len(affixes))
        cards = getattr(brush_observation, "card_rects", ())
        point = (_capture_rect_point(app, cards[index], .5, .5) if index < len(cards)
                 else _capture_rect_point(app, panel, 0.58, y_ratio))
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


def _desired_board_placement(app) -> tuple[int | None, int | None, object | None]:
    recommendations = list(getattr(app, "recommendations", ()) or ())
    install_plan = getattr(app, "install_plan", None)
    steps = tuple(getattr(install_plan, "steps", ()) or ())
    if steps:
        step = steps[0]
        matching = next(
            (
                rec
                for rec in recommendations
                if getattr(rec, "row", None) == step.row
                and getattr(rec, "col", None) == step.col
                and getattr(rec, "paint", None) == step.paint
            ),
            None,
        )
        return step.row, step.col, matching
    if recommendations:
        rec = recommendations[0]
        return getattr(rec, "row", None), getattr(rec, "col", None), rec
    return None, None, None


def _board_target_is_in_effective_scope(state, row, col) -> bool:
    if not isinstance(row, int) or not isinstance(col, int):
        return False
    if state is None:
        return True
    rule = getattr(state, "current_rule", None)
    if rule is None:
        return True
    if current_rule_is_pending(state):
        return True
    if getattr(state, "current_anchor", None) is None:
        # 没有锚点算不出几何，只能信画面金框；连金框都没有就别挡建议。
        observed = tuple(getattr(state, "observed_scope_cells", ()) or ())
        return (row, col) in observed if observed else True
    # 有锚点就一律走纯几何，和 game_model 的调色/笔触范围同口径。
    # 金框不亮虚空/失色格，而扩散能把它们转成空白地块 —— 那些格子转完就在范围内，
    # 按金框过滤会把本该有效的落点从建议里剔掉。
    return (row, col) in current_scope_cells(state, include_anchor=False)


def _high_confidence_start_visible(app) -> bool:
    scene = getattr(app, "current_scene", None)
    if scene is None:
        return False
    region = scene.region(RegionKind.START_DRAWING)
    board = getattr(scene, "board", None)
    return (
        region is not None
        and float(getattr(region, "confidence", 0.0)) >= 0.90
        and float(getattr(board, "score", 0.0)) >= 0.68
    )


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
