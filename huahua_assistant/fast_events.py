from __future__ import annotations

from dataclasses import dataclass

from .fast_vision import FastBrush, FastCurrentPaint, FastVisionSnapshot
from .models import RegionKind, TileState


@dataclass(frozen=True)
class FastVisionEvent:
    kind: str
    confidence: float
    detail: dict


def detect_fast_events(
    previous: FastVisionSnapshot | None,
    current: FastVisionSnapshot | None,
) -> tuple[FastVisionEvent, ...]:
    if current is None:
        return ()
    events: list[FastVisionEvent] = []
    if previous is None:
        events.append(
            FastVisionEvent(
                kind="snapshot_ready",
                confidence=_snapshot_confidence(current),
                detail=_snapshot_detail(current),
            )
        )
        events.extend(_single_snapshot_events(current))
        return tuple(events)

    if previous.board_signature != current.board_signature:
        events.append(_board_changed(previous, current))
    if previous.paint_signature != current.paint_signature:
        events.append(_paint_changed(previous, current))
    if previous.backpack_signature != current.backpack_signature:
        events.append(_backpack_changed(previous, current))
    current_changed = _current_key(previous.current) != _current_key(current.current)
    if current_changed:
        events.append(_current_changed(previous.current, current.current))
    brush_changed = _brush_key(previous.brush) != _brush_key(current.brush)
    if brush_changed:
        events.append(_brush_changed(previous.brush, current.brush))
    events.extend(_single_snapshot_events(current, previous=previous))
    return tuple(events)


def _single_snapshot_events(
    current: FastVisionSnapshot,
    *,
    previous: FastVisionSnapshot | None = None,
) -> tuple[FastVisionEvent, ...]:
    events: list[FastVisionEvent] = []
    if _finish_visible(current) and (previous is None or not _finish_visible(previous)):
        events.append(
            FastVisionEvent(
                kind="finish_candidate",
                confidence=0.8 if current.reliable else 0.55,
                detail={
                    "reason": "finish_region",
                    "board_signature": current.board_signature,
                    "paint_signature": current.paint_signature,
                },
            )
        )
    if _board_has_no_blank(current) and (previous is None or not _board_has_no_blank(previous)):
        events.append(
            FastVisionEvent(
                kind="finish_candidate",
                confidence=0.7 if current.reliable else 0.5,
                detail={
                    "reason": "no_blank_tiles",
                    "terrain_counts": current.terrain_counts,
                    "board_signature": current.board_signature,
                },
            )
        )
    return tuple(events)


def _board_changed(previous: FastVisionSnapshot, current: FastVisionSnapshot) -> FastVisionEvent:
    changed_cells = []
    previous_by_cell = {(tile.row, tile.col): tile for tile in previous.tiles}
    for tile in current.tiles:
        old = previous_by_cell.get((tile.row, tile.col))
        if old is None:
            continue
        if (
            old.state,
            old.paint_category,
            old.paint_level,
            old.paint_subtype,
            old.paint_is_treasure,
        ) != (
            tile.state,
            tile.paint_category,
            tile.paint_level,
            tile.paint_subtype,
            tile.paint_is_treasure,
        ):
            changed_cells.append(
                {
                    "row": tile.row,
                    "col": tile.col,
                    "from": _tile_key(old),
                    "to": _tile_key(tile),
                }
            )
    return FastVisionEvent(
        kind="board_changed",
        confidence=_snapshot_confidence(current),
        detail={
            "changed_cells": changed_cells[:12],
            "changed_count": len(changed_cells),
            "terrain_counts": current.terrain_counts,
            "board_signature": current.board_signature,
        },
    )


def _paint_changed(previous: FastVisionSnapshot, current: FastVisionSnapshot) -> FastVisionEvent:
    return FastVisionEvent(
        kind="paint_changed",
        confidence=_snapshot_confidence(current),
        detail={
            "paint_signature": current.paint_signature,
            "previous_paint_signature": previous.paint_signature,
            "paint_tiles": current.terrain_counts.get(TileState.PAINT.value, 0),
        },
    )


def _backpack_changed(previous: FastVisionSnapshot, current: FastVisionSnapshot) -> FastVisionEvent:
    previous_selected = tuple(paint.slot_index for paint in previous.backpack if paint.is_selected)
    current_selected = tuple(paint.slot_index for paint in current.backpack if paint.is_selected)
    if previous_selected != current_selected:
        kind = "selection_state_changed"
        confidence = min(0.9, _snapshot_confidence(current))
    elif (
        previous.backpack
        and current.backpack
        and len(previous.backpack) == len(current.backpack)
        and _backpack_logical_key(previous) == _backpack_logical_key(current)
        and previous.board_signature == current.board_signature
        and previous.paint_signature == current.paint_signature
    ):
        kind = "backpack_visual_changed"
        confidence = min(0.55, _snapshot_confidence(current))
    elif previous.backpack and not current.backpack:
        kind = "backpack_disappeared"
        confidence = _snapshot_confidence(current)
    elif current.backpack and not previous.backpack:
        kind = "backpack_appeared"
        confidence = _snapshot_confidence(current)
    else:
        kind = "backpack_changed"
        confidence = _snapshot_confidence(current)
    return FastVisionEvent(
        kind=kind,
        confidence=confidence,
        detail={
            "count": len(current.backpack),
            "previous_count": len(previous.backpack),
            "backpack_signature": current.backpack_signature,
            "previous_backpack_signature": previous.backpack_signature,
            "selected_slots": current_selected,
            "previous_selected_slots": previous_selected,
        },
    )


def _current_changed(previous: FastCurrentPaint | None, current: FastCurrentPaint | None) -> FastVisionEvent:
    if previous is None and current is not None:
        kind = "current_paint_appeared"
    elif previous is not None and current is None:
        kind = "current_paint_disappeared"
    else:
        kind = "current_paint_changed"
    return FastVisionEvent(
        kind=kind,
        confidence=current.confidence if current is not None else 0.6,
        detail={"previous": _current_key(previous), "current": _current_key(current)},
    )


def _brush_changed(previous: FastBrush | None, current: FastBrush | None) -> FastVisionEvent:
    if previous is None and current is not None:
        kind = "brush_stage_appeared"
    elif previous is not None and current is None:
        kind = "brush_stage_disappeared"
    else:
        kind = "brush_stage_changed"
    return FastVisionEvent(
        kind=kind,
        confidence=current.confidence if current is not None else 0.6,
        detail={"previous": _brush_key(previous), "current": _brush_key(current)},
    )


def _snapshot_detail(snapshot: FastVisionSnapshot) -> dict:
    return {
        "board_signature": snapshot.board_signature,
        "paint_signature": snapshot.paint_signature,
        "backpack_signature": snapshot.backpack_signature,
        "terrain_counts": snapshot.terrain_counts,
        "reliable": snapshot.reliable,
        "reasons": snapshot.reasons,
    }


def _snapshot_confidence(snapshot: FastVisionSnapshot) -> float:
    if snapshot.reliable:
        return max(0.0, min(1.0, float(snapshot.board_score)))
    return max(0.25, min(0.65, float(snapshot.board_score)))


def _tile_key(tile) -> dict:
    return {
        "state": tile.state,
        "category": tile.paint_category,
        "level": tile.paint_level,
        "subtype": tile.paint_subtype,
        "treasure": tile.paint_is_treasure,
    }


def _current_key(current: FastCurrentPaint | None) -> tuple | None:
    if current is None:
        return None
    return (
        current.category,
        current.level,
        current.subtype,
        current.is_treasure,
        current.direction,
        current.target_object,
        current.upgrade_amount,
        current.match_current,
        current.match_required,
    )


def _brush_key(brush: FastBrush | None) -> tuple | None:
    if brush is None:
        return None
    return (
        brush.stage1_options,
        brush.chosen_stage1,
        brush.stage2_options,
    )


def _backpack_logical_key(snapshot: FastVisionSnapshot) -> tuple:
    return tuple(
        (
            paint.slot_index,
            paint.category,
            paint.level,
            paint.subtype,
            paint.is_treasure,
        )
        for paint in snapshot.backpack
    )


def _finish_visible(snapshot: FastVisionSnapshot) -> bool:
    return RegionKind.FINISH_DRAWING.value in snapshot.regions


def _board_has_no_blank(snapshot: FastVisionSnapshot) -> bool:
    return int(snapshot.terrain_counts.get(TileState.BLANK.value, 0) or 0) == 0
