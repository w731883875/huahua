from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .models import (
    BackpackObservation,
    BoardObservation,
    BrushOptionsObservation,
    CurrentPaintObservation,
    PaintCategory,
    PaintDirection,
    PaintObject,
    Rect,
    SceneObservation,
    TileState,
)


@dataclass(frozen=True)
class FastTile:
    row: int
    col: int
    state: str
    rect: Rect
    paint_category: str
    paint_level: int | None
    paint_subtype: str | None
    paint_is_treasure: bool
    confidence: float


@dataclass(frozen=True)
class FastBackpackPaint:
    slot_index: int
    rect: Rect
    category: str
    level: int
    subtype: str | None
    is_treasure: bool
    fingerprint: str
    confidence: float


@dataclass(frozen=True)
class FastCurrentPaint:
    panel_rect: Rect
    icon_rect: Rect
    text_rect: Rect
    category: str
    level: int | None
    subtype: str | None
    is_treasure: bool
    direction: str
    target_object: str
    upgrade_amount: int | None
    match_current: int | None
    match_required: int | None
    confidence: float


@dataclass(frozen=True)
class FastBrush:
    panel_rect: Rect
    stage1_options: tuple[str, ...]
    chosen_stage1: str | None
    stage2_options: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class FastVisionSnapshot:
    board_rect: Rect
    board_score: float
    tiles: tuple[FastTile, ...]
    board_signature: str
    paint_signature: str
    terrain_counts: dict[str, int]
    backpack_rect: Rect | None
    backpack: tuple[FastBackpackPaint, ...]
    backpack_signature: str
    current: FastCurrentPaint | None
    brush: FastBrush | None
    regions: dict[str, Rect]
    reliable: bool
    reasons: tuple[str, ...]


def snapshot_from_scene(scene: SceneObservation | None) -> FastVisionSnapshot | None:
    if scene is None:
        return None
    tiles = _fast_tiles(scene.board)
    terrain_counts = _terrain_counts(tiles)
    board_signature = _stable_hash(
        "|".join(
            f"{tile.row},{tile.col},{tile.state},{tile.paint_category},{tile.paint_level},{tile.paint_subtype},{int(tile.paint_is_treasure)}"
            for tile in tiles
        )
    )
    paint_signature = _stable_hash(
        "|".join(
            f"{tile.row},{tile.col},{tile.paint_category},{tile.paint_level},{tile.paint_subtype},{int(tile.paint_is_treasure)}"
            for tile in tiles
            if tile.state == TileState.PAINT.value
        )
    )
    backpack = _fast_backpack(scene.backpack)
    current = _fast_current(scene.current_paint)
    brush = _fast_brush(scene.brush_options)
    reasons = _snapshot_reasons(scene.board, backpack, current, brush)
    return FastVisionSnapshot(
        board_rect=scene.board.rect,
        board_score=round(float(scene.board.score), 3),
        tiles=tiles,
        board_signature=board_signature,
        paint_signature=paint_signature,
        terrain_counts=terrain_counts,
        backpack_rect=scene.backpack.panel_rect if scene.backpack is not None else None,
        backpack=backpack,
        backpack_signature=_stable_hash(
            "|".join(
                f"{paint.slot_index},{paint.category},{paint.level},{paint.subtype},{int(paint.is_treasure)},{paint.fingerprint}"
                for paint in backpack
            )
        ),
        current=current,
        brush=brush,
        regions={region.kind.value: region.rect for region in scene.regions},
        reliable=not reasons,
        reasons=reasons,
    )


def _fast_tiles(board: BoardObservation) -> tuple[FastTile, ...]:
    return tuple(
        FastTile(
            row=tile.row,
            col=tile.col,
            state=tile.state.value,
            rect=tile.rect,
            paint_category=tile.paint_category.value,
            paint_level=tile.paint_level,
            paint_subtype=tile.paint_subtype,
            paint_is_treasure=tile.paint_is_treasure,
            confidence=round(float(tile.confidence), 3),
        )
        for tile in board.tiles
    )


def _fast_backpack(backpack: BackpackObservation | None) -> tuple[FastBackpackPaint, ...]:
    if backpack is None:
        return ()
    return tuple(
        FastBackpackPaint(
            slot_index=index,
            rect=paint.slot_rect,
            category=paint.category.value,
            level=int(paint.level),
            subtype=paint.subtype,
            is_treasure=paint.is_treasure,
            fingerprint=paint.fingerprint,
            confidence=round(float(paint.category_confidence) * 0.55 + float(paint.level_confidence) * 0.45, 3),
        )
        for index, paint in enumerate(backpack.paints)
    )


def _fast_current(current: CurrentPaintObservation | None) -> FastCurrentPaint | None:
    if current is None:
        return None
    confidence_parts = [
        float(current.category_confidence),
        float(current.level_confidence),
        max(current.keyword_scores.values(), default=0.0),
    ]
    return FastCurrentPaint(
        panel_rect=current.panel_rect,
        icon_rect=current.icon_rect,
        text_rect=current.text_rect,
        category=current.category.value,
        level=current.level,
        subtype=current.subtype,
        is_treasure=current.is_treasure,
        direction=current.direction.value,
        target_object=current.target_object.value,
        upgrade_amount=current.upgrade_amount,
        match_current=current.match_current,
        match_required=current.match_required,
        confidence=round(sum(confidence_parts) / len(confidence_parts), 3),
    )


def _fast_brush(brush: BrushOptionsObservation | None) -> FastBrush | None:
    if brush is None:
        return None
    return FastBrush(
        panel_rect=brush.panel_rect,
        stage1_options=brush.stage1_options,
        chosen_stage1=brush.chosen_stage1,
        stage2_options=brush.stage2_options,
        confidence=round(float(brush.confidence), 3),
    )


def _terrain_counts(tiles: tuple[FastTile, ...]) -> dict[str, int]:
    counts = {state.value: 0 for state in TileState}
    for tile in tiles:
        counts[tile.state] = counts.get(tile.state, 0) + 1
    return counts


def _snapshot_reasons(
    board: BoardObservation,
    backpack: tuple[FastBackpackPaint, ...],
    current: FastCurrentPaint | None,
    brush: FastBrush | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if float(board.score) < 0.5:
        reasons.append(f"board_score={board.score:.2f}")
    unknown_tiles = sum(1 for tile in board.tiles if tile.state == TileState.UNKNOWN)
    if unknown_tiles:
        reasons.append(f"unknown_tiles={unknown_tiles}")
    if current is not None:
        if current.category == PaintCategory.UNKNOWN.value:
            reasons.append("current_category_unknown")
        if current.direction == PaintDirection.UNKNOWN.value:
            reasons.append("current_direction_unknown")
        if current.target_object == PaintObject.UNKNOWN.value:
            reasons.append("current_target_unknown")
    if brush is not None and brush.confidence < 0.35:
        reasons.append(f"brush_confidence={brush.confidence:.2f}")
    if backpack and min(paint.confidence for paint in backpack) < 0.25:
        reasons.append("backpack_low_confidence")
    return tuple(reasons)


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:16]
