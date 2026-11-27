from __future__ import annotations

import hashlib
from collections.abc import Mapping
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
    RegionKind,
    SceneObservation,
    TileState,
)

#: 棋盘边长。``PuzzleState.tiles`` 是行主序的 36 格，坐标靠它换算。
BOARD_SIZE = 6


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
    #: 颜料自己的等级加成（`game_model.Paint.bonus_level`）。
    #: **视觉上读不到**：棋盘只渲染最终等级，加成完全不可见。
    #: 这个字段由上层从规则推演出的模型状态回填，用于记录/回放两套数的区别。
    bonus_level: int = 0
    #: 「与自身相同的颜料额外提升 x 级」这类光环给这一格带来的等级
    #: （`game_model.Tile.aura_level`）。同样视觉读不到，只能模型回填。
    aura_level: int = 0


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
    is_selected: bool = False
    selection_confidence: float = 0.0


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
    #: 面板「调色效果」正文。None = 没读到这一行。
    effect_text: str | None = None
    brush_effect_text: str | None = None


@dataclass(frozen=True)
class FastBrush:
    panel_rect: Rect
    stage1_options: tuple[str, ...]
    chosen_stage1: str | None
    stage2_options: tuple[str, ...]
    confidence: float
    card_rects: tuple[Rect, ...] = ()
    reading_pending: bool = False


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
    #: 这一帧背包是按哪个**调色目标**排的（``monster`` / ``gameplay``，读不出就是空）。
    #: 背包一页只有 3 排 15 格，整包 30 枚要靠切两次排序各记一页；没有这个字段
    #: 两页的读数会互相覆盖，第二页一来就把第一页挤掉。
    backpack_page: str = ""
    current_anchor: tuple[int, int] | None = None


def backpack_page_of(scene: SceneObservation | None) -> str:
    """这一帧背包属于哪一页：按**当前初始颜料的目标**分页。

    怪物初始颜料 → 背包按怪物排序 → ``monster`` 页；玩法 → ``gameplay`` 页。
    两页各 15 格、合计最多 30 格（同一枚颜料两页都在是正常的）。

    **目标读不出来时必须返回空串**，不能造 ``unknown`` 页：选色面板刚弹出、
    ``current_paint`` 还认不出时，造新页会往里写满 15 格空读数。
    """
    paint = getattr(scene, "current_paint", None) if scene is not None else None
    if paint is None:
        return ""
    target = getattr(paint, "target_object", None)
    value = getattr(target, "value", None)
    if value:
        return str(value)
    category = getattr(paint, "category", None)
    return str(getattr(category, "value", None) or "")


def snapshot_from_scene(
    scene: SceneObservation | None,
    model: object | None = None,
    *,
    backpack_page: str = "",
) -> FastVisionSnapshot | None:
    """把一帧观测压成快照。

    ``model`` 是规则推演出的模型状态（``game_model.PuzzleState``）。传进来时，
    每一格的 ``bonus_level`` / ``aura_level`` 会从模型回填——这两项**视觉上完全读不到**，
    棋盘只渲染最终等级，空地上叠的加成更是连像素都没有。
    """
    if scene is None:
        return None
    buffs = _tile_buffs(model)
    tiles = _fast_tiles(scene.board, buffs)
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
                f"{paint.slot_index},{paint.category},{paint.level},{paint.subtype},{int(paint.is_treasure)},{int(paint.is_selected)},{paint.fingerprint}"
                for paint in backpack
            )
        ),
        current=current,
        brush=brush,
        # 「入画」按钮在作画界面常驻，不能当成「完成绘制」区域往外报：
        # ``_finish_visible`` / 录制器 / 桌宠都只看 kind，看到它就以为本局画完了。
        regions={
            region.kind.value: region.rect
            for region in scene.regions
            if not (region.kind == RegionKind.FINISH_DRAWING and region.is_enter_painting)
        },
        reliable=not reasons,
        reasons=reasons,
        backpack_page=backpack_page or backpack_page_of(scene),
        current_anchor=next(((t.row, t.col) for t in scene.board.tiles
                             if t.is_current_anchor), None),
    )


def _fast_tiles(
    board: BoardObservation,
    buffs: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[FastTile, ...]:
    buffs = buffs or {}
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
            bonus_level=buffs.get((tile.row, tile.col), (0, 0))[0],
            aura_level=buffs.get((tile.row, tile.col), (0, 0))[1],
        )
        for tile in board.tiles
    )


def _tile_buffs(model: object | None) -> dict[tuple[int, int], tuple[int, int]]:
    """从模型状态里取出每格的 ``(bonus_level, aura_level)``。

    认三种形状，方便调用方（GUI、录制器、测试）各取所需：

    - ``None`` → 全 0；
    - 有 ``.tiles`` 的对象（``game_model.PuzzleState``）→ 按行主序换算坐标；
    - 直接给 ``{(row, col): (bonus, aura)}`` 或 ``{(row, col): tile}``。

    ``bonus_level`` 读的是**颜料**身上的加成（``Paint.bonus_level``，2026-11-23
    改成颜料加成）：地块上没有颜料时那一格就是 0，与模型口径一致。
    """
    if model is None:
        return {}
    model_tiles = getattr(model, "tiles", None)
    if model_tiles is not None:
        buffs: dict[tuple[int, int], tuple[int, int]] = {}
        for index, tile in enumerate(model_tiles):
            paint = getattr(tile, "paint", None)
            buffs[divmod(index, BOARD_SIZE)] = (
                max(0, int(getattr(paint, "bonus_level", 0) or 0)) if paint is not None else 0,
                max(0, int(getattr(tile, "aura_level", 0) or 0)),
            )
        return buffs
    if isinstance(model, Mapping):
        buffs = {}
        for key, value in model.items():
            if isinstance(value, tuple):
                bonus, aura = value
            else:
                bonus = getattr(value, "bonus_level", 0)
                aura = getattr(value, "aura_level", 0)
            buffs[(int(key[0]), int(key[1]))] = (max(0, int(bonus or 0)), max(0, int(aura or 0)))
        return buffs
    return {}


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
            is_selected=paint.is_selected,
            selection_confidence=round(float(paint.selection_confidence), 3),
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
        # 调色规则与附加笔触正文分别传入录制快照。
        effect_text=getattr(current, "effect_text", None),
        brush_effect_text=getattr(current, "brush_effect_text", None),
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
        card_rects=brush.card_rects,
        reading_pending=brush.reading_pending,
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
