from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import Enum

from .models import (
    BoardObservation,
    CurrentPaintObservation,
    PaintCategory,
    PaintDirection,
    PaintObject,
    TileState,
)


BOARD_SIZE = 6
MAX_LEVEL = 7


class BrushType(str, Enum):
    TRANSFORM = "transform"  # 转锋
    STACK = "stack"          # 堆砌
    SPREAD = "spread"        # 扩散


BRUSH_TYPE_ORDER = (BrushType.TRANSFORM, BrushType.STACK, BrushType.SPREAD)


class AffixEffect(str, Enum):
    CHANGE_DIRECTION = "change_direction"
    CHANGE_TARGET_OBJECT = "change_target_object"
    MATCH_REQUIRED_DELTA = "match_required_delta"
    RANDOMIZE_CURRENT_LEVEL = "randomize_current_level"
    RANDOMIZE_CURRENT_DIRECTION = "randomize_current_direction"
    RANDOMIZE_CURRENT_CONTENT = "randomize_current_content"
    ADD_RANDOM_SCOPE = "add_random_scope"
    REROLL_CURRENT_RULE = "reroll_current_rule"
    UPGRADE_IN_SCOPE = "upgrade_in_scope"
    UPGRADE_SCOPE_SAME_PAINT = "upgrade_scope_same_paint"
    UPGRADE_SELF = "upgrade_self"
    UPGRADE_GLOBAL_CATEGORY = "upgrade_global_category"
    UPGRADE_GLOBAL_SAME_PAINT = "upgrade_global_same_paint"
    COPY_SELF_TO_SCOPE = "copy_self_to_scope"
    RANDOMIZE_SCOPE_CATEGORY = "randomize_scope_category"
    RANDOMIZE_SCOPE_ANY = "randomize_scope_any"
    CLEAR_TILE_IN_SCOPE = "clear_tile_in_scope"
    CLEAR_RANDOM_TILE_GLOBAL = "clear_random_tile_global"
    ALLOW_PLACE_ON_BLOCKED = "allow_place_on_blocked"
    ALLOW_PLACE_ON_VOID_LOSE_SELF = "allow_place_on_void_lose_self"
    ALLOW_PLACE_ON_PAINT_LOSE_SELF = "allow_place_on_paint_lose_self"


class SpecialAction(str, Enum):
    FORCE_COMPLETE = "force_complete"
    REFRESH_STAGE1 = "refresh_stage1"
    REFRESH_STAGE2 = "refresh_stage2"


@dataclass(frozen=True)
class Paint:
    category: PaintCategory
    level: int
    subtype: str | None = None
    is_treasure: bool = False


@dataclass(frozen=True)
class Tile:
    state: TileState
    paint: Paint | None = None
    buff_applied_id: int | None = None
    affix_applied_id: int | None = None


@dataclass(frozen=True)
class CurrentPaintRule:
    paint: Paint
    direction: PaintDirection
    target_object: PaintObject
    match_required: int
    upgrade_amount: int
    match_current: int = 0
    extra_directions: tuple[PaintDirection, ...] = ()


@dataclass(frozen=True)
class AffixRule:
    brush_type: BrushType
    name: str
    effect: AffixEffect
    direction: PaintDirection | None = None
    target_object: PaintObject | None = None
    delta: int = 0
    amount: int = 0
    count: int | None = None
    persistent_until_match: bool = False
    nearest: bool = False


@dataclass(frozen=True)
class PuzzleState:
    tiles: tuple[Tile, ...]
    current_rule: CurrentPaintRule | None = None
    active_affix: AffixRule | None = None
    backpack: tuple[Paint, ...] = ()
    current_anchor: tuple[int, int] | None = None
    current_buff_id: int = 1
    current_rule_pending: bool = False

    def tile(self, row: int, col: int) -> Tile:
        return self.tiles[row * BOARD_SIZE + col]


@dataclass(frozen=True)
class PlacementRecommendation:
    row: int
    col: int
    paint: Paint | None
    score: float
    completes_match: bool
    affected: tuple[tuple[int, int], ...]
    reason: str


@dataclass(frozen=True)
class InstallStep:
    row: int
    col: int
    paint: Paint


@dataclass(frozen=True)
class InstallPlan:
    steps: tuple[InstallStep, ...]
    completes_match: bool
    reason: str


@dataclass(frozen=True)
class InstallResult:
    state: PuzzleState
    success: bool
    completed_match: bool
    message: str


@dataclass(frozen=True)
class PlacementOutcome:
    match_count: int
    completes_match: bool
    current_next_level: int
    affected_paints: int
    treasure_to_7: int
    gameplay_to_7: int
    monster_to_7: int
    any_to_7: int
    treasure_upgrades: int


@dataclass(frozen=True)
class BrushRecommendation:
    brush_type: BrushType
    score: float
    reason: str


@dataclass(frozen=True)
class AffixRecommendation:
    affix: AffixRule
    score: float
    reason: str


@dataclass(frozen=True)
class BrushChoiceScenario:
    stage1_options: tuple[BrushType, ...]
    chosen_stage1: BrushType
    stage2_options: tuple[AffixRule, ...]


@dataclass(frozen=True)
class ResourceState:
    force_remaining: int = 3
    force_used: int = 0
    refresh_available: bool = False
    refresh_remaining: int = 0


@dataclass(frozen=True)
class SpecialActionRecommendation:
    action: SpecialAction
    score: float
    recommended: bool
    reason: str


@dataclass(frozen=True)
class ForceCompleteResult:
    state: PuzzleState
    success: bool
    destroyed_cells: tuple[tuple[int, int], ...]
    message: str


@dataclass(frozen=True)
class AnchorPlaceResult:
    state: PuzzleState
    success: bool
    message: str


@dataclass(frozen=True)
class AffixApplyResult:
    state: PuzzleState
    success: bool
    message: str


def normalize_brush_options(options: tuple[BrushType, ...] | list[BrushType]) -> tuple[BrushType, ...]:
    """Stage-1 UI order appears stable: 转锋 > 堆砌 > 扩散."""
    seen = set(options)
    return tuple(brush_type for brush_type in BRUSH_TYPE_ORDER if brush_type in seen)


def is_valid_brush_option_order(options: tuple[BrushType, ...] | list[BrushType]) -> bool:
    return tuple(options) == normalize_brush_options(options)


def sort_backpack_for_rule(backpack: tuple[Paint, ...] | list[Paint], rule: CurrentPaintRule | None) -> tuple[Paint, ...]:
    target = rule.target_object if rule is not None else PaintObject.UNKNOWN

    def key(paint: Paint) -> tuple[int, int, int, str]:
        target_match = 1 if _paint_matches_target(paint, target) else 0
        treasure = 1 if paint.is_treasure else 0
        subtype = paint.subtype or ""
        return (target_match, paint.level, treasure, subtype)

    return tuple(sorted(backpack, key=key, reverse=True))


def current_scope_cells(state: PuzzleState, *, include_anchor: bool = True) -> tuple[tuple[int, int], ...]:
    if state.current_rule is None or state.current_anchor is None:
        return ()
    row, col = state.current_anchor
    cells = _rule_scope_cells_for_anchor(row, col, state.current_rule)
    if include_anchor:
        return cells
    return tuple(cell for cell in cells if cell != state.current_anchor)


def current_rule_is_pending(state: PuzzleState) -> bool:
    return state.current_rule is not None and state.current_rule_pending


def state_from_observation(
    board: BoardObservation,
    current_paint: CurrentPaintObservation | None = None,
    backpack: tuple[Paint, ...] | list[Paint] = (),
) -> PuzzleState:
    rule = rule_from_observation(current_paint)
    anchor: tuple[int, int] | None = None
    paint_cells = [tile for tile in board.tiles if tile.state == TileState.PAINT]
    if rule is not None and len(paint_cells) == 1:
        anchor = (paint_cells[0].row, paint_cells[0].col)

    tiles = []
    for tile in board.tiles:
        paint = rule.paint if anchor == (tile.row, tile.col) and rule is not None else None
        tiles.append(Tile(state=tile.state, paint=paint))
    return PuzzleState(
        tiles=tuple(tiles),
        current_rule=rule,
        backpack=sort_backpack_for_rule(tuple(backpack), rule),
        current_anchor=anchor,
    )


def rule_from_observation(current_paint: CurrentPaintObservation | None) -> CurrentPaintRule | None:
    if current_paint is None:
        return None

    level = current_paint.level or 3
    match_required, upgrade_amount = empirical_defaults_for_level(level)
    if current_paint.match_required is not None:
        match_required = current_paint.match_required
    if current_paint.upgrade_amount is not None:
        upgrade_amount = current_paint.upgrade_amount

    return CurrentPaintRule(
        paint=Paint(
            category=current_paint.category,
            level=level,
            subtype=current_paint.subtype,
            is_treasure=current_paint.is_treasure,
        ),
        direction=current_paint.direction,
        target_object=current_paint.target_object,
        match_required=max(1, match_required),
        upgrade_amount=max(1, upgrade_amount),
        match_current=current_paint.match_current or 0,
    )


def empirical_defaults_for_level(level: int) -> tuple[int, int]:
    """Observed fallback only; OCR/real recognition should override it."""
    if level >= 5:
        return 4, 2
    if level == 4:
        return 4, 1
    return 3, 1


def random_state(seed: int | None = None) -> PuzzleState:
    rng = random.Random(seed)
    blank_count = rng.randint(20, 27)
    blocked_count = rng.randint(2, 8)
    faded_count = 0
    states = (
        [TileState.BLANK] * blank_count
        + [TileState.BLOCKED] * blocked_count
        + [TileState.FADED] * faded_count
    )
    states += [TileState.VOID] * max(0, BOARD_SIZE * BOARD_SIZE - len(states))
    states = states[: BOARD_SIZE * BOARD_SIZE]
    rng.shuffle(states)

    tiles = [Tile(state=state) for state in states]
    current_rule, anchor = _random_opening_rule_and_anchor(rng, tiles)
    anchor_index = anchor[0] * BOARD_SIZE + anchor[1]
    tiles[anchor_index] = Tile(state=TileState.PAINT, paint=current_rule.paint)
    backpack = sort_backpack_for_rule(_random_backpack_for_rule(rng, current_rule), current_rule)
    return PuzzleState(
        tiles=tuple(tiles),
        current_rule=current_rule,
        backpack=backpack,
        current_anchor=anchor,
    )


def random_current_rule(rng: random.Random | None = None) -> CurrentPaintRule:
    rng = rng or random.Random()
    paint = _random_paint(rng)
    match_required, upgrade_amount = empirical_defaults_for_level(paint.level)
    return CurrentPaintRule(
        paint=paint,
        direction=rng.choice(
            [
                PaintDirection.HORIZONTAL,
                PaintDirection.VERTICAL,
                PaintDirection.DIAGONAL,
                PaintDirection.ADJACENT,
            ]
        ),
        target_object=rng.choice([PaintObject.MONSTER, PaintObject.GAMEPLAY, PaintObject.ALL]),
        match_required=match_required,
        upgrade_amount=upgrade_amount,
    )


def _random_opening_rule_and_anchor(rng: random.Random, tiles: list[Tile]) -> tuple[CurrentPaintRule, tuple[int, int]]:
    blank_cells = [(index // BOARD_SIZE, index % BOARD_SIZE) for index, tile in enumerate(tiles) if tile.state == TileState.BLANK]
    for _attempt in range(200):
        rule = random_current_rule(rng)
        if rng.random() < 0.70:
            paint = Paint(
                category=rule.paint.category,
                level=3,
                subtype=rule.paint.subtype,
                is_treasure=rule.paint.is_treasure,
            )
            match_required, upgrade_amount = empirical_defaults_for_level(3)
            rule = CurrentPaintRule(
                paint=paint,
                direction=rule.direction,
                target_object=rng.choice([PaintObject.MONSTER, PaintObject.GAMEPLAY]),
                match_required=match_required,
                upgrade_amount=upgrade_amount,
            )
        rng.shuffle(blank_cells)
        for row, col in blank_cells:
            scope = tuple(cell for cell in _rule_scope_cells_for_anchor(row, col, rule) if cell != (row, col))
            placeable = sum(1 for r, c in scope if tiles[r * BOARD_SIZE + c].state == TileState.BLANK)
            if placeable >= rule.match_required:
                return rule, (row, col)

    fallback_rule = CurrentPaintRule(
        paint=_random_paint(rng),
        direction=PaintDirection.HORIZONTAL,
        target_object=PaintObject.ALL,
        match_required=3,
        upgrade_amount=1,
    )
    return fallback_rule, blank_cells[0] if blank_cells else (0, 0)


def _random_backpack_for_rule(rng: random.Random, rule: CurrentPaintRule) -> tuple[Paint, ...]:
    size = 30
    levels = _random_backpack_levels(rng, size)
    categories = _random_backpack_categories(rng, size, rule)
    backpack = [
        _random_paint_with_category_level(rng, category, level)
        for category, level in zip(categories, levels)
    ]
    rng.shuffle(backpack)
    return tuple(backpack)


def _random_target_paint(rng: random.Random, target: PaintObject) -> Paint:
    if target == PaintObject.MONSTER:
        category = PaintCategory.MONSTER
    elif target == PaintObject.GAMEPLAY:
        category = PaintCategory.GAMEPLAY
    else:
        category = rng.choice([PaintCategory.MONSTER, PaintCategory.GAMEPLAY])
    subtype = _random_subtype(rng, category)
    level = rng.choices([3, 4, 5, 6], weights=[45, 30, 18, 7], k=1)[0]
    return Paint(category=category, level=level, subtype=subtype, is_treasure=subtype == "宝箱")


def _random_backpack_levels(rng: random.Random, size: int) -> list[int]:
    level5_count = rng.randint(0, min(1, size))
    level4_count = rng.randint(0, min(2, size - level5_count))
    levels = [5] * level5_count + [4] * level4_count
    while len(levels) < size:
        levels.append(rng.choices([1, 2, 3], weights=[36, 34, 30], k=1)[0])
    rng.shuffle(levels)
    return levels


def _random_backpack_categories(rng: random.Random, size: int, rule: CurrentPaintRule) -> list[PaintCategory]:
    categories: list[PaintCategory] = []
    target_category = _target_object_to_category(rule.target_object)
    if target_category is not None:
        categories.extend([target_category] * min(rule.match_required, size))
    while len(categories) < size:
        monster_count = sum(1 for category in categories if category == PaintCategory.MONSTER)
        gameplay_count = len(categories) - monster_count
        if monster_count < gameplay_count:
            categories.append(PaintCategory.MONSTER)
        elif gameplay_count < monster_count:
            categories.append(PaintCategory.GAMEPLAY)
        else:
            categories.append(rng.choice([PaintCategory.MONSTER, PaintCategory.GAMEPLAY]))
    rng.shuffle(categories)
    return categories


def _target_object_to_category(target: PaintObject) -> PaintCategory | None:
    if target == PaintObject.MONSTER:
        return PaintCategory.MONSTER
    if target == PaintObject.GAMEPLAY:
        return PaintCategory.GAMEPLAY
    return None


def _random_paint_with_category_level(rng: random.Random, category: PaintCategory, level: int) -> Paint:
    subtype = _random_subtype(rng, category)
    return Paint(category=category, level=level, subtype=subtype, is_treasure=subtype == "宝箱")


def recommend_placements(state: PuzzleState, limit: int = 5) -> list[PlacementRecommendation]:
    rule = state.current_rule
    if rule is None:
        return []
    if current_rule_is_pending(state):
        return _recommend_current_anchor_positions(state, rule, limit)

    candidate_paints = _candidate_backpack_paints(state, rule)
    if not candidate_paints:
        return []

    scoped_cells = _placement_scope_cells(state, rule)
    recommendations: list[PlacementRecommendation] = []
    for paint in candidate_paints:
        for row, col in scoped_cells:
            tile = state.tile(row, col)
            if not _is_placeable(tile, state.active_affix):
                continue
            outcome = _placement_outcome(state, rule, row, col, scoped_cells, paint)
            score = _score_candidate(state, rule, row, col, scoped_cells, outcome, paint)
            if score <= 0:
                continue
            recommendations.append(
                PlacementRecommendation(
                    row=row,
                    col=col,
                    paint=paint,
                    score=score,
                    completes_match=outcome.completes_match,
                    affected=scoped_cells,
                    reason=_reason(rule, outcome, paint),
                )
            )

    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations[:limit]


def place_current_rule_anchor(state: PuzzleState, row: int, col: int) -> AnchorPlaceResult:
    rule = state.current_rule
    if rule is None:
        return AnchorPlaceResult(state=state, success=False, message="缺少待放置的初始颜料")
    if not current_rule_is_pending(state):
        return AnchorPlaceResult(state=state, success=False, message="当前初始颜料已经在地图上")
    if not 0 <= row < BOARD_SIZE or not 0 <= col < BOARD_SIZE:
        return AnchorPlaceResult(state=state, success=False, message="地图格子越界")

    tile = state.tile(row, col)
    if not _is_current_anchor_placeable(tile, state.active_affix):
        return AnchorPlaceResult(state=state, success=False, message="新的初始颜料只能放在空白地块，除非已选择特殊扩散词条")

    tiles = list(state.tiles)
    tiles[row * BOARD_SIZE + col] = _tile_after_current_anchor_placement(tile, rule.paint, state.active_affix)
    placed_state = replace(
        state,
        tiles=tuple(tiles),
        current_anchor=(row, col),
        current_rule_pending=False,
    )
    match_count = _current_match_count(placed_state, rule)
    placed_state = replace(
        placed_state,
        current_rule=replace(rule, match_current=min(match_count, rule.match_required)),
        backpack=sort_backpack_for_rule(placed_state.backpack, rule),
    )
    upgraded_state, upgraded_count = _apply_current_rule_upgrade(placed_state, placed_state.current_rule)
    affix_state, affix_message = _resolve_anchor_affix(upgraded_state, state.active_affix)
    message = (
        f"已放置新的初始颜料：{_paint_label(rule.paint)}，"
        f"安装即触发，范围内{upgraded_count}个目标颜料提升{rule.upgrade_amount}级，"
        "请选择背包颜料完成匹配"
    )
    if affix_message:
        message = f"{message}；{affix_message}"
    return AnchorPlaceResult(
        state=affix_state,
        success=True,
        message=message,
    )


def recommend_install_plan(state: PuzzleState) -> InstallPlan | None:
    rule = state.current_rule
    if rule is None:
        return None
    if current_rule_is_pending(state):
        return InstallPlan(steps=(), completes_match=False, reason="请选择一个空白地块放置新的初始颜料")
    initial_match = _current_match_count(state, rule)
    if initial_match >= rule.match_required:
        return InstallPlan(steps=(), completes_match=True, reason="当前金色范围已满足匹配条件")

    sim_state = replace(state, current_rule=replace(rule, match_current=min(initial_match, rule.match_required)))
    steps: list[InstallStep] = []
    completed_current = False
    max_steps = min(len(sim_state.backpack), BOARD_SIZE * BOARD_SIZE)

    for _ in range(max_steps):
        current_rule = sim_state.current_rule
        if current_rule is None:
            break
        current_match = _current_match_count(sim_state, current_rule)
        if current_match >= current_rule.match_required:
            sim_state = replace(sim_state, current_rule=replace(current_rule, match_current=current_rule.match_required))
            break

        recs = recommend_placements(sim_state, limit=24)
        if not recs:
            break
        best = recs[0]
        if best.paint is None:
            break

        backpack_index = next(
            (index for index, paint in enumerate(sim_state.backpack) if paint == best.paint),
            None,
        )
        if backpack_index is None:
            break

        result = install_backpack_paint(sim_state, backpack_index, best.row, best.col)
        if not result.success:
            break

        steps.append(InstallStep(row=best.row, col=best.col, paint=best.paint))
        sim_state = result.state
        if result.completed_match:
            completed_current = True
            break

    final_rule = sim_state.current_rule
    final_match = _current_match_count(sim_state, final_rule) if final_rule is not None else 0
    if completed_current:
        reason = (
            f"按计划安装{len(steps)}步可完成{rule.match_required}/{rule.match_required}，"
            "完成后刷新下一张初始颜料"
        )
    elif steps and final_rule is not None:
        reason = (
            f"已推演{len(steps)}步，当前最多到{min(final_match, final_rule.match_required)}/{final_rule.match_required}，"
            "后续缺少可用目标颜料或可安装格"
        )
    else:
        reason = "当前没有可执行的安装方案，可能缺少目标颜料或金框内没有可安装格"
    return InstallPlan(steps=tuple(steps), completes_match=completed_current, reason=reason)


def install_backpack_paint(state: PuzzleState, backpack_index: int, row: int, col: int) -> InstallResult:
    rule = state.current_rule
    if rule is None:
        return InstallResult(state=state, success=False, completed_match=False, message="缺少当前初始颜料规则")
    if current_rule_is_pending(state):
        return InstallResult(state=state, success=False, completed_match=False, message="请先选择空白地块放置新的初始颜料")
    if not 0 <= backpack_index < len(state.backpack):
        return InstallResult(state=state, success=False, completed_match=False, message="背包颜料索引无效")
    if not 0 <= row < BOARD_SIZE or not 0 <= col < BOARD_SIZE:
        return InstallResult(state=state, success=False, completed_match=False, message="地图格子越界")
    scope = set(_placement_scope_cells(state, rule))
    if (row, col) not in scope:
        return InstallResult(state=state, success=False, completed_match=False, message="只能安装在当前初始颜料金色作用范围内")
    target_tile = state.tile(row, col)
    if not _is_placeable(target_tile, state.active_affix):
        return InstallResult(state=state, success=False, completed_match=False, message="该格子当前不可安装颜料")

    paint = state.backpack[backpack_index]
    tiles = list(state.tiles)
    tiles[row * BOARD_SIZE + col] = Tile(state=TileState.PAINT, paint=paint, buff_applied_id=None, affix_applied_id=None)
    backpack = sort_backpack_for_rule(
        tuple(item for index, item in enumerate(state.backpack) if index != backpack_index),
        rule,
    )

    placed_state = replace(state, tiles=tuple(tiles), backpack=backpack)
    match_count = _current_match_count(placed_state, rule)
    upgraded_state, upgraded_count = _apply_current_rule_upgrade(placed_state, rule)
    upgraded_state, persistent_affix_count = _apply_persistent_affix_after_install(upgraded_state)
    completed = match_count >= rule.match_required
    next_rule = replace(rule, match_current=min(match_count, rule.match_required))
    progress = f"{min(match_count, rule.match_required)}/{rule.match_required}"
    message = f"已安装{_paint_label(paint)}，安装即触发，范围内{upgraded_count}个目标颜料提升{rule.upgrade_amount}级，当前进度{progress}"
    if persistent_affix_count:
        message = f"{message}；持续词缀额外提升{persistent_affix_count}个颜料"
    if completed:
        upgraded_state, next_rule_message = _advance_to_next_current_rule(upgraded_state)
        message = f"完成匹配，{message}；{next_rule_message}"
    return InstallResult(
        state=replace(upgraded_state, current_rule=next_rule if not completed else upgraded_state.current_rule),
        success=True,
        completed_match=completed,
        message=message,
    )


def force_complete_current_rule(state: PuzzleState, force_used: int, seed: int | None = None) -> ForceCompleteResult:
    rule = state.current_rule
    if rule is None:
        return ForceCompleteResult(state=state, success=False, destroyed_cells=(), message="缺少当前初始颜料规则")
    if current_rule_is_pending(state):
        return ForceCompleteResult(state=state, success=False, destroyed_cells=(), message="新的初始颜料尚未放置，不能强制完成")

    damage_count = min(3, max(1, force_used + 1))
    match_count = max(rule.match_required, _current_match_count(state, rule))
    progressed = replace(state, current_rule=replace(rule, match_current=rule.match_required))
    upgraded_state, upgraded_count = _apply_current_rule_upgrade(progressed, rule)

    paint_cells = [
        (row, col)
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
        if upgraded_state.tile(row, col).state == TileState.PAINT
    ]
    rng = random.Random(seed)
    destroyed = tuple(sorted(rng.sample(paint_cells, min(damage_count, len(paint_cells))))) if paint_cells else ()
    tiles = list(upgraded_state.tiles)
    for row, col in destroyed:
        tiles[row * BOARD_SIZE + col] = Tile(state=TileState.FADED, paint=None, buff_applied_id=None, affix_applied_id=None)
    damaged_state = replace(upgraded_state, tiles=tuple(tiles))
    next_state, next_message = _advance_to_next_current_rule(damaged_state)
    message = (
        f"已强制完成当前匹配，范围内{upgraded_count}个目标颜料完成本轮结算，"
        f"随机炸毁{len(destroyed)}格变为失色；{next_message}"
    )
    return ForceCompleteResult(
        state=next_state,
        success=True,
        destroyed_cells=destroyed,
        message=message,
    )


def _advance_to_next_current_rule(state: PuzzleState) -> tuple[PuzzleState, str]:
    rng = random.Random()
    tiles = list(state.tiles)
    blank_cells = [
        (index // BOARD_SIZE, index % BOARD_SIZE)
        for index, tile in enumerate(tiles)
        if tile.state == TileState.BLANK
    ]
    next_buff_id = state.current_buff_id + 1
    if not blank_cells:
        return (
            replace(
                state,
                current_rule=None,
                current_anchor=None,
                active_affix=None,
                current_buff_id=next_buff_id,
            ),
            "没有空白地块可生成新的初始颜料",
        )

    next_rule = random_current_rule(rng)
    next_state = replace(
        state,
        tiles=tuple(tiles),
        current_rule=next_rule,
        current_anchor=None,
        current_buff_id=next_buff_id,
        current_rule_pending=True,
        active_affix=None,
        backpack=sort_backpack_for_rule(state.backpack, next_rule),
    )
    return (
        next_state,
        f"已生成新的初始颜料：{_paint_label(next_rule.paint)}，请选择空白地块放置",
    )


def recommend_brush_types(
    state: PuzzleState,
    options: tuple[BrushType, ...] | list[BrushType],
) -> list[BrushRecommendation]:
    normalized = normalize_brush_options(options)
    recommendations = [
        BrushRecommendation(brush_type=brush_type, score=_score_brush_type(state, brush_type), reason=_brush_reason(state, brush_type))
        for brush_type in normalized
    ]
    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations


def recommend_affixes(
    state: PuzzleState,
    options: tuple[AffixRule, ...] | list[AffixRule],
) -> list[AffixRecommendation]:
    recommendations = [
        AffixRecommendation(affix=affix, score=_score_affix(state, affix), reason=_affix_reason(state, affix))
        for affix in options
    ]
    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations


def recommend_special_actions(
    state: PuzzleState,
    resources: ResourceState,
    brush_scenario: BrushChoiceScenario | None = None,
) -> list[SpecialActionRecommendation]:
    recommendations: list[SpecialActionRecommendation] = []
    force = _recommend_force_complete(state, resources)
    if force is not None:
        recommendations.append(force)
    if resources.refresh_available and resources.refresh_remaining > 0:
        stage1 = _recommend_refresh_stage1(state, brush_scenario)
        if stage1 is not None:
            recommendations.append(stage1)
        stage2 = _recommend_refresh_stage2(state, brush_scenario)
        if stage2 is not None:
            recommendations.append(stage2)
    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations


def random_brush_choice_scenario(seed: int | None = None) -> BrushChoiceScenario:
    rng = random.Random(seed)
    stage1 = normalize_brush_options(tuple(rng.sample(list(BRUSH_TYPE_ORDER), 2)))
    catalog_by_type: dict[BrushType, list[AffixRule]] = {brush_type: [] for brush_type in BRUSH_TYPE_ORDER}
    for affix in build_affix_catalog():
        catalog_by_type[affix.brush_type].append(affix)
    chosen_type = rng.choice(stage1)
    stage2 = tuple(rng.sample(catalog_by_type[chosen_type], min(2, len(catalog_by_type[chosen_type]))))
    return BrushChoiceScenario(stage1_options=stage1, chosen_stage1=chosen_type, stage2_options=stage2)


def apply_affix_to_state(state: PuzzleState, affix: AffixRule, seed: int | None = None) -> AffixApplyResult:
    rule = state.current_rule
    if rule is None:
        return AffixApplyResult(state=state, success=False, message="缺少当前初始颜料，无法应用词缀")

    rng = random.Random(seed)

    if affix.effect in {
        AffixEffect.CHANGE_DIRECTION,
        AffixEffect.CHANGE_TARGET_OBJECT,
        AffixEffect.MATCH_REQUIRED_DELTA,
        AffixEffect.RANDOMIZE_CURRENT_LEVEL,
        AffixEffect.RANDOMIZE_CURRENT_DIRECTION,
        AffixEffect.RANDOMIZE_CURRENT_CONTENT,
        AffixEffect.ADD_RANDOM_SCOPE,
        AffixEffect.REROLL_CURRENT_RULE,
    }:
        next_rule = _apply_rule_affix(rule, affix, state, rng)
        next_state = _replace_current_rule(
            state,
            next_rule,
            update_anchor=affix.effect
            in {
                AffixEffect.RANDOMIZE_CURRENT_LEVEL,
                AffixEffect.RANDOMIZE_CURRENT_CONTENT,
                AffixEffect.REROLL_CURRENT_RULE,
            },
        )
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}")

    if affix.effect in {
        AffixEffect.UPGRADE_IN_SCOPE,
        AffixEffect.COPY_SELF_TO_SCOPE,
        AffixEffect.RANDOMIZE_SCOPE_CATEGORY,
        AffixEffect.RANDOMIZE_SCOPE_ANY,
        AffixEffect.CLEAR_TILE_IN_SCOPE,
    } and state.current_anchor is None:
        return AffixApplyResult(
            state=replace(state, active_affix=affix),
            success=True,
            message=f"已选择词缀：{affix.name}，放置初始颜料后按其范围结算",
        )

    if affix.effect == AffixEffect.UPGRADE_SELF:
        next_paint = replace(rule.paint, level=min(MAX_LEVEL, rule.paint.level + max(1, affix.amount)))
        next_state = _replace_current_rule(state, replace(rule, paint=next_paint), update_anchor=True)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，当前初始颜料提升到Lv{next_paint.level}")

    if affix.effect == AffixEffect.UPGRADE_IN_SCOPE:
        next_state, count = _upgrade_scope_paints(state, affix, rng)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，范围内{count}个颜料已提升")

    if affix.effect == AffixEffect.UPGRADE_GLOBAL_CATEGORY:
        next_state, count = _upgrade_global_category_paints(state, affix)
        if affix.persistent_until_match:
            next_state = replace(next_state, active_affix=affix)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，全图{count}个颜料已提升")

    if affix.effect == AffixEffect.UPGRADE_GLOBAL_SAME_PAINT:
        next_state, count = _upgrade_global_same_paints(state, affix)
        if affix.persistent_until_match:
            next_state = replace(next_state, active_affix=affix)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，全图{count}个同类颜料已提升")

    if affix.effect == AffixEffect.UPGRADE_SCOPE_SAME_PAINT:
        next_state, count = _upgrade_scope_same_paints(state, affix)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，范围内{count}个同类颜料已提升")

    if affix.effect == AffixEffect.COPY_SELF_TO_SCOPE:
        next_state, count = _copy_self_to_scope(state)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，范围内{count}个颜料替换为自身")

    if affix.effect in {AffixEffect.RANDOMIZE_SCOPE_CATEGORY, AffixEffect.RANDOMIZE_SCOPE_ANY}:
        next_state, count = _randomize_scope_paints(state, affix, rng)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，范围内{count}个颜料已随机变化")

    if affix.effect == AffixEffect.CLEAR_TILE_IN_SCOPE:
        next_state, count = _clear_scope_tiles(state, affix, rng)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，范围内{count}个地块变为空白")

    if affix.effect == AffixEffect.CLEAR_RANDOM_TILE_GLOBAL:
        next_state, count = _clear_global_tiles(state, affix, rng)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，全图{count}个地块变为空白")

    if affix.effect in {
        AffixEffect.ALLOW_PLACE_ON_BLOCKED,
        AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF,
    }:
        return AffixApplyResult(
            state=replace(state, active_affix=affix),
            success=True,
            message=f"已选择词缀：{affix.name}，请把当前初始颜料放到对应地块",
        )

    return AffixApplyResult(state=replace(state, active_affix=affix), success=True, message=f"已选择词缀：{affix.name}")


def _recommend_force_complete(
    state: PuzzleState,
    resources: ResourceState,
) -> SpecialActionRecommendation | None:
    rule = state.current_rule
    if rule is None or resources.force_remaining <= 0 or current_rule_is_pending(state):
        return None
    damage_count = min(3, resources.force_used + 1)
    reward = _force_completion_reward(rule)
    risk = _expected_force_damage(state, damage_count)
    natural = recommend_placements(state, limit=1)
    natural_completes = bool(natural and natural[0].completes_match)
    score = reward - risk
    reason_parts = [
        f"立即完成当前匹配，预计收益{reward:.0f}",
        f"第{resources.force_used + 1}次会随机破坏{damage_count}个颜料，风险{risk:.0f}",
    ]
    if natural_completes:
        score -= 900.0
        reason_parts.append("已有自然完成落点，强制完成应保留")
    else:
        score += 450.0
        reason_parts.append("当前没有自然完成，救场价值上升")
    if rule.paint.is_treasure:
        reason_parts.append("当前是宝箱，强制收益更高")
    recommended = score >= 350.0
    if not recommended:
        reason_parts.append("暂不建议，除非局面即将卡死")
    return SpecialActionRecommendation(
        action=SpecialAction.FORCE_COMPLETE,
        score=score,
        recommended=recommended,
        reason="；".join(reason_parts),
    )


def _recommend_refresh_stage1(
    state: PuzzleState,
    brush_scenario: BrushChoiceScenario | None,
) -> SpecialActionRecommendation | None:
    if brush_scenario is None:
        return None
    current = recommend_brush_types(state, brush_scenario.stage1_options)
    ideal = recommend_brush_types(state, BRUSH_TYPE_ORDER)
    if not current or not ideal:
        return None
    current_best = current[0]
    ideal_best = ideal[0]
    missing_ideal = ideal_best.brush_type not in brush_scenario.stage1_options
    gap = ideal_best.score - current_best.score
    score = gap - 70.0
    reason = (
        f"当前最佳{current_best.brush_type.value}={current_best.score:.0f}，"
        f"理想最佳{ideal_best.brush_type.value}={ideal_best.score:.0f}"
    )
    if missing_ideal:
        score += 120.0
        reason += "；理想大类未出现，可考虑阶段1刷新"
    else:
        reason += "；理想大类已出现，通常不刷新"
    return SpecialActionRecommendation(
        action=SpecialAction.REFRESH_STAGE1,
        score=score,
        recommended=score >= 120.0,
        reason=reason,
    )


def _recommend_refresh_stage2(
    state: PuzzleState,
    brush_scenario: BrushChoiceScenario | None,
) -> SpecialActionRecommendation | None:
    if brush_scenario is None:
        return None
    current = recommend_affixes(state, brush_scenario.stage2_options)
    same_type = [affix for affix in build_affix_catalog() if affix.brush_type == brush_scenario.chosen_stage1]
    ideal = recommend_affixes(state, same_type)
    if not current or not ideal:
        return None
    current_best = current[0]
    ideal_best = ideal[0]
    gap = ideal_best.score - current_best.score
    score = gap - 60.0
    reason = (
        f"当前最佳词条{current_best.score:.0f}，"
        f"{brush_scenario.chosen_stage1.value}理想词条{ideal_best.score:.0f}"
    )
    if score >= 120.0:
        reason += "；当前二阶段词条偏弱，可考虑刷新"
    else:
        reason += "；当前词条可接受，刷新价值不足"
    return SpecialActionRecommendation(
        action=SpecialAction.REFRESH_STAGE2,
        score=score,
        recommended=score >= 120.0,
        reason=reason,
    )


def _force_completion_reward(rule: CurrentPaintRule) -> float:
    next_level = min(MAX_LEVEL, rule.paint.level + rule.upgrade_amount)
    reward = 900.0
    if rule.paint.is_treasure:
        reward += 1400.0
        if next_level >= MAX_LEVEL:
            reward += 6500.0
    elif next_level >= MAX_LEVEL and rule.paint.category == PaintCategory.GAMEPLAY:
        reward += 3000.0
    elif next_level >= MAX_LEVEL and rule.paint.category == PaintCategory.MONSTER:
        reward += 2400.0
    elif next_level >= 6:
        reward += 450.0
    return reward


def _expected_force_damage(state: PuzzleState, damage_count: int) -> float:
    paint_values = [_paint_damage_value(tile.paint) for tile in state.tiles if tile.state == TileState.PAINT]
    if not paint_values:
        return 0.0
    average_value = sum(paint_values) / len(paint_values)
    return average_value * min(damage_count, len(paint_values))


def _paint_damage_value(paint: Paint | None) -> float:
    if paint is None:
        return 320.0
    value = paint.level * 90.0
    if paint.is_treasure:
        value += 2200.0
    elif paint.category == PaintCategory.GAMEPLAY:
        value += 520.0
    elif paint.category == PaintCategory.MONSTER:
        value += 420.0
    if paint.level >= MAX_LEVEL:
        value += 2400.0
    elif paint.level >= 6:
        value += 800.0
    return value


def _score_brush_type(state: PuzzleState, brush_type: BrushType) -> float:
    catalog = [affix for affix in build_affix_catalog() if affix.brush_type == brush_type]
    if not catalog:
        return 0.0
    scored = sorted((_score_affix(state, affix) for affix in catalog), reverse=True)
    top = scored[:3]
    average_top = sum(top) / len(top)
    baseline = {
        BrushType.TRANSFORM: 18.0,
        BrushType.STACK: 24.0,
        BrushType.SPREAD: 16.0,
    }[brush_type]
    return average_top + baseline


def _brush_reason(state: PuzzleState, brush_type: BrushType) -> str:
    if brush_type == BrushType.TRANSFORM:
        if state.current_rule and state.current_rule.match_required >= 4:
            return "可救场：改方向/改目标/降低匹配数"
        return "可修正调色目标，适合当前目标不顺时"
    if brush_type == BrushType.STACK:
        if state.current_rule and state.current_rule.paint.level >= 5:
            return "冲级优先：当前颜料接近7级"
        if state.current_rule and state.current_rule.paint.is_treasure:
            return "宝箱优先：可赌复制或额外升级"
        return "提升收益稳定，适合做7级"
    if _count_tiles(state, TileState.BLOCKED) + _count_tiles(state, TileState.VOID) >= 8:
        return "清场优先：当前可用空间偏少"
    return "扩展落点，适合被阻挡/虚空卡住时"


def _score_affix(state: PuzzleState, affix: AffixRule) -> float:
    rule = state.current_rule
    if rule is None:
        return 0.0

    if affix.effect == AffixEffect.MATCH_REQUIRED_DELTA:
        return 140.0 if rule.match_required >= 4 else 70.0
    if affix.effect == AffixEffect.CHANGE_DIRECTION:
        return _direction_affix_score(state, affix)
    if affix.effect == AffixEffect.CHANGE_TARGET_OBJECT:
        return 55.0
    if affix.effect == AffixEffect.REROLL_CURRENT_RULE:
        current_fit = _current_rule_fit_score(state, rule)
        if current_fit < 0.35:
            return 230.0
        if current_fit < 0.65:
            return 130.0
        return 35.0
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_DIRECTION:
        return 80.0 if _current_rule_fit_score(state, rule) < 0.65 else 30.0
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_CONTENT:
        return 75.0 if not rule.paint.is_treasure else 20.0
    if affix.effect == AffixEffect.ADD_RANDOM_SCOPE:
        return 120.0 + _count_category_paints(state, rule.target_object) * 18.0 - max(0, affix.delta) * 35.0
    if affix.effect == AffixEffect.UPGRADE_SELF:
        return _level_gain_score(rule.paint, affix.amount, multiplier=1.15)
    if affix.effect == AffixEffect.UPGRADE_IN_SCOPE:
        affected = _best_scope_paint_count(state, rule.direction)
        count = affix.count if affix.count is not None else affected
        return min(affected, count) * _level_gain_score(rule.paint, affix.amount, multiplier=0.45)
    if affix.effect == AffixEffect.UPGRADE_SCOPE_SAME_PAINT:
        same_scope = sum(
            1
            for row, col in current_scope_cells(state, include_anchor=False)
            if state.tile(row, col).paint is not None and _is_same_paint_kind(state.tile(row, col).paint, rule.paint)
        )
        return same_scope * _level_gain_score(rule.paint, affix.amount, multiplier=0.52)
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_CATEGORY:
        target_count = _count_category_paints(state, affix.target_object)
        return target_count * affix.amount * 65.0 + (35.0 if affix.persistent_until_match else 0.0)
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_SAME_PAINT:
        same_count = _count_same_paints_as_current(state)
        return same_count * affix.amount * 95.0 + (55.0 if affix.persistent_until_match else 0.0)
    if affix.effect == AffixEffect.COPY_SELF_TO_SCOPE:
        scope_count = _best_scope_paint_count(state, rule.direction)
        treasure_bonus = 400.0 if rule.paint.is_treasure else 0.0
        return scope_count * (95.0 + rule.paint.level * 12.0) + treasure_bonus
    if affix.effect == AffixEffect.RANDOMIZE_SCOPE_CATEGORY:
        scope_count = _best_scope_paint_count(state, rule.direction)
        treasure_expectation = 85.0 if affix.target_object == PaintObject.GAMEPLAY else 0.0
        return scope_count * (55.0 + treasure_expectation)
    if affix.effect == AffixEffect.RANDOMIZE_SCOPE_ANY:
        return _best_scope_paint_count(state, rule.direction) * 45.0
    if affix.effect == AffixEffect.CLEAR_TILE_IN_SCOPE:
        clearable = _best_clearable_count(state, affix.target_object or PaintObject.BLOCKED_TILE, affix.direction or rule.direction)
        count = affix.count if affix.count is not None else clearable
        return min(clearable, count) * 95.0
    if affix.effect == AffixEffect.CLEAR_RANDOM_TILE_GLOBAL:
        tile_state = _target_object_to_tile_state(affix.target_object)
        clearable = _count_tiles(state, tile_state) if tile_state else _count_tiles(state, TileState.FADED)
        count = affix.count or 1
        return min(clearable, count) * (90.0 if tile_state != TileState.VOID else 75.0)
    if affix.effect == AffixEffect.ALLOW_PLACE_ON_BLOCKED:
        return _count_tiles(state, TileState.BLOCKED) * 38.0 + 45.0
    if affix.effect == AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF:
        return _count_tiles(state, TileState.VOID) * 25.0 + 30.0
    if affix.effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF:
        return _count_tiles(state, TileState.PAINT) * 20.0 + 25.0
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_LEVEL:
        return 45.0
    return 0.0


def _current_rule_fit_score(state: PuzzleState, rule: CurrentPaintRule) -> float:
    target_paints = sum(1 for paint in state.backpack if _paint_matches_target(paint, rule.target_object))
    required = max(1, rule.match_required)
    supply_fit = min(1.0, target_paints / required)

    if state.current_anchor is not None and not current_rule_is_pending(state):
        scope = _placement_scope_cells(state, rule)
        match_count = _current_match_count(state, rule)
        placeable = sum(1 for row, col in scope if _is_placeable(state.tile(row, col), state.active_affix))
        possible = min(required, match_count + min(placeable, target_paints))
        progress_fit = min(1.0, match_count / required)
        completion_fit = possible / required
        space_fit = min(1.0, placeable / required)
        return _clamp01(progress_fit * 0.35 + completion_fit * 0.35 + supply_fit * 0.20 + space_fit * 0.10)

    best_completion = 0.0
    best_space = 0.0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            tile = state.tile(row, col)
            if not _is_current_anchor_placeable(tile, state.active_affix):
                continue
            scope = tuple(cell for cell in _rule_scope_cells_for_anchor(row, col, rule) if cell != (row, col))
            existing_matches = sum(1 for r, c in scope if _tile_matches_target(state.tile(r, c), rule.target_object))
            placeable = sum(1 for r, c in scope if _is_placeable(state.tile(r, c), state.active_affix))
            possible = min(required, existing_matches + min(placeable, target_paints))
            best_completion = max(best_completion, possible / required)
            best_space = max(best_space, min(1.0, placeable / required))
    return _clamp01(best_completion * 0.62 + supply_fit * 0.26 + best_space * 0.12)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _affix_reason(state: PuzzleState, affix: AffixRule) -> str:
    rule = state.current_rule
    if affix.effect == AffixEffect.MATCH_REQUIRED_DELTA:
        return "降低完成门槛，救场价值高"
    if affix.effect == AffixEffect.CHANGE_DIRECTION:
        return f"改为{_direction_label(affix.direction or PaintDirection.UNKNOWN)}，寻找更高匹配范围"
    if affix.effect == AffixEffect.REROLL_CURRENT_RULE:
        return "重新随机一张初始颜料，当前目标/地形不顺时救场"
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_LEVEL:
        return "随机改变当前初始颜料等级，赌高等级收益"
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_DIRECTION:
        return "随机改变当前初始颜料方向，适合当前范围卡住时救场"
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_CONTENT:
        return "随机改变当前初始颜料内容但保留等级，可赌更高价值类型"
    if affix.effect == AffixEffect.ADD_RANDOM_SCOPE:
        return "增加一种作用范围但提高进度要求，适合背包供给充足时扩大收益"
    if affix.effect == AffixEffect.UPGRADE_SELF:
        return f"自身提升{affix.amount}级，直接冲级"
    if affix.effect == AffixEffect.UPGRADE_SCOPE_SAME_PAINT:
        return "范围内同内容颜料额外提升，适合同类已铺开时冲级"
    if affix.effect == AffixEffect.COPY_SELF_TO_SCOPE:
        if rule and rule.paint.is_treasure:
            return "宝箱复制潜力最高"
        return "把范围内颜料替换为当前颜料，适合复制高等级"
    if affix.effect == AffixEffect.RANDOMIZE_SCOPE_CATEGORY and affix.target_object == PaintObject.GAMEPLAY:
        return "随机玩法可赌宝箱"
    if affix.effect in {AffixEffect.CLEAR_TILE_IN_SCOPE, AffixEffect.CLEAR_RANDOM_TILE_GLOBAL}:
        return "清理特殊地块，扩大后续空间"
    if affix.effect in {
        AffixEffect.ALLOW_PLACE_ON_BLOCKED,
        AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF,
    }:
        return "增加特殊落点，注意可能失去自身内容"
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_CATEGORY:
        return "疑似持续全图升级，长期收益较高"
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_SAME_PAINT:
        return "全图同名/同宝箱颜料提升，适合当前颜料价值高或同类多时"
    return "按当前局面估算收益"


def build_affix_catalog() -> tuple[AffixRule, ...]:
    return (
        AffixRule(BrushType.TRANSFORM, "作用方向改为横向", AffixEffect.CHANGE_DIRECTION, direction=PaintDirection.HORIZONTAL),
        AffixRule(BrushType.TRANSFORM, "作用方向改为纵向", AffixEffect.CHANGE_DIRECTION, direction=PaintDirection.VERTICAL),
        AffixRule(BrushType.TRANSFORM, "作用方向改为对角斜向", AffixEffect.CHANGE_DIRECTION, direction=PaintDirection.DIAGONAL),
        AffixRule(BrushType.TRANSFORM, "作用方向改为相邻", AffixEffect.CHANGE_DIRECTION, direction=PaintDirection.ADJACENT),
        AffixRule(BrushType.TRANSFORM, "作用方向改为全图", AffixEffect.CHANGE_DIRECTION, direction=PaintDirection.GLOBAL),
        AffixRule(BrushType.TRANSFORM, "作用对象改为另一类型", AffixEffect.CHANGE_TARGET_OBJECT),
        AffixRule(BrushType.TRANSFORM, "完成条件所需数量-2", AffixEffect.MATCH_REQUIRED_DELTA, delta=-2),
        AffixRule(BrushType.TRANSFORM, "调色目标的进度要求减少1", AffixEffect.MATCH_REQUIRED_DELTA, delta=-1),
        AffixRule(BrushType.TRANSFORM, "当前颜料等级随机改变", AffixEffect.RANDOMIZE_CURRENT_LEVEL),
        AffixRule(BrushType.TRANSFORM, "随机改变自身的等级，内容不变", AffixEffect.RANDOMIZE_CURRENT_LEVEL),
        AffixRule(BrushType.TRANSFORM, "随机改变自身的作用方向", AffixEffect.RANDOMIZE_CURRENT_DIRECTION),
        AffixRule(BrushType.TRANSFORM, "随机改变自身的内容，等级不变", AffixEffect.RANDOMIZE_CURRENT_CONTENT),
        AffixRule(BrushType.TRANSFORM, "重新获得一个随机初始颜料", AffixEffect.REROLL_CURRENT_RULE),
        AffixRule(BrushType.TRANSFORM, "重新获得1个随机调色颜料", AffixEffect.REROLL_CURRENT_RULE),
        AffixRule(BrushType.TRANSFORM, "作用范围内颜料变成随机类型颜料", AffixEffect.RANDOMIZE_SCOPE_ANY),
        AffixRule(BrushType.TRANSFORM, "作用范围内颜料变成随机玩法类型颜料", AffixEffect.RANDOMIZE_SCOPE_CATEGORY, target_object=PaintObject.GAMEPLAY),
        AffixRule(BrushType.TRANSFORM, "作用范围内颜料变成随机怪物类型颜料", AffixEffect.RANDOMIZE_SCOPE_CATEGORY, target_object=PaintObject.MONSTER),
        AffixRule(BrushType.STACK, "随机增加一种作用范围，调色进度要求增加2", AffixEffect.ADD_RANDOM_SCOPE, delta=2),
        AffixRule(BrushType.STACK, "范围内随机1个颜料提升1级", AffixEffect.UPGRADE_IN_SCOPE, amount=1, count=1),
        AffixRule(BrushType.STACK, "范围内随机2个颜料提升1级", AffixEffect.UPGRADE_IN_SCOPE, amount=1, count=2),
        AffixRule(BrushType.STACK, "范围内所有颜料提升1级", AffixEffect.UPGRADE_IN_SCOPE, amount=1),
        AffixRule(BrushType.STACK, "范围内所有颜料提升2级", AffixEffect.UPGRADE_IN_SCOPE, amount=2),
        AffixRule(BrushType.STACK, "对调色范围内的随机1个颜料额外提升1级", AffixEffect.UPGRADE_IN_SCOPE, amount=1, count=1),
        AffixRule(BrushType.STACK, "对调色范围内的随机2个颜料额外提升1级", AffixEffect.UPGRADE_IN_SCOPE, amount=1, count=2),
        AffixRule(BrushType.STACK, "对调色范围内与自身内容相同的颜料额外提升1级", AffixEffect.UPGRADE_SCOPE_SAME_PAINT, amount=1),
        AffixRule(BrushType.STACK, "对调色范围内与自身内容相同的颜料额外提升2级", AffixEffect.UPGRADE_SCOPE_SAME_PAINT, amount=2),
        AffixRule(BrushType.STACK, "自身提升1级", AffixEffect.UPGRADE_SELF, amount=1),
        AffixRule(BrushType.STACK, "自身提升2级", AffixEffect.UPGRADE_SELF, amount=2),
        AffixRule(BrushType.STACK, "自身提升3级", AffixEffect.UPGRADE_SELF, amount=3),
        AffixRule(BrushType.STACK, "全图玩法颜料提升1级", AffixEffect.UPGRADE_GLOBAL_CATEGORY, target_object=PaintObject.GAMEPLAY, amount=1, persistent_until_match=True),
        AffixRule(BrushType.STACK, "全图玩法颜料提升2级", AffixEffect.UPGRADE_GLOBAL_CATEGORY, target_object=PaintObject.GAMEPLAY, amount=2, persistent_until_match=True),
        AffixRule(BrushType.STACK, "全图怪物颜料提升1级", AffixEffect.UPGRADE_GLOBAL_CATEGORY, target_object=PaintObject.MONSTER, amount=1, persistent_until_match=True),
        AffixRule(BrushType.STACK, "全图怪物颜料提升2级", AffixEffect.UPGRADE_GLOBAL_CATEGORY, target_object=PaintObject.MONSTER, amount=2, persistent_until_match=True),
        AffixRule(BrushType.STACK, "全图和自身相同的颜料提升1级", AffixEffect.UPGRADE_GLOBAL_SAME_PAINT, amount=1, persistent_until_match=True),
        AffixRule(BrushType.STACK, "全图和自身相同的颜料提升2级", AffixEffect.UPGRADE_GLOBAL_SAME_PAINT, amount=2, persistent_until_match=True),
        AffixRule(BrushType.STACK, "范围内颜料类型与等级替换为自身", AffixEffect.COPY_SELF_TO_SCOPE),
        AffixRule(BrushType.STACK, "范围内颜料变成随机玩法类型颜料", AffixEffect.RANDOMIZE_SCOPE_CATEGORY, target_object=PaintObject.GAMEPLAY),
        AffixRule(BrushType.STACK, "范围内颜料变成随机怪物类型颜料", AffixEffect.RANDOMIZE_SCOPE_CATEGORY, target_object=PaintObject.MONSTER),
        AffixRule(BrushType.STACK, "范围内颜料变成随机类型颜料", AffixEffect.RANDOMIZE_SCOPE_ANY),
        AffixRule(BrushType.STACK, "将作用范围内的颜料转变为同等级的随机颜料", AffixEffect.RANDOMIZE_SCOPE_ANY),
        AffixRule(BrushType.SPREAD, "横向随机阻挡地块变为空白地块", AffixEffect.CLEAR_TILE_IN_SCOPE, direction=PaintDirection.HORIZONTAL, target_object=PaintObject.BLOCKED_TILE, count=1),
        AffixRule(BrushType.SPREAD, "纵向随机阻挡地块变为空白地块", AffixEffect.CLEAR_TILE_IN_SCOPE, direction=PaintDirection.VERTICAL, target_object=PaintObject.BLOCKED_TILE, count=1),
        AffixRule(BrushType.SPREAD, "相邻随机阻挡地块变为空白地块", AffixEffect.CLEAR_TILE_IN_SCOPE, direction=PaintDirection.ADJACENT, target_object=PaintObject.BLOCKED_TILE, count=1),
        AffixRule(BrushType.SPREAD, "对角斜向随机阻挡地块变为空白地块", AffixEffect.CLEAR_TILE_IN_SCOPE, direction=PaintDirection.DIAGONAL, target_object=PaintObject.BLOCKED_TILE, count=1),
        AffixRule(BrushType.SPREAD, "将距离最近的1个阻挡地块转变为空白地块", AffixEffect.CLEAR_TILE_IN_SCOPE, target_object=PaintObject.BLOCKED_TILE, count=1, nearest=True),
        AffixRule(BrushType.SPREAD, "将作用范围中最近1个阻挡地块转变为空白地块", AffixEffect.CLEAR_TILE_IN_SCOPE, target_object=PaintObject.BLOCKED_TILE, count=1, nearest=True),
        AffixRule(BrushType.SPREAD, "范围内所有阻挡地块变为空白地块", AffixEffect.CLEAR_TILE_IN_SCOPE, target_object=PaintObject.BLOCKED_TILE),
        AffixRule(BrushType.SPREAD, "全图随机1块阻挡地块变为空白地块", AffixEffect.CLEAR_RANDOM_TILE_GLOBAL, target_object=PaintObject.BLOCKED_TILE, count=1),
        AffixRule(BrushType.SPREAD, "全图随机1块失色地块变为空白地块", AffixEffect.CLEAR_RANDOM_TILE_GLOBAL, count=1),
        AffixRule(BrushType.SPREAD, "全图随机1或2块虚空地块变为空白地块", AffixEffect.CLEAR_RANDOM_TILE_GLOBAL, target_object=PaintObject.VOID_TILE, count=2),
        AffixRule(BrushType.SPREAD, "当前颜料可放置在阻挡地块", AffixEffect.ALLOW_PLACE_ON_BLOCKED),
        AffixRule(BrushType.SPREAD, "当前颜料可放置在虚空地块但失去自身内容", AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF, persistent_until_match=True),
        AffixRule(BrushType.SPREAD, "当前颜料可放置在已放置颜料地块但失去自身内容", AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF, persistent_until_match=True),
    )


def _random_paint(rng: random.Random) -> Paint:
    category = rng.choice([PaintCategory.MONSTER, PaintCategory.GAMEPLAY])
    level = rng.choices([3, 4, 5, 6], weights=[35, 35, 20, 10], k=1)[0]
    subtype = _random_subtype(rng, category)
    return Paint(category=category, level=level, subtype=subtype, is_treasure=subtype == "宝箱")


def _random_subtype(rng: random.Random, category: PaintCategory) -> str:
    if category == PaintCategory.MONSTER:
        return rng.choice(["矮人", "幽灵", "觉醒者", "沙怪", "英灵"])
    if category == PaintCategory.GAMEPLAY:
        return rng.choices(
            ["黑帆", "黑潮", "迷城", "赛季", "怪谈", "存活", "宝箱"],
            weights=[17, 17, 17, 16, 16, 15, 2],
            k=1,
        )[0]
    return "未知"


def _is_placeable(tile: Tile, affix: AffixRule | None) -> bool:
    if tile.state == TileState.BLANK:
        return True
    if affix is None:
        return False
    if tile.state == TileState.BLOCKED and affix.effect == AffixEffect.ALLOW_PLACE_ON_BLOCKED:
        return True
    if tile.state == TileState.VOID and affix.effect == AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF:
        return True
    if tile.state == TileState.PAINT and affix.effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF:
        return True
    return False


def _is_current_anchor_placeable(tile: Tile, affix: AffixRule | None) -> bool:
    if tile.state == TileState.BLANK:
        return True
    if affix is None:
        return False
    if tile.state == TileState.BLOCKED and affix.effect == AffixEffect.ALLOW_PLACE_ON_BLOCKED:
        return True
    if tile.state == TileState.VOID and affix.effect == AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF:
        return True
    if tile.state == TileState.PAINT and affix.effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF:
        return True
    return False


def _tile_after_current_anchor_placement(tile: Tile, paint: Paint, affix: AffixRule | None) -> Tile:
    if affix is None:
        return Tile(state=TileState.PAINT, paint=paint, buff_applied_id=None, affix_applied_id=None)
    if tile.state == TileState.BLOCKED and affix.effect == AffixEffect.ALLOW_PLACE_ON_BLOCKED:
        return Tile(state=TileState.PAINT, paint=paint, buff_applied_id=None, affix_applied_id=None)
    if tile.state == TileState.VOID and affix.effect == AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF:
        return Tile(state=TileState.FADED, paint=None, buff_applied_id=None, affix_applied_id=None)
    if tile.state == TileState.PAINT and affix.effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF:
        return tile
    return Tile(state=TileState.PAINT, paint=paint, buff_applied_id=None, affix_applied_id=None)


def _tile_matches_target(tile: Tile, target: PaintObject) -> bool:
    if target == PaintObject.ALL:
        return tile.state == TileState.PAINT
    if target == PaintObject.MONSTER:
        if tile.paint is None:
            return tile.state == TileState.PAINT
        return tile.paint.category == PaintCategory.MONSTER
    if target == PaintObject.GAMEPLAY:
        if tile.paint is None:
            return tile.state == TileState.PAINT
        return tile.paint.category == PaintCategory.GAMEPLAY
    if target == PaintObject.BLOCKED_TILE:
        return tile.state == TileState.BLOCKED
    if target == PaintObject.BLANK_TILE:
        return tile.state == TileState.BLANK
    if target == PaintObject.VOID_TILE:
        return tile.state == TileState.VOID
    return tile.state == TileState.PAINT


def _candidate_backpack_paints(state: PuzzleState, rule: CurrentPaintRule) -> tuple[Paint, ...]:
    if not state.backpack:
        return ()
    candidates = tuple(paint for paint in state.backpack if _paint_matches_target(paint, rule.target_object))
    return candidates if candidates else state.backpack


def _paint_matches_target(paint: Paint, target: PaintObject) -> bool:
    if target == PaintObject.ALL:
        return True
    if target == PaintObject.MONSTER:
        return paint.category == PaintCategory.MONSTER
    if target == PaintObject.GAMEPLAY:
        return paint.category == PaintCategory.GAMEPLAY
    if target in {PaintObject.UNKNOWN, PaintObject.BLANK_TILE, PaintObject.BLOCKED_TILE, PaintObject.VOID_TILE}:
        return True
    return True


def _paint_label(paint: Paint) -> str:
    if paint.is_treasure:
        return f"宝箱Lv{paint.level}"
    category = "怪物" if paint.category == PaintCategory.MONSTER else "玩法" if paint.category == PaintCategory.GAMEPLAY else "未知"
    subtype = f"-{paint.subtype}" if paint.subtype else ""
    return f"{category}{subtype}Lv{paint.level}"


def _tile_can_receive_current_buff(tile: Tile, state: PuzzleState) -> bool:
    return tile.buff_applied_id != state.current_buff_id


def _tile_can_receive_persistent_affix(tile: Tile, state: PuzzleState) -> bool:
    return tile.affix_applied_id != state.current_buff_id


def _current_match_count(state: PuzzleState, rule: CurrentPaintRule) -> int:
    return sum(
        1
        for row, col in _placement_scope_cells(state, rule)
        if _tile_matches_target(state.tile(row, col), rule.target_object)
    )


def _apply_current_rule_upgrade(state: PuzzleState, rule: CurrentPaintRule) -> tuple[PuzzleState, int]:
    tiles = list(state.tiles)
    upgraded_count = 0
    for row, col in _placement_scope_cells(state, rule):
        index = row * BOARD_SIZE + col
        tile = tiles[index]
        if tile.paint is None or not _tile_matches_target(tile, rule.target_object):
            continue
        if not _tile_can_receive_current_buff(tile, state):
            continue
        upgraded = replace(tile.paint, level=min(MAX_LEVEL, tile.paint.level + rule.upgrade_amount))
        tiles[index] = Tile(
            state=TileState.PAINT,
            paint=upgraded,
            buff_applied_id=state.current_buff_id,
            affix_applied_id=tile.affix_applied_id,
        )
        upgraded_count += 1
    return replace(state, tiles=tuple(tiles)), upgraded_count


def _direction_affix_score(state: PuzzleState, affix: AffixRule) -> float:
    rule = state.current_rule
    if rule is None or affix.direction is None:
        return 0.0
    current_best = _best_scope_paint_count(state, rule.direction)
    new_best = _best_scope_paint_count(state, affix.direction)
    gain = max(0, new_best - current_best)
    return 45.0 + gain * 45.0


def _level_gain_score(paint: Paint, amount: int, multiplier: float = 1.0) -> float:
    next_level = min(MAX_LEVEL, paint.level + amount)
    score = amount * 55.0
    if paint.is_treasure:
        score += 220.0
        if next_level >= MAX_LEVEL:
            score += 900.0
    elif next_level >= MAX_LEVEL and paint.category == PaintCategory.GAMEPLAY:
        score += 430.0
    elif next_level >= MAX_LEVEL:
        score += 340.0
    elif next_level >= 6:
        score += 90.0
    return score * multiplier


def _best_scope_paint_count(state: PuzzleState, direction: PaintDirection) -> int:
    best = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            count = sum(1 for r, c in _scope_cells(row, col, direction) if state.tile(r, c).state == TileState.PAINT)
            best = max(best, count)
    return best


def _best_clearable_count(state: PuzzleState, target: PaintObject, direction: PaintDirection) -> int:
    tile_state = _target_object_to_tile_state(target) or TileState.BLOCKED
    best = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            count = sum(1 for r, c in _scope_cells(row, col, direction) if state.tile(r, c).state == tile_state)
            best = max(best, count)
    return best


def _count_category_paints(state: PuzzleState, target: PaintObject | None) -> int:
    count = 0
    for tile in state.tiles:
        if tile.state != TileState.PAINT:
            continue
        if tile.paint is None:
            count += 1
        elif target == PaintObject.MONSTER and tile.paint.category == PaintCategory.MONSTER:
            count += 1
        elif target == PaintObject.GAMEPLAY and tile.paint.category == PaintCategory.GAMEPLAY:
            count += 1
        elif target in {PaintObject.ALL, None, PaintObject.UNKNOWN}:
            count += 1
    return count


def _count_tiles(state: PuzzleState, tile_state: TileState) -> int:
    return sum(1 for tile in state.tiles if tile.state == tile_state)


def _target_object_to_tile_state(target: PaintObject | None) -> TileState | None:
    if target == PaintObject.BLOCKED_TILE:
        return TileState.BLOCKED
    if target == PaintObject.BLANK_TILE:
        return TileState.BLANK
    if target == PaintObject.VOID_TILE:
        return TileState.VOID
    return None


def _placement_scope_cells(state: PuzzleState, rule: CurrentPaintRule) -> tuple[tuple[int, int], ...]:
    if state.current_anchor is None and not current_rule_is_pending(state):
        return tuple((row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE))
    if state.current_anchor is None:
        return ()
    if state.current_anchor is None:
        return ()
    row, col = state.current_anchor
    return tuple(cell for cell in _rule_scope_cells_for_anchor(row, col, rule) if cell != state.current_anchor)


def _recommend_current_anchor_positions(
    state: PuzzleState,
    rule: CurrentPaintRule,
    limit: int,
) -> list[PlacementRecommendation]:
    recommendations: list[PlacementRecommendation] = []
    target_paints = tuple(paint for paint in state.backpack if _paint_matches_target(paint, rule.target_object))
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            tile = state.tile(row, col)
            if not _is_current_anchor_placeable(tile, state.active_affix):
                continue
            scope = tuple(cell for cell in _rule_scope_cells_for_anchor(row, col, rule) if cell != (row, col))
            placeable = sum(1 for r, c in scope if _is_placeable(state.tile(r, c), state.active_affix))
            existing_matches = sum(1 for r, c in scope if _tile_matches_target(state.tile(r, c), rule.target_object))
            possible_matches = existing_matches + min(placeable, len(target_paints))
            progress_ratio = min(1.0, possible_matches / max(1, rule.match_required))
            center_bonus = _cell_priority((row, col)) * 4.0
            score = progress_ratio * 600.0 + placeable * 18.0 + existing_matches * 60.0 + center_bonus
            if possible_matches >= rule.match_required:
                score += 500.0
            if tile.state != TileState.BLANK:
                score += 120.0
            reason = (
                f"放置初始颜料，{_direction_label(rule.direction)}范围可凑"
                f"{min(possible_matches, rule.match_required)}/{rule.match_required}"
            )
            if tile.state != TileState.BLANK:
                reason += f"，使用特殊词条落在{tile.state.value}"
            recommendations.append(
                PlacementRecommendation(
                    row=row,
                    col=col,
                    paint=rule.paint,
                    score=score,
                    completes_match=possible_matches >= rule.match_required,
                    affected=scope,
                    reason=reason,
                )
            )

    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations[:limit]


def _scope_cells(row: int, col: int, direction: PaintDirection) -> tuple[tuple[int, int], ...]:
    if direction == PaintDirection.HORIZONTAL:
        return tuple((row, c) for c in range(BOARD_SIZE))
    if direction == PaintDirection.VERTICAL:
        return tuple((r, col) for r in range(BOARD_SIZE))
    if direction == PaintDirection.DIAGONAL:
        cells = {(row, col)}
        for delta in range(-BOARD_SIZE, BOARD_SIZE + 1):
            for r, c in ((row + delta, col + delta), (row + delta, col - delta)):
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    cells.add((r, c))
        return tuple(sorted(cells))
    if direction == PaintDirection.ADJACENT:
        cells = []
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            r = row + dr
            c = col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                cells.append((r, c))
        return tuple(cells)
    if direction == PaintDirection.GLOBAL:
        return tuple((r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
    return ((row, col),)


def _rule_scope_cells_for_anchor(row: int, col: int, rule: CurrentPaintRule) -> tuple[tuple[int, int], ...]:
    cells: list[tuple[int, int]] = []
    for direction in (rule.direction, *rule.extra_directions):
        cells.extend(_scope_cells(row, col, direction))
    return tuple(dict.fromkeys(cells))


def _score_candidate(
    state: PuzzleState,
    rule: CurrentPaintRule,
    row: int,
    col: int,
    affected: tuple[tuple[int, int], ...],
    outcome: PlacementOutcome,
    placed_paint: Paint,
) -> float:
    # Mature match/merge heuristics: objective completion first, special/high
    # value pieces second, then cascades/space management.
    progress_ratio = outcome.match_count / max(1, rule.match_required)
    score = progress_ratio * 120.0
    if outcome.completes_match:
        score += 900.0
    else:
        missing = rule.match_required - outcome.match_count
        score += max(0, 4 - missing) * 45.0

    if placed_paint.is_treasure:
        score += 1200.0
        if outcome.current_next_level >= MAX_LEVEL:
            score += 6000.0
    elif outcome.current_next_level >= MAX_LEVEL:
        if placed_paint.category == PaintCategory.GAMEPLAY:
            score += 2800.0
        elif placed_paint.category == PaintCategory.MONSTER:
            score += 2200.0
        else:
            score += 1600.0
    elif outcome.current_next_level >= 6:
        score += 350.0

    score += outcome.treasure_to_7 * 5000.0
    score += outcome.gameplay_to_7 * 1800.0
    score += outcome.monster_to_7 * 1400.0
    score += max(0, outcome.any_to_7 - outcome.treasure_to_7 - outcome.gameplay_to_7 - outcome.monster_to_7) * 900.0
    score += outcome.treasure_upgrades * 600.0
    score += outcome.affected_paints * (22.0 if outcome.completes_match else 6.0)
    score += _future_space_value(state, row, col, affected)
    return score


def _placement_outcome(
    state: PuzzleState,
    rule: CurrentPaintRule,
    row: int,
    col: int,
    affected: tuple[tuple[int, int], ...],
    placed_paint: Paint,
) -> PlacementOutcome:
    match_count = sum(
        1
        for r, c in affected
        if ((r, c) == (row, col) and _paint_matches_target(placed_paint, rule.target_object))
        or _tile_matches_target(state.tile(r, c), rule.target_object)
    )
    completes = match_count >= rule.match_required
    current_next_level = min(
        MAX_LEVEL,
        placed_paint.level + (rule.upgrade_amount if _paint_matches_target(placed_paint, rule.target_object) else 0),
    )
    affected_paints = 0
    treasure_to_7 = 0
    gameplay_to_7 = 0
    monster_to_7 = 0
    any_to_7 = 0
    treasure_upgrades = 0

    for r, c in affected:
        if (r, c) == (row, col):
            if not _paint_matches_target(placed_paint, rule.target_object):
                continue
            paint = placed_paint
            can_receive_buff = True
        else:
            tile = state.tile(r, c)
            if not _tile_matches_target(tile, rule.target_object):
                continue
            paint = tile.paint
            can_receive_buff = _tile_can_receive_current_buff(tile, state)
        if paint is None:
            continue
        if not can_receive_buff:
            continue
        affected_paints += 1
        next_level = min(MAX_LEVEL, paint.level + rule.upgrade_amount)
        if paint.is_treasure:
            treasure_upgrades += 1
            if paint.level < MAX_LEVEL <= next_level:
                treasure_to_7 += 1
        if paint.level < MAX_LEVEL <= next_level:
            any_to_7 += 1
            if paint.is_treasure:
                continue
            if paint.category == PaintCategory.GAMEPLAY:
                gameplay_to_7 += 1
            elif paint.category == PaintCategory.MONSTER:
                monster_to_7 += 1

    return PlacementOutcome(
        match_count=match_count,
        completes_match=completes,
        current_next_level=current_next_level,
        affected_paints=affected_paints,
        treasure_to_7=treasure_to_7,
        gameplay_to_7=gameplay_to_7,
        monster_to_7=monster_to_7,
        any_to_7=any_to_7,
        treasure_upgrades=treasure_upgrades,
    )


def _future_space_value(
    state: PuzzleState,
    row: int,
    col: int,
    affected: tuple[tuple[int, int], ...],
) -> float:
    blanks = sum(1 for r, c in affected if state.tile(r, c).state == TileState.BLANK)
    blocked = sum(1 for r, c in affected if state.tile(r, c).state == TileState.BLOCKED)
    voids = sum(1 for r, c in affected if state.tile(r, c).state == TileState.VOID)
    center_bonus = 8.0 - abs(row - 2.5) - abs(col - 2.5)
    return blanks * 1.5 + blocked * 0.8 + voids * 0.4 + center_bonus


def _cell_priority(cell: tuple[int, int]) -> float:
    row, col = cell
    return 8.0 - abs(row - 2.5) - abs(col - 2.5)


def _reason(rule: CurrentPaintRule, outcome: PlacementOutcome, placed_paint: Paint) -> str:
    parts = [
        f"用{_paint_label(placed_paint)}",
        f"{_direction_label(rule.direction)} {_target_label(rule.target_object)} {outcome.match_count}/{rule.match_required}",
    ]
    if outcome.completes_match:
        parts.append(f"完成匹配，触发提升{rule.upgrade_amount}级")
    if placed_paint.is_treasure and outcome.current_next_level >= MAX_LEVEL and outcome.completes_match:
        parts.append("该宝箱可到7级")
    elif placed_paint.is_treasure:
        parts.append("宝箱优先")
    elif outcome.current_next_level >= MAX_LEVEL and outcome.completes_match:
        parts.append("当前颜料可到7级")
    if outcome.treasure_to_7:
        parts.append(f"范围内宝箱到7级x{outcome.treasure_to_7}")
    if outcome.gameplay_to_7:
        parts.append(f"玩法到7级x{outcome.gameplay_to_7}")
    if outcome.monster_to_7:
        parts.append(f"怪物到7级x{outcome.monster_to_7}")
    return "，".join(parts)


def _direction_label(direction: PaintDirection) -> str:
    return {
        PaintDirection.HORIZONTAL: "横向",
        PaintDirection.VERTICAL: "纵向",
        PaintDirection.DIAGONAL: "对角斜向",
        PaintDirection.ADJACENT: "相邻",
        PaintDirection.GLOBAL: "全图",
        PaintDirection.UNKNOWN: "未知方向",
    }[direction]


def _target_label(target: PaintObject) -> str:
    return {
        PaintObject.MONSTER: "怪物",
        PaintObject.GAMEPLAY: "玩法",
        PaintObject.ALL: "所有",
        PaintObject.BLOCKED_TILE: "阻挡地块",
        PaintObject.BLANK_TILE: "空白地块",
        PaintObject.VOID_TILE: "虚空地块",
        PaintObject.UNKNOWN: "颜料",
    }[target]


def apply_transform_affix(rule: CurrentPaintRule, affix: AffixRule) -> CurrentPaintRule:
    if affix.effect == AffixEffect.CHANGE_DIRECTION and affix.direction is not None:
        return replace(rule, direction=affix.direction)
    if affix.effect == AffixEffect.CHANGE_TARGET_OBJECT and affix.target_object is not None:
        return replace(rule, target_object=affix.target_object)
    if affix.effect == AffixEffect.MATCH_REQUIRED_DELTA:
        return replace(rule, match_required=max(1, rule.match_required + affix.delta))
    return rule


def _apply_rule_affix(
    rule: CurrentPaintRule,
    affix: AffixRule,
    state: PuzzleState,
    rng: random.Random,
) -> CurrentPaintRule:
    if affix.effect == AffixEffect.REROLL_CURRENT_RULE:
        return random_current_rule(rng)
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_LEVEL:
        return replace(rule, paint=replace(rule.paint, level=rng.randint(1, MAX_LEVEL)))
    if affix.effect == AffixEffect.CHANGE_TARGET_OBJECT and affix.target_object is None:
        return replace(rule, target_object=_alternate_target_object(rule.target_object, rule.paint.category))
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_DIRECTION:
        return replace(
            rule,
            direction=rng.choice(
                [
                    PaintDirection.HORIZONTAL,
                    PaintDirection.VERTICAL,
                    PaintDirection.DIAGONAL,
                    PaintDirection.ADJACENT,
                    PaintDirection.GLOBAL,
                ]
            ),
        )
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_CONTENT:
        next_category = _randomized_category(rule.paint.category, affix.target_object, rng)
        return replace(rule, paint=_random_paint_with_category_level(rng, next_category, rule.paint.level))
    if affix.effect == AffixEffect.ADD_RANDOM_SCOPE:
        candidates = [
            PaintDirection.HORIZONTAL,
            PaintDirection.VERTICAL,
            PaintDirection.DIAGONAL,
            PaintDirection.ADJACENT,
            PaintDirection.GLOBAL,
        ]
        existing = {rule.direction, *rule.extra_directions}
        available = [direction for direction in candidates if direction not in existing]
        extra = rng.choice(available or candidates)
        return replace(
            rule,
            extra_directions=tuple(dict.fromkeys((*rule.extra_directions, extra))),
            match_required=max(1, rule.match_required + max(0, affix.delta)),
        )
    return apply_transform_affix(rule, affix)


def _alternate_target_object(target: PaintObject, paint_category: PaintCategory) -> PaintObject:
    if target == PaintObject.MONSTER:
        return PaintObject.GAMEPLAY
    if target == PaintObject.GAMEPLAY:
        return PaintObject.MONSTER
    if paint_category == PaintCategory.MONSTER:
        return PaintObject.GAMEPLAY
    return PaintObject.MONSTER


def _replace_current_rule(
    state: PuzzleState,
    rule: CurrentPaintRule,
    *,
    update_anchor: bool = False,
) -> PuzzleState:
    tiles = list(state.tiles)
    if update_anchor and state.current_anchor is not None:
        row, col = state.current_anchor
        index = row * BOARD_SIZE + col
        tile = tiles[index]
        if tile.state == TileState.PAINT and tile.paint is not None:
            tiles[index] = replace(tile, paint=rule.paint, buff_applied_id=None)
    temp_state = replace(state, tiles=tuple(tiles))
    match_current = 0
    if temp_state.current_anchor is not None:
        match_current = min(_current_match_count(temp_state, rule), rule.match_required)
    next_rule = replace(rule, match_current=match_current)
    return replace(
        temp_state,
        current_rule=next_rule,
        backpack=sort_backpack_for_rule(temp_state.backpack, next_rule),
    )


def _resolve_anchor_affix(state: PuzzleState, affix: AffixRule | None) -> tuple[PuzzleState, str]:
    if affix is None:
        return state, ""
    if affix.effect == AffixEffect.UPGRADE_IN_SCOPE:
        next_state, count = _upgrade_scope_paints(state, affix, random.Random())
        return replace(next_state, active_affix=None), f"词缀额外提升{count}个范围内颜料"
    if affix.effect == AffixEffect.COPY_SELF_TO_SCOPE:
        next_state, count = _copy_self_to_scope(state)
        return replace(next_state, active_affix=None), f"词缀把范围内{count}个颜料替换为当前初始颜料"
    if affix.effect in {AffixEffect.RANDOMIZE_SCOPE_CATEGORY, AffixEffect.RANDOMIZE_SCOPE_ANY}:
        next_state, count = _randomize_scope_paints(state, affix, random.Random())
        return replace(next_state, active_affix=None), f"词缀随机变化范围内{count}个颜料"
    if affix.effect == AffixEffect.CLEAR_TILE_IN_SCOPE:
        next_state, count = _clear_scope_tiles(state, affix, random.Random())
        return replace(next_state, active_affix=None), f"词缀清理范围内{count}个地块"
    if affix.effect in {
        AffixEffect.ALLOW_PLACE_ON_BLOCKED,
        AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF,
    }:
        return replace(state, active_affix=None), "特殊放置词条已消耗"
    return state, ""


def _apply_persistent_affix_after_install(state: PuzzleState) -> tuple[PuzzleState, int]:
    affix = state.active_affix
    if affix is None or not affix.persistent_until_match:
        return state, 0
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_CATEGORY:
        return _upgrade_global_category_paints(state, affix)
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_SAME_PAINT:
        return _upgrade_global_same_paints(state, affix)
    return state, 0


def _upgrade_scope_paints(state: PuzzleState, affix: AffixRule, rng: random.Random) -> tuple[PuzzleState, int]:
    amount = max(1, affix.amount)
    candidates = [
        row * BOARD_SIZE + col
        for row, col in _affix_scope_cells(state, affix)
        if state.tile(row, col).state == TileState.PAINT and state.tile(row, col).paint is not None
    ]
    if affix.count is not None:
        candidates = rng.sample(candidates, min(affix.count, len(candidates))) if candidates else []
    tiles = list(state.tiles)
    for index in candidates:
        tile = tiles[index]
        if tile.paint is None:
            continue
        tiles[index] = Tile(
            state=TileState.PAINT,
            paint=replace(tile.paint, level=min(MAX_LEVEL, tile.paint.level + amount)),
            buff_applied_id=tile.buff_applied_id,
            affix_applied_id=tile.affix_applied_id,
        )
    return replace(state, tiles=tuple(tiles)), len(candidates)


def _upgrade_global_category_paints(state: PuzzleState, affix: AffixRule) -> tuple[PuzzleState, int]:
    amount = max(1, affix.amount)
    tiles = list(state.tiles)
    count = 0
    for index, tile in enumerate(tiles):
        if tile.paint is None or not _paint_matches_target(tile.paint, affix.target_object or PaintObject.ALL):
            continue
        if affix.persistent_until_match and not _tile_can_receive_persistent_affix(tile, state):
            continue
        tiles[index] = Tile(
            state=TileState.PAINT,
            paint=replace(tile.paint, level=min(MAX_LEVEL, tile.paint.level + amount)),
            buff_applied_id=tile.buff_applied_id,
            affix_applied_id=state.current_buff_id if affix.persistent_until_match else tile.affix_applied_id,
        )
        count += 1
    return replace(state, tiles=tuple(tiles)), count


def _upgrade_global_same_paints(state: PuzzleState, affix: AffixRule) -> tuple[PuzzleState, int]:
    rule = state.current_rule
    if rule is None:
        return state, 0
    amount = max(1, affix.amount)
    tiles = list(state.tiles)
    count = 0
    for index, tile in enumerate(tiles):
        if tile.paint is None or not _is_same_paint_kind(tile.paint, rule.paint):
            continue
        if affix.persistent_until_match and not _tile_can_receive_persistent_affix(tile, state):
            continue
        tiles[index] = Tile(
            state=TileState.PAINT,
            paint=replace(tile.paint, level=min(MAX_LEVEL, tile.paint.level + amount)),
            buff_applied_id=tile.buff_applied_id,
            affix_applied_id=state.current_buff_id if affix.persistent_until_match else tile.affix_applied_id,
        )
        count += 1
    return replace(state, tiles=tuple(tiles)), count


def _upgrade_scope_same_paints(state: PuzzleState, affix: AffixRule) -> tuple[PuzzleState, int]:
    rule = state.current_rule
    if rule is None:
        return state, 0
    amount = max(1, affix.amount)
    tiles = list(state.tiles)
    count = 0
    for row, col in _affix_scope_cells(state, affix):
        index = row * BOARD_SIZE + col
        tile = tiles[index]
        if tile.paint is None or not _is_same_paint_kind(tile.paint, rule.paint):
            continue
        tiles[index] = Tile(
            state=TileState.PAINT,
            paint=replace(tile.paint, level=min(MAX_LEVEL, tile.paint.level + amount)),
            buff_applied_id=tile.buff_applied_id,
            affix_applied_id=tile.affix_applied_id,
        )
        count += 1
    return replace(state, tiles=tuple(tiles)), count


def _copy_self_to_scope(state: PuzzleState) -> tuple[PuzzleState, int]:
    rule = state.current_rule
    if rule is None:
        return state, 0
    tiles = list(state.tiles)
    count = 0
    for row, col in _affix_scope_cells(state, None):
        index = row * BOARD_SIZE + col
        tile = tiles[index]
        if tile.state != TileState.PAINT:
            continue
        tiles[index] = Tile(state=TileState.PAINT, paint=rule.paint, buff_applied_id=None, affix_applied_id=None)
        count += 1
    return replace(state, tiles=tuple(tiles)), count


def _count_same_paints_as_current(state: PuzzleState) -> int:
    rule = state.current_rule
    if rule is None:
        return 0
    return sum(
        1
        for tile in state.tiles
        if tile.paint is not None and _is_same_paint_kind(tile.paint, rule.paint)
    )


def _is_same_paint_kind(paint: Paint, reference: Paint) -> bool:
    if paint.is_treasure or reference.is_treasure:
        return paint.is_treasure and reference.is_treasure
    return paint.category == reference.category and paint.subtype == reference.subtype


def _randomize_scope_paints(state: PuzzleState, affix: AffixRule, rng: random.Random) -> tuple[PuzzleState, int]:
    tiles = list(state.tiles)
    count = 0
    for row, col in _affix_scope_cells(state, affix):
        index = row * BOARD_SIZE + col
        tile = tiles[index]
        if tile.state != TileState.PAINT or tile.paint is None:
            continue
        category = _randomized_category(tile.paint.category, affix.target_object, rng)
        next_paint = _random_paint_with_category_level(rng, category, tile.paint.level)
        tiles[index] = Tile(state=TileState.PAINT, paint=next_paint, buff_applied_id=None, affix_applied_id=None)
        count += 1
    return replace(state, tiles=tuple(tiles)), count


def _clear_scope_tiles(state: PuzzleState, affix: AffixRule, rng: random.Random) -> tuple[PuzzleState, int]:
    tile_state = _target_object_to_tile_state(affix.target_object) or TileState.BLOCKED
    candidates = [
        row * BOARD_SIZE + col
        for row, col in _affix_scope_cells(state, affix)
        if state.tile(row, col).state == tile_state
    ]
    if affix.count is not None:
        count = min(affix.count, len(candidates))
        if affix.nearest:
            candidates = _nearest_indices_to_anchor(state, candidates)[:count]
        else:
            candidates = rng.sample(candidates, count) if candidates else []
    return _clear_tile_indices(state, candidates)


def _clear_global_tiles(state: PuzzleState, affix: AffixRule, rng: random.Random) -> tuple[PuzzleState, int]:
    tile_state = _target_object_to_tile_state(affix.target_object) or TileState.FADED
    candidates = [index for index, tile in enumerate(state.tiles) if tile.state == tile_state]
    count = affix.count if affix.count is not None else 1
    if tile_state == TileState.VOID and count == 2:
        count = rng.randint(1, 2)
    chosen = rng.sample(candidates, min(count, len(candidates))) if candidates else []
    return _clear_tile_indices(state, chosen)


def _clear_tile_indices(state: PuzzleState, indices: list[int]) -> tuple[PuzzleState, int]:
    tiles = list(state.tiles)
    for index in indices:
        tiles[index] = Tile(state=TileState.BLANK, paint=None, buff_applied_id=None, affix_applied_id=None)
    return replace(state, tiles=tuple(tiles)), len(indices)


def _affix_scope_cells(state: PuzzleState, affix: AffixRule | None) -> tuple[tuple[int, int], ...]:
    rule = state.current_rule
    if rule is None or state.current_anchor is None:
        return ()
    row, col = state.current_anchor
    directions = [affix.direction if affix and affix.direction is not None else rule.direction]
    if affix is None or affix.direction is None:
        directions.extend(rule.extra_directions)
    cells: list[tuple[int, int]] = []
    for direction in directions:
        cells.extend(cell for cell in _scope_cells(row, col, direction) if cell != state.current_anchor)
    return tuple(dict.fromkeys(cells))


def _nearest_indices_to_anchor(state: PuzzleState, indices: list[int]) -> list[int]:
    if state.current_anchor is None:
        return indices
    anchor_row, anchor_col = state.current_anchor
    return sorted(
        indices,
        key=lambda index: (
            abs(index // BOARD_SIZE - anchor_row) + abs(index % BOARD_SIZE - anchor_col),
            index,
        ),
    )


def _randomized_category(
    fallback: PaintCategory,
    target_object: PaintObject | None,
    rng: random.Random,
) -> PaintCategory:
    if target_object == PaintObject.MONSTER:
        return PaintCategory.MONSTER
    if target_object == PaintObject.GAMEPLAY:
        return PaintCategory.GAMEPLAY
    if target_object == PaintObject.ALL or target_object is None:
        return rng.choice([PaintCategory.MONSTER, PaintCategory.GAMEPLAY])
    return fallback
