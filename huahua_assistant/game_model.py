from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from enum import Enum

from .config import config_value
from . import s8_official_data
from . import s8_plot_effect
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

#: 强制调色最多可用 3 次（官方 ``config_s8_gameplay_revive`` 只有
#: 摧毁 1/2/3 个颜料三档，说明单局最多重置三轮）。
MAX_FORCE_TIMES = 3

#: 宠物 Lv5 提供 2 次「刷新词缀」机会；官方无独立宠物表
#: 宠物等级 → 词缀刷新次数（实机确认，2026-11-12）。
#:
#: * Lv1–2：0 次
#: * Lv3–4：1 次
#: * Lv5–6：2 次
#:
#: **只能刷新第二阶段的词缀**；第一阶段（画笔类型：转锋/堆砌/扩散）无法刷新。
PET_REFRESH_BY_LEVEL: dict[int, int] = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}

#: 默认按「没有 Lv5 宠物」算，需要时由调用方显式开启。
DEFAULT_PET_LEVEL = 1

# Value scale: one gameplay Lv7 is the base unit. Treasure values follow the
# user's explicit box-to-gameplay equivalence. Monster Lv7 is a top-end
# approximation because the supplied box-to-monster ratios are not mutually
# consistent across treasure Lv5/Lv6/Lv7.
GAMEPLAY_7_VALUE = 1000.0
MONSTER_7_VALUE = GAMEPLAY_7_VALUE / 1.4
TREASURE_LEVEL_VALUES = {
    5: GAMEPLAY_7_VALUE,
    6: GAMEPLAY_7_VALUE * 3.0,
    7: GAMEPLAY_7_VALUE * 5.0,
}


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
    ALLOW_PLACE_ON_FADED = "allow_place_on_faded"
    COPY_SELF_CONTENT_GLOBAL = "copy_self_content_global"
    RELOCATE_GLOBAL_BLOCKED = "relocate_global_blocked"


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
    # 等级加成（服务端 FacadeLevel）：挂在**地块**上，不是挂在颜料上。
    # 颜料被覆盖、被炸毁、被清空后，加成仍留在这一格，下一次落笔直接吃满。
    bonus_level: int = 0
    # 「与自身相同的颜料额外提升x级」这类光环的**强度**，记在锚点地块上。
    # 光环的作用对象不是固定的：它永远等于「和这一格当前颜料同种」的全部颜料。
    # 所以初始颜料被炸掉（这一格变失色）→ 光环失效，别处同种颜料掉级；
    # 失色地块恢复后放一枚新颜料 → 光环自动改挂到新颜料上，别处同种又涨回来。
    aura_amount: int = 0
    # 派生缓存：这一格从全场所有光环里吃到的等级。每次棋盘变动后由
    # ``_recompute_auras`` 重算，不要手写。
    aura_level: int = 0


def tile_bonus_level(tile: Tile) -> int:
    """地块上累计的等级加成（不含光环）。"""
    return max(0, int(tile.bonus_level))


def tile_aura_level(tile: Tile) -> int:
    """这一格当前从「同种颜料光环」里吃到的等级。"""
    return max(0, int(tile.aura_level))


#: **赛季家族**。官方 `201103 / 201203 / 201303 / 201403`（玩法颜料-赛季 Lv3/Lv4）原文：
#: 「雪原中出现随机赛季怪物；**该颜料升至5级及以上时，转变为随机赛季玩法**」。
#: 也就是说 Lv5 起会显示成具体赛季玩法（黑潮/黑帆/魔方/迷城/梦魇/怪谈/玩偶），
#: 但**属性仍属赛季家族**——用户 2026-11-18 明确：「2 个赛季 Lv3 升级后变成了一个玩偶
#: 6 级一个黑帆 5 级，但他的属性还是属于赛季颜料家族」。
#: 所以「与自身内容相同」「同种颜料光环」的判定必须把这一组归一，否则赛季显形之后
#: 就丢掉自己那份加成（官方 2014 的文案本身就是「与自身内容相同」）。
SEASON_FAMILY_SUBTYPES: frozenset[str] = frozenset(
    {"赛季", "黑潮", "黑帆", "魔方", "迷城", "梦魇", "怪谈", "玩偶"}
)


def _paint_kind_key(paint: Paint) -> tuple[str, ...]:
    """「同种颜料」的判据：宝箱单独一类，其余按 类别 + 子类型（赛季系归一）。"""
    if paint.is_treasure:
        return ("treasure",)
    subtype = paint.subtype or ""
    if subtype in SEASON_FAMILY_SUBTYPES:
        subtype = "赛季"
    return (paint.category.value, subtype)


def tile_final_level(tile: Tile) -> int:
    """最终等级 = min(MaxPlotLevel, 颜料等级 + 地块加成 + 光环加成)（服务端 FinalItemLevel）。"""
    base = tile.paint.level if tile.paint is not None else 0
    return max(0, min(MAX_LEVEL, base + tile_bonus_level(tile) + tile_aura_level(tile)))


def tile_effective_paint(tile: Tile) -> Paint | None:
    """把地块加成折算进等级后的颜料，用于估值与报表。

    模型里"等级"和"等级加成"是两套数，只有相加后的最终等级才对应棋盘上真实显示的东西。
    """
    if tile.paint is None:
        return None
    final = tile_final_level(tile)
    if final == tile.paint.level:
        return tile.paint
    return replace(tile.paint, level=final)


def _tile_with_upgrade(tile: Tile, amount: int, *, as_bonus: bool) -> Tile:
    """把一次"N 级提升"落到地块上。

    ``as_bonus=True`` 对应官方「额外提升」类笔触 → 记到**地块加成**上，
    所以颜料被覆盖后这笔加成还在；否则加到颜料自身等级上。
    两条路径都受 7 级封顶约束：加成已经顶满时，再提升颜料等级不产生任何收益。
    """
    if tile.paint is None or amount <= 0:
        return tile
    headroom = max(0, MAX_LEVEL - tile_bonus_level(tile) - tile_aura_level(tile))
    if as_bonus:
        return replace(tile, bonus_level=min(MAX_LEVEL, tile_bonus_level(tile) + amount))
    return replace(
        tile,
        paint=replace(tile.paint, level=max(tile.paint.level, min(headroom, tile.paint.level + amount))),
    )


def _recompute_auras(state: PuzzleState) -> PuzzleState:
    """按当前棋盘重算全场光环。

    光环 = 某个地块上的 ``aura_amount``，作用对象是「和这一格当前颜料同种」的全部颜料
    （含这一格自己）。所以：

    - 锚点地块的颜料被炸毁/清空 → 这一格没有颜料，锚点失效 → 别处同种颜料掉级；
    - 失色地块恢复后放一枚新颜料 → 锚点自动改挂新颜料，别处同种又涨回来。

    这是「buff 落在地块上」的直接推论，也是实机验证过的行为。
    """
    anchors = [
        (tile.aura_amount, _paint_kind_key(tile.paint))
        for tile in state.tiles
        if tile.aura_amount > 0 and tile.state == TileState.PAINT and tile.paint is not None
    ]
    tiles: list[Tile] = []
    changed = False
    for tile in state.tiles:
        level = 0
        if anchors and tile.state == TileState.PAINT and tile.paint is not None:
            kind = _paint_kind_key(tile.paint)
            level = sum(amount for amount, anchor_kind in anchors if anchor_kind == kind)
        if tile.aura_level != level:
            changed = True
            tile = replace(tile, aura_level=level)
        tiles.append(tile)
    return replace(state, tiles=tuple(tiles)) if changed else state


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
    canonical_affix_id: str = field(default="", compare=False)
    # 官方「额外提升」类笔触：提升记在地块加成上，而不是颜料等级上。
    grants_bonus: bool = False
    aliases: tuple[str, ...] = field(default=(), compare=False)
    legacy_codes: tuple[str, ...] = field(default=(), compare=False)
    official_ids: tuple[int, ...] = field(default=(), compare=False)
    official_text: str = field(default="", compare=False)
    verified: bool = field(default=True, compare=False)
    """官方 S8_Gameplay_SelectableEffect 中是否存在对应记录。

    数据来源见 ``huahua_assistant.s8_official_data``。``False`` 表示该条目的
    语义在官方配置里找不到出处（早期版本凭实机观察推断），评分仍按原逻辑走，
    但界面/导出中应区别对待。
    """

    @property
    def display_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class PuzzleState:
    tiles: tuple[Tile, ...]
    current_rule: CurrentPaintRule | None = None
    active_affix: AffixRule | None = None
    backpack: tuple[Paint, ...] = ()
    current_anchor: tuple[int, int] | None = None
    current_buff_id: int = 1
    current_rule_pending: bool = False
    observed_scope_cells: tuple[tuple[int, int], ...] = ()
    #: 「使自身等级提升N级」这类词缀攒下的待结算等级。
    #:
    #: 官方语义（用户实机确认）：笔触词缀里写「自身」的，只作用在**这枚初始颜料自己**身上；
    #: 写「范围内XX颜料」的，才作用到别的颜料（那条走 :data:`SCOPE_BUFF_EFFECTS`）。
    #: 所以选词缀时不能直接改棋盘——那时候颜料还没落下去。这里只记一笔，
    #: 等 :func:`install_backpack_paint` 真正落子时加到**放下去那枚**颜料的等级上。
    #:
    #: 为什么不改 ``current_rule.paint``：那条路只改「面板上显示的当前颜料」，而落子用的是
    #: ``state.backpack[i]``，两个对象不联动，实测选了「自身提升2级」落下去还是原等级（等于白选）。
    self_upgrade_pending: int = 0

    def tile(self, row: int, col: int) -> Tile:
        return self.tiles[row * BOARD_SIZE + col]


@dataclass(frozen=True)
class RecommendationExplanation:
    immediate: float = 0.0
    terrain: float = 0.0
    risk: float = 0.0
    resource: float = 0.0
    expected: float = 0.0
    value: float = 0.0
    tags: tuple[str, ...] = ()
    short: str = ""


@dataclass(frozen=True)
class PlacementRecommendation:
    row: int
    col: int
    paint: Paint | None
    score: float
    completes_match: bool
    affected: tuple[tuple[int, int], ...]
    reason: str
    explanation: RecommendationExplanation | None = None


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
class _InstallSearchNode:
    state: PuzzleState
    steps: tuple[InstallStep, ...]
    match_count: int
    score: float
    completed: bool = False


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
    value_gain: float
    placed_value_gain: float
    overwritten_paint: Paint | None
    overwrite_penalty: float
    terrain_delta: float
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
    force_remaining: int = MAX_FORCE_TIMES
    force_used: int = 0
    refresh_available: bool = False
    refresh_remaining: int = 0


def resources_for_pet_level(pet_level: int, *, refresh_used: int = 0, force_used: int = 0) -> ResourceState:
    """按宠物等级算本局可用资源（目前只有 Lv5 给刷新次数）。"""
    refresh_total = PET_REFRESH_BY_LEVEL.get(int(pet_level), 0)
    return ResourceState(
        force_remaining=max(0, MAX_FORCE_TIMES - int(force_used)),
        force_used=int(force_used),
        refresh_available=refresh_total > 0,
        refresh_remaining=max(0, refresh_total - int(refresh_used)),
    )


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
    """按游戏里的真实顺序重排背包：**目标类别优先，然后等级从高到低**。

    这个顺序是实机读到的规则（用户 2026-11-16 确认），``sort_backpack_for_rule``
    要复现它才能算「下一步该点第几枚」。

    **同分不再比家族名**：以前 key 末尾带了 ``subtype``，那时候背包识别从不给
    subtype 赋值（恒为 ``""``），这一位等于没有；现在背包能认出家族了，字典序
    会真的介入，而游戏里同等级的家族顺序**不是字典序**（按官方 ID）——按字典序
    排会让「第 3 枚」跟游戏里不是同一格。所以这里去掉它，交给稳定排序保持
    上一帧读到的实际顺序，那才是游戏排出来的顺序。
    """
    target = rule.target_object if rule is not None else PaintObject.UNKNOWN

    def key(paint: Paint) -> tuple[int, int, int]:
        target_match = 1 if _paint_matches_target(paint, target) else 0
        treasure = 1 if paint.is_treasure else 0
        return (target_match, paint.level, treasure)

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


def _rule_is_uninformative(rule: CurrentPaintRule | None) -> bool:
    """规则三项全 UNKNOWN 视为「这一帧没读到」，不能当真值。

    实机里面板会有短暂过渡帧：点完「重新获取随机调色颜料」、面板还没刷出
    调色目标/效果，只剩标题（甚至标题都读不到）。这时候把规则清空，提示会
    整片消失，玩家会以为工具挂了。正确做法是沿用上一帧的认知，下一帧读到就覆盖。
    """
    if rule is None:
        return True
    return (
        rule.paint.category == PaintCategory.UNKNOWN
        and rule.direction == PaintDirection.UNKNOWN
        and rule.target_object == PaintObject.UNKNOWN
    )


def state_from_observation(
    board: BoardObservation,
    current_paint: CurrentPaintObservation | None = None,
    backpack: tuple[Paint, ...] | list[Paint] = (),
    previous: PuzzleState | None = None,
) -> PuzzleState:
    """把一帧视觉观测翻成模型状态。

    ``previous`` 是上一帧的状态。传进来时会把**地块加成**（``bonus_level`` /
    ``aura_amount``）按格子继承过来——这两项视觉上完全看不见，只能靠规则推演攒，
    而每帧都重建状态会把它们清零，于是「叠 buff 到 7 级」这条主线永远学不到。
    """
    rule = rule_from_observation(current_paint)
    rule_carried = False
    if previous is not None and _rule_is_uninformative(rule):
        previous_rule = previous.current_rule
        if not _rule_is_uninformative(previous_rule):
            rule = previous_rule
            rule_carried = True
    anchor: tuple[int, int] | None = None
    marked_cells = [tile for tile in board.tiles if getattr(tile, "is_current_anchor", False)]
    observed_scope_cells = tuple(
        (tile.row, tile.col)
        for tile in board.tiles
        if getattr(tile, "is_scope_highlight", False)
    )
    paint_cells = [tile for tile in board.tiles if tile.state == TileState.PAINT]
    if rule is not None and marked_cells:
        marked = max(
            marked_cells,
            key=lambda tile: float(getattr(tile, "current_anchor_confidence", 0.0) or 0.0),
        )
        anchor = (marked.row, marked.col)
    elif rule is not None and len(paint_cells) == 1:
        anchor = (paint_cells[0].row, paint_cells[0].col)
    elif rule is not None:
        anchor = _infer_anchor_from_progress(board.tiles, rule)
    if rule_carried and previous is not None:
        # 规则沿用上一帧，锚点和金框范围也跟着沿用，否则作用范围会算成空集，
        # 落点提示全被安全阀挡掉。
        if anchor is None:
            anchor = previous.current_anchor
        if not observed_scope_cells:
            observed_scope_cells = previous.observed_scope_cells

    carried = _carried_tile_buffs(previous)
    tiles = []
    for tile in board.tiles:
        bonus, aura_amount = carried.get((tile.row, tile.col), (0, 0))
        paint = _paint_from_tile_observation(tile)
        if anchor == (tile.row, tile.col) and rule is not None:
            # 锚点格要区分两件事：面板给的是颜料**自己的**等级，棋盘角标给的是
            # **最终等级**（颜料等级 + 地块加成 + 光环）。
            #
            # **初始颜料也是颜料**（用户 2026-11-22）：它能吃到**其他**初始颜料的
            # 调色效果与笔触效果加成，而且加成装到地块之后持续生效。所以锚点那格的
            # 画面等级高于面板等级是正常现象，不是识别错误。
            #
            # 以前这里直接 ``paint = rule.paint``，等于用面板值顶掉实测值。
            # 实机样本 r1c6：3 级赛季颜料 + 「对全图中与自身内容相同的颜料额外提升
            # 3 级」→ 画面 6 级、点击地块加成 +3，面板仍是 3 级；覆盖之后那 3 级
            # 就没了，对账于是永远对不上。
            #
            # 现在改成：颜料等级取面板值，**超出面板的差额记到地块加成上**。
            observed_level = paint.level if paint is not None else None
            paint = rule.paint
            if observed_level is not None:
                surplus = int(observed_level) - int(getattr(rule.paint, "level", 0) or 0)
                if surplus > 0:
                    bonus = max(bonus, min(MAX_LEVEL, surplus))
        tiles.append(
            Tile(
                state=tile.state,
                paint=paint,
                bonus_level=bonus,
                aura_amount=aura_amount,
            )
        )
    # 规则已选定、但棋盘上找不到锚点 → 处于「待落初始颜料」状态。
    # 判据是「有没有锚点」**且**「进度还是 0」：进度 ≥1 说明颜料已经落过笔，
    # 只是金框高亮没被视觉抓出来。这时候再判 pending，桌宠会把「安装」当成
    # 「放初始颜料」往空白格上怼——实机表现就是点了范围外的格子触发
    # ConfText1「放入颜料的位置不在调色范围内」，白扔一枚颜料。
    pending = bool(rule is not None and anchor is None and int(rule.match_current or 0) <= 0)
    state = PuzzleState(
        tiles=tuple(tiles),
        current_rule=rule,
        backpack=sort_backpack_for_rule(tuple(backpack), rule),
        current_anchor=anchor,
        current_rule_pending=pending,
        observed_scope_cells=observed_scope_cells,
    )
    # 光环是派生的，先把 aura_level 算出来，才能把「棋盘上的最终等级」反推回颜料自身等级。
    state = _recompute_auras(state)
    return _recompute_auras(_rebase_observed_levels(state, anchor=anchor))


def _infer_anchor_from_progress(
    tiles,
    rule: CurrentPaintRule,
) -> tuple[int, int] | None:
    """进度已经 ≥1 但金框/锚点高亮没被视觉抓到时的兜底。

    实机上「当前调色目标 2/4」这类进度**本身就是证据**：已经有两枚颜料落进
    调色范围了。金框高亮是一条很淡的金色描边，视觉经常扫不到，于是
    ``state_from_observation`` 会把锚点判成 None、把局面判成「待落初始颜料」，
    桌宠就去点空白格——点了范围外的格子会弹 ConfText1 确认框，颜料白扔。

    这里按「已落颜料里哪一条线最长」反推锚点。对 横向/纵向/对角斜向 来说，
    作用范围**只由这条线决定**，锚点具体落在线上的哪一格不影响范围计算，
    所以随便取线上一个匹配格都是安全的；只有 相邻 是真的依赖锚点位置，
    取「同种颜料邻居最多」的那一格。

    选线只看「线上有几枚颜料」，**不先按颜料家族过滤**：实机上纵向那条线里
    常常混着怪物/玩法两类（甚至就是宝箱那两格），先按家族过滤会把真正的那条
    线筛没、只剩一条只有一格颜料的空线，然后推荐一堆「范围外」的落点。
    家族只用来在选定的线里挑一格当锚点。
    """
    cells = [
        (tile, paint)
        for tile in tiles
        if tile.state == TileState.PAINT
        for paint in (_paint_from_tile_observation(tile),)
        if paint is not None and _paint_matches_target(paint, rule.target_object)
    ]
    if not cells:
        return None

    if rule.direction == PaintDirection.HORIZONTAL:
        groups: dict[object, list] = {}
        for tile, _paint in cells:
            groups.setdefault(tile.row, []).append(tile)
    elif rule.direction == PaintDirection.VERTICAL:
        groups = {}
        for tile, _paint in cells:
            groups.setdefault(tile.col, []).append(tile)
    elif rule.direction == PaintDirection.DIAGONAL:
        groups = {}
        for tile, _paint in cells:
            groups.setdefault(tile.row - tile.col, []).append(tile)
        for tile, _paint in cells:
            groups.setdefault(tile.row + tile.col + 1000, []).append(tile)
    else:
        # 全图：范围与锚点无关，取任意一格即可；相邻：取邻居最多的一格。
        if rule.direction == PaintDirection.ADJACENT:
            occupied = {(tile.row, tile.col) for tile, _paint in cells}
            best, _best_paint = max(
                cells,
                key=lambda item: sum(
                    1 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if (item[0].row + dr, item[0].col + dc) in occupied
                ),
            )
            return (best.row, best.col)
        first, _first_paint = cells[0]
        return (first.row, first.col)

    longest = max(groups.values(), key=len)
    same_family = [tile for tile in longest if _tile_is_family(tile, rule.paint.category)]
    line = same_family or longest
    line = sorted(line, key=lambda tile: (tile.row, tile.col))
    pick = line[len(line) // 2]
    return (pick.row, pick.col)


def _tile_is_family(tile, category: PaintCategory) -> bool:
    paint = _paint_from_tile_observation(tile)
    return paint is not None and paint.category == category


def _carried_tile_buffs(previous: PuzzleState | None) -> dict[tuple[int, int], tuple[int, int]]:
    """从上一帧状态里取出每格要继承的 ``(bonus_level, aura_amount)``。

    只带这两项，不带 ``aura_level``——它是派生缓存，交给 :func:`_recompute_auras` 重算。
    按格子继承而不是按颜料继承，是因为这两项挂在**地块**上：颜料被炸掉、地块失色、
    恢复后重新落笔，加成仍在原地（实机验证过）。
    """
    if previous is None:
        return {}
    carried: dict[tuple[int, int], tuple[int, int]] = {}
    for index, tile in enumerate(previous.tiles):
        bonus = tile_bonus_level(tile)
        aura = max(0, int(tile.aura_amount))
        if bonus or aura:
            carried[divmod(index, BOARD_SIZE)] = (bonus, aura)
    return carried


def _rebase_observed_levels(state: PuzzleState, anchor: tuple[int, int] | None = None) -> PuzzleState:
    """把「棋盘上的最终等级」换算回「颜料自身等级」。

    棋盘右下角的 ``LvN`` 角标显示的是**最终等级**（颜料等级 + 地块加成 + 光环），
    而模型里 ``Paint.level`` 是颜料自己的等级，``tile_final_level`` 负责相加。
    两边不换算就会重复计一次加成。

    三条护栏：

    - 加成是 0 的格子完全不动（老路径一字不改）；
    - 观测到 7 级说明已经封顶，反推不唯一，保持原样；
    - 反推出来小于 1 级，说明模型比现实多算了 → 就地清掉这一格的加成，让它自我纠正。
    """
    tiles: list[Tile] = []
    for index, tile in enumerate(state.tiles):
        cell = divmod(index, BOARD_SIZE)
        if tile.paint is None or cell == anchor:
            tiles.append(tile)
            continue
        bonus = tile_bonus_level(tile)
        aura = tile_aura_level(tile)
        if bonus + aura <= 0 or tile.paint.level >= MAX_LEVEL:
            tiles.append(tile)
            continue
        base = tile.paint.level - bonus - aura
        if base < 1:
            tiles.append(replace(tile, bonus_level=0, aura_amount=0, aura_level=0))
            continue
        tiles.append(replace(tile, paint=replace(tile.paint, level=base)))
    return replace(state, tiles=tuple(tiles))


def _paint_from_tile_observation(tile) -> Paint | None:
    if tile.state != TileState.PAINT:
        return None
    category = getattr(tile, "paint_category", PaintCategory.UNKNOWN)
    level = getattr(tile, "paint_level", None)
    category_confidence = float(getattr(tile, "paint_category_confidence", 0.0) or 0.0)
    level_confidence = float(getattr(tile, "paint_level_confidence", 0.0) or 0.0)
    min_category = float(config_value("quality.board_paint_min_category_confidence", 0.46))
    min_level = float(config_value("quality.board_paint_min_level_confidence", 0.36))
    if category == PaintCategory.UNKNOWN or level is None:
        return None
    if category_confidence < min_category or level_confidence < min_level:
        return None
    return Paint(
        category=category,
        level=max(1, min(MAX_LEVEL, int(level))),
        subtype=getattr(tile, "paint_subtype", None),
        is_treasure=bool(getattr(tile, "paint_is_treasure", False)),
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
    direction, target_object = official_rule_axes(rng, upgrade_amount)
    return CurrentPaintRule(
        paint=paint,
        direction=direction,
        target_object=target_object,
        match_required=match_required,
        upgrade_amount=upgrade_amount,
    )


def official_rule_axes(
    rng: random.Random,
    amount: int,
) -> tuple[PaintDirection, PaintObject]:
    """按官方 27 条「调色目标/效果」抽方向与对象。

    官方只有 27 组组合（见 ``s8_plot_effect``），旧实现从 4 方向 × 3 对象里
    均匀抽，**永远抽不到「全图」**，而全图正是 3005/3006/4001 三条。
    这里按官方记录等概率抽取，全图自然占 3/27。
    """
    pool = [item for item in s8_plot_effect.PLOT_EFFECTS if item.amount == amount]
    if not pool:
        pool = list(s8_plot_effect.PLOT_EFFECTS)
    picked = rng.choice(pool)
    return picked.direction, picked.target


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
    level = _draw_official_level(rng, minimum=3)
    subtype = _random_subtype(rng, category, level)
    return Paint(category=category, level=level, subtype=subtype, is_treasure=subtype == "宝箱")


def _random_backpack_levels(rng: random.Random, size: int) -> list[int]:
    """按官方权重抽背包等级。

    官方唯一给出「1..7 级 → 权重」的数据是 ``4005`` 内嵌的
    ``1:700;2:1800;3:2000;4:3000;5:2000;6:400;7:100``（万分比，期望 3.54）。
    旧实现用自设的 ``[1,2,3] weights=[36,34,30]`` 再加 1~2 张固定 4/5 级，
    既没有官方出处，也把 6/7 级排除在外。这里改为直接按官方权重抽，
    并保留「一局里 4 级以上不会退化成清一色」的手感约束：先按权重抽，
    再保证至少有一张不低于 3 级。
    """
    distribution = _official_level_distribution()
    if not distribution:
        # 官方数据缺失时回退旧行为，保证不劣化。
        level5_count = rng.randint(0, min(1, size))
        level4_count = rng.randint(0, min(2, size - level5_count))
        levels = [5] * level5_count + [4] * level4_count
        while len(levels) < size:
            levels.append(rng.choices([1, 2, 3], weights=[36, 34, 30], k=1)[0])
        rng.shuffle(levels)
        return levels
    levels = list(rng.choices(
        list(distribution.keys()),
        weights=list(distribution.values()),
        k=size,
    ))
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


def _official_level_distribution() -> dict[int, float]:
    """官方颜料等级分布 ``{1..7: 概率}``；官方数据不可用时返回 ``{}``。

    来源见 :data:`s8_official_data.PAINT_LEVEL_WEIGHT_SOURCE`。
    """
    try:
        distribution = s8_official_data.paint_level_distribution()
    except (AttributeError, TypeError, ValueError):
        return {}
    if not distribution:
        return {}
    if any(level < 1 or level > MAX_LEVEL for level in distribution):
        return {}
    return distribution


def _draw_official_level(rng: random.Random, *, minimum: int = 1) -> int:
    """按官方权重抽一个等级，可限制下界（用于'目标颜料不会太低'的场合）。"""
    distribution = _official_level_distribution()
    candidates = {
        level: probability for level, probability in distribution.items()
        if level >= minimum
    }
    if not candidates:
        return rng.randint(max(1, minimum), MAX_LEVEL)
    total = sum(candidates.values())
    if total <= 0:
        return rng.randint(max(1, minimum), MAX_LEVEL)
    return rng.choices(
        list(candidates.keys()),
        weights=list(candidates.values()),
        k=1,
    )[0]


def _random_paint_with_category_level(rng: random.Random, category: PaintCategory, level: int) -> Paint:
    subtype = _random_subtype(rng, category, level)
    return Paint(category=category, level=level, subtype=subtype, is_treasure=subtype == "宝箱")


def recommend_placements(
    state: PuzzleState,
    limit: int = 5,
    *,
    include_lookahead: bool = True,
    learning_stats: dict | None = None,
) -> list[PlacementRecommendation]:
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
            score += placement_data_bias(state, paint, row, col)
            lookahead_score = (
                _candidate_lookahead_score(state, rule, row, col, paint, outcome)
                if include_lookahead
                else 0.0
            )
            score += lookahead_score
            prior_score = _placement_policy_prior_score(learning_stats, row, col) if learning_stats else 0.0
            terrain_prior_score = _terrain_policy_prior_score(learning_stats, outcome.terrain_delta) if learning_stats else 0.0
            bucket_prior_score = _terrain_bucket_policy_score(learning_stats, state) if learning_stats else 0.0
            score += prior_score
            score += terrain_prior_score
            score += bucket_prior_score
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
                    reason=_reason(rule, outcome, paint, lookahead_score, prior_score + terrain_prior_score + bucket_prior_score),
                    explanation=_placement_explanation(
                        state,
                        rule,
                        row,
                        col,
                        scoped_cells,
                        outcome,
                        paint,
                        lookahead_score,
                        prior_score + terrain_prior_score + bucket_prior_score,
                    ),
                )
            )

    recommendations = _dedupe_placement_recommendations(recommendations)
    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations[:limit]


def place_current_rule_anchor(
    state: PuzzleState,
    row: int,
    col: int,
    *,
    rng: random.Random | None = None,
) -> AnchorPlaceResult:
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
        # 「自身提升N级」已经在选词缀时加到 ``rule.paint`` 上了（这一笔落的正是它），
        # 这里清掉，免得下一次落笔再加一遍。
        self_upgrade_pending=0,
    )
    match_count = _current_match_count(placed_state, rule)
    placed_state = replace(
        placed_state,
        current_rule=replace(rule, match_current=min(match_count, rule.match_required)),
        backpack=sort_backpack_for_rule(placed_state.backpack, rule),
    )
    # 结算顺序由 settle_place_effects 统一：覆盖类笔触先改写范围内容，
    # 调色再按改写后的类型给等级（用户 2026-11-22 实机口径）。
    settled_state, upgraded_count, scope_count, settle_note = settle_place_effects(
        placed_state, placed_state.current_rule, rng or random.Random()
    )
    affix_state = _recompute_auras(settled_state)
    message = (
        f"已放置新的初始颜料：{_paint_label(rule.paint)}，"
        f"安装即触发，范围内{upgraded_count}个目标颜料提升{rule.upgrade_amount}级，"
        "请选择背包颜料完成匹配"
    )
    scope_phrase = _scope_affix_phrase(scope_count, placed_state.active_affix, unit="目标")
    if scope_phrase:
        message = f"{message}；{scope_phrase}"
    if settle_note:
        message = f"{message}；{settle_note}"
    return AnchorPlaceResult(
        state=affix_state,
        success=True,
        message=message,
    )


@dataclass(frozen=True)
class FinishAdvice:
    """「这一局还能不能凑满调色目标」的结论，以及凑不满时该怎么收尾。

    存在的理由：当调色目标已经凑不满时，继续按分数推荐落点毫无意义——
    玩家真正要决定的是「要不要用强制调色」和「什么时候点完成绘制」。
    这两件事都不能只看分数，要看**盘上有没有不可再生的颜料**。
    """

    #: 理论上还能凑满吗。
    reachable: bool
    #: 当前进度 / 目标进度。
    current: int
    required: int
    #: 范围内还剩几个能落子的格；范围未知时为 None。
    legal_cells: int | None
    #: 范围来源："observed"（金框高亮）/ "anchor"（按锚点推算）/ "unknown"。
    scope_source: str
    #: 盘上会被强制调色波及的贵重颜料：(行, 列, 等级)。
    treasures: tuple[tuple[int, int, int], ...]
    #: 凑不满时可以「随便塞」的合法空格（已排除调色范围本身）。
    fillers: tuple[tuple[int, int], ...]
    #: 一句话结论。
    headline: str
    #: 建议动作，按优先级排列。
    actions: tuple[str, ...]

    @property
    def summary(self) -> str:
        return "；".join((self.headline, *self.actions))


#: 值得为它放弃「凑满目标」的颜料等级门槛。7 级是通关目标，且 4005 抽到 7 级
#: 只有 1%（期望 3.54 级），丢了这一局就再也刷不出来。
TREASURE_GUARD_LEVEL = 7


def assess_finish_mode(state: PuzzleState) -> FinishAdvice | None:
    """判断当前调色目标还能不能凑满；凑不满就给出「保住贵重颜料、收尾结算」的建议。

    为什么需要它：官方 ``config_s8_gameplay_revive`` 的强制调色是
    「随机摧毁地图上 1/2/3 个颜料连同地块」，**摧毁对象不可指定**。
    盘上只要有一枚 7 级颜料，强制调色就是拿「必然存在的摧毁风险」去换
    「自动完成当前调色目标」——7 级是 1% 的产物，这笔账永远不划算。
    所以「凑不满」时唯一正确的动作是：随便塞完剩余颜料，点完成绘制结算。
    """
    rule = state.current_rule
    if rule is None:
        return None

    required = max(1, int(rule.match_required or 1))
    current = int(rule.match_current or 0)
    treasures = tuple(
        (row, col, int(tile.paint.level))
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
        for tile in (state.tile(row, col),)
        if tile.paint is not None
        and (tile.paint.level >= TREASURE_GUARD_LEVEL or tile.paint.is_treasure)
    )
    # 「随便塞完」也得知道往哪塞：全盘所有能落子的空格，排除调色范围本身。
    # 落点本身不摧毁任何东西，而且颜料自己的调色效果照样触发（比如
    # 「纵向的所有颜料提升1级」），所以塞在范围外并不是纯浪费。
    scope_now = _placement_scope_cells(state, rule)
    scope_set = set(scope_now)
    fillers = tuple(
        (row, col)
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
        if (row, col) not in scope_set and _is_placeable(state.tile(row, col), state.active_affix)
    )

    if current >= required:
        return FinishAdvice(
            reachable=True,
            current=current,
            required=required,
            legal_cells=None,
            scope_source="observed" if state.observed_scope_cells else "anchor",
            treasures=treasures,
            fillers=fillers,
            headline=f"调色目标已达成 {current}/{required}",
            actions=("可以点「完成绘制」结算，或继续铺颜料抬等级",),
        )

    if current_rule_is_pending(state):
        return FinishAdvice(
            reachable=True,
            current=current,
            required=required,
            legal_cells=None,
            scope_source="unknown",
            treasures=treasures,
            fillers=fillers,
            headline="还没落初始颜料",
            actions=("先选一个空白地块放入初始颜料",),
        )

    scoped = _placement_scope_cells(state, rule)
    if scoped:
        legal = tuple(
            cell for cell in scoped if _is_placeable(state.tile(cell[0], cell[1]), state.active_affix)
        )
        scope_source = "observed" if state.observed_scope_cells else "anchor"
        legal_cells: int | None = len(legal)
    else:
        legal = ()
        scope_source = "unknown"
        legal_cells = None

    target_paints = sum(1 for paint in state.backpack if _paint_matches_target(paint, rule.target_object))
    missing = required - current
    # 范围未知时不要断言「凑不满」——宁可说「看不出范围」，也不要把玩家往收尾上推。
    reachable = True if legal_cells is None else (current + min(legal_cells, target_paints) >= required)

    if reachable:
        return FinishAdvice(
            reachable=True,
            current=current,
            required=required,
            legal_cells=legal_cells,
            scope_source=scope_source,
            treasures=treasures,
            fillers=fillers,
            headline=(
                f"还差 {missing} 个（{current}/{required}），范围内可落子 {legal_cells} 个"
                if legal_cells is not None
                else f"还差 {missing} 个（{current}/{required}），调色范围没识别出来"
            ),
            actions=("按下面的落点继续铺，别动强制调色",),
        )

    actions: list[str] = []
    if treasures:
        levels = sorted({level for _row, _col, level in treasures}, reverse=True)
        cells = "、".join(f"r{row + 1}c{col + 1}(Lv{level})" for row, col, level in treasures)
        actions.append(
            f"**不要用强制调色**：它会随机摧毁 1~3 个颜料（含地块），盘上有 {cells}，"
            f"Lv{levels[0]} 被炸掉这一局就白打了"
        )
    else:
        actions.append("强制调色会随机摧毁 1~3 个颜料，收益不值，别用")
    actions.append("把背包里剩余颜料随便塞进任意合法地块（塞在不压高等级颜料的位置）")
    if fillers:
        shown = "、".join(f"r{row + 1}c{col + 1}" for row, col in fillers[:6])
        more = f" 等 {len(fillers)} 格" if len(fillers) > 6 else ""
        actions.append(f"可塞的空格：{shown}{more}")
    actions.append("点「完成绘制」结算，进入下一张画纸")
    return FinishAdvice(
        reachable=False,
        current=current,
        required=required,
        legal_cells=legal_cells,
        scope_source=scope_source,
        treasures=treasures,
        fillers=fillers,
        headline=f"调色目标凑不满了：{current}/{required}，范围内只剩 {legal_cells} 个可落子格",
        actions=tuple(actions),
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
    best = _search_install_plan(sim_state, rule)
    steps = best.steps if best is not None else ()
    completed_current = bool(best and best.completed)
    final_match = best.match_count if best is not None else initial_match
    if completed_current:
        reason = (
            f"按计划安装{len(steps)}步可完成{rule.match_required}/{rule.match_required}，"
            "完成后刷新下一张初始颜料"
        )
    elif steps:
        reason = (
            f"已推演{len(steps)}步，当前最多到{min(final_match, rule.match_required)}/{rule.match_required}，"
            "后续缺少可用目标颜料或可安装格"
        )
    else:
        reason = "当前没有可执行的安装方案，可能缺少目标颜料或金框内没有可安装格"
    return InstallPlan(steps=tuple(steps), completes_match=completed_current, reason=reason)


def _search_install_plan(state: PuzzleState, rule: CurrentPaintRule) -> _InstallSearchNode | None:
    initial_match = _current_match_count(state, rule)
    initial = _InstallSearchNode(
        state=state,
        steps=(),
        match_count=initial_match,
        score=0.0,
    )
    needed = max(0, rule.match_required - initial_match)
    if needed == 0:
        return replace(initial, completed=True)

    placeable = sum(
        1
        for row, col in _placement_scope_cells(state, rule)
        if _is_placeable(state.tile(row, col), state.active_affix)
    )
    supply = sum(1 for paint in state.backpack if _paint_matches_target(paint, rule.target_object))
    max_steps = min(needed, placeable, supply)
    if max_steps <= 0:
        return None

    planned_paints = sort_backpack_for_rule(
        tuple(paint for paint in state.backpack if _paint_matches_target(paint, rule.target_object)),
        rule,
    )[:max_steps]
    candidate_cells = _install_search_cells(state, rule, planned_paints[0], max_steps)
    initial_board_value = _board_position_value(state)
    frontier = [initial]
    best_partial = initial
    completed: list[_InstallSearchNode] = []
    beam_width = 16

    for paint in planned_paints:
        next_by_state: dict[tuple, _InstallSearchNode] = {}
        for node in frontier:
            current_rule = node.state.current_rule
            if current_rule is None:
                continue
            cells = tuple(
                (row, col)
                for row, col in candidate_cells
                if _is_placeable(node.state.tile(row, col), node.state.active_affix)
            )
            for row, col in cells:
                simulated = _simulate_install_without_next_rule(node.state, current_rule, row, col, paint)
                if simulated is None:
                    continue
                next_state, is_complete = simulated
                next_match = (
                    current_rule.match_required
                    if is_complete
                    else _current_match_count(next_state, current_rule)
                )
                if next_match <= node.match_count:
                    continue
                steps = (*node.steps, InstallStep(row=row, col=col, paint=paint))
                candidate = _InstallSearchNode(
                    state=next_state,
                    steps=steps,
                    match_count=next_match,
                    score=_install_plan_score(initial_board_value, next_state, steps),
                    completed=is_complete,
                )
                key = (
                    next_state.tiles,
                    next_state.backpack,
                    next_state.current_rule,
                    next_state.active_affix,
                )
                previous = next_by_state.get(key)
                if previous is None or _install_node_rank(candidate) > _install_node_rank(previous):
                    next_by_state[key] = candidate

        if not next_by_state:
            break
        next_nodes = list(next_by_state.values())
        completed.extend(node for node in next_nodes if node.completed)
        partials = [node for node in next_nodes if not node.completed]
        if partials:
            best_partial = max((best_partial, *partials), key=_install_node_rank)
        if completed:
            break
        partials.sort(key=_install_node_rank, reverse=True)
        frontier = partials[:beam_width]

    if completed:
        return max(completed, key=_install_node_rank)
    return best_partial if best_partial.steps else None


def _install_search_cells(
    state: PuzzleState,
    rule: CurrentPaintRule,
    paint: Paint,
    needed: int,
) -> tuple[tuple[int, int], ...]:
    cells = [
        (row, col)
        for row, col in _placement_scope_cells(state, rule)
        if _is_placeable(state.tile(row, col), state.active_affix)
    ]
    cells.sort(
        key=lambda cell: (
            _cell_priority(cell) * 12.0 + _placement_terrain_delta(state, cell[0], cell[1], paint),
            _cell_priority(cell),
            -cell[0],
            -cell[1],
        ),
        reverse=True,
    )
    if len(cells) <= 14:
        return tuple(cells)
    # Global/diagonal scopes can contain dozens of equivalent blanks. Keep a
    # generous shortlist so planning stays inside the live capture budget.
    limit = min(12, max(8, needed * 2 + 2))
    return tuple(cells[:limit])


def _install_plan_score(
    initial_board_value: float,
    current_state: PuzzleState,
    steps: tuple[InstallStep, ...],
) -> float:
    score = _board_position_value(current_state) - initial_board_value
    for step in steps:
        placed = current_state.tile(step.row, step.col).paint
        if placed is None:
            continue
        # Pair valuable paints with central, reusable cells after the set of
        # paints and the set of occupied cells have been chosen globally.
        score += _paint_strategic_value(placed) * _cell_priority((step.row, step.col)) * 0.08
    return score


def _install_node_rank(node: _InstallSearchNode) -> tuple:
    selected_paints = tuple(
        sorted(
            (
                (step.paint.level, int(step.paint.is_treasure), _paint_strategic_value(step.paint))
                for step in node.steps
            ),
            reverse=True,
        )
    )
    step_order = tuple(
        (step.paint.level, _paint_strategic_value(step.paint), _cell_priority((step.row, step.col)))
        for step in node.steps
    )
    return (node.match_count, selected_paints, node.score, step_order)


def _paint_with_self_upgrade(state: PuzzleState, paint: Paint) -> Paint:
    """把「自身提升N级」词缀结算到**要落下去的这枚**颜料上。

    ``self_upgrade_pending`` 是选择词缀时攒下的（见对应字段说明），落子这一刻才兑现。
    只动这一枚，不动背包里别的颜料——官方文案是「使自身」，范围类效果走另一条路。
    """
    delta = max(0, int(getattr(state, "self_upgrade_pending", 0) or 0))
    if not delta:
        return paint
    return replace(paint, level=min(MAX_LEVEL, int(paint.level) + delta))


def install_backpack_paint(
    state: PuzzleState,
    backpack_index: int,
    row: int,
    col: int,
    *,
    rng: random.Random | None = None,
) -> InstallResult:
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

    paint = _paint_with_self_upgrade(state, state.backpack[backpack_index])
    tiles = list(state.tiles)
    tiles[row * BOARD_SIZE + col] = Tile(
        state=TileState.PAINT,
        paint=paint,
        buff_applied_id=None,
        affix_applied_id=None,
        bonus_level=tile_bonus_level(target_tile),
        aura_amount=max(0, int(target_tile.aura_amount)),
    )
    backpack = sort_backpack_for_rule(
        tuple(item for index, item in enumerate(state.backpack) if index != backpack_index),
        rule,
    )

    # 词缀已经兑现到落下去的那枚颜料上了，下笔不再重复加。
    placed_state = replace(state, tiles=tuple(tiles), backpack=backpack, self_upgrade_pending=0)
    match_count = _current_match_count(placed_state, rule)
    upgraded_state, upgraded_count, scope_affix_count, settle_note = settle_place_effects(
        placed_state, rule, rng or random.Random(), placed=(row, col)
    )
    upgraded_state, persistent_affix_count = _apply_persistent_affix_after_install(upgraded_state)
    upgraded_state = _recompute_auras(upgraded_state)
    completed = match_count >= rule.match_required
    next_rule = replace(rule, match_current=min(match_count, rule.match_required))
    progress = f"{min(match_count, rule.match_required)}/{rule.match_required}"
    message = f"已安装{_paint_label(paint)}，安装即触发，范围内{upgraded_count}个目标颜料提升{rule.upgrade_amount}级，当前进度{progress}"
    scope_phrase = _scope_affix_phrase(scope_affix_count, state.active_affix)
    if scope_phrase:
        message = f"{message}；{scope_phrase}"
    if settle_note:
        message = f"{message}；{settle_note}"
    if persistent_affix_count:
        message = f"{message}；持续词缀额外提升{persistent_affix_count}个颜料"
    if completed:
        upgraded_state, next_rule_message = _advance_to_next_current_rule(upgraded_state, rng)
        message = f"完成匹配，{message}；{next_rule_message}"
    return InstallResult(
        state=replace(upgraded_state, current_rule=next_rule if not completed else upgraded_state.current_rule),
        success=True,
        completed_match=completed,
        message=message,
    )


def _simulate_install_without_next_rule(
    state: PuzzleState,
    rule: CurrentPaintRule,
    row: int,
    col: int,
    paint: Paint,
) -> tuple[PuzzleState, bool] | None:
    scope = set(_placement_scope_cells(state, rule))
    if (row, col) not in scope or not _is_placeable(state.tile(row, col), state.active_affix):
        return None
    # 推演必须和真实落子走同一套结算，否则「自身提升N级」在规划里等于没选。
    # 背包里要删的是**原对象**：升级后的 Paint 已不是同一个值，用它会删不掉。
    source_paint = paint
    paint = _paint_with_self_upgrade(state, paint)
    tiles = list(state.tiles)
    tiles[row * BOARD_SIZE + col] = Tile(
        state=TileState.PAINT,
        paint=paint,
        buff_applied_id=None,
        affix_applied_id=None,
        bonus_level=tile_bonus_level(state.tile(row, col)),
        aura_amount=max(0, int(state.tile(row, col).aura_amount)),
    )
    backpack = sort_backpack_for_rule(_remove_one_paint(state.backpack, source_paint), rule)
    placed_state = replace(state, tiles=tuple(tiles), backpack=backpack, self_upgrade_pending=0)
    match_count = _current_match_count(placed_state, rule)
    upgraded_state, _upgraded_count, _scope_affix_count, _note = settle_place_effects(
        placed_state, rule, random.Random(0), placed=(row, col)
    )
    upgraded_state, _persistent_affix_count = _apply_persistent_affix_after_install(upgraded_state)
    upgraded_state = _recompute_auras(upgraded_state)
    completed = match_count >= rule.match_required
    next_rule = replace(rule, match_current=min(match_count, rule.match_required))
    if completed:
        upgraded_state = replace(
            upgraded_state,
            current_rule=None,
            current_anchor=None,
            active_affix=None,
            current_buff_id=state.current_buff_id + 1,
        )
    else:
        upgraded_state = replace(upgraded_state, current_rule=next_rule)
    return upgraded_state, completed


def _remove_one_paint(backpack: tuple[Paint, ...], paint: Paint) -> tuple[Paint, ...]:
    removed = False
    result: list[Paint] = []
    for item in backpack:
        if not removed and item == paint:
            removed = True
            continue
        result.append(item)
    return tuple(result)


def backpack_paint_placed(state: PuzzleState) -> bool:
    """棋盘上是否已经有**玩家放上去的**颜料（不含初始锚点那枚）。

    官方的强制调色是"把画到一半的调色目标直接结算"，前提是已经动过手：
    只放了初始颜料时按钮不可用。所以开局想强制调色，必须先落一枚背包颜料，
    此时棋盘上共 2 枚颜料（1 枚初始 + 1 枚背包）。
    """
    anchor = state.current_anchor
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if anchor is not None and (row, col) == tuple(anchor):
                continue
            if state.tile(row, col).state is TileState.PAINT:
                return True
    return False


def force_complete_current_rule(
    state: PuzzleState,
    force_used: int,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> ForceCompleteResult:
    rule = state.current_rule
    if rule is None:
        return ForceCompleteResult(state=state, success=False, destroyed_cells=(), message="缺少当前初始颜料规则")
    if current_rule_is_pending(state):
        return ForceCompleteResult(state=state, success=False, destroyed_cells=(), message="新的初始颜料尚未放置，不能强制完成")
    if not backpack_paint_placed(state):
        return ForceCompleteResult(
            state=state,
            success=False,
            destroyed_cells=(),
            message="还没有放入任何颜料，不能强制调色（必须先放 1 枚背包颜料）",
        )

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
    rng = rng or random.Random(seed)
    # 第 N 次强制调色要炸 N 个颜料（1 / 2 / 3）。棋盘上颜料不足时按实际数量炸，
    # 但**必须如实报告**，不能静默降级——否则"第三次该炸 3 个"会变成悄悄只炸 2 个，
    # 复盘时看到的摧毁数与规则不符，排查会走错方向。
    actual_damage = min(damage_count, len(paint_cells))
    destroyed = tuple(sorted(rng.sample(paint_cells, actual_damage))) if paint_cells else ()
    tiles = list(upgraded_state.tiles)
    for row, col in destroyed:
        # 炸毁只炸颜料，地块加成留在原地——这是"炸完再放一枚直接 7 级"的来源。
        # 但锚点上的颜料没了，同种光环随之失效，别处同种颜料会掉级。
        tiles[row * BOARD_SIZE + col] = replace(
            tiles[row * BOARD_SIZE + col],
            state=TileState.FADED,
            paint=None,
            buff_applied_id=None,
            affix_applied_id=None,
        )
    damaged_state = _recompute_auras(replace(upgraded_state, tiles=tuple(tiles)))
    next_state, next_message = _advance_to_next_current_rule(damaged_state, rng)
    shortfall = damage_count - actual_damage
    if shortfall > 0:
        # 棋盘上可炸的颜料少于规则要求：说清楚，别让人以为规则只炸这么点。
        damage_text = (
            f"随机炸毁{len(destroyed)}格变为失色"
            f"（本次应炸{damage_count}个，棋盘上只剩{len(paint_cells)}个颜料）"
        )
    else:
        damage_text = f"随机炸毁{len(destroyed)}格变为失色"
    message = (
        f"已强制完成当前匹配，范围内{upgraded_count}个目标颜料完成本轮结算，"
        f"{damage_text}；{next_message}"
    )
    return ForceCompleteResult(
        state=next_state,
        success=True,
        destroyed_cells=destroyed,
        message=message,
    )


def _advance_to_next_current_rule(
    state: PuzzleState,
    rng: random.Random | None = None,
) -> tuple[PuzzleState, str]:
    rng = rng or random.Random()
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
    *,
    learning_stats: dict | None = None,
) -> list[BrushRecommendation]:
    normalized = normalize_brush_options(options)
    recommendations = [
        BrushRecommendation(
            brush_type=brush_type,
            score=_score_brush_type(state, brush_type) + _brush_policy_prior_score(learning_stats, brush_type),
            reason=_brush_reason(state, brush_type, _brush_policy_prior_score(learning_stats, brush_type)),
        )
        for brush_type in normalized
    ]
    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations


#: 宝箱局词缀优先级的可读标签（见 :func:`_treasure_affix_priority`）。
_TREASURE_PRIORITY_LABEL = {
    1: '宝箱首选：替换（拿更多宝箱）',
    2: '宝箱次选：同类型升级',
    3: '宝箱三选：自身提升',
    4: '宝箱四选：玩法提升',
    6: '宝箱末选：复制（收尾已无空位，铺不开）',
}


def _remaining_rule_slots(state: PuzzleState, remaining_rules: int | None = None) -> int:
    """还剩几次词缀选择机会（= 还能结算几条规则）。

    用户口径（2026-11-12）：地图 36 格，空白+阻挡约 20 来格，一条规则要
    1 枚初始颜料 + 3~5 个连接，所以一局大约能打 4~5 条规则，
    也就是 4~5 次选词缀的机会。

    这个数字决定"现在该选什么"：前期选打开局面/铺开的，收尾阶段选直接抬等级的。
    之前给词缀排一个固定优先级，等于假设"哪个词缀好"与时机无关——数据证明是错的。

    ``remaining_rules`` 由调用方（练习会话）给出；``PuzzleState`` 本身不含
    规则计数，所以不能从 state 里读。
    """
    if remaining_rules is not None:
        return max(0, int(remaining_rules))
    return 1


def _affix_stage(state: PuzzleState, remaining_rules: int | None = None) -> str:
    """当前处在早期还是收尾阶段。"""
    return 'opening' if _remaining_rule_slots(state, remaining_rules) >= 2 else 'closing'


def _treasure_affix_priority(
    state: PuzzleState,
    affix: AffixRule,
    settled_paint: Paint | None = None,
    remaining_rules: int | None = None,
) -> int | None:
    """宝箱局的**词缀优先级**（数值越小越优先），**分阶段**。

    用户规则（2026-11-12 确认并补充）：

    * 初始颜料是宝箱时，优先「替换」拿更多宝箱；
    * 但**如果已经到收尾、棋盘没空位了，再拿替换意义不大**——
      用户原话："如果最后来宝箱替换无更多空位意义也不大了"；
    * 前期（还剩 ≥2 条规则）优先"打开局面"的词缀；
    * 收尾（最后 1 条规则）优先"直接抬等级"的词缀。

    分阶段依据来自穷举实验（200 局 × 3 决策点，46% 可突破）：
    第 1 步最常成为最优解的是「全图玩法提升」「改方向」；
    第 3 步是「全图所有颜料提升」「范围内所有颜料提升」。

    ``settled_paint`` 是**刚结算那条规则**的初始颜料。必须显式传进来：
    选词缀时 ``state.current_rule`` 已经推进到下一条规则了，直接读它
    会把宝箱局误判成普通局（实测就是这么漏掉的）。

    返回 ``None`` 表示这条词缀不在宝箱优先级体系内（交给通用打分）。
    """
    paint = settled_paint if settled_paint is not None else (
        state.current_rule.paint if state.current_rule is not None else None
    )
    if paint is None or not paint.is_treasure:
        return None
    effect = affix.effect
    stage = _affix_stage(state, remaining_rules)
    is_all_paint = (
        effect is AffixEffect.UPGRADE_GLOBAL_CATEGORY
        and affix.target_object is PaintObject.ALL
    )
    if effect is AffixEffect.COPY_SELF_TO_SCOPE:
        # 收尾且没空位时，再复制也铺不开了，降级处理。
        if stage == 'closing' and _scope_blank_room(state) <= 1:
            return 6
        return 1
    if stage == 'closing' and is_all_paint:
        # 收尾阶段「全图所有颜料提升」是穷举里胜率最高的一条（66.7%），
        # 直接抬等级比继续铺开更值钱。
        return 2
    if effect in (AffixEffect.UPGRADE_SCOPE_SAME_PAINT, AffixEffect.UPGRADE_GLOBAL_SAME_PAINT):
        # 同类型升级要有"另一个宝箱"可升。注意初始颜料此时还没落到棋盘上，
        # 所以统计时要排除锚点格，只要**别处**还有宝箱就算成立。
        return 2 if _other_treasure_count(state) >= 1 else None
    if effect is AffixEffect.UPGRADE_SELF:
        return 3
    if effect is AffixEffect.UPGRADE_GLOBAL_CATEGORY and affix.target_object is PaintObject.GAMEPLAY:
        return 4
    return None


def _scope_blank_room(state: PuzzleState) -> int:
    """当前作用范围内还能落子的空格数；算不出来返回 0。"""
    rule = state.current_rule
    if rule is None or state.current_anchor is None:
        return 0
    try:
        cells = _placement_scope_cells(state, rule)
    except Exception:  # noqa: BLE001 - 估分不该因边界情况崩掉
        return 0
    return sum(
        1
        for row, col in cells
        if state.tile(row, col).state in (TileState.BLANK, TileState.FADED)
    )


def _other_treasure_count(state: PuzzleState) -> int:
    """棋盘上**除自身锚点格之外**的宝箱颜料数。

    初始颜料在选词缀时通常还没落到棋盘上；即使已经落下，锚点格上那枚
    就是"自身"，不能算作"其他宝箱"，否则同类型升级会被误判为空放。
    """
    anchor = state.current_anchor
    return sum(
        1
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
        if (row, col) != anchor
        and (tile := state.tile(row, col)).state is TileState.PAINT
        and tile.paint is not None
        and tile.paint.is_treasure
    )


def recommend_affixes(
    state: PuzzleState,
    options: tuple[AffixRule, ...] | list[AffixRule],
    *,
    learning_stats: dict | None = None,
    settled_paint: Paint | None = None,
    remaining_rules: int | None = None,
) -> list[AffixRecommendation]:
    recommendations = [
        AffixRecommendation(
            affix=affix,
            score=_score_affix(state, affix) + _affix_policy_prior_score(learning_stats, affix),
            reason=_affix_reason(state, affix, _affix_policy_prior_score(learning_stats, affix)),
        )
        for affix in options
    ]
    # 宝箱局有硬优先级：分数只在同一优先级内部用来排序，
    # 不能让"通用分高的杂项词缀"盖掉"替换（拿更多宝箱）"。
    priority = {
        item.affix: _treasure_affix_priority(
            state, item.affix, settled_paint, remaining_rules
        )
        for item in recommendations
    }
    if any(value is not None for value in priority.values()):
        # 已识别的 4 档严格按 1→4 排在前面；其余词缀排在它们之后、内部按分数排。
        recommendations.sort(
            key=lambda item: (
                priority[item.affix] if priority[item.affix] is not None else 90,
                -item.score,
            )
        )
        # AffixRecommendation 是 frozen，改写 reason 必须重建实例。
        annotated: list[AffixRecommendation] = []
        for item in recommendations:
            rank = priority[item.affix]
            if rank is None:
                annotated.append(item)
            else:
                # 用 get 兜底：优先级函数将来再加档位时，这里不该直接 KeyError
                # 把整份推荐搞崩（历史上 rank=6 就是这么炸的）。
                label = _TREASURE_PRIORITY_LABEL.get(rank, f'宝箱第{rank}选')
                annotated.append(replace(item, reason=f'{label}：{item.reason}'))
        return annotated
    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations


def recommend_special_actions(
    state: PuzzleState,
    resources: ResourceState,
    brush_scenario: BrushChoiceScenario | None = None,
    *,
    learning_stats: dict | None = None,
) -> list[SpecialActionRecommendation]:
    recommendations: list[SpecialActionRecommendation] = []
    force = _recommend_force_complete(state, resources, learning_stats=learning_stats)
    if force is not None:
        recommendations.append(force)
    if resources.refresh_available and resources.refresh_remaining > 0:
        # 实机确认：**第一阶段（画笔类型）无法刷新**，只能刷新第二阶段的词缀。
        # 所以这里只产出 REFRESH_STAGE2，不再生成 stage1 建议——
        # 给一个做不到的操作建议，比不给建议更糟。
        stage2 = _recommend_refresh_stage2(state, brush_scenario, learning_stats=learning_stats)
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


# 官方「额外提升」类笔触（2004~2008、2014~2016）在落笔瞬间才结算：
# 作用范围里必须包含刚放下的那枚颜料，否则实机上就是「范围内 0 个目标」。
DEFERRED_SCOPE_UPGRADE_EFFECTS = frozenset(
    {AffixEffect.UPGRADE_IN_SCOPE, AffixEffect.UPGRADE_SCOPE_SAME_PAINT}
)

#: 作用范围类 buff —— 实机通用规则（2026-11-12 用户确认）：
#:
#: * **落笔才结算**：选词缀时棋盘不变化，必须把初始颜料安装到地块才生效；
#: * **整局有效**：结算后 ``active_affix`` **不清空**，之后每次往范围内
#:   安装颜料都会重新结算一遍，除非某个地块被再次替换。
#:
#: 之前只有「额外提升」类按落笔结算，其余是"选完立刻改棋盘 + 用完即弃"，
#: 两个方向都错。这里统一成一个集合，不再逐条判断。
SCOPE_BUFF_EFFECTS = frozenset(
    {
        AffixEffect.UPGRADE_IN_SCOPE,
        AffixEffect.UPGRADE_SCOPE_SAME_PAINT,
        AffixEffect.COPY_SELF_TO_SCOPE,
        AffixEffect.RANDOMIZE_SCOPE_CATEGORY,
        AffixEffect.RANDOMIZE_SCOPE_ANY,
        AffixEffect.CLEAR_TILE_IN_SCOPE,
    }
)

#: 「覆盖」类作用范围效果 —— 改写范围内**颜料本身**（类型 / 等级），不是加等级。
#:
#: 官方文案（`S8_Selectable_Effect` 2009~2012）：
#:
#: * 2009 将作用范围内的颜料转变为**同等级**的随机颜料
#: * 2010 / 2011 同上，但限定为随机怪物 / 随机玩法颜料
#: * 2012 将作用范围内所有颜料的**等级和内容替换为自身**
#:
#: 这一类必须**先于调色效果结算**，理由见 :func:`settle_place_effects`。
COVER_EFFECTS = frozenset(
    {
        AffixEffect.RANDOMIZE_SCOPE_ANY,
        AffixEffect.RANDOMIZE_SCOPE_CATEGORY,
        AffixEffect.COPY_SELF_TO_SCOPE,
    }
)

#: 选完就消耗掉的一次性放置词条（3010~3013）：落一笔之后 ``active_affix`` 清空。
CONSUMABLE_PLACE_EFFECTS = frozenset(
    {
        AffixEffect.ALLOW_PLACE_ON_BLOCKED,
        AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_FADED,
    }
)


def is_cover_affix(affix: AffixRule | None) -> bool:
    """这条笔触是不是「覆盖」类（改写范围内颜料类型/等级）。

    界面与对账都靠它区分「这一笔会换掉场上颜料」和「这一笔只是加等级」——
    前者会让对账里的「同类型才吃调色」失效，不能按老值直接推。
    """
    return affix is not None and affix.effect in COVER_EFFECTS


def _scope_affix_phrase(count: int, affix: AffixRule | None, *, unit: str = "颜料") -> str:
    """这一笔笔触的影响措辞；没有影响就返回空串。

    为什么要按效果分流（9931fc5 把三种效果统一成一句「笔触影响范围内 N 个」，
    结果「额外提升」的**级数**被吞掉了，界面上看不出这一笔到底加了几级）：

    * **额外提升类**（``DEFERRED_SCOPE_UPGRADE_EFFECTS``）—— 官方文案是
      「额外提升 N 级」，所以必须把级数说出来，并点明它记在**地块加成**上；
    * **覆盖类**（``COVER_EFFECTS``）—— ``settle_note`` 里已经有
      「笔触改写范围内 N 枚颜料」，这里不重复；
    * **其余范围类**（清除地块等）—— 不是"提升"，只报影响范围。
    """
    if count <= 0 or affix is None:
        return ""
    if affix.effect in COVER_EFFECTS:
        return ""
    if _is_deferred_scope_upgrade(affix):
        amount = max(0, int(getattr(affix, "amount", 0) or 0))
        tail = f"（+{amount}级，记在地块加成）" if amount else ""
        return f"笔触额外提升{count}个{unit}{tail}"
    return f"笔触影响范围内{count}个{unit}"


def settle_place_effects(
    state: PuzzleState,
    rule: CurrentPaintRule,
    rng: random.Random,
    *,
    placed: tuple[int, int] | None = None,
) -> tuple[PuzzleState, int, int, str]:
    """一次落笔的结算：**覆盖 → 调色 → 其余范围 buff**。

    返回 ``(新状态, 调色提升的格数, 笔触额外影响的格数, 说明文字)``。

    为什么覆盖必须排在调色前面（用户 2026-11-22 实机口径）：

        3 级初始颜料；笔触 =「将作用范围内的颜料转变为同等级的随机颜料」；
        调色效果 =「使玩法颜料提升 1 级」；范围内玩家自己装了 4 枚颜料。

        实际结果：4 枚先被随机成 **3 个玩法 + 1 个怪物（仍是 3 级）**，
        随后调色按**新类型**只给那 3 个玩法 +1
        → **3 个 4 级玩法 + 1 个 3 级怪物**。

    反过来（先调色再覆盖）会得到 4 个 4 级、且覆盖把这次升级冲掉，
    与画面永远对不上。也就是说覆盖类笔触结算后，调色要**重新判定一次**——
    覆盖清掉了 ``buff_applied_id``，所以这一刻天然满足「重新吃一次调色」。

    非覆盖类（纯加等级 / 清地块）笔触不受顺序影响，仍按「调色 → 笔触」结算：
    调色先把等级顶高，笔触再用当时的等级判断 7 级封顶，与实机一致。
    """
    affix = state.active_affix
    covered = 0
    if affix is not None and affix.effect in COVER_EFFECTS:
        state, covered = _settle_scope_buff(state, rng, placed=placed)
        affix = state.active_affix

    state, upgraded = _apply_current_rule_upgrade(state, rule)

    extra = 0
    notes: list[str] = []
    if covered:
        notes.append(f"笔触改写范围内{covered}枚颜料")
    if affix is not None and affix.effect in SCOPE_BUFF_EFFECTS and affix.effect not in COVER_EFFECTS:
        state, extra = _settle_scope_buff(state, rng, placed=placed)
    elif affix is not None and affix.effect in CONSUMABLE_PLACE_EFFECTS:
        state = replace(state, active_affix=None)
        notes.append("特殊放置词条已消耗")
    return state, upgraded, covered + extra, "；".join(notes)


def _settle_scope_buff(
    state: PuzzleState,
    rng: random.Random,
    *,
    placed: tuple[int, int] | None = None,
) -> tuple[PuzzleState, int]:
    """按当前 ``active_affix`` 结算一次作用范围 buff。

    ``placed`` 是这一笔刚落下的坐标：作用范围里一个颜料都没有时，
    「随机 N 个提升」兜底提升的就是它（实机口径）。
    **不清空 ``active_affix``**：这几条是整局有效的 buff，每次落笔都要重算。
    返回 ``(新状态, 本次影响的格数)``。
    """
    affix = state.active_affix
    if affix is None or affix.effect not in SCOPE_BUFF_EFFECTS:
        return state, 0
    if affix.effect is AffixEffect.COPY_SELF_TO_SCOPE:
        return _copy_self_to_scope(state)
    if affix.effect is AffixEffect.UPGRADE_IN_SCOPE:
        return _upgrade_scope_paints(state, affix, rng, placed=placed)
    if affix.effect is AffixEffect.UPGRADE_SCOPE_SAME_PAINT:
        return _upgrade_scope_same_paints(state, affix)
    if affix.effect in {
        AffixEffect.RANDOMIZE_SCOPE_CATEGORY,
        AffixEffect.RANDOMIZE_SCOPE_ANY,
    }:
        return _randomize_scope_paints(state, affix, rng)
    if affix.effect is AffixEffect.CLEAR_TILE_IN_SCOPE:
        return _clear_scope_tiles(state, affix, rng)
    return state, 0


def _is_deferred_scope_upgrade(affix: AffixRule | None) -> bool:
    return affix is not None and affix.effect in DEFERRED_SCOPE_UPGRADE_EFFECTS


def _apply_deferred_scope_affix(
    state: PuzzleState,
    rng: random.Random,
    *,
    placed: tuple[int, int] | None = None,
) -> tuple[PuzzleState, int]:
    """落笔后结算作用范围 buff（**单步**，不含调色）。

    通用规则：**结算但不清空** ``active_affix``——这几条整局有效，
    每次落笔都要重算。旧实现会在这里清掉词缀，导致只生效一次。

    注意：真正的主流程走 :func:`settle_place_effects`——它负责「覆盖先、调色后」
    的顺序。本函数只保留给单步测试用，**不要再在落笔路径里直接调用**，
    否则覆盖类笔触会排到调色后面，等级算出来比实机高 1 级。
    """
    return _settle_scope_buff(state, rng, placed=placed)


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

    if affix.effect in SCOPE_BUFF_EFFECTS:
        # 通用规则（实机确认）：作用范围类 buff **落笔才结算**，且**整局有效**。
        # 选词缀时只挂 buff，不动棋盘——提前结算会得到 0 个目标（此时范围里
        # 还没有刚放下的颜料），等于白选。
        return AffixApplyResult(
            state=replace(state, active_affix=affix),
            success=True,
            message=f"已选择词缀：{affix.name}，将初始颜料安装到地块后按作用范围结算",
        )

    if affix.effect == AffixEffect.UPGRADE_SELF:
        # 「使自身等级提升N级」：只提升**这枚初始颜料自己**（用户实机口径）。
        # 两件事都要做：
        #   1. 记一笔 ``self_upgrade_pending``，落子时加到放下去那枚颜料上
        #      （这才是真正决定棋盘等级的地方，见 :attr:`PuzzleState.self_upgrade_pending`）；
        #   2. 同步改 ``rule.paint``，让面板/推演里看到的当前颜料等级也跟着变。
        delta = max(1, int(affix.amount))
        next_paint = replace(rule.paint, level=min(MAX_LEVEL, rule.paint.level + delta))
        next_state = _replace_current_rule(state, replace(rule, paint=next_paint), update_anchor=True)
        next_state = replace(
            next_state,
            self_upgrade_pending=max(0, int(state.self_upgrade_pending)) + delta,
        )
        return AffixApplyResult(
            state=next_state,
            success=True,
            message=f"已选择词缀：{affix.name}，落笔时这枚颜料自身 Lv{rule.paint.level} 提升到 Lv{next_paint.level}",
        )

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

    if affix.effect == AffixEffect.CLEAR_RANDOM_TILE_GLOBAL:
        next_state, count = _clear_global_tiles(state, affix, rng)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，全图{count}个地块变为空白")

    if affix.effect == AffixEffect.COPY_SELF_CONTENT_GLOBAL:
        next_state, count = _copy_self_content_global(state)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，全图{count}个颜料的内容替换为自身")

    if affix.effect == AffixEffect.RELOCATE_GLOBAL_BLOCKED:
        next_state, count = _relocate_global_blocked(state, rng)
        return AffixApplyResult(state=next_state, success=True, message=f"已选择词缀：{affix.name}，全图{count}个阻挡地块被挪走")

    if affix.effect in {
        AffixEffect.ALLOW_PLACE_ON_BLOCKED,
        AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_FADED,
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
    *,
    learning_stats: dict | None = None,
) -> SpecialActionRecommendation | None:
    rule = state.current_rule
    if rule is None or resources.force_remaining <= 0 or current_rule_is_pending(state):
        return None
    damage_count = min(3, resources.force_used + 1)
    reward = _force_completion_reward(rule)
    risk = _expected_force_damage(state, damage_count)
    natural = recommend_placements(state, limit=1)
    current_already_complete = _current_match_count(state, rule) >= rule.match_required
    natural_completes = current_already_complete or bool(natural and natural[0].completes_match)
    score = reward - risk
    protected_treasures = _protected_treasure_count(state)
    reason_parts = [
        f"立即完成当前匹配，预计收益{reward:.0f}",
        f"第{resources.force_used + 1}次会随机破坏{damage_count}个颜料，风险{risk:.0f}",
    ]
    if protected_treasures:
        score -= protected_treasures * float(config_value("force.protected_treasure_penalty", 260.0))
        reason_parts.append(f"棋盘已有{protected_treasures}个6/7级宝箱，强制破坏风险需保守")
    if natural_completes:
        score -= float(config_value("force.natural_completion_penalty", 900.0))
        reason_parts.append("已有自然完成落点，强制完成应保留")
    else:
        score += float(config_value("force.no_natural_completion_bonus", 450.0))
        reason_parts.append("当前没有自然完成，救场价值上升")
    if rule.paint.is_treasure:
        if rule.paint.level >= 6 and not natural_completes:
            score += float(config_value("force.high_treasure_rescue_bonus", 520.0))
            reason_parts.append("当前是高等级宝箱且自然完成困难，强制救场价值高")
        else:
            reason_parts.append("当前是宝箱，强制收益更高")
    bucket_adjust, bucket_reason = _resource_bucket_adjustment(
        learning_stats,
        state,
        good_bonus=float(config_value("force.good_bucket_bonus", 90.0)),
        bad_penalty=float(config_value("force.bad_bucket_penalty", 160.0)),
        allow_aggressive=_state_has_resource_opportunity(state),
    )
    score += bucket_adjust
    if bucket_reason:
        reason_parts.append(bucket_reason)
    recommended = score >= float(config_value("force.recommend_threshold", 350.0))
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
    *,
    learning_stats: dict | None = None,
) -> SpecialActionRecommendation | None:
    """**已废弃**：实机确认第一阶段（画笔类型）无法刷新。

    保留函数只是为了兼容旧调用方（``batch_evaluation`` 等处需要个签名），
    它**永远返回 None**，不再产出建议。判断逻辑一并保留在
    :func:`recommend_brush_types` 里，用于"这一阶段抽得好不好"的事后分析，
    但不会变成"去刷新它"的动作建议。
    """
    return None


def _recommend_refresh_stage2(
    state: PuzzleState,
    brush_scenario: BrushChoiceScenario | None,
    *,
    learning_stats: dict | None = None,
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
    rule = state.current_rule
    has_premium_treasure_affix = any(_is_treasure_premium_affix(affix) for affix in brush_scenario.stage2_options)
    if (
        rule is not None
        and rule.paint.is_treasure
        and brush_scenario.chosen_stage1 == BrushType.STACK
        and not has_premium_treasure_affix
    ):
        score += float(config_value("refresh.treasure_missing_premium_stage2_bonus", 520.0))
        reason += "；当前是宝箱堆砌但没出复制/同宝箱/随机玩法赌箱子，建议刷新二阶段"
    elif rule is not None and rule.paint.is_treasure and has_premium_treasure_affix:
        score -= float(config_value("refresh.treasure_has_premium_stage2_penalty", 760.0))
        reason += "；已命中宝箱优质词条，通常不再刷新"
    stage2_threshold = float(config_value("refresh.stage2_threshold", 120.0))
    if score >= stage2_threshold:
        reason += "；当前二阶段词条偏弱，可考虑刷新"
    else:
        reason += "；当前词条可接受，刷新价值不足"
    bucket_adjust, bucket_reason = _resource_bucket_adjustment(
        learning_stats,
        state,
        good_bonus=float(config_value("refresh.good_bucket_bonus", 80.0)),
        bad_penalty=float(config_value("refresh.bad_bucket_penalty", 130.0)),
        allow_aggressive=_state_has_resource_opportunity(state),
    )
    score += bucket_adjust
    if bucket_reason:
        reason += f"；{bucket_reason}"
    return SpecialActionRecommendation(
        action=SpecialAction.REFRESH_STAGE2,
        score=score,
        recommended=score >= stage2_threshold,
        reason=reason,
    )


def _is_treasure_premium_affix(affix: AffixRule) -> bool:
    if affix.effect == AffixEffect.COPY_SELF_TO_SCOPE:
        return True
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_SAME_PAINT:
        return True
    if affix.effect == AffixEffect.UPGRADE_SCOPE_SAME_PAINT:
        return True
    if affix.effect == AffixEffect.RANDOMIZE_SCOPE_CATEGORY and affix.target_object == PaintObject.GAMEPLAY:
        return True
    if affix.effect == AffixEffect.RANDOMIZE_SCOPE_ANY:
        return True
    return False


def _resource_bucket_adjustment(
    stats: dict | None,
    state: PuzzleState,
    *,
    good_bonus: float,
    bad_penalty: float,
    allow_aggressive: bool,
) -> tuple[float, str]:
    item = _terrain_bucket_item_for_state(stats, state)
    if item is None:
        return 0.0, ""
    total = int(item.get("total", 0) or 0)
    if total < int(config_value("learning.terrain_bucket_min_samples", 8)):
        return 0.0, "当前地形桶样本不足，资源建议不受历史分桶影响"
    success_rate = float(item.get("success_rate", 0.0) or 0.0)
    pct = float(item.get("success_pct", success_rate * 100.0) or 0.0)
    if success_rate >= 0.58 and allow_aggressive:
        return good_bonus, f"当前地形桶历史高分率{pct:.1f}%，且有宝箱/冲7机会，资源可略积极"
    if success_rate <= 0.32:
        return -bad_penalty, f"当前地形桶历史高分率{pct:.1f}%，资源建议更保守"
    return 0.0, f"当前地形桶历史高分率{pct:.1f}%，资源按常规评估"


def _terrain_bucket_item_for_state(stats: dict | None, state: PuzzleState) -> dict | None:
    if not stats:
        return None
    prior = stats.get("terrain_prior") if isinstance(stats.get("terrain_prior"), dict) else {}
    if prior.get("ready") is not True:
        return None
    buckets = prior.get("terrain_buckets") if isinstance(prior.get("terrain_buckets"), dict) else {}
    items = buckets.get("items") if isinstance(buckets.get("items"), list) else []
    key = _terrain_bucket_key_for_state(state)
    for item in items:
        if isinstance(item, dict) and item.get("key") == key:
            return item
    return None


def _state_has_resource_opportunity(state: PuzzleState) -> bool:
    rule = state.current_rule
    if rule is not None and (rule.paint.is_treasure or rule.paint.level >= 6):
        return True
    for tile in state.tiles:
        paint = tile.paint
        if paint is None:
            continue
        if paint.is_treasure and paint.level >= 5:
            return True
        if paint.level >= 6:
            return True
    return False


def _force_completion_reward(rule: CurrentPaintRule) -> float:
    next_level = min(MAX_LEVEL, rule.paint.level + rule.upgrade_amount)
    return 900.0 + _paint_upgrade_gain(rule.paint, next_level) * 1.35


def _expected_force_damage(state: PuzzleState, damage_count: int) -> float:
    paint_values = [_paint_damage_value(tile.paint) for tile in state.tiles if tile.state == TileState.PAINT]
    if not paint_values:
        return 0.0
    hits = min(damage_count, len(paint_values))
    average_value = sum(paint_values) / len(paint_values)
    tail_values = sorted(paint_values, reverse=True)[:hits]
    tail_average = sum(tail_values) / len(tail_values)
    high_value_pressure = sum(
        1
        for tile in state.tiles
        if tile.paint is not None and _paint_strategic_value(tile.paint) >= TREASURE_LEVEL_VALUES[5]
    )
    pressure_ratio = min(0.75, high_value_pressure / max(1, len(paint_values)))
    return hits * (average_value * (1.0 - pressure_ratio) + tail_average * pressure_ratio)


def _paint_damage_value(paint: Paint | None) -> float:
    if paint is None:
        return 320.0
    value = _paint_strategic_value(paint)
    if paint.is_treasure and paint.level >= 7:
        return value * 1.65
    if paint.is_treasure and paint.level >= 6:
        return value * 1.45
    if paint.is_treasure and paint.level >= 5:
        return value * 1.25
    if paint.level >= 7:
        return value * 1.15
    return value


def _protected_treasure_count(state: PuzzleState) -> int:
    return sum(
        1
        for tile in state.tiles
        if tile.paint is not None and tile.paint.is_treasure and tile.paint.level >= 6
    )


def _score_brush_type(state: PuzzleState, brush_type: BrushType) -> float:
    """笔触类型的推荐分。

    **2026-11-12 用户要求：三种笔触一律同分**，不掺任何先验偏置。

    之前这里给三种笔触各加了 baseline（转锋 18 / 堆砌 24 / 扩散 16）和
    宝箱局的 +420 策略加成，等于把"哪个笔触更好"直接写死。后果是永远测不出
    真实偏好——推荐器只会重复我自己的假设。

    现在一律返回 0，让三种笔触站在完全同一条起跑线上；真实偏好交给大批量
    随机模拟的结果去回答（``practice_batch --explore 1.0``）。
    等数据出来后再决定要不要、以及怎么加回偏置。
    """
    return 0.0


def _brush_strategy_bonus(state: PuzzleState, brush_type: BrushType) -> float:
    rule = state.current_rule
    if rule is None:
        return 0.0
    if rule.paint.is_treasure and brush_type == BrushType.STACK:
        return 420.0 + rule.paint.level * 35.0
    if rule.paint.is_treasure and brush_type == BrushType.TRANSFORM:
        return -90.0 if _current_rule_fit_score(state, rule) >= 0.55 else 0.0
    if rule.paint.is_treasure and brush_type == BrushType.SPREAD:
        blocked_pressure = _count_tiles(state, TileState.BLOCKED) + _count_tiles(state, TileState.VOID)
        return 80.0 if blocked_pressure >= 10 else -60.0
    return 0.0


# ---------------------------------------------------------------------------
# 实测数据驱动的推荐修正
#
# 来源：10000 局模拟（`practice_batch`）+ 800 局特征相关分析。与总分的相关系数：
#   l7 +0.666   ml +0.616   gc +0.345   tc +0.296   steps +0.139   mc -0.323
# 含义很明确：**等级和 7 级宝箱是分数的主因，怪物颜料是拖累**。
# 这些常数是"经验修正项"，量级压在 1.0~4.0，只用来打破近似平手，不覆盖地形判断。
# ---------------------------------------------------------------------------

#: 实测：玩法颜料每个约贡献 +0.345 相关度，怪物约 -0.323。
CATEGORY_PLACEMENT_BIAS: dict[str, float] = {
    'gameplay': 3.0,
    'monster': -3.0,
}

#: 实测：平均等级相关度 +0.616 —— 等级是硬通货，落一枚高等级比铺两枚低级更值。
LEVEL_PLACEMENT_BIAS = 1.2

#: 实测：7 级宝箱相关度 +0.666，是单点最高的收益形态。
LEVEL7_TREASURE_BIAS = 6.0


def placement_data_bias(state: PuzzleState, paint: Paint, row: int, col: int) -> float:
    """按实测相关度给出的落子倾向分。

    只处理"往哪落、落哪枚"这件事，**不碰地形与匹配逻辑**——那些由
    ``_score_candidate`` 负责。这样即使数据口径以后变了，也只影响这几个常数。
    """
    bias = 0.0
    bias += CATEGORY_PLACEMENT_BIAS.get(str(getattr(paint.category, 'value', paint.category)), 0.0)
    bias += LEVEL_PLACEMENT_BIAS * (int(paint.level) - 3)
    # 宝箱值得一路堆到上限：等级越高追加越多（与 l7 相关度最高的观察一致）。
    if getattr(paint, 'is_treasure', False):
        if int(paint.level) >= MAX_LEVEL:
            bias += LEVEL7_TREASURE_BIAS
        else:
            bias += 1.5 * int(paint.level) - 4.5
    tile = state.tile(row, col)
    # 已经有地块加成的格子，再放一枚等于白拿加成，实测这类格子更容易出高等级。
    if tile.bonus_level > 0:
        bias += 2.0 * int(tile.bonus_level)
    return bias


def _brush_reason(state: PuzzleState, brush_type: BrushType, prior_score: float = 0.0) -> str:
    suffix = f"；经验先验+{prior_score:.0f}" if prior_score >= 15 else ""
    if brush_type == BrushType.TRANSFORM:
        if state.current_rule and state.current_rule.match_required >= 4:
            return "可救场：改方向/改目标/降低匹配数" + suffix
        return "可修正调色目标，适合当前目标不顺时" + suffix
    if brush_type == BrushType.STACK:
        if state.current_rule and state.current_rule.paint.is_treasure:
            return "宝箱优先堆砌：赌复制/同宝箱变化/额外升级，除非地形严重卡死" + suffix
        if state.current_rule and state.current_rule.paint.level >= 5:
            return "冲级优先：当前颜料接近7级" + suffix
        return "提升收益稳定，适合做7级" + suffix
    if _count_tiles(state, TileState.BLOCKED) + _count_tiles(state, TileState.VOID) >= 8:
        return "清场优先：当前可用空间偏少" + suffix
    return "扩展落点，适合被阻挡/虚空卡住时" + suffix


# ---------------------------------------------------------------------------
# 官方权重表接入（依据：S8画中雪原_作画玩法_官方规则.md §4.3）
#
# 游戏把「随机」类效果的几率以 `档位:权重` 原文写在配置记录里。本项目此前一律按
# 等概率模拟、按固定常数估分，导致「赌随机等级」被系统性高估（实际 7 级只有 1%）。
# 这里把三张官方权重表同时接进**估分**与**模拟**两条链路，两处口径保持一致。
#
# 档位 -> 语义的对应按官方文案顺序推定（横向/纵向/对角/相邻/全图、怪物/玩法/所有），
# 未在游戏内逐档验证；表中的权重值本身是官方原文，未做任何改动。
# ---------------------------------------------------------------------------
_RANDOMIZE_LEVEL_OFFICIAL_ID = 4005       # 随机转变自身的等级，内容不变
_ADD_RANDOM_SCOPE_OFFICIAL_ID = 2019      # 随机增加一种作用范围
_CHANGE_TARGET_OBJECT_OFFICIAL_ID = 1006  # 转变自身的作用对象为另一类型

_OFFICIAL_SCOPE_BY_TIER: dict[int, PaintDirection] = {
    1: PaintDirection.HORIZONTAL,
    2: PaintDirection.VERTICAL,
    3: PaintDirection.DIAGONAL,
    4: PaintDirection.ADJACENT,
    5: PaintDirection.GLOBAL,
}
_OFFICIAL_TARGET_BY_TIER: dict[int, PaintObject] = {
    1: PaintObject.MONSTER,
    2: PaintObject.GAMEPLAY,
    3: PaintObject.ALL,
}

# 拿不到官方表时的保守回退（与接入前的旧常数一致，保证行为不劣化）
_RANDOMIZE_LEVEL_FALLBACK = 45.0
_ADD_RANDOM_SCOPE_FALLBACK = 120.0
_CHANGE_TARGET_OBJECT_FALLBACK = 55.0

# 作用范围里每格的价值：命中调色目标 > 可落点空格 > 已有颜料
_SCOPE_CELL_TARGET_WEIGHT = 46.0
_SCOPE_CELL_ROOM_WEIGHT = 17.0
_SCOPE_CELL_PAINT_WEIGHT = 9.0

# 一次性不可逆的赌博：对期望上行打折，对下行风险再加一层惩罚
_GAMBLE_UPSIDE_DISCOUNT = 0.85
_GAMBLE_DOWNSIDE_PENALTY = 0.55

_SCOPE_DIRECTION_ORDER: tuple[PaintDirection, ...] = (
    PaintDirection.HORIZONTAL,
    PaintDirection.VERTICAL,
    PaintDirection.DIAGONAL,
    PaintDirection.ADJACENT,
    PaintDirection.GLOBAL,
)


def _official_option_probabilities(official_id: int) -> dict[int, float]:
    """官方权重表 -> ``{档位: 概率}``；缺表返回空 dict，由调用方回退到旧常数。"""
    try:
        from .s8_official_data import affix_option_distribution
    except Exception:  # pragma: no cover - 官方数据模块缺失不应影响主流程
        return {}
    return affix_option_distribution(official_id)


def _official_scope_probabilities() -> dict[PaintDirection, float]:
    """「随机增加一种作用范围」的官方分布：四个方向各约 24.4%，全图仅 2.44%。"""
    probabilities = _official_option_probabilities(_ADD_RANDOM_SCOPE_OFFICIAL_ID)
    return {
        _OFFICIAL_SCOPE_BY_TIER[tier]: probability
        for tier, probability in probabilities.items()
        if tier in _OFFICIAL_SCOPE_BY_TIER
    }


def _official_target_probabilities() -> dict[PaintObject, float]:
    """「作用对象改为另一类型」的官方分布：怪物/玩法各约 47.6%，所有仅 4.76%。"""
    probabilities = _official_option_probabilities(_CHANGE_TARGET_OBJECT_OFFICIAL_ID)
    return {
        _OFFICIAL_TARGET_BY_TIER[tier]: probability
        for tier, probability in probabilities.items()
        if tier in _OFFICIAL_TARGET_BY_TIER
    }


def _weighted_option(
    rng: random.Random,
    options: tuple,
    probabilities: dict,
    fallback: object | None = None,
) -> object | None:
    """按官方概率在候选中抽取一个。

    候选若被官方表排除（例如某方向已存在、旧目标不可再抽），按**剩余候选重新归一化**；
    若整张表缺失或候选全部不在表内，退回等概率抽取。
    """
    if not options:
        return fallback
    weights = [max(0.0, probabilities.get(option, 0.0)) for option in options]
    if sum(weights) <= 0:
        return rng.choice(list(options))
    return rng.choices(list(options), weights=weights, k=1)[0]


def _randomize_current_level_score(state: PuzzleState) -> float:
    """官方 4005 分布下「随机改变自身等级」的期望收益。

    官方分布：Lv1 7% / Lv2 18% / Lv3 20% / Lv4 30% / Lv5 20% / Lv6 4% / Lv7 1%，
    期望 **3.54 级**，赌到 7 级只有 **1%**。因为一次性且不可逆，对下行风险额外折价：
    当前已是 7 级时必然只降不升，得分归零。
    """
    rule = state.current_rule
    if rule is None:
        return 0.0
    probabilities = _official_option_probabilities(_RANDOMIZE_LEVEL_OFFICIAL_ID)
    if not probabilities:
        return _RANDOMIZE_LEVEL_FALLBACK
    paint = rule.paint
    current_value = _paint_strategic_value(paint)
    expected_value = 0.0
    downside = 0.0
    for level, probability in probabilities.items():
        level = max(1, min(MAX_LEVEL, level))
        value = _paint_strategic_value(replace(paint, level=level))
        expected_value += probability * value
        if level < paint.level:
            downside += probability * (value - current_value)
    expected_gain = expected_value - current_value
    score = max(0.0, expected_gain) * _GAMBLE_UPSIDE_DISCOUNT + downside * _GAMBLE_DOWNSIDE_PENALTY
    return max(0.0, score)


def _best_scope_coverage_value(
    state: PuzzleState,
    extra_direction: PaintDirection | None = None,
) -> float:
    """当前初始颜料在最优落点上，作用范围所覆盖格子的加权价值。

    ``extra_direction`` 非空时把该方向并入范围，用于衡量「多一种作用范围」的**边际**收益。
    """
    rule = state.current_rule
    if rule is None:
        return 0.0
    directions = (rule.direction, *rule.extra_directions)
    if extra_direction is not None and extra_direction not in directions:
        directions = (*directions, extra_direction)
    best = 0.0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if not _is_current_anchor_placeable(state.tile(row, col), state.active_affix):
                continue
            cells: list[tuple[int, int]] = []
            for direction in directions:
                cells.extend(cell for cell in _scope_cells(row, col, direction) if cell != (row, col))
            value = 0.0
            for r, c in dict.fromkeys(cells):
                tile = state.tile(r, c)
                if _tile_matches_target(tile, rule.target_object):
                    value += _SCOPE_CELL_TARGET_WEIGHT
                elif tile.state == TileState.PAINT:
                    value += _SCOPE_CELL_PAINT_WEIGHT
                elif _is_placeable(tile, state.active_affix):
                    value += _SCOPE_CELL_ROOM_WEIGHT
            best = max(best, value)
    return best


def _random_scope_addition_score(state: PuzzleState, affix: AffixRule) -> float:
    """官方 2019 分布下「随机增加一种作用范围」的期望收益。

    官方分布：横向 / 纵向 / 对角 / 相邻各约 **24.4%**，全图仅 **2.44%**。
    所以不能按"平均多一种范围"定价，必须逐个范围估值再加权 —— 这也正是把
    「抽到全图」当成常态的老做法被高估的原因。
    """
    probabilities = _official_scope_probabilities()
    if not probabilities:
        return _ADD_RANDOM_SCOPE_FALLBACK
    base_value = _best_scope_coverage_value(state)
    expected = 0.0
    for direction, probability in probabilities.items():
        gain = _best_scope_coverage_value(state, direction) - base_value
        expected += probability * max(0.0, gain)
    return expected


def _change_target_object_score(state: PuzzleState, affix: AffixRule) -> float:
    """官方 1006 分布下「作用对象改为另一类型」的期望收益。

    官方分布：怪物 / 玩法各约 47.6%，**所有仅 4.76%**；且抽签排除当前类型后重新归一化。
    价值用「当前局面与该目标的匹配度」之差衡量，避免只看类别名字就给定分。
    """
    rule = state.current_rule
    if rule is None:
        return 0.0
    if affix.target_object is not None:
        return _CHANGE_TARGET_OBJECT_FALLBACK
    probabilities = {
        target: probability
        for target, probability in _official_target_probabilities().items()
        if target != rule.target_object
    }
    total = sum(probabilities.values())
    if total <= 0:
        return _CHANGE_TARGET_OBJECT_FALLBACK
    current_fit = _current_rule_fit_score(state, rule)
    expected_gain = 0.0
    for target, probability in probabilities.items():
        fit = _current_rule_fit_score(state, replace(rule, target_object=target))
        expected_gain += (probability / total) * max(0.0, fit - current_fit)
    return _CHANGE_TARGET_OBJECT_FALLBACK + expected_gain * 320.0


def _score_affix(state: PuzzleState, affix: AffixRule) -> float:
    rule = state.current_rule
    if rule is None:
        return 0.0

    if affix.effect == AffixEffect.MATCH_REQUIRED_DELTA:
        return 140.0 if rule.match_required >= 4 else 70.0
    if affix.effect == AffixEffect.CHANGE_DIRECTION:
        return _direction_affix_score(state, affix)
    if affix.effect == AffixEffect.CHANGE_TARGET_OBJECT:
        return _change_target_object_score(state, affix)
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
        # 基础项按官方范围分布对每种范围分别估值；供给与门槛项沿用原口径。
        return (
            _random_scope_addition_score(state, affix)
            + _count_category_paints(state, rule.target_object) * 18.0
            - max(0, affix.delta) * 35.0
        )
    if affix.effect == AffixEffect.UPGRADE_SELF:
        return _level_gain_score(rule.paint, affix.amount, multiplier=1.15)
    if affix.effect == AffixEffect.UPGRADE_IN_SCOPE:
        return (
            _best_scope_upgrade_gain(state, affix, amount=affix.amount, count=affix.count) * 0.70
            # 已叠加成的格子吃到升级的边际收益远高于普通格子
            + _buffed_scope_gain(state, affix.amount) * 0.85
        )
    if affix.effect == AffixEffect.UPGRADE_SCOPE_SAME_PAINT:
        return (
            _scope_same_upgrade_gain(state, affix.amount) * 0.82
            + _buffed_scope_gain(state, affix.amount) * 0.55
        )
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_CATEGORY:
        return (
            _global_category_upgrade_gain(state, affix.target_object, affix.amount) * 0.80
            + (180.0 if affix.persistent_until_match else 0.0)
            + _buffed_global_gain(state, affix.amount) * 0.65
        )
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_SAME_PAINT:
        return (
            _global_same_upgrade_gain(state, affix.amount) * 0.92
            + (220.0 if affix.persistent_until_match else 0.0)
            + _buffed_global_gain(state, affix.amount) * 0.85
            + _treasure_affix_bonus(state, affix)
        )
    if affix.effect == AffixEffect.COPY_SELF_TO_SCOPE:
        return _copy_self_value_gain(state) * 1.15 + _treasure_affix_bonus(state, affix)
    if affix.effect == AffixEffect.RANDOMIZE_SCOPE_CATEGORY:
        return _randomize_scope_expected_gain(state, affix.target_object) * 0.75 + _treasure_affix_bonus(state, affix)
    if affix.effect == AffixEffect.RANDOMIZE_SCOPE_ANY:
        return _randomize_scope_expected_gain(state, PaintObject.ALL) * 0.58 + _treasure_affix_bonus(state, affix)
    if affix.effect == AffixEffect.CLEAR_TILE_IN_SCOPE:
        clearable = _best_clearable_count(state, affix.target_object or PaintObject.BLOCKED_TILE, affix.direction or rule.direction)
        count = affix.count if affix.count is not None else clearable
        return min(clearable, count) * 95.0
    if affix.effect == AffixEffect.CLEAR_RANDOM_TILE_GLOBAL:
        tile_state = _target_object_to_tile_state(affix.target_object)
        clearable = _count_tiles(state, tile_state) if tile_state else _count_tiles(state, TileState.FADED)
        count = affix.count if affix.count is not None else clearable
        return min(clearable, count) * (90.0 if tile_state != TileState.VOID else 75.0)
    if affix.effect == AffixEffect.COPY_SELF_CONTENT_GLOBAL:
        return _global_same_upgrade_gain(state, 1) * 0.9 + _copy_self_value_gain(state) * 0.35
    if affix.effect == AffixEffect.RELOCATE_GLOBAL_BLOCKED:
        return _count_tiles(state, TileState.BLOCKED) * 62.0
    if affix.effect == AffixEffect.ALLOW_PLACE_ON_BLOCKED:
        return _count_tiles(state, TileState.BLOCKED) * 38.0 + 45.0
    if affix.effect == AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF:
        return _count_tiles(state, TileState.VOID) * 25.0 + 30.0
    if affix.effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF:
        return _count_tiles(state, TileState.PAINT) * 20.0 + 25.0
    if affix.effect == AffixEffect.ALLOW_PLACE_ON_FADED:
        return _count_tiles(state, TileState.FADED) * 34.0 + 35.0
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_LEVEL:
        return _randomize_current_level_score(state)
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


def _treasure_affix_bonus(state: PuzzleState, affix: AffixRule) -> float:
    rule = state.current_rule
    if rule is None or not rule.paint.is_treasure or affix.brush_type != BrushType.STACK:
        return 0.0
    if affix.effect == AffixEffect.COPY_SELF_TO_SCOPE:
        return 900.0 + rule.paint.level * 120.0
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_SAME_PAINT:
        return 520.0 + affix.amount * 180.0
    if affix.effect == AffixEffect.RANDOMIZE_SCOPE_CATEGORY and affix.target_object == PaintObject.GAMEPLAY:
        return 360.0
    if affix.effect == AffixEffect.RANDOMIZE_SCOPE_ANY:
        return 240.0
    return 0.0


def _affix_reason(state: PuzzleState, affix: AffixRule, prior_score: float = 0.0) -> str:
    rule = state.current_rule
    suffix = f"；经验先验+{prior_score:.0f}" if prior_score >= 15 else ""
    if affix.effect == AffixEffect.MATCH_REQUIRED_DELTA:
        return "降低完成门槛，救场价值高" + suffix
    if affix.effect == AffixEffect.CHANGE_DIRECTION:
        return f"改为{_direction_label(affix.direction or PaintDirection.UNKNOWN)}，寻找更高匹配范围" + suffix
    if affix.effect == AffixEffect.REROLL_CURRENT_RULE:
        return "重新随机一张初始颜料，当前目标/地形不顺时救场" + suffix
    if affix.effect == AffixEffect.CHANGE_TARGET_OBJECT:
        return "作用对象改为另一类型（官方权重：怪物/玩法各约47.6%、所有仅4.76%）" + suffix
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_LEVEL:
        return "随机改变当前初始颜料等级（官方权重：期望3.54级、7级仅1%），低等级才值得赌" + suffix
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_DIRECTION:
        return "随机改变当前初始颜料方向，适合当前范围卡住时救场" + suffix
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_CONTENT:
        return "随机改变当前初始颜料内容但保留等级，可赌更高价值类型" + suffix
    if affix.effect == AffixEffect.ADD_RANDOM_SCOPE:
        return "增加一种随机作用范围（官方权重：四向各约24.4%、全图仅2.44%），但提高进度要求" + suffix
    if affix.effect == AffixEffect.UPGRADE_SELF:
        return f"自身提升{affix.amount}级，直接冲级" + suffix
    if affix.effect == AffixEffect.UPGRADE_SCOPE_SAME_PAINT:
        if rule and rule.paint.is_treasure:
            return "范围内同宝箱额外提升，适合已有同宝箱时继续冲级" + suffix
        return "范围内同内容颜料额外提升，适合同类已铺开时冲级" + suffix
    if affix.effect == AffixEffect.COPY_SELF_TO_SCOPE:
        if rule and rule.paint.is_treasure:
            return "宝箱复制潜力最高：赌把范围内颜料变成同宝箱" + suffix
        return "把范围内颜料替换为当前颜料，适合复制高等级" + suffix
    if affix.effect == AffixEffect.RANDOMIZE_SCOPE_CATEGORY and affix.target_object == PaintObject.GAMEPLAY:
        if rule and rule.paint.is_treasure:
            return "随机玩法可赌宝箱，当前初始是宝箱时可作为次优赌法" + suffix
        return "随机玩法可赌宝箱" + suffix
    if affix.effect in {AffixEffect.CLEAR_TILE_IN_SCOPE, AffixEffect.CLEAR_RANDOM_TILE_GLOBAL}:
        return "清理特殊地块，扩大后续空间" + suffix
    if affix.effect in {
        AffixEffect.ALLOW_PLACE_ON_BLOCKED,
        AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_FADED,
    }:
        return "增加特殊落点，注意可能失去自身内容" + suffix
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_CATEGORY:
        return "疑似持续全图升级，按实际7级/宝箱价值收益估分" + suffix
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_SAME_PAINT:
        if rule and rule.paint.is_treasure:
            return "全图同宝箱提升，当前宝箱局优先级很高" + suffix
        return "全图同名/同宝箱颜料提升，适合当前颜料价值高或同类多时" + suffix
    return "按当前局面估算收益" + suffix


def build_affix_catalog() -> tuple[AffixRule, ...]:
    legacy_catalog = (
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
        # 这两条与下面「对调色范围内的随机 N 个颜料额外提升1级」语义键相同，会被合并；
        # 官方文案是「额外提升」，所以合并后保留的那条也按"加地块加成"处理。
        AffixRule(BrushType.STACK, "范围内随机1个颜料提升1级", AffixEffect.UPGRADE_IN_SCOPE, amount=1, count=1, grants_bonus=True),
        AffixRule(BrushType.STACK, "范围内随机2个颜料提升1级", AffixEffect.UPGRADE_IN_SCOPE, amount=1, count=2, grants_bonus=True),
        # 官方 2006/2007 原文是「对作用范围内的所有颜料**额外**提升N级」，
        # 与 2008 同族，必须按地块加成记账（见 s8_official_data.S8_SELECTABLE_EFFECTS）。
        AffixRule(BrushType.STACK, "范围内所有颜料提升1级", AffixEffect.UPGRADE_IN_SCOPE, amount=1, grants_bonus=True),
        AffixRule(BrushType.STACK, "范围内所有颜料提升2级", AffixEffect.UPGRADE_IN_SCOPE, amount=2, grants_bonus=True),
        AffixRule(BrushType.STACK, "对调色范围内的随机1个颜料额外提升1级", AffixEffect.UPGRADE_IN_SCOPE, amount=1, count=1, grants_bonus=True),
        AffixRule(BrushType.STACK, "对调色范围内的随机2个颜料额外提升1级", AffixEffect.UPGRADE_IN_SCOPE, amount=1, count=2, grants_bonus=True),
        AffixRule(BrushType.STACK, "对调色范围内与自身内容相同的颜料额外提升1级", AffixEffect.UPGRADE_SCOPE_SAME_PAINT, amount=1, grants_bonus=True),
        AffixRule(BrushType.STACK, "对调色范围内与自身内容相同的颜料额外提升2级", AffixEffect.UPGRADE_SCOPE_SAME_PAINT, amount=2, grants_bonus=True),
        AffixRule(BrushType.STACK, "自身提升1级", AffixEffect.UPGRADE_SELF, amount=1),
        AffixRule(BrushType.STACK, "自身提升2级", AffixEffect.UPGRADE_SELF, amount=2),
        AffixRule(BrushType.STACK, "自身提升3级", AffixEffect.UPGRADE_SELF, amount=3),
        AffixRule(BrushType.STACK, "全图玩法颜料提升1级", AffixEffect.UPGRADE_GLOBAL_CATEGORY, target_object=PaintObject.GAMEPLAY, amount=1, persistent_until_match=True),
        AffixRule(BrushType.STACK, "全图玩法颜料提升2级", AffixEffect.UPGRADE_GLOBAL_CATEGORY, target_object=PaintObject.GAMEPLAY, amount=2, persistent_until_match=True),
        AffixRule(BrushType.STACK, "全图怪物颜料提升1级", AffixEffect.UPGRADE_GLOBAL_CATEGORY, target_object=PaintObject.MONSTER, amount=1, persistent_until_match=True),
        AffixRule(BrushType.STACK, "全图怪物颜料提升2级", AffixEffect.UPGRADE_GLOBAL_CATEGORY, target_object=PaintObject.MONSTER, amount=2, persistent_until_match=True),
        # 官方 2014/2015 同样带「额外提升」，与 2016 同族。
        AffixRule(BrushType.STACK, "全图和自身相同的颜料提升1级", AffixEffect.UPGRADE_GLOBAL_SAME_PAINT, amount=1, persistent_until_match=True, grants_bonus=True),
        AffixRule(BrushType.STACK, "全图和自身相同的颜料提升2级", AffixEffect.UPGRADE_GLOBAL_SAME_PAINT, amount=2, persistent_until_match=True, grants_bonus=True),
        AffixRule(BrushType.STACK, "范围内颜料类型与等级替换为自身", AffixEffect.COPY_SELF_TO_SCOPE),
        AffixRule(BrushType.STACK, "范围内颜料变成随机玩法类型颜料", AffixEffect.RANDOMIZE_SCOPE_CATEGORY, target_object=PaintObject.GAMEPLAY),
        AffixRule(BrushType.STACK, "范围内颜料变成随机怪物类型颜料", AffixEffect.RANDOMIZE_SCOPE_CATEGORY, target_object=PaintObject.MONSTER),
        AffixRule(BrushType.STACK, "将作用范围内的颜料转变为同等级的随机怪物颜料", AffixEffect.RANDOMIZE_SCOPE_CATEGORY, target_object=PaintObject.MONSTER),
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
        AffixRule(BrushType.SPREAD, "全图随机2块虚空地块变为空白地块", AffixEffect.CLEAR_RANDOM_TILE_GLOBAL, target_object=PaintObject.VOID_TILE, count=2),
        AffixRule(BrushType.SPREAD, "当前颜料可放置在阻挡地块", AffixEffect.ALLOW_PLACE_ON_BLOCKED),
        AffixRule(BrushType.SPREAD, "当前颜料可放置在虚空地块但失去自身内容", AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF, persistent_until_match=True),
        AffixRule(BrushType.SPREAD, "当前颜料可放置在已放置颜料地块但失去自身内容", AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF, persistent_until_match=True),
        AffixRule(BrushType.SPREAD, "可放置于失色地块中", AffixEffect.ALLOW_PLACE_ON_FADED),
        # ---- 以下条目依据官方 S8_Gameplay_SelectableEffect 补齐（见 s8_official_data）----
        AffixRule(BrushType.TRANSFORM, "调色目标的进度要求减少3", AffixEffect.MATCH_REQUIRED_DELTA, delta=-3),
        AffixRule(BrushType.STACK, "对作用范围内的所有颜料额外提升3级", AffixEffect.UPGRADE_IN_SCOPE, amount=3, grants_bonus=True),
        AffixRule(BrushType.STACK, "对全图中与自身内容相同的颜料额外提升3级", AffixEffect.UPGRADE_GLOBAL_SAME_PAINT, amount=3, persistent_until_match=True, grants_bonus=True),
        AffixRule(BrushType.STACK, "将全图中所有颜料的内容替换为自身", AffixEffect.COPY_SELF_CONTENT_GLOBAL),
        AffixRule(BrushType.STACK, "随机增加一种作用范围，调色进度要求增加1", AffixEffect.ADD_RANDOM_SCOPE, delta=1),
        AffixRule(BrushType.STACK, "随机增加一种作用范围", AffixEffect.ADD_RANDOM_SCOPE, delta=0),
        AffixRule(BrushType.STACK, "使全图中所有颜料的等级提升1级", AffixEffect.UPGRADE_GLOBAL_CATEGORY, target_object=PaintObject.ALL, amount=1, persistent_until_match=True),
        AffixRule(BrushType.SPREAD, "全图随机2块阻挡地块变为空白地块", AffixEffect.CLEAR_RANDOM_TILE_GLOBAL, target_object=PaintObject.BLOCKED_TILE, count=2),
        AffixRule(BrushType.SPREAD, "全图所有阻挡地块变为空白地块", AffixEffect.CLEAR_RANDOM_TILE_GLOBAL, target_object=PaintObject.BLOCKED_TILE),
        AffixRule(BrushType.SPREAD, "随机转变全图所有阻挡地块的位置至其他空白地块", AffixEffect.RELOCATE_GLOBAL_BLOCKED, target_object=PaintObject.BLOCKED_TILE),
        AffixRule(BrushType.SPREAD, "全图随机1块虚空地块变为空白地块", AffixEffect.CLEAR_RANDOM_TILE_GLOBAL, target_object=PaintObject.VOID_TILE, count=1),
        AffixRule(BrushType.SPREAD, "范围内所有失色地块变为空白地块", AffixEffect.CLEAR_TILE_IN_SCOPE, target_object=PaintObject.FADED_TILE),
        AffixRule(BrushType.SPREAD, "将距离最近的1个失色地块转变为空白地块", AffixEffect.CLEAR_TILE_IN_SCOPE, target_object=PaintObject.FADED_TILE, count=1, nearest=True),
    )
    return _canonicalize_affix_catalog(legacy_catalog)


_CANONICAL_AFFIX_ID_BY_LEGACY_CODE: dict[str, str] = {
    "1-1": "transform.direction.horizontal",
    "1-2": "transform.direction.vertical",
    "1-3": "transform.direction.diagonal",
    "1-4": "transform.direction.adjacent",
    "1-5": "transform.direction.global",
    "1-6": "transform.target.other_type",
    "1-7": "transform.match_required.minus_2",
    "1-8": "transform.match_required.minus_1",
    "1-9": "transform.current_level.random",
    "1-11": "transform.current_direction.random",
    "1-12": "transform.current_content.random",
    "1-13": "transform.current_rule.reroll",
    "1-15": "transform.randomize_scope.any",
    "1-16": "transform.randomize_scope.gameplay",
    "1-17": "transform.randomize_scope.monster",
    "2-1": "stack.scope.add_random.match_plus_2",
    "2-2": "stack.upgrade_scope.random_1.lv1",
    "2-3": "stack.upgrade_scope.random_2.lv1",
    "2-4": "stack.upgrade_scope.all.lv1",
    "2-5": "stack.upgrade_scope.all.lv2",
    "2-8": "stack.upgrade_scope.same_paint.lv1",
    "2-9": "stack.upgrade_scope.same_paint.lv2",
    "2-10": "stack.upgrade_self.lv1",
    "2-11": "stack.upgrade_self.lv2",
    "2-12": "stack.upgrade_self.lv3",
    "2-13": "stack.upgrade_global.gameplay.lv1",
    "2-14": "stack.upgrade_global.gameplay.lv2",
    "2-15": "stack.upgrade_global.monster.lv1",
    "2-16": "stack.upgrade_global.monster.lv2",
    "2-17": "stack.upgrade_global.same_paint.lv1",
    "2-18": "stack.upgrade_global.same_paint.lv2",
    "2-19": "stack.copy_self_to_scope",
    "2-20": "stack.randomize_scope.gameplay",
    "2-21": "stack.randomize_scope.monster",
    "2-23": "stack.randomize_scope.any",
    "3-1": "spread.clear_scope.blocked.horizontal.random_1",
    "3-2": "spread.clear_scope.blocked.vertical.random_1",
    "3-3": "spread.clear_scope.blocked.adjacent.random_1",
    "3-4": "spread.clear_scope.blocked.diagonal.random_1",
    "3-5": "spread.clear_scope.blocked.nearest_1",
    "3-7": "spread.clear_scope.blocked.all",
    "3-8": "spread.clear_global.blocked.random_1",
    "3-9": "spread.clear_global.faded.random_1",
    "3-10": "spread.clear_global.void.random_2",
    "3-11": "spread.place_on.blocked",
    "3-12": "spread.place_on.void.lose_self",
    "3-13": "spread.place_on.paint.lose_self",
    "3-14": "spread.place_on.faded",
    # 官方补齐条目（见 s8_official_data，对应官方记录 1012 / 2008 / 2016 / 2013 / 2018 / 2019 / 2022 / 3008~3018）
    "1-18": "transform.match_required.minus_3",
    "2-25": "stack.upgrade_scope.all.lv3",
    "2-26": "stack.upgrade_global.same_paint.lv3",
    "2-27": "stack.copy_self_content_global",
    "2-28": "stack.scope.add_random.match_plus_1",
    "2-29": "stack.scope.add_random",
    "2-30": "stack.upgrade_global.all.lv1",
    "3-15": "spread.clear_global.blocked.random_2",
    "3-16": "spread.clear_global.blocked.all",
    "3-17": "spread.relocate_global.blocked",
    "3-18": "spread.clear_global.void.random_1",
    "3-19": "spread.clear_scope.faded",
    "3-20": "spread.clear_nearest.faded",
}


# 官方 S8_Gameplay_SelectableEffect 记录 ID -> 规范化词缀 ID。
# 59 条官方记录收敛到 52 种语义：4001~4007 与 1xxx/2xxx 完全重复。
_OFFICIAL_SEMANTIC_BY_ID: dict[int, str] = {
    # 转锋
    1002: "transform.direction.horizontal",
    4001: "transform.direction.horizontal",
    1003: "transform.direction.vertical",
    1005: "transform.direction.diagonal",
    4003: "transform.direction.diagonal",
    1004: "transform.direction.adjacent",
    4002: "transform.direction.adjacent",
    1001: "transform.current_direction.random",
    1007: "transform.current_content.random",
    1008: "transform.current_level.random",
    4005: "transform.current_level.random",
    1009: "transform.current_rule.reroll",
    4004: "transform.current_rule.reroll",
    1006: "transform.target.other_type",
    1010: "transform.match_required.minus_1",
    1011: "transform.match_required.minus_2",
    1012: "transform.match_required.minus_3",
    # 堆砌
    2001: "stack.upgrade_self.lv1",
    4006: "stack.upgrade_self.lv1",
    2002: "stack.upgrade_self.lv2",
    2003: "stack.upgrade_self.lv3",
    2004: "stack.upgrade_scope.random_1.lv1",
    4007: "stack.upgrade_scope.random_1.lv1",
    2005: "stack.upgrade_scope.random_2.lv1",
    2006: "stack.upgrade_scope.all.lv1",
    2007: "stack.upgrade_scope.all.lv2",
    2008: "stack.upgrade_scope.all.lv3",
    2009: "stack.randomize_scope.any",
    2010: "stack.randomize_scope.monster",
    2011: "stack.randomize_scope.gameplay",
    2012: "stack.copy_self_to_scope",
    2013: "stack.copy_self_content_global",
    2014: "stack.upgrade_global.same_paint.lv1",
    2015: "stack.upgrade_global.same_paint.lv2",
    2016: "stack.upgrade_global.same_paint.lv3",
    2017: "stack.scope.add_random.match_plus_2",
    2018: "stack.scope.add_random.match_plus_1",
    2019: "stack.scope.add_random",
    2020: "stack.upgrade_global.monster.lv1",
    2021: "stack.upgrade_global.gameplay.lv1",
    2022: "stack.upgrade_global.all.lv1",
    # 扩散
    3001: "spread.clear_scope.blocked.adjacent.random_1",
    3002: "spread.clear_scope.blocked.horizontal.random_1",
    3003: "spread.clear_scope.blocked.vertical.random_1",
    3004: "spread.clear_scope.blocked.diagonal.random_1",
    3005: "spread.clear_scope.blocked.nearest_1",
    3006: "spread.clear_scope.blocked.all",
    3007: "spread.clear_global.blocked.random_1",
    3008: "spread.clear_global.blocked.random_2",
    3009: "spread.clear_global.blocked.all",
    3010: "spread.place_on.blocked",
    3011: "spread.relocate_global.blocked",
    3012: "spread.place_on.void.lose_self",
    3013: "spread.place_on.paint.lose_self",
    3014: "spread.clear_global.void.random_1",
    3015: "spread.clear_global.void.random_2",
    3016: "spread.clear_scope.faded",
    3017: "spread.place_on.faded",
    3018: "spread.clear_nearest.faded",
}

_OFFICIAL_IDS_BY_SEMANTIC: dict[str, tuple[int, ...]] = {}
for _official_id, _semantic_id in sorted(_OFFICIAL_SEMANTIC_BY_ID.items()):
    _OFFICIAL_IDS_BY_SEMANTIC.setdefault(_semantic_id, ())
    _OFFICIAL_IDS_BY_SEMANTIC[_semantic_id] += (_official_id,)


def _official_text_by_id() -> dict[int, str]:
    from .s8_official_data import S8_SELECTABLE_EFFECTS

    return {item.id: item.text for item in S8_SELECTABLE_EFFECTS}


def _canonicalize_affix_catalog(legacy_catalog: tuple[AffixRule, ...]) -> tuple[AffixRule, ...]:
    counters: dict[BrushType, int] = {brush_type: 0 for brush_type in BrushType}
    canonical: dict[tuple[object, ...], AffixRule] = {}
    for affix in legacy_catalog:
        counters[affix.brush_type] += 1
        legacy_code = f"{BRUSH_TYPE_ORDER.index(affix.brush_type) + 1}-{counters[affix.brush_type]}"
        key = _affix_semantic_key(affix)
        existing = canonical.get(key)
        if existing is None:
            canonical[key] = replace(
                affix,
                canonical_affix_id=_CANONICAL_AFFIX_ID_BY_LEGACY_CODE[legacy_code],
                legacy_codes=(legacy_code,),
            )
            continue
        aliases = existing.aliases
        if affix.name != existing.name and affix.name not in aliases:
            aliases += (affix.name,)
        canonical[key] = replace(
            existing,
            aliases=aliases,
            legacy_codes=existing.legacy_codes + (legacy_code,),
        )
    return _attach_official_metadata(tuple(canonical.values()))


def _attach_official_metadata(catalog: tuple[AffixRule, ...]) -> tuple[AffixRule, ...]:
    """把官方 S8 记录 ID / 官方文案挂到目录条目上，并标记是否有官方出处。

    - ``official_ids``：该语义在官方表中的记录 ID（4001~4007 等重复项会一并列出）
    - ``official_text``：官方 EffectDes 原文，同时加入 ``aliases`` 以便按游戏内文字识别
    - ``verified``：官方表里是否存在对应记录；``False`` 表示早期凭实机推断、官方无出处
    """
    texts = _official_text_by_id()
    enriched: list[AffixRule] = []
    for affix in catalog:
        ids = _OFFICIAL_IDS_BY_SEMANTIC.get(affix.canonical_affix_id, ())
        text = texts.get(ids[0], "") if ids else ""
        aliases = affix.aliases
        if text and text != affix.name and text not in aliases:
            aliases += (text,)
        enriched.append(
            replace(
                affix,
                aliases=aliases,
                official_ids=ids,
                official_text=text,
                verified=bool(ids),
            )
        )
    return tuple(enriched)


def _affix_semantic_key(affix: AffixRule) -> tuple[object, ...]:
    """合并"同一语义"的 legacy 条目时用的键。

    ``grants_bonus`` 必须参与：它决定提升记在**颜料等级**还是**地块加成**上，
    是两条不同的规则。早先漏掉这一项，导致同一键里 legacy 条目先到先得，
    把官方「额外提升」条目（2006/2007/2014/2015）并进普通提升条目后
    丢掉了 ``grants_bonus=True``。
    """
    return (
        affix.brush_type,
        affix.effect,
        affix.direction,
        affix.target_object,
        affix.delta,
        affix.amount,
        affix.count,
        affix.persistent_until_match,
        affix.nearest,
        affix.grants_bonus,
    )


def _random_paint(rng: random.Random) -> Paint:
    category = rng.choice([PaintCategory.MONSTER, PaintCategory.GAMEPLAY])
    level = _draw_official_level(rng, minimum=3)
    subtype = _random_subtype(rng, category, level)
    return Paint(category=category, level=level, subtype=subtype, is_treasure=subtype == "宝箱")


def _random_subtype(rng: random.Random, category: PaintCategory, level: int | None = None) -> str:
    """按官方 PlotItem 表抽颜料家族。

    官方表里家族与等级段是**绑定**的（见
    :data:`s8_official_data.S8_PARTIAL_LEVEL_FAMILIES`）：
    ``黑帆 / 黑潮 / 迷城 / 梦魇 / 怪谈 / 魔方 / 玩偶`` 只存在 5~7 级，
    ``赛季`` 只存在 3~4 级。旧实现从 7 个自选家族里等概率抽，
    既漏掉了 19 个 1~7 级通吃的家族，也会造出现实里不存在的「1 级迷城」。
    传入 ``level`` 时只在该等级真实存在的家族里抽。
    """
    key = 'MONSTER' if category == PaintCategory.MONSTER else 'GAMEPLAY'
    families = [
        name for name, meta in s8_official_data.S8_PAINT_FAMILIES.items()
        if meta['category'] == key
        and (level is None or level in meta['levels'])
    ]
    if families:
        return rng.choice(families)
    # 官方表不可用时回退旧硬编码。
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
    if tile.state == TileState.FADED and affix.effect == AffixEffect.ALLOW_PLACE_ON_FADED:
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
    if tile.state == TileState.FADED and affix.effect == AffixEffect.ALLOW_PLACE_ON_FADED:
        return True
    return False


def _tile_after_current_anchor_placement(tile: Tile, paint: Paint, affix: AffixRule | None) -> Tile:
    # 地块上的加成和光环锚点都跟着地块走，不跟着颜料走：
    # 初始颜料被炸掉、地块恢复后换一枚新颜料，光环就改挂到新颜料上。
    bonus = tile_bonus_level(tile)
    aura = max(0, int(tile.aura_amount))

    def _placed(state: TileState = TileState.PAINT, new_paint: Paint | None = paint) -> Tile:
        return Tile(
            state=state,
            paint=new_paint,
            buff_applied_id=None,
            affix_applied_id=None,
            bonus_level=bonus,
            aura_amount=aura,
            aura_level=0,
        )

    if affix is None:
        return _placed()
    if tile.state == TileState.BLOCKED and affix.effect == AffixEffect.ALLOW_PLACE_ON_BLOCKED:
        return _placed()
    if tile.state == TileState.VOID and affix.effect == AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF:
        return _placed(TileState.FADED, None)
    if tile.state == TileState.PAINT and affix.effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF:
        return tile
    if tile.state == TileState.FADED and affix.effect == AffixEffect.ALLOW_PLACE_ON_FADED:
        return _placed()
    return _placed()


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
    source = candidates if candidates else state.backpack
    unique: list[Paint] = []
    seen: set[tuple | None] = set()
    for paint in source:
        key = _paint_recommendation_key(paint)
        if key in seen:
            continue
        seen.add(key)
        unique.append(paint)
    return tuple(unique)


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
    """结算初始颜料的「调色效果：范围内目标提升 N 级」。

    **加成落在地块上，不落在颜料身上**（用户 2026-11-22 实机口径）。

    证据链：用户这局 c4r5 原本是 3 级怪物，被第三个初始颜料转成 3 级机械之神，
    随后吃到第二个初始颜料（横向玩法 +2）与第三个（相邻玩法 +1），最终显示 6 级；
    且**点击该地块，游戏提示的正是 +3** —— 2+1=3，说明这两笔加成是记在地块上的。

    之前实现写 ``paint.level``，那格颜料一被覆盖/转换，先前积累的加成跟着旧颜料一起
    消失，模型只能算出 3 级，而画面是 6 级、``breakdown`` 还是空的，对账天天报红。
    """

    tiles = list(state.tiles)
    upgraded_count = 0
    for row, col in _placement_scope_cells(state, rule):
        index = row * BOARD_SIZE + col
        tile = tiles[index]
        if tile.paint is None or not _tile_matches_target(tile, rule.target_object):
            continue
        if not _tile_can_receive_current_buff(tile, state):
            continue
        tiles[index] = replace(
            _tile_with_upgrade(tile, max(0, int(rule.upgrade_amount)), as_bonus=True),
            buff_applied_id=state.current_buff_id,
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


def _paint_strategic_value(paint: Paint) -> float:
    level = max(1, min(MAX_LEVEL, paint.level))
    if paint.is_treasure:
        if level >= 7:
            return TREASURE_LEVEL_VALUES[7]
        if level == 6:
            return TREASURE_LEVEL_VALUES[6]
        if level == 5:
            return TREASURE_LEVEL_VALUES[5]
        return {1: 260.0, 2: 420.0, 3: 620.0, 4: 820.0}.get(level, 260.0)
    if paint.category == PaintCategory.GAMEPLAY:
        return {1: 25.0, 2: 60.0, 3: 120.0, 4: 220.0, 5: 380.0, 6: 560.0, 7: GAMEPLAY_7_VALUE}[level]
    if paint.category == PaintCategory.MONSTER:
        return {1: 18.0, 2: 44.0, 3: 88.0, 4: 160.0, 5: 280.0, 6: 410.0, 7: MONSTER_7_VALUE}[level]
    return level * 45.0


def _paint_upgrade_gain(paint: Paint, next_level: int) -> float:
    next_paint = replace(paint, level=max(1, min(MAX_LEVEL, next_level)))
    return max(0.0, _paint_strategic_value(next_paint) - _paint_strategic_value(paint))


def _paint_level_gain(paint: Paint, from_level: int, to_level: int) -> float:
    """从 ``from_level`` 升到 ``to_level`` 的估值增量（地块加成也算进等级）。"""
    start = max(1, min(MAX_LEVEL, from_level))
    return max(0.0, _paint_strategic_value(replace(paint, level=max(1, min(MAX_LEVEL, to_level))))
               - _paint_strategic_value(replace(paint, level=start)))


def _tile_strategic_value(tile: Tile) -> float:
    """地块上这枚颜料的实际估值：按「颜料等级 + 地块加成」的最终等级算。"""
    effective = tile_effective_paint(tile)
    return 0.0 if effective is None else _paint_strategic_value(effective)


# ---------------------------------------------------------------------------
# 已有 buff 地块的追加收益
#
# 用户指出的缺口（2026-11-12）：前 4 条优先级只是"围绕初始颜料"选词缀，
# 但一局里真正值钱的是**已经被 buff 叠过的地块**——那些格子吃到一次升级
# 就能冲到 7 级，和普通格子完全不是一个量级。之前的词缀估值只看"当前等级"，
# 完全没把"这格已经存了 +3 加成"算成额外价值，等于对最高价值的格子视而不见。
# ---------------------------------------------------------------------------

def _buffed_scope_gain(state: PuzzleState, amount: int = 1) -> float:
    """范围内**已叠了地块加成/光环**的颜料再升一级能多赚多少。

    与普通升级的区别：加成越高的格子，升一级的边际收益越大（7 级墙前尤其猛），
    所以这里按每格实际最终等级逐格算增量，而不是用平均值。
    """
    rule = state.current_rule
    if rule is None:
        return 0.0
    best = 0.0
    for scope in _candidate_scope_sets(state):
        value = 0.0
        for row, col in scope:
            tile = state.tile(row, col)
            if tile.state != TileState.PAINT or tile.paint is None:
                continue
            buff = tile_bonus_level(tile) + tile_aura_level(tile)
            if buff <= 0:
                continue
            current_final = tile_final_level(tile)
            if current_final >= MAX_LEVEL:
                continue
            gain = _paint_level_gain(
                tile.paint, current_final, min(MAX_LEVEL, current_final + max(1, amount))
            )
            # 加成越高说明这格越是被专门养出来的，给一点额外权重
            value += gain * (1.0 + buff * 0.25)
        best = max(best, value)
    return best


def _buffed_global_gain(state: PuzzleState, amount: int = 1) -> float:
    """全图上已叠加成的颜料再升一级的边际收益。"""
    total = 0.0
    for tile in state.tiles:
        if tile.state != TileState.PAINT or tile.paint is None:
            continue
        buff = tile_bonus_level(tile) + tile_aura_level(tile)
        if buff <= 0:
            continue
        current_final = tile_final_level(tile)
        if current_final >= MAX_LEVEL:
            continue
        gain = _paint_level_gain(
            tile.paint, current_final, min(MAX_LEVEL, current_final + max(1, amount))
        )
        total += gain * (1.0 + buff * 0.25)
    return total


def _level_gain_score(paint: Paint, amount: int, multiplier: float = 1.0) -> float:
    next_level = min(MAX_LEVEL, paint.level + amount)
    score = _paint_upgrade_gain(paint, next_level)
    if score <= 0:
        return amount * 12.0 * multiplier
    return score * multiplier


def _candidate_scope_sets(
    state: PuzzleState,
    direction: PaintDirection | None = None,
    *,
    include_extra_directions: bool = True,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    rule = state.current_rule
    if rule is None:
        return ()
    if state.current_anchor is not None:
        anchors = (state.current_anchor,)
    else:
        anchors = tuple(
            (row, col)
            for row in range(BOARD_SIZE)
            for col in range(BOARD_SIZE)
            if _is_current_anchor_placeable(state.tile(row, col), state.active_affix)
        )
    scopes: list[tuple[tuple[int, int], ...]] = []
    for row, col in anchors:
        directions = [direction or rule.direction]
        if direction is None and include_extra_directions:
            directions.extend(rule.extra_directions)
        cells: list[tuple[int, int]] = []
        for item in directions:
            cells.extend(cell for cell in _scope_cells(row, col, item) if cell != (row, col))
        scopes.append(tuple(dict.fromkeys(cells)))
    return tuple(scopes)


def _best_scope_upgrade_gain(
    state: PuzzleState,
    affix: AffixRule,
    *,
    amount: int,
    count: int | None = None,
) -> float:
    best = 0.0
    step = max(1, amount)
    # 落笔瞬间结算时，刚放下的那枚颜料也在范围内，是稳定命中的一个目标。
    placed_gain = _paint_upgrade_gain(
        Paint(category=PaintCategory.MONSTER, level=3),
        min(MAX_LEVEL, 3 + step),
    )
    for scope in _candidate_scope_sets(state, affix.direction):
        gains = [
            _paint_level_gain(tile.paint, tile_final_level(tile), min(MAX_LEVEL, tile_final_level(tile) + step))
            for row, col in scope
            if (tile := state.tile(row, col)).state == TileState.PAINT and tile.paint is not None
        ]
        pool = gains + [placed_gain]
        if count is None:
            value = sum(pool)
        else:
            value = (sum(pool) / len(pool)) * min(count, len(pool))
        best = max(best, value)
    return best if best > 0 else placed_gain


def _scope_same_upgrade_gain(state: PuzzleState, amount: int) -> float:
    rule = state.current_rule
    if rule is None:
        return 0.0
    best = 0.0
    for scope in _candidate_scope_sets(state):
        value = 0.0
        for row, col in scope:
            tile = state.tile(row, col)
            if tile.paint is None or not _is_same_paint_kind(tile.paint, rule.paint):
                continue
            value += _paint_level_gain(tile.paint, tile_final_level(tile), min(MAX_LEVEL, tile_final_level(tile) + max(1, amount)))
        best = max(best, value)
    return best


def _global_category_upgrade_gain(
    state: PuzzleState,
    target: PaintObject | None,
    amount: int,
) -> float:
    return sum(
        _paint_level_gain(tile.paint, tile_final_level(tile), min(MAX_LEVEL, tile_final_level(tile) + max(1, amount)))
        for tile in state.tiles
        if tile.paint is not None and _paint_matches_target(tile.paint, target or PaintObject.ALL)
    )


def _global_same_upgrade_gain(state: PuzzleState, amount: int) -> float:
    rule = state.current_rule
    if rule is None:
        return 0.0
    return sum(
        _paint_level_gain(tile.paint, tile_final_level(tile), min(MAX_LEVEL, tile_final_level(tile) + max(1, amount)))
        for tile in state.tiles
        if tile.paint is not None and _is_same_paint_kind(tile.paint, rule.paint)
    )


def _copy_self_value_gain(state: PuzzleState) -> float:
    """「范围内颜料替换为自身」的期望收益。

    实机规则（2026-11-12 确认）：这是**整局有效的 buff**，该值两处：

    1. 范围内**已有**的异类颜料 → 变成自身；
    2. 之后**落进范围内**的颜料 → 也变成自身。

    旧实现只算第 1 项。选词缀那一刻范围内通常还没有颜料，于是它恒等于 0，
    推荐器把这条永远排在最后、从不选择——实测 5 万局里从未走到
    "宝箱 + 全图 + 替换为自身"这条最高分路线。这里把第 2 项补上。
    """
    rule = state.current_rule
    if rule is None:
        return 0.0
    current_value = _paint_strategic_value(rule.paint)
    # 本规则还要再放几枚才完成；这就是"之后会落进范围"的枚数。
    remaining = max(0, rule.match_required - rule.match_current)
    # 本来会放上去的通常是低阶颜料，用一个保守基准代表它
    baseline = _paint_strategic_value(
        Paint(
            category=PaintCategory.GAMEPLAY,
            level=max(1, min(MAX_LEVEL, rule.paint.level - 2)),
            subtype='存活',
        )
    )
    best = 0.0
    for scope in _candidate_scope_sets(state):
        value = 0.0
        existing = 0
        blanks = 0
        for row, col in scope:
            tile = state.tile(row, col)
            if tile.state == TileState.PAINT and tile.paint is not None:
                existing += 1
                value += current_value - _paint_strategic_value(tile.paint)
            elif tile.state == TileState.BLANK:
                blanks += 1
        # 之后落进范围内的颜料也会变成自身
        future = min(blanks, max(remaining, 1))
        value += future * (current_value - baseline)
        if rule.paint.is_treasure:
            value += (existing + future) * current_value * 0.18
        best = max(best, value)
    return best


def _randomize_scope_expected_gain(state: PuzzleState, target: PaintObject | None) -> float:
    """「范围内颜料变成随机 XX 类型」的期望收益。

    实机规则（2026-11-12 用户确认），**分成两种局面**：

    * 范围内**没有**宝箱 → 这是在赌，价值 = 随机后期望 − 现在；
    * 范围内**有**宝箱 → 宝箱会被覆盖成别的玩法颜料，等级也可能变。
      这不是赌，是**明确损失**，必须给负分，否则推荐器会把"毁掉已有宝箱"
      当成没影响。

    旧实现一律按"随机后期望 − 现在"算，所以覆写宝箱的损失根本看不出来。
    """
    best: float | None = None
    for scope in _candidate_scope_sets(state):
        value = 0.0
        for row, col in scope:
            tile = state.tile(row, col)
            if tile.state != TileState.PAINT or tile.paint is None:
                continue
            now = _paint_strategic_value(tile.paint)
            if tile.paint.is_treasure:
                # 宝箱被覆盖：等级会重掷，所以不能假定保级，按放大后的差算损失。
                expected = _expected_randomized_paint_value(
                    tile_final_level(tile), target
                )
                value += (expected - now) * 1.35
            else:
                expected = _expected_randomized_paint_value(
                    tile_final_level(tile), target
                )
                value += expected - now
        # 不能用 0 当下限：覆写宝箱是**净损失**，取 max 会把负分抹成 0，
        # 推荐器就永远看不到"这条会毁掉宝箱"。
        best = value if best is None else max(best, value)
    return 0.0 if best is None else best


def _expected_randomized_paint_value(level: int, target: PaintObject | None) -> float:
    level = max(1, min(MAX_LEVEL, level))
    monster_value = _paint_strategic_value(Paint(PaintCategory.MONSTER, level, subtype="随机怪物"))
    gameplay_value = _paint_strategic_value(Paint(PaintCategory.GAMEPLAY, level, subtype="随机玩法"))
    treasure_value = _paint_strategic_value(Paint(PaintCategory.GAMEPLAY, level, subtype="宝箱", is_treasure=True))
    treasure_chance = 0.03
    gameplay_expected = gameplay_value * (1.0 - treasure_chance) + treasure_value * treasure_chance
    if target == PaintObject.MONSTER:
        return monster_value
    if target == PaintObject.GAMEPLAY:
        return gameplay_expected
    return monster_value * 0.50 + gameplay_expected * 0.50


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
    if target == PaintObject.FADED_TILE:
        return TileState.FADED
    return None


def _placement_scope_cells(state: PuzzleState, rule: CurrentPaintRule) -> tuple[tuple[int, int], ...]:
    """调色的作用范围 = **纯几何**（锚点所在行/列/斜线/相邻/全图，去掉锚点自己）。

    为什么不再用「画面金框」收窄（用户 2026-11-15 追问后定的，别再改回去）：

    * 金框只画在**能放颜料**的格子上（``vision.is_scope_highlight`` 要求该格是空白或
      看起来像空白），虚空（``S8_Nihility``）、失色（``S8_Ruins``）这类不可放置的格子
      本来就不亮框；
    * 扩散可以把虚空/失色转成空白地块（官方 3014/3015「将全图随机 N 个虚空地块转变为
      空白地块」）。转完之后它就是范围内的普通空白格，落笔时理应吃范围加成 ——
      一旦按金框收窄，这些格子会被永久判成「不在范围内」，加成凭空少算；
    * 收窄原本的作用是纠正方向识别，而方向已经由
      ``vision._correct_current_paint_from_scope`` 用高亮格单独推断并校正过，
      这里再收窄不增加信息，只会把识别漏掉的格子一起吞掉。

    ``current_anchor is None``（还没放初始颜料）时返回识别到的高亮格：
    那时没有锚点就算不出几何，全空比瞎猜安全。
    """
    if state.current_anchor is None:
        return tuple(dict.fromkeys(state.observed_scope_cells))
    row, col = state.current_anchor
    return tuple(
        cell for cell in _rule_scope_cells_for_anchor(row, col, rule) if cell != state.current_anchor
    )


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
            # Special placement affixes apply to the pending current-paint
            # anchor only. They are consumed after that anchor is placed, so
            # blocked/void cells in its scope cannot count as future installs.
            placeable = sum(1 for r, c in scope if _is_placeable(state.tile(r, c), None))
            existing_matches = sum(1 for r, c in scope if _tile_matches_target(state.tile(r, c), rule.target_object))
            existing_value_gain = _scope_upgrade_value_gain(state, scope, rule)
            possible_matches = existing_matches + min(placeable, len(target_paints))
            progress_ratio = min(1.0, possible_matches / max(1, rule.match_required))
            center_bonus = _cell_priority((row, col)) * 4.0
            terrain_delta = _current_anchor_terrain_delta(state, row, col, rule.paint)
            special_loss_penalty = _special_anchor_loss_penalty(tile, rule.paint, state.active_affix)
            score = (
                progress_ratio * 600.0
                + placeable * 18.0
                + existing_matches * 60.0
                + existing_value_gain * 1.3
                + center_bonus
                + _special_anchor_tile_bonus(tile, state.active_affix)
                + terrain_delta * float(config_value("placement.terrain_delta_weight", 1.15))
                - special_loss_penalty
            )
            if terrain_delta < 0:
                score += terrain_delta * float(config_value("placement.bad_terrain_extra_weight", 0.65))
            if possible_matches >= rule.match_required:
                score += 500.0
            reason = (
                f"放置初始颜料，{_direction_label(rule.direction)}范围可凑"
                f"{min(possible_matches, rule.match_required)}/{rule.match_required}"
            )
            if tile.state != TileState.BLANK:
                reason += f"，{_special_anchor_reason(tile, rule.paint, state.active_affix)}"
            if abs(terrain_delta) >= 30:
                sign = "+" if terrain_delta > 0 else ""
                reason += f"，地形{sign}{terrain_delta:.0f}"
            recommendations.append(
                PlacementRecommendation(
                    row=row,
                    col=col,
                    paint=rule.paint,
                    score=score,
                    completes_match=possible_matches >= rule.match_required,
                    affected=scope,
                    reason=reason,
                    explanation=_anchor_explanation(
                        state,
                        rule,
                        row,
                        col,
                        scope,
                        possible_matches,
                        placeable,
                        existing_matches,
                        existing_value_gain,
                        terrain_delta,
                        tile,
                        special_loss_penalty,
                    ),
                )
            )

    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations[:limit]


def _scope_upgrade_value_gain(
    state: PuzzleState,
    scope: tuple[tuple[int, int], ...],
    rule: CurrentPaintRule,
) -> float:
    gain = 0.0
    for row, col in scope:
        tile = state.tile(row, col)
        if tile.paint is None or not _tile_matches_target(tile, rule.target_object):
            continue
        if not _tile_can_receive_current_buff(tile, state):
            continue
        gain += _paint_upgrade_gain(tile.paint, min(MAX_LEVEL, tile.paint.level + rule.upgrade_amount))
    return gain


def _special_anchor_tile_bonus(tile: Tile, affix: AffixRule | None) -> float:
    if affix is None:
        return 0.0
    if tile.state == TileState.BLOCKED and affix.effect == AffixEffect.ALLOW_PLACE_ON_BLOCKED:
        return 120.0
    if tile.state == TileState.VOID and affix.effect == AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF:
        return 55.0
    if tile.state == TileState.PAINT and affix.effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF:
        # This special case keeps the old paint and only uses the cell as an
        # anchor, so it is not the same risk profile as overwriting a backpack
        # paint onto an existing high-value piece.
        return 75.0
    if tile.state == TileState.FADED and affix.effect == AffixEffect.ALLOW_PLACE_ON_FADED:
        return 85.0
    return 0.0


def _special_anchor_loss_penalty(tile: Tile, paint: Paint, affix: AffixRule | None) -> float:
    if affix is None:
        return 0.0
    loses_self = (
        tile.state == TileState.VOID
        and affix.effect == AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF
    ) or (
        tile.state == TileState.PAINT
        and affix.effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF
    )
    return _paint_strategic_value(paint) if loses_self else 0.0


def _special_anchor_reason(tile: Tile, paint: Paint, affix: AffixRule | None) -> str:
    if tile.state == TileState.BLOCKED and affix is not None and affix.effect == AffixEffect.ALLOW_PLACE_ON_BLOCKED:
        return "特殊词条落在阻挡地块"
    if tile.state == TileState.VOID and affix is not None and affix.effect == AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF:
        return f"特殊词条落在虚空地块，失去{_paint_label(paint)}自身内容"
    if tile.state == TileState.PAINT and affix is not None and affix.effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF:
        return f"特殊词条借用已有颜料地块，失去{_paint_label(paint)}自身内容"
    if tile.state == TileState.FADED and affix is not None and affix.effect == AffixEffect.ALLOW_PLACE_ON_FADED:
        return "特殊词条落在失色地块"
    return f"使用特殊词条落在{tile.state.value}"


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

    score += outcome.value_gain * 2.4
    score += outcome.placed_value_gain * (0.9 if placed_paint.is_treasure else 0.35)
    if placed_paint.is_treasure:
        score += _paint_strategic_value(placed_paint) * 0.95
    if outcome.treasure_to_7:
        score += outcome.treasure_to_7 * TREASURE_LEVEL_VALUES[7] * 1.25
    if outcome.gameplay_to_7:
        score += outcome.gameplay_to_7 * GAMEPLAY_7_VALUE * 0.95
    if outcome.monster_to_7:
        score += outcome.monster_to_7 * MONSTER_7_VALUE * 0.95
    if outcome.current_next_level >= 6 and not placed_paint.is_treasure:
        score += _paint_strategic_value(replace(placed_paint, level=outcome.current_next_level)) * 0.18
    score += outcome.affected_paints * (22.0 if outcome.completes_match else 6.0)
    score += _future_space_value(state, row, col, affected)
    score -= outcome.overwrite_penalty
    score += outcome.terrain_delta * float(config_value("placement.terrain_delta_weight", 1.15))
    return score


def _candidate_lookahead_score(
    state: PuzzleState,
    rule: CurrentPaintRule,
    row: int,
    col: int,
    paint: Paint,
    outcome: PlacementOutcome,
) -> float:
    simulated = _simulate_install_without_next_rule(state, rule, row, col, paint)
    if simulated is None:
        return 0.0
    next_state, completed = simulated
    settled_terrain_delta = _terrain_value(next_state) - _terrain_value(state) - outcome.terrain_delta
    score = settled_terrain_delta * 0.35
    score += _high_value_position_delta(state, next_state) * 0.12
    if completed:
        score += _post_match_board_value(next_state) * 0.20
        score += _pending_anchor_opportunity_score(next_state, rule) * 0.18
    else:
        next_rule = next_state.current_rule
        if next_rule is not None:
            next_recs = recommend_placements(next_state, limit=3, include_lookahead=False)
            if next_recs:
                score += next_recs[0].score * 0.22
                if any(rec.completes_match for rec in next_recs[:3]):
                    score += 180.0
            score += _current_rule_completion_potential(next_state, next_rule) * 160.0
    if outcome.completes_match and _paint_strategic_value(paint) >= TREASURE_LEVEL_VALUES[5]:
        score += 120.0
    if outcome.terrain_delta < 0:
        score += outcome.terrain_delta * float(config_value("placement.bad_terrain_extra_weight", 0.65))
    return score


def _placement_explanation(
    state: PuzzleState,
    rule: CurrentPaintRule,
    row: int,
    col: int,
    affected: tuple[tuple[int, int], ...],
    outcome: PlacementOutcome,
    placed_paint: Paint,
    lookahead_score: float,
    prior_score: float,
) -> RecommendationExplanation:
    missing = max(0, rule.match_required - outcome.match_count)
    progress_ratio = outcome.match_count / max(1, rule.match_required)
    immediate = progress_ratio * 120.0
    immediate += 900.0 if outcome.completes_match else max(0, 4 - missing) * 45.0
    immediate += outcome.affected_paints * (22.0 if outcome.completes_match else 6.0)

    value = outcome.value_gain * 2.4
    value += outcome.placed_value_gain * (0.9 if placed_paint.is_treasure else 0.35)
    if placed_paint.is_treasure:
        value += _paint_strategic_value(placed_paint) * 0.95
    value += outcome.treasure_to_7 * TREASURE_LEVEL_VALUES[7] * 1.25
    value += outcome.gameplay_to_7 * GAMEPLAY_7_VALUE * 0.95
    value += outcome.monster_to_7 * MONSTER_7_VALUE * 0.95
    if outcome.current_next_level >= 6 and not placed_paint.is_treasure:
        value += _paint_strategic_value(replace(placed_paint, level=outcome.current_next_level)) * 0.18

    terrain = outcome.terrain_delta * float(config_value("placement.terrain_delta_weight", 1.15))
    resource = _future_space_value(state, row, col, affected)
    risk = -outcome.overwrite_penalty
    expected = lookahead_score + prior_score
    tags = _placement_explanation_tags(outcome, placed_paint, lookahead_score, prior_score)
    return RecommendationExplanation(
        immediate=round(immediate, 3),
        terrain=round(terrain, 3),
        risk=round(risk, 3),
        resource=round(resource, 3),
        expected=round(expected, 3),
        value=round(value, 3),
        tags=tags,
        short=_placement_explanation_short(tags, outcome),
    )


def _anchor_explanation(
    state: PuzzleState,
    rule: CurrentPaintRule,
    row: int,
    col: int,
    scope: tuple[tuple[int, int], ...],
    possible_matches: int,
    placeable: int,
    existing_matches: int,
    existing_value_gain: float,
    terrain_delta: float,
    tile: Tile,
    special_loss_penalty: float,
) -> RecommendationExplanation:
    progress_ratio = min(1.0, possible_matches / max(1, rule.match_required))
    completes = possible_matches >= rule.match_required
    immediate = progress_ratio * 600.0 + existing_matches * 60.0
    if completes:
        immediate += 500.0
    terrain = terrain_delta * float(config_value("placement.terrain_delta_weight", 1.15))
    value = existing_value_gain * 1.3
    resource = placeable * 18.0 + _cell_priority((row, col)) * 4.0 + _special_anchor_tile_bonus(tile, state.active_affix)
    risk = (0.0 if tile.state == TileState.BLANK else -35.0) - special_loss_penalty
    tags = _anchor_explanation_tags(completes, tile, terrain_delta, existing_value_gain)
    return RecommendationExplanation(
        immediate=round(immediate, 3),
        terrain=round(terrain, 3),
        risk=round(risk, 3),
        resource=round(resource, 3),
        expected=0.0,
        value=round(value, 3),
        tags=tags,
        short=_anchor_explanation_short(tags),
    )


def _placement_explanation_tags(
    outcome: PlacementOutcome,
    placed_paint: Paint,
    lookahead_score: float,
    prior_score: float,
) -> tuple[str, ...]:
    tags: list[str] = []
    if outcome.completes_match:
        tags.append("complete_match")
    if placed_paint.is_treasure:
        tags.append("treasure")
    if outcome.current_next_level >= MAX_LEVEL:
        tags.append("placed_to_7")
    if outcome.treasure_to_7:
        tags.append("treasure_to_7")
    if outcome.gameplay_to_7:
        tags.append("gameplay_to_7")
    if outcome.monster_to_7:
        tags.append("monster_to_7")
    if outcome.value_gain + outcome.placed_value_gain >= 250:
        tags.append("high_value_gain")
    if outcome.overwrite_penalty >= 120:
        tags.append("overwrite_risk")
    if outcome.terrain_delta >= 30:
        tags.append("terrain_gain")
    elif outcome.terrain_delta <= -30:
        tags.append("terrain_loss")
    if lookahead_score >= 90:
        tags.append("lookahead_gain")
    elif lookahead_score <= -90:
        tags.append("lookahead_risk")
    if prior_score >= 25:
        tags.append("learned_prior")
    return tuple(dict.fromkeys(tags))


def _anchor_explanation_tags(
    completes: bool,
    tile: Tile,
    terrain_delta: float,
    existing_value_gain: float,
) -> tuple[str, ...]:
    tags: list[str] = ["anchor"]
    if completes:
        tags.append("can_complete")
    if tile.state != TileState.BLANK:
        tags.append("special_tile")
    if tile.state == TileState.BLOCKED:
        tags.append("blocked_anchor")
    elif tile.state == TileState.VOID:
        tags.append("void_lose_self")
    elif tile.state == TileState.PAINT:
        tags.append("paint_lose_self")
    elif tile.state == TileState.FADED:
        tags.append("faded_anchor")
    if terrain_delta >= 30:
        tags.append("terrain_gain")
    elif terrain_delta <= -30:
        tags.append("terrain_loss")
    if existing_value_gain >= 250:
        tags.append("scope_value_gain")
    return tuple(dict.fromkeys(tags))


def _placement_explanation_short(tags: tuple[str, ...], outcome: PlacementOutcome) -> str:
    if "treasure_to_7" in tags:
        return "宝箱到7"
    if "complete_match" in tags and "treasure" in tags:
        return "宝箱匹配"
    if "complete_match" in tags:
        return "完成匹配"
    if "placed_to_7" in tags or "gameplay_to_7" in tags or "monster_to_7" in tags:
        return "冲7级"
    if "overwrite_risk" in tags:
        return "有覆盖风险"
    if "terrain_gain" in tags:
        return "地形更好"
    if outcome.match_count:
        return f"推进{outcome.match_count}格"
    return "保留后续"


def _anchor_explanation_short(tags: tuple[str, ...]) -> str:
    if "can_complete" in tags and "scope_value_gain" in tags:
        return "范围收益高"
    if "can_complete" in tags:
        return "可完成地形"
    if "terrain_gain" in tags:
        return "地形更好"
    if "void_lose_self" in tags:
        return "虚空失去自身"
    if "paint_lose_self" in tags:
        return "借位失去自身"
    if "blocked_anchor" in tags:
        return "放到阻挡"
    if "faded_anchor" in tags:
        return "放到失色"
    if "special_tile" in tags:
        return "特殊落点"
    return "初始锚点"


def _dedupe_placement_recommendations(
    recommendations: list[PlacementRecommendation],
) -> list[PlacementRecommendation]:
    best_by_key: dict[tuple[int, int, tuple | None], PlacementRecommendation] = {}
    for recommendation in recommendations:
        paint_key = _paint_recommendation_key(recommendation.paint)
        key = (recommendation.row, recommendation.col, paint_key)
        current = best_by_key.get(key)
        if current is None or recommendation.score > current.score:
            best_by_key[key] = recommendation
    return list(best_by_key.values())


def _paint_recommendation_key(paint: Paint | None) -> tuple | None:
    if paint is None:
        return None
    return (paint.category.value, paint.level, paint.is_treasure)


def _placement_policy_prior_score(stats: dict | None, row: int, col: int) -> float:
    if not stats:
        return 0.0
    samples = (stats.get("policy_prior_samples") or {}).get("placements", 0)
    if samples < int(config_value("learning.placement_min_samples", 30)):
        return 0.0
    cells = stats.get("placement_cells") or {}
    zones = stats.get("placement_zones") or {}
    cell_key = f"{row},{col}"
    zone = _cell_zone_for_policy(row, col)
    cell_ratio = cells.get(cell_key, 0) / max(1, samples)
    zone_total = sum(zones.values()) or samples
    zone_ratio = zones.get(zone, 0) / max(1, zone_total)
    score = cell_ratio * 420.0 + zone_ratio * 160.0
    return min(float(config_value("learning.placement_max_bonus", 180.0)), score)


def _terrain_policy_prior_score(stats: dict | None, terrain_delta: float) -> float:
    if not stats:
        return 0.0
    prior = stats.get("terrain_prior") if isinstance(stats.get("terrain_prior"), dict) else {}
    if prior.get("ready") is not True:
        return 0.0
    samples = prior.get("samples") if isinstance(prior.get("samples"), dict) else {}
    total_samples = int(samples.get("high", 0) or 0) + int(samples.get("compressed", 0) or 0)
    if total_samples < int(config_value("learning.terrain_min_samples", 20)):
        return 0.0
    deltas = prior.get("delta_high_minus_compressed") if isinstance(prior.get("delta_high_minus_compressed"), dict) else {}
    high_prefers_open = float(deltas.get("open_cells", 0.0) or 0.0)
    high_avoids_blocked = -float(deltas.get("blocked", 0.0) or 0.0)
    confidence = min(1.0, total_samples / max(1.0, float(config_value("learning.terrain_min_samples", 20)) * 3.0))
    if terrain_delta > 0:
        raw = min(abs(terrain_delta) * 0.22, float(config_value("learning.terrain_prior_max_bonus", 95.0)))
        preference = 1.0 + max(0.0, high_prefers_open) * 0.035 + max(0.0, high_avoids_blocked) * 0.025
        return min(float(config_value("learning.terrain_prior_max_bonus", 95.0)), raw * preference * confidence)
    if terrain_delta < 0:
        raw = min(abs(terrain_delta) * 0.20, float(config_value("learning.terrain_prior_max_penalty", 85.0)))
        preference = 1.0 + max(0.0, high_prefers_open) * 0.030
        return -min(float(config_value("learning.terrain_prior_max_penalty", 85.0)), raw * preference * confidence)
    return 0.0


def _terrain_bucket_policy_score(stats: dict | None, state: PuzzleState) -> float:
    if not stats:
        return 0.0
    prior = stats.get("terrain_prior") if isinstance(stats.get("terrain_prior"), dict) else {}
    if prior.get("ready") is not True:
        return 0.0
    buckets = prior.get("terrain_buckets") if isinstance(prior.get("terrain_buckets"), dict) else {}
    items = buckets.get("items") if isinstance(buckets.get("items"), list) else []
    if not items:
        return 0.0
    key = _terrain_bucket_key_for_state(state)
    for item in items:
        if not isinstance(item, dict) or item.get("key") != key:
            continue
        total = int(item.get("total", 0) or 0)
        if total < int(config_value("learning.terrain_bucket_min_samples", 8)):
            return 0.0
        success_rate = float(item.get("success_rate", 0.0) or 0.0)
        centered = success_rate - 0.5
        confidence = min(1.0, total / max(1.0, float(config_value("learning.terrain_bucket_min_samples", 8)) * 3.0))
        raw = centered * 120.0 * confidence
        if raw >= 0:
            return min(float(config_value("learning.terrain_prior_max_bonus", 95.0)), raw)
        return max(-float(config_value("learning.terrain_prior_max_penalty", 85.0)), raw)
    return 0.0


def _terrain_bucket_key_for_state(state: PuzzleState) -> str:
    blank = _count_tiles(state, TileState.BLANK)
    paint = _count_tiles(state, TileState.PAINT)
    blocked = _count_tiles(state, TileState.BLOCKED)
    void = _count_tiles(state, TileState.VOID)
    center_open = sum(
        1
        for row, col in ((2, 2), (2, 3), (3, 2), (3, 3))
        if state.tile(row, col).state in {TileState.BLANK, TileState.PAINT}
    )
    return "|".join(
        (
            f"open:{_bucket_range_for_policy(blank + paint, (16, 22))}",
            f"block:{_bucket_range_for_policy(blocked, (4, 8))}",
            f"void:{_bucket_range_for_policy(void, (8, 13))}",
            f"center:{_bucket_range_for_policy(center_open, (1, 3))}",
        )
    )


def _bucket_range_for_policy(value: int, cuts: tuple[int, int]) -> str:
    low, high = cuts
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def _brush_policy_prior_score(stats: dict | None, brush_type: BrushType) -> float:
    if not stats:
        return 0.0
    samples = (stats.get("policy_prior_samples") or {}).get("brushes", 0)
    if samples < int(config_value("learning.brush_min_samples", 20)):
        return 0.0
    brush_code = {
        BrushType.TRANSFORM: "1",
        BrushType.STACK: "2",
        BrushType.SPREAD: "3",
    }[brush_type]
    brush_codes = stats.get("trajectory_brush_codes") or {}
    brushes = stats.get("trajectory_brushes") or {}
    count = brush_codes.get(brush_code, brushes.get(brush_type.value, 0))
    ratio = count / max(1, samples)
    return min(float(config_value("learning.brush_max_bonus", 130.0)), ratio * 260.0)


def _affix_policy_prior_score(stats: dict | None, affix: AffixRule) -> float:
    if not stats:
        return 0.0
    samples = (stats.get("policy_prior_samples") or {}).get("affixes", 0)
    if samples < int(config_value("learning.affix_min_samples", 20)):
        return 0.0
    affix_ids = stats.get("trajectory_affix_ids") or {}
    affix_codes = stats.get("trajectory_affix_codes") or {}
    affixes = stats.get("trajectory_affixes") or {}
    code = _affix_policy_code(affix)
    count = affix_ids.get(affix.canonical_affix_id)
    if count is None:
        count = affix_codes.get(code, affixes.get(affix.name, 0)) if code else affixes.get(affix.name, 0)
    ratio = count / max(1, samples)
    return min(float(config_value("learning.affix_max_bonus", 150.0)), ratio * 360.0)


def _affix_policy_code(affix: AffixRule) -> str | None:
    return affix.legacy_codes[0] if affix.legacy_codes else None


def _cell_zone_for_policy(row: int, col: int) -> str:
    if row in {0, BOARD_SIZE - 1} and col in {0, BOARD_SIZE - 1}:
        return "corner"
    if row in {0, BOARD_SIZE - 1} or col in {0, BOARD_SIZE - 1}:
        return "edge"
    if row in {2, 3} and col in {2, 3}:
        return "center"
    return "inner"


def _board_position_value(state: PuzzleState) -> float:
    paint_value = 0.0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            tile = state.tile(row, col)
            if tile.paint is None:
                continue
            paint_value += _paint_strategic_value(tile.paint)
            paint_value += _cell_priority((row, col)) * (18.0 if tile.paint.is_treasure else 4.0)
    return paint_value + _terrain_value(state)


def _high_value_position_delta(previous: PuzzleState, current: PuzzleState) -> float:
    previous_value = _high_value_position_value(previous)
    current_value = _high_value_position_value(current)
    return current_value - previous_value


def _high_value_position_value(state: PuzzleState) -> float:
    value = 0.0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            tile = state.tile(row, col)
            if tile.paint is None:
                continue
            strategic = _paint_strategic_value(tile.paint)
            if strategic < GAMEPLAY_7_VALUE:
                continue
            value += _cell_priority((row, col)) * (35.0 if tile.paint.is_treasure else 12.0)
    return value


def _terrain_value(state: PuzzleState) -> float:
    blanks = _count_tiles(state, TileState.BLANK)
    blocked = _count_tiles(state, TileState.BLOCKED)
    voids = _count_tiles(state, TileState.VOID)
    faded = _count_tiles(state, TileState.FADED)
    open_components = _blank_components(state)
    largest_open = max(open_components, default=0)
    fragmentation_penalty = max(0, len(open_components) - 1) * 18.0
    return (
        blanks * 18.0
        + largest_open * 9.0
        - blocked * 22.0
        - voids * 10.0
        - faded * 35.0
        - fragmentation_penalty
    )


def _blank_components(state: PuzzleState) -> list[int]:
    seen: set[tuple[int, int]] = set()
    components: list[int] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if (row, col) in seen or state.tile(row, col).state != TileState.BLANK:
                continue
            stack = [(row, col)]
            seen.add((row, col))
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                        continue
                    if (nr, nc) in seen or state.tile(nr, nc).state != TileState.BLANK:
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
            components.append(size)
    return components


def _post_match_board_value(state: PuzzleState) -> float:
    blank_count = _count_tiles(state, TileState.BLANK)
    paint_count = _count_tiles(state, TileState.PAINT)
    high_value_count = sum(1 for tile in state.tiles if tile.paint is not None and _tile_strategic_value(tile) >= GAMEPLAY_7_VALUE)
    return blank_count * 28.0 + paint_count * 10.0 + high_value_count * 160.0 + _terrain_value(state) * 0.35


def _pending_anchor_opportunity_score(state: PuzzleState, previous_rule: CurrentPaintRule) -> float:
    blanks = [(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if state.tile(row, col).state == TileState.BLANK]
    if not blanks:
        return -500.0
    directions = (
        previous_rule.direction,
        PaintDirection.HORIZONTAL,
        PaintDirection.VERTICAL,
        PaintDirection.DIAGONAL,
        PaintDirection.ADJACENT,
    )
    target_objects = (previous_rule.target_object, PaintObject.GAMEPLAY, PaintObject.MONSTER, PaintObject.ALL)
    best = 0.0
    for row, col in blanks:
        for direction in dict.fromkeys(directions):
            scope = tuple(cell for cell in _scope_cells(row, col, direction) if cell != (row, col))
            placeable = sum(1 for r, c in scope if _is_placeable(state.tile(r, c), state.active_affix))
            for target in dict.fromkeys(target_objects):
                existing = sum(1 for r, c in scope if _tile_matches_target(state.tile(r, c), target))
                target_supply = sum(1 for paint in state.backpack if _paint_matches_target(paint, target))
                possible = existing + min(placeable, target_supply)
                fit = min(1.0, possible / max(1, previous_rule.match_required))
                value_gain = _scope_upgrade_value_gain(state, scope, replace(previous_rule, direction=direction, target_object=target))
                best = max(best, fit * 260.0 + placeable * 10.0 + existing * 24.0 + value_gain * 0.18)
    return best


def _current_rule_completion_potential(state: PuzzleState, rule: CurrentPaintRule) -> float:
    current = _current_match_count(state, rule)
    needed = max(0, rule.match_required - current)
    if needed == 0:
        return 1.0
    scope = _placement_scope_cells(state, rule)
    placeable = sum(1 for row, col in scope if _is_placeable(state.tile(row, col), state.active_affix))
    supply = sum(1 for paint in state.backpack if _paint_matches_target(paint, rule.target_object))
    possible = min(needed, placeable, supply)
    return min(1.0, (current + possible) / max(1, rule.match_required))


def _overwritten_paint_for_install(tile: Tile, affix: AffixRule | None) -> Paint | None:
    if tile.state != TileState.PAINT or tile.paint is None:
        return None
    if affix is not None and affix.effect == AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF:
        return tile.paint
    return None


def _overwrite_penalty(overwritten: Paint | None, placed: Paint) -> float:
    if overwritten is None:
        return 0.0

    overwritten_value = _paint_strategic_value(overwritten)
    placed_value = _paint_strategic_value(placed)
    value_loss = max(0.0, overwritten_value - placed_value)
    penalty = overwritten_value * float(config_value("placement.overwrite_value_weight", 0.72)) + value_loss * 1.45

    if overwritten.is_treasure:
        penalty += float(config_value("placement.overwrite_treasure_base_penalty", 900.0))
        if not placed.is_treasure:
            penalty += float(config_value("placement.overwrite_treasure_with_non_treasure_penalty", 950.0))
        elif placed.level < overwritten.level:
            penalty += (overwritten.level - placed.level) * float(
                config_value("placement.low_treasure_over_high_treasure_penalty", 1400.0)
            )
        if overwritten.level >= 6:
            penalty += float(config_value("placement.overwrite_high_treasure_penalty", 2200.0))
    elif overwritten.level >= 7:
        penalty += float(config_value("placement.overwrite_level7_penalty", 1000.0))
    elif overwritten.level >= 6:
        penalty += float(config_value("placement.overwrite_level6_penalty", 420.0))

    if placed.is_treasure and not overwritten.is_treasure and placed.level >= 5:
        penalty *= float(config_value("placement.treasure_overwrite_discount", 0.65))
    return penalty


def _placement_terrain_delta(state: PuzzleState, row: int, col: int, placed_paint: Paint) -> float:
    tiles = list(state.tiles)
    tiles[row * BOARD_SIZE + col] = replace(
        tiles[row * BOARD_SIZE + col],
        state=TileState.PAINT,
        paint=placed_paint,
        buff_applied_id=None,
        affix_applied_id=None,
    )
    next_state = replace(state, tiles=tuple(tiles))
    return _terrain_value(next_state) - _terrain_value(state)


def _current_anchor_terrain_delta(state: PuzzleState, row: int, col: int, paint: Paint) -> float:
    tiles = list(state.tiles)
    tiles[row * BOARD_SIZE + col] = _tile_after_current_anchor_placement(state.tile(row, col), paint, state.active_affix)
    next_state = replace(state, tiles=tuple(tiles))
    return _terrain_value(next_state) - _terrain_value(state)


def _placement_outcome(
    state: PuzzleState,
    rule: CurrentPaintRule,
    row: int,
    col: int,
    affected: tuple[tuple[int, int], ...],
    placed_paint: Paint,
) -> PlacementOutcome:
    overwritten_paint = _overwritten_paint_for_install(state.tile(row, col), state.active_affix)
    overwrite_penalty = _overwrite_penalty(overwritten_paint, placed_paint)
    terrain_delta = _placement_terrain_delta(state, row, col, placed_paint)
    match_count = sum(
        1
        for r, c in affected
        if ((r, c) == (row, col) and _paint_matches_target(placed_paint, rule.target_object))
        or _tile_matches_target(state.tile(r, c), rule.target_object)
    )
    completes = match_count >= rule.match_required
    # 落笔时同步结算的「额外提升」笔触：全体命中时一定吃到，随机 N 个按期望折算。
    affix = state.active_affix
    affix_amount = 0
    affix_hit_rate = 0.0
    if _is_deferred_scope_upgrade(affix):
        affix_amount = max(1, affix.amount)
        if affix.count is None:
            affix_hit_rate = 1.0
        else:
            affix_hit_rate = min(1.0, affix.count / max(1, len(affected)))
    current_extra = affix_amount if affix_hit_rate >= 1.0 else 0
    # 落点这一格的地块加成先算进来：加成已经堆高时，落一枚低等级颜料也可能直接到 7。
    placed_from_level = min(MAX_LEVEL, placed_paint.level + tile_bonus_level(state.tile(row, col)))
    current_next_level = min(
        MAX_LEVEL,
        placed_from_level
        + (rule.upgrade_amount if _paint_matches_target(placed_paint, rule.target_object) else 0)
        + current_extra,
    )
    placed_value_gain = (
        _paint_level_gain(placed_paint, placed_from_level, current_next_level)
        if _paint_matches_target(placed_paint, rule.target_object)
        else 0.0
    )
    affected_paints = 0
    value_gain = 0.0
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
            from_level = placed_from_level
            can_receive_buff = True
        else:
            tile = state.tile(r, c)
            if not _tile_matches_target(tile, rule.target_object):
                continue
            paint = tile.paint
            from_level = tile_final_level(tile)
            can_receive_buff = _tile_can_receive_current_buff(tile, state)
        if paint is None:
            continue
        if not can_receive_buff:
            continue
        affected_paints += 1
        base_level = min(MAX_LEVEL, from_level + rule.upgrade_amount)
        next_level = min(MAX_LEVEL, base_level + affix_amount)
        if affix_hit_rate >= 1.0:
            value_gain += _paint_level_gain(paint, from_level, next_level)
        elif affix_hit_rate > 0.0:
            value_gain += (
                _paint_level_gain(paint, from_level, base_level) * (1.0 - affix_hit_rate)
                + _paint_level_gain(paint, from_level, next_level) * affix_hit_rate
            )
            next_level = base_level
        else:
            value_gain += _paint_level_gain(paint, from_level, base_level)
        if paint.is_treasure:
            treasure_upgrades += 1
            if from_level < MAX_LEVEL <= next_level:
                treasure_to_7 += 1
        if from_level < MAX_LEVEL <= next_level:
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
        value_gain=value_gain,
        placed_value_gain=placed_value_gain,
        overwritten_paint=overwritten_paint,
        overwrite_penalty=overwrite_penalty,
        terrain_delta=terrain_delta,
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


def _reason(
    rule: CurrentPaintRule,
    outcome: PlacementOutcome,
    placed_paint: Paint,
    lookahead_score: float = 0.0,
    prior_score: float = 0.0,
) -> str:
    parts = [
        f"用{_paint_label(placed_paint)}",
        f"{_direction_label(rule.direction)} {_target_label(rule.target_object)} {outcome.match_count}/{rule.match_required}",
    ]
    if outcome.completes_match:
        parts.append(f"完成匹配，触发提升{rule.upgrade_amount}级")
    total_gain = outcome.value_gain
    if total_gain >= 250:
        parts.append(f"战略价值+{total_gain:.0f}")
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
    if outcome.overwritten_paint is not None and outcome.overwrite_penalty >= 120:
        parts.append(f"覆盖{_paint_label(outcome.overwritten_paint)} -{outcome.overwrite_penalty:.0f}")
    if abs(outcome.terrain_delta) >= 30:
        sign = "+" if outcome.terrain_delta > 0 else ""
        parts.append(f"落点地形{sign}{outcome.terrain_delta:.0f}")
    if abs(lookahead_score) >= 90:
        sign = "+" if lookahead_score > 0 else ""
        parts.append(f"后续地形{sign}{lookahead_score:.0f}")
    if prior_score >= 25:
        parts.append(f"经验先验+{prior_score:.0f}")
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
        # 官方 4005 权重（7 级仅 1%）抽样，而不是等概率，与估分口径一致。
        levels = tuple(range(1, MAX_LEVEL + 1))
        level = _weighted_option(rng, levels, _official_option_probabilities(_RANDOMIZE_LEVEL_OFFICIAL_ID))
        if level is None:
            level = rng.randint(1, MAX_LEVEL)
        return replace(rule, paint=replace(rule.paint, level=int(level)))
    if affix.effect == AffixEffect.CHANGE_TARGET_OBJECT and affix.target_object is None:
        # 实机规则（2026-11-12 用户确认）：**不是随机**，而是由当前初始颜料
        # 的类型直接决定对换——玩法↔怪物；本来是「作用所有颜料」就不需要改。
        # 早期实现按官方 1006 权重随机抽，会抽出"换回自己"这种无效结果。
        return replace(
            rule, target_object=_alternate_target_object(rule.target_object, rule.paint.category)
        )
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_DIRECTION:
        # 官方未给该效果的权重表，维持等概率（与规则文档 §4.2 一致）。
        return replace(rule, direction=rng.choice(list(_SCOPE_DIRECTION_ORDER)))
    if affix.effect == AffixEffect.RANDOMIZE_CURRENT_CONTENT:
        next_category = _randomized_category(rule.paint.category, affix.target_object, rng)
        return replace(rule, paint=_random_paint_with_category_level(rng, next_category, rule.paint.level))
    if affix.effect == AffixEffect.ADD_RANDOM_SCOPE:
        # 官方 2019 权重：四个方向各约 24.4%，全图仅 2.44%；已存在的方向排除后重新归一化。
        candidates = _SCOPE_DIRECTION_ORDER
        existing = {rule.direction, *rule.extra_directions}
        pool = tuple(direction for direction in candidates if direction not in existing) or candidates
        extra = _weighted_option(rng, pool, _official_scope_probabilities())
        if extra is None:
            extra = pool[0]
        return replace(
            rule,
            extra_directions=tuple(dict.fromkeys((*rule.extra_directions, extra))),
            match_required=max(1, rule.match_required + max(0, affix.delta)),
        )
    return apply_transform_affix(rule, affix)


def _alternate_target_object(target: PaintObject, paint_category: PaintCategory) -> PaintObject:
    """「改成另一类型」的目标类型。

    实机规则（2026-11-12 用户确认）：由**当前初始颜料的类型**决定对换，
    玩法↔怪物；本来就是「作用所有颜料」时不需要改变。
    """
    if target == PaintObject.ALL:
        return PaintObject.ALL
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


def _resolve_anchor_affix(
    state: PuzzleState,
    affix: AffixRule | None,
    rng: random.Random | None = None,
) -> tuple[PuzzleState, str]:
    """落笔时结算一条词缀（**单步**，不含调色）。

    主流程已改走 :func:`settle_place_effects`（覆盖先、调色后），本函数只保留
    给单步测试用。直接在落笔路径调用会让覆盖类笔触排到调色后面，等级偏高。
    """
    if affix is None:
        return state, ""
    rng = rng or random.Random()
    if affix.effect in SCOPE_BUFF_EFFECTS:
        # 通用规则：落笔结算，但 **active_affix 不清空**——这几条整局有效，
        # 之后每次往范围内安装颜料都会重新结算。
        next_state, count = _settle_scope_buff(state, rng)
        return next_state, f"词缀影响范围内{count}个目标"
    if affix.effect in {
        AffixEffect.ALLOW_PLACE_ON_BLOCKED,
        AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF,
        AffixEffect.ALLOW_PLACE_ON_FADED,
    }:
        return replace(state, active_affix=None), "特殊放置词条已消耗"
    return state, ""


def _apply_persistent_affix_after_install(state: PuzzleState) -> tuple[PuzzleState, int]:
    """安装颜料之后结算持续型词缀。

    ``SCOPE_BUFF_EFFECTS`` 已在 :func:`_apply_deferred_scope_affix` 里结算过，
    这里只处理「全图持续提升」那几条，避免同一次安装被结算两遍。
    """
    affix = state.active_affix
    if affix is None or not affix.persistent_until_match:
        return state, 0
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_CATEGORY:
        return _upgrade_global_category_paints(state, affix)
    if affix.effect == AffixEffect.UPGRADE_GLOBAL_SAME_PAINT:
        return _upgrade_global_same_paints(state, affix)
    return state, 0


@dataclass(frozen=True)
class LevelSource:
    """**一个**给等级的来源：作用范围 + 目标类别 + 级数。

    用户 2026-11-15 定的口径：等级必须按来源分开算，每个来源都自带范围与目标类别，
    谁也不许借谁的。

    * 调色效果（:attr:`key` ``tint``）—— 范围取初始颜料规则的作用范围，
      目标取 ``rule.target_object``（玩法 / 怪物 / 所有）；
    * 笔触效果（``brush``）—— 范围取这条笔触自己的方向，目标取
      ``affix.target_object``；官方文案写「范围内所有颜料」时它为空，表示不筛类别；
    * 自身提升（``self``）—— **只**给这枚初始颜料自己，范围就是锚点那一格。

    为什么要单独建这个结构：以前调色和笔触各写一套「谁该涨」，两边对范围的收窄方式
    不同（调色吃 ``observed_scope_cells``，笔触不吃），一旦某格只被一边算中就表现成
    「模型 4 画面 5」的红框，而人看不出是哪条来源漏了。
    """

    key: str
    label: str
    #: 这个来源覆盖的格坐标（不含锚点自己）
    scope: tuple[tuple[int, int], ...]
    target: PaintObject | None
    amount: int
    #: 官方「随机 N 个」的 N；``None`` = 范围内全中
    random_count: int | None = None
    #: 真 = 记在地块加成上（「额外提升」），假 = 加到颜料等级上
    as_bonus: bool = False
    #: 只对「和自身内容相同」的颜料生效（``UPGRADE_SCOPE_SAME_PAINT``）
    same_paint_only: bool = False

    def covers(self, row: int, col: int) -> bool:
        return (row, col) in self.scope

    @property
    def target_text(self) -> str:
        if self.same_paint_only:
            return "同种颜料"
        if self.target is None or self.target == PaintObject.ALL:
            return "所有颜料"
        return {
            PaintObject.MONSTER: "怪物颜料",
            PaintObject.GAMEPLAY: "玩法颜料",
            PaintObject.BLOCKED_TILE: "阻挡地块",
            PaintObject.BLANK_TILE: "空白地块",
            PaintObject.VOID_TILE: "失色地块",
        }.get(self.target, str(self.target))

    @property
    def count_text(self) -> str:
        if self.random_count is None:
            return f"范围内全部 +{self.amount}"
        return f"范围内随机 {self.random_count} 个 +{self.amount}"

    def describe(self) -> str:
        return (
            f"{self.label}：目标={self.target_text} 范围={len(self.scope)}格 "
            f"{self.count_text}"
        )


def _paint_matches_category(paint: Paint, target: PaintObject | None) -> bool:
    """颜料类别是否过得了某个目标。``None`` / ``ALL`` = 不筛。"""
    if target is None or target is PaintObject.ALL:
        return True
    if target is PaintObject.MONSTER:
        return paint.category is PaintCategory.MONSTER
    if target is PaintObject.GAMEPLAY:
        return paint.category is PaintCategory.GAMEPLAY
    return True


def _brush_target_accepts_paint(
    paint: Paint | None,
    affix: AffixRule,
    rule: CurrentPaintRule,
) -> bool:
    """这枚颜料在**类别上**吃不吃这条笔触。

    目标类别只认 **笔触自己的** ``target_object``，绝不借调色规则的：
    两条词缀各有各的目标（调色写「玩法/怪物/所有」，笔触写「范围内XX颜料」）。
    拿调色的目标去过滤笔触，就会出现「明明该 +1 却算出没涨」的假差异。

    ``target_object`` 为空 = 官方文案没限定类别（「范围内所有颜料」「随机 1 个颜料」）
    → 不筛类别。``UPGRADE_SCOPE_SAME_PAINT`` 另外要求与初始颜料同种。
    """
    if paint is None:
        return False
    if affix.effect is AffixEffect.UPGRADE_SCOPE_SAME_PAINT and not _is_same_paint_kind(
        paint, rule.paint
    ):
        return False
    return _paint_matches_category(paint, affix.target_object)


def scope_upgrade_targets(
    state: PuzzleState,
    affix: AffixRule | None = None,
    *,
    placed: tuple[int, int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """「作用范围提升」类 buff 这一次会落到哪些格子。

    返回 ``(必中格索引, 可能命中格索引, 每格级数)``。两组互斥：

    * 必中 —— 范围内的目标数 **不超过** buff 的随机个数（``count`` 为空表示全范围）；
    * 可能 —— 官方「随机 N 个」在候选多于 N 时只能命中一部分，谁中谁不中由服务端摇。

    实机口径（用户 2026-11-15 两次修正后的最终版）：

    * 目标 = 作用范围里**已有颜料**的格子，且类别必须过得了**笔触自己的**目标；
    * **初始颜料不吃这条**。它只吃「使自身等级提升 N 级」，而「作用范围」按定义
      就是锚点以外的格（:func:`_affix_scope_cells` 已排除锚点），所以拿锚点兜底
      等于凭空给初始颜料加级——旧实现就是这么错的，已删除；
    * 范围里一个颜料都没有 → **这一笔什么也不涨**。用户说的「范围全是空白、放下第一枚
      背包颜料就提升它」不是兜底，而是：那一枚落在范围**内**，落笔后它自然成了范围内
      唯一的候选，随机 1 个必然抽中它；
    * ``placed`` 只为「结算早于落笔」的场景保留：只有它**落在范围里**才并入候选，
      在范围外（含锚点格）一律不计。
    """
    chosen_affix = affix if affix is not None else state.active_affix
    if chosen_affix is None or chosen_affix.effect not in {
        AffixEffect.UPGRADE_IN_SCOPE,
        AffixEffect.UPGRADE_SCOPE_SAME_PAINT,
    }:
        return (), (), 0
    rule = state.current_rule
    if rule is None:
        return (), (), 0
    amount = max(1, int(chosen_affix.amount))
    # 范围是「锚点以外」的几何定义：里面天然不含锚点，初始颜料因此吃不到这条。
    scope = _affix_scope_cells(state, chosen_affix)

    indices: list[int] = []
    for row, col in scope:
        index = row * BOARD_SIZE + col
        if not 0 <= index < len(state.tiles):
            continue
        tile = state.tiles[index]
        if tile.state != TileState.PAINT:
            continue
        if _brush_target_accepts_paint(tile.paint, chosen_affix, rule):
            indices.append(index)

    if placed is not None and tuple(placed) in scope:
        index = placed[0] * BOARD_SIZE + placed[1]
        if 0 <= index < len(state.tiles) and index not in indices:
            tile = state.tiles[index]
            if tile.state != TileState.PAINT:
                # 结算跑在落笔之前：这一格还是空的，但这一笔就要放进去的那枚符合条件
                # （调用方只会在范围里传它，类别由调用方保证）。
                indices.append(index)

    count = chosen_affix.count
    if count is not None and len(indices) > count:
        return (), tuple(indices), amount
    return tuple(indices), (), amount


def level_sources(
    state: PuzzleState,
    *,
    affix: AffixRule | None = None,
    self_upgrade: int = 0,
) -> tuple[LevelSource, ...]:
    """当前这一笔涉及的**全部等级来源**，每个都带自己的范围与目标类别。

    顺序固定：调色 → 笔触 → 自身。``self`` 只有 ``self_upgrade > 0`` 时才出现，
    范围就是锚点那一格（初始颜料只吃这一条，见 :func:`scope_upgrade_targets`）。
    """
    rule = state.current_rule
    if rule is None:
        return ()
    sources: list[LevelSource] = []
    if rule.upgrade_amount > 0:
        sources.append(
            LevelSource(
                key="tint",
                label="调色效果",
                scope=_placement_scope_cells(state, rule),
                target=rule.target_object,
                amount=max(0, int(rule.upgrade_amount)),
                random_count=None,
                # 初始颜料的调色加成落在**地块**上（用户 2026-11-22 实机口径：
                # c4r5 被转换后仍吃到 +2/+1 到 6 级，且点击地块提示 +3 = 2+1）。
                as_bonus=True,
            )
        )
    chosen_affix = affix if affix is not None else state.active_affix
    if chosen_affix is not None and chosen_affix.effect in {
        AffixEffect.UPGRADE_IN_SCOPE,
        AffixEffect.UPGRADE_SCOPE_SAME_PAINT,
    }:
        sources.append(
            LevelSource(
                key="brush",
                label="笔触效果",
                scope=_affix_scope_cells(state, chosen_affix),
                target=chosen_affix.target_object,
                amount=max(1, int(chosen_affix.amount)),
                random_count=chosen_affix.count,
                as_bonus=bool(chosen_affix.grants_bonus),
                same_paint_only=(
                    chosen_affix.effect is AffixEffect.UPGRADE_SCOPE_SAME_PAINT
                ),
            )
        )
    if self_upgrade > 0 and state.current_anchor is not None:
        sources.append(
            LevelSource(
                key="self",
                label="自身提升",
                scope=(state.current_anchor,),
                target=None,
                amount=max(0, int(self_upgrade)),
                as_bonus=False,
            )
        )
    return tuple(sources)


def source_accepts_paint(
    source: LevelSource,
    paint: Paint | None,
    rule: CurrentPaintRule | None,
) -> bool:
    """这枚颜料吃不吃这个来源（类别 + 同种过滤）。"""
    if paint is None:
        return False
    if source.same_paint_only and rule is not None and not _is_same_paint_kind(
        paint, rule.paint
    ):
        return False
    return _paint_matches_category(paint, source.target)


def tile_level_breakdown(
    state: PuzzleState,
    row: int,
    col: int,
    sources: tuple[LevelSource, ...],
    *,
    base_level: int | None = None,
) -> tuple[int, tuple[tuple[str, int], ...]]:
    """一格等级 = 基础 + 各来源。返回 ``(基础等级, ((来源名, 级数), ...))``。

    ``base_level`` 是这一笔**之前**该格的等级：

    * 新落下的格子传 ``None`` → 用面板等级（``rule.paint.level``）；
    * 早就在盘上的格子传它上一帧的画面等级，否则会把已经生效的加成再算一遍。
    """
    tile = state.tile(row, col)
    paint = tile.paint
    if paint is None:
        return 0, ()
    rule = state.current_rule
    if base_level is not None:
        base = max(0, int(base_level))
    else:
        base = int(getattr(rule.paint, "level", 0)) if rule is not None else int(paint.level)
    parts: list[tuple[str, int]] = []
    for source in sources:
        if not source.covers(row, col):
            continue
        if not source_accepts_paint(source, paint, rule):
            continue
        parts.append((source.label, source.amount))
    return base, tuple(parts)


def describe_level_sources(sources: tuple[LevelSource, ...]) -> tuple[str, ...]:
    """给界面用的逐来源文字（范围 + 目标类别 + 级数）。"""
    return tuple(source.describe() for source in sources)


def _upgrade_scope_paints(
    state: PuzzleState,
    affix: AffixRule,
    rng: random.Random,
    *,
    placed: tuple[int, int] | None = None,
) -> tuple[PuzzleState, int]:
    sure, maybe, amount = scope_upgrade_targets(state, affix, placed=placed)
    chosen = list(sure)
    if maybe:
        limit = affix.count if affix.count is not None else len(maybe)
        chosen.extend(rng.sample(list(maybe), min(max(0, limit), len(maybe))))
    tiles = list(state.tiles)
    for index in chosen:
        tile = tiles[index]
        if tile.paint is None:
            continue
        tiles[index] = _tile_with_upgrade(tile, amount, as_bonus=affix.grants_bonus)
    return replace(state, tiles=tuple(tiles)), len(chosen)


def _upgrade_global_category_paints(state: PuzzleState, affix: AffixRule) -> tuple[PuzzleState, int]:
    amount = max(1, affix.amount)
    tiles = list(state.tiles)
    count = 0
    for index, tile in enumerate(tiles):
        if tile.paint is None or not _paint_matches_target(tile.paint, affix.target_object or PaintObject.ALL):
            continue
        if affix.persistent_until_match and not _tile_can_receive_persistent_affix(tile, state):
            continue
        tiles[index] = replace(
            _tile_with_upgrade(tile, amount, as_bonus=affix.grants_bonus),
            affix_applied_id=state.current_buff_id if affix.persistent_until_match else tile.affix_applied_id,
        )
        count += 1
    return replace(state, tiles=tuple(tiles)), count


def _upgrade_global_same_paints(state: PuzzleState, affix: AffixRule) -> tuple[PuzzleState, int]:
    rule = state.current_rule
    if rule is None:
        return state, 0
    amount = max(1, affix.amount)
    if affix.grants_bonus:
        return _attach_same_paint_aura(state, amount)
    tiles = list(state.tiles)
    count = 0
    for index, tile in enumerate(tiles):
        if tile.paint is None or not _is_same_paint_kind(tile.paint, rule.paint):
            continue
        if affix.persistent_until_match and not _tile_can_receive_persistent_affix(tile, state):
            continue
        tiles[index] = replace(
            _tile_with_upgrade(tile, amount, as_bonus=False),
            affix_applied_id=state.current_buff_id if affix.persistent_until_match else tile.affix_applied_id,
        )
        count += 1
    return replace(state, tiles=tuple(tiles)), count


def _attach_same_paint_aura(state: PuzzleState, amount: int) -> tuple[PuzzleState, int]:
    """把「与自身相同的颜料额外提升 x 级」挂到**初始颜料所在的地块**上。

    实机验证过的行为：这条词缀不跟着颜料走，而是跟着地块走。

    - 初始颜料被炸掉（地块变失色）→ 光环失效，别处同种颜料掉级；
    - 失色地块恢复后放一枚新颜料（比如从幽灵换成存活）→ 光环改挂到存活上，
      全图存活立刻吃到这 x 级。

    所以这里只写 ``aura_amount``，具体加给谁交给 :func:`_recompute_auras` 按当前
    棋盘重新算——锚点上的颜料是什么，光环就照什么算。
    """
    anchor = state.current_anchor
    if anchor is None:
        return state, 0
    row, col = anchor
    index = row * BOARD_SIZE + col
    if not 0 <= index < len(state.tiles):
        return state, 0
    tile = state.tiles[index]
    if tile.paint is None or tile.state != TileState.PAINT:
        return state, 0
    tiles = list(state.tiles)
    tiles[index] = replace(tile, aura_amount=max(tile.aura_amount, amount))
    next_state = _recompute_auras(replace(state, tiles=tuple(tiles)))
    count = sum(1 for item in next_state.tiles if tile_aura_level(item) > 0)
    return next_state, count


def _upgrade_scope_same_paints(state: PuzzleState, affix: AffixRule) -> tuple[PuzzleState, int]:
    if state.current_rule is None:
        return state, 0
    sure, maybe, amount = scope_upgrade_targets(state, affix)
    tiles = list(state.tiles)
    count = 0
    for index in (*sure, *maybe):
        tile = tiles[index]
        if tile.paint is None:
            continue
        tiles[index] = _tile_with_upgrade(tile, amount, as_bonus=affix.grants_bonus)
        count += 1
    return replace(state, tiles=tuple(tiles)), count


def _copy_self_to_scope(state: PuzzleState) -> tuple[PuzzleState, int]:
    """笔触效果：把作用范围内**已有颜料**的等级与内容替换为自身。

    实机口径（2026-11-12）：

    * 只替换**已经是颜料**的格子；空白格不会凭空长出颜料——
      这条 buff 不会无中生有，必须是"往范围内安装了背包颜料"才会变；
    * 替换后等级 = 自身等级 **+ 该地块自己的升级 buff**。
      这里只把 ``paint`` 换成 ``rule.paint``（等级＝自身等级），
      地块上的 ``bonus_level`` / ``aura_amount`` 原样保留，
      由 :func:`tile_final_level` 相加得出 ``自身 + x``，
      所以"5 级宝箱 + 提升 1 级"落上去就是 6 级，再有别的升级 buff 就是 6+x；
    * 整局有效：调用方（:func:`settle_place_effects` /
      :func:`_apply_persistent_affix_after_install`）不清 ``active_affix``；
    * **先于调色结算**：覆盖后 ``buff_applied_id`` 被清空，这一格会重新吃一次
      调色效果——这正是实机「3 个玩法变 4 级、1 个怪物还是 3 级」的来源。
    """
    rule = state.current_rule
    if rule is None:
        return state, 0
    tiles = list(state.tiles)
    count = 0
    for row, col in _affix_scope_cells(state, state.active_affix):
        index = row * BOARD_SIZE + col
        tile = tiles[index]
        if tile.state != TileState.PAINT or tile.paint is None:
            continue
        # 同一块地反复刷时不重复计数：内容本来就已经是自身了。
        if tile.paint.category == rule.paint.category and tile.paint.subtype == rule.paint.subtype:
            continue
        tiles[index] = replace(tile, paint=rule.paint, buff_applied_id=None, affix_applied_id=None)
        count += 1
    # 内容被换掉，同种光环的归属也跟着变
    return _recompute_auras(replace(state, tiles=tuple(tiles))), count


def _copy_self_content_global(state: PuzzleState) -> tuple[PuzzleState, int]:
    """官方 2013：将全图中所有颜料的内容替换为自身（等级保持不变）。"""
    rule = state.current_rule
    if rule is None:
        return state, 0
    tiles = list(state.tiles)
    count = 0
    for index, tile in enumerate(tiles):
        if tile.state != TileState.PAINT or tile.paint is None:
            continue
        tiles[index] = replace(
            tile,
            state=TileState.PAINT,
            paint=replace(rule.paint, level=tile.paint.level),
            buff_applied_id=tile.buff_applied_id,
            affix_applied_id=tile.affix_applied_id,
        )
        count += 1
    # 全图内容都换成「自身」，同种光环重新归属
    return _recompute_auras(replace(state, tiles=tuple(tiles))), count


def _relocate_global_blocked(state: PuzzleState, rng: random.Random) -> tuple[PuzzleState, int]:
    """官方 3011：随机转变全图所有阻挡地块的位置至其他空白地块。

    即把阻挡地块全部搬到别的空白格——可用空间总量不变，但会重新洗牌落点位置。
    """
    blocked = [i for i, tile in enumerate(state.tiles) if tile.state == TileState.BLOCKED]
    blanks = [i for i, tile in enumerate(state.tiles) if tile.state == TileState.BLANK]
    if not blocked:
        return state, 0
    tiles = list(state.tiles)
    targets = rng.sample(blanks, min(len(blocked), len(blanks))) if blanks else []
    for index in blocked:
        tiles[index] = replace(tiles[index], state=TileState.BLANK, paint=None, buff_applied_id=None, affix_applied_id=None)
    for index in targets:
        tiles[index] = replace(tiles[index], state=TileState.BLOCKED, paint=None, buff_applied_id=None, affix_applied_id=None)
    return replace(state, tiles=tuple(tiles)), len(blocked)


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
    return _paint_kind_key(paint) == _paint_kind_key(reference)


def _randomize_scope_paints(state: PuzzleState, affix: AffixRule, rng: random.Random) -> tuple[PuzzleState, int]:
    """范围内颜料变成随机颜料（官方 2009~2011，用户口径里的「覆盖」）。

    实机设定：

    * **类型随机、等级不变**——新颜料沿用被替换那枚的**基础等级**
      （``paint.level``，不含地块加成）；
    * **地块加成保留**——覆盖只换颜料，不清地块 buff。这是 2026-11-22 补上的
      实机样本：用户那局 c4r5 原本 3 级，被转成 3 级机械之神后仍吃到此前积累的
      +2/+1，显示 6 级，且点击该地块游戏提示 +3。旧实现按「随机化会覆盖之前累积的
      buff」把 ``bonus_level`` 清零（注释里本来就标着「待验证」），已被该样本否掉；
    * 必须**先于调色效果**结算（见 :func:`settle_place_effects`）：清零
      ``buff_applied_id`` 之后，这一格会按**新类型**重新吃一次调色。

    2026-11-22 前的遗留问题（已被上面的样本回答）：若被替换的格子原本带地块加成
    （例如 3 级 + 1 = 显示 4），换出来的是 3 级还是 4 级。现在口径是 4 级：
    基础等级沿用 3，地块加成 +1 保留。
    """
    tiles = list(state.tiles)
    count = 0
    for row, col in _affix_scope_cells(state, affix):
        index = row * BOARD_SIZE + col
        tile = tiles[index]
        if tile.state != TileState.PAINT or tile.paint is None:
            continue
        category = _randomized_category(tile.paint.category, affix.target_object, rng)
        next_paint = _random_paint_with_category_level(rng, category, tile.paint.level)
        tiles[index] = replace(
            tile,
            state=TileState.PAINT,
            paint=next_paint,
            buff_applied_id=None,
            affix_applied_id=None,
            # 地块加成保留：用户 2026-11-22 的 c4r5 就是反例样本——被转成机械之神后
            # 仍吃到此前积累的 +2/+1 到 6 级，点击地块提示 +3。光环是派生量，
            # 交给 _recompute_auras 按当前锚点重算。
            bonus_level=tile_bonus_level(tile),
        )
        count += 1
    return _recompute_auras(replace(state, tiles=tuple(tiles))), count


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
    count = affix.count if affix.count is not None else len(candidates)  # 缺省=全图该类地块
    chosen = rng.sample(candidates, min(count, len(candidates))) if candidates else []
    return _clear_tile_indices(state, chosen)


def _clear_tile_indices(state: PuzzleState, indices: list[int]) -> tuple[PuzzleState, int]:
    tiles = list(state.tiles)
    for index in indices:
        # 清空的是颜料，地块加成留在原地（"炸完再放一枚直接 7 级"的来源）。
        # 但锚点上的颜料没了，同种光环随之失效——这一步交给 _recompute_auras。
        tiles[index] = replace(tiles[index], state=TileState.BLANK, paint=None, buff_applied_id=None, affix_applied_id=None)
    return _recompute_auras(replace(state, tiles=tuple(tiles))), len(indices)


def _affix_scope_cells(state: PuzzleState, affix: AffixRule | None) -> tuple[tuple[int, int], ...]:
    """笔触的作用范围：与调色**同一个纯几何口径**。

    用户 2026-11-15：「要统一」。以前这里用纯几何、调色用「几何 ∩ 画面金框」，
    同一笔里两条来源的目标集不一样，对账会报出「模型 4 画面 5」这种看不出哪边错的差。

    现在两边都用纯几何，理由见 :func:`_placement_scope_cells`：金框不亮虚空/失色格，
    而扩散能把它们转成空白地块，那些格子转完之后就在范围内、本该吃加成。

    ``state.current_anchor is None``（还没放初始颜料）时仍返回 ``()``：
    笔触是在落笔那一刻结算的，锚点都没有就没有「刚落下的这一枚」可谈。
    """
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
    # 纯几何，**不收窄到画面金框** —— 与 :func:`_placement_scope_cells` 同一个口径。
    # 金框不亮虚空/失色格，而那些格子被扩散转成空白后就在范围内、本该吃加成。
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
