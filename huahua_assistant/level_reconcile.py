"""「模型推算的等级」与「画面识别到的等级」逐格对账。

## 为什么需要它

工具里有两条完全独立的等级来源，用户看到差异时必须能判断差在哪：

* **画面识别**：游戏 UI 上那个数字，直接读出来（``paint.level``）；
* **模型推算**：拿上一帧的局面 + 当前调色规则，按 ``game_model`` 的规则算一遍。

两者不一致，只可能是三种原因之一：模型缺了规则（漏词缀/漏解析）、识别读错了、
或者面板读数与真实落笔不符。以前只有 ``_unexpected_level_changes`` 那种
「帧间涨幅有没有超过调色量」的启发式——既不给出「模型以为是多少」，
也不会把差异落到具体格子上。

## 为什么是「增量对账」而不是「整局重放」

画面上的等级**已经包含了加成**。如果把「画面等级」当作落笔前的等级再跑一遍落笔，
加成会被算两遍，结果永远是错的。所以这里只让模型回答一个增量问题：

    「按上一帧的规则，这一笔之后每格应该 +N？」

然后把 ``上一帧等级 + N`` 和这一帧的识别值逐格比。加成由
:func:`game_model._apply_current_rule_upgrade` 自己算（不另写一套），
调色目标/作用范围/是否已吃过本 buff 全部沿用模型口径。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .game_model import (
    BOARD_SIZE,
    MAX_LEVEL,
    AffixEffect,
    AffixRule,
    LevelSource,
    Paint,
    PuzzleState,
    Tile,
    TileState,
    _affix_scope_cells,
    _apply_current_rule_upgrade,
    is_cover_affix,
    level_sources,
    scope_upgrade_targets,
    tile_level_breakdown,
    _tile_with_upgrade,
    tile_aura_level,
    tile_bonus_level,
    tile_final_level,
)

#: 只用于让 ``describe()`` 拼一句人话
SOURCE_LABELS = {"tint": "调色", "brush": "笔触", "self": "自身"}


@dataclass(frozen=True)
class CellMismatch:
    """一格「模型预测 ≠ 画面识别」。"""

    row: int
    col: int
    predicted: int
    observed: int
    #: ``existing`` = 早就有的颜料；``new`` = 这一帧刚落下的
    kind: str = "existing"
    #: 这一格还可能是「随机 N 个」没抽中的结果时，``predicted`` 是**下限**，
    #: ``upper`` 是上限；``observed`` 落在两者之间就不算差异。
    upper: int | None = None
    #: 模型认为这一格的等级由什么组成：``((来源名, 级数), ...)``
    breakdown: tuple[tuple[str, int], ...] = ()

    @property
    def delta(self) -> int:
        return self.observed - self.predicted

    @property
    def breakdown_text(self) -> str:
        if not self.breakdown:
            return ""
        return " + ".join(f"{label}{amount}" for label, amount in self.breakdown)

    def describe(self) -> str:
        tag = "" if self.kind == "existing" else "（新落笔）"
        detail = f"（{self.breakdown_text}）" if self.breakdown_text else ""
        if self.upper is not None and self.upper != self.predicted:
            return (
                f"r{self.row + 1}c{self.col + 1}{tag} 模型{self.predicted}~{self.upper} "
                f"画面{self.observed}{detail}"
            )
        return f"r{self.row + 1}c{self.col + 1}{tag} 模型{self.predicted} 画面{self.observed}{detail}"


@dataclass(frozen=True)
class ReconcileReport:
    """一次对账的结论。"""

    #: 有没有真的算出预期（没有新落笔时是 False）
    ran: bool = False
    #: 没跑的原因，直接展示给人看
    note: str = ""
    #: 本帧新落笔的格子
    installed: tuple[tuple[int, int], ...] = ()
    #: 逐格差异
    mismatches: tuple[CellMismatch, ...] = ()
    #: 逐格预期：(row, col, predicted, observed)
    levels: tuple[tuple[int, int, int, int], ...] = field(default=())
    #: 吃到「随机 N 个」加成、但具体哪一格由服务端摇的格子
    uncertain: tuple[tuple[int, int], ...] = ()
    #: 这一笔按来源拆开的表（每个来源自带范围与目标类别），界面直接展示
    sources: tuple[LevelSource, ...] = ()
    #: 推演时做了哪些妥协（例如覆盖类笔触的类型取画面值）。空 = 纯模型推演。
    caveat: str = ""
    #: 模型认为这一笔会升等级的格数（下限 / 上限）。上限 > 下限 是因为
    #: 「范围里随机 N 个」只能确定区间。
    upgrade_cells: tuple[int, int] = (0, 0)
    #: 游戏日志里实际播了几条升级动画（``Anim_PutongShengji`` / ``CaiShengji``），
    #: ``None`` = 没接日志或这一笔还没播完。由调用方填，模型不猜。
    animation_upgrades: int | None = None

    @property
    def animation_matches(self) -> bool | None:
        """动画条数与模型预期是否吻合。``None`` = 没有可比的数据。"""
        if self.animation_upgrades is None:
            return None
        low, high = self.upgrade_cells
        return low <= self.animation_upgrades <= high

    @property
    def consistent(self) -> bool:
        return self.ran and not self.mismatches

    def source_lines(self) -> tuple[str, ...]:
        """给界面的一行行来源说明：目标类别 / 范围格数 / 级数。"""
        return tuple(source.describe() for source in self.sources)

    def _animation_text(self) -> str:
        low, high = self.upgrade_cells
        span = f"{low}" if low == high else f"{low}~{high}"
        if self.animation_upgrades is None:
            return f"，模型预计升{span}格"
        verdict = "吻合" if self.animation_matches else "**对不上**"
        return f"，模型预计升{span}格 / 动画播了{self.animation_upgrades}条（{verdict}）"

    def summary(self) -> str:
        if not self.ran:
            return self.note or "本帧没有新的落笔，无对账"
        head = f"已推演 {len(self.levels)} 格"
        extra = f"，{len(self.uncertain)} 格随机加成未定" if self.uncertain else ""
        caveat = f"（{self.caveat}）" if self.caveat else ""
        if not self.mismatches:
            return f"{head}，模型与画面一致{extra}{self._animation_text()}{caveat}"
        detail = "；".join(item.describe() for item in self.mismatches[:4])
        more = "" if len(self.mismatches) <= 4 else f" 等{len(self.mismatches)}处"
        return f"{head}，{len(self.mismatches)} 格对不上（{detail}{more}）{extra}{self._animation_text()}{caveat}"


def _paint_cells(tiles) -> set[tuple[int, int]]:
    return {
        (index // BOARD_SIZE, index % BOARD_SIZE)
        for index, tile in enumerate(tiles)
        if tile.state == TileState.PAINT and tile.paint is not None
    }


def installed_cells(
    previous: PuzzleState | None, current: PuzzleState | None
) -> tuple[tuple[int, int], ...]:
    """本帧新落笔的格子（上一帧不是颜料、这一帧是）。"""
    if previous is None or current is None:
        return ()
    return tuple(sorted(_paint_cells(current.tiles) - _paint_cells(previous.tiles)))


def _level(paint: Paint | None) -> int:
    return int(paint.level) if paint is not None else 0


def _place_paint(
    state: PuzzleState, row: int, col: int, paint: Paint
) -> PuzzleState:
    """把一枚颜料放到 ``(row, col)``，地块自带的加成/光环原样保留。"""
    tiles = list(state.tiles)
    index = row * BOARD_SIZE + col
    base = tiles[index]
    tiles[index] = Tile(
        state=TileState.PAINT,
        paint=paint,
        buff_applied_id=base.buff_applied_id,
        affix_applied_id=base.affix_applied_id,
        bonus_level=tile_bonus_level(base),
        aura_amount=tile_aura_level(base),
    )
    return replace(state, tiles=tuple(tiles))


def _bump(state: PuzzleState, index: int, amount: int, as_bonus: bool) -> PuzzleState:
    tiles = list(state.tiles)
    tiles[index] = _tile_with_upgrade(tiles[index], amount, as_bonus=as_bonus)
    return replace(state, tiles=tuple(tiles))


def _cover_intermediate(
    placed: PuzzleState,
    current: PuzzleState,
    previous: PuzzleState,
    rule,
    affix: AffixRule,
) -> PuzzleState:
    """重建「刚被覆盖完、调色还没结算」的中间态。

    覆盖类笔触把范围内颜料的**类型**交给服务端摇，模型无法预测；但**等级基准**能推：

    * 2009~2011「转变为同等级的随机颜料」→ 等级 = 上一帧这一格的等级；
    * 2012「等级和内容替换为自身」→ 等级 = 自身等级。

    **地块加成两种情况都保留**（用户 2026-11-22 实机更正）：初始颜料的加成是加在
    地块上的，换一枚颜料这 N 级仍然在。之前这里把 ``bonus`` 清零，结果是颜料被
    覆盖后模型算出的等级比画面低一截——实测 c4r5 那格：模型 3、画面 6、
    ``breakdown`` 还是空的，对账因此天天报红。

    类型一律取**这一帧画面上**的实测值——它已经是覆盖后的结果。这样对账只会因为
    「等级基准算错」报红，而不是因为随不到同一个类型就每笔都红，随机性不再是
    万能挡箭牌（用户 2026-11-22：要能"方便找到问题修改"）。
    """
    tiles = list(placed.tiles)
    scope = set(_affix_scope_cells(placed, affix))
    self_level = _level(rule.paint)
    copy_self = getattr(affix, "effect", None) is AffixEffect.COPY_SELF_TO_SCOPE
    for row, col in scope:
        index = row * BOARD_SIZE + col
        if not 0 <= index < len(tiles):
            continue
        observed = current.tile(row, col).paint
        if observed is None:
            continue
        keep = tiles[index]
        # 地块加成跟着地块走，不跟着颜料走：覆盖不改变它。
        bonus = tile_bonus_level(keep)
        if copy_self:
            base = self_level
            aura_amount = max(0, int(keep.aura_amount))
        else:
            base = _level(previous.tile(row, col).paint)
            aura_amount = max(0, int(keep.aura_amount))
        if base <= 0:
            continue
        tiles[index] = Tile(
            state=TileState.PAINT,
            paint=replace(observed, level=min(MAX_LEVEL, base)),
            bonus_level=bonus,
            aura_amount=aura_amount,
            aura_level=tile_aura_level(tiles[index]),
        )
    return replace(placed, tiles=tuple(tiles))


def reconcile(
    previous: PuzzleState | None,
    current: PuzzleState | None,
    *,
    self_upgrade: int = 0,
    active_affix: AffixRule | None = None,
) -> ReconcileReport:
    """逐格对账。

    参数：

    * ``self_upgrade`` —— 这一笔额外吃到的「使自身等级提升 N 级」（笔触词缀），
      由调用方从已记录的选择里给出，模型自身不猜；
    * ``active_affix`` —— 本局挂着的笔触词缀（作用范围类）。识别出来的局面不带
      词缀，所以只能由调用方给；缺省时退回 ``previous.active_affix``。

    **三个来源必须分开算**（用户 2026-11-15 点名，且纠正过两次）：

    * 调色效果 = 按**目标类别**给等级，走 :func:`_apply_current_rule_upgrade`；
    * 笔触效果 = 作用范围里「随机 N 个」提升，走 :func:`scope_upgrade_targets`；
    * 自身提升 = 只给这一笔落下的**初始颜料自己**，不吃范围也不吃目标。

    每个来源都有自己的范围和目标类别（见 :func:`game_model.level_sources`），
    谁也不借谁的：

    * 纵向上放一枚背包颜料，范围里有 4 级英灵（``category=monster``），调色目标是玩法
      → 英灵**只**被笔触 +1 变成 5，调色那 1 级它拿不到；
    * 同一格如果放的是 4 级玩法颜料 → 调色 +1 与笔触 +1 **都吃**，变 6；
    * 初始颜料自己不属于作用范围（范围按定义排除锚点），所以「范围内随机 1 个」
      永远不会把初始颜料顶上去——它只吃「使自身等级提升 N 级」。
    """
    if previous is None or current is None:
        return ReconcileReport(note="还没有上一帧可比")
    rule = previous.current_rule
    if rule is None:
        return ReconcileReport(note="上一帧没有调到初始颜料规则，无法推演")

    installed = installed_cells(previous, current)
    if not installed:
        # 没有新落笔就没有「这一笔」可解释。此时拿规则去套静态盘面只会误报：
        # 规则里的 +N 是在落笔那一刻结算的，颜料没落下去之前它本就不该生效。
        return ReconcileReport(note="本帧没有新的落笔，无对账")
    existing = _paint_cells(previous.tiles) & _paint_cells(current.tiles)

    # ---- 第 1 步：把这一帧新落下的颜料补进去 -----------------------------
    # 必须补：笔触的「作用范围提升」是在落笔那一刻结算的，范围里要有东西可提升。
    placed = previous
    for order, (row, col) in enumerate(installed):
        # 「使自身提升 N 级」只兑现给这一次落下的那一枚。
        bonus = max(0, int(self_upgrade)) if order == 0 else 0
        # 落下去的到底是**哪一枚**颜料：类别/名字以画面为准，只有「等级基准」取面板。
        # 面板等级 = 游戏在落笔时给的初始等级，这才是模型该假设的基础值；
        # 若落笔的其实是背包里另一枚（类别和面板不同），照面板类别去推会算错加成：
        # 实测「面板是怪物、落下去是玩法」时，调色那 1 级会被漏掉。
        template = current.tile(row, col).paint or rule.paint
        base_paint = replace(template, level=min(MAX_LEVEL, _level(rule.paint) + bonus))
        placed = _place_paint(placed, row, col, base_paint)
    if placed.current_anchor is None and installed:
        # 这一帧放的就是初始颜料：锚点从「待放置」变成落点。不补这一步，作用范围
        # 与「自身提升」都没有依据，来源表会缺项（不是识别问题，是推演缺输入）。
        placed = replace(placed, current_anchor=installed[0])

    # ---- 第 2 步：调色效果（按目标类别） ---------------------------------
    affix = active_affix if active_affix is not None else previous.active_affix
    # 覆盖类笔触（官方 2009~2012）会先把范围内颜料的**类型**改掉，再让调色按新类型
    # 判定。类型是服务端摇的，模型推不出来，但**画面已经告诉我们了**——所以这里用
    # 画面类型重建一个「刚被覆盖完」的中间态，只把等级基准交给模型。
    # 不做这一步，「3 玩法 + 1 怪物」这种结果会被逐格报红，而红的其实是随机性。
    if is_cover_affix(affix):
        expected = _cover_intermediate(placed, current, previous, rule, affix)
        cover_note = "本笔为覆盖类笔触，类型以画面为准、等级基准按文案推"
    else:
        expected = placed
        cover_note = ""
    expected, _count = _apply_current_rule_upgrade(expected, rule)

    # ---- 第 3 步：笔触效果（作用范围随机 N 个） ---------------------------
    # 「随机 N 个」在候选多于 N 时只能确定区间：下限 = 一个都没抽中，
    # 上限 = 抽中了这一格。画面落在区间内就不该报警，否则每笔都在红。
    # 范围里一个颜料都没有时**不会**兜底提升这一笔——初始颜料更不会（用户 2026-11-15）。
    # 落点若在范围里，它落笔后本来就成了范围内的候选，不需要额外照顾。
    sure, maybe, amount = scope_upgrade_targets(expected, affix)
    as_bonus = bool(getattr(affix, "grants_bonus", False)) if affix is not None else False
    lower = expected
    for index in sure:
        lower = _bump(lower, index, amount, as_bonus)
    upper = lower
    for index in maybe:
        upper = _bump(upper, index, amount, as_bonus)

    # 来源表：调色 / 笔触 / 自身，各自带范围与目标类别，界面照抄即可。
    sources = level_sources(expected, affix=affix, self_upgrade=max(0, int(self_upgrade)))

    levels: list[tuple[int, int, int, int]] = []
    mismatches: list[CellMismatch] = []
    uncertain: list[tuple[int, int]] = []
    #: 吃满上限（=可能被随机抽中）的格子：(row, col, low, high, observed, kind)
    saturated: list[tuple[int, int, int, int, int, str]] = []
    #: 模型认为这一笔会**升等级**的格数（下限 / 上限）。
    #: 游戏每升一格播一条 ``Anim_PutongShengji``，所以这两个数可以直接和
    #: 日志里的动画条数对：对不上就说明模型漏了一条来源，或识别读错了格子。
    upgrade_cells_min = 0
    upgrade_cells_max = 0

    for row, col in sorted(existing | set(installed)):
        current_paint = current.tile(row, col).paint
        if current_paint is None:
            continue
        low = tile_final_level(lower.tile(row, col))
        high = tile_final_level(upper.tile(row, col))
        observed = _level(current_paint)
        kind = "new" if (row, col) in installed else "existing"
        # 这一笔**之前**的等级：新落笔格 = 面板等级，老格子 = 上一帧画面值。
        # 传错了就会把已经生效的加成再算一遍，所以必须分开取。
        base_before = (
            _level(rule.paint)
            if kind == "new"
            else _level(previous.tile(row, col).paint)
        )
        before_final = (
            base_before if kind == "new" else tile_final_level(previous.tile(row, col))
        )
        if low > before_final:
            upgrade_cells_min += 1
        if high > before_final:
            upgrade_cells_max += 1
        _base, parts = tile_level_breakdown(
            expected, row, col, sources, base_level=base_before
        )
        levels.append((row, col, low, observed))
        if low <= observed <= high:
            # 落在区间里：要么没吃随机加成（=下限），要么吃了一部分（=上限）。
            if high != low and observed != low:
                uncertain.append((row, col))
                saturated.append((row, col, low, high, observed, kind))
            continue
        mismatches.append(
            CellMismatch(
                row=row,
                col=col,
                predicted=low,
                observed=observed,
                kind=kind,
                upper=high if high != low else None,
                breakdown=parts,
            )
        )

    # 「随机 N 个」最多只能有 N 格吃满。实测画面里吃满的格数超过 N，
    # 说明多出来的等级不是这条笔触给的 —— 必须报出来，否则随机性成了万能挡箭牌。
    random_limit = getattr(affix, "count", None) if affix is not None else None
    if random_limit is not None and len(saturated) > max(0, int(random_limit)):
        overflow = {cell[:2] for cell in saturated}
        uncertain = [cell for cell in uncertain if cell not in overflow]
        for row, col, low, high, observed, kind in saturated:
            _base, parts = tile_level_breakdown(
                expected,
                row,
                col,
                sources,
                base_level=(
                    _level(rule.paint)
                    if kind == "new"
                    else _level(previous.tile(row, col).paint)
                ),
            )
            mismatches.append(
                CellMismatch(
                    row=row,
                    col=col,
                    predicted=low,
                    observed=observed,
                    kind=kind,
                    upper=high,
                    breakdown=parts,
                )
            )

    if not levels:
        return ReconcileReport(
            note="本帧没有可比对的颜料格",
            installed=installed,
            sources=sources,
            caveat=cover_note,
        )
    return ReconcileReport(
        ran=True,
        installed=installed,
        mismatches=tuple(mismatches),
        levels=tuple(sorted(levels)),
        uncertain=tuple(uncertain),
        sources=sources,
        caveat=cover_note,
        upgrade_cells=(upgrade_cells_min, upgrade_cells_max),
    )
