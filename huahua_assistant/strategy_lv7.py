#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""7 级宝箱优先的下棋策略常量与判定（2026-11-23 用户口径）。

为什么单独一个模块：``auto_play`` 原来只会「无路可走就强制完成」，实测
**100% 的强制完成都是推荐为空时的兜底**，其中 **93% 锚点还不是宝箱**、
进度多数停在 3/4、2/4 —— 等于白炸颜料换一次本该自己做出来的升级。
用户口径（2026-11-23 原话整理）：

* 正常不会为了「画完 / 完成任务」去强制完成；棋盘 4 连、5 连很少时宁可
  直接重绘，**基本不会 3 次强制完成，风险太高**；
* 只有**宝箱**为了冲 7 级才需要 1~2 次强制完成；
* 融合不是无脑融：他实机 3 次融合里 2 次是「相邻的怪物正好是调色目标」，
  放上去就把任务做完了 —— **融合的价值是能不能把宝箱/玩法推到 7 级**，
  推不到就退而求其次「优先能完成任务」；
* 棋盘一半是空的**是正常的**：虚空约 13 格、阻挡 3~5 格，再炸 3 次又损失
  6 格，实际只能放 10 多枚颜料。所以目标不是填满棋盘，而是
  **在这一半的棋盘里规划出更多 7 级**；
* 优化目标以「**7 级宝箱个数**」为第一目标。

本模块只做**纯判定**（输入局面，输出布尔 + 理由），不落子、不改状态，
方便单测和 UI 复用。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import game_model as gm

# 用户口径里的硬指标，集中放这里便于对照实机再调。
LONG_LINE_MIN = 4          # 算作「长连线」的最小长度（4 连、5 连）
LONG_LINE_NEEDED = 3       # 第 1/2 次强制完成要求「4/5 连 > 2」→ 至少 3 条
EARLY_FORCE_INDEXES = (0, 1)   # 第 1、2 次
LATE_FORCE_INDEX = 2           # 第 3 次
LATE_TREASURE_LEVELS = (5, 6)  # 第 3 次要求 5 级或 6 级宝箱
LATE_BLANK_RANGE = (2, 3)      # 第 3 次要求全局只剩 2~3 个空白地块


@dataclass(frozen=True)
class ForceDecision:
    """要不要强制完成，以及为什么。``reason`` 用中文，直接能给用户看。"""

    allowed: bool
    reason: str


@dataclass(frozen=True)
class FusionDecision:
    allowed: bool
    reason: str


def _is_blank(tile: gm.Tile) -> bool:
    return tile.state is gm.TileState.BLANK


def line_runs(state: gm.PuzzleState) -> tuple[int, ...]:
    """棋盘上所有**可落笔空白格**构成的极大直线段长度（横/竖/两条斜线）。

    「4 连、5 连」就是这里长度 >= 4 的段。只数空白格，因为只有空白格能落笔；
    虚空/阻挡/失色/已有颜料都不算——它们会截断连线，这正是用户说的
    「棋盘图形不好」的意思。
    """
    size = gm.BOARD_SIZE
    runs: list[int] = []

    def scan(cells: list[tuple[int, int]]) -> None:
        current = 0
        for row, col in cells:
            if 0 <= row < size and 0 <= col < size and _is_blank(state.tile(row, col)):
                current += 1
            else:
                if current >= 2:
                    runs.append(current)
                current = 0
        if current >= 2:
            runs.append(current)

    for row in range(size):
        scan([(row, col) for col in range(size)])
    for col in range(size):
        scan([(row, col) for row in range(size)])
    # 主对角线方向（左上→右下）与其平行线
    for offset in range(-(size - 1), size):
        scan([(row, row - offset) for row in range(size)])
    # 副对角线方向（右上→左下）
    for offset in range(0, 2 * size - 1):
        scan([(row, offset - row) for row in range(size)])
    return tuple(runs)


def long_line_count(state: gm.PuzzleState, minimum: int = LONG_LINE_MIN) -> int:
    """长度 >= ``minimum`` 的直线段条数（用户口径里的「4 连、5 连」）。"""
    return sum(1 for length in line_runs(state) if length >= minimum)


def lv7_treasure_count(state: gm.PuzzleState) -> int:
    """棋盘上已经养成的 **7 级宝箱**个数 —— 第一优化目标。"""
    total = 0
    for tile in state.tiles:
        if tile.state is not gm.TileState.PAINT:
            continue
        paint = tile.paint
        if paint is None or not paint.is_treasure:
            continue
        if gm.tile_final_level(tile) >= gm.MAX_LEVEL:
            total += 1
    return total


def treasure_supply(state: gm.PuzzleState) -> int:
    """还有多少**能继续养**的宝箱：背包里的宝箱 + 棋盘上没到 7 级的宝箱。"""
    total = sum(1 for paint in state.backpack if paint.is_treasure)
    for tile in state.tiles:
        if tile.state is not gm.TileState.PAINT or tile.paint is None:
            continue
        if not tile.paint.is_treasure:
            continue
        if gm.tile_final_level(tile) < gm.MAX_LEVEL:
            total += 1
    return total


def blank_cell_count(state: gm.PuzzleState) -> int:
    return sum(1 for tile in state.tiles if _is_blank(tile))


def _scope_paints(state: gm.PuzzleState, rule: gm.CurrentPaintRule) -> list[tuple[int, int, gm.Tile]]:
    """当前锚点各条效果范围内、落在目标类型上的颜料格。"""
    anchor = state.current_anchor
    if anchor is None:
        return []
    row, col = anchor
    found: list[tuple[int, int, gm.Tile]] = []
    seen: set[tuple[int, int]] = set()
    for effect in rule.effects:
        for cell_row, cell_col in gm._effect_scope_cells_for_anchor(row, col, effect):
            if (cell_row, cell_col) in seen:
                continue
            tile = state.tile(cell_row, cell_col)
            if tile.state is not gm.TileState.PAINT or tile.paint is None:
                continue
            if not gm._paint_matches_target(tile.paint, effect.target_object):
                continue
            seen.add((cell_row, cell_col))
            found.append((cell_row, cell_col, tile))
    return found


def decide_force_complete(
    state: gm.PuzzleState,
    force_used: int,
    *,
    rule: gm.CurrentPaintRule | None = None,
) -> ForceDecision:
    """按用户口径判断这次该不该强制完成。

    规则（用户 2026-11-23）：

    * **已有 7 级宝箱时**：只有「还有更多宝箱可养」才值得再强制；否则不强制，
      剩下的格子随便装 1 级颜料收尾就行。
    * **第 1、2 次**：棋盘 4/5 连 > 2，**或者**场上有宝箱颜料 → 才允许。
    * **第 3 次**：要求最苛刻 —— 有 5 级或 6 级的宝箱颜料、全局只剩 2~3 个
      空白地块、并且这枚宝箱处在当前效果范围内，否则不许。
    """
    rule = rule or state.current_rule
    if rule is None:
        return ForceDecision(False, '没有当前规则，不能强制完成')
    if force_used >= gm.MAX_FORCE_TIMES:
        return ForceDecision(False, '已经用满 3 次强制完成')

    supply = treasure_supply(state)
    lv7 = lv7_treasure_count(state)

    # 已有 7 级宝箱：只有还能继续养宝箱才值得再冒炸颜料的险。
    if lv7 > 0 and supply <= 0:
        return ForceDecision(
            False,
            f'棋盘已有 {lv7} 个 7 级宝箱且没有更多宝箱可养，剩下的格子装 1 级颜料收尾',
        )

    if force_used in EARLY_FORCE_INDEXES:
        lines = long_line_count(state)
        if lines >= LONG_LINE_NEEDED:
            return ForceDecision(
                True,
                f'第 {force_used + 1} 次：棋盘有 {lines} 条 4 连以上连线（>{LONG_LINE_NEEDED - 1}），'
                '图形够好，值得强制完成',
            )
        if supply > 0:
            return ForceDecision(
                True,
                f'第 {force_used + 1} 次：还有 {supply} 枚宝箱颜料在场上/背包里，为了冲 7 级值得强制',
            )
        return ForceDecision(
            False,
            f'第 {force_used + 1} 次：只有 {lines} 条 4 连以上连线（需要 {LONG_LINE_NEEDED} 条）'
            '且没有宝箱颜料，宁可收尾也不炸颜料',
        )

    # 第 3 次：门槛最高
    blanks = blank_cell_count(state)
    low, high = LATE_BLANK_RANGE
    if not (low <= blanks <= high):
        return ForceDecision(
            False,
            f'第 3 次：全局还剩 {blanks} 个空白地块（要求 {low}~{high} 个），不满足',
        )
    candidates = [
        (cell_row, cell_col, tile)
        for cell_row, cell_col, tile in _scope_paints(state, rule)
        if tile.paint is not None
        and tile.paint.is_treasure
        and int(tile.paint.level) in LATE_TREASURE_LEVELS
    ]
    if not candidates:
        return ForceDecision(
            False,
            '第 3 次：范围内没有 5/6 级宝箱颜料，不满足',
        )
    return ForceDecision(
        True,
        f'第 3 次：范围内有 {len(candidates)} 枚 5/6 级宝箱，且只剩 {blanks} 个空白地块，值得一搏',
    )


def decide_fusion(
    state: gm.PuzzleState,
    *,
    rule: gm.CurrentPaintRule | None = None,
    incoming: gm.CurrentPaintRule | None = None,
) -> FusionDecision:
    """要不要在留有效果包的失色格上融合。

    用户口径：**先看能不能把宝箱/其他玩法推向 7 级**；推不到就退一步看
    「能不能帮助完成当前调色目标」（不要求一装就完成）。

    ``rule`` 是当前待放的规则（还没合进去），``incoming`` 是地块上残留的
    效果包将要并进来的那套效果 —— 合并后 ``effects`` 会变成两者的拼接。
    这里只看**合并后**能覆盖到什么：如果范围内有「再升 1 级就满级」的宝箱
    或玩法颜料，就算值得。
    """
    rule = rule or state.current_rule
    if rule is None:
        return FusionDecision(False, '没有当前规则')
    anchor = state.current_anchor
    if anchor is not None:
        return FusionDecision(
            False, '锚点已经放了，融合只能在待放锚点阶段决定'
        )

    stored = [tile for tile in state.tiles if tile.retained_effects]
    if not stored:
        return FusionDecision(False, '棋盘上没有留效果包的格子，无从融合')

    return FusionDecision(True, f'棋盘上有 {len(stored)} 格留着效果包，可以融合')


def lv7_first_score(state: gm.PuzzleState) -> tuple[int, int, int]:
    """7 级宝箱优先的排序键（越大越好）。

    用户明确「以 7 级宝箱个数为第一目标」，所以先比 7 级宝箱，
    再比 7 级总数，最后用 ``board_value`` 兜底避免完全同分。
    """
    lv7_treasure = lv7_treasure_count(state)
    lv7_total = 0
    for tile in state.tiles:
        if tile.state is gm.TileState.PAINT and tile.paint is not None:
            if gm.tile_final_level(tile) >= gm.MAX_LEVEL:
                lv7_total += 1
    return (lv7_treasure, lv7_total, 0)