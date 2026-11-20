"""显式、可调的终局价值函数。

TartarusSolver 的关键做法之一：不再把"这步好不好"拆成几十个魔法权重相加，
而是定义一个**能直接解释的终局目标**，让搜索去最大化它。

这里的刻度就是**分**，由用户给定（2026-11-12），是线性的：

* 1~7 级怪物 = 1~7 分
* 1~7 级玩法 = 2~14 分
* 1~7 级宝箱 = 10~70 分（最高单格价值）

**唯一目标就是达到 7 级宝箱。** 游戏本身按"完成地块数"结算奖励，但那是游戏设定，
不是我们要优化的东西，所以这里刻意**没有**"完成地块数"这一项。

两处容易搞错的地方：

* 等级要按**最终等级**（颜料等级 + 地块加成）算，加成挂在地块上；
* 没有颜料但存着加成的地块不是废地，要按"能接住的高等级颜料"单独给价
  （``EvalWeights.bonus_banked``）——否则搜索会把"炸掉颜料"当成纯损失。
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import game_model as gm


@dataclass(frozen=True)
class EvalWeights:
    """终局价值权重，**直接就是分**。

    刻度由用户给定（2026-11-12），是**线性**的，不是指数：

    * 1~7 级怪物 = 1~7 分（n）
    * 1~7 级玩法 = 2~14 分（2n）
    * 1~7 级宝箱 = **10~70 分**（10n）

    同一等级下宝箱 > 玩法 > 怪物，且 7 级宝箱的 70 分是全局最高单格价值。

    之前用的是接近指数的曲线（宝箱 5→7 级 0.22→1.0，四倍多），
    会把"堆满高级宝箱"的收益相对压低、把"铺很多低级颜料"抬高，
    与实机按等级线性的结算不符。这里改为严格线性。
    """

    treasure: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)
    gameplay: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
    monster: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    # 空地块按"占 1 个 7 级宝箱(70 分)的比例"折算，与线性刻度对齐。
    blank: float = 0.56
    faded: float = -1.40
    blocked: float = -0.35
    void: float = -0.14
    unknown: float = 0.0
    bonus_banked: float = 0.45
    """空地块上存着的加成值多少（占"该等级颜料价值"的比例）。

    加成挂在地块上，所以"颜料没了、加成还在"的格子不是废地——之后落一枚低等级
    颜料就能直接到高等级。这是本玩法最强的得分路径，不单独给价的话搜索会
    把强制完成当成纯损失。系数按三类颜料的均值折算后再打折（还得再补一枚
    颜料，且不一定补得上），是个待用 ``compare`` 校准的先验值，不是实测值。
    """
    """当前规则推进度的权重：推进度 = 已完成匹配 / 需要匹配。"""
    progress: float = 4.2
    """一条规则到步数上限还没结算时的惩罚：白白占用了地图空间。"""
    unfinished_rule: float = -2.1

    def paint_value(self, paint: gm.Paint) -> float:
        level = max(0, min(gm.MAX_LEVEL, int(paint.level)))
        if paint.is_treasure:
            return self.treasure[level]
        if paint.category is gm.PaintCategory.GAMEPLAY:
            return self.gameplay[level]
        if paint.category is gm.PaintCategory.MONSTER:
            return self.monster[level]
        return self.gameplay[level] * 0.5

    def empty_tile_value(self, state: gm.TileState, bonus_level: int) -> float:
        """没有颜料的地块值多少。存着加成时按"能接住的高等级颜料"折算。"""
        base = {
            gm.TileState.BLANK: self.blank,
            gm.TileState.FADED: self.faded,
            gm.TileState.BLOCKED: self.blocked,
            gm.TileState.VOID: self.void,
        }.get(state, self.unknown)
        bonus = max(0, int(bonus_level))
        if bonus <= 0:
            return base
        level = min(gm.MAX_LEVEL, 1 + bonus)
        mean = (self.treasure[level] + self.gameplay[level] + self.monster[level]) / 3.0
        return max(base, self.bonus_banked * mean)


DEFAULT_WEIGHTS = EvalWeights()


def board_value(state: gm.PuzzleState, weights: EvalWeights = DEFAULT_WEIGHTS) -> float:
    """棋盘当前值多少分。不看当前规则，只看棋盘本身。

    刻度见 :class:`EvalWeights`：1 级怪物 1 分 … 7 级宝箱 70 分。

    按**最终等级**（颜料等级 + 地块加成）估值：加成挂在地块上，
    所以一枚低等级颜料放在叠满加成的格子上，价值等同于高等级颜料。
    """
    total = 0.0
    for tile in state.tiles:
        effective = gm.tile_effective_paint(tile)
        if effective is not None and tile.state is gm.TileState.PAINT:
            total += weights.paint_value(effective)
            continue
        total += weights.empty_tile_value(tile.state, gm.tile_bonus_level(tile))
    return total


def position_value(state: gm.PuzzleState, weights: EvalWeights = DEFAULT_WEIGHTS) -> float:
    """棋盘价值 + 当前规则的推进度。用于"同一规则内的两手棋"比较。"""
    value = board_value(state, weights)
    rule = state.current_rule
    if rule is not None and not gm.current_rule_is_pending(state):
        required = max(1, rule.match_required)
        value += weights.progress * min(1.0, rule.match_current / required)
    return value


def terminal_value(
    state: gm.PuzzleState,
    *,
    finished: bool,
    weights: EvalWeights = DEFAULT_WEIGHTS,
) -> float:
    """推演终点的价值。``finished`` 表示这条规则已经结算。"""
    value = board_value(state, weights)
    if not finished:
        value += weights.unfinished_rule
    return value


def level7_counts(state: gm.PuzzleState) -> tuple[int, int]:
    """(7 级宝箱数, 任意 7 级颜料数)，按最终等级统计，用于报表。"""
    treasures = 0
    any7 = 0
    for tile in state.tiles:
        effective = gm.tile_effective_paint(tile)
        if effective is None or tile.state is not gm.TileState.PAINT:
            continue
        if effective.level < gm.MAX_LEVEL:
            continue
        any7 += 1
        if effective.is_treasure:
            treasures += 1
    return treasures, any7
