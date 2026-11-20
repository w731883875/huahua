"""S8「画中雪原 · 作画」官方调色目标 / 调色效果表（27 条）。

数据来源（可逐条复核）：

* 结构：``Config/LuaTable/DataCfg/config_s8_gameplay_itemeffect.lua.bin``
  字段常量池可见 ``TaskDirection`` / ``ActionDirection`` / ``TaskType`` /
  ``ActionType`` / ``TaskNum`` / ``ActionParam`` / ``LibraryId`` / ``Weight`` /
  ``UnlockDifficulty``，共 27 条记录（1101~1108、2001~2008、2010~2013、
  3001~3006、4001）。
* 文案：``ConfigView/config_multi_language.bin`` 的
  ``S8_Gameplay_ItemEffect|EffectDes|{id}`` 与 ``|TaskDes|{id}``。

官方文案成对出现，且**目标方向/类型与效果方向/类型始终一致**：

    在{方向}地块放置{类型}颜料   →   使{方向}的{类型}颜料提升{N}级

所以本表只记录 (方向, 类型, 提升级数) 三元组 + 两条官方原文。

可靠性说明：Lua 常量池会去重，``TaskNum``（调色进度要求）等数值无法逐条
归回记录，因此这里**不猜测 TaskNum**；进度要求仍由
``game_model.empirical_defaults_for_level`` 按颜料等级给经验值。
"""

from __future__ import annotations

from typing import NamedTuple

from .models import PaintDirection, PaintObject


class S8PlotEffect(NamedTuple):
    """一条官方「调色目标 + 调色效果」。"""

    id: int                     # 官方记录 ID
    direction: PaintDirection   # 作用方向（目标方向与效果方向相同）
    target: PaintObject         # 作用对象（目标类型与效果类型相同）
    amount: int                 # 提升级数：官方只有 1 或 2
    task: str                   # 官方 TaskDes 原文
    effect: str                 # 官方 EffectDes 原文

    @property
    def label(self) -> str:
        return f"{self.task} → {self.effect}"


_H = PaintDirection.HORIZONTAL
_V = PaintDirection.VERTICAL
_D = PaintDirection.DIAGONAL
_A = PaintDirection.ADJACENT
_G = PaintDirection.GLOBAL

_MONSTER = PaintObject.MONSTER
_GAMEPLAY = PaintObject.GAMEPLAY
_ALL = PaintObject.ALL


PLOT_EFFECTS: tuple[S8PlotEffect, ...] = (
    # 方向 × 单类颜料，提升 1 级（1101~1108）
    S8PlotEffect(1101, _H, _MONSTER, 1, "在横向地块放置怪物颜料", "使横向的怪物颜料提升1级"),
    S8PlotEffect(1102, _V, _MONSTER, 1, "在纵向地块放置怪物颜料", "使纵向的怪物颜料提升1级"),
    S8PlotEffect(1103, _D, _MONSTER, 1, "在对角地块放置怪物颜料", "使对角的怪物颜料提升1级"),
    S8PlotEffect(1104, _A, _MONSTER, 1, "在相邻地块放置怪物颜料", "使相邻的怪物颜料提升1级"),
    S8PlotEffect(1105, _H, _GAMEPLAY, 1, "在横向地块放置玩法颜料", "使横向的玩法颜料提升1级"),
    S8PlotEffect(1106, _V, _GAMEPLAY, 1, "在纵向地块放置玩法颜料", "使纵向的玩法颜料提升1级"),
    S8PlotEffect(1107, _D, _GAMEPLAY, 1, "在对角地块放置玩法颜料", "使对角的玩法颜料提升1级"),
    S8PlotEffect(1108, _A, _GAMEPLAY, 1, "在相邻地块放置玩法颜料", "使相邻的玩法颜料提升1级"),
    # 方向 × 单类颜料，提升 2 级（2001~2008）
    S8PlotEffect(2001, _H, _MONSTER, 2, "在横向地块放置怪物颜料", "使横向的怪物颜料提升2级"),
    S8PlotEffect(2002, _V, _MONSTER, 2, "在纵向地块放置怪物颜料", "使纵向的怪物颜料提升2级"),
    S8PlotEffect(2003, _D, _MONSTER, 2, "在对角地块放置怪物颜料", "使对角的怪物颜料提升2级"),
    S8PlotEffect(2004, _A, _MONSTER, 2, "在相邻地块放置怪物颜料", "使相邻的怪物颜料提升2级"),
    S8PlotEffect(2005, _H, _GAMEPLAY, 2, "在横向地块放置玩法颜料", "使横向的玩法颜料提升2级"),
    S8PlotEffect(2006, _V, _GAMEPLAY, 2, "在纵向地块放置玩法颜料", "使纵向的玩法颜料提升2级"),
    S8PlotEffect(2007, _D, _GAMEPLAY, 2, "在对角地块放置玩法颜料", "使对角的玩法颜料提升2级"),
    S8PlotEffect(2008, _A, _GAMEPLAY, 2, "在相邻地块放置玩法颜料", "使相邻的玩法颜料提升2级"),
    # 方向 × 任意颜料，提升 1 级（2010~2013）
    S8PlotEffect(2010, _H, _ALL, 1, "在横向地块放置任何颜料", "使横向的所有颜料提升1级"),
    S8PlotEffect(2011, _V, _ALL, 1, "在纵向地块放置任何颜料", "使纵向的所有颜料提升1级"),
    S8PlotEffect(2012, _D, _ALL, 1, "在对角地块放置任何颜料", "使对角的所有颜料提升1级"),
    S8PlotEffect(2013, _A, _ALL, 1, "在相邻地块放置任何颜料", "使相邻的所有颜料提升1级"),
    # 方向 × 任意颜料，提升 2 级（3001~3004）
    S8PlotEffect(3001, _H, _ALL, 2, "在横向地块放置任何颜料", "使横向的所有颜料提升2级"),
    S8PlotEffect(3002, _V, _ALL, 2, "在纵向地块放置任何颜料", "使纵向的所有颜料提升2级"),
    S8PlotEffect(3003, _D, _ALL, 2, "在对角地块放置任何颜料", "使对角的所有颜料提升2级"),
    S8PlotEffect(3004, _A, _ALL, 2, "在相邻地块放置任何颜料", "使相邻的所有颜料提升2级"),
    # 全图（3005、3006、4001）
    S8PlotEffect(3005, _G, _MONSTER, 1, "在全图范围放置怪物颜料", "使全图的怪物颜料提升1级"),
    S8PlotEffect(3006, _G, _GAMEPLAY, 1, "在全图范围放置玩法颜料", "使全图的玩法颜料提升1级"),
    S8PlotEffect(4001, _G, _ALL, 2, "在全图范围放置任何颜料", "使全图的所有颜料提升2级"),
)

#: 按 (方向, 对象, 级数) 反查官方记录，便于 OCR 读到的文案直接对上表。
_BY_KEY: dict[tuple[PaintDirection, PaintObject, int], S8PlotEffect] = {
    (item.direction, item.target, item.amount): item for item in PLOT_EFFECTS
}

#: 官方 27 条里「全图」占 3 条；抽取时按此比例给权重，其余方向均分。
GLOBAL_DIRECTION_SHARE = 3 / 27


def effect_for(
    direction: PaintDirection,
    target: PaintObject,
    amount: int,
) -> S8PlotEffect | None:
    """反查官方记录；不存在说明这组参数在实机里不会出现。"""
    return _BY_KEY.get((direction, target, max(1, min(2, amount))))


def is_official_combination(
    direction: PaintDirection,
    target: PaintObject,
    amount: int,
) -> bool:
    return effect_for(direction, target, amount) is not None
