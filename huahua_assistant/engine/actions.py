"""统一的动作空间。

对齐 TartarusSolver 的 ``GameAction``：决策层只认这一种"动作"，
不再把"推荐落点 / 安装计划 / 强制完成"当成三套互不相干的函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    """一个决策点上的三种可选动作。"""

    PLACE_ANCHOR = "place_anchor"
    """把待放置的初始颜料放到 (row, col)。"""

    INSTALL = "install"
    """把背包第 backpack_index 个颜料安装到 (row, col)。"""

    FORCE_COMPLETE = "force_complete"
    """强制完成当前规则（会随机炸毁若干格）。"""


@dataclass(frozen=True, order=True)
class Action:
    """一个动作。字段按 kind 取用，未使用的保持 -1。"""

    kind: ActionKind
    row: int = -1
    col: int = -1
    backpack_index: int = -1

    def label(self) -> str:
        if self.kind is ActionKind.PLACE_ANCHOR:
            return f"放置初始颜料 ({self.row},{self.col})"
        if self.kind is ActionKind.INSTALL:
            return f"安装背包#{self.backpack_index} 到 ({self.row},{self.col})"
        return "强制完成"


@dataclass(frozen=True)
class ActionValue:
    """一个动作在若干次推演后的估值。"""

    action: Action
    mean: float
    stderr: float
    playouts: int


@dataclass(frozen=True)
class SearchResult:
    actions: tuple[ActionValue, ...]
    playouts: int
    elapsed: float

    @property
    def best(self) -> ActionValue | None:
        return self.actions[0] if self.actions else None
