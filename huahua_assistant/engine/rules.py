"""规则层：只回答三个问题 —— 现在能做什么、做了会怎样、这一步算不算结束。

对齐 TartarusSolver 的 ``Engine``（``Legal`` / ``Apply`` / ``Determinize``）。
本模块不复制任何游戏规则，全部复用 ``game_model`` 里已有的状态转移，
只是把它们收进一个统一的动作接口，并让随机源可以注入（推演必须可复现）。
"""

from __future__ import annotations

import random
from dataclasses import replace

from .. import game_model as gm
from .actions import Action, ActionKind


class RuleEngine:
    """把 ``game_model`` 的状态转移包装成 (合法动作, 应用, 终局判断)。

    ``assume_supply`` 对应实机结论：**每局背包的颜料是满足的**。
    推演时背包只出不进，跨规则很快就会见底；这时按 ``supply`` 画像补一枚，
    而不是让推演停在"没有可用颜料"。``supply`` 从真实背包捕获（见 ``with_supply_from``），
    所以补出来的是"观察到的颜料"，不是凭空的魔法数字。
    """

    def __init__(
        self,
        *,
        max_steps: int = 12,
        assume_supply: bool = True,
        supply: tuple[gm.Paint, ...] | list[gm.Paint] | None = None,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self.assume_supply = bool(assume_supply)
        self.supply = tuple(supply or ())

    def with_supply_from(self, state: gm.PuzzleState) -> "RuleEngine":
        """用真实背包的颜料画像构造一个引擎：推演中见底时按它补。"""
        return RuleEngine(
            max_steps=self.max_steps,
            assume_supply=self.assume_supply,
            supply=state.backpack,
        )

    def _supply_paint(self, rule: gm.CurrentPaintRule) -> gm.Paint:
        """补一枚匹配当前目标的颜料。

        优先复用真实背包里观察到的同类颜料（取最高级，因为实际也会先挑好的用）；
        画像里没有同类时退化成保守的 Lv2。
        """
        matching = [paint for paint in self.supply if gm._paint_matches_target(paint, rule.target_object)]
        if matching:
            return max(matching, key=lambda paint: paint.level)
        if rule.target_object is gm.PaintObject.MONSTER:
            category = gm.PaintCategory.MONSTER
        elif rule.target_object is gm.PaintObject.GAMEPLAY:
            category = gm.PaintCategory.GAMEPLAY
        else:
            category = rule.paint.category
        return gm.Paint(category=category, level=2, subtype=rule.paint.subtype, is_treasure=False)

    def _with_supply(self, state: gm.PuzzleState, index: int) -> tuple[gm.PuzzleState, int]:
        """把虚拟背包下标落成真实的一枚颜料。"""
        if 0 <= index < len(state.backpack):
            return state, index
        rule = state.current_rule
        if rule is None:
            return state, index
        return replace(state, backpack=state.backpack + (self._supply_paint(rule),)), len(state.backpack)

    # ---- 合法动作 ----------------------------------------------------
    def legal_actions(self, state: gm.PuzzleState) -> tuple[Action, ...]:
        if state.current_rule is None:
            return ()
        if gm.current_rule_is_pending(state):
            cells = [
                (row, col)
                for row in range(gm.BOARD_SIZE)
                for col in range(gm.BOARD_SIZE)
                if gm._is_current_anchor_placeable(state.tile(row, col), state.active_affix)
            ]
            return tuple(Action(ActionKind.PLACE_ANCHOR, row, col) for row, col in cells)

        rule = state.current_rule
        scope = gm._placement_scope_cells(state, rule)
        placeable = [
            (row, col) for row, col in scope if gm._is_placeable(state.tile(row, col), state.active_affix)
        ]
        indices = [
            index
            for index, paint in enumerate(state.backpack)
            if gm._paint_matches_target(paint, rule.target_object)
        ]
        if not indices and self.assume_supply:
            # 虚拟下标：apply 时按 supply 画像补一枚真实颜料
            indices = [len(state.backpack)]
        actions: list[Action] = []
        for index in indices:
            for row, col in placeable:
                actions.append(Action(ActionKind.INSTALL, row, col, index))
        # 官方：必须先放入颜料才能强制调色；盘点里还没有背包颜料时这一步是非法的，
        # 否则它会是一个"点了没反应"的动作，搜索会把空操作当成一条路。
        if gm.backpack_paint_placed(state):
            actions.append(Action(ActionKind.FORCE_COMPLETE))
        return tuple(actions)

    # ---- 状态转移 ----------------------------------------------------
    def apply(
        self,
        state: gm.PuzzleState,
        action: Action,
        rng: random.Random,
        *,
        force_used: int = 0,
    ) -> gm.PuzzleState:
        if action.kind is ActionKind.PLACE_ANCHOR:
            result = gm.place_current_rule_anchor(state, action.row, action.col, rng=rng)
        elif action.kind is ActionKind.INSTALL:
            state, index = self._with_supply(state, action.backpack_index)
            result = gm.install_backpack_paint(
                state, index, action.row, action.col, rng=rng
            )
        else:
            result = gm.force_complete_current_rule(state, force_used, rng=rng)
        return result.state if result.success else state

    # ---- 终局判断 ----------------------------------------------------
    @staticmethod
    def rule_done(state: gm.PuzzleState, start_rule_id: int) -> bool:
        """当前这条规则是否已经结算（进入下一条规则 / 无规则）。"""
        return state.current_rule is None or state.current_buff_id != start_rule_id

    def is_terminal(self, state: gm.PuzzleState, *, steps: int, start_rule_id: int) -> bool:
        if steps >= self.max_steps:
            return True
        if self.rule_done(state, start_rule_id):
            return True
        return not self.legal_actions(state)

    # ---- 便利方法 ----------------------------------------------------
    @staticmethod
    def backpack_index_of(state: gm.PuzzleState, paint: gm.Paint) -> int | None:
        for index, candidate in enumerate(state.backpack):
            if candidate == paint:
                return index
        return None

    def can_complete(self, state: gm.PuzzleState) -> bool:
        """当前规则还能不能靠"安装"推进（否则只剩强制完成）。"""
        rule = state.current_rule
        if rule is None:
            return False
        if gm.current_rule_is_pending(state):
            return True
        placeable = any(
            gm._is_placeable(state.tile(row, col), state.active_affix)
            for row, col in gm._placement_scope_cells(state, rule)
        )
        if not placeable:
            return False
        if self.assume_supply:
            return True
        return any(gm._paint_matches_target(paint, rule.target_object) for paint in state.backpack)
