"""推演用的落子策略。

搜索只在**根节点**做真正的比较；根节点之后的推演交给一个便宜的策略走完，
这一点和 TartarusSolver 的 ``HeuristicPolicy`` 一致。
"""

from __future__ import annotations

import random
from typing import Protocol

from .. import game_model as gm
from .actions import Action, ActionKind
from .rules import RuleEngine
from .value import DEFAULT_WEIGHTS, EvalWeights, position_value


class Policy(Protocol):
    def choose(
        self,
        state: gm.PuzzleState,
        engine: RuleEngine,
        rng: random.Random,
        *,
        force_used: int = 0,
    ) -> Action | None: ...


class HeuristicPolicy:
    """直接用项目现有的启发式推荐当推演策略。

    好处是搜索建立在"现在的脑子"之上：同样的局面，搜索只是把根节点的
    第一手换成"推演后更值钱"的那一手，后面的走法仍然是熟悉的逻辑。
    """

    def __init__(self, *, learning_stats: dict | None = None, include_lookahead: bool = False) -> None:
        self.learning_stats = learning_stats
        self.include_lookahead = include_lookahead

    def choose(
        self,
        state: gm.PuzzleState,
        engine: RuleEngine,
        rng: random.Random,
        *,
        force_used: int = 0,
    ) -> Action | None:
        if state.current_rule is None:
            return None
        pending = gm.current_rule_is_pending(state)
        recommendations = gm.recommend_placements(
            state,
            limit=1,
            include_lookahead=self.include_lookahead,
            learning_stats=self.learning_stats,
        )
        if not recommendations:
            if not pending:
                return Action(ActionKind.FORCE_COMPLETE)
            return None
        best = recommendations[0]
        if pending:
            return Action(ActionKind.PLACE_ANCHOR, best.row, best.col)
        index = engine.backpack_index_of(state, best.paint)
        if index is None:
            return Action(ActionKind.FORCE_COMPLETE)
        return Action(ActionKind.INSTALL, best.row, best.col, index)


class GreedyValuePolicy:
    """一步贪心：只按终局价值函数挑当前最划算的一手。很便宜，用于大批量推演。"""

    def __init__(self, weights: EvalWeights = DEFAULT_WEIGHTS) -> None:
        self.weights = weights

    def choose(
        self,
        state: gm.PuzzleState,
        engine: RuleEngine,
        rng: random.Random,
        *,
        force_used: int = 0,
    ) -> Action | None:
        actions = engine.legal_actions(state)
        if not actions:
            return None
        installs = [action for action in actions if action.kind is ActionKind.INSTALL]
        if not installs:
            return actions[0]

        # 同一格只保留等级最高的那瓶颜料，把动作数压到"格子数"量级。
        best_per_cell: dict[tuple[int, int], int] = {}
        for action in installs:
            key = (action.row, action.col)
            current = best_per_cell.get(key)
            if current is None or state.backpack[action.backpack_index].level > state.backpack[current].level:
                best_per_cell[key] = action.backpack_index

        probe = random.Random(0)
        best_action: Action | None = None
        best_value = float("-inf")
        for (row, col), index in best_per_cell.items():
            result = gm.install_backpack_paint(state, index, row, col, rng=probe)
            if not result.success:
                continue
            value = position_value(result.state, self.weights)
            if value > best_value:
                best_value = value
                best_action = Action(ActionKind.INSTALL, row, col, index)
        return best_action or actions[0]
