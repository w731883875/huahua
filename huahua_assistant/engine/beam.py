"""确定性束搜索（beam search）。

为什么不是蒙特卡洛：本项目一条规则内的信息是**完全可见**的 ——
初始颜料、作用范围、目标、背包全部已知，唯一的随机只有"下一条规则"
和"强制完成炸哪几格"。在这种结构下，撒随机样本推演等于把确定的事情
重复算很多遍，得到的还是同一个数。

所以这里用确定性的束搜索：每一层保留价值最高的 W 个局面，
一直推到这条规则结算。它比"一步贪心"多看两层，又比全枚举便宜得多。
"""

from __future__ import annotations

import random

from .. import game_model as gm
from .actions import Action, ActionKind, ActionValue, SearchResult
from .policy import HeuristicPolicy, Policy
from .rules import RuleEngine
from .value import DEFAULT_WEIGHTS, EvalWeights, position_value, terminal_value


class BeamSearch:
    def __init__(
        self,
        engine: RuleEngine | None = None,
        *,
        weights: EvalWeights = DEFAULT_WEIGHTS,
        policy: Policy | None = None,
    ) -> None:
        self.engine = engine or RuleEngine()
        self.weights = weights
        # 束搜索之外的分支（例如换规则后的落锚）交给这个策略。
        self.policy = policy or HeuristicPolicy()

    def search(
        self,
        state: gm.PuzzleState,
        *,
        width: int = 8,
        max_depth: int | None = None,
        force_used: int = 0,
    ) -> SearchResult:
        started_rng = random.Random(0)
        legal = self.engine.legal_actions(state)
        if not legal:
            return SearchResult(actions=(), playouts=0, elapsed=0.0)

        depth_limit = max_depth if max_depth is not None else self.engine.max_steps
        start_rule = state.current_buff_id
        rng = random.Random(0)
        leaves: dict[Action, float] = {}
        # (价值, 局面, 根动作)：第一层就是根节点的合法动作。
        beam: list[tuple[float, gm.PuzzleState, Action]] = []
        for action in legal:
            after = self.engine.apply(state, action, rng, force_used=force_used)
            if after is state:
                continue
            if self.engine.rule_done(after, start_rule):
                leaves[action] = terminal_value(after, finished=True, weights=self.weights)
            else:
                beam.append((position_value(after, self.weights), after, action))
        beam = self._dedupe(sorted(beam, key=lambda item: item[0], reverse=True))[: max(1, width)]

        for _depth in range(max(0, depth_limit - 1)):
            expanded: list[tuple[float, gm.PuzzleState, Action]] = []
            for _, current, root in beam:
                if not self.engine.can_complete(current):
                    continue
                for action in self._actions(current):
                    after = self.engine.apply(current, action, rng, force_used=force_used)
                    if after is current:
                        continue
                    if self.engine.rule_done(after, start_rule):
                        value = terminal_value(after, finished=True, weights=self.weights)
                        leaves[root] = max(leaves.get(root, float("-inf")), value)
                    else:
                        expanded.append((position_value(after, self.weights), after, root))
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0], reverse=True)
            beam = self._dedupe(expanded)[: max(1, width)]

        if not leaves:
            return SearchResult(actions=(), playouts=0, elapsed=0.0)
        values = tuple(
            ActionValue(action=action, mean=value, stderr=0.0, playouts=1)
            for action, value in sorted(leaves.items(), key=lambda item: item[1], reverse=True)
        )
        return SearchResult(actions=values, playouts=len(values), elapsed=0.0)

    def best_action(
        self,
        state: gm.PuzzleState,
        *,
        width: int = 8,
        max_depth: int | None = None,
        force_used: int = 0,
    ) -> Action | None:
        best = self.search(state, width=width, max_depth=max_depth, force_used=force_used).best
        return best.action if best is not None else None

    # ---- 内部 --------------------------------------------------------
    def _actions(self, state: gm.PuzzleState) -> tuple[Action, ...]:
        actions = self.engine.legal_actions(state)
        installs = tuple(action for action in actions if action.kind is not ActionKind.FORCE_COMPLETE)
        return installs or actions

    def _dedupe(
        self,
        expanded: list[tuple[float, gm.PuzzleState, Action]],
    ) -> list[tuple[float, gm.PuzzleState, Action]]:
        seen: set[tuple[object, ...]] = set()
        unique: list[tuple[float, gm.PuzzleState, Action]] = []
        for item in expanded:
            signature = self._signature(item[1])
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(item)
        return unique

    @staticmethod
    def _signature(state: gm.PuzzleState) -> tuple[object, ...]:
        tiles = tuple(
            (tile.state.value, tile.paint.category.value if tile.paint else "", tile.paint.level if tile.paint else 0)
            for tile in state.tiles
        )
        backpack = tuple(sorted((paint.category.value, paint.level, paint.is_treasure) for paint in state.backpack))
        return (tiles, backpack)


class BeamPolicy:
    """每个决策点跑一次束搜索的策略，用于强度测量。"""

    def __init__(self, search: BeamSearch, *, width: int = 8) -> None:
        self.search = search
        self.width = width
        self.last_result: SearchResult | None = None

    def choose(
        self,
        state: gm.PuzzleState,
        engine: RuleEngine,
        rng: random.Random,
        *,
        force_used: int = 0,
    ) -> Action | None:
        self.last_result = self.search.search(state, width=self.width, force_used=force_used)
        best = self.last_result.best
        return best.action if best is not None else None
