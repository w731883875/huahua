"""确定性蒙特卡洛搜索（determinized Monte Carlo）。

和 TartarusSolver 的 ``MonteCarloSearch`` 同一套思路：

* 只比较**根节点**的合法动作，根节点之后交给便宜策略推演到底；
* 每个"世界"用同一个随机种子跑完所有候选动作（common random numbers），
  这样动作之间的差异来自动作本身，而不是运气；
* 落后太多的动作在后续轮次不再分配推演次数（successive rejection）；
* 可以给时间预算，随时收手。

和它的差别：本项目的隐藏信息主要是"下一条规则长什么样"和"强制完成的炸毁格"，
所以推演默认跨过规则边界（``horizon_rules``），否则同一条规则内的推演几乎是确定性的。
"""

from __future__ import annotations

import math
import random
import time

from .. import game_model as gm
from .actions import Action, ActionValue, SearchResult
from .policy import HeuristicPolicy, Policy
from .rules import RuleEngine
from .value import DEFAULT_WEIGHTS, EvalWeights, position_value, terminal_value

_MASK64 = (1 << 64) - 1


def _mix(seed: int, index: int) -> int:
    value = (seed * 0x9E3779B97F4A7C15 + index * 0xBF58476D1CE4E5B9) & _MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK64
    value ^= value >> 27
    return value & 0x7FFFFFFF


class DeterminizedSearch:
    def __init__(
        self,
        engine: RuleEngine | None = None,
        *,
        weights: EvalWeights = DEFAULT_WEIGHTS,
        policy: Policy | None = None,
        horizon_rules: int = 1,
    ) -> None:
        self.engine = engine or RuleEngine()
        self.weights = weights
        self.policy = policy or HeuristicPolicy()
        self.horizon_rules = max(1, int(horizon_rules))

    # ---- 对外入口 ----------------------------------------------------
    def search(
        self,
        state: gm.PuzzleState,
        *,
        playouts: int = 6,
        top_k: int = 6,
        time_budget: float | None = None,
        seed: int = 1,
        force_used: int = 0,
    ) -> SearchResult:
        started = time.perf_counter()
        legal = self.engine.legal_actions(state)
        if not legal:
            return SearchResult(actions=(), playouts=0, elapsed=0.0)

        candidates = self._prune(state, legal, max(1, top_k), force_used)
        if len(candidates) == 1:
            return SearchResult(
                actions=(ActionValue(candidates[0], 0.0, 0.0, 0),),
                playouts=0,
                elapsed=time.perf_counter() - started,
            )

        start_rule_id = state.current_buff_id
        count = len(candidates)
        totals = [0.0] * count
        squares = [0.0] * count
        counts = [0] * count
        alive = [True] * count

        round_size = max(1, min(4, playouts))
        rounds = max(1, math.ceil(playouts / round_size))
        for round_index in range(rounds):
            for sample in range(round_size):
                sample_seed = _mix(seed, round_index * round_size + sample)
                for index in range(count):
                    if not alive[index]:
                        continue
                    rng = random.Random(sample_seed)
                    after = self.engine.apply(state, candidates[index], rng, force_used=force_used)
                    value = self._rollout(after, rng, start_rule_id, force_used)
                    totals[index] += value
                    squares[index] += value * value
                    counts[index] += 1
            if round_index >= 1:
                self._reject_laggards(candidates, totals, squares, counts, alive)
            if all(counts[i] >= playouts for i in range(count) if alive[i]):
                break
            if time_budget is not None and time.perf_counter() - started >= time_budget:
                break

        values = [
            ActionValue(
                action=candidates[i],
                mean=totals[i] / counts[i] if counts[i] else float("-inf"),
                stderr=self._stderr(totals[i], squares[i], counts[i]),
                playouts=counts[i],
            )
            for i in range(count)
            if counts[i] > 0
        ]
        values.sort(key=lambda item: item.mean, reverse=True)
        return SearchResult(
            actions=tuple(values),
            playouts=sum(counts),
            elapsed=time.perf_counter() - started,
        )

    def best_action(
        self,
        state: gm.PuzzleState,
        *,
        playouts: int = 6,
        top_k: int = 6,
        time_budget: float | None = None,
        seed: int = 1,
        force_used: int = 0,
    ) -> Action | None:
        result = self.search(
            state,
            playouts=playouts,
            top_k=top_k,
            time_budget=time_budget,
            seed=seed,
            force_used=force_used,
        )
        best = result.best
        return best.action if best is not None else None

    # ---- 内部 --------------------------------------------------------
    def _prune(
        self,
        state: gm.PuzzleState,
        legal: tuple[Action, ...],
        top_k: int,
        force_used: int,
    ) -> list[Action]:
        if len(legal) <= top_k:
            return list(legal)
        probe = random.Random(0)
        scored: list[tuple[float, Action]] = []
        for action in legal:
            after = self.engine.apply(state, action, probe, force_used=force_used)
            scored.append((position_value(after, self.weights), action))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [action for _, action in scored[:top_k]]

    def _rollout(
        self,
        state: gm.PuzzleState,
        rng: random.Random,
        start_rule_id: int,
        force_used: int,
    ) -> float:
        steps = 0
        while steps < self.engine.max_steps:
            if state.current_rule is None:
                break
            if state.current_buff_id >= start_rule_id + self.horizon_rules:
                break
            if not self.engine.can_complete(state):
                break
            action = self.policy.choose(state, self.engine, rng, force_used=force_used)
            if action is None:
                break
            after = self.engine.apply(state, action, rng, force_used=force_used)
            if after is state:
                break
            state = after
            steps += 1
        finished = state.current_rule is None or state.current_buff_id > start_rule_id
        return terminal_value(state, finished=finished, weights=self.weights)

    @staticmethod
    def _stderr(total: float, square: float, count: int) -> float:
        if count < 2:
            return float("inf")
        mean = total / count
        variance = max(0.0, square / count - mean * mean)
        return math.sqrt(variance / count)

    @staticmethod
    def _reject_laggards(
        candidates: list[Action],
        totals: list[float],
        squares: list[float],
        counts: list[int],
        alive: list[bool],
    ) -> None:
        live = [i for i in range(len(candidates)) if alive[i] and counts[i] > 0]
        if len(live) < 2:
            return
        means = {i: totals[i] / counts[i] for i in live}
        leader = max(live, key=lambda i: means[i])
        lead_mean = means[leader]
        lead_err = DeterminizedSearch._stderr(totals[leader], squares[leader], counts[leader])
        for index in live:
            if index == leader:
                continue
            mean = means[index]
            err = DeterminizedSearch._stderr(totals[index], squares[index], counts[index])
            if mean + 4 * err < lead_mean - 4 * lead_err:
                alive[index] = False


class SearchPolicy:
    """每个决策点都跑一次搜索的策略。用于"搜索到底有多强"的测量。"""

    def __init__(
        self,
        search: DeterminizedSearch,
        *,
        playouts: int = 6,
        top_k: int = 6,
        time_budget: float | None = None,
    ) -> None:
        self.search = search
        self.playouts = playouts
        self.top_k = top_k
        self.time_budget = time_budget
        self._seed = 1
        self.last_result: SearchResult | None = None

    def choose(
        self,
        state: gm.PuzzleState,
        engine: RuleEngine,
        rng: random.Random,
        *,
        force_used: int = 0,
    ) -> Action | None:
        self.last_result = self.search.search(
            state,
            playouts=self.playouts,
            top_k=self.top_k,
            time_budget=self.time_budget,
            seed=self._seed,
            force_used=force_used,
        )
        self._seed += 1
        best = self.last_result.best
        return best.action if best is not None else None
