"""配对批量评估：同一批局面，让两套策略各打一遍，看谁留下的棋盘更值钱。

这是 TartarusSolver 里 ``tartarus-sim compare`` 的对应物，也是本项目
"批量模拟评估 / A-B 对比"一直缺的那块。没有它，任何权重改动都只是感觉。
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass

from .. import game_model as gm
from .policy import HeuristicPolicy, Policy
from .rules import RuleEngine
from .search import DeterminizedSearch, SearchPolicy
from .value import DEFAULT_WEIGHTS, EvalWeights, board_value, level7_counts


@dataclass(frozen=True)
class EpisodeResult:
    value: float
    steps: int
    finished: bool
    treasure7: int
    any7: int


@dataclass(frozen=True)
class CompareReport:
    episodes: int
    a_label: str
    b_label: str
    a_mean: float
    b_mean: float
    a_median: float
    b_median: float
    wins: int
    ties: int
    losses: int
    a_treasure7: float
    b_treasure7: float
    a_seconds: float
    b_seconds: float

    def format(self) -> str:
        total = max(1, self.wins + self.ties + self.losses)
        return "\n".join(
            [
                f"局面数            {self.episodes}",
                f"A = {self.a_label}",
                f"B = {self.b_label}",
                f"A 平均终局价值     {self.a_mean:.4f}（中位数 {self.a_median:.4f}）",
                f"B 平均终局价值     {self.b_mean:.4f}（中位数 {self.b_median:.4f}）",
                f"B 胜 / 平 / 负     {self.wins} / {self.ties} / {self.losses}"
                f"（胜率 {self.wins / total:.1%}）",
                f"A 平均 7 级宝箱    {self.a_treasure7:.3f}",
                f"B 平均 7 级宝箱    {self.b_treasure7:.3f}",
                f"耗时 A / B        {self.a_seconds:.2f}s / {self.b_seconds:.2f}s",
            ]
        )


def play_episode(
    state: gm.PuzzleState,
    policy: Policy,
    engine: RuleEngine,
    *,
    seed: int = 0,
    rules: int = 1,
    max_steps: int | None = None,
    force_used: int = 0,
    weights: EvalWeights = DEFAULT_WEIGHTS,
) -> EpisodeResult:
    """用一套策略从给定局面往下打 ``rules`` 条规则，返回终局价值。"""
    rng = random.Random(seed)
    start_rule = state.current_buff_id
    steps = 0
    limit = max_steps if max_steps is not None else engine.max_steps
    while steps < limit:
        if state.current_rule is None:
            break
        if state.current_buff_id >= start_rule + rules:
            break
        if not engine.can_complete(state):
            break
        action = policy.choose(state, engine, rng, force_used=force_used)
        if action is None:
            break
        after = engine.apply(state, action, rng, force_used=force_used)
        if after is state:
            break
        state = after
        steps += 1
    finished = state.current_rule is None or state.current_buff_id > start_rule
    treasure7, any7 = level7_counts(state)
    return EpisodeResult(
        value=board_value(state, weights),
        steps=steps,
        finished=finished,
        treasure7=treasure7,
        any7=any7,
    )


def compare(
    *,
    episodes: int = 20,
    base_seed: int = 1000,
    a_policy: Policy | None = None,
    b_policy: Policy | None = None,
    a_label: str = "现有启发式",
    b_label: str = "搜索",
    rules: int = 1,
    weights: EvalWeights = DEFAULT_WEIGHTS,
    engine: RuleEngine | None = None,
) -> CompareReport:
    engine = engine or RuleEngine()
    a_policy = a_policy or HeuristicPolicy(include_lookahead=True)
    if b_policy is None:
        search = DeterminizedSearch(engine, weights=weights, policy=HeuristicPolicy())
        b_policy = SearchPolicy(search, playouts=6, top_k=6)

    a_values: list[float] = []
    b_values: list[float] = []
    a_seconds = b_seconds = 0.0
    wins = ties = losses = 0
    a_treasure = b_treasure = 0

    for index in range(episodes):
        start = gm.random_state(base_seed + index)
        started = time.perf_counter()
        a_result = play_episode(start, a_policy, engine, seed=base_seed + index, rules=rules, weights=weights)
        a_seconds += time.perf_counter() - started

        started = time.perf_counter()
        b_result = play_episode(start, b_policy, engine, seed=base_seed + index, rules=rules, weights=weights)
        b_seconds += time.perf_counter() - started

        a_values.append(a_result.value)
        b_values.append(b_result.value)
        a_treasure += a_result.treasure7
        b_treasure += b_result.treasure7
        if b_result.value > a_result.value + 1e-9:
            wins += 1
        elif b_result.value < a_result.value - 1e-9:
            losses += 1
        else:
            ties += 1

    return CompareReport(
        episodes=episodes,
        a_label=a_label,
        b_label=b_label,
        a_mean=statistics.fmean(a_values) if a_values else 0.0,
        b_mean=statistics.fmean(b_values) if b_values else 0.0,
        a_median=statistics.median(a_values) if a_values else 0.0,
        b_median=statistics.median(b_values) if b_values else 0.0,
        wins=wins,
        ties=ties,
        losses=losses,
        a_treasure7=a_treasure / max(1, episodes),
        b_treasure7=b_treasure / max(1, episodes),
        a_seconds=a_seconds,
        b_seconds=b_seconds,
    )
