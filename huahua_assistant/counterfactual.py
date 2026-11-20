#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反事实复盘：把同一局的词缀选择换成别的，看能不能拿到更高分。

用途很直接——**验证推荐器是不是真的选对了**。
一局打出的分数只是"推荐器选了这条路"的结果；如果换成另一条词缀能达到更高分，
那就说明推荐器选错了，而不是这条路只能打到这个分。

做法：读一份复盘快照，把其中所有 ``choose_affix`` 决策当成一个分支点，
穷举每条分支的**所有可选词缀**组合，各自重放到底，比较终局价值。

注意边界：

* 只重放**同一批走法**，不改落子位置——落子由策略决定，属于另一个问题；
* 词缀决策数通常只有 1~2 个，所以穷举不会爆；
* 若中途某条分支执行失败（例如刷新次数不够），该分支标记为不可行，不参与比较。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from . import practice as practice_mod


@dataclass(frozen=True)
class AffixBranch:
    """一条完整的词缀选择分支及其结果。"""

    choices: tuple[str, ...]
    value: float
    feasible: bool
    note: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {
            'choices': list(self.choices),
            'value': round(self.value, 6),
            'feasible': self.feasible,
            'note': self.note,
        }


@dataclass(frozen=True)
class CounterfactualResult:
    """一局的反事实分析结果。"""

    recorded_value: float
    recorded_choices: tuple[str, ...]
    best: AffixBranch | None
    branches: tuple[AffixBranch, ...]

    @property
    def gain(self) -> float:
        """最优分支比实际记录高多少（负值表示原选择已是最优）。"""
        if self.best is None:
            return 0.0
        return self.best.value - self.recorded_value

    @property
    def improved(self) -> bool:
        return self.gain > 1e-6

    def as_dict(self) -> dict[str, Any]:
        return {
            'recorded_value': round(self.recorded_value, 6),
            'recorded_choices': list(self.recorded_choices),
            'best': self.best.as_dict() if self.best else None,
            'gain': round(self.gain, 6),
            'improved': self.improved,
            'branch_count': len(self.branches),
            'branches': [b.as_dict() for b in self.branches],
        }


def _affix_decision_points(snapshot: dict[str, Any]) -> list[int]:
    """找出快照里所有「选词缀」走法的下标。"""
    steps = snapshot.get('steps') or ()
    return [index for index, step in enumerate(steps) if step.get('kind') == 'choose_affix']


def _replay_with_choices(
    snapshot: dict[str, Any],
    decisions: list[int],
    choices: tuple[str, ...],
) -> tuple[float, bool, str]:
    """按给定的词缀选择重放，返回 ``(终局价值, 是否可行, 备注)``。"""
    session = practice_mod.restore_practice(snapshot)
    steps = list(snapshot.get('steps') or ())
    wanted = dict(zip(decisions, choices))
    for index, step in enumerate(steps):
        kind = step.get('kind')
        if index in wanted:
            name = wanted[index]
            options = session.affix_options
            target = next((i for i, affix in enumerate(options) if affix.name == name), None)
            if target is None:
                return 0.0, False, f'第 {index} 步词缀「{name}」不在可选项里'
            if not session.choose_affix(target):
                return 0.0, False, f'第 {index} 步选择「{name}」执行失败'
            continue
        if not practice_mod.apply_step(session, step):
            return 0.0, False, f'第 {index} 步（{kind}）执行失败'
    return float(session.value), True, ''


def _options_at(snapshot: dict[str, Any], decisions: list[int], index: int) -> tuple[str, ...]:
    """重放到第 ``index`` 个决策点，返回那一刻的可选词缀名。"""
    steps = list(snapshot.get('steps') or ())
    session = practice_mod.restore_practice(snapshot)
    for position, step in enumerate(steps):
        if position == index:
            return tuple(affix.name for affix in session.affix_options)
        if not practice_mod.apply_step(session, step):
            return ()
    return ()


def analyse(
    snapshot: dict[str, Any],
    *,
    max_branches: int = 400,
) -> CounterfactualResult:
    """穷举一局里所有词缀组合，返回最优分支。"""
    decisions = _affix_decision_points(snapshot)
    recorded = tuple(
        str((snapshot.get('steps') or ())[index].get('affix_id') or '') for index in decisions
    )
    if not decisions:
        return CounterfactualResult(0.0, (), None, ())

    # 每个决策点的可选项：只展开一次，避免重复重放
    option_lists: list[tuple[str, ...]] = []
    for index in decisions:
        options = _options_at(snapshot, decisions, index)
        option_lists.append(options if options else (recorded[decisions.index(index)],))

    total = 1
    for options in option_lists:
        total *= max(1, len(options))
    if total > max_branches:
        return CounterfactualResult(0.0, recorded, None, ())

    branches: list[AffixBranch] = []
    for combo in product(*option_lists):
        value, feasible, note = _replay_with_choices(snapshot, decisions, combo)
        branches.append(AffixBranch(tuple(combo), value, feasible, note))

    # **基准必须自己重算**，不能读快照里的 ``final_board_value``：
    # 那个数是入库那一刻的刻度，一旦估值表改过（本次就改了线性分值），
    # 拿旧刻度的数去比新刻度的分支，会得出"100% 都能突破"这种假结论。
    recorded_value = 0.0
    for branch in branches:
        if branch.feasible and branch.choices == recorded:
            recorded_value = branch.value
            break

    usable = [b for b in branches if b.feasible]
    best = max(usable, key=lambda b: b.value) if usable else None
    return CounterfactualResult(
        recorded_value, recorded, best, tuple(branches)
    )


def format_result(result: CounterfactualResult, *, limit: int = 6) -> str:
    """把反事实结果渲染成人看的几行。"""
    if not result.recorded_choices:
        return '这局没有词缀决策，无需反事实分析。'
    lines = [
        f'实际选择：{" / ".join(result.recorded_choices)}  →  {result.recorded_value:.4f} 分'
    ]
    if result.best is None:
        lines.append('  （没有可行分支）')
        return '\n'.join(lines)
    if result.improved:
        lines.append(
            f'  可突破：{" / ".join(result.best.choices)}  →  {result.best.value:.4f} 分'
            f'（+{result.gain:.4f}）'
        )
    else:
        lines.append('  已是最优：换任何词缀都不会更高')
    ranked = sorted(
        (b for b in result.branches if b.feasible), key=lambda b: -b.value
    )[:limit]
    if len(ranked) > 1:
        lines.append('  分支排行：')
        for branch in ranked:
            mark = '★' if branch is result.best else ' '
            lines.append(
                f'    {mark} {branch.value:.4f}  {" / ".join(branch.choices)}'
            )
    return '\n'.join(lines)
