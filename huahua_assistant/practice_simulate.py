#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按官方几率跑批量模拟（任务 C）。

这里只做两件事：

1. **跑**：批量开局 + 策略自动下完，收集观测数据；
2. **对账**：把观测到的颜料等级分布、宝箱出现率、技巧命中率
   与 pak 里读出来的官方数字并排放，偏差一眼能看见。

为什么要对账：客户端**没有**颜料出现几率表（见 ``s8_pak数据核实报告.md`` §3），
工程里凡是"几率"都必然是某个代理口径。对账的意义就是把这个代理的偏差量化出来，
而不是继续凭感觉调参。目前已知的三处非官方假设会被显式列在
:func:`assumptions` 里。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import game_model as gm
from . import practice as pr
from . import s8_official_data as official
from .practice_library import PracticeLibrary, format_library_report


# ---------------------------------------------------------------------------
# 已知的非官方假设（对账时要摆出来，不能当成官方数据）
# ---------------------------------------------------------------------------

def assumptions() -> list[dict[str, str]]:
    """列出当前生成器里**没有官方依据**的假设，以及它的实际取值。"""
    return [
        {
            'item': '背包等级分布',
            'value': '直接用官方 4005 权重（1..7 全档）',
            'why': '这是**官方数据**，但对账时仍要记得：4005 的官方语义是'
                   '「转锋随机改等级」，不是「拾彩掉落等级」。拾彩等级由服务端下发，'
                   '客户端无表（见 s8_pak数据核实报告.md §3）。',
        },
        {
            'item': '调色目标颜料的等级下界',
            'value': f'min={_paint_minimum()}（函数 _random_paint/_random_target_paint）',
            'why': '官方没有任何"目标颜料不会低于 N 级"的规定，这是工程里为了让'
                   '调色目标不至于过低而设的下界，属自设参数。',
        },
        {
            'item': '怪物/玩法 二分',
            'value': '50 / 50',
            'why': '官方 plotitem 表只有家族与类别，没有两者的出现比例。',
        },
        {
            'item': '宝箱出现率',
            'value': '按玩法家族等概率（回退分支才是 2%）',
            'why': '官方 2009/2010/2011 无权重表；工程里的 treasure_chance 是自设参数。',
        },
        {
            'item': '颜料家族等概率',
            'value': '在同一等级真实存在的家族里等概率',
            'why': '官方无家族权重字段（plotitem 表连 Weight 列都没有）。',
        },
    ]


def _paint_minimum() -> int:
    """探测 ``_random_paint`` 实际用的等级下界（读不到就报未知）。"""
    import inspect

    try:
        source = inspect.getsource(gm._random_paint)  # noqa: SLF001
    except (OSError, TypeError):
        return -1
    for line in source.splitlines():
        if 'minimum=' in line:
            try:
                return int(line.split('minimum=')[1].split(',')[0].split(')')[0])
            except (IndexError, ValueError):
                continue
    return -1


# ---------------------------------------------------------------------------
# 模拟
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """一次批量模拟的全部观测数据。"""

    runs: int
    pet_level: int
    rules: int | None
    prefer: str
    scores: list[float] = field(default_factory=list)
    grades: dict[str, int] = field(default_factory=dict)
    start_levels: dict[int, int] = field(default_factory=dict)
    final_levels: dict[int, int] = field(default_factory=dict)
    treasure_backpack_total: int = 0
    treasure_board_runs: int = 0
    tactic_hits: dict[str, int] = field(default_factory=dict)
    recorded: dict[str, Any] | None = None

    # ---- 派生指标 ----------------------------------------------------
    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def best_score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def high_score_runs(self) -> int:
        return sum(1 for score in self.scores if score >= 0.9)

    def observed_start_distribution(self) -> dict[int, float]:
        total = sum(self.start_levels.values())
        if not total:
            return {}
        return {level: count / total for level, count in sorted(self.start_levels.items())}

    def as_dict(self) -> dict[str, Any]:
        return {
            'runs': self.runs,
            'pet_level': self.pet_level,
            'rules': self.rules,
            'prefer': self.prefer,
            'mean_score': round(self.mean_score, 6),
            'best_score': round(self.best_score, 6),
            'high_score_runs': self.high_score_runs,
            'grades': dict(sorted(self.grades.items())),
            'start_levels': dict(sorted(self.start_levels.items())),
            'final_levels': dict(sorted(self.final_levels.items())),
            'treasure_backpack_total': self.treasure_backpack_total,
            'treasure_board_runs': self.treasure_board_runs,
            'tactic_hits': dict(sorted(self.tactic_hits.items())),
            'recorded': self.recorded,
        }


def simulate(
    *,
    runs: int = 30,
    base_seed: int = 1,
    pet_level: int = gm.DEFAULT_PET_LEVEL,
    rules: int | None = 2,
    prefer: str = 'greedy',
    record: bool = False,
    library_path: str | Path | None = None,
) -> SimulationResult:
    """批量跑局并收集观测数据。``record=True`` 时同时写进高分库。"""
    result = SimulationResult(
        runs=0, pet_level=int(pet_level), rules=rules, prefer=prefer
    )
    library = PracticeLibrary(library_path) if record else None
    grades: collections.Counter[str] = collections.Counter()
    start_levels: collections.Counter[int] = collections.Counter()
    final_levels: collections.Counter[int] = collections.Counter()
    tactic_hits: collections.Counter[str] = collections.Counter()

    for index in range(max(1, int(runs))):
        session = pr.start_practice(
            int(base_seed) + index,
            pet_level=pet_level,
            rules=rules,
            wait_for_anchor=True,
        )
        # 开局背包 = 官方等级分布的直接体现，先记下来再下棋。
        for paint in session.state.backpack:
            start_levels[int(paint.level)] += 1
            if paint.is_treasure:
                result.treasure_backpack_total += 1

        pr.auto_play(session, prefer=prefer)

        for tile in session.state.tiles:
            if tile.state is gm.TileState.PAINT and tile.paint is not None:
                final_levels[gm.tile_final_level(tile)] += 1

        explanation = pr.explain_high_score(
            session.state,
            steps=session.steps,
            session=session,
            weights=session.weights,
        )
        result.scores.append(float(explanation.score))
        grades[explanation.grade] += 1
        for tactic in explanation.tactics:
            tactic_hits[tactic.key] += 1
        if any(
            tile.paint is not None and tile.paint.is_treasure
            for tile in session.state.tiles
            if tile.state is gm.TileState.PAINT
        ):
            result.treasure_board_runs += 1

        if library is not None:
            library.record(
                explanation,
                session.snapshot(),
                source='simulation',
                note=f'prefer={prefer} pet={pet_level} rules={rules}',
            )
        result.runs += 1

    result.grades = dict(grades)
    result.start_levels = dict(start_levels)
    result.final_levels = dict(final_levels)
    result.tactic_hits = dict(tactic_hits)
    if library is not None:
        result.recorded = library.stats()
    return result


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def _bar(value: float, scale: float = 1.0, width: int = 24) -> str:
    """ASCII 柱状图：用 # 和 . 而不是方块字符，避免控制台代码页把整行糊掉。"""
    filled = int(round(min(1.0, max(0.0, value * scale)) * width))
    return '#' * filled + '.' * (width - filled)


def format_report(result: SimulationResult) -> str:
    """人看的报告：先结论，再对账，最后是差异清单。"""
    lines: list[str] = []
    lines.append(f'=== 模拟 {result.runs} 局  (宠物 Lv{result.pet_level}, '
                 f'结算 {result.rules} 条词缀, 策略 {result.prefer}) ===')
    lines.append(
        f'平均分 {result.mean_score:.4f}   最高 {result.best_score:.4f}   '
        f'高分(≥0.9) {result.high_score_runs} 局 '
        f'({result.high_score_runs / max(1, result.runs):.1%})'
    )
    if result.grades:
        grades = '  '.join(f'{name} {count}' for name, count in sorted(result.grades.items()))
        lines.append(f'等级分布：{grades}')

    # ---- 等级分布对账 ----
    expected = official.paint_level_distribution()
    observed = result.observed_start_distribution()
    if expected and observed:
        lines.append('')
        lines.append('颜料等级分布（开局背包）对账：')
        lines.append(f'  {"等级":<4}{"官方":>8}{"模拟":>8}   官方 / 模拟')
        for level in range(1, gm.MAX_LEVEL + 1):
            want = expected.get(level, 0.0)
            got = observed.get(level, 0.0)
            lines.append(
                f'  Lv{level:<2}{want:>8.2%}{got:>8.2%}   '
                f'{_bar(want)} | {_bar(got)}'
            )
        mean_want = sum(level * p for level, p in expected.items())
        mean_got = sum(level * p for level, p in observed.items())
        lines.append(f'  期望等级：官方 {mean_want:.3f} vs 模拟 {mean_got:.3f}'
                     f'（差 {mean_got - mean_want:+.3f}）')

    # ---- 宝箱 ----
    if result.runs:
        lines.append('')
        lines.append(
            f'宝箱：开局背包共 {result.treasure_backpack_total} 枚，'
            f'终局棋盘上出现宝箱的对局 {result.treasure_board_runs}/{result.runs}'
            f'（{result.treasure_board_runs / result.runs:.1%}）'
        )

    # ---- 技巧 ----
    if result.tactic_hits:
        lines.append('')
        lines.append('技巧命中：')
        for key, hits in sorted(result.tactic_hits.items(), key=lambda kv: -kv[1]):
            title = pr.TACTIC_LIBRARY.get(key, (key, '', ''))[0]
            lines.append(f'  {hits:4d} 次  {title}')

    # ---- 假设清单 ----
    lines.append('')
    lines.append('非官方假设（对账时必须记住这些不是官方数据）：')
    for item in assumptions():
        lines.append(f'  · {item["item"]} = {item["value"]}')
        lines.append(f'      {item["why"]}')

    if result.recorded:
        lines.append('')
        lines.append(format_library_report(result.recorded))
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    # 控制台默认代码页是 GBK，中文报告会糊成乱码；这里强制按 UTF-8 写。
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description='按官方几率跑批量模拟并对账')
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--base-seed', type=int, default=1)
    parser.add_argument('--pet', type=int, default=gm.DEFAULT_PET_LEVEL)
    parser.add_argument('--rules', type=int, default=2)
    parser.add_argument('--prefer', choices=('greedy', 'lookahead'), default='greedy')
    parser.add_argument('--record', action='store_true', help='同时写进高分库')
    parser.add_argument('--library', default=None)
    parser.add_argument('--json', default=None, help='把结果另存为 JSON')
    args = parser.parse_args(argv)

    result = simulate(
        runs=args.runs,
        base_seed=args.base_seed,
        pet_level=args.pet,
        rules=args.rules,
        prefer=args.prefer,
        record=args.record,
        library_path=args.library,
    )
    print(format_report(result))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=1), encoding='utf-8'
        )
        print(f'\n已写出 {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
