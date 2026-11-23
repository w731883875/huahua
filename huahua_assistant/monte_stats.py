#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按当前规则批量模拟 + 采集统计口径（2026-11-23 用户列的口径）。

用户要求的记录项：

* 当局的**空白地块 / 阻挡 / 虚空**数量；
* **4 格、5 格、6 格空白连线**数量；
* **怪物 / 玩法 / 宝箱**初始颜料几率；
* **调色效果**几率、**笔触效果**几率。

为什么单独一个模块而不是随手脚本：这些口径要能长期对照实机复核，
所以统一放包里、可被测试和 UI 复用，跑法固定、输出带 config 便于复现。

隔离原则：每次运行写到 ``runtime/monte_runs/<时间戳>_<标签>/``，
不覆盖、不混入历史数据。
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import game_model as gm
from . import practice as pr
from . import strategy_lv7 as st

# 一次运行内累计的计数器键（聚合时按这些键合并）
_COUNTER_KEYS = (
    'board_blank', 'board_blocked', 'board_void', 'board_faded', 'board_paint',
    'line_len4', 'line_len5', 'line_len6', 'line_long',
    'anchor_category', 'anchor_treasure', 'anchor_level',
    'tint_effect', 'tint_direction', 'tint_target', 'tint_amount',
    'brush_offered', 'brush_chosen', 'brush_effect',
    'stage1_offered', 'stage1_chosen',
    'force_per_run', 'grade',
)


@dataclass
class RunObservation:
    """单局的原始观测（聚合前的原料）。"""

    seed: int
    score: float
    lv7_treasure: int
    lv7_total: int
    paint_count: int
    steps: int
    force_used: int
    counters: dict[str, collections.Counter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in _COUNTER_KEYS:
            self.counters.setdefault(key, collections.Counter())


def observe_run(
    seed: int,
    *,
    pet_level: int = gm.DEFAULT_PET_LEVEL,
    rules: int | None = None,
    max_steps: int = 60,
    prefer: str = 'greedy',
    fusion_explore: bool = False,
    weights: Any = None,
) -> RunObservation:
    """跑一局并采集用户要求的全部口径。"""
    session = pr.start_practice(
        seed,
        pet_level=pet_level,
        rules=rules,
        max_steps=max_steps,
        wait_for_anchor=True,
    )

    counters: dict[str, collections.Counter] = {
        key: collections.Counter() for key in _COUNTER_KEYS
    }

    # 开局棋盘构成 + 空白连线段长度（用户要的 4/5/6 格连线）
    for tile in session.state.tiles:
        counters['board_' + tile.state.value][0] += 1
    for length in st.line_runs(session.state):
        if length == 4:
            counters['line_len4'][4] += 1
        elif length == 5:
            counters['line_len5'][5] += 1
        elif length >= 6:
            counters['line_len6'][length] += 1
        if length >= 4:
            counters['line_long'][length] += 1

    def _on_event(event: dict) -> None:
        kind = event.get('kind')
        if kind == 'anchor':
            paint = event.get('paint') or {}
            rule = event.get('rule') or {}
            counters['anchor_category'][paint.get('category')] += 1
            counters['anchor_treasure'][bool(paint.get('is_treasure'))] += 1
            counters['anchor_level'][int(paint.get('level') or 0)] += 1
            for effect in rule.get('effects') or []:
                # 调色效果 = 初始颜料自带的那套效果（官方 itemeffect 表）
                key = (
                    effect.get('direction'),
                    effect.get('target_object'),
                    int(effect.get('upgrade_amount') or 0),
                )
                counters['tint_effect']['|'.join(str(part) for part in key)] += 1
                counters['tint_direction'][effect.get('direction')] += 1
                counters['tint_target'][effect.get('target_object')] += 1
                counters['tint_amount'][int(effect.get('upgrade_amount') or 0)] += 1
        elif kind == 'affix':
            # 笔触效果 = 选中的词缀（官方 selectableeffect 表）
            for name in event.get('options') or []:
                counters['brush_offered'][name] += 1
            name = event.get('name') or ''
            if name:
                counters['brush_chosen'][name] += 1
        elif kind == 'force':
            counters['force_per_run'][1] += 1

    scenario = getattr(session, '_affix_scenario', None)
    pr.auto_play(
        session,
        prefer=prefer,
        fusion_explore=fusion_explore,
        observer=_on_event,
    )

    explanation = pr.explain_high_score(
        session.state,
        steps=session.steps,
        session=session,
        weights=weights or session.weights,
    )

    lv7_total = 0
    paint_count = 0
    for tile in session.state.tiles:
        if tile.state is not gm.TileState.PAINT or tile.paint is None:
            continue
        paint_count += 1
        if gm.tile_final_level(tile) >= gm.MAX_LEVEL:
            lv7_total += 1

    counters['grade'][explanation.grade] += 1

    return RunObservation(
        seed=seed,
        score=float(explanation.score),
        lv7_treasure=st.lv7_treasure_count(session.state),
        lv7_total=lv7_total,
        paint_count=paint_count,
        steps=len(session.steps),
        force_used=int(session._force_used),
        counters=counters,
    )


def _run_chunk(args: tuple) -> list[RunObservation]:
    lo, hi, kwargs = args
    results: list[RunObservation] = []
    for seed in range(lo, hi):
        try:
            results.append(observe_run(seed, **kwargs))
        except Exception:  # 单局异常不该拖死整批
            continue
    return results


def _merge(observations: list[RunObservation]) -> dict[str, Any]:
    merged: dict[str, collections.Counter] = {
        key: collections.Counter() for key in _COUNTER_KEYS
    }
    scores: list[float] = []
    lv7_treasure: list[int] = []
    lv7_total: list[int] = []
    paint_counts: list[int] = []
    steps: list[int] = []
    force: list[int] = []

    for item in observations:
        scores.append(item.score)
        lv7_treasure.append(item.lv7_treasure)
        lv7_total.append(item.lv7_total)
        paint_counts.append(item.paint_count)
        steps.append(item.steps)
        force.append(item.force_used)
        for key, counter in item.counters.items():
            merged[key].update(counter)

    def _pct(counter: collections.Counter) -> dict[str, float]:
        total = sum(counter.values()) or 1
        return {
            str(key): round(value / total * 100.0, 3)
            for key, value in counter.most_common()
        }

    return {
        'runs': len(observations),
        'score': {
            'mean': round(statistics.mean(scores), 2) if scores else 0.0,
            'median': round(statistics.median(scores), 2) if scores else 0.0,
            'max': round(max(scores), 2) if scores else 0.0,
        },
        'lv7_treasure': {
            'mean': round(statistics.mean(lv7_treasure), 3) if lv7_treasure else 0.0,
            'max': max(lv7_treasure) if lv7_treasure else 0,
        },
        'lv7_total': {
            'mean': round(statistics.mean(lv7_total), 3) if lv7_total else 0.0,
            'max': max(lv7_total) if lv7_total else 0,
        },
        'paint_count': {
            'mean': round(statistics.mean(paint_counts), 3) if paint_counts else 0.0,
        },
        'steps': {
            'mean': round(statistics.mean(steps), 3) if steps else 0.0,
            'max': max(steps) if steps else 0,
        },
        'force_used': {
            'mean': round(statistics.mean(force), 3) if force else 0.0,
            'hist': {str(k): v for k, v in sorted(collections.Counter(force).items())},
        },
        'counts': {key: dict(counter) for key, counter in merged.items()},
        'percent': {key: _pct(counter) for key, counter in merged.items()},
    }


def _board_summary(aggregated: dict[str, Any]) -> str:
    """把「空白/阻挡/虚空」「4/5/6 格连线」摊平成人能读的一行。"""
    counts = aggregated['counts']
    runs = aggregated['runs'] or 1
    blank = counts['board_blank'].get(0, 0) / runs
    blocked = counts['board_blocked'].get(0, 0) / runs
    void = counts['board_void'].get(0, 0) / runs
    faded = counts['board_faded'].get(0, 0) / runs
    len4 = sum(counts['line_len4'].values()) / runs
    len5 = sum(counts['line_len5'].values()) / runs
    len6 = sum(counts['line_len6'].values()) / runs
    return (
        f'开局地块(每局均值): 空白 {blank:.2f} / 阻挡 {blocked:.2f} / '
        f'虚空 {void:.2f} / 失色 {faded:.2f}\n'
        f'开局空白连线(每局均值): 4 格 {len4:.2f} 条 / 5 格 {len5:.2f} 条 / '
        f'6 格 {len6:.2f} 条'
    )


def format_report(aggregated: dict[str, Any]) -> str:
    percent = aggregated['percent']
    lines: list[str] = []
    lines.append('=' * 72)
    lines.append(f"模拟统计  共 {aggregated['runs']} 局")
    lines.append('=' * 72)
    lines.append(_board_summary(aggregated))
    lines.append('')
    score = aggregated['score']
    lines.append(
        f"分数 mean={score['mean']} median={score['median']} max={score['max']}"
    )
    lines.append(
        f"7 级宝箱 mean={aggregated['lv7_treasure']['mean']} "
        f"max={aggregated['lv7_treasure']['max']}    "
        f"7 级总数 mean={aggregated['lv7_total']['mean']}"
    )
    lines.append(
        f"颜料数 mean={aggregated['paint_count']['mean']}    "
        f"步数 mean={aggregated['steps']['mean']}    "
        f"强制次数 mean={aggregated['force_used']['mean']}"
    )
    lines.append('')
    lines.append('--- 初始颜料类别几率 ---')
    for key, value in percent['anchor_category'].items():
        lines.append(f'   {key:<12} {value:6.2f}%')
    lines.append('--- 初始颜料是否宝箱 ---')
    for key, value in percent['anchor_treasure'].items():
        lines.append(f'   {key:<12} {value:6.2f}%')
    lines.append('')
    lines.append('--- 调色效果几率（按 方向|目标|提升级数）---')
    for key, value in list(percent['tint_effect'].items())[:20]:
        lines.append(f'   {key:<34} {value:6.2f}%')
    lines.append('--- 调色效果 提升级数分布 ---')
    for key, value in percent['tint_amount'].items():
        lines.append(f'   +{key} 级      {value:6.2f}%')
    lines.append('')
    lines.append('--- 笔触效果：出现几率 top15 ---')
    for key, value in list(percent['brush_offered'].items())[:15]:
        lines.append(f'   {key:<40} {value:6.2f}%')
    lines.append('--- 笔触效果：被选中几率 top15 ---')
    for key, value in list(percent['brush_chosen'].items())[:15]:
        lines.append(f'   {key:<40} {value:6.2f}%')
    return '\n'.join(lines)


def run(
    *,
    runs: int = 2000,
    workers: int = 8,
    tag: str = 'stats',
    out_root: str = os.path.join('runtime', 'monte_runs'),
    **kwargs: Any,
) -> dict[str, Any]:
    """跑一批并落盘，返回聚合结果。"""
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(out_root, f'{stamp}_{tag}')
    os.makedirs(out_dir, exist_ok=True)

    config = {
        'runs': runs,
        'workers': workers,
        'started_at': stamp,
        'grade_thresholds': list(pr.GRADE_THRESHOLDS),
        **{key: value for key, value in kwargs.items() if key != 'weights'},
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    chunk = max(1, runs // max(1, workers * 4))
    jobs = [
        (index, min(index + chunk, runs), kwargs)
        for index in range(0, runs, chunk)
    ]

    started = time.time()
    observations: list[RunObservation] = []
    with mp.Pool(workers) as pool:
        for chunk_result in pool.imap_unordered(_run_chunk, jobs):
            observations.extend(chunk_result)

    aggregated = _merge(observations)
    aggregated['config'] = config
    aggregated['duration_seconds'] = round(time.time() - started, 1)
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as handle:
        json.dump(aggregated, handle, ensure_ascii=False)
    report = format_report(aggregated)
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as handle:
        handle.write(report)
    return aggregated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='按当前规则跑模拟并采集统计口径')
    parser.add_argument('--runs', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--rules', default='none', help='none / 2 / 3 / ...')
    parser.add_argument('--prefer', default='greedy')
    parser.add_argument('--fusion', action='store_true')
    parser.add_argument('--max-steps', type=int, default=60)
    parser.add_argument('--pet-level', type=int, default=gm.DEFAULT_PET_LEVEL)
    parser.add_argument('--tag', default='stats')
    args = parser.parse_args(argv)

    rules = None if str(args.rules).lower() in ('none', 'null', '') else int(args.rules)
    aggregated = run(
        runs=args.runs,
        workers=args.workers,
        tag=args.tag,
        rules=rules,
        prefer=args.prefer,
        fusion_explore=args.fusion,
        max_steps=args.max_steps,
        pet_level=args.pet_level,
    )
    print(format_report(aggregated))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
