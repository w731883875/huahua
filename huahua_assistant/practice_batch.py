#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量扫描器（任务 C）：自动跑满指定局数，挑出**终局棋盘上有 7 级宝箱**的复盘。

为什么要专门扫 7 级宝箱：

* 7 级是等级上限（``MAX_LEVEL``），一枚 7 级宝箱是单格价值最高的形态之一；
* 宝箱只有靠「同格叠加成 + 反复升级」才可能顶到 7，天然是"加成叠加技巧"的
  活样本，复盘价值远高于普通局；
* 它**稀有**，靠手动找不现实，只能批量跑出来。

来源标注（关键）：

扫出来的复盘一律记 ``origin='simulated'`` / ``source='simulation'``，
**与实机录制的 ``real_capture`` 复盘严格区分**。原因见
:mod:`practice_library` 的「来源区分」注释：模拟棋盘用的是工程自设的
家族/宝箱比例，只能用来研究"技巧本身"，不能当成实机结论。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import game_model as gm
from . import practice as pr
from .practice_library import (
    ORIGIN_SIMULATED,
    SOURCE_SIMULATION,
    PracticeLibrary,
    seed_of,
)

#: 7 级宝箱复盘快照的落盘目录。
LEVEL7_DIR = Path('runtime') / 'practice_sessions' / 'level7'

#: 判定"终局有 7 级宝箱"的等级门槛。7 是 ``gm.MAX_LEVEL``。
TARGET_LEVEL = 7


# ---------------------------------------------------------------------------
# 单局扫描（必须在模块顶层，供多进程 pickle）
# ---------------------------------------------------------------------------

@dataclass
class Level7Hit:
    """一局命中 7 级宝箱的结果。"""

    seed: int
    score: float
    grade: str
    level7_tiles: list[tuple[int, int, int]]   # (row, col, final_level)
    treasure_tiles: list[tuple[int, int, int]]  # 所有宝箱格
    snapshot: dict[str, Any] = field(default_factory=dict)
    summary: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            'seed': self.seed,
            'score': round(self.score, 6),
            'grade': self.grade,
            'level7_tiles': self.level7_tiles,
            'treasure_tiles': self.treasure_tiles,
            'summary': self.summary,
            'origin': ORIGIN_SIMULATED,
            'source': SOURCE_SIMULATION,
            # 复盘本体：没有这个就还原不了，落盘和入库都要它。
            'snapshot': self.snapshot,
        }


def scan_one(job: tuple[int, int, int, str, float]) -> dict[str, Any] | None:
    """跑一局；终局有 7 级宝箱就返回结果，否则 ``None``。

    ``job = (seed, pet_level, rules, prefer, explore)``。返回值是纯 dict，
    可以直接跨进程传回或落盘。
    """
    seed, pet_level, rules, prefer, explore = job
    session = pr.start_practice(
        seed, pet_level=pet_level, rules=rules, wait_for_anchor=True
    )
    pr.auto_play(session, prefer=prefer, explore=explore)

    treasures: list[tuple[int, int, int]] = []
    level7: list[tuple[int, int, int]] = []
    for row in range(gm.BOARD_SIZE):
        for col in range(gm.BOARD_SIZE):
            tile = session.state.tile(row, col)
            if tile.state is not gm.TileState.PAINT or tile.paint is None:
                continue
            if not tile.paint.is_treasure:
                continue
            final_level = gm.tile_final_level(tile)
            treasures.append((row, col, int(final_level)))
            if final_level >= TARGET_LEVEL:
                level7.append((row, col, int(final_level)))

    if not level7:
        return None

    explanation = pr.explain_high_score(
        session.state,
        steps=session.steps,
        session=session,
        weights=session.weights,
    )
    hit = Level7Hit(
        seed=int(seed),
        score=float(explanation.score),
        grade=str(explanation.grade),
        level7_tiles=level7,
        treasure_tiles=treasures,
        snapshot=session.snapshot(),
        summary=list(explanation.summary),
    )
    return hit.as_dict()


# ---------------------------------------------------------------------------
# 批量调度
# ---------------------------------------------------------------------------

@dataclass
class ScanReport:
    """一次批量扫描的汇总。"""

    runs: int
    hits: list[dict[str, Any]]
    elapsed: float
    workers: int
    pet_level: int
    rules: int
    prefer: str
    saved_paths: list[str] = field(default_factory=list)
    library_written: int = 0

    @property
    def hit_rate(self) -> float:
        return len(self.hits) / self.runs if self.runs else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            'runs': self.runs,
            'hit_count': len(self.hits),
            'hit_rate': round(self.hit_rate, 6),
            'elapsed_sec': round(self.elapsed, 2),
            'workers': self.workers,
            'pet_level': self.pet_level,
            'rules': self.rules,
            'prefer': self.prefer,
            'origin': ORIGIN_SIMULATED,
            'hits': [{k: v for k, v in hit.items() if k != 'snapshot'} for hit in self.hits],
        }


def scan(
    *,
    runs: int = 10000,
    base_seed: int = 1,
    pet_level: int = gm.DEFAULT_PET_LEVEL,
    rules: int | None = 2,
    prefer: str = 'greedy',
    explore: float = 0.0,
    workers: int | None = None,
    save: bool = True,
    record: bool = True,
    library_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    progress: bool = False,
) -> ScanReport:
    """跑 ``runs`` 局，收集所有终局含 7 级宝箱的复盘。

    ``explore`` 见 :func:`huahua_assistant.practice.auto_play`：按这个概率
    随机选词缀。纯 greedy 会反复走同一条盲区，稀有高分组合永远采不到。
    """
    started = time.perf_counter()
    jobs = [
        (int(base_seed) + index, int(pet_level), rules, prefer, float(explore))
        for index in range(int(runs))
    ]
    workers = workers or min(8, os.cpu_count() or 1)

    hits: list[dict[str, Any]] = []
    if workers <= 1 or len(jobs) < 64:
        for done, job in enumerate(jobs, start=1):
            result = scan_one(job)
            if result is not None:
                hits.append(result)
            if progress and done % 500 == 0:
                print(f'  ... {done}/{len(jobs)} 局，命中 {len(hits)}', flush=True)
    else:
        with mp.Pool(processes=workers) as pool:
            for done, result in enumerate(pool.imap_unordered(scan_one, jobs, chunksize=8), start=1):
                if result is not None:
                    hits.append(result)
                if progress and done % 500 == 0:
                    print(f'  ... {done}/{len(jobs)} 局，命中 {len(hits)}', flush=True)

    hits.sort(key=lambda hit: (-hit['score'], hit['seed']))
    elapsed = time.perf_counter() - started

    report = ScanReport(
        runs=int(runs),
        hits=hits,
        elapsed=elapsed,
        workers=workers,
        pet_level=int(pet_level),
        rules=rules,
        prefer=prefer,
    )

    if save:
        report.saved_paths = save_hits(hits, out_dir=out_dir)
    if record:
        report.library_written = record_hits(hits, library_path=library_path)
    return report


def save_hits(
    hits: list[dict[str, Any]], *, out_dir: str | Path | None = None
) -> list[str]:
    """把命中的复盘快照逐个落盘，并写一份索引，返回路径列表。"""
    target = Path(out_dir) if out_dir is not None else LEVEL7_DIR
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    index: list[dict[str, Any]] = []
    for hit in hits:
        path = target / f'level7_seed{hit["seed"]}.json'
        payload = {
            'schema': 'huahua-level7-treasure-replay-v1',
            'origin': ORIGIN_SIMULATED,
            'source': SOURCE_SIMULATION,
            'note': (
                '本局为**模拟**复盘（本地按官方等级分布生成棋盘后自动完成），'
                '不是实机录制。可用于研究技巧，不能作为实机结论引用。'
            ),
            'seed': hit['seed'],
            'score': hit['score'],
            'grade': hit['grade'],
            'level7_tiles': hit['level7_tiles'],
            'treasure_tiles': hit['treasure_tiles'],
            'summary': hit['summary'],
            'snapshot': hit['snapshot'],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8'
        )
        paths.append(str(path))
        index.append({
            'seed': hit['seed'],
            'score': hit['score'],
            'grade': hit['grade'],
            'level7_count': len(hit['level7_tiles']),
            'level7_tiles': hit['level7_tiles'],
            'file': path.name,
        })
    if index:
        (target / 'INDEX.json').write_text(
            json.dumps(
                {
                    'schema': 'huahua-level7-treasure-index-v1',
                    'origin': ORIGIN_SIMULATED,
                    'source': SOURCE_SIMULATION,
                    'note': '全部为模拟复盘；实机复盘请查练习库 origin=real 的记录。',
                    'count': len(index),
                    'entries': index,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding='utf-8',
        )
    return paths


def record_hits(
    hits: list[dict[str, Any]], *, library_path: str | Path | None = None
) -> int:
    """把命中的复盘写进练习库，``source='simulation'``（即 origin=simulated）。

    必须先 ``replay_session`` 把走法重放出来，才有**终局**局面可解释；
    直接 ``restore_practice`` 只拿到开局，解释会算错。
    """
    library = PracticeLibrary(library_path)
    written = 0
    for hit in hits:
        snapshot = hit.get('snapshot') or {}
        try:
            session, check = pr.replay_session(snapshot)
        except Exception:  # noqa: BLE001 - 单局还原失败不该拖垮整批
            continue
        if not check.ok:
            continue
        explanation = pr.explain_high_score(
            session.state,
            steps=session.steps,
            session=session,
            weights=session.weights,
        )
        entry = library.record(
            explanation,
            snapshot,
            source=SOURCE_SIMULATION,
            note=f'终局 7 级宝箱 {len(hit["level7_tiles"])} 个（模拟复盘，非实机）',
        )
        if entry is not None:
            written += 1
    return written


def rebuild_index(out_dir: str | Path | None = None) -> Path:
    """从已落盘的快照重建 ``INDEX.json``。

    扫描可能在加索引之前就跑完了（历史批次），这个函数让老目录也能补齐索引，
    不必重跑一万局。
    """
    target = Path(out_dir) if out_dir is not None else LEVEL7_DIR
    entries: list[dict[str, Any]] = []
    for path in sorted(target.glob('level7_seed*.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        entries.append({
            'seed': payload.get('seed'),
            'score': payload.get('score'),
            'grade': payload.get('grade'),
            'level7_count': len(payload.get('level7_tiles') or ()),
            'level7_tiles': payload.get('level7_tiles') or [],
            'file': path.name,
        })
    entries.sort(
        key=lambda item: (
            -float(item.get('score') or 0.0),
            # 种子 0 是合法种子，不能用 `or` 兜底（会把 0 变成默认值）。
            seed_of(item) if seed_of(item) is not None else 0,
        )
    )
    index_path = target / 'INDEX.json'
    index_path.write_text(
        json.dumps(
            {
                'schema': 'huahua-level7-treasure-index-v1',
                'origin': ORIGIN_SIMULATED,
                'source': SOURCE_SIMULATION,
                'note': '全部为模拟复盘；实机复盘请查练习库 origin=real 的记录。',
                'count': len(entries),
                'entries': entries,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding='utf-8',
    )
    return index_path


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def format_report(report: ScanReport) -> str:
    """人看的扫描报告。"""
    lines = [
        f'=== 7 级宝箱扫描：跑 {report.runs} 局 '
        f'({report.workers} 进程, 宠物 Lv{report.pet_level}, '
        f'结算 {report.rules} 条, 策略 {report.prefer}) ===',
        f'耗时 {report.elapsed:.1f}s（{report.elapsed / max(1, report.runs) * 1000:.1f} ms/局）',
        f'命中 {len(report.hits)} 局（{report.hit_rate:.4%}）',
        '',
        '⚠ 来源：以上全部是 **模拟复盘**（origin=simulated），'
        '与实机录制的 real_capture 复盘严格区分，不可混用作实机结论。',
    ]
    if report.saved_paths:
        lines.append(f'已保存 {len(report.saved_paths)} 份复盘快照到 '
                     f'{Path(report.saved_paths[0]).parent}')
    if report.library_written:
        lines.append(f'已入库 {report.library_written} 条（source=simulation）')
    if not report.hits:
        lines.append('')
        lines.append('本批没有命中。7 级宝箱需要「同格反复叠加成 + 升级」，本身很稀有，'
                     '可以加大 --runs 或换 --prefer lookahead 再试。')
        return '\n'.join(lines)

    lines.append('')
    lines.append('命中明细（按分数降序）：')
    for hit in report.hits[:30]:
        tiles = '、'.join(f'({row},{col})Lv{level}' for row, col, level in hit['level7_tiles'])
        lines.append(f'  seed={hit["seed"]:<6} {hit["score"]:.4f}（{hit["grade"]}）  7级宝箱：{tiles}')
    if len(report.hits) > 30:
        lines.append(f'  … 另有 {len(report.hits) - 30} 局，见 JSON 输出')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(
        description='批量跑局，挑出终局有 7 级宝箱的复盘（模拟，非实机）'
    )
    parser.add_argument('--runs', type=int, default=10000)
    parser.add_argument('--base-seed', type=int, default=1)
    parser.add_argument('--pet', type=int, default=gm.DEFAULT_PET_LEVEL)
    parser.add_argument('--rules', type=int, default=2)
    parser.add_argument('--prefer', choices=('greedy', 'lookahead'), default='greedy')
    parser.add_argument(
        '--explore',
        type=float,
        default=0.0,
        help=(
            '探索率 0~1：按这个概率随机选词缀，而不是一律听推荐器。'
            '纯 greedy 会反复走同一条盲区，稀有高分组合（例如宝箱+全图+替换为自身）永远采不到。'
        ),
    )
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-record', action='store_true')
    parser.add_argument('--library', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--json', default=None, help='把汇总另存为 JSON')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args(argv)

    report = scan(
        runs=args.runs,
        base_seed=args.base_seed,
        pet_level=args.pet,
        rules=args.rules,
        prefer=args.prefer,
        explore=args.explore,
        workers=args.workers,
        save=not args.no_save,
        record=not args.no_record,
        library_path=args.library,
        out_dir=args.out_dir,
        progress=args.progress,
    )
    print(format_report(report))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=1), encoding='utf-8'
        )
        print(f'\n已写出 {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
