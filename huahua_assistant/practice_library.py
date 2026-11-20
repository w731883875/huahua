"""练习高分库：把练习对局的「分数 + 为什么高分 + 怎么复现」落到磁盘。

设计要点（对应「完善记录模式」的三条要求）：

1. **高分记录** —— 每局一条 JSONL，重复写入按 :func:`record_signature` 去重，
   同一 seed + 同一走法序列不会堆出重复记录；
2. **能够还原** —— 每条记录内嵌 ``snapshot``（开局局面 + 走法序列），
   用 :func:`huahua_assistant.practice.replay_session` 可完整重放并校验；
3. **为什么高分** —— 内嵌 ``breakdown``（分数拆解）与 ``tactics``（技巧证据），
   两者都是可计算的判据，不是主观描述。

与 ``learning.py`` 的分工：那边记录的是**实机捕获**的对局（``real_capture`` 门控、
供学习用）；这里记录的是**练习/模拟**产生的对局，只服务复盘，不混进学习样本。
两边都是 JSONL，但文件、schema、写入路径完全独立。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import practice as practice_mod


PRACTICE_LIBRARY_SCHEMA = 'huahua-practice-library-v1'
DEFAULT_LIBRARY_PATH = Path('runtime') / 'practice_library.jsonl'

#: 「高分」的下限。低于这个分仍然会入库（便于对比），但 ``is_high_score=False``。
#: 取 0.9 是对应等级 A 的界线，来自 :func:`practice._grade` 的同一套阈值。
HIGH_SCORE_THRESHOLD = 0.9

#: 「值得复盘」的下限：分数不高但技巧触发得多，也算有学习价值。
MIN_TACTIC_WEIGHT = 1.0

# ---- 来源区分 -------------------------------------------------------------
#
# 库里同时存两类记录，必须一眼分得清：
#
# * ``simulated``  —— 本地按官方几率生成棋盘跑出来的**模拟**复盘。
#   棋盘、背包、走法全是算出来的，官方几率表只覆盖到"等级分布"这一层，
#   颜料家族比例、宝箱出现率都还是工程自设参数（见 ``practice_simulate.assumptions``）。
#   **不能**当成实机结论引用。
# * ``real`` —— 实机录制的复盘（``app.py`` 的 ``real_capture`` 链路），
#   棋盘来自真实截图识别，是唯一能拿去和官方对账的东西。
#
# 这条界线是硬的：分析脚本按 origin 分流，报告里分开列。

SOURCE_SIMULATION = 'simulation'
SOURCE_MANUAL = 'manual'
SOURCE_PRACTICE = 'practice'
SOURCE_REAL = 'real_capture'

#: 哪些 ``source`` 算实机。
_REAL_SOURCES = frozenset({SOURCE_REAL, 'game_log', 'real_run'})

ORIGIN_SIMULATED = 'simulated'
ORIGIN_REAL = 'real'


def origin_of(source: str | None) -> str:
    """把 ``source`` 归到「模拟」或「真实」两类。"""
    return ORIGIN_REAL if (source or '') in _REAL_SOURCES else ORIGIN_SIMULATED


def record_signature(snapshot: dict[str, Any], *, origin: str | None = None) -> str:
    """一局的身份：来源 + seed + 参数 + 走法序列。

    不含分数与时间——同一局重跑出来的分数应当相同，带进签名反而会让
    「同局两次记录」变成两条。

    **必须带 origin**：模拟与实机的 seed、走法可能完全撞上（都是 ``seed=3``
    同一串动作），若不带来源，后记的那条会被当成重复直接丢掉——模拟复盘
    覆盖真实复盘，正好是最不能发生的事。
    """
    identity = {
        'origin': origin,
        'seed': snapshot.get('seed'),
        'pet_level': snapshot.get('pet_level'),
        'rules': snapshot.get('rules'),
        'max_steps': snapshot.get('max_steps'),
        'steps': [
            {
                'kind': step.get('kind'),
                'row': step.get('row'),
                'col': step.get('col'),
                'backpack_index': step.get('backpack_index'),
            }
            for step in snapshot.get('steps') or ()
        ],
    }
    blob = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def build_entry(
    explanation: practice_mod.HighScoreExplanation,
    snapshot: dict[str, Any],
    *,
    source: str = 'practice',
    note: str = '',
) -> dict[str, Any]:
    """把一次对局打包成一条库记录（还没落盘）。"""
    origin = origin_of(source)
    return {
        'schema': PRACTICE_LIBRARY_SCHEMA,
        'ts': time.time(),
        'source': source,
        'origin': origin,
        'signature': record_signature(snapshot, origin=origin),
        'note': note,
        'seed': snapshot.get('seed'),
        'pet_level': snapshot.get('pet_level'),
        'score': round(float(explanation.score), 6),
        'grade': explanation.grade,
        'is_high_score': float(explanation.score) >= HIGH_SCORE_THRESHOLD,
        'board_value': round(float(explanation.breakdown.board_value), 6),
        'breakdown': explanation.breakdown.as_dict(),
        'tactics': [item.as_dict() for item in explanation.tactics],
        'summary': list(explanation.summary),
        'board': list(explanation.session.get('board') or ()) if explanation.session else [],
        'snapshot': snapshot,
    }


def replay_flag(snapshot: dict[str, Any], recorded_value: float) -> dict[str, Any]:
    """把这份存档重放一遍，返回**复原一致性标记**。

    这是整套记录模式的意义所在：实机怎么打、我们本地规则算出来是不是同一个
    结果。**不一致必须留痕**，否则会一直拿错的规则往下推荐。

    校验两层，**任何一层不过都算不一致**：

    1. 走法能否重放（``ReplayCheck.ok``）——落子是否都执行成功、
       每一步落下的颜料签名是否与记录相同；
    2. 终局价值是否等于记录值——``replay_session`` 只比对快照内部的
       ``final_board_value``，那是"自己和自己比"，永远相等；
       必须再拿**入库时记下的分数**对一次，才能发现真正的偏差。

    返回字段：

    * ``ok`` —— 能否还原（``None`` 表示还没校验过）；
    * ``replayed_value`` —— 回放算出的终局价值；
    * ``recorded_value`` —— 原始记录里的终局价值；
    * ``delta`` —— 两者之差；
    * ``mismatch`` —— 第一处分歧的人类可读说明，一致时为空串。
    """
    tolerance = 1e-6
    try:
        _session, check = practice_mod.replay_session(snapshot)
    except Exception as exc:  # noqa: BLE001 - 校验失败要留痕，不能中断写入
        return {
            'ok': False,
            'replayed_value': None,
            'recorded_value': round(float(recorded_value), 6),
            'delta': None,
            'mismatch': f'{type(exc).__name__}: {exc}',
        }
    replayed = float(check.actual_value)
    recorded = float(recorded_value)
    delta = replayed - recorded
    mismatch = str(check.mismatch or '')
    ok = bool(check.ok)
    if ok and abs(delta) > tolerance:
        ok = False
        mismatch = (
            f'终局分与记录不一致：记录 {recorded:.6f}，回放 {replayed:.6f}'
        )
    return {
        'ok': ok,
        'replayed_value': round(replayed, 6),
        'recorded_value': round(recorded, 6),
        'delta': round(delta, 6),
        'mismatch': mismatch,
    }


def seed_of(entry: dict[str, Any]) -> int | None:
    """读出记录里的种子；读不出就返回 ``None``。

    **不能用 ``entry.get('seed') or 默认值``**：``seed=0`` 是合法种子，是假值，
    会被 ``or`` 当成缺值——``by_seed(0)`` 从此永远查不到东西，而 ``by_seed(-1)``
    反而把所有缺种子的记录捞出来。缺失、空串、非数字、布尔一律返回 ``None``，
    让调用方自己决定怎么处理，不塞哨兵值。
    """
    if not isinstance(entry, dict):
        return None
    raw = entry.get('seed')
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        # 5.0 当 5；5.5 不是种子，别悄悄截断成 5。
        return int(raw) if float(raw).is_integer() else None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


class PracticeLibrary:
    """练习高分库。默认落在 ``runtime/practice_library.jsonl``。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_LIBRARY_PATH
        self._signatures: set[str] | None = None

    # ---- 写 ----------------------------------------------------------
    def record(
        self,
        explanation: practice_mod.HighScoreExplanation,
        snapshot: dict[str, Any],
        *,
        source: str = 'practice',
        note: str = '',
        verify: bool = True,
    ) -> dict[str, Any] | None:
        """入库。返回写入的记录；重复（同签名）时返回 ``None``。

        ``verify=True`` 时会立刻把这份存档重放一遍，把**复原一致性**写进
        ``replay`` 字段。实机记录尤其需要这个：回放对不上就说明本地规则
        和游戏不一致，必须留痕而不是当成一次正常记录。
        """
        entry = build_entry(explanation, snapshot, source=source, note=note)
        if verify:
            entry['replay'] = replay_flag(snapshot, entry['board_value'])
        else:
            entry['replay'] = {
                'ok': None,
                'replayed_value': None,
                'recorded_value': round(float(entry['board_value']), 6),
                'delta': None,
                'mismatch': '',
            }
        if entry['signature'] in self._known():
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + '\n')
        self._known().add(entry['signature'])
        return entry

    def _known(self) -> set[str]:
        if self._signatures is None:
            self._signatures = {entry['signature'] for entry in self.entries()}
        return self._signatures

    # ---- 读 ----------------------------------------------------------
    def entries(self) -> list[dict[str, Any]]:
        """读出全部记录。坏行跳过而不是炸掉整个库——库是一路追加的，
        一行写坏不该让历史记录全不可用。"""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get('signature'):
                out.append(payload)
        return out

    def top(self, limit: int = 10, *, high_only: bool = False) -> list[dict[str, Any]]:
        """按分数降序取前 ``limit`` 条。"""
        items = [item for item in self.entries() if not high_only or item.get('is_high_score')]
        items.sort(key=lambda item: float(item.get('score') or 0.0), reverse=True)
        return items[: max(0, int(limit))]

    def by_seed(self, seed: int) -> list[dict[str, Any]]:
        """按种子取记录。``seed=0`` 是合法种子，必须能查回自己的记录。"""
        try:
            target = int(seed)
        except (TypeError, ValueError):
            return []
        return [item for item in self.entries() if seed_of(item) == target]

    def tactic_stats(self, *, direction: str = 'all') -> list[tuple[str, int, float]]:
        """技巧命中统计：``(key, 次数, 命中该技巧的平均分)``，按次数降序。

        ``direction``：

        * ``'positive'`` —— 只看正向技巧（``weight > 0``）；
        * ``'negative'`` —— 只看"待改进"项（``weight < 0``）；
        * ``'all'`` —— 不区分。

        **必须能区分**：正向技巧命中多代表打得好，而"待改进"项命中多代表
        犯了这个错，两者混在一起算平均分，会把"锚点放外圈"这类扣分项
        说成"命中它分数更低"，读起来像是技巧本身有害。
        """
        counts: dict[str, int] = {}
        scores: dict[str, float] = {}
        for entry in self.entries():
            score = float(entry.get('score') or 0.0)
            for tactic in entry.get('tactics') or ():
                key = str(tactic.get('key') or '')
                if not key:
                    continue
                weight = float(tactic.get('weight') or 0.0)
                if direction == 'positive' and weight <= 0:
                    continue
                if direction == 'negative' and weight >= 0:
                    continue
                counts[key] = counts.get(key, 0) + 1
                scores[key] = scores.get(key, 0.0) + score
        rows = [
            (key, count, scores[key] / count if count else 0.0)
            for key, count in counts.items()
        ]
        rows.sort(key=lambda row: (-row[1], -row[2]))
        return rows

    def compare_tactics(self) -> list[dict[str, Any]]:
        """「这个技巧到底有没有用」：命中它的局平均分 vs 全库平均分。

        这是**相关性**不是因果——命中某技巧的局本身可能是好局面堆出来的。
        但差值方向能用来验证 :data:`practice.TACTIC_LIBRARY` 里的建议是否站得住：
        如果某条号称正向的技巧，命中它的局平均分反而低于全库，
        要么判据写错了，要么建议本身就错。

        结果带 ``direction`` 字段（``positive`` / ``negative``），
        正向技巧与"待改进"项**不再混算**。
        """
        entries = self.entries()
        if not entries:
            return []
        overall = sum(float(item.get('score') or 0.0) for item in entries) / len(entries)
        out: list[dict[str, Any]] = []
        for direction in ('positive', 'negative'):
            for key, count, mean in self.tactic_stats(direction=direction):
                out.append({
                    'key': key,
                    'direction': direction,
                    'title': practice_mod.TACTIC_LIBRARY.get(key, (key, '', ''))[0],
                    'hits': count,
                    'mean_score': round(mean, 6),
                    'overall_mean': round(overall, 6),
                    'delta': round(mean - overall, 6),
                })
        out.sort(key=lambda row: (row['direction'], -row['hits']))
        return out

    def verify_all(self) -> list[tuple[dict[str, Any], practice_mod.ReplayCheck]]:
        """把库里每一局都重放一遍，返回校验结果。用于确认"能不能还原"。

        比对口径与 :func:`replay_flag` 一致：除了 ``replay_session`` 内部的
        （快照自洽）检查，还要拿**记录里的分数**再对一次，否则永远报"一致"。
        """
        out: list[tuple[dict[str, Any], practice_mod.ReplayCheck]] = []
        for entry in self.entries():
            snapshot = entry.get('snapshot') or {}
            recorded_value = float(entry.get('board_value') or entry.get('score') or 0.0)
            try:
                _, check = practice_mod.replay_session(snapshot)
            except Exception as exc:  # noqa: BLE001 - 校验失败要报告，不该中断整批
                check = practice_mod.ReplayCheck(
                    ok=False,
                    steps_replayed=0,
                    expected_value=recorded_value,
                    actual_value=0.0,
                    mismatch=f'{type(exc).__name__}: {exc}',
                )
                out.append((entry, check))
                continue
            if check.ok and abs(float(check.actual_value) - recorded_value) > 1e-6:
                check = practice_mod.ReplayCheck(
                    ok=False,
                    steps_replayed=check.steps_replayed,
                    expected_value=recorded_value,
                    actual_value=check.actual_value,
                    mismatch=(
                        f'终局分与记录不一致：记录 {recorded_value:.6f}，'
                        f'回放 {check.actual_value:.6f}'
                    ),
                    paint_mismatch=check.paint_mismatch,
                    board=check.board,
                )
            out.append((entry, check))
        return out

    def split_by_origin(self) -> dict[str, dict[str, Any]]:
        """按「模拟 / 真实」分开统计。

        这两类**永远不能混在一起看**：模拟复盘是自己生成棋盘跑出来的，
        真实复盘来自实机截图识别。混统计会让"官方对账"变成自证。
        """
        buckets: dict[str, list[dict[str, Any]]] = {}
        for entry in self.entries():
            origin = str(entry.get('origin') or origin_of(entry.get('source')))
            buckets.setdefault(origin, []).append(entry)
        out: dict[str, dict[str, Any]] = {}
        for origin, items in buckets.items():
            scores = [float(item.get('score') or 0.0) for item in items]
            out[origin] = {
                'runs': len(items),
                'high_scores': sum(1 for item in items if item.get('is_high_score')),
                'best_score': round(max(scores), 6) if scores else 0.0,
                'mean_score': round(sum(scores) / len(scores), 6) if scores else 0.0,
                'sources': sorted({str(item.get('source') or '') for item in items}),
            }
        return out

    def consistency(self) -> dict[str, Any]:
        """复原一致性汇总：多少局能还原、多少局对不上。

        **对不上的那些就是"本地规则和游戏不一样"的证据**，比分数本身更重要：
        分数只能说明打得好不好，一致性才能说明我们算得对不对。
        """
        entries = self.entries()
        checked = [
            item for item in entries
            if isinstance(item.get('replay'), dict) and item['replay'].get('ok') is not None
        ]
        mismatched = [item for item in checked if not item['replay'].get('ok')]
        unchecked = len(entries) - len(checked)
        by_origin: dict[str, dict[str, Any]] = {}
        for origin in (ORIGIN_REAL, ORIGIN_SIMULATED):
            rows = [
                item for item in checked
                if str(item.get('origin') or origin_of(item.get('source'))) == origin
            ]
            bad = [item for item in rows if not item['replay'].get('ok')]
            by_origin[origin] = {
                'checked': len(rows),
                'mismatched': len(bad),
                'examples': [
                    {
                        'seed': item.get('seed'),
                        'source': item.get('source'),
                        'recorded_value': item['replay'].get('recorded_value'),
                        'replayed_value': item['replay'].get('replayed_value'),
                        'delta': item['replay'].get('delta'),
                        'mismatch': item['replay'].get('mismatch'),
                    }
                    for item in bad[:10]
                ],
            }
        return {
            'checked': len(checked),
            'mismatched': len(mismatched),
            'unchecked': unchecked,
            'by_origin': by_origin,
        }

    def mismatched(self) -> list[dict[str, Any]]:
        """所有回放对不上的记录，按偏差绝对值降序。"""
        rows = [
            item for item in self.entries()
            if isinstance(item.get('replay'), dict) and item['replay'].get('ok') is False
        ]
        rows.sort(
            key=lambda item: abs(float(item['replay'].get('delta') or 0.0)),
            reverse=True,
        )
        return rows

    def recheck_consistency(self) -> dict[str, Any]:
        """把库里所有记录重新校验一遍并**就地回写** ``replay`` 字段。

        规则改过（比如作用范围 buff 的结算时机）之后必须重跑一次：
        以前"对得上"的记录可能已经对不上，以前对不上的可能已经修好。
        返回 ``{checked, fixed, broken, unchanged}``。
        """
        entries = self.entries()
        fixed = broken = unchanged = 0
        for entry in entries:
            snapshot = entry.get('snapshot') or {}
            if not snapshot:
                continue
            before = (entry.get('replay') or {}).get('ok')
            entry['replay'] = replay_flag(snapshot, entry.get('board_value') or 0.0)
            after = entry['replay']['ok']
            if before is True and after is False:
                broken += 1
            elif before is False and after is True:
                fixed += 1
            else:
                unchanged += 1
        if entries:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('w', encoding='utf-8') as handle:
                for entry in entries:
                    handle.write(
                        json.dumps(entry, ensure_ascii=False, sort_keys=True) + '\n'
                    )
            self._signatures = None
        return {
            'checked': len(entries),
            'fixed': fixed,
            'broken': broken,
            'unchanged': unchanged,
        }

    def stats(self) -> dict[str, Any]:
        entries = self.entries()
        if not entries:
            return {
                'path': str(self.path),
                'runs': 0,
                'high_scores': 0,
                'best_score': 0.0,
                'mean_score': 0.0,
                'tactics': [],
            }
        scores = [float(item.get('score') or 0.0) for item in entries]
        return {
            'path': str(self.path),
            'runs': len(entries),
            'high_scores': sum(1 for item in entries if item.get('is_high_score')),
            'best_score': round(max(scores), 6),
            'mean_score': round(sum(scores) / len(scores), 6),
            'origins': self.split_by_origin(),
            'consistency': self.consistency(),
            'tactics': [
                {'key': key, 'hits': hits, 'mean_score': round(mean, 6)}
                for key, hits, mean in self.tactic_stats()
            ],
        }


def record_sessions(
    sessions: Iterable[practice_mod.PracticeSession],
    *,
    path: str | Path | None = None,
    source: str = 'simulation',
    note: str = '',
) -> dict[str, Any]:
    """批量入库。返回 ``{written, skipped, path}``。

    每局都重新算一次 ``explain_high_score``，不依赖调用方先算好——这样
    "入库的解释"与"界面上看到的解释"永远同源。
    """
    library = PracticeLibrary(path)
    written = 0
    skipped = 0
    for session in sessions:
        explanation = practice_mod.explain_high_score(
            session.state,
            steps=session.steps,
            session=session,
            weights=session.weights,
        )
        entry = library.record(
            explanation, session.snapshot(), source=source, note=note
        )
        if entry is None:
            skipped += 1
        else:
            written += 1
    return {'written': written, 'skipped': skipped, 'path': str(library.path)}


def format_library_report(stats: dict[str, Any], limit: int = 10) -> str:
    """把 :meth:`PracticeLibrary.stats` 渲染成人看的几行。"""
    lines = [f'练习库 {stats["path"]}']
    if not stats['runs']:
        lines.append('  （空，先跑几局或 ``practice --runs N --record``）')
        return '\n'.join(lines)
    lines.append(
        f'  共 {stats["runs"]} 局，其中高分 {stats["high_scores"]} 局；'
        f'最高 {stats["best_score"]:.4f}，平均 {stats["mean_score"]:.4f}'
    )
    origins = stats.get('origins') or {}
    if origins:
        # 模拟与真实必须分列：混在一起看会让"对账官方"变成自证。
        for origin, label in ((ORIGIN_REAL, '实机'), (ORIGIN_SIMULATED, '模拟')):
            row = origins.get(origin)
            if not row:
                continue
            lines.append(
                f'  [{label}] {row["runs"]} 局，高分 {row["high_scores"]}，'
                f'最高 {row["best_score"]:.4f}，平均 {row["mean_score"]:.4f}'
                f'（来源 {"/".join(row["sources"])}）'
            )
    # 复原一致性放在分数**前面**：分数只说明打得好不好，
    # 一致性才说明我们的规则算得对不对。对不上就必须立刻看见。
    consistency = stats.get('consistency') or {}
    if consistency.get('checked'):
        bad = int(consistency.get('mismatched') or 0)
        mark = '⚠' if bad else '✓'
        lines.append(
            f'  {mark} 复原一致性：已校验 {consistency["checked"]} 局，'
            f'对不上 {bad} 局'
            + (f'，未校验 {consistency["unchecked"]} 局'
               if consistency.get('unchecked') else '')
        )
        by_origin = consistency.get('by_origin') or {}
        for origin, label in ((ORIGIN_REAL, '实机'), (ORIGIN_SIMULATED, '模拟')):
            row = by_origin.get(origin) or {}
            if not row.get('checked'):
                continue
            if row.get('mismatched'):
                lines.append(
                    f'    [{label}] {row["mismatched"]}/{row["checked"]} 局与本地规则不一致'
                )
                for example in (row.get('examples') or ())[:3]:
                    lines.append(
                        f'      · seed={example.get("seed")} '
                        f'记录 {example.get("recorded_value")} '
                        f'→ 回放 {example.get("replayed_value")}'
                        f'（差 {example.get("delta")}）{example.get("mismatch") or ""}'
                    )
    if stats['tactics']:
        lines.append('  技巧命中：')
        for row in stats['tactics'][:limit]:
            title = practice_mod.TACTIC_LIBRARY.get(row['key'], (row['key'], '', ''))[0]
            lines.append(
                f'    {row["hits"]:4d} 次  平均 {row["mean_score"]:.4f}  {title}'
            )
    return '\n'.join(lines)


@dataclass(frozen=True)
class LibraryEntryView:
    """界面上显示一条记录所需的字段（不带 snapshot，避免刷屏）。"""

    seed: int
    score: float
    grade: str
    is_high_score: bool
    note: str
    top_tactics: tuple[str, ...]
    summary: tuple[str, ...]
    replay_ok: bool | None = None
    replay_mismatch: str = ''

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> 'LibraryEntryView':
        tactics = sorted(
            (item for item in entry.get('tactics') or () if float(item.get('weight') or 0) > 0),
            key=lambda item: -float(item.get('weight') or 0),
        )
        replay = entry.get('replay') or {}
        known_seed = seed_of(entry)
        return cls(
            seed=known_seed if known_seed is not None else 0,
            score=float(entry.get('score') or 0.0),
            grade=str(entry.get('grade') or ''),
            is_high_score=bool(entry.get('is_high_score')),
            note=str(entry.get('note') or ''),
            top_tactics=tuple(str(item.get('title') or '') for item in tactics[:3]),
            summary=tuple(str(line) for line in entry.get('summary') or ()),
            replay_ok=replay.get('ok'),
            replay_mismatch=str(replay.get('mismatch') or ''),
        )

    def format_text(self) -> str:
        flag = '★' if self.is_high_score else ' '
        # 复原标记与高分标记并列：对不上的记录即使分数再高也不能当范例用。
        replay_mark = ' ⚠复原不一致' if self.replay_ok is False else ''
        lines = [f'{flag} seed={self.seed}  {self.score:.4f}（{self.grade}）{replay_mark}']
        if self.replay_ok is False and self.replay_mismatch:
            lines.append(f'    ⚠ 回放对不上：{self.replay_mismatch}')
        if self.note:
            lines.append(f'    备注：{self.note}')
        for line in self.summary:
            lines.append(f'    · {line}')
        for title in self.top_tactics:
            lines.append(f'    + {title}')
        return '\n'.join(lines)
