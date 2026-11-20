#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手动练习驱动（任务 B）：一步一步下，随时存档、还原、复盘。

设计要点：

* 手动落子不需要新模型 —— :class:`practice.PracticeSession` 早就有
  ``place_anchor`` / ``install`` / ``force_complete`` / ``refresh_rule``，
  缺的只是一个**人能用**的入口；
* 每一步之后自动留一帧快照，``undo`` 直接退回上一帧并重放，
  不需要为撤销单独写反向逻辑；
* 一局结束时用 :func:`practice.explain_high_score` 说清"为什么是这个分"，
  并把这局连同走法一起写进高分库（同签名去重）。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import game_model as gm
from . import practice as pr
from .practice_library import PracticeLibrary

#: 手动对局的存档目录。
SESSION_DIR = Path('runtime') / 'practice_sessions'


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _enum_text(value: Any) -> str:
    """枚举取中文值；非枚举原样转字符串。"""
    return str(getattr(value, 'value', value))


def rule_text(rule: gm.CurrentPaintRule) -> str:
    """把调色目标渲染成一句人话。"""
    parts = [
        f'{rule.paint.subtype} Lv{rule.paint.level}',
        _enum_text(rule.direction),
        _enum_text(rule.target_object),
        f'需连 {rule.match_required}',
        f'提升 {rule.upgrade_amount}',
    ]
    extras = [_enum_text(item) for item in rule.extra_directions]
    if extras:
        parts.append('额外方向 ' + '/'.join(extras))
    return ' · '.join(parts)


def render_board(session: pr.PracticeSession) -> str:
    """棋盘 + 词缀 + 背包 + 剩余资源，一屏能看完。"""
    state = session.state
    lines = [f'第 {len(session.steps)} 步   局面价值 {session.value:.4f}']
    lines.extend('  ' + row for row in pr._board_rows(state))

    rule = state.current_rule
    if rule is None:
        lines.append('词缀：（无，本局已结算完）')
    else:
        pending = '待放锚点' if gm.current_rule_is_pending(state) else '作画中'
        lines.append(f'词缀：{rule_text(rule)}   [{pending}]')

    backpack = state.backpack
    if backpack:
        lines.append(f'背包（{len(backpack)} 枚，用 put 序号 行 列 放置）：')
        for index, paint in enumerate(backpack):
            mark = '★宝箱' if paint.is_treasure else ''
            lines.append(
                f'  [{index + 1}] {paint.subtype} Lv{paint.level} '
                f'({paint.category.value}) {mark}'
            )
    else:
        lines.append('背包：空')

    lines.append(
        f'强制调色剩余 {session.force_remaining}  '
        f'刷新剩余 {session.refresh_remaining}  '
        f'宠物 Lv{session.pet_level}'
    )
    return '\n'.join(lines)


def render_tactics(session: pr.PracticeSession) -> str:
    """当前局面能看出的技巧（正向 + 待改进）。"""
    findings = pr.find_tactics(session.state, steps=session.steps)
    if not findings:
        return '技巧：暂无触发'
    lines = ['技巧：']
    for item in findings:
        sign = '✔' if item.weight >= 0 else '✘'
        lines.append(f'  {sign} {item.title}')
        lines.append(f'      {item.evidence}')
    return '\n'.join(lines)


def render_advice(session: pr.PracticeSession, limit: int = 5) -> str:
    """让策略给出候选着法（字段名容错，策略结构变了也不至于炸）。"""
    try:
        items = gm.recommend_placements(session.state, limit=limit)
    except Exception as exc:  # noqa: BLE001 - 建议是锦上添花，不该挡着人下棋
        return f'建议不可用：{type(exc).__name__}: {exc}'
    if not items:
        return '建议：当前没有可选着法'
    lines = ['建议：']
    for rank, item in enumerate(items, start=1):
        row = getattr(item, 'row', '?')
        col = getattr(item, 'col', '?')
        score = getattr(item, 'score', getattr(item, 'value', None))
        text = getattr(item, 'reason', getattr(item, 'note', ''))
        head = f'  {rank}. ({row},{col})'
        if isinstance(score, (int, float)):
            head += f' 评分 {float(score):.3f}'
        if text:
            head += f'  {text}'
        lines.append(head)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 存档
# ---------------------------------------------------------------------------

def save_snapshot(snapshot: dict[str, Any], name: str) -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    safe = ''.join(ch for ch in name if ch.isalnum() or ch in '-_') or 'session'
    path = SESSION_DIR / f'{safe}.json'
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=1), encoding='utf-8'
    )
    return path


def load_snapshot(name: str) -> dict[str, Any]:
    path = SESSION_DIR / f'{name}.json'
    if not path.exists():
        raise FileNotFoundError(f'没有这个存档：{path}')
    return json.loads(path.read_text(encoding='utf-8'))


def rebuild(snapshot: dict[str, Any]) -> pr.PracticeSession:
    """按快照重建一局会话（含强制调色/刷新的已用次数）。"""
    session = pr.restore_practice(snapshot)
    for step in snapshot.get('steps') or ():
        pr.apply_step(session, step)
    return session


# ---------------------------------------------------------------------------
# 交互驱动
# ---------------------------------------------------------------------------

@dataclass
class ManualPractice:
    """一局手动练习。所有动作都返回 (是否成功, 说明)。"""

    session: pr.PracticeSession
    library: PracticeLibrary = field(default_factory=PracticeLibrary)
    _frames: list[dict[str, Any]] = field(default_factory=list, init=False)
    _saved: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._frames = [self.session.snapshot()]

    # ---- 动作 --------------------------------------------------------
    def anchor(self, row: int, col: int) -> tuple[bool, str]:
        ok = self.session.place_anchor(row, col)
        return self._after(ok, f'放锚点 ({row},{col})')

    def put(self, index: int, row: int, col: int) -> tuple[bool, str]:
        """``index`` 是 1-based 序号，与界面显示一致。"""
        ok = self.session.install(index - 1, row, col)
        return self._after(ok, f'落第 {index} 枚到 ({row},{col})')

    def force(self) -> tuple[bool, str]:
        return self._after(self.session.force_complete(), '强制调色')

    def refresh(self) -> tuple[bool, str]:
        return self._after(self.session.refresh_rule(), '刷新词缀')

    def _after(self, ok: bool, what: str) -> tuple[bool, str]:
        if not ok:
            return False, f'{what} 失败（条件不满足）'
        self._frames.append(self.session.snapshot())
        return True, what

    def undo(self) -> tuple[bool, str]:
        if len(self._frames) <= 1:
            return False, '已经回到开局，无法再撤'
        self._frames.pop()
        self.session = rebuild(self._frames[-1])
        return True, '已撤销上一步'

    # ---- 结算 --------------------------------------------------------
    def explain(self) -> pr.HighScoreExplanation:
        return pr.explain_high_score(
            self.session.state,
            steps=self.session.steps,
            session=self.session,
            weights=self.session.weights,
        )

    def record(self, *, note: str = '') -> dict[str, Any] | None:
        """把这一局写进高分库。返回写入的记录；重复则 ``None``。"""
        entry = self.library.record(
            self.explain(),
            self.session.snapshot(),
            source='manual',
            note=note,
        )
        self._saved = entry is not None
        return entry


def _parse(line: str) -> tuple[str, list[str]]:
    parts = line.strip().split()
    if not parts:
        return '', []
    return parts[0].lower(), parts[1:]


def _ints(tokens: list[str], count: int) -> tuple[int, ...] | None:
    if len(tokens) < count:
        return None
    try:
        return tuple(int(token) for token in tokens[:count])
    except ValueError:
        return None


HELP = """\
board              看棋盘
anchor 行 列       放初始颜料（开局第一步）
put 序号 行 列     把背包第 N 枚落到 (行,列)，行列从 0 开始
force              强制调色
refresh            刷新词缀
advice             让策略给候选着法
tactics            看当前触发的技巧
undo               撤销上一步
save 名字          存档
load 名字          读档
explain            结算并解释为什么是这个分
record             结算并写进高分库
quit               退出
"""


def play_interactive(
    session: pr.PracticeSession,
    *,
    library: PracticeLibrary | None = None,
    input_func=input,
    output_func=print,
) -> pr.HighScoreExplanation:
    """命令行交互对局。返回最终解释。"""
    driver = ManualPractice(session=session, library=library or PracticeLibrary())
    output_func(render_board(driver.session))
    output_func('输入 help 看指令。')
    while True:
        try:
            line = input_func('> ')
        except EOFError:
            break
        command, args = _parse(line)
        if not command:
            continue
        if command in ('quit', 'exit', 'q'):
            break
        if command == 'help':
            output_func(HELP)
        elif command == 'board':
            output_func(render_board(driver.session))
        elif command == 'anchor':
            values = _ints(args, 2)
            if values is None:
                output_func('用法：anchor 行 列')
                continue
            ok, message = driver.anchor(*values)
            output_func(message)
            if ok:
                output_func(render_board(driver.session))
        elif command == 'put':
            values = _ints(args, 3)
            if values is None:
                output_func('用法：put 序号 行 列')
                continue
            ok, message = driver.put(*values)
            output_func(message)
            if ok:
                output_func(render_board(driver.session))
        elif command in ('force', 'f'):
            ok, message = driver.force()
            output_func(message)
            if ok:
                output_func(render_board(driver.session))
        elif command == 'refresh':
            ok, message = driver.refresh()
            output_func(message)
            if ok:
                output_func(render_board(driver.session))
        elif command in ('advice', 'a'):
            output_func(render_advice(driver.session))
        elif command in ('tactics', 't'):
            output_func(render_tactics(driver.session))
        elif command == 'undo':
            ok, message = driver.undo()
            output_func(message)
            if ok:
                output_func(render_board(driver.session))
        elif command == 'save':
            name = args[0] if args else f'seed{driver.session.seed}'
            path = save_snapshot(driver.session.snapshot(), name)
            output_func(f'已存档 {path}')
        elif command == 'load':
            if not args:
                output_func('用法：load 名字')
                continue
            try:
                driver = ManualPractice(
                    session=rebuild(load_snapshot(args[0])),
                    library=driver.library,
                )
            except FileNotFoundError as exc:
                output_func(str(exc))
                continue
            output_func(f'已读档 {args[0]}')
            output_func(render_board(driver.session))
        elif command == 'explain':
            output_func(driver.explain().format_text())
        elif command == 'record':
            entry = driver.record()
            output_func('已入库' if entry else '这局已在库里（签名重复）')
        else:
            output_func(f'不认识的指令：{command}（help 看列表）')
    explanation = driver.explain()
    output_func(explanation.format_text())
    if not driver._saved:
        entry = driver.record()
        output_func('已自动入库' if entry else '这局已在库里（签名重复）')
    return explanation


def main(argv: list[str] | None = None) -> int:
    # 控制台默认代码页是 GBK，中文会糊；强制 UTF-8 输出。
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description='手动练习：一步步下，随时存档复盘')
    parser.add_argument('--seed', type=int, default=None, help='对局种子；不填则随机')
    parser.add_argument('--pet', type=int, default=gm.DEFAULT_PET_LEVEL, help='宠物等级 1-6')
    parser.add_argument('--rules', type=int, default=3, help='本局结算几条词缀')
    parser.add_argument('--max-steps', type=int, default=60)
    parser.add_argument('--load', default=None, help='从存档继续')
    parser.add_argument('--library', default=None, help='高分库路径')
    args = parser.parse_args(argv)

    library = PracticeLibrary(args.library)
    if args.load:
        session = rebuild(load_snapshot(args.load))
    else:
        session = pr.start_practice(
            args.seed,
            pet_level=args.pet,
            rules=args.rules,
            max_steps=args.max_steps,
            wait_for_anchor=True,
        )
    play_interactive(session, library=library)
    return 0


if __name__ == '__main__':
    sys.exit(main())
