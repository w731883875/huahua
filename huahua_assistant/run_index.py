"""把 ``auto_run_records.jsonl`` 拆成「一局一局」，**包括没录完的那些**。

存在理由：录制文件里 48 个开局只有 12 个结束。剩下那些在回放入口里根本不存在，
玩家一局录到一半，回头只能看到"空列表"，分不清是重开、退出、还是写盘失败。
这个模块只做只读分析：不改记录、不猜结局，缺证据就写"未知"。

三个概念必须分开（用户能看到的三句话）：

* ``board_full`` —— 只看到"棋盘没空格了"，这是**终止条件**，不是结算结果；
* ``finish_seen`` —— 看到了结束按钮 / 进入了下一局的开始画面；
* ``confirmed_result`` —— 真实的结算结果。**录制器单独判定不了**，必须由
  ``learning`` 那一侧用人工确认或结果画面补上，所以这里永远标"未确认"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


#: 结束原因 → 我们**知道到什么程度**。
#: ``no_blank_tiles``：棋盘没空格了，游戏可能还没结算；``finish_button``：看到结束按钮；
#: ``next_start_screen``：直接进了下一局的开始画面（上一局显然结束了，但没看到结算界面）。
REASON_OUTCOME = {
    "no_blank_tiles": "board_full",
    "finish_button": "finish_seen",
    "next_start_screen": "finish_seen",
}

OUTCOME_BOARD_FULL = "board_full"
OUTCOME_FINISH_SEEN = "finish_seen"
OUTCOME_INTERRUPTED = "interrupted"
OUTCOME_CONFIRMED = "confirmed_result"
OUTCOME_UNKNOWN = "unknown"

OUTCOME_LABELS = {
    OUTCOME_BOARD_FULL: "满盘（未确认结算）",
    OUTCOME_FINISH_SEEN: "见结束态（未确认结果）",
    OUTCOME_INTERRUPTED: "中断（没等到结束）",
    OUTCOME_CONFIRMED: "已确认结果",
    OUTCOME_UNKNOWN: "未知",
}


def outcome_of_reason(reason: str | None) -> str:
    """结束原因 → 结局类型。缺原因就是"未知"，不拿别的原因顶替。"""
    if not reason:
        return OUTCOME_UNKNOWN
    return REASON_OUTCOME.get(str(reason), OUTCOME_UNKNOWN)


@dataclass(frozen=True)
class BadLine:
    line_no: int
    error: str
    preview: str


@dataclass(frozen=True)
class SessionSummary:
    session: str
    start_reason: str | None = None
    finish_reason: str | None = None
    started_ts: float | None = None
    finished_ts: float | None = None
    event_count: int = 0
    last_sequence: int | None = None
    last_event: str | None = None
    last_ts: float | None = None
    attached_mid_run: bool = False
    duplicate_sequences: tuple[int, ...] = ()

    @property
    def complete(self) -> bool:
        """录到了结束事件。**不等于结果已确认**（见 :attr:`outcome`）。"""
        return self.finish_reason is not None

    @property
    def outcome(self) -> str:
        if self.finish_reason is not None:
            return outcome_of_reason(self.finish_reason)
        return OUTCOME_INTERRUPTED

    @property
    def outcome_label(self) -> str:
        return OUTCOME_LABELS.get(self.outcome, self.outcome)

    @property
    def last_reliable_step(self) -> str:
        """最后一条**可信**记录：中断局从这里往后就没有证据了。"""
        if not self.last_event:
            return "-"
        if self.last_sequence is None:
            return self.last_event
        return f"{self.last_event}#{self.last_sequence}"

    def as_dict(self) -> dict[str, Any]:
        return {
            'session': self.session,
            'complete': self.complete,
            'outcome': self.outcome,
            'start_reason': self.start_reason,
            'finish_reason': self.finish_reason,
            'started_ts': self.started_ts,
            'finished_ts': self.finished_ts,
            'event_count': self.event_count,
            'last_sequence': self.last_sequence,
            'last_event': self.last_event,
            'attached_mid_run': self.attached_mid_run,
            'duplicate_sequences': list(self.duplicate_sequences),
        }


@dataclass
class RunLogIndex:
    path: str | None = None
    total_lines: int = 0
    parsed_events: int = 0
    sessions: list[SessionSummary] = field(default_factory=list)
    bad_lines: list[BadLine] = field(default_factory=list)
    truncated_tail: bool = False

    @property
    def complete_sessions(self) -> list[SessionSummary]:
        return [item for item in self.sessions if item.complete]

    @property
    def partial_sessions(self) -> list[SessionSummary]:
        """有开局、没结束的会话。**不是垃圾数据**——它是"这一局没录完"的证据。"""
        return [item for item in self.sessions if not item.complete]

    def find(self, session: str) -> SessionSummary | None:
        for item in self.sessions:
            if item.session == session:
                return item
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            'path': self.path,
            'total_lines': self.total_lines,
            'parsed_events': self.parsed_events,
            'truncated_tail': self.truncated_tail,
            'bad_lines': [
                {'line_no': bad.line_no, 'error': bad.error, 'preview': bad.preview}
                for bad in self.bad_lines
            ],
            'complete': len(self.complete_sessions),
            'partial': len(self.partial_sessions),
            'sessions': [item.as_dict() for item in self.sessions],
        }

    def format_table(self, *, limit: int | None = None) -> str:
        """给终端看的一览。中断局排在后面但**不会被藏起来**。"""
        ordered = self.complete_sessions + self.partial_sessions
        if limit is not None and limit > 0:
            ordered = ordered[-limit:]
        lines = [
            f'会话                 状态      开始         结束            事件  最后可信步骤            附加',
            '-' * 108,
        ]
        for item in ordered:
            lines.append(
                ' '.join(
                    [
                        f"{item.session:<20}",
                        f"{'完整' if item.complete else '中断':<6}",
                        f"{(item.start_reason or '-'):<12}",
                        f"{(item.finish_reason or '-'):<14}",
                        f"{item.event_count:>4}",
                        f"{item.last_reliable_step:<22}",
                        ('中途接入 ' if item.attached_mid_run else '')
                        + (f'重复序号{item.duplicate_sequences} ' if item.duplicate_sequences else '')
                        + item.outcome_label,
                    ]
                )
            )
        lines.append(
            f'合计 {len(self.sessions)} 局：完整 {len(self.complete_sessions)}、'
            f'中断 {len(self.partial_sessions)}；坏行 {len(self.bad_lines)} 处'
            + ('（末尾被截断）' if self.truncated_tail else '')
        )
        return '\n'.join(lines)


def _read_lines(path: Path) -> tuple[list[tuple[int, str]], bool]:
    """读原始行，保留行号。返回 (行列表, 是否末尾截断)。

    末尾截断 = 最后一行既没换行符、又解析失败。这是"写到一半被杀"的典型形状，
    和"中间某行写坏"要分开报。
    """
    raw = path.read_bytes()
    text = raw.decode('utf-8', errors='replace')
    truncated = bool(text) and not text.endswith('\n')
    return list(enumerate(text.splitlines(), start=1)), truncated


def index_run_log(path: str | Path) -> RunLogIndex:
    """扫一遍记录文件，给出全部会话（含中断局）与坏行位置。只读。"""
    target = Path(path)
    index = RunLogIndex(path=str(target))
    if not target.exists():
        return index
    lines, truncated_hint = _read_lines(target)
    index.total_lines = len(lines)

    sessions: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    seen_sequences: dict[str, set[int]] = {}

    for line_no, line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            index.bad_lines.append(
                BadLine(line_no=line_no, error=f'{type(exc).__name__}: {exc.msg}', preview=text[:60])
            )
            continue
        if not isinstance(payload, dict):
            continue
        index.parsed_events += 1
        session = payload.get('session')
        if not session:
            continue
        session = str(session)
        if session not in sessions:
            sessions[session] = {
                'start_reason': None,
                'finish_reason': None,
                'started_ts': None,
                'finished_ts': None,
                'event_count': 0,
                'last_sequence': None,
                'last_event': None,
                'last_ts': None,
                'attached_mid_run': False,
                'duplicates': [],
            }
            order.append(session)
            seen_sequences[session] = set()
        entry = sessions[session]
        event = str(payload.get('event') or '')
        sequence = payload.get('sequence')
        ts = payload.get('ts')
        entry['event_count'] += 1
        entry['last_event'] = event or entry['last_event']
        if isinstance(sequence, int):
            if sequence in seen_sequences[session]:
                entry['duplicates'].append(sequence)
            seen_sequences[session].add(sequence)
            entry['last_sequence'] = sequence
        if isinstance(ts, (int, float)):
            entry['last_ts'] = float(ts)
        if event == 'run_started':
            body = payload.get('payload') or {}
            entry['start_reason'] = body.get('reason')
            entry['attached_mid_run'] = bool(body.get('attached_mid_run'))
            if isinstance(ts, (int, float)):
                entry['started_ts'] = float(ts)
        elif event == 'run_finished':
            body = payload.get('payload') or {}
            entry['finish_reason'] = body.get('reason')
            if isinstance(ts, (int, float)):
                entry['finished_ts'] = float(ts)

    for session in order:
        entry = sessions[session]
        index.sessions.append(
            SessionSummary(
                session=session,
                start_reason=entry['start_reason'],
                finish_reason=entry['finish_reason'],
                started_ts=entry['started_ts'],
                finished_ts=entry['finished_ts'],
                event_count=int(entry['event_count']),
                last_sequence=entry['last_sequence'],
                last_event=entry['last_event'],
                last_ts=entry['last_ts'],
                attached_mid_run=bool(entry['attached_mid_run']),
                duplicate_sequences=tuple(sorted(set(entry['duplicates']))),
            )
        )

    # 末尾截断只在最后一行确实解析失败时才算——文件正常结束不会有这个标记。
    index.truncated_tail = bool(
        truncated_hint and index.bad_lines and index.bad_lines[-1].line_no == index.total_lines
    )
    return index


def iter_sessions(path: str | Path) -> Iterable[SessionSummary]:
    yield from index_run_log(path).sessions
