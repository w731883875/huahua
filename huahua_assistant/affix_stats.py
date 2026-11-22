"""从运行记录里统计「出现」概率，而不是「选中」概率。

用户 2026-11-22 明确要求的两个口径：

1. **转锋 / 堆砌 / 扩散**（阶段一）各出现多少次；
2. **所有二阶段词缀**各出现多少次，并且按阶段一笔触类型分组。

关键在于「出现」不是「选中」：一次抽卡给两张卡，玩家挑了其中一张，另一张也
算出现过。所以只能读 ``options`` 列表，不能读 ``selected``。

统计口径的实测依据（2026-11-22 22:48 那局 ``run-20261122-144715`` 的完整序列）::

    + 0.0s  brush_stage1_options_observed  ['transform', 'stack']   ← 面板出现，2 张
    + 3.7s  brush_stage1_options_observed  ['stack']                ← 选中后窄化成 1 张

同一个面板会**被观测两次**：出现时两张、选中后一张。窄化那条不是新的一次抽卡，
算进去就会让「被选中的那个」被多数一次，概率系统性偏移。所以本模块只认
``len(options) >= 2`` 的观测，窄化（1 张）一律丢弃。

同理，``affix_refresh_clicked``（刷新）之后出现的词缀是**新的一次抽取**，必须
计入——它们和首次给出的一样，都是服务端从池子里抽的。
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

#: 默认读这份记录。BOM 是实测的：``runtime/auto_run_records.jsonl`` 带 UTF-8 BOM，
#: 用 ``utf-8`` 打开会让第一行的 ``event`` 键名带 ``\ufeff``，于是什么都匹配不上。
DEFAULT_RECORDS = Path("runtime/auto_run_records.jsonl")

#: 阶段一笔触代码 → 中文名。代码表见 :data:`huahua_assistant.codes.BRUSH_CODES`。
BRUSH_NAMES: dict[str, str] = {
    "transform": "转锋",
    "stack": "堆砌",
    "spread": "扩散",
}

#: 抽卡时一次给几张。两个阶段实测都是 2 张（2026-11-22 全库 103 次阶段一观测里
#: 有 20 次是 2 张，其余是窄化后的 1 张；词缀侧 49/50 是 2 张）。
DRAW_SIZE = 2


@dataclass(frozen=True)
class Offer:
    """一次「抽卡出现」：某一屏给出的若干张卡。"""

    kind: str
    """``stage1``（笔触）或 ``stage2``（词缀）。"""
    cards: tuple[str, ...]
    brush: str | None
    """阶段二时，这一屏属于哪个笔触（阶段一已定）。阶段一恒为 ``None``。"""
    session: str
    ts: float


@dataclass
class Tally:
    """一个统计口径的累计结果。"""

    counts: Counter[str] = field(default_factory=Counter)
    draws: int = 0
    """抽了多少次（每屏算一次）。"""
    cards_seen: int = 0
    """一共看到多少张卡 = ``draws * 每屏张数``。"""

    def add(self, cards: Iterable[str]) -> None:
        cards = tuple(cards)
        if not cards:
            return
        self.draws += 1
        self.cards_seen += len(cards)
        self.counts.update(cards)

    def rate(self, name: str) -> float:
        """出现率：该卡出现次数 / 总抽取次数（不是 / 总卡数）。

        因为每屏固定抽 :data:`DRAW_SIZE` 张不重复的卡，所以「某一项出现的概率」
        口径是「这一屏里有没有它」。用 ``draws`` 做分母才与玩家感受一致：
        一屏里几乎不会同时给出两张同名卡。
        """

        return self.counts.get(name, 0) / self.draws if self.draws else 0.0

    def wilson(self, name: str, z: float = 1.96) -> tuple[float, float]:
        """Wilson 区间。小样本下比正态近似靠谱，样本很少时也不会给出负数下界。"""

        if not self.draws:
            return (0.0, 0.0)
        n = self.draws
        p = self.counts.get(name, 0) / n
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))


def _load(records: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not records.is_file():
        return rows
    with records.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # 半行/坏行不该让整份统计失败：记录是边跑边写的，末行常被截断。
                continue
    return rows


def iter_offers(rows: Iterable[dict[str, Any]]) -> Iterator[Offer]:
    """把记录流变成「抽卡出现」流。

    只认**完整的一屏**（卡数 >= :data:`DRAW_SIZE`）：窄化后只剩一张的那条不是
    新抽卡，算进去会让被选中的那张多算一次。
    """

    for row in rows:
        event = row.get("event")
        payload = row.get("payload") or {}
        options = payload.get("options")
        if not isinstance(options, list) or not options:
            continue
        cards = tuple(str(item) for item in options if item)
        if len(cards) < DRAW_SIZE:
            continue
        session = str(row.get("session") or "")
        ts = float(row.get("ts") or 0.0)
        if event == "brush_stage1_options_observed":
            yield Offer(kind="stage1", cards=cards, brush=None, session=session, ts=ts)
        elif event == "affix_choice_observed":
            stage1 = payload.get("stage1_options") or []
            brush = str(stage1[0]) if stage1 else None
            yield Offer(kind="stage2", cards=cards, brush=brush, session=session, ts=ts)


def _dedupe(offers: Iterable[Offer]) -> list[Offer]:
    """去掉「同一屏被连记两次」的重复。

    实测有同一秒内 ``brush_stage1_options_observed`` 与 ``brush_stage1_selected``
    各记一次的情况；``_selected`` 不在 :func:`iter_offers` 里，所以这里只需处理
    同 session 内**紧邻且卡面完全相同**的重复观测（面板停留多帧时会出现）。
    """

    kept: list[Offer] = []
    for offer in offers:
        if kept:
            prev = kept[-1]
            same_screen = (
                prev.session == offer.session
                and prev.kind == offer.kind
                and prev.cards == offer.cards
                and prev.brush == offer.brush
            )
            if same_screen:
                continue
        kept.append(offer)
    return kept


@dataclass
class Report:
    stage1: Tally = field(default_factory=Tally)
    stage2: Tally = field(default_factory=Tally)
    stage2_by_brush: dict[str, Tally] = field(default_factory=dict)
    runs: int = 0
    first_at: float | None = None
    last_at: float | None = None


def build_report(rows: Iterable[dict[str, Any]]) -> Report:
    rows = list(rows)
    offers = _dedupe(iter_offers(rows))
    report = Report()
    report.runs = sum(1 for row in rows if row.get("event") == "run_started")
    for offer in offers:
        if offer.ts:
            report.first_at = offer.ts if report.first_at is None else min(report.first_at, offer.ts)
            report.last_at = offer.ts if report.last_at is None else max(report.last_at, offer.ts)
        if offer.kind == "stage1":
            report.stage1.add(offer.cards)
        else:
            report.stage2.add(offer.cards)
            if offer.brush:
                bucket = report.stage2_by_brush.setdefault(offer.brush, Tally())
                bucket.add(offer.cards)
    return report


def _fmt_time(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _print_tally(
    title: str,
    tally: Tally,
    order: Iterable[str] | None = None,
    labels: dict[str, str] | None = None,
) -> None:
    print(title)
    if not tally.draws:
        print("  （还没有样本）")
        print()
        return
    labels = labels or {}
    names = list(order) if order is not None else [n for n, _ in tally.counts.most_common()]

    def label(name: str) -> str:
        pretty = labels.get(name)
        return f"{pretty}（{name}）" if pretty else name

    print(f"  样本：{tally.draws} 次抽取，共看到 {tally.cards_seen} 张卡")
    print(f"  {'项目':<30}{'出现':>6}{'出现率':>9}   95% 区间")
    for name in names:
        count = tally.counts.get(name, 0)
        low, high = tally.wilson(name)
        print(f"  {label(name):<30}{count:>6}{tally.rate(name) * 100:>8.1f}%   [{low * 100:.1f}%, {high * 100:.1f}%]")
    # 池子里从没出现过的项也要列出来，否则会误以为「池子里没有它」。
    unseen = [n for n in order or () if tally.counts.get(n, 0) == 0]
    if unseen:
        print(f"  尚未出现：{'、'.join(label(n) for n in unseen)}")
    print()


def render(report: Report) -> str:
    import io
    import sys

    buffer = io.StringIO()
    stdout, sys.stdout = sys.stdout, buffer
    try:
        _render(report)
    finally:
        sys.stdout = stdout
    return buffer.getvalue()


def _render(report: Report) -> None:
    print("=" * 78)
    print("画中雪原 · 出现概率统计（出现 ≠ 选中）")
    print("=" * 78)
    print(f"  局数   : {report.runs}")
    print(f"  时间   : {_fmt_time(report.first_at)} ~ {_fmt_time(report.last_at)}")
    print()
    _print_tally(
        "【阶段一】笔触出现概率",
        report.stage1,
        order=BRUSH_NAMES.keys(),
        labels=BRUSH_NAMES,
    )
    _print_tally("【阶段二】词缀出现概率（不分笔触）", report.stage2)
    if report.stage2_by_brush:
        print("-" * 78)
        print("【阶段二】按阶段一笔触分组（同一词缀在不同笔触下概率不同）")
        print("-" * 78)
        print()
        for brush in BRUSH_NAMES:
            tally = report.stage2_by_brush.get(brush)
            if tally is None:
                continue
            _print_tally(f"  ← {BRUSH_NAMES[brush]}（{brush}）", tally)
        unknown = {b: t for b, t in report.stage2_by_brush.items() if b not in BRUSH_NAMES}
        for brush, tally in unknown.items():
            _print_tally(f"  ← 未识别笔触 {brush}", tally)


def _use_utf8_stdout() -> None:
    """Windows 控制台默认是 GBK，直接 print 中文会变问号/乱码。

    用户 2026-11-22 实机看到的就是一堆 ``����ѩԭ``：脚本算对了，是终端编码错了。
    这里把 stdout 包一层 UTF-8，重定向到文件时也不会再坏。
    """

    import io
    import sys

    stream = sys.stdout
    if stream is None:  # pythonw 下没有控制台
        return
    encoding = (getattr(stream, "encoding", "") or "").lower()
    if encoding.replace("-", "") in {"utf8", "utf8mb4"}:
        return
    try:
        sys.stdout = io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # 没有 .buffer（已被包装过）就原样用，别为了编码把统计搞崩。
        pass


def main(argv: list[str] | None = None) -> int:
    import argparse

    _use_utf8_stdout()
    parser = argparse.ArgumentParser(
        prog="affix-stats",
        description="统计画中雪原的阶段一笔触与二阶段词缀「出现」概率。样本随时间累积，不设上限。",
    )
    parser.add_argument(
        "-r",
        "--records",
        default=str(DEFAULT_RECORDS),
        help=f"运行记录路径，默认 {DEFAULT_RECORDS}",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是表格")
    parsed = parser.parse_args(argv)

    report = build_report(_load(Path(parsed.records)))
    if parsed.json:
        payload = {
            "runs": report.runs,
            "stage1": {"draws": report.stage1.draws, "counts": dict(report.stage1.counts)},
            "stage2": {"draws": report.stage2.draws, "counts": dict(report.stage2.counts)},
            "stage2_by_brush": {
                brush: {"draws": tally.draws, "counts": dict(tally.counts)}
                for brush, tally in report.stage2_by_brush.items()
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _render(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
