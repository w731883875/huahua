"""从 ``runtime/auto_run_records.jsonl`` 复原一局的棋盘演化。

录制器每次棋盘变化都会写一条 ``board_transition``，里面既带**变化前后的签名**
也带**变化后的完整棋盘快照**。所以只要签名字段能首尾相接，这一局就能完整重建：
它记录的不是「推测」，而是每一帧被识别到的真实状态。

用法::

    .venv/Scripts/python.exe -B scripts/replay_auto_run.py            # 最后一局 + 控制台输出
    .venv/Scripts/python.exe -B scripts/replay_auto_run.py --html out.html
    .venv/Scripts/python.exe -B scripts/replay_auto_run.py --index 3  # 倒数第三局
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

#: 允许直接 ``python scripts/replay_auto_run.py`` 跑。不加这一行，脚本目录会被
#: 当成 import 根，`huahua_assistant` 根本找不到。
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROWS = 6
COLS = 6
TILES = ROWS * COLS

STATE_LABEL = {
    "B": "空地",
    "K": "阻挡",
    "V": "虚空",
    "F": "失色",
    "P": "颜料",
    "paint": "颜料",
    "unknown": "未知",
    "?": "未知",
}

CATEGORY_LABEL = {"monster": "怪物", "gameplay": "玩法"}
CATEGORY_COLOR = {"monster": "#e2574c", "gameplay": "#4c9ee2"}
FALLBACK_COLOR = "#b06cd6"


# ---------------------------------------------------------------- 读记录


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def split_runs(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """按 ``run_started`` 切分。同一局的 ``run_finished`` 也归到该局。"""
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for record in records:
        if record.get("event") == "run_started":
            if current:
                runs.append(current)
            current = [record]
            continue
        if current:
            current.append(record)
    if current:
        runs.append(current)
    return runs


def finished_runs(runs: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    return [run for run in runs if run and run[-1].get("event") == "run_finished"]


# ---------------------------------------------------------------- 校验


def verify(run: list[dict[str, Any]]) -> list[str]:
    """检查这一局的快照链是否自洽。返回问题列表，空列表 = 通过。"""
    issues: list[str] = []
    start = next((r for r in run if r.get("event") == "run_started"), None)
    finish = next((r for r in run if r.get("event") == "run_finished"), None)
    if start is None:
        return ["缺少 run_started"]
    if finish is None:
        issues.append("缺少 run_finished：这局没有正常收尾，事件流可能被截断")

    boards: list[tuple[str, dict[str, Any]]] = []
    initial = start["payload"].get("initial_board")
    if not isinstance(initial, dict):
        return ["run_started 里没有 initial_board"]
    boards.append(("接入时", initial))

    for record in run:
        if record.get("event") != "board_transition":
            continue
        payload = record["payload"]
        board = payload.get("board")
        if not isinstance(board, dict):
            issues.append(f"ts={record.get('ts')} 的 board_transition 没有 board 快照")
            continue
        if board.get("signature") != payload.get("after_signature"):
            issues.append(
                f"ts={record.get('ts')} 快照签名 {board.get('signature')} 与 after_signature "
                f"{payload.get('after_signature')} 不一致"
            )
        boards.append(("变化", board))

    for i, (_, board) in enumerate(boards):
        states = board.get("states") or ""
        if len(states) != TILES:
            issues.append(f"第 {i} 个快照 states 长度 {len(states)} != {TILES}")
            continue
        paints = board.get("paints") or []
        expected = int((board.get("terrain_counts") or {}).get("paint", 0) or 0)
        if len(paints) != expected:
            issues.append(f"第 {i} 个快照 paints={len(paints)} 与 terrain_counts.paint={expected} 不一致")
        for paint in paints:
            index = int(paint["row"]) * COLS + int(paint["col"])
            if states[index] != "P":
                issues.append(
                    f"第 {i} 个快照 ({paint['row']},{paint['col']}) 有颜料但格子状态是 {states[index]!r}"
                )

    # 签名链：每条的 before 必须等于上一条的 after（第一条对 initial_board 的签名）
    prev_after = initial.get("signature")
    for record in run:
        if record.get("event") != "board_transition":
            continue
        payload = record["payload"]
        before = payload.get("before_signature")
        after = payload.get("after_signature")
        if before != prev_after:
            issues.append(
                f"签名断链：ts={record.get('ts')} 的 before={before} 与上一条 after={prev_after} 不一致"
            )
        prev_after = after
    if finish is not None and prev_after is not None:
        final = finish["payload"].get("final_board") or {}
        if final.get("signature") and final["signature"] != prev_after:
            issues.append(
                f"final_board 签名 {final['signature']} 与最后一条 transition 的 {prev_after} 不一致"
            )
    return issues


# ---------------------------------------------------------------- 重建步骤


#: 开局时棋盘上自带的颜料块数（与 ``run_recorder.INITIAL_PAINTS_AT_START`` 同义）。
#: 游戏开局就送一块，所以「接入时看到 1 块」不是中途接入。
INITIAL_PAINTS_AT_START = 1


def describe_capture_start(run: list[dict[str, Any]]) -> str:
    """判断这局是不是从开局录起的。

    **不直接信 ``attached_mid_run`` 字段**：2026-11-12 之前的录制器把它写成
    ``reason == "active_scene"``，而 ``active_scene`` 只表示「没看到开始画面」，
    于是每一局都被标成中途接入——包括开局自带 1 块初始颜料的完整对局。
    这里按实际数据重算：看到开始画面、或颜料没超过开局自带数量，就是从开局录起。
    """
    start = next(r for r in run if r.get("event") == "run_started")
    payload = start.get("payload") or {}
    reason = payload.get("reason")
    paints = len((payload.get("initial_board") or {}).get("paints") or [])
    if reason == "start_screen":
        return "从开局录起（看到了开始画面）"
    if paints <= INITIAL_PAINTS_AT_START:
        return f"从开局录起（接入时棋盘只有 {paints} 块初始颜料，那是游戏送的）"
    return f"中途接入（接入时棋盘已有 {paints} 块颜料，开局前几步不在事件流里）"


def _cell_name(row: int, col: int) -> str:
    return f"r{row + 1}c{col + 1}"


def _describe_cell(side: dict[str, Any] | None) -> str:
    if not side:
        return "?"
    state = side.get("state") or "?"
    if state == "paint":
        category = CATEGORY_LABEL.get(side.get("category") or "", side.get("category") or "未知")
        piece = f"{category}"
        if side.get("subtype"):
            piece += f"·{side['subtype']}"
        level = side.get("level")
        bonus = int(side.get("bonus_level") or 0)
        aura = int(side.get("aura_level") or 0)
        if level:
            if bonus or aura:
                # 记录里的 ``level`` 是**生效等级**（画面等级 + 地块加成 + 光环）。
                # 复原时必须把画面那一半还原出来，否则「游戏给的 4 级」和
                # 「我们自己算出来的 4 级」在复盘里长得一模一样。
                base = max(0, int(level) - bonus - aura)
                piece += f" Lv{base}→{level}"
                tags = []
                if bonus:
                    tags.append(f"地块+{bonus}")
                if aura:
                    tags.append(f"光环+{aura}")
                piece += f"（{' + '.join(tags)}）"
            else:
                piece += f" Lv{level}"
        else:
            piece += " Lv?"
        if side.get("treasure"):
            piece += "（宝箱）"
        return piece
    return STATE_LABEL.get(state, state)


def _describe_change(change: dict[str, Any]) -> str:
    before = change.get("from") or {}
    after = change.get("to") or {}
    where = _cell_name(int(change["row"]), int(change["col"]))
    if after.get("state") == "paint" and before.get("state") != "paint":
        verb = "落子"
    elif after.get("state") == "paint" and before.get("state") == "paint":
        before_level = before.get("level")
        after_level = after.get("level")
        if (before.get("category") or None) != (after.get("category") or None):
            verb = "改类"
        elif (before.get("subtype") or None) != (after.get("subtype") or None):
            # 子类换掉（黑帮→灾烬这类）才是实情，不能一律叫「升级」
            verb = "换子类"
        elif (
            isinstance(before_level, int)
            and isinstance(after_level, int)
            and after_level < before_level
        ):
            verb = "降级"
        elif (
            isinstance(before_level, int)
            and isinstance(after_level, int)
            and after_level > before_level
        ):
            verb = "升级"
        else:
            verb = "变更"
    elif after.get("state") == "void":
        verb = "变虚空"
    elif before.get("state") == "void" and after.get("state") != "void":
        verb = "虚空被填"
    elif after.get("state") == "blocked":
        verb = "变阻挡"
    else:
        verb = f"{STATE_LABEL.get(before.get('state'), before.get('state'))}→{STATE_LABEL.get(after.get('state'), after.get('state'))}"
    return f"{where} {verb}：{_describe_cell(before)} → {_describe_cell(after)}"


def _escape_line(text: str) -> str:
    """HTML 里只转义三个必要字符：这段文本里有 > 和 /，不能整体转义掉。"""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def describe_brush_choices(run: list[dict[str, Any]]) -> list[str]:
    """把「笔触阶段一选大类 → 阶段二选词缀」这条链翻成人类可读的时间线。

    日志里每步都记了来源（``captured_click`` / ``card_highlight`` / ``stage2_entry``
    / ``effect_inference``）：点击是确凿的，高亮和效果反推是推断，复盘时必须让人
    一眼看出哪条是哪种，否则「我到底点的是哪张」永远说不清。
    """
    lines: list[str] = []
    start = next((r for r in run if r.get("event") == "run_started"), None)
    t0 = float(start.get("ts") or 0) if start else 0.0
    for record in run:
        event = record.get("event")
        if event not in BRUSH_EVENTS:
            continue
        payload = record.get("payload") or {}
        line = _brush_event_line(event, payload)
        if line is None:
            continue
        offset = float(record.get("ts") or t0) - t0
        lines.append(f"+{max(0.0, offset):5.1f}s  {line}")
    return lines


#: 笔触相关的全部事件名。顺序无意义，只用来筛。
BRUSH_EVENTS = {
    "brush_stage1_options_observed",
    "brush_stage1_card_clicked",
    "brush_stage1_selected",
    "affix_options_observed",
    "affix_options_refreshed",
    "affix_card_clicked",
    "affix_refresh_clicked",
    "affix_choice_observed",
    "affix_choice_resolved",
    "affix_choice_unresolved",
}

SOURCE_LABEL = {
    "captured_click": "抓到你点的位置",
    "card_highlight": "选中高亮",
    "stage2_entry": "进入词缀面板时补记",
    "effect_inference": "看效果反推（非点击）",
}


def _brush_event_line(event: str, payload: dict[str, Any]) -> str | None:
    if event == "brush_stage1_options_observed":
        names = payload.get("names") or payload.get("options") or []
        return f"笔触候选：{' / '.join(str(name) for name in names)}"
    if event == "brush_stage1_card_clicked":
        label = payload.get("selected_name") or payload.get("selected")
        index = payload.get("card_index")
        where = f"（第 {int(index) + 1} 张）" if isinstance(index, int) else ""
        return f"点选笔触大类：{label}{where}"
    if event == "brush_stage1_selected":
        label = payload.get("selected_name") or payload.get("selected") or "（没读到）"
        source = SOURCE_LABEL.get(str(payload.get("source")), str(payload.get("source") or "?"))
        return f"笔触大类落定：{label}　来源={source}"
    if event == "affix_options_observed":
        options = payload.get("options") or []
        return f"词缀候选：{' / '.join(str(option) for option in options)}"
    if event == "affix_options_refreshed":
        before = payload.get("before") or []
        after = payload.get("after") or []
        # 事件本身只是「阶段二候选集变了」，不一定是按了刷新按钮：换一张笔触大类卡
        # 也会让候选整批换。宠物不足 5 级时界面根本没有刷新按钮（用户 2026-11-19 确认，
        # 2 级宠物无此按钮），所以这里不能标成「刷新词缀」。
        return f"词缀候选变化：{' / '.join(before)} → {' / '.join(after)}"
    if event == "affix_card_clicked":
        index = payload.get("card_index")
        where = f"（第 {int(index) + 1} 张）" if isinstance(index, int) else ""
        return f"点选词缀：{payload.get('selected')}{where}"
    if event == "affix_refresh_clicked":
        return "点了刷新词缀（需 5 级宠物，低于该等级不该出现）"
    if event == "affix_choice_observed":
        brush = payload.get("brush_name") or payload.get("brush") or "?"
        options = payload.get("options") or []
        return f"词缀面板收起（{brush}）：{' / '.join(str(option) for option in options)}"
    if event == "affix_choice_resolved":
        code = payload.get("code")
        suffix = f" [{code}]" if code else ""
        source = SOURCE_LABEL.get(str(payload.get("source")), str(payload.get("source") or "?"))
        return f"词缀落定：{payload.get('selected')}{suffix}　来源={source}"
    if event == "affix_choice_unresolved":
        options = payload.get("options") or []
        return f"词缀没定下来：{' / '.join(str(option) for option in options)}"
    return None


def _cell_signature(cell: dict[str, Any] | None) -> tuple | None:
    if not cell:
        return None
    return (
        cell.get("state"),
        cell.get("category"),
        cell.get("level"),
        cell.get("subtype"),
        bool(cell.get("treasure")),
    )


def mark_flip_flops(steps: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """标出「同一格在 1~2 步内又翻回原状」的变化。

    这种几乎都是识别抖动而不是游戏动作：实测本局 ``r1c2`` 在 108 秒里于
    ``颜料 ↔ 虚空`` 之间来回跳了二十多次，每次间隔约 1 秒。抖动会制造假步数，
    不标出来就等于把噪声当棋谱。
    """
    flagged: list[tuple[int, str]] = []
    for index, step in enumerate(steps):
        tags: list[str] = []
        for change in step.get("changes") or []:
            key = (int(change["row"]), int(change["col"]))
            origin = _cell_signature(change.get("from"))
            for later_index in range(index + 1, min(index + 3, len(steps))):
                back = next(
                    (
                        c
                        for c in steps[later_index].get("changes") or []
                        if (int(c["row"]), int(c["col"])) == key
                    ),
                    None,
                )
                if back is None:
                    continue
                if _cell_signature(back.get("to")) == origin:
                    tags.append(
                        f"{_cell_name(*key)} 读数抖动：第 {later_index} 步又回到本步之前的样子，"
                        "多半是等级没读出来/状态瞬时误判，不是游戏里的真实操作"
                    )
                    flagged.append((index, _cell_name(*key)))
                    break
        step["flip_flops"] = tags
    return flagged


def audit_treasure_claims(run: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """找出站不住的「宝箱」判定，返回 ``[(格子名, 理由列表), ...]``。

    两条独立判据，任一命中即存疑：

    * **Lv5**：Lv5 的橙色菱形外框本身金色占比就有 0.232（``runtime/s8_frames/
      diamond_Lv5_橙.png`` 实测，一个没有任何图案的空框都能过旧阈值），所以
      修好之前这一级的「宝箱」几乎必然是颜色兜底误判；
    * **类别翻过**：同一格在同一局里既被判过怪物、又被判过宝箱。升级不会
      改变颜料类别，两种判定同时出现，至少有一次是错的。

    这不是在否定记录，而是把「哪几格别当棋谱看」标出来——终局算出 2 个宝箱
    但实际只投了 0~1 个，通常就是这类格子。
    """
    state: dict[tuple[int, int], dict[str, Any]] = {}

    def touch(cell: dict[str, Any], row: int, col: int) -> None:
        entry = state.setdefault(
            (row, col), {"treasure": False, "monster": False, "levels": set()}
        )
        if cell.get("state") != "paint":
            return
        # 只认「子类真的是宝箱」。旧记录里存在 ``treasure=true`` 却挂着
        # 「黑帮/灾烬」这类怪物子类的脏数据（颜色标志与家族识别各说一套），
        # 拿它当宝箱会把一堆怪物格误算进来。
        if cell.get("subtype") == "宝箱":
            entry["treasure"] = True
            level = cell.get("level")
            if isinstance(level, int):
                entry["levels"].add(level)
        if cell.get("category") == "monster":
            entry["monster"] = True

    start = next((r for r in run if r.get("event") == "run_started"), None)
    if start is not None:
        for paint in (start["payload"].get("initial_board") or {}).get("paints") or []:
            touch(paint, int(paint["row"]), int(paint["col"]))
    for record in run:
        if record.get("event") != "board_transition":
            continue
        for change in record["payload"].get("changes") or []:
            row, col = int(change["row"]), int(change["col"])
            touch(change.get("from") or {}, row, col)
            touch(change.get("to") or {}, row, col)

    findings: list[tuple[str, list[str]]] = []
    for (row, col), entry in state.items():
        if not entry["treasure"]:
            continue
        reasons: list[str] = []
        if 5 in entry["levels"]:
            reasons.append(
                "Lv5 橙色外框本身就占 0.232 金色，颜色兜底会把任何 Lv5 格子判成宝箱"
            )
        if entry["monster"]:
            reasons.append(
                "同一格还判过「怪物」，而升级不会改变颜料类别，至少有一次是错的"
            )
        if reasons:
            findings.append((_cell_name(row, col), reasons))
    return findings


def build_steps(run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = next(r for r in run if r.get("event") == "run_started")
    t0 = float(start.get("ts") or 0)
    steps: list[dict[str, Any]] = [
        {
            "index": 0,
            "offset": 0.0,
            "clock": time.strftime("%H:%M:%S", time.localtime(t0)),
            "board": start["payload"]["initial_board"],
            "changes": [],
            "title": "接入时（录制器启动那一刻的棋盘）",
        }
    ]
    for record in run:
        if record.get("event") != "board_transition":
            continue
        payload = record["payload"]
        changes = payload.get("changes") or []
        ts = float(record.get("ts") or t0)
        labels = [_describe_change(c) for c in changes]
        steps.append(
            {
                "index": len(steps),
                "offset": ts - t0,
                "clock": time.strftime("%H:%M:%S", time.localtime(ts)),
                "board": payload["board"],
                "changes": changes,
                "labels": labels,
                "title": labels[0] if len(labels) == 1 else f"{labels[0]} 等 {len(labels)} 处变化",
            }
        )
    mark_flip_flops(steps)
    return steps


def grid(board: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """把扁平 states + paints 还原成 6x6 的二维表。"""
    states = board.get("states") or ""
    paints = {(int(p["row"]), int(p["col"])): p for p in (board.get("paints") or [])}
    table: list[list[dict[str, Any]]] = []
    for row in range(ROWS):
        line: list[dict[str, Any]] = []
        for col in range(COLS):
            index = row * COLS + col
            char = states[index] if index < len(states) else "?"
            line.append({"char": char, "paint": paints.get((row, col))})
        table.append(line)
    return table


def ascii_board(board: dict[str, Any]) -> str:
    """终端里看的一版：两个字符一格，颜料带等级。"""
    table = grid(board)
    header = "      " + "  ".join(f"c{c + 1}" for c in range(COLS))
    lines = [header]
    for row, cells in enumerate(table):
        parts = []
        for cell in cells:
            paint = cell["paint"]
            if paint is not None:
                level = paint.get("level")
                parts.append(f"P{level if level is not None else '?'}")
            else:
                parts.append(f"{cell['char']} ")
        lines.append(f"  r{row + 1}  " + " ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------- HTML


def _cell_html(cell: dict[str, Any], changed: bool) -> str:
    paint = cell["paint"]
    char = cell["char"]
    if paint is not None:
        category = paint.get("category") or ""
        color = CATEGORY_COLOR.get(category, FALLBACK_COLOR)
        level = paint.get("level")
        level_text = f"Lv{level}" if level is not None else "Lv?"
        subtype = paint.get("subtype") or CATEGORY_LABEL.get(category, category or "?")
        treasure = "★" if paint.get("treasure") else ""
        ring = " box-shadow:0 0 0 2px #ffd84d inset;" if changed else ""
        return (
            f'<div class="cell paint" style="background:{color};{ring}">'
            f'<b>{level_text}</b><span>{treasure}{subtype}</span></div>'
        )
    ring = " box-shadow:0 0 0 2px #ffd84d inset;" if changed else ""
    if char == "V":
        return f'<div class="cell void" style="{ring}"><span>虚空</span></div>'
    if char == "K":
        return f'<div class="cell blocked" style="{ring}"><span>阻挡</span></div>'
    if char == "F":
        return f'<div class="cell faded" style="{ring}"><span>失色</span></div>'
    if char == "?":
        return f'<div class="cell unknown" style="{ring}"><span>?</span></div>'
    return f'<div class="cell blank" style="{ring}"></div>'


def _board_html(board: dict[str, Any], changed_cells: set[tuple[int, int]]) -> str:
    table = grid(board)
    head = "<div class='hd'></div>" + "".join(f"<div class='hd'>c{c + 1}</div>" for c in range(COLS))
    body = ""
    for row, cells in enumerate(table):
        body += f"<div class='hd'>r{row + 1}</div>"
        for col, cell in enumerate(cells):
            body += _cell_html(cell, (row, col) in changed_cells)
    return f"<div class='board' style='grid-template-columns:34px repeat({COLS}, 96px)'>{head}{body}</div>"


# ---------------------------------------------------------------- 交叉校验


def cross_check(steps: list[dict[str, Any]], observations: Path, run: list[dict[str, Any]]) -> dict[str, Any]:
    """拿 ``observations.jsonl`` 里的逐帧地形，反过来验证每个快照真的发生过。

    整局记录的 12 个快照是「变化触发」的，只靠自己证明不了没瞎编；逐帧观测是
    另一条独立通道（每秒一次），如果每个快照的 states 都能在帧流里找到，那这局
    的演化就不是推演出来的。
    """
    t0 = float(run[0].get("ts") or 0)
    t1 = float(run[-1].get("ts") or 0)
    frames: list[tuple[float, str]] = []
    if observations.exists():
        for record in load_records(observations):
            if record.get("event") != "quick_capture_step":
                continue
            ts = float(record.get("ts") or 0)
            if not (t0 - 10 <= ts <= t1 + 10):
                continue
            board = (record.get("payload") or {}).get("board") or {}
            states = board.get("states")
            if states:
                frames.append((ts, states))

    hits = 0
    misses: list[int] = []
    for step in steps:
        states = step["board"].get("states")
        if any(states == s for _, s in frames):
            hits += 1
        else:
            misses.append(step["index"])
    return {
        "available": bool(frames),
        "frames": len(frames),
        "unique": len({s for _, s in frames}),
        "hits": hits,
        "total": len(steps),
        "misses": misses,
    }


def build_html(
    run: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    issues: list[str],
    source: Path,
    cross: dict[str, Any] | None = None,
) -> str:
    start = next(r for r in run if r.get("event") == "run_started")
    finish = next((r for r in run if r.get("event") == "run_finished"), None)
    payload = finish["payload"] if finish else {}
    final_board = payload.get("final_board") or steps[-1]["board"]
    counts = final_board.get("terrain_counts") or {}
    session = start.get("session") or "?"
    t0 = float(start.get("ts") or 0)
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
    duration = payload.get("duration")
    reason = payload.get("reason") or "未收尾"

    status_class = "ok" if not issues else "bad"
    status_text = "快照链自洽，可完整复原" if not issues else f"{len(issues)} 处问题"
    issue_html = "".join(f"<li>{i}</li>" for i in issues)
    cross_chip = ""
    if cross and cross.get("available"):
        cross_chip = (
            f"<div class='chip'>逐帧交叉校验 <b>{cross['hits']}/{cross['total']}</b>"
            f"（{cross['frames']} 帧 / {cross['unique']} 种局面）</div>"
        )
        if not issues and cross["hits"] == cross["total"]:
            status_text = (
                f"快照链自洽，且 {cross['hits']}/{cross['total']} 个快照都能在实时帧流里找到"
            )
        elif cross.get("misses"):
            status_class = "bad"
            status_text += f"；{len(cross['misses'])} 个快照未在帧流出现（步骤 {cross['misses']}）"

    cards = []
    toc = []
    treasure_findings = audit_treasure_claims(run)
    treasure_html = ""
    if treasure_findings:
        items = "".join(
            f"<li><b>{cell}</b>：{'；'.join(reasons)}</li>" for cell, reasons in treasure_findings
        )
        treasure_html = (
            "<div class='verdict bad' style='margin-top:8px'>"
            f"宝箱判定存疑 {len(treasure_findings)} 格（这几格的「宝箱」不是真实的，别当棋谱看）"
            f"<ul>{items}</ul></div>"
        )
    brush_lines = describe_brush_choices(run)
    if brush_lines:
        items = "".join(f"<li>{_escape_line(line)}</li>" for line in brush_lines)
        brush_html = (
            "<div class='verdict info' style='margin-top:8px'>"
            f"笔触选择 {len(brush_lines)} 条（来源里「点击」是确凿的，其余是推断）"
            f"<ul>{items}</ul></div>"
        )
    else:
        brush_html = ""
    for step in steps:
        board = step["board"]
        changed = {(int(c["row"]), int(c["col"])) for c in step.get("changes") or []}
        labels = step.get("labels") or ["初始状态"]
        bad = int((board.get("terrain_counts") or {}).get("paint", 0) or 0)
        items = "".join(f"<li>{label}</li>" for label in labels)
        warns = "".join(f"<p class='warn'>⚠ {tag}</p>" for tag in step.get("flip_flops") or [])
        cards.append(
            f"""
<section class="card" id="step{step['index']}">
  <header>
    <div class="idx">#{step['index']}</div>
    <div class="meta">
      <div class="t">{step['clock']} <span class="off">+{step['offset']:.1f}s</span></div>
      <div class="title">{step['title']}</div>
    </div>
    <div class="stat">颜料 {bad} 块</div>
  </header>
  {_board_html(board, changed)}
  <ul class="changes">{items}</ul>
  {warns}
</section>"""
        )
        toc.append(
            f"<a href='#step{step['index']}'><span>#{step['index']}</span>"
            f"{step['clock']} · {step['title']}</a>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>画中雪原 · 本局复原</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#0b1218; color:#d7e3ee;
         font-family: "Microsoft YaHei UI","PingFang SC",system-ui,sans-serif; }}
  .wrap {{ display:grid; grid-template-columns: 320px 1fr; gap:20px; padding:20px 24px 60px; }}
  header.top {{ grid-column: 1 / -1; background:#101c25; border:1px solid #1d2f3d; border-radius:12px;
                padding:16px 20px; }}
  h1 {{ margin:0 0 6px; font-size:19px; letter-spacing:.5px; }}
  .sub {{ color:#8fa9bd; font-size:13px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
  .chip {{ background:#16242f; border:1px solid #24384a; border-radius:999px; padding:4px 12px;
           font-size:12px; color:#c6d6e2; }}
  .chip b {{ color:#fff; font-weight:600; }}
  .verdict {{ margin-top:12px; padding:10px 14px; border-radius:8px; font-size:13px; }}
  .verdict.ok {{ background:#0f2f24; border:1px solid #1d6b4f; color:#8bffc8; }}
  .verdict.bad {{ background:#311717; border:1px solid #7a2b2b; color:#ffb3b3; }}
  .verdict.info {{ background:#132432; border:1px solid #27506e; color:#bcdcf5; }}
  .verdict ul {{ margin:6px 0 0 18px; padding:0; }}
  nav {{ position:sticky; top:20px; align-self:start; max-height:calc(100vh - 40px); overflow:auto;
         background:#101c25; border:1px solid #1d2f3d; border-radius:12px; padding:10px; }}
  nav h2 {{ font-size:13px; color:#8fa9bd; margin:6px 8px 10px; font-weight:600; }}
  nav a {{ display:block; padding:7px 8px; border-radius:7px; color:#b8cbd9; text-decoration:none;
           font-size:12.5px; line-height:1.45; }}
  nav a:hover {{ background:#172634; color:#fff; }}
  nav a span {{ display:inline-block; min-width:26px; color:#5f7d92; font-variant-numeric:tabular-nums; }}
  .card {{ background:#101c25; border:1px solid #1d2f3d; border-radius:12px; padding:14px 16px 16px;
           margin-bottom:16px; }}
  .card header {{ display:flex; align-items:center; gap:12px; margin-bottom:12px; }}
  .card .idx {{ width:38px; height:38px; border-radius:9px; background:#16242f; color:#7fd4ff;
                display:grid; place-items:center; font-weight:700; }}
  .card .t {{ font-variant-numeric:tabular-nums; font-size:13px; color:#9fb6c6; }}
  .card .off {{ color:#5f7d92; }}
  .card .title {{ font-size:14px; color:#eaf4fb; margin-top:2px; }}
  .card .stat {{ margin-left:auto; font-size:12px; color:#8fa9bd; }}
  .board {{ display:grid; gap:6px; }}
  .hd {{ color:#5f7d92; font-size:11px; display:grid; place-items:center; }}
  .cell {{ height:64px; border-radius:8px; display:flex; flex-direction:column; gap:2px;
           align-items:center; justify-content:center; font-size:11px; color:#0b1218; }}
  .cell b {{ font-size:15px; }}
  .cell span {{ font-size:10.5px; opacity:.92; max-width:88px; overflow:hidden;
                text-overflow:ellipsis; white-space:nowrap; }}
  .cell.void {{ background:#141f29; color:#546a7b; border:1px dashed #26384a; }}
  .cell.blocked {{ background:#2b3137; color:#9aa7b2; }}
  .cell.faded {{ background:#3a3122; color:#d8b878; }}
  .cell.unknown {{ background:#1a2430; color:#7c93a6; }}
  .cell.blank {{ background:#0d1720; border:1px solid #1d2f3d; }}
  .changes {{ margin:12px 0 0; padding-left:18px; color:#a9c0d0; font-size:12.5px; line-height:1.7; }}
  .changes li {{ margin:2px 0; }}
  .warn {{ margin:8px 0 0; padding:7px 10px; border-radius:7px; font-size:12px;
           background:#3a2f12; border:1px solid #6b551d; color:#ffd88a; }}
</style></head>
<body><div class="wrap">
  <header class="top">
    <h1>画中雪原 · 本局复原</h1>
    <div class="sub">来源 {source.name} · 会话 {session} · 开始于 {started}</div>
    <div class="chips">
      <div class="chip">时长 <b>{duration:.1f}s</b></div>
      <div class="chip">步数 <b>{len(steps) - 1}</b></div>
      <div class="chip">事件 <b>{len(run)}</b></div>
      <div class="chip">结束原因 <b>{reason}</b></div>
      <div class="chip">终局颜料 <b>{int(counts.get('paint', 0) or 0)}</b></div>
      <div class="chip">空地 <b>{int(counts.get('blank', 0) or 0)}</b></div>
      <div class="chip">虚空 <b>{int(counts.get('void', 0) or 0)}</b></div>
      {cross_chip}
      <div class="chip">{describe_capture_start(run)}</div>
    </div>
    <div class="verdict {status_class}">{status_text}{f"<ul>{issue_html}</ul>" if issues else ""}</div>
    {treasure_html}
    {brush_html}
  </header>
  <nav><h2>步骤</h2>{''.join(toc)}</nav>
  <main>{''.join(cards)}</main>
</div></body></html>"""


# ---------------------------------------------------------------- 入口


def main() -> None:
    parser = argparse.ArgumentParser(description="复原 auto_run_records.jsonl 里的一局")
    parser.add_argument("--input", default="runtime/auto_run_records.jsonl")
    parser.add_argument("--index", type=int, default=-1, help="第几局（默认最后一局，负数从末尾数）")
    parser.add_argument("--html", default="", help="输出 HTML 路径；留空则自动写到 runtime/replay_<会话>.html")
    parser.add_argument(
        "--observations",
        default="runtime/observations.jsonl",
        help="逐帧观测文件，用来交叉校验快照是否真的发生过；传空字符串可跳过",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印棋盘，只打印摘要")
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="列出全部会话后退出（**含中断局**：一局录到一半也能看见）",
    )
    parser.add_argument(
        "--session",
        default="",
        help="按会话 ID 复原。中断局也能用，看到最后一条可信记录为止",
    )
    args = parser.parse_args()

    path = Path(args.input)

    if args.list_sessions:
        from huahua_assistant.run_index import index_run_log

        index = index_run_log(path)
        print(index.format_table())
        for bad in index.bad_lines[:20]:
            print(f"  坏行 {bad.line_no}：{bad.error} | {bad.preview}")
        if len(index.bad_lines) > 20:
            print(f"  ……另有 {len(index.bad_lines) - 20} 处坏行，见 --json")
        if args.html:
            Path(args.html).write_text(
                json.dumps(index.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
            )
            print(f"已写出：{args.html}")
        return

    records = load_records(path)
    all_runs = split_runs(records)
    if args.session:
        picked = [run for run in all_runs if run and str(run[0].get("session") or "") == args.session]
        if not picked:
            raise SystemExit(f"{path} 里没有会话 {args.session}")
        run = picked[-1]
    else:
        runs = finished_runs(all_runs)
        if not runs:
            raise SystemExit(
                f"{path} 里没有已收尾的完整对局（用 --list-sessions 看中断局，"
                f"或用 --session 指定某局）"
            )
        try:
            run = runs[args.index]
        except IndexError:
            raise SystemExit(f"只有 {len(runs)} 局，索引 {args.index} 越界")

    issues = verify(run)
    steps = build_steps(run)
    start = next(r for r in run if r.get("event") == "run_started")
    finish = next((r for r in run if r.get("event") == "run_finished"), None)
    payload = finish["payload"] if finish else {}
    t0 = float(start.get("ts") or 0)

    print(f"对局：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}")
    print(f"时长：{payload.get('duration', 0):.1f}s  事件 {len(run)} 条  重建 {len(steps) - 1} 步")
    print(f"接入方式：{describe_capture_start(run)}")
    if finish is None:
        # 中断局：没有结束事件，就没有终局快照。只能恢复到"最后一条可信记录"。
        from huahua_assistant.run_index import index_run_log

        summary = index_run_log(path).find(str(run[0].get("session") or ""))
        step_text = summary.last_reliable_step if summary is not None else "未知"
        print(f"状态：中断（没等到结束事件）。最后可信步骤：{step_text}")
        print(f"结束原因：未知——别再猜是重开还是退出，先看最后一条记录停在哪")
    else:
        print(f"结束原因：{payload.get('reason') or '未收尾'}")
    final = payload.get("final_board") or steps[-1]["board"]
    print(f"{'当前' if finish is None else '终局'}地形：{final.get('terrain_counts')}")
    treasure_findings = audit_treasure_claims(run)
    if treasure_findings:
        print()
        print(f"宝箱判定存疑：{len(treasure_findings)} 格（这几格的「宝箱」不是真实的）")
        for cell, reasons in treasure_findings:
            print(f"  - {cell}：{'；'.join(reasons)}")
    brush_lines = describe_brush_choices(run)
    print()
    if brush_lines:
        print(f"笔触选择：{len(brush_lines)} 条")
        for line in brush_lines:
            print(f"  {line}")
    else:
        print("笔触选择：这一局没有记录到（面板没出现过，或记录时还没接上）")
    print()
    if finish is None:
        print("快照链校验：跳过（中断局没有终局快照，接不上是正常的，不代表记录坏了）")
    elif issues:
        print("快照链校验：失败")
        for issue in issues:
            print("  -", issue)
    else:
        print("快照链校验：通过（签名首尾相接，paints 与 states 自洽）")

    cross = None
    if args.observations:
        cross = cross_check(steps, Path(args.observations), run)
        if cross["available"]:
            print(
                f"逐帧交叉校验：{cross['hits']}/{cross['total']} 个快照在 {cross['frames']} 帧实时观测中命中"
                f"（{cross['unique']} 种不同局面）"
            )
            if cross["misses"]:
                print(f"  未命中步骤：{cross['misses']}")
        else:
            print("逐帧交叉校验：跳过（该时间窗内没有逐帧观测）")
    print()

    if not args.quiet:
        for step in steps:
            extra = ""
            if step.get("flip_flops"):
                extra = f"  ⚠ 含 {len(step['flip_flops'])} 处读数抖动"
            print(f"[{step['index']}] {step['clock']}  +{step['offset']:.1f}s  {step['title']}{extra}")
            print(ascii_board(step["board"]))
            print()
        noisy = [(s["index"], tag) for s in steps for tag in s.get("flip_flops") or []]
        if noisy:
            cells = sorted({tag.split(" ")[0] for _, tag in noisy})
            print(f"读数抖动：{len(noisy)} 处，涉及格子 {cells}（这些不是真实操作，别当棋谱看）")
            print()

    html_path = Path(args.html) if args.html else Path("runtime") / f"replay_{time.strftime('%Y%m%d-%H%M%S', time.localtime(t0))}.html"
    html_path.write_text(build_html(run, steps, issues, path, cross), encoding="utf-8")
    print(f"HTML 复盘页：{html_path.resolve()}")


if __name__ == "__main__":
    main()
