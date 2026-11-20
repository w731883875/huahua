"""提示词与输出解析。

模型要回答的是一个固定 schema 的 JSON，不写散文：

```json
{"action": "place", "row": 1, "col": 3, "option": null, "reason": "范围收益高"}
```

``parse_advice`` 是纯函数，任何机器都能测，不依赖网络。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .types import (
    ACTION_BACKPACK,
    ACTION_BRUSH1,
    ACTION_BRUSH2,
    ACTION_FINISH,
    ACTION_PLACE,
    ACTION_UNKNOWN,
    AdvisorAdvice,
)

VALID_ACTIONS = {
    ACTION_PLACE,
    ACTION_BACKPACK,
    ACTION_BRUSH1,
    ACTION_BRUSH2,
    ACTION_FINISH,
}

SYSTEM_PROMPT = """你是《火炬之光：无限》S8「画中雪原 · 作画」玩法的实时教练。

# 玩法
6x6 棋盘。你要根据画面判断当前处于哪个阶段，并指出下一步该点哪里。

阶段与动作的对应：
1. 选笔触方向：屏幕上两张卡写着【转锋】【堆砌】【扩散】之一 → action=brush_stage1，option 填卡片名（转锋/堆砌/扩散）。
2. 选笔触效果：两张卡是具体词缀描述（如「使相邻的怪物颜料提升1级」）→ action=brush_stage2，option 填那张卡的完整描述。
3. 放颜料：棋盘上有高亮可放置的格子、右侧显示「当前颜料」→ action=place，row/col 填格子的行列（0 开始，左上角是 0,0）。
4. 选背包颜料：底部背包高亮、需要挑一枚 → action=backpack，option 填第几格（从 1 开始）。
5. 入画：画面下方出现一个大的蓝色「入画」按钮（按钮上方写着「可入画次数 n/6」）→ action=finish。

# 判断顺序（重要）
**先看参考信息，再看按钮，最后数棋盘。** 依次确认：
- 参考信息里说「本地已检测到『入画』按钮」→ 直接 finish，不要再选笔触、不要再看棋盘。
- 自己看到「入画」按钮（下方一块大蓝色按钮，上方写着「可入画次数 n/6」）→ finish。
- 看到两张带词缀描述的卡 → brush_stage2。
- 看到【转锋】【堆砌】【扩散】这类方向名 → brush_stage1。
- 看到「当前颜料」和高亮空格 → place。
- 以上都没有 → action=unknown，不要硬猜。

# place 的坐标
参考信息里如果列出了「可以放颜料的高亮格」，**row/col 必须从那个列表里挑**，
不要自己目测。模型在缩略图上数格子经常数错（实测会把空地说成高亮格）。

# 决策偏好
- 优先把颜料放到能同时满足「调色目标」和吃到「调色效果」加成的位置。
- 能让某个颜料冲到 7 级的位置优先（7 级是这个玩法的核心收益）。
- 同等条件下选作用范围内空格最多的格子。
- 拿不准就选最稳妥的（能推进目标计数的），不要为了收益去赌。

# 输出
只输出一个 JSON 对象，不要解释、不要 markdown 代码块：
{"action":"place","row":1,"col":3,"option":null,"reason":"一句话理由，20 字以内"}

row/col 只对 place 有意义，其它动作用 null。option 只对笔锋和背包有意义。
看不懂画面就返回 {"action":"unknown","row":null,"col":null,"option":null,"reason":"原因"}。
"""


def build_user_content(context: str) -> str:
    """把本地链路已经知道的信息作为文字上下文附在图片后面。

    这些信息不是必须的，但能让模型少猜——尤其棋盘坐标，模型自己数格子
    经常数错，而我们已经知道每格的位置。
    """
    if not context.strip():
        return "这是游戏当前画面。请判断阶段并给出下一步动作。"
    return (
        "这是游戏当前画面。本地识别链路给出的参考信息如下"
        "（可能有错，以你看到的画面为准）：\n"
        f"{context.strip()}\n\n"
        "请判断阶段并给出下一步动作。"
    )


_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出第一个 JSON 对象。

    模型经常在前后加 ```json 或一句「好的，我的建议是：」，所以不能直接 loads。
    """
    if not text:
        return None
    stripped = text.strip()
    try:
        loaded = json.loads(stripped)
        if isinstance(loaded, dict):
            return loaded
    except ValueError:
        pass
    for match in _JSON_RE.finditer(stripped):
        try:
            loaded = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = re.sub(r"[^0-9-]", "", value)
        if digits in ("", "-"):
            return None
        try:
            return int(digits)
        except ValueError:
            return None
    return None


def parse_advice(text: str, *, elapsed: float = 0.0) -> AdvisorAdvice:
    """把模型输出解析成 :class:`AdvisorAdvice`（纯函数）。"""
    payload = _extract_json(text)
    if payload is None:
        return AdvisorAdvice(
            ok=False,
            raw=text,
            elapsed=elapsed,
            error="模型没有返回 JSON",
        )

    action = str(payload.get("action") or "").strip().lower()
    if action not in VALID_ACTIONS:
        action = ACTION_UNKNOWN

    row = _to_int(payload.get("row"))
    col = _to_int(payload.get("col"))
    option = payload.get("option")
    option_text = str(option).strip() if isinstance(option, (str, int, float)) else None
    if option_text in ("", "null", "None"):
        option_text = None

    reason = str(payload.get("reason") or "").strip()

    if action == ACTION_PLACE and (row is None or col is None):
        return AdvisorAdvice(
            ok=False,
            action=ACTION_UNKNOWN,
            reason=reason,
            raw=text,
            elapsed=elapsed,
            error="place 动作缺少 row/col",
        )
    if row is not None and not (0 <= row < 6):
        return AdvisorAdvice(
            ok=False, action=ACTION_UNKNOWN, reason=reason, raw=text, elapsed=elapsed,
            error=f"row 越界：{row}",
        )
    if col is not None and not (0 <= col < 6):
        return AdvisorAdvice(
            ok=False, action=ACTION_UNKNOWN, reason=reason, raw=text, elapsed=elapsed,
            error=f"col 越界：{col}",
        )

    return AdvisorAdvice(
        ok=action != ACTION_UNKNOWN,
        action=action,
        row=row,
        col=col,
        option=option_text,
        reason=reason,
        raw=text,
        elapsed=elapsed,
        error=None if action != ACTION_UNKNOWN else "模型判定画面无法识别",
    )
