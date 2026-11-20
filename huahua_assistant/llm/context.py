"""把本地链路已经知道的东西写成一段文字，喂给大模型当参考。

为什么要做这件事：模型自己数格子经常数错（6x6 一共 36 格，它在缩略图上数不准），
而棋盘像素坐标、当前颜料的等级/方向/目标、笔锋面板候选这些我们**已经算出来了**。
把这些写成文字附在图片后面，模型只需要做「选择」，不用做「测量」。

这些信息可能错，所以提示词里明确说了「以你看到的画面为准」。
"""

from __future__ import annotations

from typing import Any


def describe_scene(scene: Any, state: Any = None, top_recommendation: Any = None) -> str:
    """返回一段中文描述。``scene`` / ``state`` 都可以是 ``None``，缺什么就少写什么。"""
    lines: list[str] = []

    board = getattr(scene, "board", None)
    if board is not None and getattr(board, "rect", None) is not None:
        rect = board.rect
        lines.append(f"棋盘区域在画面像素 ({rect.x},{rect.y},{rect.w},{rect.h})，每格约 {rect.w / 6:.0f} 像素。")
        lines.append(f"棋盘左上角格子是 (0,0)，行列都从 0 数到 5，右下角是 (5,5)。")
        # 高亮格是本地逐格算出来的，比模型在缩略图上目测准得多。
        # 实测模型会挑一个根本不在高亮范围内的格子（说「高亮空格可放」但坐标是空的），
        # 所以直接把可选集合摊开，让它只在里面选。
        allowed = [
            (tile.row, tile.col)
            for tile in getattr(board, "tiles", ()) or ()
            if getattr(tile, "is_scope_highlight", False)
        ]
        if allowed:
            cells = "、".join(f"({row},{col})" for row, col in sorted(allowed))
            lines.append(f"本地检测到可以放颜料的高亮格：{cells}。place 只能从这些格子里选，别选别的。")

    paint = getattr(scene, "current_paint", None)
    if paint is not None:
        subtype = f"/{paint.subtype}" if getattr(paint, "subtype", None) else ""
        lines.append(
            f"当前颜料：{paint.category.value} Lv{paint.level}{subtype}，"
            f"方向={paint.direction.value}，目标={paint.target_object.value}，"
            f"进度={paint.match_current}/{paint.match_required}。"
        )

    brush = getattr(scene, "brush_options", None)
    if brush is not None:
        stage1 = "、".join(getattr(brush, "stage1_options", ()) or ()) or "未识别"
        stage2 = "、".join(getattr(brush, "stage2_options", ()) or ()) or "未识别"
        lines.append(f"笔锋面板可见：方向候选={stage1}；效果候选={stage2}。")
        chosen = getattr(brush, "chosen_stage1", None)
        if chosen:
            lines.append(f"本地判断已选中的方向={chosen}。")

    regions = getattr(scene, "regions", None) or ()
    if regions:
        names = "、".join(region.kind.value for region in regions)
        lines.append(f"画面上的按钮/区域：{names}。")
        # 「入画」按钮只有 200x40 像素左右，模型在缩略图上经常看不见（实测 1280 宽
        # 时它会说「无入画按钮」）。本地检测到就明确说出来，模型只负责决定要不要点。
        if any(region.kind.value == "finish_drawing" for region in regions):
            lines.append("本地已检测到「入画」按钮（finish_drawing），说明本局可以收工。")

    backpack = getattr(scene, "backpack", None)
    paints = getattr(backpack, "paints", None) or ()
    if paints:
        slots = []
        for index, item in enumerate(paints, start=1):
            category = getattr(getattr(item, "category", None), "value", "unknown")
            level = getattr(item, "level", None)
            selected = "（已选中）" if getattr(item, "is_selected", False) else ""
            slots.append(f"#{index} {category} Lv{level}{selected}")
        lines.append(f"背包共 {len(paints)} 格（按行优先编号，从 1 开始）：" + "；".join(slots) + "。")
        # 背包最后一行格子的 Lv 标签在游戏 UI 里被面板底边裁掉，本地只能靠边框颜色
        # 反推，经常不准。先说清楚，免得模型把它当真值。
        lines.append("注意：背包最后一行格子的等级本地读不准（游戏把 Lv 标签画在面板外了），仅供参考。")

    rule = getattr(state, "current_rule", None)
    if rule is not None and hasattr(rule, "describe"):
        lines.append(f"当前规则：{rule.describe()}。")

    anchor = getattr(state, "current_anchor", None)
    if anchor is not None:
        lines.append(f"锚点格=({anchor[0]},{anchor[1]})。")

    if top_recommendation is not None:
        row = getattr(top_recommendation, "row", None)
        col = getattr(top_recommendation, "col", None)
        if row is not None and col is not None:
            lines.append(f"本地链路首选落点=({row},{col})，仅供参考。")

    return "\n".join(lines)
