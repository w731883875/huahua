"""右侧「等级口径 / 模型对账」台账必须真的写得出来。

这条测试是为一个**只在特定数据下才崩**的 bug 准备的：``_write_level_ledger``
原来写的是 ``tile.row``，而模型层的 :class:`game_model.Tile` 根本没有 row/col
字段（坐标就是它在 ``tiles`` 元组里的下标）。全场加成都是 0 时函数提前 return，
所以一直没暴露；一旦某格带「地块+3」就会 AttributeError，整块面板直接不显示。

同一条测试顺带锁住「模型对账」小节：差异要落到具体格子上。
"""

from __future__ import annotations

import tkinter as tk
import unittest

from huahua_assistant.codes import resolve_affix
from huahua_assistant.game_model import (
    BOARD_SIZE,
    CurrentPaintRule,
    Paint,
    PaintCategory,
    PaintDirection,
    PaintObject,
    PuzzleState,
    Tile,
    TileState,
    level_sources,
)
from huahua_assistant.level_reconcile import CellMismatch, ReconcileReport

BRUSH_ONE = resolve_affix("对调色范围内的随机1个颜料额外提升1级")


def _state_with_boost() -> PuzzleState:
    tiles = [Tile(state=TileState.BLANK) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    tiles[2 * BOARD_SIZE + 3] = Tile(
        state=TileState.PAINT,
        paint=Paint(PaintCategory.MONSTER, 2, "鼠人"),
        bonus_level=3,
    )
    tiles[1 * BOARD_SIZE + 2] = Tile(
        state=TileState.PAINT,
        paint=Paint(PaintCategory.MONSTER, 2, "矮人"),
        aura_level=2,
    )
    return PuzzleState(tiles=tuple(tiles))


def _state_with_sources() -> PuzzleState:
    """锚点 r2c4、纵向范围，r3c4 有一枚玩法 Lv4 —— 调色 +1、笔触 +1 都吃。"""
    tiles = [Tile(state=TileState.BLANK) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    tiles[2 * BOARD_SIZE + 3] = Tile(
        state=TileState.PAINT,
        paint=Paint(PaintCategory.GAMEPLAY, 4, "赛季"),
    )
    rule = CurrentPaintRule(
        paint=Paint(PaintCategory.MONSTER, 5, "伊吉"),
        direction=PaintDirection.VERTICAL,
        target_object=PaintObject.GAMEPLAY,
        match_required=4,
        upgrade_amount=1,
    )
    return PuzzleState(
        tiles=tuple(tiles),
        current_rule=rule,
        current_anchor=(1, 3),
        active_affix=BRUSH_ONE,
    )


class LevelLedgerTests(unittest.TestCase):
    def test_ledger_lists_boosted_cells_without_crashing(self) -> None:
        from huahua_assistant.app import HuahuaApp

        app = HuahuaApp()
        try:
            app.update()
            app.current_state = _state_with_boost()
            app.level_reconcile = ReconcileReport(note="本帧没有新的落笔，无对账")

            app.region_text.configure(state=tk.NORMAL)
            app.region_text.delete("1.0", tk.END)
            app._write_level_ledger()
            app.region_text.configure(state=tk.DISABLED)
            text = app.region_text.get("1.0", tk.END)
        finally:
            app.destroy()

        self.assertIn("等级口径", text)
        # r3c4：base 2 + 地块3 → 5；r2c3：base 2 + 光环2 → 4
        self.assertIn("r3c4 Lv2 + 地块+3 → 5", text)
        self.assertIn("r2c3 Lv2 + 光环+2 → 4", text)
        self.assertIn("模型对账", text)

    def test_mismatch_lines_name_the_cells(self) -> None:
        from huahua_assistant.app import HuahuaApp

        app = HuahuaApp()
        try:
            app.update()
            app.current_state = _state_with_boost()
            app.level_reconcile = ReconcileReport(
                ran=True,
                installed=((4, 0),),
                mismatches=(
                    CellMismatch(row=4, col=0, predicted=5, observed=7, kind="new"),
                    CellMismatch(row=2, col=3, predicted=5, observed=3),
                ),
                levels=((4, 0, 5, 7), (2, 3, 5, 3)),
            )
            app.region_text.configure(state=tk.NORMAL)
            app.region_text.delete("1.0", tk.END)
            app._write_level_ledger()
            app.region_text.configure(state=tk.DISABLED)
            text = app.region_text.get("1.0", tk.END)
        finally:
            app.destroy()

        self.assertIn("2 格对不上", text)
        self.assertIn("r5c1（新落笔） 模型5 ≠ 画面7（差+2）", text)
        self.assertIn("r3c4 模型5 ≠ 画面3（差-2）", text)

    def test_level_sources_are_written_out_with_targets(self) -> None:
        """三个来源必须分开列：各自的目标类别、范围、级数都要出现在台账里。"""
        from huahua_assistant.app import HuahuaApp

        app = HuahuaApp()
        try:
            app.update()
            app.current_state = _state_with_sources()
            app.selected_affix = BRUSH_ONE
            app.level_reconcile = ReconcileReport(note="本帧没有新的落笔，无对账")
            app.region_text.configure(state=tk.NORMAL)
            app.region_text.delete("1.0", tk.END)
            app._write_level_ledger()
            app.region_text.configure(state=tk.DISABLED)
            text = app.region_text.get("1.0", tk.END)
        finally:
            app.destroy()

        self.assertIn("这一笔的等级来源", text)
        self.assertIn("调色效果：目标=玩法颜料", text)
        self.assertIn("笔触效果：目标=所有颜料", text)
        # 玩法 Lv4：调色 +1 与笔触 +1 都吃 → 6
        self.assertIn("r3c4", text)
        self.assertIn("调色效果1 + 笔触效果1 → Lv6", text)

    def test_reconcile_section_prints_source_table(self) -> None:
        from huahua_assistant.app import HuahuaApp

        state = _state_with_sources()
        report = ReconcileReport(
            ran=True,
            installed=((2, 3),),
            mismatches=(),
            levels=((2, 3, 6, 6),),
            sources=level_sources(state, affix=BRUSH_ONE),
        )
        app = HuahuaApp()
        try:
            app.update()
            app.current_state = state
            app.level_reconcile = report
            app.region_text.configure(state=tk.NORMAL)
            app.region_text.delete("1.0", tk.END)
            app._write_reconcile_lines()
            app.region_text.configure(state=tk.DISABLED)
            text = app.region_text.get("1.0", tk.END)
        finally:
            app.destroy()

        self.assertIn("来源（分开算，各自带范围与目标）", text)
        self.assertIn("调色效果：目标=玩法颜料 范围=5格 范围内全部 +1", text)
        self.assertIn("笔触效果：目标=所有颜料 范围=5格 范围内随机 1 个 +1", text)


if __name__ == "__main__":
    unittest.main()
