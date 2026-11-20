"""逐格对账：模型推算 vs 画面识别。

要证明两件事：
1. 模型能解释的变化**不要**报差异（否则每帧都在报警，用户会学会忽略它）；
2. 模型解释不了的变化**必须**点名到具体格子，并给出「模型以为是多少」。
"""

from __future__ import annotations

import unittest

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
)
from huahua_assistant.level_reconcile import installed_cells, reconcile

ROW = 4
PANEL = Paint(PaintCategory.MONSTER, 5, "伊吉")


def _state(
    cells: dict[tuple[int, int], Paint] | None = None,
    *,
    pending: bool = False,
    panel: Paint = PANEL,
    upgrade: int = 2,
    match_current: int = 0,
) -> PuzzleState:
    tiles = [Tile(state=TileState.BLANK) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    for (row, col), paint in (cells or {}).items():
        tiles[row * BOARD_SIZE + col] = Tile(state=TileState.PAINT, paint=paint)
    rule = CurrentPaintRule(
        paint=panel,
        direction=PaintDirection.HORIZONTAL,
        target_object=PaintObject.ALL,
        match_required=4,
        upgrade_amount=upgrade,
        match_current=match_current,
    )
    return PuzzleState(
        tiles=tuple(tiles),
        current_rule=rule,
        current_anchor=None if pending else (ROW, 0),
        current_rule_pending=pending,
        observed_scope_cells=tuple((ROW, col) for col in range(6)),
    )


class ReconcileTests(unittest.TestCase):
    def test_no_new_placement_is_reported_as_skipped(self) -> None:
        state = _state({(ROW, 1): Paint(PaintCategory.GAMEPLAY, 5, "赛季")})
        report = reconcile(state, state)
        self.assertFalse(report.ran)
        self.assertFalse(report.mismatches)

    def test_initial_paint_upgrade_is_explained_by_self_upgrade_only(self) -> None:
        """画面 5→7：面板读 5，笔触「自身提升 2 级」→ 7，能解释。

        注意这里用的是 ``self_upgrade`` 而不是 ``upgrade``（调色）。用户 2026-11-15
        的口径：**初始颜料只吃「使自身等级提升 N 级」这一条**，调色效果的作用范围
        按定义排除锚点，不能拿来解释初始颜料自己涨的级。
        """
        previous = _state({}, pending=True)
        current = _state({(ROW, 0): Paint(PaintCategory.MONSTER, 7, "伊吉")})
        report = reconcile(previous, current, self_upgrade=2)
        self.assertTrue(report.ran)
        self.assertEqual(report.installed, ((ROW, 0),))
        self.assertEqual(report.mismatches, (), report.summary())
        self.assertEqual(report.levels[0][2], 7)

    def test_initial_paint_does_not_get_tint_upgrade(self) -> None:
        """调色 +2 不给初始颜料：面板 5、落笔 5 才是自洽的，判成 7 就是错。"""
        previous = _state({}, pending=True)
        current = _state({(ROW, 0): Paint(PaintCategory.MONSTER, 5, "伊吉")})
        report = reconcile(previous, current)
        self.assertEqual(report.mismatches, (), report.summary())
        self.assertEqual(report.levels[0][2], 5)

    def test_panel_reading_that_does_not_match_board_is_flagged(self) -> None:
        """面板读到 3、画面实际 7：这不是「加成算错」，是**面板读数与落笔不符**。"""
        previous = _state({}, pending=True, panel=Paint(PaintCategory.MONSTER, 3, "伊吉"))
        current = _state({(ROW, 0): Paint(PaintCategory.MONSTER, 7, "伊吉")})
        report = reconcile(previous, current)
        self.assertEqual(len(report.mismatches), 1)
        item = report.mismatches[0]
        self.assertEqual((item.row, item.col, item.kind), (ROW, 0, "new"))
        self.assertEqual((item.predicted, item.observed), (3, 7))
        self.assertEqual(item.delta, 4)

    def test_existing_paints_upgraded_by_rule_are_consistent(self) -> None:
        previous = _state(
            {
                (ROW, 1): Paint(PaintCategory.GAMEPLAY, 3, "赛季"),
                (ROW, 2): Paint(PaintCategory.GAMEPLAY, 3, "赛季"),
            }
        )
        current = _state(
            {
                (ROW, 1): Paint(PaintCategory.GAMEPLAY, 5, "赛季"),
                (ROW, 2): Paint(PaintCategory.GAMEPLAY, 5, "赛季"),
                (ROW, 3): Paint(PaintCategory.MONSTER, 7, "伊吉"),
            }
        )
        report = reconcile(previous, current)
        self.assertTrue(report.ran)
        self.assertEqual(report.mismatches, (), report.summary())

    def test_unexplained_jump_on_existing_cell_is_flagged(self) -> None:
        """已有格子一次涨 4，模型只解释得了 +2 → 必须点名。"""
        previous = _state(
            {
                (ROW, 1): Paint(PaintCategory.GAMEPLAY, 3, "赛季"),
                (ROW, 2): Paint(PaintCategory.GAMEPLAY, 3, "赛季"),
            }
        )
        current = _state(
            {
                (ROW, 1): Paint(PaintCategory.GAMEPLAY, 7, "赛季"),
                (ROW, 2): Paint(PaintCategory.GAMEPLAY, 5, "赛季"),
                (ROW, 3): Paint(PaintCategory.MONSTER, 7, "伊吉"),
            }
        )
        report = reconcile(previous, current)
        self.assertEqual(len(report.mismatches), 1)
        item = report.mismatches[0]
        self.assertEqual((item.row, item.col, item.kind), (ROW, 1, "existing"))
        self.assertEqual((item.predicted, item.observed), (5, 7))

    def test_summary_mentions_cells(self) -> None:
        previous = _state({(ROW, 1): Paint(PaintCategory.GAMEPLAY, 3, "赛季")})
        current = _state(
            {
                (ROW, 1): Paint(PaintCategory.GAMEPLAY, 7, "赛季"),
                (ROW, 3): Paint(PaintCategory.MONSTER, 7, "伊吉"),
            }
        )
        summary = reconcile(previous, current).summary()
        self.assertIn("r5c2", summary)
        self.assertIn("模型5", summary)

    def test_installed_cells_only_reports_new_paints(self) -> None:
        previous = _state({(ROW, 1): Paint(PaintCategory.GAMEPLAY, 3, "赛季")})
        current = _state(
            {
                (ROW, 1): Paint(PaintCategory.GAMEPLAY, 5, "赛季"),
                (ROW, 2): Paint(PaintCategory.MONSTER, 7, "伊吉"),
            }
        )
        self.assertEqual(installed_cells(previous, current), ((ROW, 2),))

    def test_missing_previous_state_is_not_an_error(self) -> None:
        report = reconcile(None, _state({}))
        self.assertFalse(report.ran)
        self.assertTrue(report.note)


if __name__ == "__main__":
    unittest.main()
