"""逐格对账：模型推算 vs 画面识别。

要证明两件事：
1. 模型能解释的变化**不要**报差异（否则每帧都在报警，用户会学会忽略它）；
2. 模型解释不了的变化**必须**点名到具体格子，并给出「模型以为是多少」。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from dataclasses import replace

from huahua_assistant.game_model import (
    BOARD_SIZE,
    AffixEffect,
    AffixRule,
    BrushType,
    CurrentPaintRule,
    Paint,
    PaintCategory,
    PaintDirection,
    PaintObject,
    PuzzleState,
    Tile,
    TileState,
    _apply_current_rule_upgrade,
    place_current_rule_anchor,
    tile_bonus_level,
    tile_final_level,
)
from huahua_assistant.level_reconcile import (
    CellMismatch,
    ReconcileReport,
    _cover_intermediate,
    installed_cells,
    reconcile,
)

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


class SurfaceMismatchTests(unittest.TestCase):
    """对账差异必须能在界面上看见。

    2026-11-22 用户原话：「日志标了对不上，不在 ui 上给我看？」——之前差异只写进
    右侧「识别区域」的小节和 observations.jsonl，用户根本注意不到。
    """

    @staticmethod
    def _app():
        from huahua_assistant.app import HuahuaApp

        class Stub:
            last_reconcile_notice_key = ""
            status = ""
            notice = ""

            def __init__(self):
                self.status_var = SimpleNamespace(set=lambda v: setattr(self, "status", v))

            def _set_notice(self, message, level="info"):
                self.notice = message
                self.level = level

        stub = Stub()
        stub.surface = lambda report: HuahuaApp._surface_level_mismatch(stub, report)
        return stub

    def test_mismatch_is_pushed_to_status_and_notice(self) -> None:
        stub = self._app()
        report = ReconcileReport(
            ran=True,
            levels=((4, 3, 3, 3),),
            mismatches=(CellMismatch(row=4, col=3, predicted=3, observed=6),),
        )

        stub.surface(report)

        self.assertIn("对账不一致", stub.status)
        self.assertIn("r5c4", stub.notice)  # 对外用 1 基坐标
        self.assertIn("模型3≠画面6", stub.notice)
        self.assertEqual("warning", stub.level)

    def test_same_mismatch_is_not_repeated(self) -> None:
        """同一笔每帧都会重算，逐帧弹提示会刷屏。"""

        stub = self._app()
        report = ReconcileReport(
            ran=True,
            mismatches=(CellMismatch(row=0, col=0, predicted=1, observed=2),),
        )

        stub.surface(report)
        first = stub.notice
        stub.notice = ""
        stub.surface(report)

        self.assertNotEqual("", first)
        self.assertEqual("", stub.notice)

    def test_changed_mismatch_is_reported_again(self) -> None:
        stub = self._app()
        stub.surface(
            ReconcileReport(ran=True, mismatches=(CellMismatch(row=0, col=0, predicted=1, observed=2),))
        )
        stub.notice = ""
        stub.surface(
            ReconcileReport(ran=True, mismatches=(CellMismatch(row=0, col=0, predicted=1, observed=3),))
        )

        self.assertIn("模型1≠画面3", stub.notice)

    def test_clean_reconcile_clears_the_key(self) -> None:
        stub = self._app()
        stub.surface(
            ReconcileReport(ran=True, mismatches=(CellMismatch(row=0, col=0, predicted=1, observed=2),))
        )
        stub.surface(ReconcileReport(ran=True, mismatches=()))

        self.assertEqual("", stub.last_reconcile_notice_key)


class InitialPaintBonusLivesOnTheTileTests(unittest.TestCase):
    """初始颜料的加成落在**地块**上，不跟着颜料走。

    用户 2026-11-22 实机口径（并更正了之前的说法「覆盖会清掉地块 buff」）：

      c4r5 那格原本是 3 级怪物（M3），第三个初始颜料把作用范围内的颜料
      「转变为同等级的随机颜料」，它变成 3 级机械之神；随后吃到第二个初始颜料
      （横向玩法 +2）和第三个（相邻玩法 +1）的加成，最终 6 级。

    关键点：加成加在地块上，所以**换了颜料这 N 级仍然在**。
    此前模型把加成加到 ``paint.level``，颜料一被转换，加成连同旧颜料一起消失 ——
    实测模型 3、画面 6、``breakdown`` 为空，对账天天报红。
    """

    def test_rule_upgrade_writes_tile_bonus_not_paint_level(self) -> None:
        state = _state(
            {(ROW, 1): Paint(PaintCategory.GAMEPLAY, 3, "赛季")},
            upgrade=2,
            match_current=3,
            pending=True,  # 还没落锚点，才能放得下去
        )
        # 落笔那一刻就结算调色，所以直接看放置结果。
        placed = place_current_rule_anchor(state, ROW, 0)
        self.assertTrue(placed.success)

        tile = placed.state.tile(ROW, 1)
        self.assertEqual(3, tile.paint.level)         # 颜料自身等级没被改
        self.assertEqual(2, tile_bonus_level(tile))   # 加成记在地块上
        self.assertEqual(5, tile_final_level(tile))   # 最终等级照样是 5

    def test_tile_bonus_survives_a_paint_being_replaced(self) -> None:
        """加成在地块上：那格换成别的颜料，这 N 级不能丢。"""

        with_bonus = _state(
            {(ROW, 1): Paint(PaintCategory.GAMEPLAY, 3, "赛季")}, match_current=3, pending=True
        )
        after = place_current_rule_anchor(with_bonus, ROW, 0).state
        self.assertEqual(2, tile_bonus_level(after.tile(ROW, 1)))

        # 模拟「这格被覆盖成另一枚 3 级颜料」：只换 paint，地块加成原地不动。
        tile = after.tile(ROW, 1)
        replaced = replace(tile, paint=Paint(PaintCategory.GAMEPLAY, 3, "机械之神"))

        self.assertEqual(2, tile_bonus_level(replaced))
        self.assertEqual(5, tile_final_level(replaced))

    def test_cover_affix_keeps_the_tile_bonus(self) -> None:
        """覆盖类笔触改的是颜料类型，不该顺带把地块加成清零。"""

        previous = _state({(ROW, 1): Paint(PaintCategory.MONSTER, 3, "伊吉")}, match_current=3)
        tiles = list(previous.tiles)
        index = ROW * BOARD_SIZE + 1
        tiles[index] = replace(tiles[index], bonus_level=3)
        previous = replace(previous, tiles=tuple(tiles))

        # 画面上它已经被转成同等级的机械之神。
        current_tiles = list(previous.tiles)
        current_tiles[index] = replace(
            current_tiles[index],
            paint=Paint(PaintCategory.GAMEPLAY, 3, "机械之神"),
            bonus_level=3,
        )
        current = replace(previous, tiles=tuple(current_tiles))

        affix = AffixRule(
            BrushType.TRANSFORM,
            "将作用范围内的颜料转变为同等级的随机颜料",
            AffixEffect.UPGRADE_IN_SCOPE,
            amount=0,
        )
        intermediate = _cover_intermediate(previous, current, previous, previous.current_rule, affix)

        self.assertEqual(3, tile_bonus_level(intermediate.tile(ROW, 1)))
        self.assertEqual(6, tile_final_level(intermediate.tile(ROW, 1)))


class ReadInIsNotANewPlacementTests(unittest.TestCase):
    """「上一帧有颜料但没读出内容」不算新落笔。

    2026-11-22 实测 r1c6：首帧 ``state=P`` 但 ``paint=None``（遮挡/动画中间态），
    第二帧读到 ``G6-梦魇``。旧实现按「有没有读出内容」判定新落笔，于是拿面板等级 3
    去比画面 6，报出一条根本不该存在的 mismatch。
    """

    def test_cell_with_unread_paint_is_not_installed(self) -> None:
        previous = _state({(ROW, 1): Paint(PaintCategory.GAMEPLAY, 3, "赛季")})
        # 同一格：上一帧状态是 PAINT 但内容没读出来
        tiles = list(previous.tiles)
        index = ROW * BOARD_SIZE + 1
        tiles[index] = Tile(state=TileState.PAINT, paint=None)
        previous = replace(previous, tiles=tuple(tiles))

        current = _state({(ROW, 1): Paint(PaintCategory.GAMEPLAY, 6, "梦魇")})

        self.assertEqual((), installed_cells(previous, current))

    def test_truly_blank_cell_is_still_installed(self) -> None:
        previous = _state({})
        current = _state({(ROW, 1): Paint(PaintCategory.GAMEPLAY, 3, "赛季")})

        self.assertEqual(((ROW, 1),), installed_cells(previous, current))


if __name__ == "__main__":
    unittest.main()
