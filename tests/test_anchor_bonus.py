"""初始颜料也是颜料：能吃到**其他**初始颜料的调色/笔触加成。

用户 2026-11-22 口径：「初始颜料也是颜料，所以能吃到其他初始颜料的调色效果和
笔触效果加成」；并且加成是**装到地块之后持续生效**的。

这条规则的直接后果：锚点那格**棋盘显示等级会高于面板等级**。面板给的是颜料自己的
等级，棋盘角标是最终等级（颜料等级 + 地块加成 + 光环）。

2026-11-22 实机样本 r1c6：3 级赛季颜料 + 「对全图中与自身内容相同的颜料额外提升
3 级」→ 画面 6 级、点击地块加成 +3，而面板仍是 3 级。

旧实现 ``paint = rule.paint`` 用面板值顶掉实测值，那 3 级当场丢失，对账永远对不上。
"""

from __future__ import annotations

import unittest

from huahua_assistant.game_model import (
    CurrentPaintObservation,
    PaintCategory,
    PaintDirection,
    PaintObject,
    TileState,
    state_from_observation,
    tile_bonus_level,
    tile_final_level,
)
from huahua_assistant.models import BoardObservation, Rect, TileObservation

SIZE = 6
ANCHOR = (0, 5)  # r1c6
_EMPTY = Rect(0, 0, 10, 10)


def _empty_tile(row: int, col: int) -> TileObservation:
    return TileObservation(
        row=row,
        col=col,
        rect=_EMPTY,
        state=TileState.BLANK,
        confidence=0.9,
        features={},
        paint_category=None,
        paint_category_confidence=0.0,
        paint_level=None,
        paint_level_confidence=0.0,
        paint_subtype=None,
        paint_is_treasure=False,
        is_scope_highlight=False,
        scope_highlight_confidence=0.0,
        is_current_anchor=False,
        current_anchor_confidence=0.0,
    )


def _anchor_tile(observed_level: int, *, subtype: str = "赛季") -> TileObservation:
    return TileObservation(
        row=ANCHOR[0],
        col=ANCHOR[1],
        rect=_EMPTY,
        state=TileState.PAINT,
        confidence=0.95,
        features={},
        paint_category=PaintCategory.GAMEPLAY,
        paint_category_confidence=0.95,
        paint_level=observed_level,
        paint_level_confidence=0.9,
        paint_subtype=subtype,
        paint_is_treasure=False,
        is_scope_highlight=False,
        scope_highlight_confidence=0.0,
        is_current_anchor=True,
        current_anchor_confidence=0.95,
    )


def _panel(level: int = 3) -> CurrentPaintObservation:
    return CurrentPaintObservation(
        panel_rect=Rect(0, 0, 200, 60),
        icon_rect=Rect(0, 0, 40, 40),
        text_rect=Rect(40, 0, 160, 40),
        category=PaintCategory.GAMEPLAY,
        category_confidence=0.9,
        subtype="赛季",
        is_treasure=False,
        level=level,
        level_confidence=0.95,
        match_current=0,
        match_required=4,
        target_object=PaintObject.GAMEPLAY,
        direction=PaintDirection.HORIZONTAL,
        upgrade_amount=1,
        keyword_scores={},
        raw_text_hint="",
    )


def _state_for(anchor_level: int):
    tiles = [
        _anchor_tile(anchor_level) if (r, c) == ANCHOR else _empty_tile(r, c)
        for r in range(SIZE)
        for c in range(SIZE)
    ]
    board = BoardObservation(rect=Rect(0, 0, 60, 60), tiles=tuple(tiles), score=0.9)
    return state_from_observation(board, _panel(), (), previous=None)


class AnchorReceivesOtherBonusesTests(unittest.TestCase):
    def test_anchor_surplus_over_panel_becomes_tile_bonus(self) -> None:
        """画面 6 级、面板 3 级 → 颜料 3 级 + 地块加成 3 级，最终仍是 6。"""

        state = _state_for(6)
        tile = state.tile(*ANCHOR)

        self.assertEqual(3, tile.paint.level, "颜料等级取面板值")
        self.assertEqual(3, tile_bonus_level(tile), "超出的 3 级记在地块上")
        self.assertEqual(6, tile_final_level(tile), "最终等级必须和画面一致")

    def test_anchor_without_extra_bonus_stays_at_panel_level(self) -> None:
        state = _state_for(3)
        tile = state.tile(*ANCHOR)

        self.assertEqual(3, tile.paint.level)
        self.assertEqual(0, tile_bonus_level(tile))
        self.assertEqual(3, tile_final_level(tile))

    def test_anchor_is_never_in_its_own_scope_but_others_are(self) -> None:
        """锚点不在自己作用范围内，但其他格子照常 —— 这样别的初始颜料才吃得到加成。"""

        from huahua_assistant.game_model import _placement_scope_cells

        state = _state_for(3)
        scope = _placement_scope_cells(state, state.current_rule)

        self.assertNotIn(ANCHOR, scope)
        self.assertIn((0, 0), scope, "同一行其他格子必须在范围内")

    def test_surplus_is_capped_at_max_level(self) -> None:
        """地块加成不能超过 7 级上限。"""

        state = _state_for(7)
        tile = state.tile(*ANCHOR)

        self.assertLessEqual(tile_bonus_level(tile), 7)
        self.assertLessEqual(tile_final_level(tile), 7)


if __name__ == "__main__":
    unittest.main()
