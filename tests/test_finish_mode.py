"""收尾判定：调色目标凑不满时，该不该用强制调色。

实机场景（2026-11-05，用户当场提出的决策）：
纵向 · 所有颜料 · 2/4，纵向那条线上已经没有能落子的格，盘上还有一枚
Lv7 宝箱。这时候「强制调色」是唯一的「凑满」手段，但它会**随机**摧毁
1~3 个颜料（官方 `config_s8_gameplay_revive` 1001/2001/3001），可能把
Lv7 一起炸掉——而 Lv7 是 4005 权重表里 1% 的产物，丢了再也刷不出来。

所以这条链路的正确输出是：**判成凑不满 → 明确劝阻强制调色 → 给出随便塞完
+ 完成绘制**，而不是继续按分数推荐落点。
"""

from __future__ import annotations

import unittest

from huahua_assistant import game_model as gm
from huahua_assistant.models import (
    BoardObservation,
    CurrentPaintObservation,
    PaintCategory,
    PaintDirection,
    PaintObject,
    Rect,
    TileObservation,
    TileState,
)


def _blank_tiles() -> list[gm.Tile]:
    return [gm.Tile(TileState.BLANK) for _ in range(gm.BOARD_SIZE * gm.BOARD_SIZE)]


def _vertical_rule(match_current: int) -> gm.CurrentPaintRule:
    return gm.CurrentPaintRule(
        paint=gm.Paint(PaintCategory.GAMEPLAY, 3),
        direction=PaintDirection.VERTICAL,
        target_object=PaintObject.ALL,
        match_required=4,
        upgrade_amount=1,
        match_current=match_current,
    )


def _state_with_full_column(match_current: int = 2) -> gm.PuzzleState:
    """复刻实机那一局：纵向第 4 列塞满，只剩顶部一格是虚空。"""
    tiles = _blank_tiles()
    column = 3
    filled = {
        1: gm.Paint(PaintCategory.GAMEPLAY, 4, subtype="宝箱", is_treasure=True),
        2: gm.Paint(PaintCategory.GAMEPLAY, 3),
        3: gm.Paint(PaintCategory.GAMEPLAY, 4),
        4: gm.Paint(PaintCategory.GAMEPLAY, 7, subtype="宝箱", is_treasure=True),
        5: gm.Paint(PaintCategory.MONSTER, 4),
    }
    for row, paint in filled.items():
        tiles[row * gm.BOARD_SIZE + column] = gm.Tile(TileState.PAINT, paint)
    tiles[0 * gm.BOARD_SIZE + column] = gm.Tile(TileState.VOID)
    return gm.PuzzleState(
        tiles=tuple(tiles),
        current_rule=_vertical_rule(match_current),
        current_anchor=(4, column),
        backpack=(gm.Paint(PaintCategory.GAMEPLAY, 2),) * 3,
    )


class FinishModeTests(unittest.TestCase):
    def test_full_column_with_seven_chest_is_unreachable(self) -> None:
        advice = gm.assess_finish_mode(_state_with_full_column())
        self.assertIsNotNone(advice)
        assert advice is not None
        self.assertFalse(advice.reachable)
        self.assertEqual((advice.current, advice.required), (2, 4))
        self.assertEqual(advice.legal_cells, 0)

    def test_advice_names_the_seven_chest_and_forbids_force_complete(self) -> None:
        advice = gm.assess_finish_mode(_state_with_full_column())
        assert advice is not None
        self.assertIn((4, 3, 7), advice.treasures)
        joined = advice.summary
        self.assertIn("不要用强制调色", joined)
        # 必须点名 7 级那一格，玩家才知道风险具体落在哪。
        self.assertIn("r5c4", joined)

    def test_advice_offers_filler_cells_and_finish_drawing(self) -> None:
        advice = gm.assess_finish_mode(_state_with_full_column())
        assert advice is not None
        self.assertTrue(advice.fillers)
        # 「随便塞」不能塞回调色范围里——那几格本来就是满的。
        self.assertNotIn((0, 3), advice.fillers)
        self.assertIn("完成绘制", advice.summary)

    def test_reachable_goal_is_not_pushed_into_finish_mode(self) -> None:
        state = _state_with_full_column(match_current=2)
        # 把第 4 列清空三格，让目标重新可达。
        tiles = list(state.tiles)
        for row in (2, 3, 5):
            tiles[row * gm.BOARD_SIZE + 3] = gm.Tile(TileState.BLANK)
        state = gm.PuzzleState(
            tiles=tuple(tiles),
            current_rule=state.current_rule,
            current_anchor=state.current_anchor,
            backpack=state.backpack,
        )
        advice = gm.assess_finish_mode(state)
        assert advice is not None
        self.assertTrue(advice.reachable)
        self.assertEqual(advice.legal_cells, 3)
        self.assertNotIn("凑不满", advice.headline)

    def test_zero_progress_stays_pending(self) -> None:
        """进度 0 时才允许判「待落初始颜料」。"""
        state = gm.PuzzleState(
            tiles=tuple(_blank_tiles()),
            current_rule=_vertical_rule(0),
            backpack=(),
            current_rule_pending=True,
        )
        advice = gm.assess_finish_mode(state)
        assert advice is not None
        self.assertTrue(advice.reachable)
        self.assertIn("初始颜料", advice.headline)

    def test_no_rule_returns_nothing(self) -> None:
        state = gm.PuzzleState(tiles=tuple(_blank_tiles()))
        self.assertIsNone(gm.assess_finish_mode(state))


def _observation_board(paints: dict[tuple[int, int], gm.Paint]) -> BoardObservation:
    tiles = []
    for row in range(gm.BOARD_SIZE):
        for col in range(gm.BOARD_SIZE):
            paint = paints.get((row, col))
            tiles.append(
                TileObservation(
                    row=row,
                    col=col,
                    rect=Rect(col * 100, row * 100, 100, 100),
                    state=TileState.PAINT if paint else TileState.BLANK,
                    confidence=0.9,
                    features={},
                    paint_category=paint.category if paint else PaintCategory.UNKNOWN,
                    paint_category_confidence=0.9 if paint else 0.0,
                    paint_level=paint.level if paint else None,
                    paint_level_confidence=0.9 if paint else 0.0,
                    paint_is_treasure=bool(paint and paint.is_treasure),
                )
            )
    return BoardObservation(rect=Rect(0, 0, 600, 600), score=0.9, tiles=tuple(tiles))


class ProgressImpliesAnchorTests(unittest.TestCase):
    """进度 ≥1 时不能再判「待落初始颜料」——否则桌宠会往范围外的空格怼。"""

    def _paint(self) -> gm.CurrentPaintObservation:
        return CurrentPaintObservation(
            panel_rect=Rect(0, 0, 400, 100),
            icon_rect=Rect(0, 0, 40, 40),
            text_rect=Rect(40, 0, 360, 100),
            category=PaintCategory.GAMEPLAY,
            category_confidence=0.9,
            subtype=None,
            is_treasure=False,
            level=3,
            level_confidence=0.9,
            match_current=2,
            match_required=4,
            target_object=PaintObject.ALL,
            direction=PaintDirection.VERTICAL,
            upgrade_amount=1,
            keyword_scores={},
            raw_text_hint="",
        )

    def test_progress_over_zero_is_not_pending(self) -> None:
        board = _observation_board(
            {
                (1, 3): gm.Paint(PaintCategory.GAMEPLAY, 4),
                (4, 3): gm.Paint(PaintCategory.GAMEPLAY, 7, is_treasure=True),
            }
        )
        state = gm.state_from_observation(board, self._paint(), ())
        self.assertFalse(gm.current_rule_is_pending(state))

    def test_anchor_falls_back_to_the_longest_line(self) -> None:
        # 第 3 列两枚、第 5 列一枚：纵向的锚点应该落在第 3 列。
        board = _observation_board(
            {
                (1, 3): gm.Paint(PaintCategory.GAMEPLAY, 4),
                (4, 3): gm.Paint(PaintCategory.GAMEPLAY, 7, is_treasure=True),
                (4, 5): gm.Paint(PaintCategory.MONSTER, 3),
            }
        )
        state = gm.state_from_observation(board, self._paint(), ())
        self.assertIsNotNone(state.current_anchor)
        assert state.current_anchor is not None
        self.assertEqual(state.current_anchor[1], 3)

    def test_zero_progress_is_still_pending(self) -> None:
        board = _observation_board({})
        paint = self._paint()
        paint = gm.CurrentPaintObservation(**{**paint.__dict__, "match_current": 0})
        state = gm.state_from_observation(board, paint, ())
        self.assertTrue(gm.current_rule_is_pending(state))


if __name__ == "__main__":
    unittest.main()
