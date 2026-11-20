"""「使自身等级提升N级」必须真的作用到落下去的那枚颜料上。

改之前：词缀改的是 ``current_rule.paint``，而落子用的是 ``state.backpack[i]``，
两个对象不联动 —— 选了词缀等于白选（实测落下去还是原等级）。
"""

from __future__ import annotations

import unittest

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
    apply_affix_to_state,
    install_backpack_paint,
    place_current_rule_anchor,
)

ANCHOR_ROW = 4


def _state(*, rule_level: int = 1, pending: bool = False, backpack_level: int = 1) -> PuzzleState:
    tiles = [Tile(state=TileState.BLANK) for _ in range(BOARD_SIZE * BOARD_SIZE)]
    # 第 5 行摆两枚已有颜料，模拟「调色目标已有进度」的盘面。
    tiles[ANCHOR_ROW * BOARD_SIZE + 1] = Tile(
        state=TileState.PAINT, paint=Paint(PaintCategory.GAMEPLAY, 3, "赛季")
    )
    tiles[ANCHOR_ROW * BOARD_SIZE + 2] = Tile(
        state=TileState.PAINT, paint=Paint(PaintCategory.GAMEPLAY, 3, "赛季")
    )
    rule = CurrentPaintRule(
        paint=Paint(PaintCategory.MONSTER, rule_level, "鼠人"),
        direction=PaintDirection.HORIZONTAL,
        target_object=PaintObject.ALL,
        match_required=4,
        upgrade_amount=2,
        match_current=0,
    )
    return PuzzleState(
        tiles=tuple(tiles),
        current_rule=rule,
        backpack=(Paint(PaintCategory.MONSTER, backpack_level, "鼠人"),),
        current_anchor=None if pending else (ANCHOR_ROW, 0),
        current_rule_pending=pending,
        observed_scope_cells=tuple((ANCHOR_ROW, col) for col in range(6)),
    )


SELF_AFFIX = AffixRule(
    BrushType.STACK, "使自身等级提升2级", AffixEffect.UPGRADE_SELF, amount=2
)


class SelfUpgradeAffixTests(unittest.TestCase):
    def test_affix_records_pending_amount(self) -> None:
        result = apply_affix_to_state(_state(), SELF_AFFIX)
        self.assertTrue(result.success)
        self.assertEqual(result.state.self_upgrade_pending, 2)
        # 面板上的当前颜料等级也要跟着涨（用户看得到的那份）
        self.assertEqual(result.state.current_rule.paint.level, 3)

    def test_install_applies_pending_to_the_placed_paint(self) -> None:
        plain = install_backpack_paint(_state(), 0, ANCHOR_ROW, 3)
        self.assertTrue(plain.success)
        base_level = plain.state.tile(ANCHOR_ROW, 3).paint.level

        with_affix = apply_affix_to_state(_state(), SELF_AFFIX)
        upgraded = install_backpack_paint(with_affix.state, 0, ANCHOR_ROW, 3)
        self.assertTrue(upgraded.success)
        level = upgraded.state.tile(ANCHOR_ROW, 3).paint.level

        # 1(本体) + 2(自身词缀) + 2(调色) = 5；不选词缀只有 1 + 2 = 3
        self.assertEqual(base_level, 3)
        self.assertEqual(level, 5)

    def test_pending_is_consumed_after_install(self) -> None:
        with_affix = apply_affix_to_state(_state(), SELF_AFFIX)
        upgraded = install_backpack_paint(with_affix.state, 0, ANCHOR_ROW, 3)
        self.assertEqual(upgraded.state.self_upgrade_pending, 0)

    def test_anchor_placement_also_consumes_pending(self) -> None:
        """放初始颜料走的是另一条路（``place_current_rule_anchor``），也不能重复加。"""
        pending_state = _state(pending=True)
        with_affix = apply_affix_to_state(pending_state, SELF_AFFIX)
        self.assertEqual(with_affix.state.self_upgrade_pending, 2)
        placed = place_current_rule_anchor(with_affix.state, ANCHOR_ROW, 0)
        self.assertTrue(placed.success)
        self.assertEqual(placed.state.self_upgrade_pending, 0)
        self.assertEqual(placed.state.tile(ANCHOR_ROW, 0).paint.level, 3)

    def test_capped_at_max_level(self) -> None:
        # 封顶要看**放下去那枚**的等级：6 + 自身2 → 7（不是 8）。
        state = _state(backpack_level=6)
        with_affix = apply_affix_to_state(state, SELF_AFFIX)
        upgraded = install_backpack_paint(with_affix.state, 0, ANCHOR_ROW, 3)
        self.assertEqual(upgraded.state.tile(ANCHOR_ROW, 3).paint.level, 7)

    def test_without_affix_nothing_changes(self) -> None:
        result = install_backpack_paint(_state(), 0, ANCHOR_ROW, 3)
        self.assertEqual(result.state.self_upgrade_pending, 0)


if __name__ == "__main__":
    unittest.main()
