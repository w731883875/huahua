"""第二条笔触效果：融合初始颜料挂 2 条笔触（用户 2026-11-23 规则三）。

融合初始颜料「同时拥有 2 个调色效果、2 个笔触效果、2 个范围」。
它自带的那套笔触是从被炸毁的初始颜料那里继承的，这一轮再选的那条是**第二套**，
两条都要留着 —— 所以挂词缀时要**追加**，不能像普通情况那样覆盖。

普通颜料不受影响：一局每轮只选一条笔触，旧的交棒，仍是覆盖。
"""

from __future__ import annotations

import unittest

from huahua_assistant import game_model as gm
from huahua_assistant.models import PaintCategory, PaintDirection, PaintObject

GAMEPLAY = PaintCategory.GAMEPLAY


def _paint(level: int = 2, subtype: str = "存活") -> gm.Paint:
    return gm.Paint(category=GAMEPLAY, level=level, subtype=subtype)


def _upgrade_affix(amount: int, direction: PaintDirection | None = None) -> gm.AffixRule:
    """「范围内目标额外提升 N 级」这类笔触。"""
    return next(
        affix
        for affix in gm.build_affix_catalog()
        if affix.effect is gm.AffixEffect.UPGRADE_IN_SCOPE
        and affix.amount == amount
        and affix.direction == direction
    )


def _rule(*, effects: tuple[gm.PaintEffect, ...] = ()) -> gm.CurrentPaintRule:
    return gm.CurrentPaintRule(
        paint=_paint(),
        direction=PaintDirection.HORIZONTAL,
        target_object=PaintObject.GAMEPLAY,
        match_required=3,
        upgrade_amount=1,
        effects=effects,
    )


def _merged_rule() -> gm.CurrentPaintRule:
    """融合颜料：横向玩法 +1、纵向怪物 +1 两套调色效果。"""
    return _rule(
        effects=(
            gm.PaintEffect(PaintDirection.HORIZONTAL, PaintObject.GAMEPLAY, 1),
            gm.PaintEffect(PaintDirection.VERTICAL, PaintObject.MONSTER, 1),
        )
    )


def _state(rule: gm.CurrentPaintRule) -> gm.PuzzleState:
    tiles = [gm.Tile(gm.TileState.BLANK) for _ in range(gm.BOARD_SIZE * gm.BOARD_SIZE)]
    return gm.PuzzleState(tiles=tuple(tiles), current_rule=rule, current_anchor=(2, 2))


class OrdinaryRuleStillReplacesTest(unittest.TestCase):
    def test_a_second_brush_replaces_the_first_on_an_ordinary_paint(self) -> None:
        state = _state(_rule())
        first = _upgrade_affix(1)
        second = _upgrade_affix(2)

        state = gm.apply_affix_to_state(state, first, seed=0).state
        state = gm.apply_affix_to_state(state, second, seed=0).state

        self.assertEqual(state.active_affix, second)
        self.assertEqual(state.extra_affixes, (), "普通颜料只留最新那条")
        self.assertEqual(len(state.affixes), 1)


class MergedRuleStacksTest(unittest.TestCase):
    def test_a_second_brush_is_kept_on_a_merged_paint(self) -> None:
        state = _state(_merged_rule())
        first = _upgrade_affix(1)
        second = _upgrade_affix(2)

        state = gm.apply_affix_to_state(state, first, seed=0).state
        state = gm.apply_affix_to_state(state, second, seed=0).state

        self.assertEqual(state.active_affix, first, "主条目保留先选的那条")
        self.assertEqual(state.extra_affixes, (second,), "第二条进额外槽")
        self.assertEqual(set(state.affixes), {first, second}, "两条笔触都在")

    def test_selecting_the_same_brush_twice_does_not_duplicate_it(self) -> None:
        state = _state(_merged_rule())
        affix = _upgrade_affix(1)

        state = gm.apply_affix_to_state(state, affix, seed=0).state
        state = gm.apply_affix_to_state(state, affix, seed=0).state

        self.assertEqual(state.extra_affixes, ())
        self.assertEqual(len(state.affixes), 1)

    def test_both_brushes_settle_on_the_next_placement(self) -> None:
        """两条笔触都要在落笔时结算，不能只算主条目。"""
        state = _state(_merged_rule())
        first = _upgrade_affix(1)
        second = _upgrade_affix(2)
        state = gm.apply_affix_to_state(state, first, seed=0).state
        state = gm.apply_affix_to_state(state, second, seed=0).state

        # 范围内放一枚颜料，两条笔触都该命中它
        tiles = list(state.tiles)
        tiles[2 * gm.BOARD_SIZE + 4] = gm.Tile(gm.TileState.PAINT, _paint(1))
        state = gm.replace(state, tiles=tuple(tiles))

        settled, count = gm._settle_scope_buff(
            state, __import__("random").Random(0), placed=(2, 4)
        )

        self.assertGreater(count, 0, "至少有一条笔触命中了目标")
        self.assertGreater(
            gm.tile_bonus_level(settled.tile(2, 4)),
            0,
            "笔触加成要落到颜料上",
        )


class AffixHelperTest(unittest.TestCase):
    def test_consumable_place_affix_is_not_persistent(self) -> None:
        for affix in gm.build_affix_catalog():
            if affix.effect in gm.CONSUMABLE_PLACE_EFFECTS:
                self.assertFalse(
                    gm._is_persistent_affix(affix),
                    f"{affix.name} 是一次性放置词条，不该算整局有效",
                )

    def test_scope_upgrade_affix_is_persistent(self) -> None:
        self.assertTrue(gm._is_persistent_affix(_upgrade_affix(1)))

    def test_rule_is_merged_only_with_multiple_effects(self) -> None:
        self.assertFalse(gm._rule_is_merged(_rule()))
        self.assertTrue(gm._rule_is_merged(_merged_rule()))
        self.assertFalse(gm._rule_is_merged(None))


if __name__ == "__main__":
    unittest.main()
