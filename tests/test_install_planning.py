from __future__ import annotations

import unittest

from huahua_assistant.game_model import (
    AffixEffect,
    AffixRule,
    BrushType,
    CurrentPaintRule,
    Paint,
    PuzzleState,
    Tile,
    place_current_rule_anchor,
    recommend_install_plan,
    recommend_placements,
    sort_backpack_for_rule,
)
from huahua_assistant.models import PaintCategory, PaintDirection, PaintObject, TileState


def _vertical_four_cell_state(backpack: list[Paint]) -> PuzzleState:
    anchor = (2, 2)
    tiles = [Tile(TileState.VOID) for _ in range(36)]
    for row in (0, 1, 3, 4):
        tiles[row * 6 + 2] = Tile(TileState.BLANK)
    tiles[5 * 6 + 2] = Tile(TileState.BLOCKED)
    rule = CurrentPaintRule(
        paint=Paint(PaintCategory.GAMEPLAY, 3),
        direction=PaintDirection.VERTICAL,
        target_object=PaintObject.MONSTER,
        match_required=3,
        upgrade_amount=1,
    )
    tiles[anchor[0] * 6 + anchor[1]] = Tile(TileState.PAINT, rule.paint)
    return PuzzleState(
        tiles=tuple(tiles),
        current_rule=rule,
        backpack=tuple(backpack),
        current_anchor=anchor,
    )


class InstallPlanningTests(unittest.TestCase):
    def test_backpack_prioritizes_target_then_descending_level(self) -> None:
        rule = CurrentPaintRule(
            paint=Paint(PaintCategory.GAMEPLAY, 3),
            direction=PaintDirection.VERTICAL,
            target_object=PaintObject.MONSTER,
            match_required=3,
            upgrade_amount=1,
        )
        gameplay_7 = Paint(PaintCategory.GAMEPLAY, 7)
        monster_2 = Paint(PaintCategory.MONSTER, 2)
        monster_6 = Paint(PaintCategory.MONSTER, 6)

        ordered = sort_backpack_for_rule((gameplay_7, monster_2, monster_6), rule)

        self.assertEqual((monster_6, monster_2, gameplay_7), ordered)

    def test_four_scope_cells_only_uses_best_three_with_top_paints(self) -> None:
        paints = [
            Paint(PaintCategory.MONSTER, 3),
            Paint(PaintCategory.MONSTER, 6),
            Paint(PaintCategory.MONSTER, 4),
            Paint(PaintCategory.MONSTER, 5),
        ]

        plan = recommend_install_plan(_vertical_four_cell_state(paints))

        self.assertIsNotNone(plan)
        self.assertTrue(plan.completes_match)
        self.assertEqual(3, len(plan.steps))
        self.assertEqual([6, 5, 4], sorted((step.paint.level for step in plan.steps), reverse=True))
        self.assertNotIn((0, 2), {(step.row, step.col) for step in plan.steps})

    def test_highest_level_is_assigned_to_highest_value_cell(self) -> None:
        paints = [Paint(PaintCategory.MONSTER, level) for level in (6, 5, 4, 3)]

        plan = recommend_install_plan(_vertical_four_cell_state(paints))

        self.assertIsNotNone(plan)
        by_cell = {(step.row, step.col): step.paint.level for step in plan.steps}
        self.assertEqual(6, by_cell[(3, 2)])
        self.assertEqual(6, plan.steps[0].paint.level)

    def test_insufficient_target_supply_returns_best_partial_plan(self) -> None:
        paints = [
            Paint(PaintCategory.MONSTER, 6),
            Paint(PaintCategory.GAMEPLAY, 7),
            Paint(PaintCategory.MONSTER, 4),
        ]

        plan = recommend_install_plan(_vertical_four_cell_state(paints))

        self.assertIsNotNone(plan)
        self.assertFalse(plan.completes_match)
        self.assertEqual(2, len(plan.steps))
        self.assertEqual({PaintCategory.MONSTER}, {step.paint.category for step in plan.steps})

    def test_blocked_affix_recommends_and_places_blocked_anchor(self) -> None:
        tiles = [Tile(TileState.VOID) for _ in range(36)]
        tiles[2 * 6 + 2] = Tile(TileState.BLOCKED)
        rule = CurrentPaintRule(
            paint=Paint(PaintCategory.GAMEPLAY, 3),
            direction=PaintDirection.VERTICAL,
            target_object=PaintObject.MONSTER,
            match_required=3,
            upgrade_amount=1,
        )
        affix = AffixRule(
            BrushType.SPREAD,
            "当前颜料可放置在阻挡地块",
            AffixEffect.ALLOW_PLACE_ON_BLOCKED,
        )
        state = PuzzleState(
            tiles=tuple(tiles),
            current_rule=rule,
            current_rule_pending=True,
            active_affix=affix,
        )

        recommendations = recommend_placements(state, limit=36)
        result = place_current_rule_anchor(state, 2, 2)

        self.assertEqual([(2, 2)], [(rec.row, rec.col) for rec in recommendations])
        self.assertIn("阻挡地块", recommendations[0].reason)
        self.assertEqual("放到阻挡", recommendations[0].explanation.short)
        self.assertTrue(result.success)
        self.assertEqual(TileState.PAINT, result.state.tile(2, 2).state)
        self.assertIsNone(result.state.active_affix)

    def test_special_anchor_affix_does_not_make_scope_tiles_installable(self) -> None:
        tiles = [Tile(TileState.VOID) for _ in range(36)]
        for row in range(6):
            tiles[row * 6 + 2] = Tile(TileState.BLOCKED)
        rule = CurrentPaintRule(
            paint=Paint(PaintCategory.GAMEPLAY, 3),
            direction=PaintDirection.VERTICAL,
            target_object=PaintObject.MONSTER,
            match_required=3,
            upgrade_amount=1,
        )
        affix = AffixRule(
            BrushType.SPREAD,
            "当前颜料可放置在阻挡地块",
            AffixEffect.ALLOW_PLACE_ON_BLOCKED,
        )
        state = PuzzleState(
            tiles=tuple(tiles),
            current_rule=rule,
            backpack=tuple(Paint(PaintCategory.MONSTER, 1) for _ in range(3)),
            current_rule_pending=True,
            active_affix=affix,
        )

        recommendations = recommend_placements(state, limit=36)

        self.assertEqual(6, len(recommendations))
        self.assertFalse(any(rec.completes_match for rec in recommendations))

    def test_void_affix_warns_about_lost_paint_and_prefers_safe_blank(self) -> None:
        tiles = [Tile(TileState.BLOCKED) for _ in range(36)]
        tiles[2 * 6 + 2] = Tile(TileState.VOID)
        tiles[0] = Tile(TileState.BLANK)
        treasure = Paint(PaintCategory.GAMEPLAY, 6, subtype="宝箱", is_treasure=True)
        rule = CurrentPaintRule(
            paint=treasure,
            direction=PaintDirection.VERTICAL,
            target_object=PaintObject.MONSTER,
            match_required=3,
            upgrade_amount=1,
        )
        affix = AffixRule(
            BrushType.SPREAD,
            "当前颜料可放置在虚空地块但失去自身内容",
            AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF,
            persistent_until_match=True,
        )
        state = PuzzleState(
            tiles=tuple(tiles),
            current_rule=rule,
            current_rule_pending=True,
            active_affix=affix,
        )

        recommendations = recommend_placements(state, limit=36)
        void_rec = next(rec for rec in recommendations if (rec.row, rec.col) == (2, 2))
        result = place_current_rule_anchor(state, 2, 2)

        self.assertEqual((0, 0), (recommendations[0].row, recommendations[0].col))
        self.assertIn("失去宝箱Lv6自身内容", void_rec.reason)
        self.assertEqual("虚空失去自身", void_rec.explanation.short)
        self.assertLessEqual(void_rec.explanation.risk, -3000.0)
        self.assertTrue(result.success)
        self.assertEqual(TileState.FADED, result.state.tile(2, 2).state)
        self.assertIsNone(result.state.tile(2, 2).paint)
        self.assertIsNone(result.state.active_affix)

    def test_special_tiles_are_not_candidates_without_matching_affix(self) -> None:
        tiles = [Tile(TileState.BLOCKED) for _ in range(36)]
        rule = CurrentPaintRule(
            paint=Paint(PaintCategory.GAMEPLAY, 3),
            direction=PaintDirection.VERTICAL,
            target_object=PaintObject.MONSTER,
            match_required=3,
            upgrade_amount=1,
        )
        state = PuzzleState(
            tiles=tuple(tiles),
            current_rule=rule,
            current_rule_pending=True,
        )

        self.assertEqual([], recommend_placements(state, limit=36))

    def test_observed_gold_scope_is_a_hard_placement_boundary(self) -> None:
        tiles = [Tile(TileState.BLANK) for _ in range(36)]
        rule = CurrentPaintRule(
            paint=Paint(PaintCategory.GAMEPLAY, 3),
            direction=PaintDirection.VERTICAL,
            target_object=PaintObject.MONSTER,
            match_required=3,
            upgrade_amount=1,
        )
        state = PuzzleState(
            tiles=tuple(tiles),
            current_rule=rule,
            backpack=(Paint(PaintCategory.MONSTER, 4),),
            observed_scope_cells=((1, 2), (2, 2), (3, 2)),
        )

        recommendations = recommend_placements(state, limit=36)

        self.assertEqual(
            {(1, 2), (2, 2), (3, 2)},
            {(rec.row, rec.col) for rec in recommendations},
        )

    def test_missing_anchor_and_gold_scope_produces_no_placement_guess(self) -> None:
        tiles = [Tile(TileState.BLANK) for _ in range(36)]
        rule = CurrentPaintRule(
            paint=Paint(PaintCategory.GAMEPLAY, 3),
            direction=PaintDirection.VERTICAL,
            target_object=PaintObject.MONSTER,
            match_required=3,
            upgrade_amount=1,
        )
        state = PuzzleState(
            tiles=tuple(tiles),
            current_rule=rule,
            backpack=(Paint(PaintCategory.MONSTER, 4),),
        )

        self.assertEqual([], recommend_placements(state, limit=36))


if __name__ == "__main__":
    unittest.main()
