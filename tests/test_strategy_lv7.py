#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""强制完成门槛 / 宝箱优先策略的回归测试（2026-11-23 用户口径）。

三条被实测打脸的旧行为，这里全部锁死：

1. **强制完成曾被当成「无路可走」的兜底**：5 万局里 100% 由「推荐为空」
   触发，93% 锚点还不是宝箱，平均用满 3 次。用户口径是「基本不会 3 次，
   风险太高」，只有宝箱冲 7 级才用 1~2 次。
2. **「调色目标的进度要求减少 N」曾是写死 140 分的最高分**，完全不看局面，
   于是推荐器 200 局里选了 191 次，把词缀额度全喂给「让任务更容易」而不是
   「产出更多 7 级」。
3. 用户明确：**以 7 级宝箱个数为第一目标**；升级词缀里**优先选提升 2 级的**。
"""

from __future__ import annotations

import unittest

from huahua_assistant import game_model as gm
from huahua_assistant import practice as pr
from huahua_assistant import strategy_lv7 as st

BOARD = gm.BOARD_SIZE


def _blank_board() -> list[gm.Tile]:
    return [gm.Tile(gm.TileState.BLANK) for _ in range(BOARD * BOARD)]


def _paint(category: gm.PaintCategory, level: int, *, treasure: bool = False) -> gm.Paint:
    return gm.Paint(category=category, level=level, subtype='存活', is_treasure=treasure)


def _state(tiles: list[gm.Tile], **kwargs) -> gm.PuzzleState:
    return gm.PuzzleState(tiles=tuple(tiles), **kwargs)


class LineRunsTest(unittest.TestCase):
    """整条直线的可用地块数，空洞不切段，已有同类颜料不扣除。"""

    def test_empty_board_has_full_length_runs(self) -> None:
        runs = st.line_runs(_state(_blank_board()))
        self.assertIn(BOARD, runs, '空棋盘应当有整行/整列的 6 连')

    def test_void_cells_reduce_capacity_without_splitting(self) -> None:
        empty_runs = st.line_runs(_state(_blank_board()))
        full_six = sum(1 for length in empty_runs if length >= BOARD)
        tiles = _blank_board()
        for row in range(BOARD):
            tiles[row * BOARD + 2] = gm.Tile(gm.TileState.VOID)
        runs = st.line_runs(_state(tiles))
        self.assertLess(sum(length >= BOARD for length in runs), full_six)
        self.assertGreaterEqual(runs.count(5), BOARD, '每行中间一个虚空，应保留整条 5 格线')

    def test_long_line_count_only_counts_long_ones(self) -> None:
        self.assertGreaterEqual(
            st.long_line_count(_state(_blank_board()), minimum=4), 1
        )
        tiles = _blank_board()
        # 只留左上 2×2 的空白，其余全变虚空 → 不可能有 4 连
        for row in range(BOARD):
            for col in range(BOARD):
                if row >= 2 or col >= 2:
                    tiles[row * BOARD + col] = gm.Tile(gm.TileState.VOID)
        self.assertEqual(st.long_line_count(_state(tiles), minimum=4), 0)


class ForceGateTest(unittest.TestCase):
    """强制完成的门槛：不该再被当成兜底。"""

    def _rule(self) -> gm.CurrentPaintRule:
        return gm.CurrentPaintRule(
            paint=_paint(gm.PaintCategory.MONSTER, 3),
            direction=gm.PaintDirection.HORIZONTAL,
            target_object=gm.PaintObject.MONSTER,
            match_required=3,
            upgrade_amount=1,
        )

    def test_rejected_after_three_uses(self) -> None:
        decision = st.decide_force_complete(
            _state(_blank_board(), current_rule=self._rule()), 3
        )
        self.assertFalse(decision.allowed)
        self.assertIn('3 次', decision.reason)

    def test_rejected_when_no_long_line_and_no_treasure(self) -> None:
        """图形不好、又没宝箱 → 不许炸颜料（用户：宁可直接重绘）。"""
        tiles = _blank_board()
        for row in range(BOARD):
            for col in range(2, BOARD):
                tiles[row * BOARD + col] = gm.Tile(gm.TileState.VOID)
        decision = st.decide_force_complete(_state(tiles, current_rule=self._rule()), 0)
        self.assertFalse(decision.allowed, decision.reason)

    def test_allowed_with_treasure_supply(self) -> None:
        tiles = _blank_board()
        for row in range(BOARD):
            for col in range(2, BOARD):
                tiles[row * BOARD + col] = gm.Tile(gm.TileState.VOID)
        state = _state(
            tiles,
            current_rule=self._rule(),
            backpack=(_paint(gm.PaintCategory.GAMEPLAY, 4, treasure=True),),
        )
        decision = st.decide_force_complete(state, 0)
        self.assertTrue(decision.allowed, decision.reason)

    def test_lv7_treasure_stops_further_forcing_without_supply(self) -> None:
        """已经有 7 级宝箱、又没有更多宝箱可养 → 不再炸，直接收尾。"""
        tiles = _blank_board()
        tiles[0] = gm.Tile(
            gm.TileState.PAINT, paint=_paint(gm.PaintCategory.GAMEPLAY, 7, treasure=True)
        )
        state = _state(tiles, current_rule=self._rule())
        decision = st.decide_force_complete(state, 0)
        self.assertFalse(decision.allowed, decision.reason)

    def test_third_force_requires_high_level_treasure_and_few_blanks(self) -> None:
        """第 3 次门槛最高：5/6 级宝箱 + 只剩 2~3 个空白格。"""
        tiles = _blank_board()
        for index in range(3, BOARD * BOARD):
            tiles[index] = gm.Tile(gm.TileState.VOID)
        # 只剩 2 个空白，且范围内没有宝箱 → 不许
        state = _state(tiles, current_rule=self._rule(), current_anchor=(0, 0))
        self.assertFalse(st.decide_force_complete(state, 2).allowed)


class Lv7TreasureObjectiveTest(unittest.TestCase):
    """7 级宝箱是第一目标。"""

    def test_counts_only_level7_treasures(self) -> None:
        tiles = _blank_board()
        tiles[0] = gm.Tile(
            gm.TileState.PAINT, paint=_paint(gm.PaintCategory.GAMEPLAY, 7, treasure=True)
        )
        tiles[1] = gm.Tile(
            gm.TileState.PAINT, paint=_paint(gm.PaintCategory.GAMEPLAY, 6, treasure=True)
        )
        tiles[2] = gm.Tile(
            gm.TileState.PAINT, paint=_paint(gm.PaintCategory.GAMEPLAY, 7)
        )
        self.assertEqual(st.lv7_treasure_count(_state(tiles)), 1)

    def test_treasure_supply_counts_backpack_and_unfinished_board(self) -> None:
        tiles = _blank_board()
        tiles[0] = gm.Tile(
            gm.TileState.PAINT, paint=_paint(gm.PaintCategory.GAMEPLAY, 4, treasure=True)
        )
        tiles[1] = gm.Tile(
            gm.TileState.PAINT, paint=_paint(gm.PaintCategory.GAMEPLAY, 7, treasure=True)
        )
        state = _state(
            tiles, backpack=(_paint(gm.PaintCategory.GAMEPLAY, 2, treasure=True),)
        )
        # 背包 1 + 棋盘 4 级那枚 1 = 2；7 级那枚不再计入
        self.assertEqual(st.treasure_supply(state), 2)


class MatchRequiredDeltaScoringTest(unittest.TestCase):
    """「进度要求减少」不该再压过升级类词缀。"""

    def _setup(self, *, match_required: int = 4, match_current: int = 0):
        rule = gm.CurrentPaintRule(
            paint=_paint(gm.PaintCategory.GAMEPLAY, 4),
            direction=gm.PaintDirection.HORIZONTAL,
            target_object=gm.PaintObject.GAMEPLAY,
            match_required=match_required,
            match_current=match_current,
            upgrade_amount=1,
        )
        tiles = _blank_board()
        return _state(tiles, current_rule=rule, current_anchor=(2, 2))

    def test_no_longer_a_flat_top_score(self) -> None:
        state = self._setup()
        delta = next(
            affix
            for affix in gm.build_affix_catalog()
            if affix.effect is gm.AffixEffect.MATCH_REQUIRED_DELTA and int(affix.delta) < 0
        )
        upgrade = next(
            affix
            for affix in gm.build_affix_catalog()
            if affix.effect is gm.AffixEffect.UPGRADE_GLOBAL_SAME_PAINT and int(affix.amount) >= 2
        )
        delta_score = gm._score_affix(state, delta)
        upgrade_score = gm._score_affix(state, upgrade)
        self.assertLess(
            delta_score,
            upgrade_score,
            f'进度减少({delta_score:.0f}) 不该高过升级(+{upgrade.amount}级, {upgrade_score:.0f})',
        )

    def test_still_valuable_when_board_is_stuck(self) -> None:
        """棋盘放不下时它仍是救场词缀，要给高分。"""
        state = self._setup(match_required=6)
        # 只留 1 个空白格 → 凑不齐 6 个
        tiles = list(state.tiles)
        for index in range(1, BOARD * BOARD):
            tiles[index] = gm.Tile(gm.TileState.VOID)
        stuck = _state(tiles, current_rule=state.current_rule, current_anchor=(0, 0))
        delta = next(
            affix
            for affix in gm.build_affix_catalog()
            if affix.effect is gm.AffixEffect.MATCH_REQUIRED_DELTA and int(affix.delta) < 0
        )
        self.assertGreater(gm._score_affix(stuck, delta), 100.0)


class AmountPreferenceTest(unittest.TestCase):
    """用户口径：有「提升 2 级」时优先选 2 级。"""

    def test_higher_amount_scores_higher(self) -> None:
        self.assertGreater(
            gm._amount_preference_bonus(2), gm._amount_preference_bonus(1)
        )
        self.assertGreater(
            gm._amount_preference_bonus(3), gm._amount_preference_bonus(2)
        )

    def test_plus_one_gets_no_bonus(self) -> None:
        self.assertEqual(gm._amount_preference_bonus(1), 0.0)


class FusionGateTest(unittest.TestCase):
    """融合门槛：能推 7 级才融，推不到就看能不能帮上调色任务。

    用户 2026-11-23：「融合需要看是否将宝箱及其他玩法推向 7 级，如果不能则
    以能完成任务优先」。实机 3 次融合里有 2 次是「相邻怪物正好是调色目标，
    放上去就完成任务」——不是无脑融。
    """

    def _rule(self, target=gm.PaintObject.MONSTER):
        return gm.CurrentPaintRule(
            paint=_paint(gm.PaintCategory.GAMEPLAY, 3),
            direction=gm.PaintDirection.ADJACENT,
            target_object=target,
            match_required=3,
            upgrade_amount=1,
        )

    def _state_with_kept_effect(self, *, effect_target=gm.PaintObject.GAMEPLAY,
                                paint_level=6, treasure=True):
        """在 (0,0) 留效果包，相邻格 (0,1) 放一枚待加强的颜料。"""
        tiles = _blank_board()
        tiles[0] = gm.Tile(
            gm.TileState.FADED,
            retained_effects=(
                gm.PaintEffect(gm.PaintDirection.ADJACENT, effect_target, 1),
            ),
        )
        tiles[1] = gm.Tile(
            gm.TileState.PAINT,
            paint=_paint(gm.PaintCategory.GAMEPLAY, paint_level, treasure=treasure),
        )
        return _state(tiles, current_rule=self._rule())

    def test_fuses_when_treasure_reaches_lv7(self):
        state = self._state_with_kept_effect(paint_level=6, treasure=True)
        decision = st.decide_fusion(state, cell=(0, 0))
        self.assertTrue(decision.allowed, decision.reason)
        self.assertIn('宝箱', decision.reason)

    def test_fuses_when_gameplay_reaches_lv7(self):
        state = self._state_with_kept_effect(paint_level=6, treasure=False)
        decision = st.decide_fusion(state, cell=(0, 0))
        self.assertTrue(decision.allowed, decision.reason)
        self.assertIn('玩法', decision.reason)

    def test_refuses_when_cannot_reach_lv7_nor_help_task(self):
        """宝箱推不到 7 级，且残留效果对象不是当前调色目标 -> 不融。"""
        state = self._state_with_kept_effect(
            effect_target=gm.PaintObject.GAMEPLAY, paint_level=1, treasure=True)
        state = gm.replace(state, current_rule=self._rule(target=gm.PaintObject.MONSTER))
        decision = st.decide_fusion(state, cell=(0, 0))
        self.assertFalse(decision.allowed, decision.reason)

    def test_fuses_when_target_matches_even_if_no_lv7(self):
        """推不到 7 级，但残留效果对象和调色目标一致 -> 有助于完成任务，融。"""
        state = self._state_with_kept_effect(
            effect_target=gm.PaintObject.GAMEPLAY, paint_level=2, treasure=False)
        state = gm.replace(state, current_rule=self._rule(target=gm.PaintObject.GAMEPLAY))
        decision = st.decide_fusion(state, cell=(0, 0))
        self.assertTrue(decision.allowed, decision.reason)
        self.assertIn('完成任务', decision.reason)

    def test_refuses_without_kept_effect_cell(self):
        state = _state(_blank_board(), current_rule=self._rule())
        self.assertFalse(st.decide_fusion(state, cell=(0, 0)).allowed)


if __name__ == '__main__':
    unittest.main()
