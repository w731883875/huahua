#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""宝箱局的打法优先级回归测试。

用户规则（2026-11-12）：

* 初始颜料是宝箱时，第一阶段**首选堆砌**——因为「替换为自身」是堆砌词条，
  第一阶段选了转锋/扩散就永远刷不出它，而第一阶段不可刷新；
* 第二阶段词缀优先级：**替换（拿更多宝箱） > 同类型升级（场上有别的宝箱）
  > 自身提升 > 玩法提升**；
* 用刷新次数优先去刷「替换」。
"""
from __future__ import annotations

import unittest

from huahua_assistant import game_model as gm
from huahua_assistant import practice as pr
from huahua_assistant.engine.actions import ActionKind
from huahua_assistant.engine.rules import RuleEngine


def _catalog() -> list[gm.AffixRule]:
    return gm.build_affix_catalog()


def _find(effect: gm.AffixEffect, **kwargs) -> gm.AffixRule:
    for affix in _catalog():
        if affix.effect is not effect:
            continue
        if all(getattr(affix, key) == value for key, value in kwargs.items()):
            return affix
    raise LookupError(f'找不到词缀 {effect} {kwargs}')


def _anchored_state(treasure: bool = True) -> gm.PuzzleState:
    engine = RuleEngine()
    session = pr.start_practice(2, pet_level=6, rules=2, wait_for_anchor=True)
    action = next(
        a for a in engine.legal_actions(session.state) if a.kind is ActionKind.PLACE_ANCHOR
    )
    session.place_anchor(action.row, action.col)
    state = session.state
    paint = (
        gm.Paint(gm.PaintCategory.GAMEPLAY, 5, subtype='宝箱', is_treasure=True)
        if treasure
        else gm.Paint(gm.PaintCategory.MONSTER, 3, subtype='觉醒者')
    )
    rule = gm.replace(state.current_rule, paint=paint, direction=gm.PaintDirection.GLOBAL)
    return gm.replace(state, current_rule=rule)


def _with_other_treasure(state: gm.PuzzleState) -> gm.PuzzleState:
    """在锚点之外的空白格放一枚宝箱，模拟"场上有别的宝箱"。"""
    tiles = list(state.tiles)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, gm.BOARD_SIZE)
        if tile.state is gm.TileState.BLANK and (row, col) != state.current_anchor:
            tiles[index] = gm.replace(
                tile,
                state=gm.TileState.PAINT,
                paint=gm.Paint(gm.PaintCategory.GAMEPLAY, 5, subtype='宝箱', is_treasure=True),
            )
            return gm.replace(state, tiles=tuple(tiles))
    raise AssertionError('棋盘上找不到空白格')


class TreasureAffixPriorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.copy = _find(gm.AffixEffect.COPY_SELF_TO_SCOPE)
        self.same = _find(gm.AffixEffect.UPGRADE_SCOPE_SAME_PAINT, amount=2)
        self.self_up = _find(gm.AffixEffect.UPGRADE_SELF)
        self.gameplay = next(
            a
            for a in _catalog()
            if a.effect is gm.AffixEffect.UPGRADE_GLOBAL_CATEGORY
            and a.target_object is gm.PaintObject.GAMEPLAY
        )
        self.clear = _find(gm.AffixEffect.CLEAR_TILE_IN_SCOPE)
        # 故意打乱顺序：推荐结果必须与传入顺序无关
        self.options = (self.clear, self.gameplay, self.self_up, self.same, self.copy)

    def test_copy_self_is_first_priority(self) -> None:
        state = _anchored_state()
        ranked = gm.recommend_affixes(state, self.options)
        self.assertIs(ranked[0].affix, self.copy, '宝箱局必须首选「替换」')
        self.assertIn('宝箱首选', ranked[0].reason)

    def test_same_paint_ranks_second_when_other_treasure_exists(self) -> None:
        state = _with_other_treasure(_anchored_state())
        ranked = gm.recommend_affixes(state, self.options)
        order = [item.affix for item in ranked]
        self.assertIs(order[0], self.copy)
        self.assertIs(order[1], self.same, '场上有别的宝箱时，同类型升级排第二')
        self.assertIn('宝箱次选', ranked[1].reason)

    def test_same_paint_rank_is_skipped_without_other_treasure(self) -> None:
        state = _anchored_state()
        ranked = gm.recommend_affixes(state, self.options)
        order = [item.affix for item in ranked]
        self.assertIs(order[1], self.self_up, '没有别的宝箱时，同类型升级不该占第二')
        self.assertNotIn(self.same, order[:2])

    def test_self_upgrade_before_gameplay_upgrade(self) -> None:
        state = _anchored_state()
        order = [item.affix for item in gm.recommend_affixes(state, self.options)]
        self.assertLess(order.index(self.self_up), order.index(self.gameplay))

    def test_non_treasure_round_keeps_score_order(self) -> None:
        """非宝箱局不套用优先级，仍按分数排序。"""
        state = _anchored_state(treasure=False)
        ranked = gm.recommend_affixes(state, self.options)
        scores = [item.score for item in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_settled_paint_override_is_honoured(self) -> None:
        """选词缀时 current_rule 已推进，必须能靠 settled_paint 纠正。"""
        state = _anchored_state(treasure=False)
        treasure_paint = gm.Paint(
            gm.PaintCategory.GAMEPLAY, 5, subtype='宝箱', is_treasure=True
        )
        ranked = gm.recommend_affixes(
            state, self.options, settled_paint=treasure_paint
        )
        self.assertIs(ranked[0].affix, self.copy, 'settled_paint 说是宝箱局，就该首选替换')


class TreasureStackLockTest(unittest.TestCase):
    def test_affix_offer_locks_stack_for_treasure_rule(self) -> None:
        """宝箱局的第二阶段必须从堆砌词条里抽，否则「替换」不可能出现。"""
        session = pr.start_practice(11, pet_level=6, rules=2, wait_for_anchor=True)
        session._settled_rule_paint = gm.Paint(
            gm.PaintCategory.GAMEPLAY, 5, subtype='宝箱', is_treasure=True
        )
        seen_stack = 0
        for _ in range(20):
            session._offer_affix()
            if session._affix_scenario.chosen_stage1 is gm.BrushType.STACK:
                seen_stack += 1
                self.assertTrue(
                    all(a.brush_type is gm.BrushType.STACK for a in session.affix_options),
                    '选定堆砌后第二阶段只能是堆砌词条',
                )
        self.assertGreater(seen_stack, 0)

    def test_non_treasure_does_not_force_stack(self) -> None:
        session = pr.start_practice(11, pet_level=6, rules=2, wait_for_anchor=True)
        session._settled_rule_paint = gm.Paint(gm.PaintCategory.MONSTER, 3, subtype='觉醒者')
        types = set()
        for _ in range(20):
            session._offer_affix()
            types.add(session._affix_scenario.chosen_stage1)
        self.assertGreater(len(types), 1, '非宝箱局不应锁定堆砌')


class CopySelfScoringTest(unittest.TestCase):
    def test_copy_self_scores_beyond_zero_on_empty_scope(self) -> None:
        """范围里还没有颜料时也要有正分——它的价值主要在之后落的颜料上。"""
        state = _anchored_state()
        copy_affix = _find(gm.AffixEffect.COPY_SELF_TO_SCOPE)
        self.assertGreater(gm._score_affix(state, copy_affix), 0.0)

    def test_copy_self_beats_clear_affix_on_treasure_global(self) -> None:
        state = _anchored_state()
        copy_affix = _find(gm.AffixEffect.COPY_SELF_TO_SCOPE)
        clear_affix = _find(gm.AffixEffect.CLEAR_TILE_IN_SCOPE)
        self.assertGreater(
            gm._score_affix(state, copy_affix), gm._score_affix(state, clear_affix)
        )


if __name__ == '__main__':
    unittest.main()
