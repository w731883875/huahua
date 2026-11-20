from __future__ import annotations

import random
import unittest

from huahua_assistant import game_model as gm
from huahua_assistant.engine import (
    Action,
    ActionKind,
    BeamSearch,
    DeterminizedSearch,
    HeuristicPolicy,
    RuleEngine,
    board_value,
    compare,
    level7_counts,
    play_episode,
)
from huahua_assistant.engine.value import EvalWeights, position_value


class RuleEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RuleEngine()

    def test_legal_actions_are_applicable(self) -> None:
        state = gm.random_state(7)
        actions = self.engine.legal_actions(state)
        self.assertTrue(actions, "开局应当至少有一个合法动作")
        rng = random.Random(1)
        for action in actions:
            after = self.engine.apply(state, action, rng)
            self.assertIsNot(after, state, f"{action.label()} 应当改变局面")

    def test_pending_state_only_offers_anchor_cells(self) -> None:
        state = gm.random_state(11)
        pending = gm.replace(state, current_rule_pending=True, current_anchor=None)
        actions = self.engine.legal_actions(pending)
        self.assertTrue(actions)
        self.assertTrue(all(action.kind is ActionKind.PLACE_ANCHOR for action in actions))

    def test_rule_completes_and_advances(self) -> None:
        state = gm.random_state(3)
        engine = self.engine
        start_rule = state.current_buff_id
        rng = random.Random(0)
        steps = 0
        while steps < engine.max_steps and not engine.rule_done(state, start_rule):
            action = HeuristicPolicy().choose(state, engine, rng)
            self.assertIsNotNone(action)
            state = engine.apply(state, action, rng)
            steps += 1
        self.assertTrue(engine.rule_done(state, start_rule))
        self.assertGreater(state.current_buff_id, start_rule)

    def test_backpack_index_lookup(self) -> None:
        # 背包允许出现完全相同的颜料（同家族同等级），此时索引应返回**首个**匹配项，
        # 且该位置的颜料与查询对象相等。
        state = gm.random_state(5)
        paint = state.backpack[3]
        index = self.engine.backpack_index_of(state, paint)
        self.assertIsNotNone(index)
        self.assertEqual(state.backpack[index], paint)
        self.assertLessEqual(index, 3)


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_episode(self) -> None:
        engine = RuleEngine()
        policy = HeuristicPolicy()
        first = play_episode(gm.random_state(21), policy, engine, seed=99)
        second = play_episode(gm.random_state(21), policy, engine, seed=99)
        self.assertEqual(first, second)

    def test_force_complete_is_seeded(self) -> None:
        state = gm.random_state(31)
        first = gm.force_complete_current_rule(state, 0, seed=1234)
        second = gm.force_complete_current_rule(state, 0, seed=1234)
        self.assertEqual(first.destroyed_cells, second.destroyed_cells)


class ValueTest(unittest.TestCase):
    def test_level7_treasure_is_70_points(self) -> None:
        """刻度由用户给定（2026-11-12）：1~7 级宝箱 = 10~70 分，严格线性。

        早先的刻度把 7 级宝箱归一化成 1.0（宝箱 5→7 级 0.22→1.0），
        现已改成绝对分制，所以这里的期望值跟着变，不是实现在漂。
        """
        weights = EvalWeights()
        treasure7 = gm.Paint(category=gm.PaintCategory.GAMEPLAY, level=7, subtype="宝箱", is_treasure=True)
        treasure6 = gm.Paint(category=gm.PaintCategory.GAMEPLAY, level=6, subtype="宝箱", is_treasure=True)
        self.assertAlmostEqual(weights.paint_value(treasure7), 70.0)
        self.assertLess(weights.paint_value(treasure6), weights.paint_value(treasure7))

    def test_board_value_rewards_paint(self) -> None:
        state = gm.random_state(9)
        paint = gm.Paint(category=gm.PaintCategory.GAMEPLAY, level=7, subtype="宝箱", is_treasure=True)
        tiles = list(state.tiles)
        tiles[0] = gm.Tile(state=gm.TileState.PAINT, paint=paint)
        boosted = gm.replace(state, tiles=tuple(tiles))
        self.assertGreater(board_value(boosted), board_value(state))

    def test_level7_counts(self) -> None:
        state = gm.random_state(13)
        treasures, any7 = level7_counts(state)
        self.assertGreaterEqual(treasures, 0)
        self.assertGreaterEqual(any7, treasures)


class SearchTest(unittest.TestCase):
    def test_beam_returns_a_legal_action(self) -> None:
        engine = RuleEngine()
        state = gm.random_state(17)
        legal = set(engine.legal_actions(state))
        best = BeamSearch(engine).best_action(state, width=6)
        self.assertIn(best, legal)

    def test_beam_beats_force_complete(self) -> None:
        engine = RuleEngine()
        state = gm.random_state(19)
        result = BeamSearch(engine).search(state, width=6)
        force = [value for value in result.actions if value.action.kind is ActionKind.FORCE_COMPLETE]
        best = result.best
        self.assertIsNotNone(best)
        self.assertNotEqual(best.action.kind, ActionKind.FORCE_COMPLETE)
        if force:
            self.assertGreaterEqual(best.mean, force[0].mean)

    def test_mc_search_is_reproducible(self) -> None:
        engine = RuleEngine()
        state = gm.random_state(23)
        search = DeterminizedSearch(engine)
        first = search.search(state, playouts=2, top_k=4, seed=5)
        second = search.search(state, playouts=2, top_k=4, seed=5)
        self.assertEqual(
            [(value.action, value.mean) for value in first.actions],
            [(value.action, value.mean) for value in second.actions],
        )

    def test_position_value_tracks_rule_progress(self) -> None:
        state = gm.random_state(29)
        rule = state.current_rule
        self.assertIsNotNone(rule)
        progressed = gm.replace(state, current_rule=gm.replace(rule, match_current=rule.match_required))
        self.assertGreater(position_value(progressed), position_value(state))


class CompareTest(unittest.TestCase):
    def test_compare_runs_and_reports(self) -> None:
        report = compare(episodes=2, base_seed=77, engine=RuleEngine())
        self.assertEqual(report.episodes, 2)
        self.assertEqual(report.wins + report.ties + report.losses, 2)
        self.assertIn("平均终局价值", report.format())


if __name__ == "__main__":
    unittest.main()
