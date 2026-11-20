"""官方权重表接入的回归测试（依据：S8画中雪原_作画玩法_官方规则.md §4.3）。

覆盖三件事：
1. 权重表本身被正确读出（合计 10000、期望 3.54 级、全图 2.44%）；
2. 估分链路按官方分布算期望，而不是按等概率或固定常数；
3. 模拟链路按官方分布抽样（7 级约 1%、全图约 2.4%），与估分口径一致。
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from huahua_assistant.game_model import (
    AffixEffect,
    AffixRule,
    BrushType,
    CurrentPaintRule,
    Paint,
    PuzzleState,
    Tile,
    _random_scope_addition_score,
    _randomize_current_level_score,
    apply_affix_to_state,
    build_affix_catalog,
)
from huahua_assistant.models import PaintCategory, PaintDirection, PaintObject, TileState
from huahua_assistant.s8_official_data import (
    affix_option_distribution,
    affix_option_expectation,
)


def _open_state(level: int, *, treasure: bool = False) -> PuzzleState:
    """6x6 全空白，当前初始颜料居中，作用范围为纵向。"""
    tiles = [Tile(TileState.BLANK) for _ in range(36)]
    paint = Paint(PaintCategory.GAMEPLAY, level, is_treasure=treasure)
    rule = CurrentPaintRule(
        paint=paint,
        direction=PaintDirection.VERTICAL,
        target_object=PaintObject.ALL,
        match_required=3,
        upgrade_amount=1,
    )
    tiles[2 * 6 + 2] = Tile(TileState.PAINT, paint)
    return PuzzleState(
        tiles=tuple(tiles),
        current_rule=rule,
        backpack=(),
        current_anchor=(2, 2),
    )


def _catalog_affix(effect: AffixEffect, *, delta: int = 0) -> AffixRule:
    for affix in build_affix_catalog():
        if affix.effect == effect and affix.delta == delta:
            return affix
    raise AssertionError(f"catalog 缺少 {effect} delta={delta}")


def _official_affix(official_id: int) -> AffixRule:
    for affix in build_affix_catalog():
        if official_id in affix.official_ids:
            return affix
    raise AssertionError(f"catalog 缺少官方记录 {official_id}")


def _void_state(void_count: int) -> PuzzleState:
    state = _open_state(1)
    tiles = list(state.tiles)
    for index in range(void_count):
        tiles[index] = Tile(TileState.VOID)
    return replace(state, tiles=tuple(tiles))


class OfficialWeightTableTests(unittest.TestCase):
    def test_randomize_level_table_matches_official_original(self) -> None:
        distribution = affix_option_distribution(4005)

        self.assertAlmostEqual(1.0, sum(distribution.values()), places=9)
        self.assertAlmostEqual(0.07, distribution[1], places=9)
        self.assertAlmostEqual(0.01, distribution[7], places=9)
        # 期望等级 = 35400 / 10000，赌到 7 级只有 1%
        self.assertAlmostEqual(3.54, affix_option_expectation(4005), places=9)

    def test_scope_table_marks_global_as_rare(self) -> None:
        distribution = affix_option_distribution(2019)

        self.assertEqual(5, len(distribution))
        self.assertAlmostEqual(0.0244, distribution[5], places=4)
        # 四个常见方向各 24.4%，全图只有它们的十分之一
        for tier in (1, 2, 3, 4):
            self.assertAlmostEqual(0.2439, distribution[tier], places=4)

    def test_target_object_table_marks_all_as_rare(self) -> None:
        distribution = affix_option_distribution(1006)

        self.assertEqual(3, len(distribution))
        self.assertAlmostEqual(0.0476, distribution[3], places=4)


class RandomizeCurrentLevelScoreTests(unittest.TestCase):
    def test_top_level_paint_never_scores_positive(self) -> None:
        # 当前已是 7 级：官方分布下只可能持平或降级
        self.assertEqual(0.0, _randomize_current_level_score(_open_state(7)))

    def test_score_decreases_as_current_level_rises(self) -> None:
        scores = [_randomize_current_level_score(_open_state(level)) for level in range(1, 8)]

        for lower, higher in zip(scores, scores[1:]):
            self.assertGreaterEqual(lower, higher)
        self.assertGreater(scores[0], 0.0)
        self.assertEqual(0.0, scores[6])

    def test_treasure_upside_is_priced_higher_than_plain_paint(self) -> None:
        plain = _randomize_current_level_score(_open_state(1))
        treasure = _randomize_current_level_score(_open_state(1, treasure=True))

        self.assertGreater(treasure, plain)


class RandomScopeAdditionScoreTests(unittest.TestCase):
    def test_scope_score_is_positive_but_not_global_priced(self) -> None:
        state = _open_state(3)
        affix = _catalog_affix(AffixEffect.ADD_RANDOM_SCOPE, delta=0)

        score = _random_scope_addition_score(state, affix)

        self.assertGreater(score, 0.0)
        # 若把「全图」当常态（等概率），期望会被显著抬高；官方分布下全图只占 2.44%，
        # 所以得分必须明显低于"全图常驻"的定价。
        self.assertLess(score, 120.0)


class OfficialWeightSamplingTests(unittest.TestCase):
    def test_level_sampling_follows_official_expectation(self) -> None:
        affix = AffixRule(BrushType.TRANSFORM, "随机改变自身等级", AffixEffect.RANDOMIZE_CURRENT_LEVEL)
        levels = []
        for seed in range(3000):
            result = apply_affix_to_state(_open_state(1), affix, seed=seed)
            self.assertTrue(result.success)
            self.assertIsNotNone(result.state.current_rule)
            levels.append(result.state.current_rule.paint.level)

        mean = sum(levels) / len(levels)
        self.assertAlmostEqual(3.54, mean, delta=0.15)
        # 7 级约 1%，1 级约 7%（等概率会是 14.3%）
        level_7 = levels.count(7) / len(levels)
        level_1 = levels.count(1) / len(levels)
        self.assertLess(abs(level_1 - 0.07), 0.02)
        self.assertLess(abs(level_7 - 0.01), 0.01)
        self.assertLess(level_7, level_1)

    def test_scope_sampling_keeps_global_rare(self) -> None:
        affix = _catalog_affix(AffixEffect.ADD_RANDOM_SCOPE, delta=0)
        extras = []
        for seed in range(3000):
            result = apply_affix_to_state(_open_state(3), affix, seed=seed)
            self.assertTrue(result.success)
            rule = result.state.current_rule
            added = [direction for direction in rule.extra_directions if direction != PaintDirection.VERTICAL]
            self.assertTrue(added)
            extras.append(added[0])

        global_ratio = extras.count(PaintDirection.GLOBAL) / len(extras)
        # 当前范围是纵向，纵向被排除后重新归一化 -> 全图约 3.2%
        self.assertLess(abs(global_ratio - 0.032), 0.02)
        self.assertNotIn(PaintDirection.VERTICAL, extras)


class ClearGlobalVoidAffixTests(unittest.TestCase):
    def _cleared_void_count(self, official_id: int, seed: int) -> int:
        state = _void_state(5)
        result = apply_affix_to_state(state, _official_affix(official_id), seed=seed)

        self.assertTrue(result.success)
        return sum(
            1
            for index in range(5)
            if state.tile(index // 6, index % 6).state == TileState.VOID
            and result.state.tile(index // 6, index % 6).state == TileState.BLANK
        )

    def test_3014_clears_exactly_one_void_tile(self) -> None:
        self.assertEqual(1, self._cleared_void_count(3014, seed=1))

    def test_3015_clears_exactly_two_void_tiles(self) -> None:
        for seed in range(20):
            with self.subTest(seed=seed):
                self.assertEqual(2, self._cleared_void_count(3015, seed=seed))


class PaintLevelDistributionTests(unittest.TestCase):
    """颜料等级出现几率必须走官方权重，而不是自设常数。

    官方唯一给出「1..7 级 → 权重」的数据是 ``4005`` 内嵌的
    ``1:700;2:1800;3:2000;4:3000;5:2000;6:400;7:100``（万分比，期望 3.54）。
    """

    OFFICIAL = {1: 700, 2: 1800, 3: 2000, 4: 3000, 5: 2000, 6: 400, 7: 100}

    def test_weights_are_the_official_ones(self) -> None:
        from huahua_assistant.s8_official_data import PAINT_LEVEL_WEIGHTS

        self.assertEqual(self.OFFICIAL, PAINT_LEVEL_WEIGHTS)
        self.assertEqual(10000, sum(PAINT_LEVEL_WEIGHTS.values()))

    def test_distribution_and_expectation(self) -> None:
        from huahua_assistant.s8_official_data import (
            paint_level_distribution,
            paint_level_expectation,
        )

        distribution = paint_level_distribution()
        self.assertAlmostEqual(1.0, sum(distribution.values()), places=9)
        self.assertAlmostEqual(0.01, distribution[7], places=9)
        self.assertAlmostEqual(3.54, paint_level_expectation(), places=6)

    def test_backpack_levels_sample_the_official_distribution(self) -> None:
        import collections
        import random

        from huahua_assistant.game_model import _random_backpack_levels

        rng = random.Random(2026)
        counts: collections.Counter[int] = collections.Counter()
        for _ in range(3000):
            counts.update(_random_backpack_levels(rng, 6))
        total = sum(counts.values())
        for level, weight in self.OFFICIAL.items():
            with self.subTest(level=level):
                self.assertAlmostEqual(
                    weight / 10000,
                    counts[level] / total,
                    delta=0.015,
                    msg=f"{level} 级实测占比偏离官方权重",
                )

    def test_level_is_drawn_by_official_weight_not_uniform(self) -> None:
        """4 级应明显多于 7 级（30% vs 1%），旧实现只有 3/4/5/6 四档。"""
        import collections
        import random

        from huahua_assistant.game_model import _random_paint

        rng = random.Random(11)
        levels: collections.Counter[int] = collections.Counter()
        for _ in range(2000):
            levels[_random_paint(rng).level] += 1
        self.assertGreater(levels[4], levels[7] * 5)
        self.assertTrue(levels[7] > 0, "官方允许 7 级，模拟不能把它排除")


class PaintFamilyLevelBindingTests(unittest.TestCase):
    """家族与等级段是绑定的：黑帆/黑潮/迷城/梦魇/怪谈/魔方/玩偶 只存在 5~7 级。"""

    def test_catalog_comes_from_official_plotitem_table(self) -> None:
        from huahua_assistant.s8_official_data import (
            S8_FULL_LEVEL_FAMILIES,
            S8_PARTIAL_LEVEL_FAMILIES,
        )

        self.assertIn('宝箱', S8_FULL_LEVEL_FAMILIES)
        self.assertEqual([5, 6, 7], S8_PARTIAL_LEVEL_FAMILIES['黑帆'])
        self.assertEqual([3, 4], S8_PARTIAL_LEVEL_FAMILIES['赛季'])

    def test_low_level_paints_never_use_late_families(self) -> None:
        import random

        from huahua_assistant.game_model import _random_subtype
        from huahua_assistant.models import PaintCategory

        rng = random.Random(5)
        late_families = {'黑帆', '黑潮', '迷城', '梦魇', '怪谈', '魔方', '玩偶'}
        for _ in range(400):
            picked = _random_subtype(rng, PaintCategory.GAMEPLAY, 1)
            self.assertNotIn(picked, late_families, '1 级不可能出现只存在于 5~7 级的家族')
            self.assertNotEqual('赛季', picked, '赛季只存在于 3~4 级')

    def test_high_level_paints_never_use_early_families(self) -> None:
        import random

        from huahua_assistant.game_model import _random_subtype
        from huahua_assistant.models import PaintCategory

        rng = random.Random(6)
        for _ in range(400):
            picked = _random_subtype(rng, PaintCategory.GAMEPLAY, 7)
            self.assertNotEqual('赛季', picked, '赛季只存在于 3~4 级')


if __name__ == "__main__":
    unittest.main()
