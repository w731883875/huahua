"""落笔结算的「额外提升」笔触 + 官方调色目标表。

这两条都来自解包取证：

* ``config_s8_gameplay_itemeffect`` 的 27 条「调色目标/调色效果」成对出现，
  目标与效果的方向/类型一致，提升级数只有 1 或 2，且包含「全图」。
* 官方文案里的「额外提升 N 级」在落笔瞬间结算——此时作用范围里才有刚放下
  的颜料。旧实现选词缀时就结算，范围里还没有目标，等于白选。
"""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from huahua_assistant import game_model as gm
from huahua_assistant import s8_plot_effect as official
from huahua_assistant.models import PaintCategory, PaintDirection, PaintObject


def _rule(*, amount: int = 2, direction: PaintDirection = PaintDirection.HORIZONTAL) -> gm.CurrentPaintRule:
    return gm.CurrentPaintRule(
        paint=gm.Paint(category=PaintCategory.GAMEPLAY, level=6),
        direction=direction,
        target_object=PaintObject.GAMEPLAY,
        match_required=4,
        upgrade_amount=amount,
    )


def _state_with(rule: gm.CurrentPaintRule, backpack_levels: tuple[int, ...]) -> gm.PuzzleState:
    state = gm.random_state(7)
    state = replace(state, current_rule=rule)
    backpack = tuple(
        gm.Paint(category=PaintCategory.GAMEPLAY, level=level) for level in backpack_levels
    )
    return replace(state, backpack=backpack + tuple(state.backpack))


def _scope_affix(amount: int, count: int | None = None) -> gm.AffixRule:
    return next(
        affix
        for affix in gm.build_affix_catalog()
        if affix.effect is gm.AffixEffect.UPGRADE_IN_SCOPE
        and affix.amount == amount
        and affix.count == count
        and affix.direction is None
    )


def _place(state: gm.PuzzleState, level: int) -> gm.InstallResult:
    scope = gm._placement_scope_cells(state, state.current_rule)
    cell = next(c for c in scope if gm._is_placeable(state.tile(*c), state.active_affix))
    index = next(i for i, paint in enumerate(state.backpack) if paint.level == level)
    return gm.install_backpack_paint(state, index, cell[0], cell[1])


class DeferredScopeUpgradeTest(unittest.TestCase):
    def test_scope_upgrade_is_not_settled_on_selection(self) -> None:
        state = _state_with(_rule(), (2,))
        result = gm.apply_affix_to_state(state, _scope_affix(3), seed=0)

        self.assertTrue(result.success)
        self.assertEqual(result.state.active_affix, _scope_affix(3))
        # 选词缀只挂 buff，不动棋盘：必须提示"安装到地块后"才结算
        self.assertIn("安装到地块后", result.message)

    def test_placement_stacks_rule_and_affix_to_level_cap(self) -> None:
        # 官方上限：Lv2 + 调色效果 +2 + 笔触额外提升 +3 → 最终等级 7。
        # 两笔都记成**颜料加成**（2026-11-23 定案），所以颜料自身仍是 Lv2，
        # 加成 2+3=5，最终 7。
        state = _state_with(_rule(amount=2), (2,))
        state = gm.apply_affix_to_state(state, _scope_affix(3), seed=1).state

        result = _place(state, 2)
        finals = [gm.tile_final_level(tile) for tile in result.state.tiles if tile.paint is not None]
        self.assertIn(gm.MAX_LEVEL, finals)
        placed = next(
            tile
            for tile in result.state.tiles
            if tile.paint is not None and gm.tile_bonus_level(tile) == 5
        )
        self.assertEqual(placed.paint.level, 2, "调色效果记加成，不改颜料等级")
        self.assertEqual(gm.tile_final_level(placed), gm.MAX_LEVEL)
        self.assertIn("笔触额外提升", result.message)

    def test_affix_persists_for_whole_run(self) -> None:
        """实机通用规则：范围类 buff 落笔才结算，且**整局有效**。

        旧口径把这类词缀当成"用完即弃"，与实机不符。
        每次落笔都会重新结算，但等级受 ``MAX_LEVEL`` 封顶，不会无限增长。
        """
        state = _state_with(_rule(amount=1), (3, 3))
        state = gm.apply_affix_to_state(state, _scope_affix(1), seed=2).state

        first = _place(state, 3)
        self.assertIsNotNone(first.state.active_affix, 'buff 应当整局保留')
        self.assertIn("笔触额外提升", first.message)

        second = _place(first.state, 3)
        self.assertIn("笔触额外提升", second.message, '第二次落笔也要重新结算')
        highest = max(
            gm.tile_final_level(tile)
            for tile in second.state.tiles
            if tile.paint is not None
        )
        self.assertLessEqual(highest, gm.MAX_LEVEL, '累加不能突破等级上限')

    def test_random_count_still_limits_targets(self) -> None:
        state = _state_with(_rule(amount=1), (2,))
        state = gm.apply_affix_to_state(state, _scope_affix(1, count=1), seed=3).state

        result = _place(state, 2)
        self.assertIn("笔触额外提升1个颜料", result.message)

    def test_same_content_scope_upgrade_also_defers(self) -> None:
        affix = next(
            a
            for a in gm.build_affix_catalog()
            if a.effect is gm.AffixEffect.UPGRADE_SCOPE_SAME_PAINT and a.amount == 2
        )
        state = _state_with(_rule(), (2,))
        result = gm.apply_affix_to_state(state, affix, seed=4)

        self.assertIs(result.state.active_affix, affix)
        self.assertIn("安装到地块后", result.message)


class OfficialPlotEffectTableTest(unittest.TestCase):
    def test_table_matches_official_shape(self) -> None:
        self.assertEqual(len(official.PLOT_EFFECTS), 27)
        self.assertEqual(len({item.id for item in official.PLOT_EFFECTS}), 27)
        self.assertEqual(
            sorted(item.id for item in official.PLOT_EFFECTS),
            sorted(
                list(range(1101, 1109))
                + list(range(2001, 2009))
                + list(range(2010, 2014))
                + list(range(3001, 3007))
                + [4001]
            ),
        )

    def test_amounts_are_one_or_two(self) -> None:
        self.assertEqual({item.amount for item in official.PLOT_EFFECTS}, {1, 2})

    def test_global_direction_exists(self) -> None:
        globals_ = [i for i in official.PLOT_EFFECTS if i.direction is PaintDirection.GLOBAL]
        self.assertEqual([i.id for i in globals_], [3005, 3006, 4001])

    def test_task_and_effect_share_direction_and_type(self) -> None:
        for item in official.PLOT_EFFECTS:
            self.assertIn("提升", item.effect)
            self.assertIn("放置", item.task)
            self.assertIs(official.effect_for(item.direction, item.target, item.amount), item)

    def test_generated_rules_stay_inside_official_combinations(self) -> None:
        directions = set()
        for seed in range(400):
            rule = gm.random_current_rule(random.Random(seed))
            directions.add(rule.direction)
            self.assertIsNotNone(
                official.effect_for(rule.direction, rule.target_object, rule.upgrade_amount),
                msg=f"seed={seed} 生成了官方不存在的组合",
            )
        self.assertIn(PaintDirection.GLOBAL, directions)


class SeasonFamilyTest(unittest.TestCase):
    """赛季 Lv3/Lv4 升到 5 级会显形成具体赛季玩法，但家族不变。

    官方 `201403 玩法颜料-赛季` 原文：「雪原中出现随机赛季怪物；
    **该颜料升至5级及以上时，转变为随机赛季玩法**」。用户 2026-11-18 实测：
    两个赛季 Lv3 升级后变成一个玩偶 6 级、一个黑帆 5 级，属性仍属赛季家族。
    所以「与自身内容相同」的判定必须把这一组归一，否则显形后丢掉自己那份加成。
    """

    def test_official_text_is_the_basis(self) -> None:
        from huahua_assistant.s8_official_data import S8_PLOT_ITEMS

        season = [item for item in S8_PLOT_ITEMS if item.name == "玩法颜料-赛季"]
        self.assertTrue(season)
        for item in season:
            self.assertIn("5级及以上", item.gameplay_des)
            self.assertIn("转变为随机赛季玩法", item.gameplay_des)

    def test_season_shapes_share_one_kind(self) -> None:
        base = gm.Paint(category=PaintCategory.GAMEPLAY, level=3, subtype="赛季")
        doll = gm.Paint(category=PaintCategory.GAMEPLAY, level=6, subtype="玩偶")
        sail = gm.Paint(category=PaintCategory.GAMEPLAY, level=5, subtype="黑帆")
        self.assertEqual(gm._paint_kind_key(base), gm._paint_kind_key(doll))
        self.assertEqual(gm._paint_kind_key(doll), gm._paint_kind_key(sail))

    def test_unrelated_gameplay_is_not_merged(self) -> None:
        base = gm.Paint(category=PaintCategory.GAMEPLAY, level=3, subtype="赛季")
        other = gm.Paint(category=PaintCategory.GAMEPLAY, level=5, subtype="机械之神")
        self.assertNotEqual(gm._paint_kind_key(base), gm._paint_kind_key(other))
        # 怪物家族不能被赛季归一影响
        monster = gm.Paint(category=PaintCategory.MONSTER, level=3, subtype="鼠人")
        self.assertNotEqual(gm._paint_kind_key(base), gm._paint_kind_key(monster))


if __name__ == "__main__":
    unittest.main()
