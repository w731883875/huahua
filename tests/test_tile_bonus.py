"""地块等级加成（服务端 ``FacadeLevel``）。

实机确认的三条事实，构成了这套模型的全部约束：

1. **加成挂在「地块」上，不挂在颜料上**——颜料被覆盖、炸毁、清空后加成仍在原地，
   所以「先把一格加成叠满，再放一枚颜料」可以直接到 7 级。
2. 最终等级 = min(MaxPlotLevel, 颜料等级 + 地块加成)，上限 7。
3. 游戏内没有「拆下颜料」，只有强制调色；强制调色炸毁的是颜料，不是加成。

各级加成的落点（2026-11-22 口径）：

* 「额外提升 N 级」类**笔触** → 地块加成；
* 普通「提升 N 级」**笔触** → 颜料等级；
* **初始颜料的调色效果** → 地块加成。依据：用户那局 c4r5 被转成机械之神后仍吃到
  +2/+1 到 6 级，点击该地块游戏提示 +3（= 2+1）。写进颜料等级的话，颜料一被覆盖
  这 N 级就没了，与画面永远对不上。
"""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from huahua_assistant import game_model as gm
from huahua_assistant.engine.value import DEFAULT_WEIGHTS, board_value
from huahua_assistant.models import PaintCategory, PaintDirection, PaintObject

GAMEPLAY = PaintCategory.GAMEPLAY


def _rule(*, amount: int = 2, required: int = 4) -> gm.CurrentPaintRule:
    return gm.CurrentPaintRule(
        paint=gm.Paint(category=GAMEPLAY, level=1),
        direction=PaintDirection.HORIZONTAL,
        target_object=PaintObject.GAMEPLAY,
        match_required=required,
        upgrade_amount=amount,
    )


def _state(*, rule: gm.CurrentPaintRule | None = None, backpack: tuple[gm.Paint, ...] = ()) -> gm.PuzzleState:
    tiles = tuple(gm.Tile(gm.TileState.BLANK) for _ in range(gm.BOARD_SIZE * gm.BOARD_SIZE))
    return gm.PuzzleState(
        tiles=tiles,
        current_rule=rule,
        current_anchor=(2, 2),
        backpack=backpack,
    )


def _with_tile(state: gm.PuzzleState, row: int, col: int, tile: gm.Tile) -> gm.PuzzleState:
    tiles = list(state.tiles)
    tiles[row * gm.BOARD_SIZE + col] = tile
    return replace(state, tiles=tuple(tiles))


def _bonus_affix(amount: int = 3) -> gm.AffixRule:
    """官方「对作用范围内的所有颜料额外提升 N 级」——加的是地块加成。"""
    return next(
        affix
        for affix in gm.build_affix_catalog()
        if affix.effect is gm.AffixEffect.UPGRADE_IN_SCOPE
        and affix.amount == amount
        and affix.count is None
        and affix.direction is None
        and affix.grants_bonus
    )


def _plain_affix(amount: int = 1) -> gm.AffixRule:
    """同形但**不加地块加成**的规则——提升直接写在颜料等级上。

    2026-11-19 核对官方表后发现：``UPGRADE_IN_SCOPE`` 这一族在
    ``S8_Gameplay_SelectableEffect`` 里**全部**是「额外提升」（2004~2008），
    目录里已经没有 ``grants_bonus=False`` 的同 effect 条目。所以这里手工构造，
    只用来测 ``as_bonus=False`` 的记账分支，不再依赖目录内容。
    """
    return gm.AffixRule(
        gm.BrushType.STACK,
        f"范围内所有颜料提升{amount}级",
        gm.AffixEffect.UPGRADE_IN_SCOPE,
        amount=amount,
    )


class TileBonusArithmeticTest(unittest.TestCase):
    def test_final_level_is_paint_level_plus_bonus(self) -> None:
        tile = gm.Tile(gm.TileState.PAINT, gm.Paint(GAMEPLAY, 3), bonus_level=2)
        self.assertEqual(gm.tile_bonus_level(tile), 2)
        self.assertEqual(gm.tile_final_level(tile), 5)

    def test_final_level_is_capped_at_seven(self) -> None:
        tile = gm.Tile(gm.TileState.PAINT, gm.Paint(GAMEPLAY, 5), bonus_level=6)
        self.assertEqual(gm.tile_final_level(tile), gm.MAX_LEVEL)

    def test_effective_paint_carries_final_level(self) -> None:
        tile = gm.Tile(gm.TileState.PAINT, gm.Paint(GAMEPLAY, 1), bonus_level=4)
        effective = gm.tile_effective_paint(tile)
        self.assertIsNotNone(effective)
        self.assertEqual(effective.level, 5)

    def test_empty_tile_has_no_effective_paint(self) -> None:
        self.assertIsNone(gm.tile_effective_paint(gm.Tile(gm.TileState.BLANK, bonus_level=6)))


class BonusSurvivesOnTileTest(unittest.TestCase):
    def test_bonus_survives_clearing_the_paint(self) -> None:
        state = _with_tile(_state(), 2, 3, gm.Tile(gm.TileState.PAINT, gm.Paint(GAMEPLAY, 3), bonus_level=4))

        cleared, count = gm._clear_tile_indices(state, [2 * gm.BOARD_SIZE + 3])

        self.assertEqual(count, 1)
        tile = cleared.tile(2, 3)
        self.assertEqual(tile.state, gm.TileState.BLANK)
        self.assertIsNone(tile.paint)
        self.assertEqual(gm.tile_bonus_level(tile), 4)

    def test_bonus_survives_force_complete_destruction(self) -> None:
        # 规则本身不额外提升，单看「炸毁后加成还在」这件事
        state = _with_tile(_state(rule=_rule(amount=0)), 2, 3, gm.Tile(gm.TileState.PAINT, gm.Paint(GAMEPLAY, 3), bonus_level=4))
        # 只留这一格有颜料，强制完成必定炸它
        result = gm.force_complete_current_rule(state, force_used=0, rng=random.Random(0))

        self.assertTrue(result.success)
        tile = result.state.tile(2, 3)
        self.assertEqual(tile.state, gm.TileState.FADED)
        self.assertIsNone(tile.paint)
        self.assertEqual(gm.tile_bonus_level(tile), 4)

    def test_new_paint_inherits_tile_bonus(self) -> None:
        # 清空后剩下 BLANK + 加成 4，落一枚 Lv1（规则本身不额外提升）直接变 5
        state = _with_tile(
            _state(rule=_rule(amount=0), backpack=(gm.Paint(GAMEPLAY, 1),)),
            2,
            3,
            gm.Tile(gm.TileState.BLANK, bonus_level=4),
        )

        result = gm.install_backpack_paint(state, 0, 2, 3, rng=random.Random(0))

        self.assertTrue(result.success)
        tile = result.state.tile(2, 3)
        self.assertEqual(tile.paint.level, 1)
        self.assertEqual(gm.tile_bonus_level(tile), 4)
        self.assertEqual(gm.tile_final_level(tile), 5)


class UpgradeLandsOnTheRightLayerTest(unittest.TestCase):
    def test_extra_upgrade_writes_bonus_not_paint_level(self) -> None:
        state = _with_tile(_state(rule=_rule()), 2, 3, gm.Tile(gm.TileState.PAINT, gm.Paint(GAMEPLAY, 3)))

        upgraded, count = gm._upgrade_scope_paints(state, _bonus_affix(3), random.Random(0))

        self.assertEqual(count, 1)
        tile = upgraded.tile(2, 3)
        self.assertEqual(tile.paint.level, 3, "「额外提升」不该改颜料等级")
        self.assertEqual(gm.tile_bonus_level(tile), 3)
        self.assertEqual(gm.tile_final_level(tile), 6)

    def test_plain_upgrade_writes_paint_level(self) -> None:
        state = _with_tile(_state(rule=_rule()), 2, 3, gm.Tile(gm.TileState.PAINT, gm.Paint(GAMEPLAY, 3)))

        upgraded, count = gm._upgrade_scope_paints(state, _plain_affix(1), random.Random(0))

        self.assertEqual(count, 1)
        tile = upgraded.tile(2, 3)
        self.assertEqual(tile.paint.level, 4)
        self.assertEqual(gm.tile_bonus_level(tile), 0)

    def test_rule_effect_writes_tile_bonus(self) -> None:
        """初始颜料的调色加成也落在地块上（用户 2026-11-22 口径）。

        证据：c4r5 被转成机械之神后仍吃到 +2/+1 到 6 级，且点击该地块游戏提示 +3。
        所以断言最终等级，并锁住「加成没写进颜料等级」——写进颜料等级的话，
        那格颜料一被覆盖，这 N 级就跟着旧颜料一起没了。
        """

        state = _with_tile(_state(rule=_rule(amount=2)), 2, 3, gm.Tile(gm.TileState.PAINT, gm.Paint(GAMEPLAY, 2)))

        upgraded, count = gm._apply_current_rule_upgrade(state, state.current_rule)

        self.assertEqual(count, 1)
        tile = upgraded.tile(2, 3)
        self.assertEqual(tile.paint.level, 2, "调色加成不该改颜料等级")
        self.assertEqual(gm.tile_bonus_level(tile), 2)
        self.assertEqual(gm.tile_final_level(tile), 4)

    def test_rule_effect_is_wasted_once_bonus_fills_the_cap(self) -> None:
        # 加成 +6 时 Lv1 已经是 7 级，再提升颜料等级不产生收益
        state = _with_tile(_state(rule=_rule(amount=2)), 2, 3, gm.Tile(gm.TileState.PAINT, gm.Paint(GAMEPLAY, 1), bonus_level=6))

        upgraded, _count = gm._apply_current_rule_upgrade(state, state.current_rule)

        tile = upgraded.tile(2, 3)
        self.assertEqual(tile.paint.level, 1)
        self.assertEqual(gm.tile_final_level(tile), gm.MAX_LEVEL)


class ValueUsesFinalLevelTest(unittest.TestCase):
    def test_level_one_treasure_with_six_bonus_is_worth_a_level_seven(self) -> None:
        treasure = gm.Paint(GAMEPLAY, 1, subtype="宝箱", is_treasure=True)
        empty = _state()
        plain = _with_tile(empty, 2, 3, gm.Tile(gm.TileState.PAINT, treasure))
        stacked = _with_tile(empty, 2, 3, gm.Tile(gm.TileState.PAINT, treasure, bonus_level=6))

        baseline = board_value(empty)
        # 落一枚 Lv1 宝箱：这一格从 blank 变成 paint，差值应等于 Lv1 宝箱价值
        self.assertAlmostEqual(
            board_value(plain) - (baseline - DEFAULT_WEIGHTS.blank), DEFAULT_WEIGHTS.treasure[1]
        )
        # 同样的 Lv1 宝箱，地块加成 +6 之后价值等于 7 级宝箱
        self.assertAlmostEqual(
            board_value(stacked) - (baseline - DEFAULT_WEIGHTS.blank), DEFAULT_WEIGHTS.treasure[gm.MAX_LEVEL]
        )
        self.assertAlmostEqual(
            board_value(stacked) - board_value(plain),
            DEFAULT_WEIGHTS.treasure[gm.MAX_LEVEL] - DEFAULT_WEIGHTS.treasure[1],
        )

    def test_level7_counts_uses_final_level(self) -> None:
        treasure = gm.Paint(GAMEPLAY, 1, subtype="宝箱", is_treasure=True)
        state = _with_tile(_state(), 2, 3, gm.Tile(gm.TileState.PAINT, treasure, bonus_level=6))

        from huahua_assistant.engine.value import level7_counts

        self.assertEqual(level7_counts(state), (1, 1))


class BankedBonusValueTest(unittest.TestCase):
    def test_blank_tile_holding_bonus_is_not_worthless(self) -> None:
        empty = _state()
        loaded = _with_tile(empty, 2, 3, gm.Tile(gm.TileState.BLANK, bonus_level=6))

        self.assertAlmostEqual(
            board_value(loaded) - board_value(empty),
            DEFAULT_WEIGHTS.empty_tile_value(gm.TileState.BLANK, 6) - DEFAULT_WEIGHTS.blank,
        )
        self.assertGreater(board_value(loaded), board_value(empty))

    def test_loaded_blank_beats_a_fresh_level_one_treasure(self) -> None:
        treasure = gm.Paint(GAMEPLAY, 1, subtype="宝箱", is_treasure=True)
        empty = _state()
        loaded = _with_tile(empty, 2, 3, gm.Tile(gm.TileState.BLANK, bonus_level=6))
        fresh = _with_tile(empty, 2, 3, gm.Tile(gm.TileState.PAINT, treasure))

        # 一格 +6 的空地比一枚 Lv1 宝箱更值钱——这正是"先叠 buff 再落一枚"的理由
        self.assertGreater(board_value(loaded), board_value(fresh))


class SelfCorrectionKeepsAuraTest(unittest.TestCase):
    """反推基础等级「小于 1 级」时，只能清地块加成，**不许动光环**。

    2026-11-22 全库实测：``runtime/auto_run_records.jsonl`` 里 2559 个颜料格样本
    的 ``aura_level`` **全是 0**。根因就是 ``_rebase_observed_levels`` 的自纠正分支
    把 ``aura_amount`` 一起清零了 —— 而 ``aura_amount`` 是「全图和自身相同的颜料
    提升N级」（官方 2014/2015，即工具里的 ``2-17``/``2-18``）挂在地块上的**强度**，
    清掉等于凭空删掉一条玩家已经选过的词缀。

    取舍：光环单独就超过画面读数时，把颜料等级钳到 1、光环原样留着，让
    ``level_reconcile`` 把差异报出来。宁可看见对账红框，也不静默丢词缀。
    """

    def _rebased(self, observed: int, *, bonus: int, aura: int) -> gm.Tile:
        """造一格「画面读数 observed、模型认为加成 bonus+aura」的局面再反推。"""
        state = _with_tile(
            _state(),
            2,
            3,
            gm.Tile(
                gm.TileState.PAINT,
                gm.Paint(GAMEPLAY, observed),
                bonus_level=bonus,
                aura_amount=aura,
                aura_level=aura,
            ),
        )
        return gm._rebase_observed_levels(state).tile(2, 3)

    def test_base_below_one_clears_bonus_but_keeps_aura_amount(self) -> None:
        # 画面 4 级，模型认为 +5 地块加成、+3 光环 → 反推 -4 级，落在自纠正分支
        tile = self._rebased(4, bonus=5, aura=3)

        self.assertEqual(gm.tile_bonus_level(tile), 0, "多算的地块加成该清掉")
        self.assertEqual(tile.aura_amount, 3, "光环强度是词缀，不许清")
        self.assertEqual(tile.aura_level, 3, "派生缓存也不许清")

    def test_rebased_level_still_reproduces_the_observed_level(self) -> None:
        """清掉加成后，颜料等级 + 光环必须仍等于画面读数（这里是 4）。"""

        tile = self._rebased(4, bonus=5, aura=3)

        self.assertEqual(tile.paint.level, 1, "4 - 3 光环 = 1")
        self.assertEqual(gm.tile_final_level(tile), 4, "最终等级不能和画面对不上")

    def test_aura_alone_over_the_reading_clamps_instead_of_wiping(self) -> None:
        """光环单独就超过画面读数：颜料等级钳到 1，光环保留，差异交给对账。"""

        tile = self._rebased(2, bonus=0, aura=5)

        self.assertEqual(tile.paint.level, 1, "钳到最低等级，不出现 0 级")
        self.assertEqual(tile.aura_amount, 5, "光环依然不许清")
        self.assertGreater(
            gm.tile_final_level(tile), 2, "宁可对账报差异，也不静默删词缀"
        )

    def test_no_correction_needed_leaves_aura_untouched(self) -> None:
        tile = self._rebased(6, bonus=1, aura=2)

        self.assertEqual(tile.paint.level, 3, "6 - 1 - 2 = 3")
        self.assertEqual(tile.aura_amount, 2)
        self.assertEqual(gm.tile_bonus_level(tile), 1)


if __name__ == "__main__":
    unittest.main()
