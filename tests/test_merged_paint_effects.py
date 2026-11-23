"""融合初始颜料：一枚颜料同时拥有 2 套调色效果 / 范围。

用户 2026-11-23 定案：

* 「范围取并集」——融合颜料有两个范围时，两个方向的格子都算在范围内；
* 「相当于这个初始颜料融合了 2 种初始颜料，同时拥有 2 个调色效果、
  2 个笔触效果、2 个范围」；
* 融合的产生方式：初始颜料被炸毁后，它的「范围和笔触效果」**留在锚点地块上**，
  失色恢复成空白后往上落一枚新初始颜料，两套就叠在一起。

配置层面的旁证（见 ``docs/s8_bonus_resource_evidence.md``）：
``config_s8_gameplay_plotitem`` 的 ``EffectLibraryId`` 是**单值**、
``config_s8_gameplay_itemeffect`` 的 ``LibraryId`` 也是**单值**，所以一枚颜料
在配置里只有一条效果 —— 融合必然是运行期把两套并起来的结果。
"""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from huahua_assistant import game_model as gm
from huahua_assistant.models import PaintCategory, PaintDirection, PaintObject

GAMEPLAY = PaintCategory.GAMEPLAY
MONSTER = PaintCategory.MONSTER
ANCHOR = (2, 2)


def _paint(category: PaintCategory = GAMEPLAY, level: int = 2, subtype: str = "存活") -> gm.Paint:
    return gm.Paint(category=category, level=level, subtype=subtype)


def _rule(
    *,
    direction: PaintDirection = PaintDirection.HORIZONTAL,
    target: PaintObject = PaintObject.GAMEPLAY,
    amount: int = 1,
    required: int = 3,
    effects: tuple[gm.PaintEffect, ...] = (),
) -> gm.CurrentPaintRule:
    return gm.CurrentPaintRule(
        paint=_paint(),
        direction=direction,
        target_object=target,
        match_required=required,
        upgrade_amount=amount,
        effects=effects,
    )


def _state(
    cells: dict[tuple[int, int], gm.Paint] | None = None,
    *,
    rule: gm.CurrentPaintRule | None = None,
) -> gm.PuzzleState:
    tiles = [gm.Tile(gm.TileState.BLANK) for _ in range(gm.BOARD_SIZE * gm.BOARD_SIZE)]
    for (row, col), paint in (cells or {}).items():
        tiles[row * gm.BOARD_SIZE + col] = gm.Tile(gm.TileState.PAINT, paint)
    return gm.PuzzleState(
        tiles=tuple(tiles),
        current_rule=rule,
        current_anchor=ANCHOR,
    )


class EffectsAreIndependentTest(unittest.TestCase):
    """各条效果**各自独立结算**，不取范围并集（用户 2026-11-23 纠正）。"""

    def test_identical_effects_are_kept_and_stack(self) -> None:
        """3 枚融合的实机例子里就有「2 条相邻怪物提升1级」——两条各自生效。

        用户 2026-11-23：「融合颜料可以不设置上限，但是颜料加成需要单独计算」。
        所以同方向 + 同对象的两条**不能去重**，必须都留着、都结算（合计 +2）。
        """
        rule = _rule()
        merged = gm._merge_retained_effects(
            rule,
            (gm.PaintEffect(PaintDirection.HORIZONTAL, PaintObject.GAMEPLAY, 1),),
        )

        self.assertEqual(len(merged.effects), 2, "重复的两条都要留")
        self.assertEqual(
            [effect.upgrade_amount for effect in merged.effects],
            [1, 1],
            "不是取高者，是两条并存",
        )

    def test_two_identical_effects_upgrade_a_target_twice(self) -> None:
        """两条「相邻怪物 +1」命中同一枚怪物 → 合计 +2。"""
        rule = _rule(
            effects=(
                gm.PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1),
                gm.PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1),
            )
        )
        state = _state({(1, 2): _paint(MONSTER, 2, "幽灵")}, rule=rule)

        upgraded, _count = gm._apply_current_rule_upgrade(state, rule)

        self.assertEqual(gm.tile_bonus_level(upgraded.tile(1, 2)), 2)

    def test_different_targets_on_the_same_direction_are_kept_apart(self) -> None:
        rule = _rule(target=PaintObject.GAMEPLAY, amount=1)
        merged = gm._merge_retained_effects(
            rule,
            (gm.PaintEffect(PaintDirection.HORIZONTAL, PaintObject.MONSTER, 2),),
        )

        self.assertEqual(len(merged.effects), 2, "同方向不同对象是两条")
        self.assertEqual(
            {effect.target_object for effect in merged.effects},
            {PaintObject.GAMEPLAY, PaintObject.MONSTER},
        )

    def test_mixed_ranges_each_hit_only_their_own_cells(self) -> None:
        """「2 条相邻怪物 +1、1 条斜对角怪物 +1」：各按各的范围算。

        锚点 (2,2)。相邻只看上下左右，斜对角只看四个角。
        """
        adj = gm.PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1)
        diag = gm.PaintEffect(PaintDirection.DIAGONAL, PaintObject.MONSTER, 1)
        rule = _rule(effects=(adj, adj, diag))
        state = _state(
            {
                (1, 2): _paint(MONSTER, 2, "幽灵"),   # 上相邻
                (3, 2): _paint(MONSTER, 2, "幽灵"),   # 下相邻
                (1, 1): _paint(MONSTER, 2, "幽灵"),   # 斜对角
            },
            rule=rule,
        )

        upgraded, _count = gm._apply_current_rule_upgrade(state, rule)

        self.assertEqual(
            gm.tile_bonus_level(upgraded.tile(1, 2)), 2, "相邻两条都命中上下两格 → +2"
        )
        self.assertEqual(gm.tile_bonus_level(upgraded.tile(3, 2)), 2)
        self.assertEqual(
            gm.tile_bonus_level(upgraded.tile(1, 1)), 1, "斜对角那条只给斜对角 +1"
        )


class MergedPaintAppliesBothEffectsTest(unittest.TestCase):
    def test_each_effect_upgrades_only_its_own_target(self) -> None:
        """融合颜料：横向给玩法 +2，纵向给怪物 +1，各管各的目标。"""
        rule = _rule(
            effects=(
                gm.PaintEffect(PaintDirection.HORIZONTAL, PaintObject.GAMEPLAY, 2),
                gm.PaintEffect(PaintDirection.VERTICAL, PaintObject.MONSTER, 1),
            )
        )
        state = _state(
            {
                (2, 4): _paint(GAMEPLAY, 2, "存活"),   # 横向同排、锚点右侧
                (4, 2): _paint(MONSTER, 2, "幽灵"),    # 纵向同列、锚点下方
            },
            rule=rule,
        )

        upgraded, count = gm._apply_current_rule_upgrade(state, rule)

        self.assertEqual(count, 2)
        self.assertEqual(gm.tile_bonus_level(upgraded.tile(2, 4)), 2, "玩法吃到横向 +2")
        self.assertEqual(gm.tile_bonus_level(upgraded.tile(4, 2)), 1, "怪物吃到纵向 +1")

    def test_a_tile_hit_by_both_effects_takes_the_sum(self) -> None:
        """两条效果命中同一格就把两笔加起来 —— 这正是点击地块提示 +3 的由来。"""
        rule = _rule(
            effects=(
                gm.PaintEffect(PaintDirection.HORIZONTAL, PaintObject.GAMEPLAY, 2),
                gm.PaintEffect(PaintDirection.VERTICAL, PaintObject.GAMEPLAY, 1),
            )
        )
        # (2,2) 是锚点；(2,3) 只在横向；(3,2) 只在纵向；(2,2) 自己两向都覆盖但不算
        state = _state({(2, 3): _paint(GAMEPLAY, 2, "存活")}, rule=rule)

        upgraded, _count = gm._apply_current_rule_upgrade(state, rule)

        self.assertEqual(gm.tile_bonus_level(upgraded.tile(2, 3)), 2)


class DestroyedInitialPaintLeavesItsEffectsTest(unittest.TestCase):
    """规则二：初始颜料被炸毁后，效果包留在地块上，等下一枚颜料继承。"""

    def test_force_complete_leaves_the_effect_pack_on_the_anchor(self) -> None:
        rule = _rule(amount=2, required=1, effects=(
            gm.PaintEffect(PaintDirection.HORIZONTAL, PaintObject.GAMEPLAY, 2),
            gm.PaintEffect(PaintDirection.ADJACENT, PaintObject.GAMEPLAY, 1),
        ))
        state = _state(
            {
                ANCHOR: _paint(GAMEPLAY, 2, "存活"),
                (0, 0): _paint(GAMEPLAY, 3, "幽灵"),  # 必须另有一枚背包颜料才能强制完成
            },
            rule=rule,
        )

        # force_used=2 → 本次炸 3 个，棋盘只有 2 枚，两枚都被炸掉，锚点必然在内。
        result = gm.force_complete_current_rule(state, force_used=2, rng=random.Random(0))

        self.assertTrue(result.success)
        self.assertIn(ANCHOR, result.destroyed_cells)
        tile = result.state.tile(*ANCHOR)
        self.assertEqual(tile.state, gm.TileState.FADED)
        self.assertIsNone(tile.paint)
        self.assertEqual(len(tile.retained_effects), 2, "两套效果都留在锚点地块上")

    def test_a_normal_backpack_paint_leaves_nothing_behind(self) -> None:
        """规则一：普通背包颜料被炸 → 全新空白地块，无任何残留。"""
        rule = _rule(amount=0, required=2)
        state = _state(
            {
                ANCHOR: _paint(GAMEPLAY, 2, "存活"),        # 锚点
                (2, 4): _paint(GAMEPLAY, 3, "幽灵"),        # 普通背包颜料
                (0, 0): _paint(GAMEPLAY, 3, "幽灵"),        # 保证可强制完成
            },
            rule=rule,
        )

        result = gm.force_complete_current_rule(state, force_used=2, rng=random.Random(0))

        self.assertTrue(result.success)
        for row, col in result.destroyed_cells:
            tile = result.state.tile(row, col)
            self.assertEqual(gm.tile_bonus_level(tile), 0, "加成随颜料走")
            if (row, col) != ANCHOR:
                self.assertEqual(tile.retained_effects, (), "普通颜料不留效果包")


class MergedPaintIsBuiltByAnchorPlacementTest(unittest.TestCase):
    def test_placing_an_initial_paint_on_a_retained_cell_merges(self) -> None:
        retained = (
            gm.PaintEffect(PaintDirection.VERTICAL, PaintObject.MONSTER, 2),
        )
        tiles = [gm.Tile(gm.TileState.BLANK) for _ in range(gm.BOARD_SIZE * gm.BOARD_SIZE)]
        tiles[ANCHOR[0] * gm.BOARD_SIZE + ANCHOR[1]] = gm.Tile(
            gm.TileState.BLANK, retained_effects=retained
        )
        rule = _rule(
            direction=PaintDirection.HORIZONTAL,
            target=PaintObject.GAMEPLAY,
            amount=1,
            required=1,
        )
        state = gm.PuzzleState(
            tiles=tuple(tiles), current_rule=rule, current_rule_pending=True
        )

        result = gm.place_current_rule_anchor(state, *ANCHOR, rng=random.Random(0))

        self.assertTrue(result.success)
        merged = result.state.current_rule
        self.assertEqual(len(merged.effects), 2, "自己一条 + 留着的一条")
        self.assertEqual(
            {effect.direction for effect in merged.effects},
            {PaintDirection.HORIZONTAL, PaintDirection.VERTICAL},
        )
        self.assertEqual(
            result.state.tile(*ANCHOR).retained_effects, (), "效果包已被继承走，不再重复"
        )


class InitialPaintBasicPartsTest(unittest.TestCase):
    """一枚初始颜料的基本属性，以及两条「只剩一部分」的实机规则。

    用户 2026-11-23 截图口径：初始颜料 = 基础等级 + 颜料内容 + 调色效果 + 笔触效果。

    这两条规则说明「调色效果 / 笔触效果」和「颜料内容」是**两套独立的东西**：

    * 放到虚空 → **只失去颜料内容**，调色效果和笔触效果都还在；
    * 放到普通颜料上 → **变成初始颜料**（1 调色 + 1 笔触 + 1 内容），
      因为普通颜料本来就只有内容、没有那两套效果。
    """

    def _void_affix(self) -> gm.AffixRule:
        return next(
            affix
            for affix in gm.build_affix_catalog()
            if affix.effect is gm.AffixEffect.ALLOW_PLACE_ON_VOID_LOSE_SELF
        )

    def test_placed_on_void_loses_content_but_keeps_both_effects(self) -> None:
        rule = _rule(
            effects=(gm.PaintEffect(PaintDirection.HORIZONTAL, PaintObject.MONSTER, 2),)
        )
        void_tile = gm.Tile(
            gm.TileState.VOID,
            retained_effects=(gm.PaintEffect(PaintDirection.VERTICAL, PaintObject.GAMEPLAY, 1),),
        )

        placed = gm._tile_after_current_anchor_placement(
            void_tile, _paint(GAMEPLAY, 2, "存活"), self._void_affix(), rule=rule
        )

        self.assertIsNone(placed.paint, "颜料内容没了")

        kept = set(placed.retained_effects)
        self.assertIn(
            gm.PaintEffect(PaintDirection.HORIZONTAL, PaintObject.MONSTER, 2),
            kept,
            "这枚初始颜料自带的调色效果还在",
        )
        self.assertIn(
            gm.PaintEffect(PaintDirection.VERTICAL, PaintObject.GAMEPLAY, 1),
            kept,
            "地块原本留着的调色效果也还在",
        )
        self.assertEqual(len(kept), 2, "两条调色效果都在，没有互相顶掉")

    def test_placed_on_a_normal_paint_keeps_that_paint_and_takes_the_effects(self) -> None:
        """用户 2026-11-23 的具体例子。

        背包颜料 = **3 级宝箱**；初始颜料 = **4 级怪物**，
        调色效果 = 相邻范围玩法提升1级，笔触 = 放置到已放置颜料但失去自身内容。

        放上去之后变成：**初始颜料 3 级宝箱** + 相邻范围玩法提升1级 + 那条笔触。
        也就是 **初始颜料自己的 4 级怪物内容丢了**，地块保留原来的 3 级宝箱；
        初始颜料只贡献「调色效果 + 笔触效果」。
        """
        affix = next(
            affix
            for affix in gm.build_affix_catalog()
            if affix.effect is gm.AffixEffect.ALLOW_PLACE_ON_PAINT_LOSE_SELF
        )
        rule = _rule(
            effects=(gm.PaintEffect(PaintDirection.ADJACENT, PaintObject.GAMEPLAY, 1),)
        )
        # 地块上已放着的背包颜料：3 级宝箱
        treasure = gm.Paint(category=GAMEPLAY, level=3, subtype="宝箱", is_treasure=True)
        normal = gm.Tile(gm.TileState.PAINT, treasure)
        # 要放下去的初始颜料：4 级怪物
        initial = _paint(MONSTER, 4, "幽灵")

        placed = gm._tile_after_current_anchor_placement(
            normal, initial, affix, rule=rule
        )

        self.assertEqual(placed.state, gm.TileState.PAINT)
        self.assertIsNotNone(placed.paint)
        self.assertEqual(placed.paint.level, 3, "等级来自地块上的宝箱，不是初始颜料的 4")
        self.assertTrue(placed.paint.is_treasure, "内容仍是宝箱")
        self.assertEqual(placed.paint.subtype, "宝箱")
        self.assertNotEqual(
            placed.paint.category, MONSTER, "初始颜料自己的怪物内容已经失去"
        )
        self.assertEqual(len(rule.effects), 1, "调色效果来自初始颜料，只此一条")
        self.assertIsNone(placed.buff_applied_id, "清掉旧去重标记，好让这次调色生效")
        self.assertIsNone(placed.affix_applied_id)

    def test_a_plain_paint_contributes_no_effects(self) -> None:
        """普通背包颜料被融合时，只贡献内容，不贡献调色/笔触效果。"""
        normal = gm.Paint(category=GAMEPLAY, level=3, subtype="幽灵")

        self.assertEqual(
            getattr(normal, "bonus_parts", ()), (), "普通颜料没有效果来源账"
        )


if __name__ == "__main__":
    unittest.main()
