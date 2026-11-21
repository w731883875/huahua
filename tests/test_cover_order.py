#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「覆盖」类笔触与调色效果的结算顺序回归测试。

用户 2026-11-22 实机口径：

    3 级初始颜料；笔触 =「将作用范围内的颜料转变为同等级的随机颜料」（官方 2009）；
    调色效果 =「使玩法颜料提升 1 级」；范围内玩家已经装了 4 枚颜料。

    实际画面：4 枚先被随机成 **3 个玩法 + 1 个怪物（仍是 3 级）**，
    随后调色按**新类型**只给那 3 个玩法 +1
    → **3 个 4 级玩法 + 1 个 3 级怪物**。

也就是说：**覆盖先结算、调色后结算**，且调色判定用的是覆盖**之后**的类型。
反过来（先调色再覆盖）会把怪物那格也升到 4 级，与画面永远对不上。

本文件用一条与随机结果无关的不变式锁住这件事：

    被覆盖的格子，等级 = 3 + (1 if 覆盖后是玩法 else 0)

无论服务端把它们随机成什么类型，这条都必须成立；顺序错了就会大量出现
「怪物 4 级」或「玩法 3 级」。下面另有一条负例，用旧顺序跑同一判据，
证明这个断言确实会红（不是恒真检查）。
"""
from __future__ import annotations

import random
import unittest

from huahua_assistant import game_model as gm

SIZE = gm.BOARD_SIZE
ANCHOR = (3, 2)
#: 作用范围（第 3 行除锚点）：玩家已经装好的 4 枚颜料，初始等级都是 3。
FILLED = ((3, 0), (3, 1), (3, 3), (3, 4))
#: 不在作用范围内的一个空格，用来做「本帧新落笔」
OUTSIDE = (0, 0)


def _affix(effect: gm.AffixEffect) -> gm.AffixRule:
    return next(a for a in gm.build_affix_catalog() if a.effect is effect)


def _state(*, target: gm.PaintObject = gm.PaintObject.GAMEPLAY, upgrade: int = 1) -> gm.PuzzleState:
    """6x6 空盘 + 锚点 + 作用范围内 4 枚 3 级颜料（2 玩法 2 怪物）。"""
    tiles = [gm.Tile(state=gm.TileState.BLANK) for _ in range(SIZE * SIZE)]
    tiles[ANCHOR[0] * SIZE + ANCHOR[1]] = gm.Tile(
        state=gm.TileState.PAINT,
        paint=gm.Paint(gm.PaintCategory.GAMEPLAY, 3, "玩法甲"),
    )
    for order, (row, col) in enumerate(FILLED):
        category = gm.PaintCategory.GAMEPLAY if order < 2 else gm.PaintCategory.MONSTER
        tiles[row * SIZE + col] = gm.Tile(
            state=gm.TileState.PAINT,
            paint=gm.Paint(category, 3, f"颜{order}"),
        )
    rule = gm.CurrentPaintRule(
        paint=gm.Paint(gm.PaintCategory.GAMEPLAY, 3, "玩法甲"),
        direction=gm.PaintDirection.HORIZONTAL,
        target_object=target,
        upgrade_amount=upgrade,
        match_required=4,
    )
    return gm.PuzzleState(
        tiles=tuple(tiles),
        current_rule=rule,
        current_anchor=ANCHOR,
        backpack=(gm.Paint(gm.PaintCategory.MONSTER, 3, "背包怪"),),
    )


def _violations(state: gm.PuzzleState) -> list[tuple[int, int, str, int]]:
    """不变式检查：覆盖后是玩法就该 4 级，是怪物就该 3 级。"""
    bad: list[tuple[int, int, str, int]] = []
    for row, col in FILLED:
        paint = state.tile(row, col).paint
        if paint is None:
            bad.append((row, col, "空白", 0))
            continue
        expected = 4 if paint.category is gm.PaintCategory.GAMEPLAY else 3
        if paint.level != expected:
            bad.append((row, col, paint.category.value, paint.level))
    return bad


class CoverBeforeTintTest(unittest.TestCase):
    """覆盖先、调色后 —— 用户 2026-11-22 实机口径。"""

    def test_covered_cells_only_level_up_when_they_end_up_gameplay(self) -> None:
        state = gm.replace(_state(), active_affix=_affix(gm.AffixEffect.RANDOMIZE_SCOPE_ANY))
        settled, _upgraded, covered, _note = gm.settle_place_effects(
            state, state.current_rule, random.Random(1)
        )

        self.assertTrue(covered, "范围内有 4 枚颜料，覆盖应当真的动手")
        self.assertEqual(_violations(settled), [])
        for row, col in FILLED:
            # 2009「同等级」替换会清掉这一格之前累积的地块加成
            self.assertEqual(settled.tile(row, col).bonus_level, 0)

    def test_anchor_is_never_covered(self) -> None:
        state = gm.replace(_state(), active_affix=_affix(gm.AffixEffect.RANDOMIZE_SCOPE_ANY))
        settled, _upgraded, _covered, _note = gm.settle_place_effects(
            state, state.current_rule, random.Random(1)
        )
        anchor = settled.tile(*ANCHOR).paint
        self.assertIsNotNone(anchor)
        # 锚点不在作用范围内，类型必须原样保留
        self.assertIs(anchor.category, gm.PaintCategory.GAMEPLAY)

    def test_copy_self_makes_every_covered_cell_the_same_kind(self) -> None:
        """2012「等级和内容替换为自身」：覆盖后全是自身类型，随后调色再 +1。"""
        state = gm.replace(_state(), active_affix=_affix(gm.AffixEffect.COPY_SELF_TO_SCOPE))
        settled, _upgraded, covered, _note = gm.settle_place_effects(
            state, state.current_rule, random.Random(1)
        )

        self.assertTrue(covered)
        rule_paint = state.current_rule.paint
        for row, col in FILLED:
            paint = settled.tile(row, col).paint
            self.assertIsNotNone(paint)
            self.assertIs(paint.category, rule_paint.category)
            self.assertEqual(paint.subtype, rule_paint.subtype)
            # 等级 = 自身 3 级 + 调色 1 级
            self.assertEqual(paint.level, 4)

    def test_random_category_affix_respects_its_target(self) -> None:
        """2011「随机玩法颜料」：覆盖后全变成玩法，于是全吃调色 → 全 4 级。"""
        state = gm.replace(
            _state(), active_affix=_affix(gm.AffixEffect.RANDOMIZE_SCOPE_CATEGORY)
        )
        settled, _upgraded, _covered, _note = gm.settle_place_effects(
            state, state.current_rule, random.Random(1)
        )
        for row, col in FILLED:
            paint = settled.tile(row, col).paint
            self.assertIsNotNone(paint)
            self.assertIs(paint.category, gm.PaintCategory.GAMEPLAY)
            self.assertEqual(paint.level, 4)


class LegacyOrderWouldFailTest(unittest.TestCase):
    """负例自检：按旧顺序（先调色、后覆盖）跑，同一判据必须报出差异。

    不做这条，就无法排除「测试恒真」——比如覆盖根本没生效、调色没被算进去，
    或者不变式本身写成了永远成立的空条件。
    """

    def test_old_order_violates_the_invariant(self) -> None:
        state = gm.replace(_state(), active_affix=_affix(gm.AffixEffect.RANDOMIZE_SCOPE_ANY))
        rng = random.Random(1)

        # 旧顺序：先调色，再覆盖
        legacy, _count = gm._apply_current_rule_upgrade(state, state.current_rule)
        legacy, _covered = gm._settle_scope_buff(legacy, rng)

        self.assertNotEqual(
            _violations(legacy),
            [],
            "旧顺序居然也满足不变式，说明这条断言抓不到顺序错误，需要换判据",
        )


class CoverReconcileTest(unittest.TestCase):
    """对账侧：覆盖类笔触以画面类型为准，不该因为随机类型而逐格报红。"""

    def test_reconcile_accepts_the_observed_types(self) -> None:
        from huahua_assistant.level_reconcile import reconcile

        affix = _affix(gm.AffixEffect.RANDOMIZE_SCOPE_ANY)
        previous = gm.replace(_state(), active_affix=affix)

        # 模拟服务端摇出来的画面：3 个玩法（4 级）+ 1 个怪物（3 级）
        tiles = list(previous.tiles)
        categories = [
            gm.PaintCategory.GAMEPLAY,
            gm.PaintCategory.GAMEPLAY,
            gm.PaintCategory.GAMEPLAY,
            gm.PaintCategory.MONSTER,
        ]
        for (row, col), category in zip(FILLED, categories):
            level = 4 if category is gm.PaintCategory.GAMEPLAY else 3
            tiles[row * SIZE + col] = gm.Tile(
                state=gm.TileState.PAINT,
                paint=gm.Paint(category, level, f"摇{row}{col}"),
            )
        # 本帧在范围外落了一枚（对账要求「有新的落笔」才会跑）
        tiles[OUTSIDE[0] * SIZE + OUTSIDE[1]] = gm.Tile(
            state=gm.TileState.PAINT,
            paint=gm.Paint(gm.PaintCategory.MONSTER, 3, "新落"),
        )
        current = gm.replace(previous, tiles=tuple(tiles))

        report = reconcile(previous, current, active_affix=affix)

        self.assertTrue(report.ran, report.note)
        self.assertEqual(report.mismatches, (), report.summary())
        self.assertIn("覆盖", report.caveat)
        # 3 个玩法各升 1 级 → 模型预计升 3 格；怪物那格不动
        self.assertEqual(report.upgrade_cells, (3, 3), report.summary())

    def test_reconcile_still_flags_a_genuinely_wrong_level(self) -> None:
        """负例：画面把怪物那格读成 4 级 → 必须报红，不能拿「覆盖」当挡箭牌。"""
        from huahua_assistant.level_reconcile import reconcile

        affix = _affix(gm.AffixEffect.RANDOMIZE_SCOPE_ANY)
        previous = gm.replace(_state(), active_affix=affix)

        tiles = list(previous.tiles)
        categories = [
            gm.PaintCategory.GAMEPLAY,
            gm.PaintCategory.GAMEPLAY,
            gm.PaintCategory.GAMEPLAY,
            gm.PaintCategory.MONSTER,
        ]
        for (row, col), category in zip(FILLED, categories):
            level = 4 if category is gm.PaintCategory.GAMEPLAY else 3
            tiles[row * SIZE + col] = gm.Tile(
                state=gm.TileState.PAINT,
                paint=gm.Paint(category, level, f"摇{row}{col}"),
            )
        # 故意把怪物那格写成 4 级（覆盖不改等级，它最多 3）
        monster_cell = FILLED[3]
        tiles[monster_cell[0] * SIZE + monster_cell[1]] = gm.Tile(
            state=gm.TileState.PAINT,
            paint=gm.Paint(gm.PaintCategory.MONSTER, 4, "错读"),
        )
        tiles[OUTSIDE[0] * SIZE + OUTSIDE[1]] = gm.Tile(
            state=gm.TileState.PAINT,
            paint=gm.Paint(gm.PaintCategory.MONSTER, 3, "新落"),
        )
        current = gm.replace(previous, tiles=tuple(tiles))

        report = reconcile(previous, current, active_affix=affix)

        self.assertTrue(report.ran, report.note)
        self.assertTrue(report.mismatches, "怪物那格被读成 4 级，应当报红")


class AnimationCrossCheckTest(unittest.TestCase):
    """「模型预计升几格」对「动画播了几条」——纯数据层判定。"""

    def _report(self, *, low: int, high: int, animations: int | None):
        from huahua_assistant.level_reconcile import ReconcileReport

        return ReconcileReport(
            ran=True,
            levels=((0, 0, 4, 4),),
            upgrade_cells=(low, high),
            animation_upgrades=animations,
        )

    def test_count_inside_range_is_a_match(self) -> None:
        report = self._report(low=2, high=4, animations=3)
        self.assertTrue(report.animation_matches)
        self.assertIn("吻合", report.summary())

    def test_count_below_range_is_flagged(self) -> None:
        report = self._report(low=3, high=3, animations=1)
        self.assertFalse(report.animation_matches)
        self.assertIn("对不上", report.summary())

    def test_count_above_range_is_flagged(self) -> None:
        report = self._report(low=1, high=2, animations=5)
        self.assertFalse(report.animation_matches)

    def test_missing_log_is_not_a_verdict(self) -> None:
        report = self._report(low=3, high=3, animations=None)
        self.assertIsNone(report.animation_matches)
        self.assertIn("预计升3格", report.summary())


if __name__ == "__main__":
    unittest.main()
