#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用户 2026-11-25 用四张实机截图定下的加成/融合规则。

四张截图分别在 ``C:\\Users\\8838\\Desktop\\jijiji\\``：

* ``放置虚空失去颜料内容.png``
    「虚空地块 / 调色效果：使横向的怪物颜料提升1级 / 笔触效果：可放置于虚空
    地块中，但会失去自身内容」
* ``2个初始颜料融合颜料.png``
    「怪物颜料-沙怪 LV3」，2 条调色 + 2 条笔触，**无绿字 +N**；
* ``3个初始颜料融合颜料.png``
    「怪物颜料-沙怪 LV3」，3 条调色 + 3 条笔触，**无绿字 +N**；
* ``初始放普通颜料融合变成初始.png``
    「玩法颜料-巨力之神 LV4 +2」，1 条调色 + 1 条笔触。

用户口径原文：

* 「2+2分别代表的是前者是颜料自身等级2级，后者是这个颜料享受的等级加成」
    —— 即 ``tile_final_level`` 的「自身 + 加成 = 显示等级」，**不是**两个颜料相加；
* 「初始颜料的融合不涨等级，只叠加效果（等级由别处决定）」；
* 「2 个融合/3 个融合 暂时并没有吃到对应调色效果、笔触词缀、虚空光环的加成所以
  是没有的，这是正常现象」；
* 「虚空光环只剩调色效果和笔触词缀，**无颜料内容是吃不到任何加成**」；
* 「+2 是因为吃到了其他调色效果+笔触词缀+虚空光环，最终提升了 2 级达到了 lv4」；
* 「别处 = 其他颜料照过来的调色效果 + 笔触词缀 + 虚空光环」。
"""

from __future__ import annotations

import unittest

from huahua_assistant import game_model as gm
from huahua_assistant.game_model import (
    MAX_LEVEL,
    Paint,
    PaintCategory,
    PaintDirection,
    PaintEffect,
    PaintObject,
    PuzzleState,
    Tile,
    TileState,
)
from huahua_assistant.models import Rect


def _monster(level: int, subtype: str = "沙怪", bonus: int = 0, **kwargs: object) -> Paint:
    return Paint(
        category=PaintCategory.MONSTER,
        level=level,
        subtype=subtype,
        bonus_level=bonus,
        **kwargs,  # type: ignore[arg-type]
    )


def _gameplay(level: int, subtype: str = "巨力之神", bonus: int = 0, **kwargs: object) -> Paint:
    return Paint(
        category=PaintCategory.GAMEPLAY,
        level=level,
        subtype=subtype,
        bonus_level=bonus,
        **kwargs,  # type: ignore[arg-type]
    )


def _state(tiles: list[Tile], **kwargs: object) -> PuzzleState:
    return PuzzleState(tiles=tuple(tiles), **kwargs)  # type: ignore[arg-type]


def _blank_board() -> list[Tile]:
    return [Tile(state=TileState.BLANK) for _ in range(36)]


class FusionDoesNotRaiseLevelTests(unittest.TestCase):
    """规则一：初始颜料融合只叠效果，不涨等级。"""

    def test_two_fusions_stack_two_effects_but_level_unchanged(self) -> None:
        """2 枚融合 → 2 调色 + 2 笔触、1 个颜料内容、等级不变（截图②）。

        用户 2026-11-25：「初始颜料的融合不涨等级，只叠加效果（等级由别处决定）」
        以及截图里 2 枚融合仍是 LV3、且无绿字加成。
        """
        paint = _monster(3)
        own = (
            PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1),
        )
        retained = (
            PaintEffect(PaintDirection.DIAGONAL, PaintObject.BLOCKED_TILE, 1),
            PaintEffect(PaintDirection.DIAGONAL, PaintObject.MONSTER, 1),
            PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1),
        )
        rule = gm.CurrentPaintRule(
            paint=_monster(3),
            direction=PaintDirection.ADJACENT,
            target_object=PaintObject.MONSTER,
            match_required=1,
            upgrade_amount=1,
            effects=own,
        )
        merged = gm._merge_retained_effects(rule, retained)  # noqa: SLF001

        self.assertEqual(len(merged.effects), 4, "2 条调色 + 2 条笔触应全部保留")
        # 合并只动 effects，等级相关字段一个都不许变
        self.assertEqual(merged.paint, rule.paint, "融合不该换掉颜料本体")
        self.assertEqual(merged.upgrade_amount, rule.upgrade_amount, "融合不该改提升级数")
        self.assertEqual(merged.target_object, rule.target_object)
        self.assertEqual(merged.match_required, rule.match_required)
        # 关键：融合不碰等级。颜料还是 3 级、加成还是 0，最终等级仍是 3。
        tile = Tile(state=TileState.PAINT, paint=paint)
        self.assertEqual(tile.paint.level, 3)
        self.assertEqual(tile.paint.bonus_level, 0)
        self.assertEqual(gm.tile_final_level(tile), 3, "融合后显示等级必须仍是 LV3")

    def test_three_fusions_keep_six_effects_without_level_gain(self) -> None:
        """3 枚融合 → 3 调色 + 3 笔触、等级仍不变（截图③）。"""
        own = (PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1),)
        retained = (
            PaintEffect(PaintDirection.DIAGONAL, PaintObject.BLOCKED_TILE, 1),
            PaintEffect(PaintDirection.DIAGONAL, PaintObject.MONSTER, 1),
            PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1),
            PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1),
            PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1),
        )
        rule = gm.CurrentPaintRule(
            paint=_monster(3),
            direction=PaintDirection.ADJACENT,
            target_object=PaintObject.MONSTER,
            match_required=1,
            upgrade_amount=1,
            effects=own,
        )
        merged = gm._merge_retained_effects(rule, retained)  # noqa: SLF001
        self.assertEqual(len(merged.effects), 6)
        tile = Tile(state=TileState.PAINT, paint=_monster(3))
        self.assertEqual(gm.tile_final_level(tile), 3, "3 枚融合也不涨等级")

    def test_fusion_never_deduplicates(self) -> None:
        """两条一模一样的调色效果要各自生效（合起来 +2），不能去重。

        用户 2026-11-25 口径 + 截图③的「2 条使相邻的怪物颜料提升1级」：
        两条各自生效，对相邻怪物的真实加成是 +2。
        """
        own = (PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1),)
        retained = (PaintEffect(PaintDirection.ADJACENT, PaintObject.MONSTER, 1),)
        rule = gm.CurrentPaintRule(
            paint=_monster(3),
            direction=PaintDirection.ADJACENT,
            target_object=PaintObject.MONSTER,
            match_required=1,
            upgrade_amount=1,
            effects=own,
        )
        merged = gm._merge_retained_effects(rule, retained)  # noqa: SLF001
        self.assertEqual(len(merged.effects), 2)
        self.assertEqual(tuple(merged.effects), (own[0], retained[0]))


class FinalLevelIsSelfPlusBonusTests(unittest.TestCase):
    """规则二：2 + 2 = 4 指「自身等级 2 + 加成 2」。"""

    def test_self_plus_bonus_equals_displayed_level(self) -> None:
        """截图④：巨力之神 LV4 + 2 —— 自身 2 级 + 加成 2 级（用户 2026-11-25）。"""
        tile = Tile(state=TileState.PAINT, paint=_gameplay(2, bonus=2))
        self.assertEqual(gm.tile_final_level(tile), 4, "2 自身 + 2 加成 = 4")
        self.assertEqual(gm.tile_bonus_level(tile), 2)

    def test_bonus_sources_all_accumulate(self) -> None:
        """别处 = 调色效果 + 笔触词缀 + 虚空光环，三者都算进加成。"""
        paint = _gameplay(
            1,
            bonus=3,
            bonus_parts=(
                (gm.BONUS_SOURCE_RULE, 1),
                (gm.BONUS_SOURCE_AFFIX, 1),
                (gm.BONUS_SOURCE_AURA, 1),
            ),
        )
        tile = Tile(state=TileState.PAINT, paint=paint)
        self.assertEqual(gm.tile_final_level(tile), 4, "1 自身 + 3 加成 = 4")
        self.assertEqual(dict(paint.bonus_parts), {
            gm.BONUS_SOURCE_RULE: 1,
            gm.BONUS_SOURCE_AFFIX: 1,
            gm.BONUS_SOURCE_AURA: 1,
        })

    def test_final_level_caps_at_max(self) -> None:
        tile = Tile(state=TileState.PAINT, paint=_gameplay(5, bonus=5))
        self.assertEqual(gm.tile_final_level(tile), MAX_LEVEL)


class NoContentReceivesNoBonusTests(unittest.TestCase):
    """规则三：没有颜料内容的地块吃不到任何加成。"""

    def test_void_tile_without_paint_has_no_bonus_and_no_level(self) -> None:
        """虚空格只剩调色效果/笔触词缀、无颜料内容 → 吃不到加成（用户 2026-11-25）。"""
        for state in (TileState.VOID, TileState.BLANK, TileState.FADED):
            tile = Tile(state=state)
            self.assertEqual(gm.tile_bonus_level(tile), 0, "%s 格不该有加成" % state)
            self.assertEqual(gm.tile_final_level(tile), 0, "%s 格不该有等级" % state)

    def test_bonus_never_lingers_after_paint_destroyed(self) -> None:
        """颜料被炸毁 → 加成跟着颜料走，不残留在地块上。

        ``tile_bonus_level`` 的注释写明这是 2026-11-23 定案的修正点：
        以前把加成存 ``Tile`` 上，才会出现「颜料被炸掉加成还留在原地」。
        """
        painted = Tile(state=TileState.PAINT, paint=_gameplay(2, bonus=2))
        destroyed = Tile(state=TileState.FADED, paint=None)
        self.assertEqual(gm.tile_final_level(painted), 4)
        self.assertEqual(gm.tile_bonus_level(destroyed), 0)
        self.assertEqual(gm.tile_final_level(destroyed), 0)


class AuraOnlyOnPaintTilesTests(unittest.TestCase):
    """规则四：光环只给「有颜料内容」的格子，没有内容的格恒为 0。"""

    def test_recompute_auras_skips_tiles_without_paint(self) -> None:
        """锚点光环照到空白/虚空/失色格时，那些格子的 aura_level 必须是 0。"""
        tiles = _blank_board()
        # (1,1) 放一枚带光环强度的颜料（aura_amount=3）
        tiles[1 * 6 + 1] = Tile(
            state=TileState.PAINT, paint=_monster(5), aura_amount=3
        )
        # 周围一圈：虚空 / 空白 / 失色 各来一个
        tiles[1 * 6 + 2] = Tile(state=TileState.VOID)
        tiles[2 * 6 + 1] = Tile(state=TileState.BLANK)
        tiles[2 * 6 + 2] = Tile(state=TileState.FADED)

        after = gm._recompute_auras(_state(tiles))  # noqa: SLF001

        for row, col, expect_state in ((1, 2, TileState.VOID), (2, 1, TileState.BLANK), (2, 2, TileState.FADED)):
            tile = after.tile(row, col)
            self.assertEqual(tile.state, expect_state)
            self.assertEqual(tile.aura_level, 0, "%s 格没有颜料内容，光环必须是 0" % expect_state)
            self.assertEqual(gm.tile_final_level(tile), 0)

    def test_void_tile_aura_cannot_raise_its_own_level(self) -> None:
        """虚空格上残留的光环是「照别人」的，不能加到它自己头上。

        用户 2026-11-25：「虚空光环只剩调色效果和笔触词缀，无颜料内容是吃不到
        任何加成」。所以即使视觉上看到该格有光环，它自己的等级仍是 0。
        """
        tiles = _blank_board()
        tiles[3 * 6 + 3] = Tile(state=TileState.VOID, retained_aura_level=4)
        after = gm._recompute_auras(_state(tiles))  # noqa: SLF001
        tile = after.tile(3, 3)
        self.assertEqual(tile.aura_level, 0)
        self.assertEqual(gm.tile_final_level(tile), 0, "虚空格不能因为残留光环变成 4 级")

    def test_aura_reaches_paint_tiles_of_same_kind(self) -> None:
        """同一套规则的正向验证：光环确实会加到同种且有内容的颜料上。"""
        tiles = _blank_board()
        tiles[1 * 6 + 1] = Tile(state=TileState.PAINT, paint=_monster(5), aura_amount=3)
        tiles[1 * 6 + 3] = Tile(state=TileState.PAINT, paint=_monster(2))
        tiles[1 * 6 + 4] = Tile(state=TileState.PAINT, paint=_gameplay(2))

        after = gm._recompute_auras(_state(tiles))  # noqa: SLF001
        same_kind = after.tile(1, 3)
        other_kind = after.tile(1, 4)
        self.assertEqual(same_kind.aura_level, 3, "同种（怪物-沙怪）应吃到 3 级光环")
        self.assertEqual(gm.tile_final_level(same_kind), 5, "2 + 3 = 5")
        self.assertEqual(other_kind.aura_level, 0, "不同种（玩法颜料）吃不到")
        self.assertEqual(gm.tile_final_level(other_kind), 2)


class ScreenshotNumericMatchTests(unittest.TestCase):
    """把四张截图的数字直接对照一遍，防止以后改坏。"""

    def test_screenshot_three_fusions_show_lv3_without_bonus(self) -> None:
        tile = Tile(state=TileState.PAINT, paint=_monster(3))
        self.assertEqual(gm.tile_final_level(tile), 3)
        self.assertEqual(gm.tile_bonus_level(tile), 0, "绿字 +N 为 0 时界面不显示")

    def test_screenshot_juli_zhishen_shows_lv4_plus2(self) -> None:
        tile = Tile(state=TileState.PAINT, paint=_gameplay(2, bonus=2))
        self.assertEqual(gm.tile_final_level(tile), 4, "界面上那行 LV4")
        self.assertEqual(gm.tile_bonus_level(tile), 2, "界面上那行绿色的 +2")


if __name__ == "__main__":
    unittest.main()