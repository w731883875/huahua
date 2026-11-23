"""地块 buff 的两条特殊设定（实机确认）。

**设定 1：「与自身相同的颜料额外提升 x 级」是地块 buff，不是颜料 buff。**

实机现象（用户确认）：

1. 初始颜料（比如幽灵）放在某一格，全图幽灵额外 +x；
2. 这枚初始颜料被炸掉、地块变失色 → **别处的幽灵掉级**；
3. 失色地块恢复后放一枚**别的**颜料（比如存活）→ 全图存活立刻 +x。

第 3 步是判据：如果 buff 挂在颜料上，幽灵被炸走时 buff 就跟着消失了，换一枚存活
不可能继承。它继承了，说明 **buff 挂在那一格地块上**，「自身」指的是这一格当前
放着的那枚颜料。

**设定 2：「范围内颜料变成随机颜料」是**改写内容**，不是炸毁。**

被随机化的格子，颜料实例还在（只是内容变了），所以：

- 它身上那笔**颜料加成**（``Paint.bonus_level``）跟着这枚颜料继续存在；
- 它作为锚点挂着的**笔触光环强度**（``Tile.aura_amount``）也还在，只是作用对象
  改成了改写后的新内容 —— 原来吃这条光环的旧种颜料立刻掉级。

只有**炸毁**（强制完成 → 失色）才是真销毁：颜料没了，它自己的加成一并消失；
但光环强度因为挂在**地块**上，仍然留着，等这一格重新落笔后改挂到新颜料上。

模型上的落地：

- ``Paint.bonus_level`` 是颜料自己的加成，跟着颜料生命周期；
- ``Tile.aura_amount`` 记在锚点地块上，是光环强度；
- ``Tile.aura_level`` 是派生缓存，每次棋盘变动后由 ``_recompute_auras`` 重算；
- 最终等级 = 颜料等级 + 颜料加成 + ``aura_level``，封顶 7。
"""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from huahua_assistant import game_model as gm
from huahua_assistant.models import PaintCategory, PaintDirection, PaintObject

GHOST = gm.Paint(PaintCategory.MONSTER, 1, subtype="幽灵")
SURVIVE = gm.Paint(PaintCategory.GAMEPLAY, 1, subtype="存活")


def _rule(paint: gm.Paint = GHOST, *, direction: PaintDirection = PaintDirection.HORIZONTAL) -> gm.CurrentPaintRule:
    return gm.CurrentPaintRule(
        paint=paint,
        direction=direction,
        target_object=PaintObject.ALL,
        match_required=4,
        upgrade_amount=0,
    )


def _state(rule: gm.CurrentPaintRule | None = None) -> gm.PuzzleState:
    tiles = tuple(gm.Tile(gm.TileState.BLANK) for _ in range(gm.BOARD_SIZE * gm.BOARD_SIZE))
    return gm.PuzzleState(tiles=tiles, current_rule=rule, current_anchor=(2, 2))


def _with_tile(state: gm.PuzzleState, row: int, col: int, tile: gm.Tile) -> gm.PuzzleState:
    tiles = list(state.tiles)
    tiles[row * gm.BOARD_SIZE + col] = tile
    return replace(state, tiles=tuple(tiles))


def _paint_at(state: gm.PuzzleState, row: int, col: int, paint: gm.Paint) -> gm.PuzzleState:
    return _with_tile(state, row, col, gm.Tile(gm.TileState.PAINT, paint))


def _global_same_affix(amount: int = 3) -> gm.AffixRule:
    """官方「对全图中与自身内容相同的颜料额外提升 3 级」。"""
    return next(
        affix
        for affix in gm.build_affix_catalog()
        if affix.effect is gm.AffixEffect.UPGRADE_GLOBAL_SAME_PAINT
        and affix.grants_bonus
        and affix.amount == amount
    )


def _randomize_affix() -> gm.AffixRule:
    return gm.AffixRule(
        gm.BrushType.STACK,
        "范围内颜料变成随机类型颜料",
        gm.AffixEffect.RANDOMIZE_SCOPE_ANY,
    )


def _board_with_aura() -> gm.PuzzleState:
    """锚点(2,2)是幽灵，(2,3)是另一枚幽灵，(2,4)是存活，三格都装着 Lv1。"""
    state = _state(_rule())
    state = _paint_at(state, 2, 2, GHOST)
    state = _paint_at(state, 2, 3, GHOST)
    state = _paint_at(state, 2, 4, SURVIVE)
    state, _count = gm._upgrade_global_same_paints(state, _global_same_affix(3))
    return state


class AuraAnchorsOnTheTileTest(unittest.TestCase):
    def test_aura_lands_on_the_anchor_tile(self) -> None:
        state = _board_with_aura()

        self.assertEqual(state.tile(2, 2).aura_amount, 3, "强度应记在锚点地块上")
        self.assertEqual(state.tile(2, 3).aura_amount, 0, "别的格子不该背强度")
        self.assertEqual(gm.tile_bonus_level(state.tile(2, 2)), 0, "光环不是普通地块加成")

    def test_same_kind_paints_get_the_aura(self) -> None:
        state = _board_with_aura()

        self.assertEqual(state.tile(2, 3).aura_level, 3)
        self.assertEqual(gm.tile_final_level(state.tile(2, 3)), 4, "Lv1 幽灵 + 3 = 4")
        self.assertEqual(state.tile(2, 4).aura_level, 0, "存活不该吃到幽灵的光环")

    def test_aura_is_capped_at_seven(self) -> None:
        tile = gm.Tile(gm.TileState.PAINT, gm.Paint(PaintCategory.GAMEPLAY, 5, subtype="存活"), aura_level=3)
        self.assertEqual(gm.tile_final_level(tile), gm.MAX_LEVEL)


class AuraFollowsTheTileTest(unittest.TestCase):
    def test_destroying_the_anchor_drops_the_other_paints(self) -> None:
        state = _board_with_aura()

        # 锚点颜料被炸掉 → 地块变失色、没有颜料
        destroyed = _with_tile(state, 2, 2, replace(state.tile(2, 2), state=gm.TileState.FADED, paint=None))
        destroyed = gm._recompute_auras(destroyed)

        self.assertEqual(destroyed.tile(2, 2).aura_amount, 3, "buff 还在这一格上")
        self.assertEqual(destroyed.tile(2, 3).aura_level, 0, "别处的幽灵掉级")
        self.assertEqual(gm.tile_final_level(destroyed.tile(2, 3)), 1)

    def test_restoring_with_a_new_paint_rekeys_the_aura(self) -> None:
        state = _board_with_aura()
        destroyed = _with_tile(state, 2, 2, replace(state.tile(2, 2), state=gm.TileState.FADED, paint=None))
        destroyed = gm._recompute_auras(destroyed)

        # 失色地块恢复，放一枚「存活」
        restored_tile = gm._tile_after_current_anchor_placement(destroyed.tile(2, 2), SURVIVE, None)
        restored = _with_tile(destroyed, 2, 2, restored_tile)
        restored = gm._recompute_auras(restored)

        self.assertEqual(restored.tile(2, 2).aura_amount, 3, "buff 继承下来了")
        self.assertEqual(restored.tile(2, 4).aura_level, 3, "全图存活吃到 +3")
        self.assertEqual(gm.tile_final_level(restored.tile(2, 4)), 4)
        self.assertEqual(restored.tile(2, 3).aura_level, 0, "幽灵不再吃")

    def test_bonus_level_alone_is_not_an_aura(self) -> None:
        """颜料加成不参与光环重算；颜料被清掉时它跟着走。"""
        state = _with_tile(_state(), 2, 3, gm.Tile(gm.TileState.PAINT, replace(GHOST, bonus_level=4)))

        cleared, _count = gm._clear_tile_indices(state, [2 * gm.BOARD_SIZE + 3])

        self.assertEqual(gm.tile_bonus_level(cleared.tile(2, 3)), 0, "颜料没了，加成也没了")
        self.assertEqual(cleared.tile(2, 3).aura_level, 0)


class RandomizeOverwritesTheBuffTest(unittest.TestCase):
    def test_randomize_keeps_the_paint_bonus(self) -> None:
        """随机化是改写内容、不是炸毁 → 颜料身上那笔加成跟着走。"""
        state = _state(_rule())
        state = _paint_at(state, 2, 2, GHOST)
        state = _with_tile(state, 2, 3, gm.Tile(gm.TileState.PAINT, replace(GHOST, bonus_level=4)))

        randomized, count = gm._randomize_scope_paints(state, _randomize_affix(), random.Random(0))

        self.assertEqual(count, 1)
        tile = randomized.tile(2, 3)
        self.assertEqual(gm.tile_bonus_level(tile), 4, "转化不是销毁，加成不丢")

    def test_randomize_retargets_the_anchor_aura(self) -> None:
        """光环强度留在地块上，但作用对象改成改写后的新内容。"""
        state = _board_with_aura()
        # 再给 (2,3) 也挂一条 +3 光环：两条同种光环叠加，幽灵吃 6 级
        tiles = list(state.tiles)
        tiles[2 * gm.BOARD_SIZE + 3] = replace(tiles[2 * gm.BOARD_SIZE + 3], aura_amount=3)
        state = gm._recompute_auras(replace(state, tiles=tuple(tiles)))
        self.assertEqual(state.tile(2, 2).aura_level, 6, "两条 +3 光环叠加")

        randomized, _count = gm._randomize_scope_paints(state, _randomize_affix(), random.Random(0))

        self.assertEqual(
            randomized.tile(2, 3).aura_amount, 3, "笔触强度挂在锚点地块上，改写内容不删它"
        )
        if not gm._is_same_paint_kind(randomized.tile(2, 3).paint, GHOST):
            self.assertEqual(
                randomized.tile(2, 2).aura_level, 3, "那格改成别的颜料了，幽灵只剩锚点自己那条 +3"
            )


if __name__ == "__main__":
    unittest.main()
