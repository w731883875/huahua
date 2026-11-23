#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「作用范围内颜料类型与等级替换为自身」口径回归测试。

实机确认（2026-11-12）：

1. 这是 **buff**，不会凭空生成颜料：选词缀那一刻棋盘不变；
2. 必须把初始颜料**安装到地块**才结算；
3. 等级 = 自身等级 **+ 该地块自己的升级 buff**
   （5 级宝箱 + 提升 1 级 = 6 级；该地块再有别的升级 buff 就是 6+x）；
4. **整局有效**，除非该地块被再次替换；
5. 只有空白格吃这个 buff；虚空/失色/阻挡**变成空白之后**也会吃。
"""
from __future__ import annotations

import unittest
from collections import Counter
import random

from huahua_assistant import game_model as gm
from huahua_assistant import practice as pr
from huahua_assistant.engine.actions import ActionKind
from huahua_assistant.engine.rules import RuleEngine


def _affix() -> gm.AffixRule:
    return next(
        a for a in gm.build_affix_catalog()
        if a.effect is gm.AffixEffect.COPY_SELF_TO_SCOPE
    )


def _anchored_state(pet_level: int = 5) -> gm.PuzzleState:
    """开一局并落好锚点，返回局面。"""
    engine = RuleEngine()
    session = pr.start_practice(2, pet_level=pet_level, rules=2, wait_for_anchor=True)
    action = next(
        a for a in engine.legal_actions(session.state)
        if a.kind is ActionKind.PLACE_ANCHOR
    )
    session.place_anchor(action.row, action.col)
    return session.state


def _global_state(state: gm.PuzzleState, extra: list[gm.Paint]) -> tuple[gm.PuzzleState, list[tuple[int, int]]]:
    """把作用方向改成全图，并往空白格塞若干枚普通颜料。"""
    state = gm.replace(
        state, current_rule=gm.replace(state.current_rule, direction=gm.PaintDirection.GLOBAL)
    )
    tiles = list(state.tiles)
    placed: list[tuple[int, int]] = []
    for index, tile in enumerate(tiles):
        if len(placed) >= len(extra):
            break
        if tile.state is not gm.TileState.BLANK:
            continue
        row, col = divmod(index, gm.BOARD_SIZE)
        if (row, col) == state.current_anchor:
            continue
        tiles[index] = gm.replace(tile, state=gm.TileState.PAINT, paint=extra[len(placed)])
        placed.append((row, col))
    return gm.replace(state, tiles=tuple(tiles)), placed


class CopySelfIsBuffTest(unittest.TestCase):
    """第 1、2 条：是 buff，不凭空生成。"""

    def test_selecting_affix_does_not_change_board(self) -> None:
        state = _anchored_state()
        before = Counter(tile.state.value for tile in state.tiles)
        result = gm.apply_affix_to_state(state, _affix(), seed=1)
        after = Counter(tile.state.value for tile in result.state.tiles)
        self.assertEqual(before, after, '选词缀那一刻棋盘不应变化')
        self.assertIsNotNone(result.state.active_affix, '必须要挂成 buff')
        self.assertIn('安装', result.message)

    def test_blank_cells_never_become_paint_by_themselves(self) -> None:
        state = _anchored_state()
        state = gm.apply_affix_to_state(state, _affix(), seed=1).state
        state = gm.replace(
            state,
            current_rule=gm.replace(state.current_rule, direction=gm.PaintDirection.GLOBAL),
        )
        blanks_before = [
            index for index, tile in enumerate(state.tiles)
            if tile.state is gm.TileState.BLANK
        ]
        settled, _count = gm._copy_self_to_scope(state)
        for index in blanks_before:
            self.assertIs(
                settled.tiles[index].state, gm.TileState.BLANK,
                '没有被安装过的空白格不能凭空变成颜料',
            )


class CopySelfLevelTest(unittest.TestCase):
    """第 3 条：等级 = 自身等级 + 地块升级 buff。"""

    def test_converted_paint_takes_self_type_and_level(self) -> None:
        state = _anchored_state()
        state = gm.apply_affix_to_state(state, _affix(), seed=1).state
        other = gm.Paint(category=gm.PaintCategory.GAMEPLAY, level=1, subtype='存活')
        state, placed = _global_state(state, [other, other])
        self_paint = state.current_rule.paint

        settled, count = gm._copy_self_to_scope(state)
        self.assertEqual(count, len(placed))
        for row, col in placed:
            tile = settled.tile(row, col)
            self.assertEqual(tile.paint.category, self_paint.category)
            self.assertEqual(tile.paint.subtype, self_paint.subtype)
            self.assertEqual(gm.tile_final_level(tile), self_paint.level)

    def test_tile_bonus_adds_on_top(self) -> None:
        """自身 Lv5 + 地块 +1 → 最终 6 级；再加 x → 6+x。"""
        state = _anchored_state()
        state = gm.apply_affix_to_state(state, _affix(), seed=1).state
        other = gm.Paint(category=gm.PaintCategory.GAMEPLAY, level=1, subtype='存活')
        state, placed = _global_state(state, [other])
        settled, _count = gm._copy_self_to_scope(state)

        row, col = placed[0]
        index = row * gm.BOARD_SIZE + col
        self_level = state.current_rule.paint.level

        for bonus in (1, 2):
            tiles = list(settled.tiles)
            tiles[index] = gm.replace(
                tiles[index], paint=gm.replace(tiles[index].paint, bonus_level=bonus)
            )
            boosted = gm.replace(settled, tiles=tuple(tiles))
            self.assertEqual(
                gm.tile_final_level(boosted.tile(row, col)),
                min(gm.MAX_LEVEL, self_level + bonus),
                f'颜料 +{bonus} 应当叠加到自身等级上',
            )

    def test_level_capped_at_max(self) -> None:
        state = _anchored_state()
        state = gm.apply_affix_to_state(state, _affix(), seed=1).state
        other = gm.Paint(category=gm.PaintCategory.GAMEPLAY, level=1, subtype='存活')
        state, placed = _global_state(state, [other])
        settled, _count = gm._copy_self_to_scope(state)
        row, col = placed[0]
        tiles = list(settled.tiles)
        tiles[row * gm.BOARD_SIZE + col] = gm.replace(
            tiles[row * gm.BOARD_SIZE + col],
            paint=gm.replace(
                tiles[row * gm.BOARD_SIZE + col].paint, bonus_level=gm.MAX_LEVEL
            ),
        )
        boosted = gm.replace(settled, tiles=tuple(tiles))
        self.assertEqual(gm.tile_final_level(boosted.tile(row, col)), gm.MAX_LEVEL)


class CopySelfPersistenceTest(unittest.TestCase):
    """第 4 条：整局有效。"""

    def test_affix_survives_anchor_resolution(self) -> None:
        state = _anchored_state()
        state = gm.apply_affix_to_state(state, _affix(), seed=1).state
        resolved, _message = gm._resolve_anchor_affix(state, _affix())
        self.assertIsNotNone(
            resolved.active_affix, '落笔结算后 buff 不能消失，实机是整局有效'
        )

    def test_affix_survives_repeated_installs(self) -> None:
        state = _anchored_state()
        state = gm.apply_affix_to_state(state, _affix(), seed=1).state
        other = gm.Paint(category=gm.PaintCategory.GAMEPLAY, level=1, subtype='存活')
        state, placed = _global_state(state, [other])
        for _ in range(3):
            # 把颜色刷回「别的颜料」，模拟又装了一枚
            tiles = list(state.tiles)
            for row, col in placed:
                index = row * gm.BOARD_SIZE + col
                tiles[index] = gm.replace(tiles[index], paint=other)
            state = gm.replace(state, tiles=tuple(tiles))
            state, count = gm._apply_deferred_scope_affix(state, random.Random(1))
            self.assertIsNotNone(state.active_affix, '每轮安装后 buff 都要保留')
            self.assertEqual(count, len(placed), '每轮都要重新结算范围内颜料')


class CopySelfScopeTest(unittest.TestCase):
    """第 5 条：范围决定一切；变色格转空白后也吃 buff。"""

    def test_only_paints_inside_scope_change(self) -> None:
        state = _anchored_state()
        state = gm.apply_affix_to_state(state, _affix(), seed=1).state
        self_paint = state.current_rule.paint
        other = gm.Paint(category=gm.PaintCategory.GAMEPLAY, level=1, subtype='存活')
        state = gm.replace(
            state,
            current_rule=gm.replace(state.current_rule, direction=gm.PaintDirection.GLOBAL),
        )
        # 锚点格是作用域的唯一例外，放一枚别的颜料在那儿
        anchor = state.current_anchor
        tiles = list(state.tiles)
        if anchor is not None:
            index = anchor[0] * gm.BOARD_SIZE + anchor[1]
            tiles[index] = gm.replace(
                tiles[index], state=gm.TileState.PAINT, paint=other
            )
        state = gm.replace(state, tiles=tuple(tiles))

        settled, _count = gm._copy_self_to_scope(state)
        if anchor is not None:
            tile = settled.tile(*anchor)
            inside = anchor in set(gm._affix_scope_cells(state, state.active_affix))
            if not inside and tile.paint is not None:
                self.assertNotEqual(
                    tile.paint.category, self_paint.category,
                    '作用范围之外的颜料不能被替换',
                )

    def test_faded_to_blank_still_gets_buff(self) -> None:
        state = _anchored_state()
        state = gm.apply_affix_to_state(state, _affix(), seed=1).state
        other = gm.Paint(category=gm.PaintCategory.GAMEPLAY, level=1, subtype='存活')
        state = gm.replace(
            state,
            current_rule=gm.replace(state.current_rule, direction=gm.PaintDirection.GLOBAL),
        )
        scope = set(gm._affix_scope_cells(state, state.active_affix))
        target = next(
            (row, col) for row, col in scope
            if state.tile(row, col).state is not gm.TileState.PAINT
        )
        tiles = list(state.tiles)
        index = target[0] * gm.BOARD_SIZE + target[1]
        # 失色 → 空白 → 装入颜料，应当吃 buff
        tiles[index] = gm.replace(
            tiles[index], state=gm.TileState.FADED, paint=None
        )
        tiles[index] = gm.replace(
            tiles[index], state=gm.TileState.BLANK, paint=None
        )
        tiles[index] = gm.replace(
            tiles[index], state=gm.TileState.PAINT, paint=other
        )
        state = gm.replace(state, tiles=tuple(tiles))
        settled, _count = gm._apply_deferred_scope_affix(state, random.Random(1))
        self.assertEqual(
            settled.tile(*target).paint.category,
            state.current_rule.paint.category,
            '范围内由失色转成空白的格子，装上颜料后也要变自身',
        )


if __name__ == '__main__':
    unittest.main()
