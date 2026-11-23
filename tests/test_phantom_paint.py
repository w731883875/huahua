#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「状态=颜料、内容=None」幽灵格的回归测试（2026-11-23 实机事故）。

事故现场：``run-20261123-234819`` 第 56 步格 ``c5r6`` 读出了
``G1-狩猎之神 Lv1``，第 57 步短暂读成空白，第 58 步状态又回到 P，
但颜料内容从此再也读不出来。

旧代码在 ``state_from_observation`` 里直接 ``state=tile.state`` +
``paint=None``，于是构造出 ``Tile(state=PAINT, paint=None)``：

* ``_count_tiles(PAINT)`` 把它算进颜料数（看板 paint=21）；
* ``_compact_tile_paints`` 又给它输出 ``None``（落盘只有 20 枚）。

同一帧自相矛盾，而且之后连续 11 步都带着这个幽灵格。这类坏格一旦进状态，
还会被算成「棋盘上有颜料」，污染范围、加成和推荐。

修法（用户 2026-11-23 口径「日志/画面为真」）：
* 上一帧这格有颜料 → **沿用上一帧**（颜料持久，一帧读不出内容≠它没了）；
* 上一帧也没有 → **降级成 UNKNOWN**，交给下一帧重读，绝不伪造空壳颜料。
"""

from __future__ import annotations

import unittest

from huahua_assistant import game_model as gm

BOARD = gm.BOARD_SIZE
TARGET = (5, 4)  # 事故现场 c5r6 → 0-based r5c4


class _ObsTile:
    """最小化的视觉观测格（只带 state 和被读出来的 paint_* 属性）。"""

    def __init__(self, row: int, col: int, state: gm.TileState, **paint_fields: object) -> None:
        self.row = row
        self.col = col
        self.state = state
        for key, value in paint_fields.items():
            setattr(self, key, value)


class _ObsBoard:
    def __init__(self, tiles: list[_ObsTile]) -> None:
        self.tiles = tiles


def _board_with_target(state: gm.TileState, *, readable: bool) -> _ObsBoard:
    """造一块棋盘：目标格按 ``state``，其余全空白；``readable`` 控制内容能否读出。"""
    fields: dict[str, object] = {}
    if readable:
        fields = {
            'paint_category': gm.PaintCategory.GAMEPLAY,
            'paint_level': 1,
            'paint_category_confidence': 0.9,
            'paint_level_confidence': 0.9,
            'paint_subtype': '狩猎之神',
            'paint_is_treasure': False,
        }
    tiles = [
        _ObsTile(row, col, state if (row, col) == TARGET else gm.TileState.BLANK, **fields)
        for row in range(BOARD)
        for col in range(BOARD)
    ]
    return _ObsBoard(tiles)


class PhantomPaintTest(unittest.TestCase):
    """状态读到颜料、内容读不出来时的处置。"""

    def test_carries_previous_paint_when_content_unreadable(self) -> None:
        previous = gm.state_from_observation(_board_with_target(gm.TileState.PAINT, readable=True))
        self.assertIsNotNone(previous.tiles[TARGET[0] * BOARD + TARGET[1]].paint)

        current = gm.state_from_observation(
            _board_with_target(gm.TileState.PAINT, readable=False),
            previous=previous,
        )
        tile = current.tiles[TARGET[0] * BOARD + TARGET[1]]
        self.assertIs(tile.state, gm.TileState.PAINT, '上一帧有颜料时状态应保持 PAINT')
        self.assertIsNotNone(tile.paint, '应沿用上一帧的颜料，而不是留下空壳')
        self.assertEqual(tile.paint.subtype, '狩猎之神', '沿用的应是具体那枚颜料')
        self.assertEqual(tile.paint.level, 1)

    def test_never_leaves_paint_state_without_paint(self) -> None:
        """核心不变式：状态是 PAINT 的格子，paint 一定不是 None。"""
        current = gm.state_from_observation(_board_with_target(gm.TileState.PAINT, readable=False))
        for tile in current.tiles:
            if tile.state is gm.TileState.PAINT:
                self.assertIsNotNone(tile.paint, '出现状态=PAINT 但 paint=None 的幽灵格')

    def test_degrades_to_unknown_without_previous(self) -> None:
        current = gm.state_from_observation(_board_with_target(gm.TileState.PAINT, readable=False))
        tile = current.tiles[TARGET[0] * BOARD + TARGET[1]]
        self.assertIs(tile.state, gm.TileState.UNKNOWN, '没有历史可依时应降级为 UNKNOWN')
        self.assertIsNone(tile.paint)

    def test_readable_paint_is_untouched(self) -> None:
        """内容读得出来时不能走沿用逻辑，必须用这一帧的实测值。"""
        previous = gm.state_from_observation(_board_with_target(gm.TileState.PAINT, readable=True))
        readable = _board_with_target(gm.TileState.PAINT, readable=True)
        # 把这一帧改成 3 级，验证不会被上一帧的 1 级顶掉
        readable.tiles[TARGET[0] * BOARD + TARGET[1]].paint_level = 3
        current = gm.state_from_observation(readable, previous=previous)
        tile = current.tiles[TARGET[0] * BOARD + TARGET[1]]
        self.assertEqual(tile.paint.level, 3, '读得出来时必须用实测等级')

    def test_blank_cell_stays_blank(self) -> None:
        """空白格不能被沿用逻辑塞上颜料。"""
        previous = gm.state_from_observation(_board_with_target(gm.TileState.PAINT, readable=True))
        current = gm.state_from_observation(
            _board_with_target(gm.TileState.BLANK, readable=False),
            previous=previous,
        )
        tile = current.tiles[TARGET[0] * BOARD + TARGET[1]]
        self.assertIs(tile.state, gm.TileState.BLANK)
        self.assertIsNone(tile.paint)


class CountConsistencyTest(unittest.TestCase):
    """颜料计数与逐格数据必须一致——事故里就是这里对不上。"""

    def test_paint_count_matches_paint_objects(self) -> None:
        current = gm.state_from_observation(_board_with_target(gm.TileState.PAINT, readable=False))
        counted = sum(1 for tile in current.tiles if tile.state is gm.TileState.PAINT)
        with_paint = sum(
            1 for tile in current.tiles
            if tile.state is gm.TileState.PAINT and tile.paint is not None
        )
        self.assertEqual(counted, with_paint, '颜料计数与有内容的颜料数必须相等')


if __name__ == '__main__':
    unittest.main()
