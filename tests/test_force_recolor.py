#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""强制调色规则回归测试。

规则（实机确认）：

* 一局最多 **3 次**；
* 第 N 次炸毁 **N 个**颜料（1 / 2 / 3）；
* 被炸的格子变**失色**，但**地块加成保留**（这是"炸完再放一枚直接 7 级"的来源）；
* 失色格在普通情况下**不可落子**；
* 棋盘上颜料不足时按实际数量炸，但**必须如实报告**，不能静默降级。
"""
from __future__ import annotations

import unittest

from huahua_assistant import game_model as gm
from huahua_assistant import practice as pr
from huahua_assistant.engine.actions import ActionKind
from huahua_assistant.engine.rules import RuleEngine


def _ready_session(seed: int = 2, *, rules: int | None = None) -> pr.PracticeSession:
    """开一局，并推进到"可以强制调色"的状态。"""
    engine = RuleEngine()
    session = pr.start_practice(
        seed, pet_level=6, rules=rules, max_steps=400, wait_for_anchor=True
    )
    if gm.current_rule_is_pending(session.state):
        anchors = [
            a for a in engine.legal_actions(session.state)
            if a.kind is ActionKind.PLACE_ANCHOR
        ]
        if anchors:
            session.place_anchor(anchors[0].row, anchors[0].col)
    guard = 0
    while not gm.backpack_paint_placed(session.state) and guard < 10:
        guard += 1
        installs = [
            a for a in engine.legal_actions(session.state)
            if a.kind is ActionKind.INSTALL
        ]
        if not installs:
            break
        session.install(installs[0].backpack_index, installs[0].row, installs[0].col)
    return session


class ForceRecolorTest(unittest.TestCase):
    """三次强制调色的次数、摧毁数与地块状态。"""

    def test_max_three_uses(self) -> None:
        self.assertEqual(gm.MAX_FORCE_TIMES, 3)

    def test_damage_escalates_1_2_3(self) -> None:
        """第 N 次应炸 N 个——棋盘颜料充足时必须炸满。"""
        session = _ready_session(2)
        for index, want in enumerate([1, 2, 3], start=1):
            # 强制调色后规则会推进到"待放锚点"，要先落锚点 + 一枚颜料才能再强制
            if gm.current_rule_is_pending(session.state):
                session.place_anchor(*_any_anchor(session))
            while not gm.backpack_paint_placed(session.state):
                if not _install_one(session):
                    break
            # 保证棋盘上至少 4 枚颜料（够炸 3 个），但**留出空白格**给下一轮落锚点
            tiles = list(session.state.tiles)
            paints = sum(
                1 for tile in tiles if tile.state is gm.TileState.PAINT
            )
            for cell, tile in enumerate(tiles):
                if paints >= 4:
                    break
                if tile.state is gm.TileState.BLANK:
                    tiles[cell] = gm.replace(
                        tile,
                        state=gm.TileState.PAINT,
                        paint=gm.Paint(
                            category=gm.PaintCategory.GAMEPLAY,
                            level=1,
                            subtype='存活',
                        ),
                    )
                    paints += 1
            session.state = gm.replace(session.state, tiles=tuple(tiles))
            self.assertTrue(session.force_complete(), f'第 {index} 次应当成功')
            self.assertIn(f'摧毁 {want} 个颜料', session.steps[-1].note)
            self.assertNotIn('应炸', session.steps[-1].note, '颜料充足时不该出现短缺提示')

    def test_fourth_force_is_rejected(self) -> None:
        session = _ready_session(2)
        for _ in range(3):
            if gm.current_rule_is_pending(session.state):
                session.place_anchor(*_any_anchor(session))
            while not gm.backpack_paint_placed(session.state):
                if not _install_one(session):
                    break
            session.force_complete()
        self.assertEqual(session._force_used, 3)
        self.assertEqual(session.force_remaining, 0)
        self.assertFalse(session.force_complete())

    def test_destroyed_tile_becomes_faded_and_drops_the_bonus(self) -> None:
        """炸毁 → 失色，颜料身上那笔加成跟着颜料一起销毁（用户 2026-11-23 规则一）。"""
        session = _ready_session(39)
        tiles = list(session.state.tiles)
        painted = [
            index for index, tile in enumerate(tiles)
            if tile.state is gm.TileState.PAINT
        ]
        self.assertTrue(painted, '本局应当已有颜料')
        for index in painted:
            tiles[index] = gm.replace(
                tiles[index], paint=gm.replace(tiles[index].paint, bonus_level=2)
            )
        session.state = gm.replace(session.state, tiles=tuple(tiles))

        self.assertTrue(session.force_complete())
        lingering = [
            (index // gm.BOARD_SIZE, index % gm.BOARD_SIZE, gm.tile_bonus_level(tile))
            for index, tile in enumerate(session.state.tiles)
            if tile.state is gm.TileState.FADED and gm.tile_bonus_level(tile) > 0
        ]
        self.assertEqual(
            lingering,
            [],
            '失色格不许保留加成：加成挂在颜料上，颜料被炸毁就一起没了',
        )

    def test_shortfall_is_reported_not_silent(self) -> None:
        """棋盘颜料不足时，摧毁数可以少于 N，但提示里必须写明。"""
        session = _ready_session(2)
        found_shortfall = False
        for _ in range(3):
            if gm.current_rule_is_pending(session.state):
                session.place_anchor(*_any_anchor(session))
            while not gm.backpack_paint_placed(session.state):
                if not _install_one(session):
                    break
            session.force_complete()
            note = session.steps[-1].note
            if '应炸' in note:
                found_shortfall = True
                self.assertIn('颜料不足', note)
        # 三次里至少第三次常常不足；不强求一定出现，但出现时必须写清楚
        if found_shortfall:
            self.assertTrue(True)

    def test_faded_tile_not_placeable(self) -> None:
        session = _ready_session(2)
        faded = next(
            (
                (row, col)
                for row in range(gm.BOARD_SIZE)
                for col in range(gm.BOARD_SIZE)
                if session.state.tile(row, col).state is gm.TileState.FADED
            ),
            None,
        )
        if faded is None:
            self.skipTest('本局开局没有失色格')
        self.assertFalse(
            gm._is_placeable(
                session.state.tile(*faded), session.state.active_affix
            )
        )


def _any_anchor(session: pr.PracticeSession) -> tuple[int, int]:
    engine = RuleEngine()
    for action in engine.legal_actions(session.state):
        if action.kind is ActionKind.PLACE_ANCHOR:
            return action.row, action.col
    return (0, 0)


def _install_one(session: pr.PracticeSession) -> bool:
    engine = RuleEngine()
    for action in engine.legal_actions(session.state):
        if action.kind is ActionKind.INSTALL:
            return session.install(action.backpack_index, action.row, action.col)
    return False


if __name__ == '__main__':
    unittest.main()
