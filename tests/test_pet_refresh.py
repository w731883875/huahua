#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""宠物刷新规则回归测试。

规则（实机确认，2026-11-12）：

* 刷新次数按宠物等级：**Lv1–2 为 0、Lv3–4 为 1、Lv5–6 为 2**；
* **只能刷新第二阶段的词缀**；第一阶段（画笔类型：转锋/堆砌/扩散）**不可刷新**；
* 但第一阶段仍然要有**选择推荐**（不能刷新 ≠ 不用推荐）。
"""
from __future__ import annotations

import unittest

from huahua_assistant import game_model as gm
from huahua_assistant import practice as pr
from huahua_assistant.engine.actions import ActionKind
from huahua_assistant.engine.rules import RuleEngine

EXPECTED_REFRESH = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


class PetRefreshTableTest(unittest.TestCase):
    """各等级刷新次数与资源换算。"""

    def test_table_matches_rule(self) -> None:
        for level, want in EXPECTED_REFRESH.items():
            self.assertEqual(
                gm.PET_REFRESH_BY_LEVEL[level], want,
                f'Lv{level} 刷新次数应为 {want}',
            )

    def test_resources_follow_level(self) -> None:
        for level, want in EXPECTED_REFRESH.items():
            resources = gm.resources_for_pet_level(level)
            self.assertEqual(resources.refresh_available, want > 0, f'Lv{level} 可用性')
            self.assertEqual(resources.refresh_remaining, want, f'Lv{level} 剩余次数')

    def test_practice_session_refresh_remaining(self) -> None:
        for level, want in EXPECTED_REFRESH.items():
            session = pr.start_practice(
                2, pet_level=level, rules=1, wait_for_anchor=True
            )
            self.assertEqual(session.refresh_remaining, want, f'Lv{level} 会话剩余')


class Stage1NotRefreshableTest(unittest.TestCase):
    """第一阶段不可刷新，但仍要推荐。"""

    def test_special_actions_never_offer_stage1_refresh(self) -> None:
        session = pr.start_practice(2, pet_level=5, rules=2, wait_for_anchor=True)
        scenario = gm.random_brush_choice_scenario(11)
        resources = gm.resources_for_pet_level(5)
        actions = gm.recommend_special_actions(session.state, resources, scenario)
        kinds = {action.action.value for action in actions}
        self.assertNotIn('refresh_stage1', kinds)

    def test_deprecated_stage1_refresh_returns_none(self) -> None:
        session = pr.start_practice(2, pet_level=5, rules=2, wait_for_anchor=True)
        scenario = gm.random_brush_choice_scenario(11)
        self.assertIsNone(gm._recommend_refresh_stage1(session.state, scenario))

    def test_stage1_selection_recommendation_still_works(self) -> None:
        """不能刷新 ≠ 不用推荐：第一阶段必须仍给出选哪个画笔类型。"""
        session = pr.start_practice(2, pet_level=5, rules=2, wait_for_anchor=True)
        scenario = gm.random_brush_choice_scenario(11)
        recs = gm.recommend_brush_types(session.state, scenario.stage1_options)
        self.assertTrue(recs, '第一阶段选择推荐不能为空')
        self.assertTrue(all(rec.reason for rec in recs), '每条推荐都要有理由')


class RefreshOnlyStage2Test(unittest.TestCase):
    """刷新只换第二阶段选项，画笔类型保持不动。"""

    def _session_at_affix_choice(self, pet_level: int) -> pr.PracticeSession:
        engine = RuleEngine()
        session = pr.start_practice(
            2, pet_level=pet_level, rules=3, max_steps=200, wait_for_anchor=True
        )
        guard = 0
        while not session.affix_pending and not session.done() and guard < 60:
            guard += 1
            if gm.current_rule_is_pending(session.state):
                anchors = [
                    a for a in engine.legal_actions(session.state)
                    if a.kind is ActionKind.PLACE_ANCHOR
                ]
                if anchors:
                    session.place_anchor(anchors[0].row, anchors[0].col)
                    continue
            if session.force_complete():
                continue
            installs = [
                a for a in engine.legal_actions(session.state)
                if a.kind is ActionKind.INSTALL
            ]
            if not installs:
                break
            session.install(
                installs[0].backpack_index, installs[0].row, installs[0].col
            )
        return session

    def test_refresh_keeps_stage1_and_changes_stage2(self) -> None:
        session = self._session_at_affix_choice(5)
        if not session.affix_pending:
            self.skipTest('本局未走到词缀选择')
        stage1_before = session._affix_scenario.stage1_options
        stage2_before = [affix.name for affix in session.affix_options]
        self.assertTrue(session.refresh_rule())
        self.assertEqual(
            session._affix_scenario.stage1_options, stage1_before,
            '刷新不得改动第一阶段画笔选项',
        )
        stage2_after = [affix.name for affix in session.affix_options]
        self.assertNotEqual(stage2_before, stage2_after, '第二阶段词缀应当被重抽')

    def test_refresh_respects_level_limit(self) -> None:
        session = self._session_at_affix_choice(3)
        if not session.affix_pending:
            self.skipTest('本局未走到词缀选择')
        self.assertEqual(session.refresh_remaining, 1)
        self.assertTrue(session.refresh_rule())
        self.assertEqual(session.refresh_remaining, 0)
        self.assertFalse(session.refresh_rule(), 'Lv3 只有 1 次')

    def test_level2_cannot_refresh(self) -> None:
        session = self._session_at_affix_choice(2)
        if not session.affix_pending:
            self.skipTest('本局未走到词缀选择')
        self.assertEqual(session.refresh_remaining, 0)
        self.assertFalse(session.refresh_rule())

    def test_refresh_without_pending_affix_is_rejected(self) -> None:
        session = pr.start_practice(2, pet_level=5, rules=2, wait_for_anchor=True)
        self.assertFalse(session.affix_pending)
        self.assertFalse(session.refresh_rule())


if __name__ == '__main__':
    unittest.main()
