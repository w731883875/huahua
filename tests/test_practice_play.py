#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手动练习驱动（任务 B）与官方几率模拟（任务 C）的回归测试。

覆盖点：

* 手动动作能把局面推进，并且撤销后**局面价值回到原值**；
* 存档 → 重建 → 价值一致（还原链路不能有隐性状态漂移）；
* 同一局重复入库会被签名去重；
* 模拟报告的等级分布与官方 ``4005`` 权重同向（期望等级接近 3.54）；
* 报告里必须显式列出非官方假设，避免把代理口径当官方数据。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from huahua_assistant import game_model as gm
from huahua_assistant import practice as pr
from huahua_assistant import practice_batch as ps_scan
from huahua_assistant import practice_play as pp
from huahua_assistant import practice_simulate as ps
from huahua_assistant import s8_official_data as official
from huahua_assistant.practice_library import PracticeLibrary


class ManualPracticeTest(unittest.TestCase):
    """手动落子 / 撤销 / 存档还原。"""

    def _session(self, seed: int = 7) -> pr.PracticeSession:
        return pr.start_practice(seed, pet_level=3, rules=2, wait_for_anchor=True)

    def test_place_anchor_then_undo_restores_value(self):
        driver = pp.ManualPractice(session=self._session())
        before = driver.session.value
        ok, _message = driver.anchor(2, 2)
        self.assertTrue(ok)
        self.assertEqual(len(driver.session.steps), 1)
        ok, _message = driver.undo()
        self.assertTrue(ok)
        self.assertEqual(len(driver.session.steps), 0)
        self.assertAlmostEqual(driver.session.value, before, places=9)

    def test_undo_at_opening_is_rejected(self):
        driver = pp.ManualPractice(session=self._session())
        ok, message = driver.undo()
        self.assertFalse(ok)
        self.assertIn('开局', message)

    def test_snapshot_roundtrip_preserves_value(self):
        driver = pp.ManualPractice(session=self._session(11))
        driver.anchor(2, 2)
        snapshot = driver.session.snapshot()
        rebuilt = pp.rebuild(snapshot)
        self.assertAlmostEqual(rebuilt.value, driver.session.value, places=9)
        self.assertEqual(len(rebuilt.steps), len(driver.session.steps))

    def test_save_and_load_snapshot_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = pp.SESSION_DIR
            pp.SESSION_DIR = Path(tmp)
            try:
                driver = pp.ManualPractice(session=self._session(3))
                driver.anchor(1, 1)
                path = pp.save_snapshot(driver.session.snapshot(), 'unittest')
                self.assertTrue(path.exists())
                loaded = pp.rebuild(pp.load_snapshot('unittest'))
                self.assertAlmostEqual(loaded.value, driver.session.value, places=9)
            finally:
                pp.SESSION_DIR = original

    def test_install_uses_one_based_index(self):
        """界面显示序号从 1 开始；传 0 必须被拒绝，传 1 必须能落子。"""
        from huahua_assistant.engine.actions import ActionKind
        from huahua_assistant.engine.rules import RuleEngine

        driver = pp.ManualPractice(session=self._session(5))
        engine = RuleEngine()
        anchor_actions = [
            action for action in engine.legal_actions(driver.session.state)
            if action.kind is ActionKind.PLACE_ANCHOR
        ]
        self.assertTrue(anchor_actions)
        anchor = anchor_actions[0]
        self.assertTrue(driver.anchor(anchor.row, anchor.col))

        install_actions = [
            action for action in engine.legal_actions(driver.session.state)
            if action.kind is ActionKind.INSTALL
        ]
        self.assertTrue(install_actions, '落完锚点后应当有合法落子点')
        target = install_actions[0]
        # 序号 0 不合法（人类序号从 1 开始）
        ok, _message = driver.put(0, target.row, target.col)
        self.assertFalse(ok)
        ok, _message = driver.put(target.backpack_index + 1, target.row, target.col)
        self.assertTrue(ok)

    def test_record_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = PracticeLibrary(Path(tmp) / 'lib.jsonl')
            session = pr.start_practice(9, pet_level=2, rules=1, wait_for_anchor=True)
            pr.auto_play(session, prefer='greedy')
            driver = pp.ManualPractice(session=session, library=library)
            self.assertIsNotNone(driver.record(note='第一次'))
            self.assertIsNone(driver.record(note='第二次'))
            self.assertEqual(library.stats()['runs'], 1)


class RenderTest(unittest.TestCase):
    """渲染不能因为字段改名就炸。"""

    def test_render_board_contains_key_sections(self):
        session = pr.start_practice(13, pet_level=5, rules=1, wait_for_anchor=True)
        text = pp.render_board(session)
        for keyword in ('词缀：', '背包', '强制调色剩余', '宠物 Lv5'):
            self.assertIn(keyword, text)

    def test_rule_text_has_no_placeholder(self):
        session = pr.start_practice(13, pet_level=1, rules=1, wait_for_anchor=True)
        rule = session.state.current_rule
        if rule is None:
            self.skipTest('该 seed 开局没有词缀')
        text = pp.rule_text(rule)
        self.assertIn(f'Lv{rule.paint.level}', text)


class SimulationTest(unittest.TestCase):
    """官方几率模拟与对账。"""

    def test_level_distribution_tracks_official_weights(self):
        """模拟出来的背包等级分布应当贴着官方 4005 权重。"""
        result = ps.simulate(runs=25, base_seed=1, pet_level=1, rules=1)
        observed = result.observed_start_distribution()
        self.assertTrue(observed)
        expected = official.paint_level_distribution()
        mean_expected = sum(level * p for level, p in expected.items())
        mean_observed = sum(level * p for level, p in observed.items())
        # 25 局 × 30 枚 ≈ 750 个样本，均值不应偏离官方期望太多
        self.assertLess(abs(mean_observed - mean_expected), 0.35)

    def test_report_lists_assumptions(self):
        result = ps.simulate(runs=3, base_seed=1, pet_level=1, rules=1)
        report = ps.format_report(result)
        self.assertIn('非官方假设', report)
        self.assertIn('官方', report)
        self.assertIn('对账', report)

    def test_assumptions_cover_known_gaps(self):
        items = {item['item'] for item in ps.assumptions()}
        self.assertTrue(any('宝箱' in name for name in items))
        self.assertTrue(any('等级' in name for name in items))

    def test_simulation_records_into_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ps.simulate(
                runs=3,
                base_seed=2,
                pet_level=1,
                rules=1,
                record=True,
                library_path=Path(tmp) / 'lib.jsonl',
            )
            self.assertIsNotNone(result.recorded)
            self.assertEqual(result.recorded['runs'], 3)


class TacticCoverageTest(unittest.TestCase):
    """用户点名的那个技巧必须能产出说明。"""

    def test_isolated_treasure_tactic_is_reachable(self):
        """「地块连接少但有宝箱」在落完锚点后应能命中。

        开局（还没落锚点）**不能**触发：作用范围是相对锚点定义的，
        那时空位数恒为 0，会让这条对任何带宝箱的开局都成立。
        """
        from huahua_assistant.engine.actions import ActionKind
        from huahua_assistant.engine.rules import RuleEngine

        engine = RuleEngine()
        hit = False
        tested = 0
        for seed in range(1, 61):
            session = pr.start_practice(seed, pet_level=1, rules=1, wait_for_anchor=True)
            opening = pr.find_tactics(session.state, steps=session.steps)
            self.assertFalse(
                any(item.key == pr.TACTIC_FORCE_ON_ISOLATED_TREASURE for item in opening),
                f'seed {seed}：没落锚点就误报了孤立宝箱技巧',
            )
            anchor_actions = [
                action for action in engine.legal_actions(session.state)
                if action.kind is ActionKind.PLACE_ANCHOR
            ]
            if not anchor_actions:
                continue
            anchor = anchor_actions[0]
            session.place_anchor(anchor.row, anchor.col)
            tested += 1
            findings = pr.find_tactics(session.state, steps=session.steps)
            if any(item.key == pr.TACTIC_FORCE_ON_ISOLATED_TREASURE for item in findings):
                hit = True
        self.assertGreater(tested, 10, '可用种子太少，测试没有说服力')
        self.assertTrue(hit, f'{tested} 个种子里都没命中孤立宝箱技巧')


class Level7ScanTest(unittest.TestCase):
    """7 级宝箱扫描：必须真能挑出含 7 级宝箱的复盘，且标注为模拟。"""

    def test_scan_finds_level7_treasures(self):
        report = ps_scan.scan(
            runs=120, base_seed=1, pet_level=1, rules=2,
            workers=1, save=False, record=False,
        )
        self.assertEqual(report.runs, 120)
        self.assertTrue(report.hits, '120 局应当至少命中一局含 7 级宝箱')
        for hit in report.hits:
            self.assertTrue(hit['level7_tiles'], '命中必须带 7 级宝箱坐标')
            for _row, _col, level in hit['level7_tiles']:
                self.assertGreaterEqual(level, ps_scan.TARGET_LEVEL)
            self.assertIn('snapshot', hit, '复盘必须带快照，否则无法还原')

    def test_saved_replay_restores_with_level7_treasure(self):
        import tempfile as _tempfile
        from huahua_assistant.practice_library import PracticeLibrary as _Lib

        with _tempfile.TemporaryDirectory() as tmp:
            report = ps_scan.scan(
                runs=120, base_seed=1, pet_level=1, rules=2, workers=1,
                save=True, record=False, out_dir=Path(tmp) / 'l7',
            )
            self.assertTrue(report.saved_paths)
            index_path = ps_scan.rebuild_index(Path(tmp) / 'l7')
            self.assertTrue(index_path.exists())

            payload = json.loads(Path(report.saved_paths[0]).read_text(encoding='utf-8'))
            self.assertEqual(payload['origin'], 'simulated')
            self.assertEqual(payload['source'], 'simulation')

            session, check = pr.replay_session(payload['snapshot'])
            self.assertTrue(check.ok, check.mismatch)
            found = any(
                tile.state is gm.TileState.PAINT
                and tile.paint is not None
                and tile.paint.is_treasure
                and gm.tile_final_level(tile) >= ps_scan.TARGET_LEVEL
                for tile in session.state.tiles
            )
            self.assertTrue(found, '还原后 7 级宝箱必须仍在棋盘上')

    def test_scan_hits_are_recorded_as_simulation(self):
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as tmp:
            report = ps_scan.scan(
                runs=120, base_seed=1, pet_level=1, rules=2, workers=1,
                save=False, record=True,
                library_path=Path(tmp) / 'lib.jsonl',
            )
            self.assertGreater(report.library_written, 0)
            library = PracticeLibrary(Path(tmp) / 'lib.jsonl')
            split = library.split_by_origin()
            self.assertIn('simulated', split)
            self.assertEqual(split['simulated']['sources'], ['simulation'])
            self.assertNotIn('real', split, '模拟复盘不得混进实机分类')


class OriginSeparationTest(unittest.TestCase):
    """模拟复盘与真实复盘必须分得开。"""

    def test_origin_of(self):
        from huahua_assistant.practice_library import origin_of

        self.assertEqual(origin_of('real_capture'), 'real')
        self.assertEqual(origin_of('simulation'), 'simulated')
        self.assertEqual(origin_of('manual'), 'simulated')
        self.assertEqual(origin_of(None), 'simulated')

    def test_same_run_can_be_both_origins(self):
        """同一局在模拟与实机下必须能各存一条——带上 origin 的签名。"""
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as tmp:
            library = PracticeLibrary(Path(tmp) / 'lib.jsonl')
            session = pr.start_practice(3, pet_level=2, rules=1, wait_for_anchor=True)
            pr.auto_play(session, prefer='greedy')
            explanation = pr.explain_high_score(
                session.state, steps=session.steps, session=session,
                weights=session.weights,
            )
            snapshot = session.snapshot()
            self.assertIsNotNone(library.record(explanation, snapshot, source='simulation'))
            self.assertIsNotNone(library.record(explanation, snapshot, source='real_capture'))
            self.assertIsNone(library.record(explanation, snapshot, source='simulation'))
            split = library.split_by_origin()
            self.assertEqual(split['simulated']['runs'], 1)
            self.assertEqual(split['real']['runs'], 1)


if __name__ == '__main__':
    unittest.main()
