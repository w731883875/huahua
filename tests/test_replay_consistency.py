#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记录模式的「复原一致性」标记回归测试。

记录与复盘的意义：搞清楚游戏里实际怎么结算，以及**本地规则算出来是否一样**。
所以回放对不上必须被标记出来，不能当成一次正常记录吞掉。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from huahua_assistant import practice as pr
from huahua_assistant import practice_library as pl


def _one_session(seed: int = 3) -> pr.PracticeSession:
    session = pr.start_practice(seed, pet_level=5, rules=2, wait_for_anchor=True)
    pr.auto_play(session, prefer='greedy')
    return session


def _entry_for(library: pl.PracticeLibrary, session: pr.PracticeSession, source: str = 'real_capture'):
    explanation = pr.explain_high_score(
        session.state, steps=session.steps, session=session, weights=session.weights
    )
    return library.record(explanation, session.snapshot(), source=source, note='测试')


class _TempLibraryCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / 'lib.jsonl'
        self.library = pl.PracticeLibrary(self.path)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _rewrite(self, payload: dict) -> None:
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n',
            encoding='utf-8',
        )

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding='utf-8').strip())


class RecordCarriesReplayFlagTest(_TempLibraryCase):
    def test_entry_gets_replay_field(self) -> None:
        entry = _entry_for(self.library, _one_session())
        self.assertIsNotNone(entry)
        self.assertIn('replay', entry)
        self.assertTrue(entry['replay']['ok'])
        self.assertEqual(entry['replay']['delta'], 0.0)
        self.assertEqual(entry['replay']['mismatch'], '')

    def test_verify_can_be_skipped(self) -> None:
        session = _one_session()
        explanation = pr.explain_high_score(
            session.state, steps=session.steps, session=session, weights=session.weights
        )
        entry = self.library.record(
            explanation, session.snapshot(), source='real_capture', verify=False
        )
        self.assertIsNone(entry['replay']['ok'], '跳过校验时 ok 应为 None（未知）')

    def test_consistency_counts_unchecked(self) -> None:
        session = _one_session()
        explanation = pr.explain_high_score(
            session.state, steps=session.steps, session=session, weights=session.weights
        )
        self.library.record(
            explanation, session.snapshot(), source='real_capture', verify=False
        )
        summary = self.library.consistency()
        self.assertEqual(summary['checked'], 0)
        self.assertEqual(summary['unchecked'], 1)


class MismatchIsFlaggedTest(_TempLibraryCase):
    def test_value_drift_is_flagged(self) -> None:
        """记录分与回放分不一致 → 必须标记，而不是静默通过。"""
        _entry_for(self.library, _one_session())
        raw = self._read()
        raw['board_value'] = round(float(raw['board_value']) + 0.25, 6)
        raw['replay'] = pl.replay_flag(raw['snapshot'], raw['board_value'])
        self._rewrite(raw)

        library = pl.PracticeLibrary(self.path)
        self.assertFalse(raw['replay']['ok'])
        self.assertIn('不一致', raw['replay']['mismatch'])
        self.assertEqual(len(library.mismatched()), 1)

    def test_report_shows_warning(self) -> None:
        _entry_for(self.library, _one_session())
        raw = self._read()
        raw['board_value'] = round(float(raw['board_value']) + 0.25, 6)
        raw['replay'] = pl.replay_flag(raw['snapshot'], raw['board_value'])
        self._rewrite(raw)

        library = pl.PracticeLibrary(self.path)
        report = pl.format_library_report(library.stats())
        self.assertIn('⚠', report)
        self.assertIn('对不上 1 局', report)
        self.assertIn('与本地规则不一致', report)

    def test_entry_view_marks_mismatch(self) -> None:
        _entry_for(self.library, _one_session())
        raw = self._read()
        raw['board_value'] = round(float(raw['board_value']) + 0.25, 6)
        raw['replay'] = pl.replay_flag(raw['snapshot'], raw['board_value'])
        view = pl.LibraryEntryView.from_entry(raw)
        self.assertIs(view.replay_ok, False)
        self.assertIn('⚠复原不一致', view.format_text())

    def test_recheck_repairs_after_rule_fix(self) -> None:
        """规则改好后重跑一次校验，应该从"不一致"变回"一致"。"""
        _entry_for(self.library, _one_session())
        raw = self._read()
        good_value = float(raw['board_value'])
        raw['board_value'] = round(good_value + 0.25, 6)
        raw['replay'] = pl.replay_flag(raw['snapshot'], raw['board_value'])
        self._rewrite(raw)

        library = pl.PracticeLibrary(self.path)
        self.assertEqual(len(library.mismatched()), 1)

        # 把记录分改回真实回放值，模拟"本地规则已修正"
        raw = self._read()
        raw['board_value'] = good_value
        self._rewrite(raw)
        result = pl.PracticeLibrary(self.path).recheck_consistency()
        self.assertEqual(result['fixed'], 1)
        self.assertEqual(len(pl.PracticeLibrary(self.path).mismatched()), 0)


class ReplayDeterminismTest(unittest.TestCase):
    """回放必须**逐字节可复现**。

    这条守的是一个具体事故：加入词缀层时，生成词缀选项从主随机流
    （``_rng``）里取了数，导致之后 ``force_complete`` 的炸格位置整体错位，
    1782 局里有 389 局重放出不同终局——而且全部集中在含强制调色的对局。
    修法是给词缀单独开一条随机流（``_affix_rng``）。

    这里的断言就是"探针"：只要哪天又有代码往主随机流里插了额外消耗，
    这个测试会先炸，而不是等到重跑几千局才发现。
    """

    def test_replay_reproduces_same_value(self) -> None:
        for seed in range(1, 26):
            session = pr.start_practice(
                seed, pet_level=6, rules=1, max_steps=40, wait_for_anchor=True
            )
            snapshot = session.snapshot()
            _replayed, check = pr.replay_session(snapshot)
            self.assertTrue(
                check.ok, f'seed={seed} 回放对不上：{check.mismatch}'
            )
            self.assertEqual(
                round(check.actual_value, 6),
                round(float(snapshot['final_board_value']), 6),
                f'seed={seed} 回放终局分与记录不符',
            )

    def test_replay_with_force_complete_is_stable(self) -> None:
        """强制调色最吃随机流，必须重点覆盖。"""
        tested = 0
        for seed in range(1, 40):
            session = pr.start_practice(
                seed, pet_level=6, rules=None, max_steps=60, wait_for_anchor=True
            )
            pr.auto_play(session, prefer='greedy')
            snapshot = session.snapshot()
            if not any(step.get('kind') == 'force_complete' for step in snapshot['steps']):
                continue
            tested += 1
            _replayed, check = pr.replay_session(snapshot)
            self.assertTrue(
                check.ok, f'seed={seed}（含强制调色）回放对不上：{check.mismatch}'
            )
        self.assertGreater(tested, 0, '抽样里应当至少有含强制调色的对局')

    def test_affix_offer_does_not_disturb_main_rng(self) -> None:
        """生成词缀选项不能改变主随机流的后续输出。"""
        session = pr.start_practice(7, pet_level=6, rules=3, wait_for_anchor=True)
        before = [session._rng.random() for _ in range(5)]
        # 重置后先走一次词缀选项生成，再取同样的 5 个数
        refreshed = pr.start_practice(7, pet_level=6, rules=3, wait_for_anchor=True)
        refreshed._offer_affix()
        after = [refreshed._rng.random() for _ in range(5)]
        self.assertEqual(before, after, '词缀选项生成污染了主随机流')

    def test_affix_rng_is_independent_seed_stream(self) -> None:
        session = pr.start_practice(7, pet_level=6, rules=3, wait_for_anchor=True)
        self.assertIsNot(session._affix_rng, session._rng)
        # 两条流的首个输出不应相同（同一 seed 派生会撞上）
        self.assertNotEqual(session._affix_rng.random(), session._rng.random())


if __name__ == '__main__':
    unittest.main()
