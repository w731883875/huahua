"""Archive replay must preserve uncertainty, source identity and original data."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from huahua_assistant.learning import compact_trajectory_payload
from scripts import replay_recent_runs as replay


def snapshot(index=0, t=0, paints=1):
    return {
        'step_index': index, 't': t,
        'board': {'states': 'P'*paints+'B'*(36-paints), 'paint': paints, 'blank': 36-paints},
        'tiles': [{'c': 'monster', 's': None, 'l': 3, 't': 0} for _ in range(paints)]+[None]*(36-paints),
        'current': {'paint': {'c': 'monster', 's': None, 'l': 3}, 'direction': 'unknown', 'target': 'monster'},
        'quality': {'ok': True}, 'inferred_action': {'type': 'install_paint', 'confidence': .74},
    }


def archive(session='game-a', start=100, steps=None):
    return {'session': session, 'started_at': start, 'finished_at': start+30, 'finish_seen': True,
            'steps': steps if steps is not None else [snapshot(), snapshot(1, 10, 2)]}


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf8')


class RecentReplayTest(unittest.TestCase):
    def test_selects_latest_session_and_expands_compact_tiles(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'archives.jsonl'
            newer = archive(steps=[snapshot(), snapshot(1, 2, 3)])
            compacted = compact_trajectory_payload(copy.deepcopy(newer))
            write_rows(path, [{'ts': 1, 'payload': archive()}, {'ts': 2, 'payload': compacted},
                              {'ts': 4, 'payload': archive('single', 200, [snapshot()])},
                              {'ts': 3, 'payload': archive('game-b', 150)}])
            original = path.read_bytes()
            selected, excluded, bad = replay.select_archives(path, 5)
            self.assertEqual([p['session'] for _, p, _ in selected], ['game-b', 'game-a'])
            self.assertEqual(selected[1][1]['steps'], newer['steps'])
            self.assertEqual(selected[1][2], 2)
            self.assertEqual([p['session'] for p in excluded], ['single'])
            self.assertEqual(bad, [])
            self.assertEqual(path.read_bytes(), original)

    def test_same_step_from_other_session_cannot_corroborate_frame(self):
        a = archive()
        foreign = {'event': 'quick_capture_step', '_line': 3,
                   'payload': {**copy.deepcopy(a['steps'][0]), 'session': 'different-game'}}
        exact = {'event': 'quick_capture_step', '_line': 8,
                 'payload': {**copy.deepcopy(a['steps'][1]), 'session': 'game-a'}}
        run = replay.make_run(a, 7, {'game-a': [foreign, exact]}, [])
        self.assertEqual(run['stats']['corroborated'], 1)
        self.assertEqual(run['steps'][0]['evidence']['observation_lines'], [])
        self.assertEqual(run['steps'][1]['evidence'],
                         {'archive_line': 7, 'archive_step': 1, 'observation_lines': [8]})

    def test_source_conflicts_and_unknowns_are_preserved(self):
        step = snapshot()
        step['board']['states'] = 'B?'+'B'*34
        original = copy.deepcopy(step)
        run = replay.make_run(archive(steps=[step]), 1, {}, [])
        output = run['steps'][0]
        self.assertEqual(output['board'], original['board'])
        self.assertEqual(output['tiles'], original['tiles'])
        self.assertIn('1行1列：地形与颜料明细冲突', output['issues'])
        self.assertIn('1行2列：地形未知', output['issues'])
        self.assertIn('初始颜料家族未识别', output['issues'])
        self.assertEqual(output['current']['direction'], 'unknown')
        self.assertIn(0, output['line_capacity']['uncertain_cells'])
        self.assertEqual(run['stats']['conflict_frames'], 1)
        self.assertEqual(step, original)

    def test_gaps_and_time_reversal_are_not_hidden(self):
        run = replay.make_run(archive(steps=[snapshot(), snapshot(1, 45, 2), snapshot(2, 40, 3)]), 1, {}, [])
        self.assertEqual(run['steps'][1]['gap'], 45)
        self.assertTrue(any('可能包含多次操作' in text for text in run['steps'][1]['issues']))
        self.assertIn('存档时间倒退，保留原始记录顺序', run['steps'][2]['issues'])
        self.assertEqual([s['source_index'] for s in run['steps']], [0, 1, 2])

    def test_inference_events_are_distinct_from_clicks_and_bounded_by_time(self):
        events = [
            {'ts': 99, 'event': 'brush_stage1_card_clicked', 'payload': {'source': 'captured_click'}, '_line': 1},
            {'ts': 103, 'event': 'brush_stage1_selected',
             'payload': {'source': 'effect_inference', 'selected': 'stack'}, '_line': 2},
            {'ts': 104, 'event': 'brush_stage1_card_clicked',
             'payload': {'source': 'captured_click', 'selected': 'stack'}, '_line': 3},
            {'ts': 111, 'event': 'affix_card_clicked', 'payload': {'source': 'captured_click'}, '_line': 4},
        ]
        run = replay.make_run(archive(), 1, {}, events)
        associated = run['steps'][1]['events']
        self.assertEqual([e['line'] for e in associated], [2, 3])
        self.assertIn('非点击证据', associated[0]['evidence'])
        self.assertTrue(associated[1]['evidence'].startswith('点击记录'))
        self.assertEqual(run['stats']['click_records'], 1)
        self.assertEqual(run['steps'][1]['inferred_action']['confidence'], .74)

    def test_added_paint_and_identity_reading_are_not_equivalent(self):
        before = snapshot()
        after = copy.deepcopy(before)
        after['tiles'][0]['s'] = '黑帮'
        change = replay.diff_steps(before, after)[0]
        self.assertTrue(change['detail_only'])
        self.assertFalse(change['added'])
        self.assertEqual((change['row'], change['col']), (1, 1))
        after['tiles'][0]['t'] = 1
        self.assertFalse(replay.diff_steps(before, after)[0]['detail_only'])
        self.assertTrue(replay.diff_steps(before, snapshot(1, 10, 2))[0]['added'])

    def test_short_lived_flip_is_visible(self):
        run = replay.make_run(archive(steps=[snapshot(), snapshot(1, 2, 2), snapshot(2, 4)]), 1, {}, [])
        self.assertTrue(any('可能是识别抖动' in s for s in run['steps'][1]['issues']))

    def test_embedded_json_cannot_inject_script_but_round_trips(self):
        payload = {'text': '</script><script>alert(1)</script>&中文'}
        html = replay.render_html(payload)
        body = html.split('<script id="replay-data" type="application/json">', 1)[1].split('</script>', 1)[0]
        self.assertNotIn('<', body)
        self.assertEqual(json.loads(body), payload)

    def test_report_is_read_only_and_ignores_foreign_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder)
            a = archive()
            write_rows(runtime/'high_score_trajectories.jsonl', [{'payload': compact_trajectory_payload(copy.deepcopy(a))}])
            write_rows(runtime/'observations.jsonl', [
                {'event': 'quick_capture_step', 'payload': {**copy.deepcopy(s), 'session': a['session']}}
                for s in a['steps']
            ]+[{'event': 'quick_capture_ignored', 'payload': {'session': 'another-session'}}])
            write_rows(runtime/'auto_run_records.jsonl', [])
            original = {p.name: p.read_bytes() for p in runtime.iterdir()}
            report = replay.build_report(runtime)
            self.assertEqual(report['runs'][0]['stats']['corroborated'], 2)
            self.assertEqual(report['runs'][0]['stats']['ignored_observations'], 0)
            self.assertIn('存档状态回放', replay.render_html(report))
            self.assertEqual({p.name: p.read_bytes() for p in runtime.iterdir()}, original)


if __name__ == '__main__':
    unittest.main()
