from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from huahua_assistant import app, brush_panel, vision
from huahua_assistant.capture import CaptureResult
from huahua_assistant.frame_pump import FramePump
from huahua_assistant.codes import brush_effect_identity, compact_rule_with_code
from huahua_assistant.fast_vision import _fast_brush
from huahua_assistant.game_model import CurrentPaintRule, Paint, PaintEffect, PuzzleState, ResourceState
from huahua_assistant.learning import compact_trajectory_payload, expand_trajectory_payload
from huahua_assistant.models import (BrushOptionsObservation, PaintCategory, PaintDirection,
                                     PaintObject, Rect, RegionKind, RegionObservation)
from huahua_assistant.replay_effects import scope_comparison, selection_evidence
from huahua_assistant.run_recorder import AutomaticRunRecorder
from scripts.replay_recent_runs import make_run
from tests.test_brush_choices_recorded import _blank_tile, _brush, _snapshot
from tests.test_recent_replay import archive, snapshot


def panel():
    return _brush(chosen_stage1='stack', stage1_options=('stack',),
                  stage2_options=('自身提升1级', '范围内颜料变成随机类型颜料'),
                  card_rects=(Rect(110, 60, 270, 60), Rect(110, 145, 270, 70)))


class RecordingEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.recorder = AutomaticRunRecorder(path=None, stable_frames=3)
        self.tiles = [_blank_tile(i) for i in range(36)]
        self.recorder.process(_snapshot(self.tiles), observed_at=100)
        self.addCleanup(self.recorder.close)

    def test_transient_panel_survives_board_stability_and_coalescing(self):
        # Only one frame of cards, then gone: stable_frames must not erase it.
        self.recorder.process(_snapshot(self.tiles, brush=panel()), observed_at=101,
                              clicks=((100.9, 160, 85),))
        result = self.recorder.process(_snapshot(self.tiles), observed_at=102)
        choices = [e for e in result.evidence['events'] if e['event'] == 'affix_choice_resolved']
        self.assertEqual(choices[0]['payload']['selected'], '自身提升1级')
        self.assertEqual(choices[0]['payload']['source'], 'captured_click')
        self.assertIsNone(choices[0]['payload']['brush_effect_id'])
        self.assertEqual(choices[0]['payload']['brush_effect_ids'], [2001, 4006])
        # The UI only receives the next frame; the full evidence remains available.
        latest = self.recorder.process(_snapshot(self.tiles), observed_at=103)
        self.assertEqual(latest.evidence, result.evidence)
        preserved = copy.deepcopy(latest.evidence)
        self.recorder.process(_snapshot(self.tiles, brush=panel()), observed_at=104)
        self.assertEqual(latest.evidence, preserved)
        fresh = self.recorder.process(_snapshot(self.tiles), observed_at=105, run_boundary='new_run')
        self.assertNotEqual(fresh.evidence['session'], preserved['session'])
        self.assertFalse(fresh.evidence['events'])

    def test_card_rects_exclude_gap_and_confirm_after_resize(self):
        for scale in (.65, 1, 1.4):
            with self.subTest(scale=scale):
                p = panel()
                resize = lambda r: Rect(*(round(v * scale) for v in (r.x, r.y, r.w, r.h)))
                p = replace(p, panel_rect=resize(p.panel_rect), card_rects=tuple(map(resize, p.card_rects)))
                recorder = AutomaticRunRecorder(path=None, stable_frames=1)
                self.addCleanup(recorder.close)
                recorder.process(_snapshot(self.tiles, brush=p), observed_at=100,
                                 clicks=((99.9, 160*scale, 85*scale),))
                recorder.process(_snapshot(self.tiles, brush=p), observed_at=101,
                                 clicks=((100.8, 160*scale, 130*scale), (100.9, 160*scale, 230*scale)))
                result = recorder.process(_snapshot(self.tiles), observed_at=102)
                choices = [e for e in result.evidence['events'] if e['event'] == 'affix_choice_resolved']
                self.assertEqual(choices[0]['payload']['selected'], '自身提升1级')

    def test_buffered_confirmation_cannot_override_first_card(self):
        unknown = _brush(reading_pending=True)
        self.recorder.process(_snapshot(self.tiles, brush=unknown), observed_at=101,
                              clicks=((100.8, 160, 85), (100.9, 160, 230)))
        self.recorder.process(_snapshot(self.tiles, brush=panel()), observed_at=102)
        result = self.recorder.process(_snapshot(self.tiles), observed_at=103)
        chosen = [e for e in result.evidence['events'] if e['event'] == 'affix_choice_resolved'][0]
        self.assertEqual(chosen['payload']['selected'], '自身提升1级')
        self.assertEqual(chosen['payload']['source'], 'buffered_click')
        self.assertEqual(chosen['payload']['confidence'], .90)
        self.assertIsNone(chosen['payload']['brush_effect_id'])
        self.assertEqual(chosen['payload']['brush_effect_ids'], [2001, 4006])

    def test_official_identity_survives_disk_archive_and_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'events.jsonl'
            recorder = AutomaticRunRecorder(path=path, stable_frames=1)
            unique_panel = replace(panel(), chosen_stage1='transform', stage1_options=('transform',),
                                   stage2_options=('调色目标的进度要求减少2', '作用对象改为另一类型'))
            try:
                recorder.process(_snapshot(self.tiles, brush=unique_panel), observed_at=100,
                                 clicks=((99.9, 160, 85),))
                result = recorder.process(_snapshot(self.tiles), observed_at=101)
            finally:
                recorder.close()
            rows = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        choice = next(e for e in rows if e['event'] == 'affix_choice_resolved')['payload']
        self.assertEqual(choice['brush_effect_id'], 1011)
        self.assertEqual(choice['brush_effect_table'], 'S8_Gameplay_SelectableEffect')
        self.assertEqual(choice['brush_effect_id_source'], 'catalog_mapping')
        options = next(e for e in rows if e['event'] == 'affix_options_observed')['payload']
        self.assertEqual(options['option_effects'][0]['brush_effect_id'], 1011)
        step = snapshot()
        step['recording_evidence'] = result.evidence
        payload = archive(steps=[step])
        expanded = expand_trajectory_payload(compact_trajectory_payload(copy.deepcopy(payload)))
        selected = make_run(expanded, 1, {}, [])['steps'][0]['selection']
        self.assertEqual(selected['brush_effect_id'], 1011)
        self.assertEqual(selected['source'], 'captured_click')

    def test_id_ambiguity_and_unknowns_do_not_become_unique(self):
        ambiguous = brush_effect_identity('转变自身的作用方向为横向')
        self.assertEqual(ambiguous['brush_effect_ids'], [1002, 4001])
        self.assertIsNone(ambiguous['brush_effect_id'])
        self.assertEqual(ambiguous['brush_effect_id_status'], 'ambiguous')
        self.assertEqual(brush_effect_identity('调色目标的进度要求减少2')['brush_effect_id'], 1011)
        for text in ('没有这一种笔触', '使纵向的玩法颜料提升2级'):
            self.assertEqual(brush_effect_identity(text)['brush_effect_ids'], [])
            self.assertIsNone(brush_effect_identity(text)['brush_effect_id'])

    def test_inferred_selection_keeps_inference_source_with_id(self):
        self.recorder._emit('affix_choice_resolved', {'selected': '调色目标的进度要求减少2',
                             'brush': 'transform', 'source': 'effect_inference'}, 101)
        choice = self.recorder._events[-1]['payload']
        self.assertEqual(choice['brush_effect_id'], 1011)
        self.assertEqual(choice['source'], 'effect_inference')

    def test_resized_panel_does_not_remap_old_buffered_click(self):
        self.recorder.process(_snapshot(self.tiles, brush=_brush(reading_pending=True)), observed_at=101,
                              clicks=((100.9, 160, 85),))
        p = replace(panel(), panel_rect=Rect(80, 0, 360, 220))
        result = self.recorder.process(_snapshot(self.tiles, brush=p), observed_at=102)
        self.assertFalse(any(e['event'] == 'affix_card_clicked' for e in result.evidence['events']))

    def test_frame_pump_carries_evidence_separately_from_timings(self):
        pump = FramePump(capture_func=lambda: CaptureResult(np.zeros((8, 8, 3), np.uint8), 'test'),
                         analyze_func=lambda _: None, needs_full_refresh=lambda: False,
                         full_analyze_func=lambda _: None,
                         record_func=lambda s, _: self.recorder.process(s, observed_at=101))
        with patch('huahua_assistant.frame_pump.snapshot_from_scene',
                   return_value=_snapshot(self.tiles, brush=panel())):
            frame = pump._capture_and_analyze()
        self.assertIsNone(frame.error)
        self.assertTrue(frame.recording_evidence['events'])
        self.assertNotIn('recording_evidence', frame.perf)

    def test_metadata_only_choice_is_saved_by_quick_capture(self):
        state = PuzzleState(tiles=())
        before = snapshot()
        harness = SimpleNamespace(
            last_real_capture_step=before, recommendations=[], brush_scenario=None,
            _backpack_cache_details=lambda: [], _backpack_pages_payload=lambda: [],
            _resource_state=lambda: ResourceState(), _intern_board=lambda _: 1,
            _persist_backpack_snapshot=lambda _: None, _handle_tentative_step=lambda *a, **k: None,
            last_quick_step_key=None, _stamp_real_step=lambda s: s, _ensure_initial_snapshot=lambda _: None,
            real_trajectory_steps=[], quick_capture_record_count=0)
        evidence = {'session': 'recorder-a', 'events': [{'sequence': 5}]}
        with patch.object(app, '_quick_capture_quality', return_value={'ok': True}), \
             patch.object(app, '_quality_is_trusted_for_trajectory', return_value=True), \
             patch.object(app, '_compact_real_capture_step', return_value=copy.deepcopy(before)), \
             patch.object(app, '_infer_real_capture_action', return_value={'type': 'capture_update'}), \
             patch.object(app, 'record_observation'):
            saved = app.HuahuaApp._record_quick_capture_step(harness, None, state, .1,
                                                           recording_evidence=evidence)
        self.assertTrue(saved)
        self.assertEqual(harness.real_trajectory_steps[0]['recording_evidence'], evidence)

    def test_stale_previous_panel_is_not_reused_just_because_rect_is_equal(self):
        rect = Rect(0, 0, 200, 200)
        previous = BrushOptionsObservation(rect, ('stack', 'spread'), {}, None, (), {}, {}, .9)
        region = RegionObservation(RegionKind.BRUSH_OPTIONS, rect, .9, 'test')
        with patch.object(brush_panel, 'read_brush_panel_async', return_value=None), \
             patch.object(vision, '_match_brush_stage2_raster_templates', return_value=((), {})), \
             patch.object(vision, '_match_brush_stage2_windows_ocr', return_value=((), {})), \
             patch.object(vision, '_brush_stage1_options_from_cards', return_value=((), None)), \
             patch.object(vision, '_brush_card_pattern_scores', return_value={}):
            observed = vision._analyze_brush_options_realtime(np.zeros((200, 200, 3), np.uint8),
                                                             (region,), previous=previous)
        self.assertIsNot(observed, previous)
        self.assertEqual(observed.stage1_options, ())
        self.assertTrue(observed.reading_pending)
        self.assertTrue(_fast_brush(observed).reading_pending)

    def test_recording_ignores_ui_default_and_preserves_effects(self):
        effects = (PaintEffect(PaintDirection.HORIZONTAL, PaintObject.MONSTER, 1),
                   PaintEffect(PaintDirection.VERTICAL, PaintObject.GAMEPLAY, 2),
                   PaintEffect(PaintDirection.HORIZONTAL, PaintObject.MONSTER, 1))
        rule = CurrentPaintRule(Paint(PaintCategory.MONSTER, 3, '黑帮'),
                                PaintDirection.HORIZONTAL, PaintObject.MONSTER, 3, 1, effects=effects)
        self.assertEqual(len(compact_rule_with_code(rule)['effects']), 3)
        scene = SimpleNamespace(brush_options=None, backpack=None,
                                current_paint=SimpleNamespace(effect_text='面板实际原文', raw_text_hint=''))
        with patch.object(app, '_compact_board_summary', return_value={'states': 'B'*36}):
            step = app._compact_real_capture_step(scene=scene, state=PuzzleState(tiles=(), current_rule=rule),
                backpack=(), recommendations=[], brush_scenario=object(), resources=ResourceState())
        self.assertIsNone(step['brush']['chosen'])
        self.assertEqual(step['current']['effect'], '面板实际原文')
        self.assertEqual(step['current']['effects'][0], step['current']['effects'][2])
        changed = copy.deepcopy(step)
        changed['recording_evidence'] = {'session': 'recording', 'events': [{'sequence': 1}]}
        self.assertTrue(app._recording_metadata_changed(step, changed))
        raw = {'steps': [step, changed]}
        self.assertEqual(expand_trajectory_payload(compact_trajectory_payload(copy.deepcopy(raw))), raw)


class ScopeReplayTest(unittest.TestCase):
    def pair(self):
        before = snapshot(paints=7)
        for t in before['tiles'][:7]:
            t['s'] = '黑帮'
        before.update(anchor=[0, 0], pending=False)
        before['tiles'][6].update(c='gameplay', s='征战之神')
        before['current']['effects'] = [
            {'direction': 'horizontal', 'target': 'monster', 'upgrade': 1},
            {'direction': 'vertical', 'target': 'gameplay', 'upgrade': 2},
            {'direction': 'horizontal', 'target': 'monster', 'upgrade': 1}]
        after = copy.deepcopy(before)
        after['tiles'][1]['l'] = 5
        after['tiles'][6]['l'] = 5
        return before, after

    def test_effects_keep_separate_targets_duplicates_and_previous_anchor(self):
        before, after = self.pair()
        after['anchor'] = [5, 5]
        report = scope_comparison(before, after, 1)
        self.assertFalse(report['causal'])
        self.assertEqual(report['anchor'], [0, 0])
        self.assertEqual([e['increased'] for e in report['effects']], [1, 1, 1])
        self.assertEqual([t['cell'] for t in report['effects'][1]['targets']], [6])
        self.assertNotIn(0, report['effects'][0]['cells'])

    def test_unlike_unknown_or_distant_readings_are_not_confirmed_upgrades(self):
        before, after = self.pair()
        after['tiles'][1]['s'] = '鼠人'
        report = scope_comparison(before, after, 1)
        self.assertEqual(report['effects'][0]['increased'], 0)
        self.assertIn(1, report['effects'][0]['uncertain_cells'])
        self.assertEqual(scope_comparison(before, after, 20)['effects'][1]['increased'], 0)
        before['pending'] = True
        self.assertFalse(scope_comparison(before, after, 1)['available'])

    def test_observed_displayed_level_is_not_base_plus_buffs(self):
        before, after = self.pair()
        for s in (before, after):
            s['observed_tiles'] = [{**(t or {}), 'state': s['board']['states'][i],
                                    'confidence': .95, 'level_confidence': .95}
                                   for i, t in enumerate(s['tiles'])]
        before['observed_tiles'][1]['l'] = after['observed_tiles'][1]['l'] = 7
        after['tiles'][1].update(b=3, au=2)
        report = scope_comparison(before, after, 1)
        self.assertEqual(report['level_source'], 'displayed_observation')
        self.assertEqual(report['effects'][0]['increased'], 0)

    def test_old_missing_affix_stays_unknown_and_embedded_events_survive_archive(self):
        before, after = self.pair()
        self.assertEqual(selection_evidence(before)['status'], 'unknown')
        event = {'session': 'recorder-a', 'sequence': 3, 'ts': 101, 'event': 'affix_choice_resolved',
                 'payload': {'selected': '自身提升1级', 'source': 'effect_inference',
                             'observed_current': {'category': 'monster', 'subtype': '黑帮', 'level': 3,
                                                  'direction': 'horizontal', 'target': 'monster'}}}
        after['recording_evidence'] = {'session': 'recorder-a', 'events': [event]}
        run = make_run(archive(steps=[before, after]), 1, {}, [])
        self.assertEqual(run['steps'][1]['selection']['source'], 'effect_inference')
        self.assertIsNone(run['steps'][1]['selection']['brush_effect_id'])
        self.assertEqual(run['steps'][1]['events'][0]['sequence'], 3)
        after['recording_evidence']['events'].append({'event': 'affix_options_observed', 'payload': {}})
        self.assertEqual(selection_evidence(after)['status'], 'unknown')


if __name__ == '__main__':
    unittest.main()
