import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import cv2

from huahua_assistant.paint_panel import parse_lines
from huahua_assistant.models import PaintDirection, PaintObject, Rect
from huahua_assistant.brush_recording import (
    EvidenceStore, ChoiceTracker, match_card, audit_report,
    detect_cards,
    BrushEvidenceService, recognizer_revision, choice_at_time,
)
from huahua_assistant.replay_effects import selection_evidence
from huahua_assistant.mouse_monitor import MouseClickMonitor, MouseClick


class SeparateEffectsTest(unittest.TestCase):
    def test_wrapped_sections_do_not_mix_rules(self):
        reading = parse_lines([
            '怪物颜料·伊吉LV3', '笔触效果',
            '将纵向范围中最近1个阻挡地块', '转变为空白地块',
            '调色效果', '使相邻的怪物颜料提升2级', '颜料内容', '玩法提升7级',
        ])
        self.assertEqual(reading.brush_effect_text, '将纵向范围中最近1个阻挡地块转变为空白地块')
        self.assertEqual(reading.effect_text, '使相邻的怪物颜料提升2级')
        self.assertEqual(reading.direction, PaintDirection.ADJACENT)
        self.assertEqual(reading.target_object, PaintObject.MONSTER)
        self.assertEqual(reading.upgrade_amount, 2)

    def test_tint_is_not_a_selected_brush(self):
        reading = parse_lines(['调色效果:使相邻的怪物颜料提升2级'])
        self.assertIsNone(reading.brush_effect_text)

    def test_progress_effect_body_is_not_a_new_heading(self):
        reading = parse_lines(['笔触效果', '调色目标的进度要求减少2', '颜料内容', '内容'])
        self.assertEqual(reading.brush_effect_text, '调色目标的进度要求减少2')

    def test_real_tooltip_lines_keep_title_level_and_two_effects(self):
        reading = parse_lines(['怪物颜料·伊吉', 'LV4', '调色效果', '使相邻的怪物颜料提升2级',
                               '笔触效果', '将纵向范围中最近1个阻挡地块', '转变为空白地块',
                               '颜料内容', '雪原中出现伊吉怪物；', '调色目标',
                               '在相邻地块上放置怪物颜料', '(2/4)'])
        self.assertEqual(reading.subtype, '伊吉')
        self.assertEqual(reading.level, 4)
        self.assertEqual(reading.match_required, 4)
        self.assertEqual(match_card([reading.brush_effect_text])['brush_effect_id'], 3003)


class MatchTest(unittest.TestCase):
    def test_unknown_and_duplicate_slots(self):
        self.assertIsNone(match_card([])['name'])
        result = match_card(['当前颜料可放置在阻挡地块'])
        self.assertEqual(result['brush_effect_id'], 3010)
        diagonal = match_card(['作用方向改为斜对角'])
        self.assertIsNone(diagonal['brush_effect_id'])
        self.assertEqual(diagonal['brush_effect_ids'], [1005, 4003])

    def test_partial_numbers_are_not_guessed(self):
        self.assertIsNone(match_card(['将纵向范围中最近个阻挡地块转变为空白地块'])['name'])

    def test_real_geometry_scales_without_calibration(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'jietu/扩散后选择前.png'
        if not path.exists():
            self.skipTest('local real screenshot unavailable')
        original = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        for scale in (.65, .75, 1, 1.2):
            with self.subTest(scale=scale):
                frame = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                _, found = detect_cards(frame)
                self.assertIsNotNone(found)
                board, cards, confirm = found
                self.assertLess(abs(board.y/scale-222), 8)
                self.assertEqual(len(cards), 2)
                self.assertIsNotNone(confirm)
                self.assertGreater(confirm.y, cards[-1].bottom)

    def test_resize_keeps_real_choice_identity(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'jietu/扩散后选择前.png'
        if not path.exists():
            self.skipTest('local real screenshot unavailable')
        original = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        with tempfile.TemporaryDirectory() as folder:
            tracker = ChoiceTracker(EvidenceStore(folder))
            identities = []
            for i, scale in enumerate((1., .75, 1.2, .65, 1.)):
                frame = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                _, (_, rects, confirm) = detect_cards(frame)
                cards = [frame[r.y:r.bottom, r.x:r.right] for r in rects]
                identity, _ = tracker.observe(cards, rects, confirm, geometry=(1, 0, 0, frame.shape[1], frame.shape[0]),
                                               session='s', ts=10+i)
                identities.append(identity)
            self.assertEqual(len(set(identities)), 1)


class RecordingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = EvidenceStore(Path(self.temp.name), max_bytes=100_000)
        self.tracker = ChoiceTracker(self.store)
        self.cards = (np.full((40, 160, 3), 80, np.uint8), np.full((40, 160, 3), 100, np.uint8))
        self.rects = (Rect(10, 10, 160, 40), Rect(10, 65, 160, 40))
        self.confirm = Rect(110, 130, 60, 20)

    def show(self, ts=10, session='s1', geometry=(1, 0, 0, 200, 200)):
        return self.tracker.observe(self.cards, self.rects, self.confirm,
                                    geometry=geometry, session=session, ts=ts)

    def test_late_ocr_keeps_slots_and_original_session(self):
        choice, _ = self.show()
        self.tracker.click(10.1, 20, 75, (1, 0, 0, 200, 200))
        self.tracker.click(10.2, 130, 140, (1, 0, 0, 200, 200))
        self.tracker.absent(13)
        self.tracker.absent(14)
        next_choice, _ = self.show(15, 's2')
        self.tracker.reading(choice, [match_card([]), match_card(['当前颜料可放置在阻挡地块'])])
        record = self.store.records()[choice]
        self.assertEqual(record['session'], 's1')
        self.assertEqual(record['selected_index'], 1)
        self.assertEqual(record['brush_effect_id'], 3010)
        self.assertEqual(record['status'], 'confirmed')
        self.assertNotEqual(choice, next_choice)
        self.assertIsNone(self.store.records()[next_choice].get('brush_effect_id'))

    def test_disappearance_is_not_confirmation(self):
        choice, _ = self.show()
        self.tracker.reading(choice, [match_card(['当前颜料可放置在阻挡地块'])] * 2)
        self.tracker.click(10.1, 20, 20, (1, 0, 0, 200, 200))
        self.tracker.absent(12)
        self.tracker.absent(13)
        record = self.store.records()[choice]
        self.assertEqual(record['status'], 'unconfirmed')
        self.assertIsNone(record.get('brush_effect_id'))

    def test_resize_rejects_old_geometry_click_and_gap(self):
        choice, _ = self.show()
        self.show(11, geometry=(1, 10, 10, 200, 200))
        self.tracker.click(12, 20, 20, (1, 0, 0, 200, 200))
        self.tracker.click(12, 20, 58, (1, 10, 10, 200, 200))
        self.assertIsNone(self.store.records()[choice]['selected_index'])

    def test_cap_dedupe_and_gate(self):
        image = self.cards[0]
        first = self.store.save_crop(image)
        self.assertEqual(first, self.store.save_crop(image))
        self.assertEqual(len(list((Path(self.temp.name) / 'images').glob('*'))), 1)
        tiny = EvidenceStore(Path(self.temp.name) / 'tiny', max_bytes=1, mode='id_only')
        self.assertEqual(tiny.mode, 'validation')
        self.assertIsNone(tiny.save_crop(image))
        self.assertEqual(tiny.image_bytes, 0)

    def test_same_effect_in_two_choices_not_deduplicated(self):
        first, _ = self.show()
        self.tracker.absent(11)
        self.tracker.absent(12)
        second, _ = self.show(13)
        self.assertNotEqual(first, second)

    def test_audit_counts_unknown_and_missing_as_errors(self):
        records = {'x': {'choice_id': 'x', 'session': 's', 'status': 'unconfirmed'}}
        reviews = [{'choice_id': 'x', 'expected_canonical_id': 'x', 'reviewer': 'human'},
                   {'choice_id': 'missed-1', 'expected_canonical_id': 'x', 'reviewer': 'human'}]
        report = audit_report(records, reviews)
        self.assertEqual(report['reviewed'], 2)
        self.assertEqual(report['correct'], 0)
        self.assertFalse(report['eligible'])

    def test_confirm_timeout_cannot_validate_later_disappearance(self):
        key, _ = self.show()
        self.tracker.reading(key, [match_card(['当前颜料可放置在阻挡地块'])]*2)
        self.tracker.click(10.1, 20, 20, (1, 0, 0, 200, 200))
        self.tracker.click(10.2, 130, 140, (1, 0, 0, 200, 200))
        self.show(20)
        self.tracker.absent(21)
        self.tracker.absent(22)
        self.assertEqual(self.store.records()[key]['status'], 'unconfirmed')

    def test_late_ocr_conflict_revokes_confirmation(self):
        key, _ = self.show()
        self.tracker.click(10.1, 20, 20, (1, 0, 0, 200, 200))
        self.tracker.click(10.2, 130, 140, (1, 0, 0, 200, 200))
        self.tracker.absent(10.3)
        self.tracker.absent(10.7)
        self.tracker.reading(key, [match_card(['当前颜料可放置在阻挡地块'])]*2)
        self.assertEqual(self.store.records()[key]['status'], 'confirmed')
        self.tracker.reading(key, [match_card(['作用方向改为横向'])]*2)
        self.assertEqual(self.store.records()[key]['status'], 'unconfirmed')
        self.assertIsNone(self.store.records()[key]['brush_effect_id'])

    def test_interrupted_ocr_recovers_from_original_choice(self):
        key, _ = self.show()
        paths = [self.store.save_crop(card) for card in self.cards]
        self.tracker.reading(key, [], {'captured_at': 10, 'images': paths, 'ocr_pending': True})
        service = BrushEvidenceService(MouseClickMonitor(), store=self.store)
        service._recover_pending()
        job = service.jobs.get_nowait()
        self.assertEqual(job[0], key)
        self.assertEqual(job[2], 10)
        self.assertEqual(len(job[1]), 2)

    def test_id_only_stores_no_images_or_ocr_and_replays_name_from_id(self):
        with patch.object(EvidenceStore, 'audit', return_value={'eligible': True}):
            store = EvidenceStore(Path(self.temp.name)/'ids', mode='id_only')
        record = {'choice_id': 'c', 'session': 's', 'stage': 'effect', 'status': 'confirmed',
                  'selected_index': 0, 'selected': '当前颜料可放置在阻挡地块',
                  'cards': [match_card(['当前颜料可放置在阻挡地块'])],
                  'samples': [{'ocr_lines': ['large text']}],
                  **match_card(['当前颜料可放置在阻挡地块'])}
        store.write(record)
        self.assertIsNone(store.save_crop(self.cards[0]))
        saved = store.records()['c']
        self.assertNotIn('selected', saved)
        self.assertEqual(saved['samples'], [])
        self.assertNotIn('name', saved['cards'][0])
        step = {'recording_evidence': {'brush_choices': {'schema': 'brush-evidence-v1', 'choices': [saved]}}}
        self.assertEqual(selection_evidence(step)['selected'], '当前颜料可放置在阻挡地块')
        from huahua_assistant.brush_review import write_report
        output = Path(self.temp.name)/'review.html'
        write_report(store, output)
        self.assertIn('当前颜料可放置在阻挡地块', output.read_text(encoding='utf8'))
        self.assertIn('尚无样本', output.read_text(encoding='utf8'))

    def test_replay_never_uses_future_clicks_or_placement(self):
        record = {'choice_id': 'c', 'stage': 'effect', 'status': 'confirmed',
                  'observed_at': 10, 'clicked_at': 11, 'selected_index': 1,
                  'confirmed_at': 12, 'closed_at': 13, 'selected': 'effect',
                  'canonical_affix_id': 'effect', 'brush_effect_ids': [3010],
                  'placement': {'ts': 14}, 'paint_observation': {'ts': 15}}
        early = choice_at_time(record, 10.5)
        self.assertIsNone(early['selected_index'])
        self.assertIsNone(early['selected'])
        self.assertIsNone(early.get('brush_effect_id'))
        after = choice_at_time(record, 13.5)
        self.assertEqual(after['selected'], 'effect')
        self.assertIsNone(after['placement'])
        self.assertIsNone(after['paint_observation'])
        self.assertEqual(record['placement']['ts'], 14)

    def test_resize_in_flight_clicks_never_use_previous_bounds(self):
        key, _ = self.show()
        service = BrushEvidenceService(MouseClickMonitor(), store=self.store)
        service.tracker, service.session = self.tracker, 's1'
        service._accept_sample(None, (), (1, 10, 10, 100, 100), 11, 's1',
                               [MouseClick(10.8, 25, 25, (1, 10, 10, 100, 100))])
        self.assertIsNone(self.store.records()[key]['selected_index'])

    def test_frame_from_previous_session_is_discarded(self):
        service = BrushEvidenceService(MouseClickMonitor(), store=self.store)
        service.session = 'new'
        service._accept_sample(np.zeros((200, 200, 3), np.uint8),
                               (Rect(0, 0, 80, 80), self.rects, self.confirm),
                               (1, 0, 0, 200, 200), 10, 'old', [])
        self.assertFalse(self.store.records())

    def test_fast_placement_during_panel_close_is_retained(self):
        key, _ = self.show()
        geometry = (1, 0, 0, 200, 200)
        self.tracker.click(10.1, 20, 20, geometry)
        self.tracker.click(10.2, 130, 140, geometry)
        service = BrushEvidenceService(MouseClickMonitor(), store=self.store)
        service.tracker, service.session = self.tracker, 's1'
        board = Rect(0, 0, 60, 60)
        service._accept_sample(None, None, geometry, 10.5, 's1',
                               [MouseClick(10.4, 25, 35, geometry)], board=board)
        self.tracker.reading(key, [match_card(['当前颜料可放置在阻挡地块'])]*2)
        record = self.store.records()[key]
        self.assertEqual(record['status'], 'confirmed')
        self.assertEqual((record['placement_click']['row'], record['placement_click']['col']), (3, 2))

    def test_click_feeds_are_independent(self):
        monitor = MouseClickMonitor()
        click = MouseClick(10, 1, 2, (1, 0, 0, 100, 100))
        monitor._clicks.append(click)
        monitor._history.append(click)
        with patch('huahua_assistant.mouse_monitor.time.time', return_value=10):
            self.assertEqual(monitor.drain(), (click,))
        self.assertEqual(monitor.since(9), (click,))

    def test_mouse_hook_uses_same_visible_bounds_as_capture(self):
        from huahua_assistant.mouse_monitor import _click_window_geometry
        from ctypes import wintypes
        def get_rect(hwnd, pointer):
            rect = pointer._obj
            rect.left, rect.top, rect.right, rect.bottom = (91, 91, 509, 509)
            return 1
        user32 = SimpleNamespace(GetForegroundWindow=lambda: 1, GetWindowRect=get_rect)
        with patch('huahua_assistant.capture._visible_bounds', return_value=wintypes.RECT(100, 100, 500, 500)):
            self.assertEqual(_click_window_geometry(user32), (1, 100, 100, 400, 400))

    def test_recording_does_not_promote_legacy_inference(self):
        from huahua_assistant.run_recorder import AutomaticRunRecorder
        recorder = AutomaticRunRecorder(path=None, external_brush_choices=True)
        self.addCleanup(recorder.close)
        recorder._emit('affix_choice_resolved', {'selected': '当前颜料可放置在阻挡地块',
                                               'source': 'effect_inference'}, 10)
        event = recorder._events[-1]
        self.assertEqual(event['event'], 'affix_choice_legacy_candidate')
        self.assertNotIn('selected', event['payload'])
        self.assertNotIn('brush_effect_id', event['payload'])

    def test_audit_requires_coverage_and_does_not_double_count_labels(self):
        revision = recognizer_revision()
        records, reviews = {}, []
        for i in range(400):
            key = str(i)
            records[key] = {'choice_id': key, 'session': str(i%10), 'frame_size': [100+i%3, 100],
                            'stage': 'effect', 'status': 'confirmed', 'selected_index': 0,
                            'canonical_affix_id': 'x', 'brush_effect_ids': [3010], 'recognizer_revision': revision}
            reviews.append({'choice_id': key, 'expected_index': 0, 'expected_canonical_id': 'x',
                            'expected_official_ids': [3010], 'reviewer': 'test-human', 'recognizer_revision': revision})
        report = audit_report(records, reviews*2)
        self.assertEqual(report['reviewed'], 400)
        self.assertTrue(report['eligible'])
        records['unreviewed'] = {**records['0'], 'choice_id': 'unreviewed', 'status': 'unconfirmed'}
        self.assertFalse(audit_report(records, reviews)['eligible'])


if __name__ == '__main__':
    unittest.main()
