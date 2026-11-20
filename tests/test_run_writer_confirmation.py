#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""写盘确认 + 中断局可见 的回归测试。

要钉住的两件事：
1. **"入队"不等于"落盘"**——慢盘、写失败、关闭超时都必须能被看见，
   而不是界面显示"已保存"、实际那一局不在文件里；
2. **没录完的局也要能列出来**，并且说清"停在哪一步"，不拿别的原因冒充结局。
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from huahua_assistant import run_index as ri
from huahua_assistant.run_recorder import _AsyncJsonlWriter, _board_payload  # noqa: F401


def _event(session: str, sequence: int, event: str = 'current_paint_changed', **payload) -> dict:
    return {
        'schema': 'huahua-auto-record-v1',
        'ts': 1000.0 + sequence,
        'session': session,
        'sequence': sequence,
        't': float(sequence),
        'event': event,
        'payload': payload,
    }


class _SlowWriter(_AsyncJsonlWriter):
    """每写一行睡一下，模拟慢盘。"""

    def __init__(self, path: Path, delay: float) -> None:
        self.delay = delay
        super().__init__(path)
        self.written: list[int] = []

    def _write_line(self, handle, item: dict) -> None:
        time.sleep(self.delay)
        super()._write_line(handle, item)
        self.written.append(int(item.get('sequence') or 0))


class _FailingWriter(_AsyncJsonlWriter):
    """前 N 行写失败（权限/磁盘满），之后正常。"""

    def __init__(self, path: Path, fail_first: int) -> None:
        self.fail_first = fail_first
        self.attempts = 0
        super().__init__(path)

    def _write_line(self, handle, item: dict) -> None:
        self.attempts += 1
        if self.attempts <= self.fail_first:
            raise OSError('disk full')
        super()._write_line(handle, item)


class WriterConfirmationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / 'rec.jsonl'

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_normal_close_settles(self) -> None:
        writer = _AsyncJsonlWriter(self.path)
        for index in range(5):
            self.assertTrue(writer.submit(_event('s1', index)))
        status = writer.close(timeout=2.0)
        self.assertTrue(status.settled, '正常关闭后最后入队的一条必须已经落盘')
        self.assertEqual(status.pending, 0)
        self.assertFalse(status.degraded)
        lines = self.path.read_text(encoding='utf-8').strip().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual([json.loads(line)['sequence'] for line in lines], [0, 1, 2, 3, 4])

    def test_submit_before_write_is_not_confirmation(self) -> None:
        """入队成功 ≠ 已落盘。慢盘下这条差额必须能被看到。"""
        writer = _SlowWriter(self.path, delay=0.05)
        for index in range(4):
            writer.submit(_event('s1', index))
        mid = writer.status()
        self.assertFalse(mid.settled, '刚入队就宣称写完，正是要消灭的假安慰')
        self.assertGreater(mid.pending, 0)
        final = writer.close(timeout=5.0)
        self.assertTrue(final.settled)

    def test_queue_full_drops_and_reports(self) -> None:
        writer = _SlowWriter(self.path, delay=0.05)
        writer.QUEUE_LIMIT = 2
        # 用一个小队列直接验证背压：满了要丢最晚的事件并记数，不能卡死采集线程。
        writer._queue.maxsize = 2
        accepted = 0
        for index in range(12):
            if writer.submit(_event('s1', index)):
                accepted += 1
        status = writer.close(timeout=5.0)
        self.assertLess(accepted, 12, '队列满了必须拒绝入队，而不是无界堆积')
        self.assertGreater(status.dropped, 0)
        self.assertTrue(status.degraded)
        self.assertEqual(status.last_error, 'queue_full')

    def test_write_failure_is_visible_and_does_not_kill_writer(self) -> None:
        writer = _FailingWriter(self.path, fail_first=2)
        for index in range(5):
            writer.submit(_event('s1', index))
        status = writer.close(timeout=5.0)
        self.assertEqual(status.failures, 2)
        self.assertIn('disk full', status.last_error or '')
        self.assertTrue(status.degraded)
        # 坏两条之后还写进了三条，说明单条失败没有把线程带死
        lines = self.path.read_text(encoding='utf-8').strip().splitlines()
        self.assertEqual(len(lines), 3)

    def test_close_timeout_is_reported(self) -> None:
        writer = _SlowWriter(self.path, delay=0.4)
        for index in range(4):
            writer.submit(_event('s1', index))
        status = writer.close(timeout=0.05)
        self.assertFalse(status.settled, '超时没排空就不能说"已保存"')
        self.assertEqual(status.last_error, 'close_timeout')
        # 收尾：别把线程留在后台（它会在 duetime 之后自然写完）
        writer._thread.join(timeout=5.0)

    def test_second_close_is_idempotent(self) -> None:
        writer = _AsyncJsonlWriter(self.path)
        writer.submit(_event('s1', 0))
        writer.close(timeout=2.0)
        again = writer.close(timeout=2.0)
        self.assertEqual(again.pending, 0)
        self.assertFalse(writer.submit(_event('s1', 1)), '关闭之后不再接收')


class RunIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / 'rec.jsonl'

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _write(self, lines: list[dict | str], *, trailing_newline: bool = True) -> None:
        parts = []
        for item in lines:
            parts.append(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False))
        text = '\n'.join(parts)
        if trailing_newline:
            text += '\n'
        self.path.write_text(text, encoding='utf-8')

    def test_complete_and_interrupted_sessions(self) -> None:
        self._write(
            [
                _event('s1', 0, 'run_started', reason='start_screen'),
                _event('s1', 1),
                _event('s1', 2, 'run_finished', reason='finish_button'),
                _event('s2', 0, 'run_started', reason='active_scene'),
                _event('s2', 1),
            ]
        )
        index = ri.index_run_log(self.path)
        self.assertEqual(len(index.sessions), 2)
        self.assertEqual([item.session for item in index.complete_sessions], ['s1'])
        self.assertEqual([item.session for item in index.partial_sessions], ['s2'])
        interrupted = index.find('s2')
        self.assertEqual(interrupted.outcome, ri.OUTCOME_INTERRUPTED)
        self.assertEqual(interrupted.last_reliable_step, 'current_paint_changed#1')

    def test_board_full_is_not_a_confirmed_result(self) -> None:
        """「满盘」只是终止条件，不能当成"这局结算成功了"。"""
        self._write(
            [
                _event('s1', 0, 'run_started', reason='start_screen'),
                _event('s1', 1, 'run_finished', reason='no_blank_tiles'),
            ]
        )
        summary = ri.index_run_log(self.path).find('s1')
        self.assertTrue(summary.complete)
        self.assertEqual(summary.outcome, ri.OUTCOME_BOARD_FULL)
        self.assertNotEqual(summary.outcome, ri.OUTCOME_CONFIRMED)
        self.assertEqual(ri.outcome_of_reason('finish_button'), ri.OUTCOME_FINISH_SEEN)
        self.assertEqual(ri.outcome_of_reason(None), ri.OUTCOME_UNKNOWN)
        self.assertEqual(ri.outcome_of_reason('some_new_reason'), ri.OUTCOME_UNKNOWN)

    def test_bad_line_in_middle_is_reported_with_line_number(self) -> None:
        self._write(
            [
                _event('s1', 0, 'run_started', reason='start_screen'),
                '{"broken": ',
                _event('s1', 2, 'run_finished', reason='finish_button'),
            ]
        )
        index = ri.index_run_log(self.path)
        self.assertEqual(len(index.bad_lines), 1)
        self.assertEqual(index.bad_lines[0].line_no, 2)
        self.assertFalse(index.truncated_tail, '中间写坏不是"末尾截断"')
        self.assertEqual(len(index.complete_sessions), 1)

    def test_truncated_tail_is_flagged(self) -> None:
        self._write(
            [
                _event('s1', 0, 'run_started', reason='start_screen'),
                '{"schema": "huahua-auto-record-v1", "session": "s1", "seque',
            ],
            trailing_newline=False,
        )
        index = ri.index_run_log(self.path)
        self.assertTrue(index.truncated_tail)
        self.assertEqual(index.bad_lines[-1].line_no, index.total_lines)

    def test_duplicate_sequences_are_reported(self) -> None:
        self._write(
            [
                _event('s1', 0, 'run_started', reason='start_screen'),
                _event('s1', 1),
                _event('s1', 1),
            ]
        )
        summary = ri.index_run_log(self.path).find('s1')
        self.assertEqual(summary.duplicate_sequences, (1,))

    def test_missing_file_is_empty_index(self) -> None:
        index = ri.index_run_log(Path(self._dir.name) / 'nope.jsonl')
        self.assertEqual(index.sessions, [])
        self.assertEqual(index.total_lines, 0)

    def test_table_mentions_partial_sessions(self) -> None:
        self._write([_event('s2', 0, 'run_started', reason='active_scene')])
        text = ri.index_run_log(self.path).format_table()
        self.assertIn('s2', text)
        self.assertIn('中断', text)
        self.assertIn('合计 1 局', text)


class RecorderStatusWiringTest(unittest.TestCase):
    """录制器状态要能带出写盘侧的三个量。"""

    def test_status_reports_writer_fields(self) -> None:
        from huahua_assistant.run_recorder import AutomaticRunRecorder

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rec.jsonl'
            recorder = AutomaticRunRecorder(path)
            status = recorder.status(now=time.time())
            self.assertEqual(status.writer_pending, 0)
            self.assertIsNone(status.writer_error)
            self.assertIsNone(status.writer_written_sequence)
            recorder.close()
            report = recorder.close_report()
            self.assertTrue(report['settled'])


class RecordStatusTextTest(unittest.TestCase):
    """灯上的文案：写盘降级必须说出来。"""

    def test_degraded_text_mentions_dropped(self) -> None:
        from huahua_assistant.run_recorder import RecorderStatus, describe_record_status

        status = RecorderStatus(
            phase='playing',
            session_id='s1',
            event_count=12,
            duration=61.0,
            frames=400,
            missing_frames=0,
            last_event_age=0.5,
            path='runtime/auto_run_records.jsonl',
            writer_dropped=3,
            writer_failures=1,
        )
        text, level = describe_record_status(status, auto_follow=True)
        self.assertEqual(level, 'recording')
        self.assertIn('写盘降级', text)
        self.assertIn('丢3条', text)

    def test_clean_text_has_no_warning(self) -> None:
        from huahua_assistant.run_recorder import RecorderStatus, describe_record_status

        status = RecorderStatus(
            phase='playing',
            session_id='s1',
            event_count=12,
            duration=61.0,
            frames=400,
            missing_frames=0,
            last_event_age=0.5,
            path='runtime/auto_run_records.jsonl',
        )
        text, _ = describe_record_status(status, auto_follow=True)
        self.assertNotIn('写盘降级', text)


if __name__ == '__main__':
    unittest.main()
