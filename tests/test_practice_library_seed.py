#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""练习库按种子查询的回归测试。

**种子 0 是合法种子**。旧实现 ``int(item.get('seed') or -1)`` 把 0 当成了缺值，
于是 ``by_seed(0)`` 永远查不到东西、``by_seed(-1)`` 反而会把所有缺种子记录捞出来。
这里用合成记录逐个钉住：0、负数、字符串形式、缺值、非数字、浮点、布尔。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from huahua_assistant import practice_library as pl


#: 「这个字段压根不存在」的哨兵。区别于显式写入 ``None``（那是"有字段但值未知"）。
_MISSING = object()


def _record(seed, *, signature: str, score: float = 0.5) -> dict:
    """最小可被 ``entries()`` 收下的记录：只要求 signature 存在。"""
    entry = {
        'schema': pl.PRACTICE_LIBRARY_SCHEMA,
        'signature': signature,
        'score': score,
        'grade': 'C',
        'origin': 'simulated',
        'source': 'test',
    }
    if seed is not _MISSING:
        entry['seed'] = seed
    return entry


class SeedQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / 'lib.jsonl'
        self.library = pl.PracticeLibrary(self.path)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _write(self, *records: dict) -> None:
        self.path.write_text(
            ''.join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n'
                for item in records
            ),
            encoding='utf-8',
        )

    # ---- 核心回归：0 ------------------------------------------------
    def test_seed_zero_is_found(self) -> None:
        self._write(_record(0, signature='s0'), _record(1, signature='s1'))
        hits = self.library.by_seed(0)
        self.assertEqual([item['signature'] for item in hits], ['s0'])

    def test_seed_zero_does_not_leak_into_other_queries(self) -> None:
        self._write(_record(0, signature='s0'))
        self.assertEqual(self.library.by_seed(1), [])
        self.assertEqual(self.library.by_seed(-1), [], '种子 0 不该被当成缺值匹配 -1')

    # ---- 缺值与无效值不能凑数 ---------------------------------------
    def test_missing_seed_not_returned_for_negative_query(self) -> None:
        self._write(_record(_MISSING, signature='noneseed'), _record(-1, signature='neg'))
        hits = self.library.by_seed(-1)
        self.assertEqual([item['signature'] for item in hits], ['neg'],
                         '只有真的种子 -1 能命中，缺种子记录不能')

    def test_missing_seed_not_returned_for_zero_query(self) -> None:
        self._write(_record(_MISSING, signature='noneseed'))
        self.assertEqual(self.library.by_seed(0), [])

    def test_non_numeric_seed_does_not_raise(self) -> None:
        self._write(_record('abc', signature='bad'), _record(2, signature='ok'))
        self.assertEqual([item['signature'] for item in self.library.by_seed(2)], ['ok'])
        self.assertEqual(self.library.by_seed('abc'), [], '无效查询值返回空而不是抛异常')

    # ---- 字符串形式 --------------------------------------------------
    def test_string_seed_matches_integer_query(self) -> None:
        self._write(_record('0', signature='str0'), _record(' 12 ', signature='str12'))
        self.assertEqual([item['signature'] for item in self.library.by_seed(0)], ['str0'])
        self.assertEqual([item['signature'] for item in self.library.by_seed(12)], ['str12'])

    # ---- seed_of 本身 -------------------------------------------------
    def test_seed_of_variants(self) -> None:
        cases = [
            (0, 0),
            (5, 5),
            (-3, -3),
            ('7', 7),
            (' 7 ', 7),
            ('', None),
            ('abc', None),
            (None, None),
            (5.0, 5),
            (5.5, None),
            (True, None),  # bool 是 int 子类，不能当 1
            (False, None),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(pl.seed_of({'seed': raw}), expected)

    def test_seed_of_missing_key(self) -> None:
        self.assertIsNone(pl.seed_of({'signature': 'x'}))
        self.assertIsNone(pl.seed_of({}))

    # ---- 视图不因 0 而说谎 -------------------------------------------
    def test_view_keeps_seed_zero(self) -> None:
        view = pl.LibraryEntryView.from_entry(_record(0, signature='s0', score=0.8))
        self.assertEqual(view.seed, 0)
        self.assertIn('seed=0', view.format_text())

    def test_view_survives_non_numeric_seed(self) -> None:
        view = pl.LibraryEntryView.from_entry(_record('abc', signature='s'))
        self.assertEqual(view.seed, 0)


if __name__ == '__main__':
    unittest.main()
