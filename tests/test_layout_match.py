#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""布局标定匹配诊断的回归测试（2026-11-23「昨天稳定今天全错」事故）。

事故特征：每一格都读错，但工具**自评质量分没掉**、``unknown_tiles`` 还是 0。
原因不是分类器变差，而是棋盘定位错了——定位是按窗口尺寸分别标定的，
当前尺寸在标定表里找不到匹配时，``load_layout`` 会静默回落到内置默认比例。

``layout.py`` 的注释里还有一次同形事故（2026-11-19）：

> ``1922x1030 -> x_ratio=0.0, y_ratio=0.0``，也就是「棋盘 = 窗口左上角」。
> 真盘面是 空18/阻挡6/虚空11/颜料1，程序读到的是 空3/阻挡10/虚空17/颜料6，
> 整个开局全错。

所以必须把「这帧用的是哪组标定、有没有匹配上」记进观测，否则无从复盘。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from huahua_assistant import layout


def _write_layout(entries: list[tuple[int, int, float, float, float]]) -> Path:
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8')
    payload = {
        'canvas': [1600, 900],
        'calibrations': [
            {
                'width': width,
                'height': height,
                'x_ratio': x_ratio,
                'y_ratio': y_ratio,
                'size_ratio': size_ratio,
            }
            for width, height, x_ratio, y_ratio, size_ratio in entries
        ],
    }
    json.dump(payload, handle)
    handle.close()
    return Path(handle.name)


class LayoutMatchDiagnosticTest(unittest.TestCase):
    """last_layout_match() 要能说清「命中了没有」。"""

    def test_reports_match_for_calibrated_size(self) -> None:
        path = _write_layout([(1922, 1030, 0.176, 0.251, 0.602)])
        layout.load_layout(1922, 1030, path=path)
        report = layout.last_layout_match()
        self.assertTrue(report['matched'])
        self.assertFalse(report['fallback_to_default'])
        self.assertEqual(report['width'], 1922)
        self.assertAlmostEqual(report['entry']['x_ratio'], 0.176)

    def test_reports_fallback_for_unknown_size(self) -> None:
        """换分辨率/最大化后尺寸没标定过 → 必须明确报「回落默认」。"""
        path = _write_layout([(1922, 1030, 0.176, 0.251, 0.602)])
        layout.load_layout(2560, 1440, path=path)
        report = layout.last_layout_match()
        self.assertFalse(report['matched'], '没标定过的尺寸不该报「已匹配」')
        self.assertTrue(report['fallback_to_default'])
        self.assertIsNone(report['entry'])
        self.assertEqual(report['entries'], 1)

    def test_nearby_size_still_matches(self) -> None:
        """相差几个像素（DPI 取整）仍算同一个窗口。"""
        path = _write_layout([(1922, 1030, 0.176, 0.251, 0.602)])
        layout.load_layout(1920, 1024, path=path)
        self.assertTrue(layout.last_layout_match()['matched'])

    def test_missing_file_reports_fallback(self) -> None:
        layout.load_layout(1922, 1030, path=Path('definitely-not-here.json'))
        report = layout.last_layout_match()
        self.assertFalse(report['matched'])
        self.assertEqual(report['entries'], 0)

    def test_fallback_returns_default_ratios(self) -> None:
        """回落时拿到的确实是内置默认比例——正是「静默读错」的来源。"""
        path = _write_layout([(1922, 1030, 0.176, 0.251, 0.602)])
        fallback = layout.load_layout(2560, 1440, path=path)
        default = layout.ScreenLayout()
        self.assertEqual(fallback.active[0].x_ratio, default.active[0].x_ratio)


if __name__ == '__main__':
    unittest.main()
