#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""区域标定：相对比例必须跨窗口尺寸稳定。

用户 2026-11-24 要求把识别范围（棋盘 / 调色范围 / 2 个笔触卡片）画成可拖动的
绿框并固定。之所以用相对比例而不是像素，是因为同一局里两条抓图路径给出的
窗口尺寸并不一致（实测 1602x939 与 800x469，正好差 2 倍），用像素存盘必然
有一条永远匹配不上。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from huahua_assistant import region_calibration as calib


class RegionRoundTripTests(unittest.TestCase):
    def test_pixels_ratio_round_trip(self) -> None:
        region = calib.Region.from_pixels("board", 300, 240, 558, 558, 1602, 939)
        self.assertEqual(region.to_pixels(1602, 939), (300, 240, 558, 558))

    def test_ratio_survives_resize(self) -> None:
        """同一块区域在 1602x939 与 800x469 下必须等比——这是本模块存在的理由。"""
        big = calib.Region.from_pixels("board", 300, 240, 558, 558, 1602, 939)
        small = big.to_pixels(800, 469)
        self.assertEqual(small, (150, 120, 279, 279))
        # 反算比例一致（允许 1px 整数取整误差）
        back = calib.Region.from_pixels("board", *small, 800, 469)
        self.assertAlmostEqual(back.x_ratio, big.x_ratio, places=2)
        self.assertAlmostEqual(back.w_ratio, big.w_ratio, places=2)

    def test_board_stays_square_across_sizes(self) -> None:
        """棋盘必须永远是正方形。

        2026-11-24 端到端测试实测到：正方形若按「相对宽 / 相对高」分别存，
        1602x939 下 w=h=558 只是巧合，换到 1536x864 就成了 535x513 的长方形，
        6x6 网格被拉伸，格子裁切跟着偏。
        """
        board = calib.Region.from_pixels("board", 300, 240, 558, 558, 1602, 939)
        ratio_short = []
        for width, height in ((1602, 939), (1536, 864), (800, 469), (1920, 1080), (1280, 720)):
            x, y, w, h = board.to_pixels(width, height)
            self.assertEqual(w, h, "棋盘在 %dx%d 下不是正方形" % (width, height))
            ratio_short.append(w / min(width, height))
        # 边长相对短边必须稳定（0.594 左右），否则换尺寸就跑偏
        for value in ratio_short:
            self.assertAlmostEqual(value, ratio_short[0], places=2)

    def test_square_resize_keeps_square(self) -> None:
        board = calib.Region.from_pixels("board", 300, 240, 558, 558, 1602, 939)
        bigger = board.resized_by(60, 10, 1602, 939)
        x, y, w, h = bigger.to_pixels(1602, 939)
        self.assertEqual(w, h)
        self.assertGreater(w, 558)

    def test_minimum_side_enforced(self) -> None:
        region = calib.Region.from_pixels("scope", 10, 10, 1, 1, 1600, 900)
        _, _, w, h = region.to_pixels(1600, 900)
        self.assertGreaterEqual(w, calib.MIN_SIDE_PX)
        self.assertGreaterEqual(h, calib.MIN_SIDE_PX)

    def test_region_is_clamped_inside_window(self) -> None:
        region = calib.Region.from_pixels("board", 1500, 800, 400, 400, 1600, 900)
        x, y, w, h = region.to_pixels(1600, 900)
        self.assertLessEqual(x + w, 1600)
        self.assertLessEqual(y + h, 900)

    def test_move_keeps_size(self) -> None:
        region = calib.Region.from_pixels("board", 300, 240, 400, 400, 1600, 900)
        moved = region.moved_by(50, -20, 1600, 900)
        before = region.to_pixels(1600, 900)
        after = moved.to_pixels(1600, 900)
        self.assertEqual((before[2], before[3]), (after[2], after[3]))
        self.assertEqual(after[0] - before[0], 50)
        self.assertEqual(after[1] - before[1], -20)

    def test_resize_changes_size(self) -> None:
        """长方形区域（笔触卡片）各轴独立缩放。"""
        region = calib.Region.from_pixels("brush_stage1", 300, 240, 400, 300, 1600, 900)
        bigger = region.resized_by(60, 40, 1600, 900)
        before = region.to_pixels(1600, 900)
        after = bigger.to_pixels(1600, 900)
        self.assertEqual(after[2] - before[2], 60)
        self.assertEqual(after[3] - before[3], 40)


class MergeRegionTests(unittest.TestCase):
    """三层合成必须保证四块区域都在。

    用户 2026-11-25 实测：「固定了棋盘和调色区域，但是笔触卡片1、2都未出现」。
    原因是识别结果只在拿到笔触面板时才带 brush_stage1/2，用 ``or`` 串联会被
    非空字典短路，默认值整层被跳过。
    """

    def test_partial_layer_keeps_all_four(self) -> None:
        partial = {
            "board": calib.Region.from_pixels("board", 240, 190, 450, 450, 1294, 758),
            "scope": calib.Region.from_pixels("scope", 260, 210, 410, 410, 1294, 758),
        }
        merged = calib.merge_regions(1294, 758, partial)
        self.assertEqual(set(merged), set(calib.REGION_LABELS))
        self.assertIn("brush_stage1", merged)
        self.assertIn("brush_stage2", merged)

    def test_later_layers_win(self) -> None:
        current = {"board": calib.Region.from_pixels("board", 100, 100, 300, 300, 1294, 758)}
        saved = {"board": calib.Region.from_pixels("board", 200, 150, 400, 400, 1294, 758)}
        merged = calib.merge_regions(1294, 758, current, saved)
        self.assertEqual(merged["board"].to_pixels(1294, 758), (200, 150, 400, 400))

    def test_empty_layers_give_defaults(self) -> None:
        merged = calib.merge_regions(1294, 758, None, {})
        self.assertEqual(set(merged), set(calib.REGION_LABELS))
        defaults = calib.default_regions(1294, 758)
        for key in calib.REGION_LABELS:
            self.assertEqual(merged[key], defaults[key])


class DefaultAndSaveTests(unittest.TestCase):
    def test_default_regions_cover_all_keys(self) -> None:
        regions = calib.default_regions(1602, 939)
        self.assertEqual(set(regions), set(calib.REGION_LABELS))

    def test_cells_of_yields_36(self) -> None:
        cells = list(calib.cells_of((339, 259, 620, 620)))
        self.assertEqual(len(cells), 36)
        # 首格在左上、末格在右下
        self.assertEqual(cells[0][:2], (339, 259))
        last_x, last_y, _, _ = cells[-1]
        self.assertEqual(last_x, 339 + round(5 * 620 / 6))
        self.assertEqual(last_y, 259 + round(5 * 620 / 6))

    def test_save_and_load_round_trip(self) -> None:
        regions = calib.default_regions(1602, 939)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            original = calib.CALIBRATION_PATH
            try:
                calib.CALIBRATION_PATH = path
                saved = calib.save_regions(regions, width=1602, height=939)
                self.assertTrue(saved.exists())
                payload = json.loads(saved.read_text(encoding="utf-8"))
                self.assertEqual(payload["width"], 1602)
                self.assertEqual(len(payload["regions"]), 4)
                back = calib.load_regions()
                self.assertEqual(set(back), set(calib.REGION_LABELS))
                for key in calib.REGION_LABELS:
                    self.assertAlmostEqual(back[key].x_ratio, regions[key].x_ratio, places=5)
                    self.assertAlmostEqual(back[key].h_ratio, regions[key].h_ratio, places=5)
            finally:
                calib.CALIBRATION_PATH = original

    def test_load_missing_file_is_empty(self) -> None:
        original = calib.CALIBRATION_PATH
        try:
            calib.CALIBRATION_PATH = Path(tempfile.gettempdir()) / "no_such_calibration_文件.json"
            self.assertEqual(calib.load_regions(), {})
        finally:
            calib.CALIBRATION_PATH = original

    def test_load_ignores_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "regions": [
                            {"key": "board", "x_ratio": 0.1, "y_ratio": 0.2, "w_ratio": 0.3, "h_ratio": 0.35},
                            {"key": "不存在的区域", "x_ratio": 0.1, "y_ratio": 0.2, "w_ratio": 0.3, "h_ratio": 0.35},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            original = calib.CALIBRATION_PATH
            try:
                calib.CALIBRATION_PATH = path
                loaded = calib.load_regions()
                self.assertEqual(set(loaded), {"board"})
            finally:
                calib.CALIBRATION_PATH = original

    def test_resolved_fills_missing_with_defaults(self) -> None:
        one = {"board": calib.Region.from_pixels("board", 100, 100, 500, 500, 1600, 900)}
        out = calib.resolved(one, 1600, 900)
        self.assertEqual(set(out), set(calib.REGION_LABELS))
        self.assertEqual(out["board"], (100, 100, 500, 500))


if __name__ == "__main__":
    unittest.main()