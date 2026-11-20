"""面板「调色效果」原文 → 官方 ID：两张表必须分开查。

背景（2026-11-19 实机复盘）：

* 面板第三行的原文，有时是**地块调色效果**（「使纵向的玩法颜料提升2级」），
  有时是**笔触条目**（「调色目标的进度要求减少2」）。
* 这两句话分别属于两张官方表：
  ``S8_Gameplay_ItemEffect``（27 条，:mod:`huahua_assistant.s8_plot_effect`）与
  ``S8_Gameplay_SelectableEffect``（59 条，:mod:`huahua_assistant.s8_official_data`）。
* 两张表 **19 个 ID 撞号且语义完全不同**（2001 在调色表是「使横向的怪物颜料提升2级」，
  在笔触表是「使自身等级提升1级」），文案本身没有交集。
  旧实现只查笔触表，导致那一局 9 段调色效果 ID 全空。

所以录制层写成两个字段：``effect_id`` 查调色表，``brush_effect_id`` 查笔触表。
"""

from __future__ import annotations

import unittest

from huahua_assistant.codes import brush_effect_ids, official_effect_ids, plot_effect_ids
from huahua_assistant.s8_official_data import S8_SELECTABLE_EFFECTS
from huahua_assistant.s8_plot_effect import PLOT_EFFECTS


class PlotEffectIdTest(unittest.TestCase):
    def test_plot_text_resolves_to_plot_table(self) -> None:
        # 实机 2026-11-19 读到的原文，旧实现查笔触表结果为空。
        self.assertEqual((1108,), plot_effect_ids("使相邻的玩法颜料提升1级"))
        self.assertEqual((1101,), plot_effect_ids("使横向的怪物颜料提升1级"))
        self.assertEqual((2006,), plot_effect_ids("使纵向的玩法颜料提升2级"))
        self.assertEqual((2010,), plot_effect_ids("使横向的所有颜料提升1级"))
        self.assertEqual((4001,), plot_effect_ids("使全图的所有颜料提升2级"))

    def test_plot_lookup_never_returns_brush_rows(self) -> None:
        """笔触文案不该被调色表捞走。"""
        for text in ("调色目标的进度要求减少2", "使自身等级提升1级", "转变自身的作用方向为横向"):
            self.assertEqual((), plot_effect_ids(text), text)

    def test_brush_text_resolves_to_brush_table(self) -> None:
        self.assertEqual((1011,), official_effect_ids("调色目标的进度要求减少2"))
        self.assertEqual((1006,), brush_effect_ids("转变自身的作用对象为另一类型"))
        # 1002/4001 是官方表里的同文案重复记录，必须一起返回。
        self.assertEqual((1002, 4001), brush_effect_ids("转变自身的作用方向为横向"))

    def test_brush_lookup_never_returns_plot_rows(self) -> None:
        for text in ("使相邻的玩法颜料提升1级", "使纵向的玩法颜料提升2级"):
            self.assertEqual((), brush_effect_ids(text), text)

    def test_two_tables_have_disjoint_texts_and_overlapping_ids(self) -> None:
        """这张表是"为什么必须分两个字段"的证据本身。"""
        plot_texts = {item.effect for item in PLOT_EFFECTS}
        brush_texts = {item.text for item in S8_SELECTABLE_EFFECTS}
        self.assertEqual(set(), plot_texts & brush_texts)
        shared_ids = {item.id for item in PLOT_EFFECTS} & {item.id for item in S8_SELECTABLE_EFFECTS}
        self.assertTrue(shared_ids, "两表 ID 应当撞号，撞号正是不能混用的原因")

    def test_whitespace_is_normalized(self) -> None:
        # 面板截图 OCR 常带首尾空格/全角空格。
        self.assertEqual((2006,), plot_effect_ids(" 使纵向的玩法颜料提升2级 "))

    def test_unrelated_text_returns_empty(self) -> None:
        """匹配不到就留空，不硬猜——同构文案（只差「怪物/玩法」）之间相似度会打平。"""
        self.assertEqual((), plot_effect_ids("完全无关的一句话"))
        self.assertEqual((), brush_effect_ids("完全无关的一句话"))

    def test_empty_input(self) -> None:
        for text in (None, "", "   "):
            self.assertEqual((), plot_effect_ids(text))
            self.assertEqual((), brush_effect_ids(text))


if __name__ == "__main__":
    unittest.main()
