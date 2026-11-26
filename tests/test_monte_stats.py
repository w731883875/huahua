#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计口径回归测试（2026-11-23 用户列的记录项）。

用户要求统计脚本记录：

* 当局的**空白 / 阻挡 / 虚空**地块数量；
* **4 格、5 格、6 格普通可用地块连线**数量；
* **怪物 / 玩法 / 宝箱**初始颜料几率；
* **调色效果**几率、**笔触效果**几率。

这里锁住两件事：口径**算得出来**（不会因为字段改名/接口调整而静默归零），
以及``observe_run`` 采到的事件确实覆盖了「初始颜料 + 调色效果 + 笔触效果」。
"""

from __future__ import annotations

import unittest

from huahua_assistant import game_model as gm
from huahua_assistant import monte_stats as ms
from huahua_assistant import strategy_lv7 as st


class LineBucketTest(unittest.TestCase):
    """4/5/6 格连线要分桶，不能只给一个总数。"""

    def test_buckets_follow_line_runs(self) -> None:
        observation = ms.observe_run(3, rules=None, max_steps=60)
        counters = observation.counters

        # 分桶之和必须等于「>=4 的长连线」总数
        bucketed = (
            sum(counters['line_len4'].values())
            + sum(counters['line_len5'].values())
            + sum(counters['line_len6'].values())
        )
        self.assertEqual(
            bucketed,
            sum(counters['line_long'].values()),
            '4/5/6 分桶之和应等于长连线总数',
        )

    def test_line_buckets_match_strategy_module(self) -> None:
        """同一局的 4 格连线数，两个模块必须一致。"""
        session = __import__(
            'huahua_assistant.practice', fromlist=['start_practice']
        ).start_practice(5, pet_level=6, rules=None, wait_for_anchor=True)
        expected = sum(1 for length in st.line_runs(session.state) if length == 4)
        observation = ms.observe_run(5, rules=None)
        self.assertEqual(sum(observation.counters['line_len4'].values()), expected)


class BoardCompositionTest(unittest.TestCase):
    """地块构成：空白 + 阻挡 + 虚空 + 失色必须铺满 36 格。"""

    def test_composition_sums_to_board(self) -> None:
        observation = ms.observe_run(7, rules=None)
        counters = observation.counters
        total = (
            counters['board_blank'].get(0, 0)
            + counters['board_blocked'].get(0, 0)
            + counters['board_void'].get(0, 0)
            + counters['board_faded'].get(0, 0)
            + counters['board_paint'].get(0, 0)
        )
        self.assertEqual(total, gm.BOARD_SIZE * gm.BOARD_SIZE)


class AnchorAndEffectTest(unittest.TestCase):
    """初始颜料类别 + 调色效果 + 笔触效果都要采到。"""

    def test_anchor_category_is_recorded(self) -> None:
        observation = ms.observe_run(11, rules=None)
        counters = observation.counters['anchor_category']
        self.assertGreater(sum(counters.values()), 0, '一局里至少要有一次放初始颜料')
        for category in counters:
            self.assertIn(
                category,
                {item.value for item in gm.PaintCategory},
                f'未知的颜料类别 {category}',
            )

    def test_tint_effects_are_recorded(self) -> None:
        """调色效果 = 初始颜料自带的效果，每放一次锚点至少记一条。"""
        observation = ms.observe_run(11, rules=None)
        tint = observation.counters['tint_effect']
        anchors = sum(observation.counters['anchor_category'].values())
        self.assertGreaterEqual(
            sum(tint.values()), anchors, '每次放初始颜料至少要有一条调色效果'
        )
        for key in tint:
            parts = key.split('|')
            self.assertEqual(len(parts), 3, f'调色效果键格式应为 方向|目标|级数，实际 {key}')

    def test_brush_effects_offered_and_chosen(self) -> None:
        observation = ms.observe_run(11, rules=None)
        offered = observation.counters['brush_offered']
        chosen = observation.counters['brush_chosen']
        self.assertGreater(sum(offered.values()), 0, '应该有笔触效果被提供')
        self.assertGreater(sum(chosen.values()), 0, '应该有笔触效果被选中')
        # 选中的一定在提供过的里面
        self.assertTrue(
            set(chosen) <= set(offered),
            f'选中的笔触效果必须来自提供列表：{set(chosen) - set(offered)}',
        )


class AggregateReportTest(unittest.TestCase):
    """聚合结果要能直接出报告，且百分比归一。"""

    def test_merge_produces_reportable_numbers(self) -> None:
        observations = [ms.observe_run(seed, rules=None) for seed in (1, 2, 3)]
        aggregated = ms._merge(observations)
        self.assertEqual(aggregated['runs'], 3)
        self.assertGreater(aggregated['score']['mean'], 0)
        for key, distribution in aggregated['percent'].items():
            if not distribution:
                continue
            total = sum(distribution.values())
            self.assertAlmostEqual(total, 100.0, places=0, msg=f'{key} 百分比没有归一')

    def test_report_contains_every_requested_section(self) -> None:
        observations = [ms.observe_run(seed, rules=None) for seed in (1, 2)]
        report = ms.format_report(ms._merge(observations))
        for section in ('空白', '阻挡', '虚空', '4 格', '5 格', '6 格', '调色效果', '笔触效果'):
            self.assertIn(section, report, f'报告缺少「{section}」这一节')


if __name__ == '__main__':
    unittest.main()
