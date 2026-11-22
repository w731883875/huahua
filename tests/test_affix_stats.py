"""锁住 ``affix_stats`` 的统计口径。

核心是「出现 ≠ 选中」：一次抽卡给两张卡，玩家挑一张，另一张也算出现过。
只要有人把口径改回读 ``selected``，或者把窄化后的单张也算成一次抽卡，
这里的测试就会红。
"""

from __future__ import annotations

import unittest

from huahua_assistant.affix_stats import (
    BRUSH_NAMES,
    DRAW_SIZE,
    build_report,
    iter_offers,
)


def _row(event: str, payload: dict, session: str = "s1", ts: float = 100.0) -> dict:
    return {"event": event, "payload": payload, "session": session, "ts": ts}


class OfferExtractionTest(unittest.TestCase):
    def test_narrowed_single_card_is_not_a_draw(self) -> None:
        """窄化后只剩一张的观测不是新的一次抽卡。

        实机 2026-11-22 22:48 ``run-20261122-144715`` 实测同一面板被观测两次：
            +0.0s  ['transform', 'stack']   ← 面板出现
            +3.7s  ['stack']                ← 选中后窄化
        把窄化那条也算进去，被选中的 ``stack`` 就会平白多算一次。
        """

        rows = [
            _row("brush_stage1_options_observed", {"options": ["transform", "stack"]}, ts=100.0),
            _row("brush_stage1_options_observed", {"options": ["stack"]}, ts=103.7),
        ]
        offers = list(iter_offers(rows))
        self.assertEqual(1, len(offers))
        self.assertEqual(("transform", "stack"), offers[0].cards)

    def test_selected_field_is_ignored(self) -> None:
        """统计只看 ``options``（出现过什么），不看 ``selected``（选了什么）。"""

        rows = [
            _row(
                "affix_choice_observed",
                {
                    "options": ["自身提升1级", "全图和自身相同的颜料提升1级"],
                    "selected": "自身提升1级",
                    "stage1_options": ["stack"],
                },
            )
        ]
        report = build_report(rows)
        self.assertEqual(1, report.stage2.counts["自身提升1级"])
        self.assertEqual(1, report.stage2.counts["全图和自身相同的颜料提升1级"], "没被选中的那张也要算")

    def test_stage2_is_grouped_by_the_stage1_brush(self) -> None:
        """词缀池由阶段一笔触决定，所以必须按笔触分组统计。"""

        rows = [
            _row(
                "affix_choice_observed",
                {"options": ["自身提升1级", "自身提升2级"], "stage1_options": ["stack"]},
            ),
            _row(
                "affix_choice_observed",
                {"options": ["作用方向改为横向", "作用方向改为纵向"], "stage1_options": ["transform"]},
                ts=110.0,
            ),
        ]
        report = build_report(rows)
        self.assertEqual({"stack", "transform"}, set(report.stage2_by_brush))
        self.assertEqual(1, report.stage2_by_brush["stack"].draws)
        self.assertEqual(1, report.stage2_by_brush["transform"].draws)
        self.assertNotIn("作用方向改为横向", report.stage2_by_brush["stack"].counts)

    def test_brush_names_cover_all_three(self) -> None:
        self.assertEqual({"transform", "stack", "spread"}, set(BRUSH_NAMES))

    def test_draw_size_matches_the_observed_screen(self) -> None:
        """每屏两张，是实测出来的：阶段一 103 次观测里 2 张的那批，词缀 49/50。"""

        self.assertEqual(2, DRAW_SIZE)


class AccumulationTest(unittest.TestCase):
    def test_identical_consecutive_screens_are_deduped(self) -> None:
        """同一屏停留多帧会连记好几次，只能算一次抽取。"""

        rows = [
            _row("brush_stage1_options_observed", {"options": ["transform", "stack"]}, ts=1.0),
            _row("brush_stage1_options_observed", {"options": ["transform", "stack"]}, ts=1.2),
            _row("brush_stage1_options_observed", {"options": ["transform", "stack"]}, ts=1.4),
        ]
        report = build_report(rows)
        self.assertEqual(1, report.stage1.draws)

    def test_same_cards_after_a_different_screen_count_again(self) -> None:
        """中间隔了别的屏，同样的两张卡就是新的一次抽取，不能误判成重复。"""

        rows = [
            _row("brush_stage1_options_observed", {"options": ["transform", "stack"]}, ts=1.0),
            _row("brush_stage1_options_observed", {"options": ["stack", "spread"]}, ts=5.0),
            _row("brush_stage1_options_observed", {"options": ["transform", "stack"]}, ts=9.0),
        ]
        report = build_report(rows)
        self.assertEqual(3, report.stage1.draws)

    def test_no_cap_on_sample_size(self) -> None:
        """样本随时间累积，没有 500 次之类的上限。"""

        rows = [
            _row("brush_stage1_options_observed", {"options": ["transform", "stack"]}, ts=float(i))
            for i in range(1200)
        ]
        report = build_report(rows)
        # 相邻且完全相同会被去重，所以这里交替两个屏来模拟真实抽卡节奏。
        rows = []
        for i in range(1200):
            options = ["transform", "stack"] if i % 2 == 0 else ["stack", "spread"]
            rows.append(_row("brush_stage1_options_observed", {"options": options}, ts=float(i)))
        report = build_report(rows)
        self.assertEqual(1200, report.stage1.draws, "1200 次抽取必须全部计入，不能被截断")


class RateTest(unittest.TestCase):
    def test_rate_uses_draws_not_card_count(self) -> None:
        """出现率的分母是「抽了多少次」，不是「看到多少张卡」。

        一屏给两张、两张不会同名，所以某项的出现率上限是 1，不会出现 200%。
        """

        rows = [
            _row("brush_stage1_options_observed", {"options": ["transform", "stack"]}, ts=1.0),
            _row("brush_stage1_options_observed", {"options": ["transform", "spread"]}, ts=5.0),
        ]
        report = build_report(rows)
        self.assertEqual(2, report.stage1.draws)
        self.assertEqual(2, report.stage1.counts["transform"])
        self.assertAlmostEqual(1.0, report.stage1.rate("transform"))

    def test_wilson_interval_shrinks_with_more_samples(self) -> None:
        few = build_report(
            [_row("brush_stage1_options_observed", {"options": ["transform", "stack"]}, ts=1.0)]
        ).stage1.wilson("transform")
        many = build_report(
            [
                _row(
                    "brush_stage1_options_observed",
                    {"options": ["transform", "stack"] if i % 2 else ["transform", "spread"]},
                    ts=float(i),
                )
                for i in range(200)
            ]
        ).stage1.wilson("transform")
        self.assertLess(many[1] - many[0], few[1] - few[0], "样本变多，区间应变窄")

    def test_empty_report_does_not_crash(self) -> None:
        report = build_report([])
        self.assertEqual(0, report.stage1.draws)
        self.assertEqual(0.0, report.stage1.rate("transform"))
        self.assertEqual((0.0, 0.0), report.stage1.wilson("transform"))

    def test_unresolved_choices_are_ignored(self) -> None:
        """未定案的条目不影响「出现」统计——它本来就只是没读出选中项。

        用户 2026-11-22：「对未定案那种可以忽略」。因为统计读的是 ``options``，
        而 ``affix_choice_unresolved`` 不带 ``options``，天然不进统计。
        """

        rows = [
            _row("affix_choice_observed", {"options": ["自身提升1级", "自身提升2级"], "stage1_options": ["stack"]}),
            _row("affix_choice_unresolved", {"selected": None, "confidence": 0.35}, ts=120.0),
        ]
        report = build_report(rows)
        self.assertEqual(1, report.stage2.draws)


if __name__ == "__main__":
    unittest.main()
