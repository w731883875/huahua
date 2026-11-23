#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""融合专线 + 阶级阈值的回归测试（2026-11-23）。

背景：5 万局 greedy 模拟里**融合颜料 0 次**。原因是融合要三个条件同时成立：

1. 用掉一次强制调色，且随机炸毁**恰好炸到当前锚点格**（炸到才留效果包）；
2. 下一轮抽到「可放置于失色地块中」（官方 3017）—— 失色格默认放不了锚点；
3. 在这一格放新锚点，才把地块上的效果包并进新规则。

默认策略连强制完成都不主动选，概率趋近 0，于是新规则（融合无上限、
逐条结算、虚空只丢内容）**一局都没验证到**。这里锁住两件事：

* ``fusion_explore`` 打开后确实能产生融合（effects > 1）；
* 阶级阈值按绝对分标定，不再出现「全盘皆 S」。

同时锁住 ``_kept_effect_placeable`` 这一融合入口的判定：
**有残留效果包 + 词缀允许落上去**才为真。
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from huahua_assistant import game_model as gm
from huahua_assistant import practice as pr

BOARD = gm.BOARD_SIZE


def _faded_affix() -> gm.AffixRule:
    return next(
        affix
        for affix in gm.build_affix_catalog()
        if affix.effect is gm.AffixEffect.ALLOW_PLACE_ON_FADED
    )


def _blank_state() -> gm.PuzzleState:
    return gm.PuzzleState(
        tiles=tuple(
            gm.Tile(gm.TileState.BLANK) for _ in range(BOARD * BOARD)
        )
    )


class KeptEffectPlaceableTest(unittest.TestCase):
    """``_kept_effect_placeable`` 是融合的唯一入口，判定不能松也不能紧。"""

    def _state_with_kept_effect(self, *, affix: gm.AffixRule | None) -> gm.PuzzleState:
        tiles = list(_blank_state().tiles)
        tiles[2 * BOARD + 1] = gm.Tile(
            gm.TileState.FADED,
            retained_effects=(
                gm.PaintEffect(gm.PaintDirection.ADJACENT, gm.PaintObject.MONSTER, 1),
            ),
        )
        return gm.PuzzleState(tiles=tuple(tiles), active_affix=affix)

    def test_needs_both_kept_effect_and_permitting_affix(self) -> None:
        # 有效果包、没词缀 → 失色格放不了，不能算可融合
        self.assertFalse(gm._kept_effect_placeable(self._state_with_kept_effect(affix=None)))
        # 有效果包 + 「可放置于失色地块中」 → 可以
        self.assertTrue(
            gm._kept_effect_placeable(self._state_with_kept_effect(affix=_faded_affix()))
        )

    def test_no_kept_effect_means_no_fusion(self) -> None:
        state = gm.PuzzleState(
            tiles=tuple(gm.Tile(gm.TileState.BLANK) for _ in range(BOARD * BOARD)),
            active_affix=_faded_affix(),
        )
        self.assertFalse(gm._kept_effect_placeable(state), "没有效果包就谈不上融合")

    def test_affix_must_match_the_faded_state(self) -> None:
        """「可放置于阻挡地块」不解锁失色格 —— 词缀和目标地块状态必须对上。"""
        blocked_affix = next(
            affix
            for affix in gm.build_affix_catalog()
            if affix.effect is gm.AffixEffect.ALLOW_PLACE_ON_BLOCKED
        )
        state = self._state_with_kept_effect(affix=blocked_affix)
        self.assertFalse(gm._kept_effect_placeable(state))


class FusionExploreTest(unittest.TestCase):
    """融合专线必须真的产出融合，否则新规则永远测不到。"""

    def test_fusion_explore_produces_merged_paints(self) -> None:
        """跑一批种子，至少要出现一次 effects > 1 的融合颜料。

        实测融合率约 3%（炸中锚点 ~51% × 抽到 3017 ~3.6% × 还有机会刷新），
        60 局里 0 次的概率约 16%，所以取 200 局留足余量。
        """
        merged = 0
        for seed in range(200):
            session = pr.start_practice(
                seed, pet_level=6, rules=None, max_steps=60, wait_for_anchor=True
            )
            pr.auto_play(session, prefer="greedy", fusion_explore=True)
            rule = session.state.current_rule
            if rule is not None and len(rule.effects) > 1:
                merged += 1
        self.assertGreater(merged, 0, "融合专线一局都没融合，说明链路又断了")

    def test_without_fusion_explore_never_merges(self) -> None:
        """默认策略从不主动强制完成，所以正常情况下不该出现融合。"""
        for seed in range(60):
            session = pr.start_practice(
                seed, pet_level=6, rules=None, max_steps=60, wait_for_anchor=True
            )
            pr.auto_play(session, prefer="greedy")
            rule = session.state.current_rule
            if rule is not None:
                self.assertEqual(
                    len(rule.effects),
                    1,
                    f"seed={seed} 没开融合专线却出现了融合，判定条件过松",
                )


class GradeThresholdTest(unittest.TestCase):
    """阶级阈值按 ``board_value`` 绝对分标定，不能全盘皆 S。"""

    def test_thresholds_are_monotonic(self) -> None:
        scores = [threshold for threshold, _ in pr.GRADE_THRESHOLDS]
        self.assertEqual(scores, sorted(scores, reverse=True), "阈值必须从高到低")

    def test_grades_are_spread_out(self) -> None:
        """典型的模拟分数要落在不同档位，而不是清一色 S。

        取样值从阈值表本身推导，避免以后重标阈值时测试又和实现脱节
        （这正是当初 ``_grade`` 阈值失效却没人发现的原因）。
        """
        samples = [0.0]
        for threshold, _ in pr.GRADE_THRESHOLDS:
            samples.append(float(threshold))
            samples.append(float(threshold) + 1.0)
        seen = {pr._grade(value) for value in samples}
        self.assertEqual(
            len(seen),
            len(pr.GRADE_THRESHOLDS) + 1,
            f"阶级档位没有全部用上：{sorted(seen)}（阈值 {pr.GRADE_THRESHOLDS}）",
        )

    def test_low_score_is_not_top_grade(self) -> None:
        self.assertEqual(pr._grade(0.0), "D 低分")
        self.assertEqual(pr._grade(10.0), "D 低分")
        self.assertEqual(pr._grade(1000.0), "S 顶级")


if __name__ == "__main__":
    unittest.main()
