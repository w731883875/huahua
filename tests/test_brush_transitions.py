from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from huahua_assistant.app import HuahuaApp, _state_for_visible_brush_panel
from huahua_assistant.game_model import (
    BrushType,
    CurrentPaintRule,
    Paint,
    PuzzleState,
    Tile,
    build_affix_catalog,
    recommend_placements,
)
from huahua_assistant.models import (
    BrushOptionsObservation,
    PaintCategory,
    PaintDirection,
    PaintObject,
    Rect,
    TileState,
)


def _observation(
    *,
    stage1: tuple[str, ...],
    chosen: str | None = None,
    stage2: tuple[str, ...] = (),
) -> BrushOptionsObservation:
    return BrushOptionsObservation(
        panel_rect=Rect(400, 300, 760, 250),
        stage1_options=stage1,
        stage1_scores={name: 0.9 for name in stage1},
        chosen_stage1=chosen,
        stage2_options=stage2,
        stage2_scores={name: 0.9 for name in stage2},
        keyword_scores={},
        confidence=0.9,
    )


class _BrushHarness:
    _apply_detected_brush_options = HuahuaApp._apply_detected_brush_options
    _merge_quick_brush_stage2 = HuahuaApp._merge_quick_brush_stage2


def _pending_state() -> PuzzleState:
    rule = CurrentPaintRule(
        paint=Paint(PaintCategory.GAMEPLAY, 3),
        direction=PaintDirection.VERTICAL,
        target_object=PaintObject.MONSTER,
        match_required=3,
        upgrade_amount=1,
    )
    return PuzzleState(
        tiles=tuple(Tile(TileState.BLANK) for _ in range(36)),
        current_rule=rule,
        backpack=tuple(Paint(PaintCategory.MONSTER, level) for level in (4, 3, 2)),
        current_rule_pending=True,
    )


class BrushTransitionTests(unittest.TestCase):
    def test_visible_stage1_panel_does_not_erase_a_placed_anchor(self) -> None:
        """笔触面板开着 ≠ 初始颜料没落（2026-11-14 实机纠正）。

        实测流程是**先落初始颜料、再选笔触**：伊吉已经落在 r5c6、面板进度 2/4，
        笔触面板这时仍然开着。旧实现在这里强制 ``current_anchor=None`` +
        ``current_rule_pending=True``，推荐就变成「先选一个空白地块放入初始颜料」，
        玩家会被推着往调色范围外的空白格落锚点。锚点只认棋盘证据。
        """
        state = replace(
            _pending_state(),
            current_rule_pending=False,
            current_anchor=(2, 2),
        )
        observation = _observation(
            stage1=(BrushType.STACK.value, BrushType.SPREAD.value),
        )

        updated = _state_for_visible_brush_panel(state, observation)

        self.assertFalse(updated.current_rule_pending)
        self.assertEqual((2, 2), updated.current_anchor)

    def test_stage1_guidance_survives_transient_panel_misses(self) -> None:
        app = _BrushHarness()
        app.current_state = _pending_state()
        app.brush_scenario = None
        app.selected_brush_type = None
        app.selected_affix = None
        app.selected_backpack_index = None
        app.brush_stage1_rects = {}
        app.brush_affix_rects = {}
        app._brush_panel_absent_frames = 0
        app._last_brush_observation = None
        app._brush_stage1_vote_observations = []
        app.status_var = SimpleNamespace(set=lambda _value: None)
        observation = _observation(stage1=(BrushType.STACK.value, BrushType.SPREAD.value))
        app._merge_quick_brush_stage2(observation)
        app._merge_quick_brush_stage2(observation)

        for _ in range(11):
            app._merge_quick_brush_stage2(None)

        self.assertIsNotNone(app.brush_scenario)
        self.assertIsNotNone(app._last_brush_observation)
        app._merge_quick_brush_stage2(None)
        self.assertIsNone(app.brush_scenario)

    def test_stage1_choice_latches_after_two_votes_and_ignores_conflicts(self) -> None:
        app = _BrushHarness()
        app.current_state = _pending_state()
        app.brush_scenario = None
        app.selected_brush_type = None
        app.selected_affix = None
        app.selected_backpack_index = None
        app.brush_stage1_rects = {}
        app.brush_affix_rects = {}
        app._brush_panel_absent_frames = 0
        app._last_brush_observation = None
        app._brush_stage1_vote_observations = []
        app.status_var = SimpleNamespace(set=lambda _value: None)
        conflicting = _observation(
            stage1=(BrushType.TRANSFORM.value, BrushType.STACK.value),
        )
        stable = _observation(
            stage1=(BrushType.STACK.value, BrushType.SPREAD.value),
        )

        app._merge_quick_brush_stage2(conflicting)
        self.assertIsNone(app.brush_scenario)
        app._merge_quick_brush_stage2(stable)
        app._merge_quick_brush_stage2(stable)
        self.assertEqual(
            (BrushType.STACK, BrushType.SPREAD),
            app.brush_scenario.stage1_options,
        )

        app._merge_quick_brush_stage2(conflicting)
        self.assertEqual(
            (BrushType.STACK, BrushType.SPREAD),
            app.brush_scenario.stage1_options,
        )

    def test_quick_frames_cover_stage1_stage2_and_release_anchor_guidance(self) -> None:
        app = _BrushHarness()
        app.current_state = _pending_state()
        app.brush_scenario = None
        app.selected_brush_type = None
        app.selected_affix = None
        app.selected_backpack_index = None
        app.brush_stage1_rects = {}
        app.brush_affix_rects = {}
        app._brush_panel_absent_frames = 0
        app._last_brush_observation = None
        app._brush_stage1_vote_observations = []
        app.status_var = SimpleNamespace(set=lambda _value: None)

        observation = _observation(stage1=(BrushType.STACK.value, BrushType.SPREAD.value))
        app._merge_quick_brush_stage2(observation)
        app._merge_quick_brush_stage2(observation)

        self.assertEqual(
            (BrushType.STACK, BrushType.SPREAD),
            app.brush_scenario.stage1_options,
        )
        self.assertIsNone(app.selected_brush_type)

        stack_affixes = tuple(
            affix.name
            for affix in build_affix_catalog()
            if affix.brush_type == BrushType.STACK
        )[:2]
        app._merge_quick_brush_stage2(
            _observation(
                stage1=(BrushType.STACK.value,),
                chosen=BrushType.STACK.value,
                stage2=stack_affixes,
            )
        )

        self.assertEqual(BrushType.STACK, app.selected_brush_type)
        self.assertEqual(2, len(app.brush_scenario.stage2_options))

        app._merge_quick_brush_stage2(None)
        self.assertIsNotNone(app.brush_scenario)
        # 记录模式（默认开）下不猜词缀，见 recording_mode_enabled 的注释；
        # 这里模拟「非记录模式」，验证旧的猜测路径还在。
        with patch("huahua_assistant.app.recording_mode_enabled", return_value=False):
            app._merge_quick_brush_stage2(None)

        self.assertIsNone(app.brush_scenario)
        self.assertIsNotNone(app.selected_affix)
        self.assertTrue(recommend_placements(app.current_state))

    def test_recording_mode_does_not_guess_the_affix(self) -> None:
        """记录模式绝不能靠「推荐排序第一」去猜玩家选了哪条词缀。

        用户 2026-11-22 明确要求：「现在是记录模式，不是推荐模式，任何推荐的都需要
        关掉」。猜错比留空更糟——留空时下游知道「这条没坐实」，猜错就成了假数据。
        """

        app = _BrushHarness()
        app.current_state = _pending_state()
        app.brush_scenario = None
        app.selected_brush_type = None
        app.selected_affix = None
        app.selected_backpack_index = None
        app.brush_stage1_rects = {}
        app.brush_affix_rects = {}
        app._brush_panel_absent_frames = 0
        app._last_brush_observation = None
        app._brush_stage1_vote_observations = []
        app.status_var = SimpleNamespace(set=lambda _value: None)

        stack_affixes = tuple(
            affix.name
            for affix in build_affix_catalog()
            if affix.brush_type == BrushType.STACK
        )[:2]
        app._merge_quick_brush_stage2(
            _observation(
                stage1=(BrushType.STACK.value,),
                chosen=BrushType.STACK.value,
                stage2=stack_affixes,
            )
        )
        app._merge_quick_brush_stage2(None)
        app._merge_quick_brush_stage2(None)

        self.assertIsNone(app.selected_affix, "记录模式不得凭空猜一条词缀")


if __name__ == "__main__":
    unittest.main()
