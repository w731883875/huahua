from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from huahua_assistant.fast_events import FastVisionEvent
from huahua_assistant.game_model import BrushType
from huahua_assistant.guidance import GamePhase, PhaseEvidence, TargetKind
from huahua_assistant.models import PaintCategory, Rect, RegionKind
from huahua_assistant import pet_brain
from huahua_assistant.pet_brain import evaluate_pet_state


def _app(**overrides):
    values = {
        "last_fast_events": (),
        "last_fast_snapshot": object(),
        "last_capture_quality": {"ok": True},
        "last_pet_state": None,
        "current_state": None,
        "current_scene": None,
        "recommendations": [],
        "install_plan": None,
        "brush_scenario": None,
        "selected_brush_type": None,
        "selected_affix": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PetBrainTargetBaselineTests(unittest.TestCase):
    def test_raw_card_highlight_does_not_fake_stage1_selection(self) -> None:
        brush = SimpleNamespace(
            chosen_stage1=BrushType.STACK.value,
            stage2_options=(),
        )
        app = _app(
            last_fast_snapshot=SimpleNamespace(
                board_signature="brush-board",
                board_score=0.9,
                reliable=True,
                brush=brush,
            ),
            current_scene=SimpleNamespace(
                brush_options=SimpleNamespace(),
                region=lambda _kind: None,
            ),
            current_state=SimpleNamespace(current_rule=object(), current_rule_pending=True),
            brush_scenario=SimpleNamespace(stage2_options=()),
            selected_brush_type=None,
        )

        evidence = pet_brain._phase_evidence(app, "raw-highlight")

        self.assertTrue(evidence.brush_visible)
        self.assertFalse(evidence.brush_type_selected)

    def test_stage1_target_uses_vertical_card_position_and_locks_first_decision(self) -> None:
        stack = SimpleNamespace(brush_type=BrushType.STACK, reason="堆砌优先")
        spread = SimpleNamespace(brush_type=BrushType.SPREAD, reason="扩散优先")
        panel = Rect(400, 300, 760, 250)
        app = _app(
            current_state=SimpleNamespace(current_rule=object(), current_rule_pending=True),
            current_scene=SimpleNamespace(
                brush_options=SimpleNamespace(panel_rect=panel, confidence=0.9),
            ),
            brush_scenario=SimpleNamespace(
                stage1_options=(BrushType.STACK, BrushType.SPREAD),
                stage2_options=(),
            ),
            _locked_brush_stage1=None,
            _screen_point_for_capture_rect=lambda rect, x_ratio, y_ratio: (
                int(rect.x + rect.w * x_ratio),
                int(rect.y + rect.h * y_ratio),
            ),
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
        )
        evidence = PhaseEvidence(
            frame_id="brush-lock",
            board_detected=True,
            board_confidence=0.9,
            quality_ok=True,
            current_rule_pending=True,
            brush_visible=True,
        )

        with patch.object(
            pet_brain,
            "recommend_brush_types",
            side_effect=([stack, spread], [spread, stack]),
        ):
            first = pet_brain._brush_target_proposal(app, evidence, 0.9)
            second = pet_brain._brush_target_proposal(app, evidence, 0.9)

        self.assertEqual("brush_stage1:stack", first.key)
        self.assertEqual(first.key, second.key)
        # 第一张卡片的中心实测在面板高度的 0.30（8 张实拍一致），
        # 不是按张数均分出来的 0.25——按均分会把点打到卡片上沿。
        self.assertEqual((818, 375), first.screen_point)
        self.assertEqual(first.screen_point, second.screen_point)

    def test_start_button_target_is_released_after_stable_frames(self) -> None:
        def snapshot():
            return SimpleNamespace(
                board_signature="start-board",
                board_score=0.82,
                reliable=True,
                brush=None,
                board_rect=Rect(590, 225, 522, 522),
            )

        start_region = SimpleNamespace(rect=Rect(1210, 760, 250, 88), confidence=0.92)
        scene = SimpleNamespace(
            brush_options=None,
            backpack=None,
            region=lambda kind: start_region if kind == RegionKind.START_DRAWING else None,
        )
        app = _app(
            last_fast_snapshot=snapshot(),
            last_capture_quality={"ok": True, "score": 0.82},
            current_scene=scene,
            current_state=SimpleNamespace(current_rule=None, current_rule_pending=False),
            last_capture_window=SimpleNamespace(left=0, top=0, width=1689, height=886),
            last_capture_size=(1689, 886),
            _screen_point_for_capture_rect=lambda rect, x_ratio, y_ratio: (
                int(rect.x + rect.w * x_ratio),
                int(rect.y + rect.h * y_ratio),
            ),
        )

        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=40.0):
            first = evaluate_pet_state(app)
        app.last_fast_snapshot = snapshot()
        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=40.31):
            second = evaluate_pet_state(app)

        self.assertEqual(GamePhase.UNSTABLE, first.phase)
        self.assertEqual(GamePhase.INITIAL_MAP, second.phase)
        self.assertIsNotNone(second.recommendation_target)
        self.assertEqual(TargetKind.START, second.recommendation_target.kind)
        self.assertEqual((1335, 804), second.recommendation_target.screen_point)

    def test_high_confidence_start_button_overrides_composited_capture_gate(self) -> None:
        start_region = SimpleNamespace(rect=Rect(1210, 760, 250, 88), confidence=0.96)
        scene = SimpleNamespace(
            board=SimpleNamespace(score=0.73),
            brush_options=None,
            backpack=None,
            region=lambda kind: start_region if kind == RegionKind.START_DRAWING else None,
        )
        app = _app(
            last_fast_snapshot=SimpleNamespace(
                board_signature="centered",
                board_score=0.73,
                reliable=False,
                brush=None,
                board_rect=Rect(590, 225, 522, 522),
            ),
            last_capture_quality={"ok": False, "score": 0.73, "reason": "capture_source_composited"},
            current_scene=scene,
            current_state=SimpleNamespace(current_rule=None, current_rule_pending=False),
            last_capture_window=SimpleNamespace(left=0, top=0, width=1689, height=886),
            last_capture_size=(1689, 886),
            _screen_point_for_capture_rect=lambda rect, x_ratio, y_ratio: (
                int(rect.x + rect.w * x_ratio),
                int(rect.y + rect.h * y_ratio),
            ),
        )

        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=50.0):
            first = evaluate_pet_state(app)
        app.last_fast_snapshot = SimpleNamespace(
            board_signature="centered",
            board_score=0.73,
            reliable=False,
            brush=None,
            board_rect=Rect(590, 225, 522, 522),
        )
        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=50.31):
            second = evaluate_pet_state(app)

        self.assertEqual(GamePhase.INITIAL_MAP, second.phase)
        self.assertIsNotNone(second.recommendation_target)
        self.assertEqual(TargetKind.START, second.recommendation_target.kind)
        self.assertNotEqual("capture_source_composited", second.event_kind)

    def test_unstable_capture_freezes_all_action_targets(self) -> None:
        app = _app(
            last_capture_quality={"ok": False, "reason": "board_score_low"},
            recommendations=[SimpleNamespace(paint=object())],
        )

        state = evaluate_pet_state(app)

        self.assertEqual("observing", state.mode)
        self.assertIsNone(state.target)
        self.assertEqual("quality_low", state.event_kind)

    def test_composited_capture_uses_specific_unstable_reason(self) -> None:
        app = _app(
            last_capture_quality={
                "ok": False,
                "reason": "capture_source_composited; board_score_low",
            }
        )

        state = evaluate_pet_state(app)

        self.assertEqual("capture_source_composited", state.event_kind)
        self.assertEqual("截图不稳", state.message)

    def test_finish_event_has_priority_over_backpack_and_board(self) -> None:
        finish = FastVisionEvent(kind="finish_candidate", confidence=0.92, detail={})
        app = _app(
            last_fast_events=(finish,),
            recommendations=[SimpleNamespace(paint=object())],
        )

        state = evaluate_pet_state(app)

        self.assertEqual("finish_hint", state.mode)
        self.assertEqual("finish", state.target)
        self.assertEqual(0.92, state.confidence)

    def test_selection_state_is_ignored_and_board_recommendation_continues(self) -> None:
        selection = FastVisionEvent(kind="selection_state_changed", confidence=0.64, detail={})
        app = _app(
            last_fast_events=(selection,),
            recommendations=[SimpleNamespace(paint=object())],
        )

        state = evaluate_pet_state(app)

        self.assertEqual("recommend_board_cell", state.mode)
        self.assertEqual("board", state.target)

    def test_paint_requirement_points_directly_to_board_target(self) -> None:
        app = _app(recommendations=[SimpleNamespace(paint=object())])

        state = evaluate_pet_state(app)

        self.assertEqual("recommend_board_cell", state.mode)
        self.assertEqual("board", state.target)

    def test_board_target_is_used_when_recommendation_has_no_backpack_paint(self) -> None:
        app = _app(recommendations=[SimpleNamespace(paint=None)])

        state = evaluate_pet_state(app)

        self.assertEqual("recommend_board_cell", state.mode)
        self.assertEqual("board", state.target)

    def test_selected_paint_points_to_first_global_plan_cell(self) -> None:
        paint = object()
        app = _app(
            install_plan=SimpleNamespace(
                steps=(SimpleNamespace(row=4, col=2, paint=paint),),
            ),
            recommendations=[
                SimpleNamespace(row=1, col=5, paint=paint, reason="单步榜首", explanation=None),
            ],
            last_capture_quality={"ok": True, "score": 0.9},
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_size=(1920, 1080),
            _screen_point_for_board_cell=lambda row, col: (1000 + col, 600 + row),
        )
        evidence = PhaseEvidence(
            frame_id="selected",
            board_detected=True,
            board_signature="board",
            board_confidence=0.9,
            quality_ok=True,
            board_target=True,
        )

        proposal = pet_brain._target_proposal(app, evidence)

        self.assertIsNotNone(proposal)
        self.assertEqual(TargetKind.BOARD, proposal.kind)
        self.assertEqual("board:4,2", proposal.key)
        self.assertEqual((1002, 604), proposal.screen_point)

    def test_selected_backpack_state_does_not_change_first_plan_cell(self) -> None:
        lv2 = SimpleNamespace(category=PaintCategory.MONSTER, level=2, is_treasure=False)
        lv1 = SimpleNamespace(category=PaintCategory.MONSTER, level=1, is_treasure=False)
        selected_lv1 = SimpleNamespace(
            category=PaintCategory.MONSTER,
            level=1,
            is_treasure=False,
            is_selected=True,
            selection_confidence=0.95,
        )
        app = _app(
            current_scene=SimpleNamespace(
                backpack=SimpleNamespace(paints=(selected_lv1,)),
                region=lambda _kind: None,
            ),
            install_plan=SimpleNamespace(
                steps=(
                    SimpleNamespace(row=4, col=3, paint=lv2),
                    SimpleNamespace(row=1, col=3, paint=lv1),
                ),
            ),
            recommendations=[],
            last_capture_quality={"ok": True, "score": 0.9},
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_size=(1920, 1080),
            _screen_point_for_board_cell=lambda row, col: (1000 + col, 600 + row),
        )
        evidence = PhaseEvidence(
            frame_id="selected-lv1",
            board_detected=True,
            board_signature="board",
            board_confidence=0.9,
            quality_ok=True,
            board_target=True,
        )

        proposal = pet_brain._target_proposal(app, evidence)

        self.assertIsNotNone(proposal)
        self.assertEqual("board:4,3", proposal.key)
        self.assertEqual((1003, 604), proposal.screen_point)

    def test_unselected_visible_paint_points_to_backpack_slot_first(self) -> None:
        def snapshot():
            return SimpleNamespace(
                board_signature="backpack-board",
                board_score=0.91,
                reliable=True,
                brush=None,
                board_rect=Rect(100, 100, 600, 600),
                terrain_counts={},
                backpack=(
                    SimpleNamespace(
                        slot_index=0,
                        category="monster",
                        level=2,
                        subtype=None,
                        is_treasure=False,
                        is_selected=False,
                        selection_confidence=0.0,
                    ),
                ),
            )

        paint = SimpleNamespace(category=PaintCategory.MONSTER, level=2, is_treasure=False)
        observed = SimpleNamespace(
            category=PaintCategory.MONSTER,
            level=2,
            subtype=None,
            is_treasure=False,
            is_selected=False,
            selection_confidence=0.0,
            slot_rect=Rect(1500, 900, 80, 80),
        )
        scene = SimpleNamespace(
            brush_options=None,
            backpack=SimpleNamespace(paints=(observed,)),
            region=lambda _kind: None,
        )
        app = _app(
            last_fast_snapshot=snapshot(),
            last_capture_quality={"ok": True, "score": 0.9},
            current_scene=scene,
            current_state=SimpleNamespace(current_rule=None, current_rule_pending=False),
            install_plan=SimpleNamespace(steps=(SimpleNamespace(row=1, col=1, paint=paint),)),
            recommendations=[],
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_size=(1920, 1080),
            _visible_backpack_index_for_paint=lambda _paint: 0,
            _screen_point_for_backpack_slot=lambda index: (1540 + index, 940),
            _screen_point_for_board_cell=lambda row, col: (1200 + col, 700 + row),
        )

        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=10.0):
            first = evaluate_pet_state(app)
        app.last_fast_snapshot = snapshot()
        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=10.31):
            second = evaluate_pet_state(app)

        self.assertEqual(GamePhase.UNSTABLE, first.phase)
        self.assertEqual("recommend_backpack", second.mode)
        self.assertEqual("backpack", second.target)
        self.assertEqual(GamePhase.BACKPACK_SELECT, second.phase)
        self.assertIsNotNone(second.recommendation_target)
        self.assertEqual(TargetKind.BACKPACK, second.recommendation_target.kind)
        self.assertEqual((1540, 940), second.recommendation_target.screen_point)

    def test_selected_desired_paint_releases_board_target(self) -> None:
        def snapshot():
            return SimpleNamespace(
                board_signature="selected-board",
                board_score=0.92,
                reliable=True,
                brush=None,
                board_rect=Rect(100, 100, 600, 600),
                terrain_counts={},
                backpack=(
                    SimpleNamespace(
                        slot_index=0,
                        category="monster",
                        level=2,
                        subtype=None,
                        is_treasure=False,
                        is_selected=True,
                        selection_confidence=0.93,
                    ),
                ),
            )

        paint = SimpleNamespace(category=PaintCategory.MONSTER, level=2, is_treasure=False)
        observed = SimpleNamespace(
            category=PaintCategory.MONSTER,
            level=2,
            subtype=None,
            is_treasure=False,
            is_selected=True,
            selection_confidence=0.93,
            slot_rect=Rect(1500, 900, 80, 80),
        )
        app = _app(
            last_fast_snapshot=snapshot(),
            last_capture_quality={"ok": True, "score": 0.92},
            current_scene=SimpleNamespace(
                brush_options=None,
                backpack=SimpleNamespace(paints=(observed,)),
                region=lambda _kind: None,
            ),
            current_state=SimpleNamespace(current_rule=None, current_rule_pending=False),
            install_plan=SimpleNamespace(steps=(SimpleNamespace(row=4, col=3, paint=paint),)),
            recommendations=[],
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_size=(1920, 1080),
            _visible_backpack_index_for_paint=lambda _paint: 0,
            _screen_point_for_backpack_slot=lambda index: (1540 + index, 940),
            _screen_point_for_board_cell=lambda row, col: (1000 + col, 600 + row),
        )

        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=12.0):
            evaluate_pet_state(app)
        app.last_fast_snapshot = snapshot()
        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=12.31):
            state = evaluate_pet_state(app)

        self.assertEqual("recommend_board_cell", state.mode)
        self.assertEqual(GamePhase.BOARD_PLACE, state.phase)
        self.assertIsNotNone(state.recommendation_target)
        self.assertEqual(TargetKind.BOARD, state.recommendation_target.kind)
        self.assertEqual((1003, 604), state.recommendation_target.screen_point)

    def test_hidden_desired_paint_without_selection_has_no_false_target(self) -> None:
        lv2 = SimpleNamespace(category=PaintCategory.MONSTER, level=2, is_treasure=False)
        lv1_observed = SimpleNamespace(
            category=PaintCategory.MONSTER,
            level=1,
            subtype=None,
            is_treasure=False,
            is_selected=False,
            selection_confidence=0.0,
            slot_rect=Rect(1500, 900, 80, 80),
        )
        app = _app(
            last_fast_snapshot=SimpleNamespace(
                board_signature="hidden-paint",
                board_score=0.9,
                reliable=True,
                brush=None,
                board_rect=Rect(100, 100, 600, 600),
                terrain_counts={},
                backpack=(),
            ),
            last_capture_quality={"ok": True, "score": 0.9},
            current_scene=SimpleNamespace(
                brush_options=None,
                backpack=SimpleNamespace(paints=(lv1_observed,)),
                region=lambda _kind: None,
            ),
            current_state=SimpleNamespace(current_rule=object(), current_rule_pending=False),
            install_plan=SimpleNamespace(steps=(SimpleNamespace(row=1, col=1, paint=lv2),)),
            recommendations=[],
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_size=(1920, 1080),
            _visible_backpack_index_for_paint=lambda _paint: None,
            _screen_point_for_board_cell=lambda row, col: (1000 + col, 600 + row),
        )

        evidence = pet_brain._phase_evidence(app, "hidden")

        self.assertFalse(evidence.backpack_target)
        self.assertFalse(evidence.board_target)
        self.assertIsNone(pet_brain._target_proposal(app, evidence))

    def test_stale_board_target_outside_observed_gold_scope_is_rejected(self) -> None:
        app = _app(
            current_state=SimpleNamespace(
                current_rule=object(),
                current_rule_pending=False,
                observed_scope_cells=((1, 2), (2, 2), (3, 2)),
            ),
            recommendations=[
                SimpleNamespace(row=5, col=5, paint=None, reason="stale", explanation=None),
            ],
            last_capture_quality={"ok": True, "score": 0.9},
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_size=(1920, 1080),
            _screen_point_for_board_cell=lambda row, col: (1000 + col, 600 + row),
        )
        evidence = PhaseEvidence(
            frame_id="stale-scope",
            board_detected=True,
            board_signature="board",
            board_confidence=0.9,
            quality_ok=True,
            board_target=True,
        )

        self.assertIsNone(pet_brain._target_proposal(app, evidence))

    def test_structured_board_target_is_released_after_stable_capture_frames(self) -> None:
        def snapshot():
            return SimpleNamespace(
                board_signature="board-a",
                board_score=0.91,
                reliable=True,
                brush=None,
                board_rect=Rect(100, 100, 600, 600),
            )

        scene = SimpleNamespace(
            brush_options=None,
            backpack=None,
            region=lambda _kind: None,
        )
        recommendation = SimpleNamespace(
            row=2,
            col=3,
            paint=None,
            reason="宝箱到7",
            explanation=None,
        )
        app = _app(
            last_fast_snapshot=snapshot(),
            last_capture_quality={"ok": True, "score": 0.91},
            current_scene=scene,
            current_state=SimpleNamespace(current_rule=None, current_rule_pending=False),
            recommendations=[recommendation],
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_size=(1920, 1080),
            _screen_point_for_board_cell=lambda row, col: (1200 + col, 700 + row),
        )

        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=10.0):
            first = evaluate_pet_state(app)
        app.last_fast_snapshot = snapshot()
        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=10.31):
            second = evaluate_pet_state(app)

        self.assertEqual(GamePhase.UNSTABLE, first.phase)
        self.assertIsNone(first.recommendation_target)
        self.assertEqual(GamePhase.BOARD_PLACE, second.phase)
        self.assertIsNotNone(second.recommendation_target)
        self.assertEqual(TargetKind.BOARD, second.recommendation_target.kind)
        self.assertEqual((1203, 702), second.recommendation_target.screen_point)

    def test_finish_evidence_persists_across_frames_for_stable_target(self) -> None:
        def snapshot():
            return SimpleNamespace(
                board_signature="terminal-board",
                board_score=0.94,
                reliable=True,
                brush=None,
                board_rect=Rect(100, 100, 600, 600),
                terrain_counts={"blank": 0, "paint": 36},
            )

        finish_region = SimpleNamespace(rect=Rect(1400, 900, 180, 70), confidence=0.93)
        scene = SimpleNamespace(
            brush_options=None,
            backpack=None,
            region=lambda kind: finish_region if kind == RegionKind.FINISH_DRAWING else None,
        )
        app = _app(
            last_fast_snapshot=snapshot(),
            last_capture_quality={"ok": True, "score": 0.94},
            current_scene=scene,
            current_state=SimpleNamespace(current_rule=None, current_rule_pending=False),
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_size=(1920, 1080),
            _screen_point_for_capture_rect=lambda rect, x_ratio, y_ratio: (
                int(rect.x + rect.w * x_ratio),
                int(rect.y + rect.h * y_ratio),
            ),
        )

        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=20.0):
            first = evaluate_pet_state(app)
        app.last_fast_snapshot = snapshot()
        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=20.31):
            second = evaluate_pet_state(app)

        self.assertEqual(GamePhase.UNSTABLE, first.phase)
        self.assertEqual(GamePhase.FINISH_READY, second.phase)
        self.assertIsNotNone(second.recommendation_target)
        self.assertEqual(TargetKind.FINISH, second.recommendation_target.kind)
        self.assertEqual((1490, 935), second.recommendation_target.screen_point)

    def test_incomplete_terrain_counts_do_not_confirm_finish(self) -> None:
        def snapshot():
            return SimpleNamespace(
                board_signature="partial-board",
                board_score=0.94,
                reliable=True,
                brush=None,
                board_rect=Rect(100, 100, 600, 600),
                terrain_counts={"blank": 0, "paint": 4},
            )

        finish_region = SimpleNamespace(rect=Rect(1400, 900, 180, 70), confidence=0.93)
        no_blank = FastVisionEvent(
            kind="finish_candidate",
            confidence=0.7,
            detail={"reason": "no_blank_tiles", "terrain_counts": {"blank": 0, "paint": 4}},
        )
        app = _app(
            last_fast_snapshot=snapshot(),
            last_fast_events=(no_blank,),
            last_capture_quality={"ok": True, "score": 0.94},
            current_scene=SimpleNamespace(
                brush_options=None,
                backpack=None,
                region=lambda kind: finish_region if kind == RegionKind.FINISH_DRAWING else None,
            ),
            current_state=SimpleNamespace(current_rule=None, current_rule_pending=False),
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_size=(1920, 1080),
            _screen_point_for_capture_rect=lambda rect, x_ratio, y_ratio: (
                int(rect.x + rect.w * x_ratio),
                int(rect.y + rect.h * y_ratio),
            ),
        )

        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=25.0):
            evaluate_pet_state(app)
        app.last_fast_snapshot = snapshot()
        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=25.31):
            state = evaluate_pet_state(app)

        self.assertEqual(GamePhase.WAITING, state.phase)
        self.assertIsNone(state.recommendation_target)

    def test_offscreen_or_outside_window_target_is_rejected(self) -> None:
        def snapshot():
            return SimpleNamespace(
                board_signature="board-a",
                board_score=0.9,
                reliable=True,
                brush=None,
                board_rect=Rect(100, 100, 600, 600),
            )

        app = _app(
            last_fast_snapshot=snapshot(),
            last_capture_quality={"ok": True, "score": 0.9},
            current_scene=SimpleNamespace(
                brush_options=None,
                backpack=None,
                region=lambda _kind: None,
            ),
            current_state=SimpleNamespace(current_rule=None, current_rule_pending=False),
            recommendations=[SimpleNamespace(row=2, col=3, paint=None, reason="推荐", explanation=None)],
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_size=(1920, 1080),
            _screen_point_for_board_cell=lambda _row, _col: (-20, 700),
            winfo_vrootx=lambda: 0,
            winfo_vrooty=lambda: 0,
            winfo_vrootwidth=lambda: 1920,
            winfo_vrootheight=lambda: 1080,
        )

        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=30.0):
            evaluate_pet_state(app)
        app.last_fast_snapshot = snapshot()
        with patch("huahua_assistant.pet_brain.time.monotonic", return_value=30.31):
            state = evaluate_pet_state(app)

        self.assertEqual(GamePhase.BOARD_PLACE, state.phase)
        self.assertIsNone(state.recommendation_target)

    def test_windows_monitor_check_takes_precedence_over_tk_virtual_root(self) -> None:
        secondary_monitor_app = SimpleNamespace(
            last_capture_window=SimpleNamespace(left=-1920, top=0, width=1920, height=1080),
            winfo_exists=lambda: True,
            winfo_vrootx=lambda: 0,
            winfo_vrooty=lambda: 0,
            winfo_vrootwidth=lambda: 1920,
            winfo_vrootheight=lambda: 1080,
        )
        with patch.object(pet_brain, "_windows_point_is_on_monitor", return_value=True):
            self.assertTrue(pet_brain._target_point_is_safe(secondary_monitor_app, (-100, 700)))

        primary_monitor_app = SimpleNamespace(
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            winfo_exists=lambda: True,
        )
        with patch.object(pet_brain, "_windows_point_is_on_monitor", return_value=False):
            self.assertFalse(pet_brain._target_point_is_safe(primary_monitor_app, (1200, 700)))

    def test_non_finite_target_and_malformed_counts_are_rejected(self) -> None:
        app = SimpleNamespace(last_capture_window=None)
        malformed_window_app = SimpleNamespace(
            last_capture_window=SimpleNamespace(left=0, top=0, width=None, height=1080)
        )

        self.assertFalse(pet_brain._target_point_is_safe(app, (float("nan"), 700)))
        self.assertFalse(pet_brain._target_point_is_safe(app, (1200, float("inf"))))
        self.assertFalse(pet_brain._target_point_is_safe(malformed_window_app, (1200, 700)))
        self.assertFalse(pet_brain._terrain_counts_show_terminal_map({"blank": "0", "paint": 36}))
        self.assertFalse(
            pet_brain._terrain_counts_show_terminal_map({"blank": float("nan"), "paint": 36})
        )


if __name__ == "__main__":
    unittest.main()
