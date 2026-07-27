from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from huahua_assistant.fast_events import FastVisionEvent
from huahua_assistant.guidance import GamePhase, TargetKind
from huahua_assistant.models import Rect
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

    def test_selection_state_waits_for_board_drop_before_new_recommendation(self) -> None:
        selection = FastVisionEvent(kind="selection_state_changed", confidence=0.64, detail={})
        app = _app(
            last_fast_events=(selection,),
            recommendations=[SimpleNamespace(paint=object())],
        )

        state = evaluate_pet_state(app)

        self.assertEqual("wait_for_drop", state.mode)
        self.assertEqual("board", state.target)

    def test_backpack_target_precedes_board_target_when_paint_is_required(self) -> None:
        app = _app(recommendations=[SimpleNamespace(paint=object())])

        state = evaluate_pet_state(app)

        self.assertEqual("recommend_backpack", state.mode)
        self.assertEqual("backpack", state.target)

    def test_board_target_is_used_when_recommendation_has_no_backpack_paint(self) -> None:
        app = _app(recommendations=[SimpleNamespace(paint=None)])

        state = evaluate_pet_state(app)

        self.assertEqual("recommend_board_cell", state.mode)
        self.assertEqual("board", state.target)

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
            region=lambda _kind: finish_region,
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
                region=lambda _kind: finish_region,
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
