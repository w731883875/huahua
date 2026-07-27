from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from huahua_assistant.fast_events import FastVisionEvent
from huahua_assistant.guidance import GamePhase, TargetKind
from huahua_assistant.models import Rect
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


if __name__ == "__main__":
    unittest.main()
