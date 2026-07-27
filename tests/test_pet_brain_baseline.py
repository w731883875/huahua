from __future__ import annotations

import unittest
from types import SimpleNamespace

from huahua_assistant.fast_events import FastVisionEvent
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


if __name__ == "__main__":
    unittest.main()
