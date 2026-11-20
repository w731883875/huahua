from __future__ import annotations

import unittest

from huahua_assistant.guidance import (
    GamePhase,
    GamePhaseStateMachine,
    GuidanceController,
    PhaseEvidence,
    RecommendationTargetStabilizer,
    TargetKind,
    TargetProposal,
)


def _evidence(frame_id: str, **overrides) -> PhaseEvidence:
    values = {
        "frame_id": frame_id,
        "board_detected": True,
        "board_signature": "board-a",
        "board_confidence": 0.9,
        "quality_ok": True,
        "board_target": True,
    }
    values.update(overrides)
    return PhaseEvidence(**values)


def _proposal(frame_id: str, **overrides) -> TargetProposal:
    values = {
        "key": "board:2,3",
        "kind": TargetKind.BOARD,
        "priority": 40,
        "screen_point": (1250, 730),
        "confidence": 0.87,
        "reason_short": "宝箱到7",
        "source_frame_id": frame_id,
    }
    values.update(overrides)
    return TargetProposal(**values)


class GamePhaseStateMachineTests(unittest.TestCase):
    def test_brush_phase_stabilizes_even_when_board_signature_changes(self) -> None:
        machine = GamePhaseStateMachine()

        first = machine.update(
            _evidence(
                "f1",
                board_signature="noisy-a",
                board_target=False,
                current_rule_pending=True,
                brush_visible=True,
            )
        )
        second = machine.update(
            _evidence(
                "f2",
                board_signature="noisy-b",
                board_target=False,
                current_rule_pending=True,
                brush_visible=True,
            )
        )

        self.assertEqual(GamePhase.BRUSH_STAGE1, first.observed_phase)
        self.assertEqual(GamePhase.BRUSH_STAGE1, second.phase)
        self.assertTrue(second.stable)

    def test_start_button_stabilizes_as_initial_map(self) -> None:
        machine = GamePhaseStateMachine()

        pending = machine.update(
            _evidence("f1", board_target=False, start_visible=True)
        )
        stable = machine.update(
            _evidence("f2", board_target=False, start_visible=True)
        )

        self.assertEqual(GamePhase.INITIAL_MAP, pending.observed_phase)
        self.assertEqual(GamePhase.INITIAL_MAP, stable.phase)
        self.assertTrue(stable.stable)

    def test_phase_and_key_fields_require_two_distinct_frames(self) -> None:
        machine = GamePhaseStateMachine()

        first = machine.update(_evidence("f1"))
        repeated = machine.update(_evidence("f1"))
        second = machine.update(_evidence("f2"))

        self.assertEqual(GamePhase.UNSTABLE, first.phase)
        self.assertEqual("phase_transition_pending", first.reason)
        self.assertEqual(GamePhase.UNSTABLE, repeated.phase)
        self.assertEqual(GamePhase.BOARD_PLACE, second.phase)
        self.assertTrue(second.stable)

    def test_low_confidence_freezes_without_overwriting_last_trusted_phase(self) -> None:
        machine = GamePhaseStateMachine()
        machine.update(_evidence("f1"))
        machine.update(_evidence("f2"))

        unstable = machine.update(
            _evidence("f3", board_confidence=0.31, board_signature="noise")
        )
        recovered = machine.update(_evidence("f4"))

        self.assertEqual(GamePhase.UNSTABLE, unstable.phase)
        self.assertEqual(GamePhase.BOARD_PLACE, unstable.last_trusted_phase)
        self.assertEqual(GamePhase.BOARD_PLACE, recovered.phase)
        self.assertTrue(recovered.stable)

    def test_illegal_phase_jump_is_rejected(self) -> None:
        machine = GamePhaseStateMachine()
        machine.update(_evidence("f1"))
        machine.update(_evidence("f2"))

        jumped = machine.update(
            _evidence(
                "f3",
                board_target=False,
                current_rule_pending=True,
                brush_visible=True,
                brush_stage2_visible=True,
                brush_type_selected=True,
            )
        )

        self.assertEqual(GamePhase.UNSTABLE, jumped.phase)
        self.assertIn("illegal_transition", jumped.reason)
        self.assertEqual(GamePhase.BOARD_PLACE, jumped.last_trusted_phase)

    def test_finish_ready_requires_two_distinct_evidence_categories(self) -> None:
        one_signal = GamePhaseStateMachine()
        first = one_signal.update(
            _evidence("f1", board_target=False, finish_signals=frozenset({"finish_button"}))
        )

        two_signals = GamePhaseStateMachine()
        pending = two_signals.update(
            _evidence(
                "f1",
                board_target=False,
                finish_signals=frozenset({"finish_button", "terminal_map"}),
            )
        )
        ready = two_signals.update(
            _evidence(
                "f2",
                board_target=False,
                finish_signals=frozenset({"finish_button", "terminal_map"}),
            )
        )

        self.assertNotEqual(GamePhase.FINISH_READY, first.observed_phase)
        self.assertEqual(GamePhase.FINISH_READY, pending.observed_phase)
        self.assertEqual(GamePhase.FINISH_READY, ready.phase)


class RecommendationTargetStabilizerTests(unittest.TestCase):
    def test_start_target_is_accepted_on_initial_map(self) -> None:
        stabilizer = RecommendationTargetStabilizer()
        context = ("window-a",)
        first = stabilizer.update(
            _proposal("f1", key="start_drawing", kind=TargetKind.START),
            phase=GamePhase.INITIAL_MAP,
            context_signature=context,
            now=1.0,
        )
        second = stabilizer.update(
            _proposal("f2", key="start_drawing", kind=TargetKind.START),
            phase=GamePhase.INITIAL_MAP,
            context_signature=context,
            now=1.31,
        )

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(TargetKind.START, second.kind)

    def test_target_requires_two_frames_and_300ms_then_serializes_contract(self) -> None:
        stabilizer = RecommendationTargetStabilizer()
        context = (100, 100, 1920, 1080, 1.0)

        first = stabilizer.update(
            _proposal("f1"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=context,
            now=10.0,
        )
        second = stabilizer.update(
            _proposal("f2"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=context,
            now=10.31,
        )

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual("board", second.to_dict()["kind"])
        self.assertEqual([1250, 730], second.to_dict()["screen_point"])
        self.assertEqual(10.0, second.stable_since)
        self.assertGreater(second.expires_at, 10.31)

    def test_same_target_ignores_coordinate_jitter_and_refreshes_expiry(self) -> None:
        stabilizer = RecommendationTargetStabilizer()
        context = ("window-a",)
        stabilizer.update(
            _proposal("f1"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=context,
            now=1.0,
        )
        stable = stabilizer.update(
            _proposal("f2"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=context,
            now=1.31,
        )
        refreshed = stabilizer.update(
            _proposal("f3", screen_point=(1254, 728), confidence=0.89),
            phase=GamePhase.BOARD_PLACE,
            context_signature=context,
            now=1.6,
        )

        self.assertEqual((1250, 730), stable.screen_point)
        self.assertEqual((1250, 730), refreshed.screen_point)
        self.assertGreater(refreshed.expires_at, stable.expires_at)

    def test_repeated_ui_render_of_same_frame_does_not_extend_ttl(self) -> None:
        stabilizer = RecommendationTargetStabilizer()
        context = ("window-a",)
        stabilizer.update(
            _proposal("f1"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=context,
            now=1.0,
        )
        stable = stabilizer.update(
            _proposal("f2"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=context,
            now=1.31,
        )
        repeated = stabilizer.update(
            _proposal("f2"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=context,
            now=2.5,
        )
        expired = stabilizer.update(
            _proposal("f2"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=context,
            now=stable.expires_at + 0.01,
        )

        self.assertEqual(stable.expires_at, repeated.expires_at)
        self.assertIsNone(expired)

    def test_static_brush_target_refreshes_until_player_selects_it(self) -> None:
        stabilizer = RecommendationTargetStabilizer()
        context = ("window-a",)
        proposal = lambda frame: _proposal(
            frame,
            key="brush_stage1:stack",
            kind=TargetKind.BRUSH_STAGE1,
        )
        stabilizer.update(
            proposal("f1"),
            phase=GamePhase.BRUSH_STAGE1,
            context_signature=context,
            now=1.0,
        )
        stable = stabilizer.update(
            proposal("f2"),
            phase=GamePhase.BRUSH_STAGE1,
            context_signature=context,
            now=1.31,
        )
        refreshed = stabilizer.update(
            proposal("f2"),
            phase=GamePhase.BRUSH_STAGE1,
            context_signature=context,
            now=2.5,
        )

        self.assertIsNotNone(stable)
        self.assertGreater(refreshed.expires_at, stable.expires_at)

    def test_low_confidence_context_change_and_expiry_hide_target(self) -> None:
        stabilizer = RecommendationTargetStabilizer()
        old_context = ("window-a",)
        stabilizer.update(
            _proposal("f1"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=old_context,
            now=1.0,
        )
        stable = stabilizer.update(
            _proposal("f2"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=old_context,
            now=1.31,
        )

        low = stabilizer.update(
            _proposal("f3", confidence=0.4),
            phase=GamePhase.BOARD_PLACE,
            context_signature=old_context,
            now=1.5,
        )
        changed = stabilizer.update(
            _proposal("f4"),
            phase=GamePhase.BOARD_PLACE,
            context_signature=("window-b",),
            now=1.6,
        )

        self.assertIsNotNone(stable)
        self.assertIsNone(low)
        self.assertIsNone(changed)
        self.assertIsNone(
            stabilizer.active(context_signature=old_context, now=stable.expires_at + 0.01)
        )

    def test_controller_releases_target_when_phase_and_target_stabilize_together(self) -> None:
        controller = GuidanceController()
        context = ("window-a",)

        first = controller.update(
            _evidence("f1"),
            _proposal("f1"),
            context_signature=context,
            now=5.0,
        )
        second = controller.update(
            _evidence("f2"),
            _proposal("f2"),
            context_signature=context,
            now=5.31,
        )

        self.assertEqual(GamePhase.UNSTABLE, first.phase.phase)
        self.assertIsNone(first.target)
        self.assertEqual(GamePhase.BOARD_PLACE, second.phase.phase)
        self.assertIsNotNone(second.target)


if __name__ == "__main__":
    unittest.main()
