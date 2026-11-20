"""实机提示链路的「恢复能力」测试。

这一轮把「识别 → 阶段 → 提示」整条链路接到真实截图上时，暴露出来的都不是
识别精度问题，而是**卡死**问题：任何一次误判、任何一帧读不到面板，提示就会
永久消失，只能重启程序。这些用例锁的就是「能自己爬起来」。
"""
from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from huahua_assistant import pet_brain
from huahua_assistant.app import _capture_quality_gate
from huahua_assistant.game_model import (
    PuzzleState,
    current_rule_is_pending,
    state_from_observation,
)
from huahua_assistant.guidance import (
    GamePhase,
    GamePhaseStateMachine,
    PhaseEvidence,
)
from huahua_assistant.models import (
    BoardObservation,
    BrushOptionsObservation,
    CurrentPaintObservation,
    PaintCategory,
    PaintDirection,
    PaintObject,
    Rect,
    RegionKind,
    RegionObservation,
    SceneObservation,
    TileObservation,
    TileState,
)


def _evidence(frame_id: str, **overrides) -> PhaseEvidence:
    values = {
        "frame_id": frame_id,
        "board_detected": True,
        "board_signature": "board-a",
        "board_confidence": 0.9,
        "quality_ok": True,
    }
    values.update(overrides)
    return PhaseEvidence(**values)


def _tile(index: int, state: TileState = TileState.BLANK, **overrides) -> TileObservation:
    values = {
        "row": index // 6,
        "col": index % 6,
        "rect": Rect((index % 6) * 10, (index // 6) * 10, 10, 10),
        "state": state,
        "confidence": 0.95,
        "features": {},
    }
    values.update(overrides)
    return TileObservation(**values)


def _board(overrides: dict[int, TileObservation] | None = None) -> BoardObservation:
    overrides = overrides or {}
    return BoardObservation(
        rect=Rect(0, 0, 60, 60),
        score=0.9,
        tiles=tuple(overrides.get(index, _tile(index)) for index in range(36)),
    )


def _current_paint(
    *,
    category: PaintCategory = PaintCategory.MONSTER,
    level: int = 3,
    direction: PaintDirection = PaintDirection.VERTICAL,
    target_object: PaintObject = PaintObject.ALL,
    match_required: int = 3,
    match_current: int = 0,
) -> CurrentPaintObservation:
    return CurrentPaintObservation(
        panel_rect=Rect(0, 0, 100, 30),
        icon_rect=Rect(0, 0, 20, 20),
        text_rect=Rect(20, 0, 80, 20),
        category=category,
        category_confidence=0.9,
        level=level,
        level_confidence=0.9,
        subtype="邪影",
        is_treasure=False,
        direction=direction,
        target_object=target_object,
        upgrade_amount=1,
        match_current=match_current,
        match_required=match_required,
        keyword_scores={},
        raw_text_hint="",
    )


def _unreadable_paint() -> CurrentPaintObservation:
    return _current_paint(
        category=PaintCategory.UNKNOWN,
        direction=PaintDirection.UNKNOWN,
        target_object=PaintObject.UNKNOWN,
    )


class PhaseRecoveryTests(unittest.TestCase):
    """阶段状态机不能把自己锁死。"""

    def _to_stage2(self, machine: GamePhaseStateMachine) -> None:
        for index in range(3):
            machine.update(
                _evidence(
                    f"b{index}",
                    board_signature="brush",
                    brush_visible=True,
                    current_rule_pending=True,
                )
            )
            machine.update(
                _evidence(
                    f"s{index}",
                    board_signature="brush",
                    brush_visible=True,
                    brush_stage2_visible=True,
                    current_rule_pending=True,
                )
            )
        state = machine.update(
            _evidence(
                "s-final",
                board_signature="brush",
                brush_visible=True,
                brush_stage2_visible=True,
                current_rule_pending=True,
            )
        )
        self.assertEqual(GamePhase.BRUSH_STAGE2, state.phase)

    def test_stage2_can_return_to_initial_map(self) -> None:
        """选完词缀回到对局，是最常见的一条路，必须算合法转移。"""
        machine = GamePhaseStateMachine()
        self._to_stage2(machine)

        state = machine.update(_evidence("m1", current_rule_present=True))
        state = machine.update(_evidence("m2", current_rule_present=True))

        self.assertEqual(GamePhase.INITIAL_MAP, state.phase)
        self.assertTrue(state.stable)

    def test_illegal_transition_is_accepted_after_tolerance_frames(self) -> None:
        """一直看到同一个「非法」阶段，说明现实确实变了，必须接受，不能永久卡死。"""
        machine = GamePhaseStateMachine(illegal_tolerance=3)
        self._to_stage2(machine)

        states = [
            machine.update(_evidence(f"w{index}", board_detected=False))
            for index in range(3)
        ]

        self.assertEqual(GamePhase.UNSTABLE, states[0].phase)
        self.assertEqual(GamePhase.UNSTABLE, states[1].phase)
        self.assertEqual(GamePhase.WAITING, states[2].phase)
        self.assertTrue(states[2].stable)

    def test_illegal_transition_is_not_accepted_on_the_first_frame(self) -> None:
        """容忍不等于放弃防抖：第一帧仍然报不稳。"""
        machine = GamePhaseStateMachine(illegal_tolerance=3)
        self._to_stage2(machine)

        first = machine.update(_evidence("w", board_detected=False))

        self.assertEqual(GamePhase.UNSTABLE, first.phase)
        self.assertEqual(GamePhase.WAITING, first.observed_phase)
        self.assertIn("illegal_transition", first.reason)


class RuleCarryOverTests(unittest.TestCase):
    """面板读不到的那几帧，不能把规则清空。"""

    def _state_with_rule(self) -> PuzzleState:
        board = _board(
            {
                8: _tile(
                    8,
                    TileState.PAINT,
                    paint_category=PaintCategory.MONSTER,
                    paint_level=3,
                    is_current_anchor=True,
                    current_anchor_confidence=0.9,
                )
            }
        )
        return state_from_observation(board, _current_paint(), ())

    def test_unreadable_panel_keeps_previous_rule(self) -> None:
        previous = self._state_with_rule()
        self.assertIsNotNone(previous.current_rule)

        state = state_from_observation(_board(), _unreadable_paint(), (), previous=previous)

        self.assertIsNotNone(state.current_rule)
        self.assertEqual(PaintDirection.VERTICAL, state.current_rule.direction)
        self.assertEqual(PaintObject.ALL, state.current_rule.target_object)

    def test_anchor_and_scope_carry_over_with_the_rule(self) -> None:
        previous = self._state_with_rule()
        self.assertEqual((1, 2), previous.current_anchor)

        state = state_from_observation(_board(), _unreadable_paint(), (), previous=previous)

        self.assertEqual((1, 2), state.current_anchor)
        self.assertEqual(previous.observed_scope_cells, state.observed_scope_cells)

    def test_readable_panel_replaces_previous_rule(self) -> None:
        previous = self._state_with_rule()
        new_paint = _current_paint(direction=PaintDirection.HORIZONTAL, target_object=PaintObject.MONSTER)

        state = state_from_observation(_board(), new_paint, (), previous=previous)

        self.assertEqual(PaintDirection.HORIZONTAL, state.current_rule.direction)
        self.assertEqual(PaintObject.MONSTER, state.current_rule.target_object)

    def test_without_previous_an_unreadable_panel_still_yields_no_rule(self) -> None:
        """不传 previous 时行为不变：读不到就是读不到。"""
        state = state_from_observation(_board(), _unreadable_paint(), ())
        self.assertIsNotNone(state.current_rule)
        self.assertEqual(PaintCategory.UNKNOWN, state.current_rule.paint.category)


class PendingRuleTests(unittest.TestCase):
    """选完笔锋、还没落笔的那一段，要判成「待放初始颜料」。"""

    def test_rule_without_anchor_is_pending(self) -> None:
        state = state_from_observation(_board(), _current_paint(), ())
        self.assertTrue(current_rule_is_pending(state))

    def test_anchor_clears_pending(self) -> None:
        board = _board(
            {
                8: _tile(
                    8,
                    TileState.PAINT,
                    paint_category=PaintCategory.MONSTER,
                    paint_level=3,
                    is_current_anchor=True,
                    current_anchor_confidence=0.9,
                )
            }
        )
        state = state_from_observation(board, _current_paint(), ())
        self.assertEqual((1, 2), state.current_anchor)
        self.assertFalse(current_rule_is_pending(state))

    def test_pending_state_still_recommends_a_placement(self) -> None:
        from huahua_assistant.game_model import recommend_placements

        state = state_from_observation(_board(), _current_paint(), ())
        self.assertTrue(recommend_placements(state, limit=5))


class QualityGateTests(unittest.TestCase):
    """笔锋面板会挡住背包，这是游戏设定，不该扣分。"""

    def _brush_scene(self) -> SceneObservation:
        return SceneObservation(
            board=_board(),
            regions=(),
            current_paint=_current_paint(),
            brush_options=BrushOptionsObservation(
                panel_rect=Rect(600, 300, 700, 250),
                stage1_options=("transform", "stack"),
                stage1_scores={},
                chosen_stage1=None,
                stage2_options=(),
                stage2_scores={},
                keyword_scores={},
                confidence=0.95,
            ),
        )

    def test_brush_panel_hides_backpack_without_penalty(self) -> None:
        quality = _capture_quality_gate(self._brush_scene(), ())
        self.assertNotIn("backpack=", quality["reason"])
        self.assertTrue(quality["ok"])

    def test_missing_backpack_still_penalized_without_brush_panel(self) -> None:
        scene = replace(self._brush_scene(), brush_options=None)
        quality = _capture_quality_gate(scene, ())
        self.assertFalse(quality["ok"])
        self.assertIn("backpack=", quality["reason"])


class FinishStateTests(unittest.TestCase):
    """结束态格子状态难分，但「完成绘制」按钮是硬信号。

    注意别拿右下角那颗「入画」按钮当证据：它在作画界面常驻（实机每帧检出，
    置信 0.92），只是「可以进下一张画纸」。以前把它算成 ``finish_button``，
    新一局的第一帧就被判可结算并自动收尾（见 tests/test_new_run_detection.py）。
    """

    def _app(self, label: str = "finish drawing button") -> SimpleNamespace:
        scene = SceneObservation(
            board=_board(
                {
                    index: _tile(index, TileState.UNKNOWN)
                    for index in (4, 5, 11, 23, 29, 35)
                }
            ),
            regions=(
                RegionObservation(
                    kind=RegionKind.FINISH_DRAWING,
                    rect=Rect(1084, 665, 264, 49),
                    confidence=0.99,
                    label=label,
                ),
            ),
        )
        return SimpleNamespace(
            last_fast_snapshot=SimpleNamespace(board_signature="b", board_score=0.75, reliable=True),
            current_scene=scene,
            current_state=None,
            last_fast_events=(),
            last_capture_quality={"ok": False, "score": 0.66, "reason": "unknown_tiles=6"},
            last_capture_window=None,
        )

    def test_finish_region_survives_unknown_tiles(self) -> None:
        app = self._app()
        self.assertIsNone(pet_brain._unstable_capture_state(app))
        evidence = pet_brain._phase_evidence(app, "f1")
        self.assertTrue(evidence.quality_ok)
        self.assertIn("finish_button", evidence.finish_signals)

    def test_enter_painting_button_is_not_finish_evidence(self) -> None:
        app = self._app(label="enter painting button fast")
        evidence = pet_brain._phase_evidence(app, "f1")
        self.assertNotIn("finish_button", evidence.finish_signals)


if __name__ == "__main__":
    unittest.main()
