from __future__ import annotations

import unittest
from dataclasses import replace

from huahua_assistant.fast_events import detect_fast_events
from huahua_assistant.fast_vision import FastBackpackPaint, FastVisionSnapshot
from huahua_assistant.models import Rect, TileState


def _paint(*, fingerprint: str, selected: bool = False) -> FastBackpackPaint:
    return FastBackpackPaint(
        slot_index=0,
        rect=Rect(10, 10, 40, 40),
        category="monster",
        level=2,
        subtype=None,
        is_treasure=False,
        fingerprint=fingerprint,
        confidence=0.9,
        is_selected=selected,
        selection_confidence=0.9 if selected else 0.0,
    )


def _snapshot(paint: FastBackpackPaint, *, signature: str) -> FastVisionSnapshot:
    return FastVisionSnapshot(
        board_rect=Rect(0, 0, 600, 600),
        board_score=0.82,
        tiles=(),
        board_signature="board",
        paint_signature="paint",
        terrain_counts={TileState.BLANK.value: 4},
        backpack_rect=Rect(650, 0, 300, 500),
        backpack=(paint,),
        backpack_signature=signature,
        current=None,
        brush=None,
        regions={},
        reliable=True,
        reasons=(),
    )


class FastEventTests(unittest.TestCase):
    def test_backpack_fingerprint_jitter_is_only_visual_change(self) -> None:
        previous = _snapshot(_paint(fingerprint="frame-a"), signature="signature-a")
        current = _snapshot(_paint(fingerprint="frame-b"), signature="signature-b")

        events = detect_fast_events(previous, current)

        self.assertEqual(["backpack_visual_changed"], [event.kind for event in events])
        self.assertEqual((), events[0].detail["selected_slots"])
        self.assertEqual((), events[0].detail["previous_selected_slots"])

    def test_selected_slot_change_emits_selection_state_changed(self) -> None:
        previous = _snapshot(_paint(fingerprint="frame-a"), signature="signature-a")
        current_paint = replace(_paint(fingerprint="frame-b"), is_selected=True)
        current = _snapshot(current_paint, signature="signature-b")

        events = detect_fast_events(previous, current)

        self.assertEqual(["selection_state_changed"], [event.kind for event in events])
        self.assertEqual((0,), events[0].detail["selected_slots"])
        self.assertEqual((), events[0].detail["previous_selected_slots"])


if __name__ == "__main__":
    unittest.main()
