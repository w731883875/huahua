from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from huahua_assistant.desktop_pet import (
    PET_ACTIVE_ASSET,
    DesktopPetActor,
    _active_recommendation_target,
    _build_markers,
)
from huahua_assistant.guidance import RecommendationTarget, TargetKind
from huahua_assistant.models import Rect
from huahua_assistant.pet_layout import choose_pet_guidance_placement, pointer_geometry


def _target(
    *,
    kind: TargetKind = TargetKind.BOARD,
    point: tuple[int, int] = (960, 540),
    expires_at: float | None = None,
) -> RecommendationTarget:
    return RecommendationTarget(
        key="board:2,3",
        kind=kind,
        priority=1,
        screen_point=point,
        confidence=0.91,
        reason_short="完成匹配",
        source_frame_id="frame-2",
        stable_since=1.0,
        expires_at=time.monotonic() + 10.0 if expires_at is None else expires_at,
    )


class PetLayoutTests(unittest.TestCase):
    def test_prefers_above_and_never_covers_target(self) -> None:
        placement = choose_pet_guidance_placement(
            (960, 540),
            (120, 220),
            (0, 0, 1920, 1080),
        )

        self.assertIsNotNone(placement)
        self.assertEqual("above", placement.side)
        self.assertFalse(placement.covers((960, 540)))
        self.assertEqual(30, 540 - placement.bottom)

    def test_falls_back_right_left_then_below_near_edges(self) -> None:
        right = choose_pet_guidance_placement((960, 120), (120, 220), (0, 0, 1920, 1080))
        left = choose_pet_guidance_placement((1850, 120), (120, 220), (0, 0, 1920, 1080))
        below = choose_pet_guidance_placement((130, 100), (120, 220), (0, 0, 260, 1080))

        self.assertEqual("right", right.side)
        self.assertEqual("left", left.side)
        self.assertEqual("below", below.side)

    def test_supports_negative_monitor_coordinates_and_rejects_impossible_layout(self) -> None:
        placement = choose_pet_guidance_placement(
            (-960, 540),
            (120, 220),
            (-1920, 0, 1920, 1080),
        )
        impossible = choose_pet_guidance_placement((40, 40), (120, 220), (0, 0, 80, 80))

        self.assertIsNotNone(placement)
        self.assertEqual("above", placement.side)
        self.assertFalse(placement.covers((-960, 540)))
        self.assertIsNone(impossible)

    def test_pointer_tip_is_exactly_at_target_center_for_every_side(self) -> None:
        for side in ("above", "right", "left", "below"):
            with self.subTest(side=side):
                line, head = pointer_geometry(side, 76)
                self.assertEqual((38, 38), line[2:])
                self.assertEqual((38, 38), head[:2])


class DesktopPetGuidanceTests(unittest.TestCase):
    def test_target_expiry_is_checked_at_render_time(self) -> None:
        live = SimpleNamespace(recommendation_target=_target(expires_at=20.0))
        expired = SimpleNamespace(recommendation_target=_target(expires_at=19.99))

        self.assertIsNotNone(_active_recommendation_target(live, now=19.99))
        self.assertIsNone(_active_recommendation_target(expired, now=20.0))

    def test_actor_uses_guidance_position_over_manual_position_without_repeated_move(self) -> None:
        actor = DesktopPetActor(SimpleNamespace())
        actor.manual_position = (1700, 800)
        actor.window = _FakeWindow()
        actor.label = _FakeLabel()
        actor.bubble_label = _FakeLabel(req_width=96, req_height=24)
        actor.photo = _FakePhoto(110, 190)
        actor.photo_asset_name = PET_ACTIVE_ASSET
        pet_state = SimpleNamespace(recommendation_target=_target(), message="推荐地块")

        with (
            patch("huahua_assistant.desktop_pet._screen_bounds_for_target", return_value=(0, 0, 1920, 1080)),
            patch("huahua_assistant.desktop_pet._place_window") as place_window,
            patch.object(actor, "_show_pointer") as show_pointer,
            patch.object(actor, "_apply_input_mode"),
        ):
            actor.show(active=True, pet_state=pet_state)
            actor.show(active=True, pet_state=pet_state)

        self.assertTrue(actor.guidance_active)
        self.assertEqual(1, place_window.call_count)
        _, width, height, left, top = place_window.call_args.args
        self.assertNotEqual(actor.manual_position, (left, top))
        self.assertLessEqual(left, 960)
        self.assertLess(top + height, 540)
        self.assertEqual(2, show_pointer.call_count)

    def test_guidance_is_clickthrough_by_default_and_explicit_drag_is_interactive(self) -> None:
        actor = DesktopPetActor(SimpleNamespace())
        actor.window = _FakeWindow()
        actor.guidance_active = True

        with (
            patch("huahua_assistant.desktop_pet._make_desktop_pet_window_clickthrough") as clickthrough,
            patch("huahua_assistant.desktop_pet._make_desktop_pet_window_interactive") as interactive,
        ):
            actor._apply_input_mode()
            actor.draggable = True
            actor._apply_input_mode()

        clickthrough.assert_called_once()
        interactive.assert_called_once()

    def test_explicit_drag_temporarily_uses_manual_position_and_hides_guidance(self) -> None:
        actor = DesktopPetActor(SimpleNamespace())
        actor.draggable = True
        actor.manual_position = (1700, 800)
        actor.window = _FakeWindow()
        actor.label = _FakeLabel()
        actor.bubble_label = _FakeLabel(req_width=96, req_height=24)
        actor.photo = _FakePhoto(110, 190)
        actor.photo_asset_name = PET_ACTIVE_ASSET
        pet_state = SimpleNamespace(recommendation_target=_target(), message="推荐地块")

        with (
            patch("huahua_assistant.desktop_pet._clamp_pet_position", return_value=(1700, 800)),
            patch("huahua_assistant.desktop_pet._place_window") as place_window,
            patch.object(actor, "_hide_pointer") as hide_pointer,
            patch.object(actor, "_apply_input_mode"),
        ):
            actor.show(active=True, pet_state=pet_state)

        self.assertFalse(actor.guidance_active)
        self.assertEqual((1700, 800), place_window.call_args.args[-2:])
        hide_pointer.assert_called_once()

    def test_same_pointer_target_is_not_redrawn_or_repositioned(self) -> None:
        actor = DesktopPetActor(SimpleNamespace())
        actor.pointer_window = _FakeWindow()
        actor.pointer_canvas = _FakeCanvas()
        target = _target()
        placement = choose_pet_guidance_placement(
            target.screen_point,
            (110, 218),
            (0, 0, 1920, 1080),
        )

        with (
            patch("huahua_assistant.desktop_pet._place_window") as place_window,
            patch("huahua_assistant.desktop_pet._make_clickthrough_widgets"),
        ):
            actor._show_pointer(target, placement)
            actor._show_pointer(target, placement)

        self.assertEqual(1, place_window.call_count)
        self.assertEqual(1, actor.pointer_canvas.delete_count)

    def test_capture_hide_withdraws_both_pet_and_pointer(self) -> None:
        actor = DesktopPetActor(SimpleNamespace())
        actor.window = _FakeWindow()
        actor.window.visible = True
        actor.pointer_window = _FakeWindow()
        actor.pointer_window.visible = True
        actor.last_render_key = ("visible",)

        hidden = actor.hide()

        self.assertEqual(2, hidden)
        self.assertFalse(actor.window.visible)
        self.assertFalse(actor.pointer_window.visible)
        self.assertIsNone(actor.last_render_key)

    def test_board_overlay_contains_only_top2_and_top3_for_stable_board_target(self) -> None:
        recommendations = [
            SimpleNamespace(row=0, col=index, explanation=None, completes_match=False)
            for index in range(3)
        ]
        board = SimpleNamespace(tile=lambda _row, _col: SimpleNamespace(rect=Rect(0, 0, 100, 100)))
        app = SimpleNamespace(
            current_scene=SimpleNamespace(board=board),
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_quality={"ok": True},
            last_pet_state=SimpleNamespace(recommendation_target=_target()),
            recommendations=recommendations,
            _screen_point_for_board_cell=lambda row, col: (800 + col * 100, 600 + row * 100),
        )

        with patch("huahua_assistant.desktop_pet._screen_bounds_for_target", return_value=(0, 0, 1920, 1080)):
            markers = _build_markers(app)

        self.assertEqual(["2", "3"], [marker.label for marker in markers])
        self.assertEqual(["board-2", "board-3"], [marker.key for marker in markers])
        self.assertNotIn((800, 600), [(marker.x, marker.y) for marker in markers])

        app._screen_point_for_board_cell = lambda row, col: (
            (-20, 600) if col == 1 else (800 + col * 100, 600 + row * 100)
        )
        with patch("huahua_assistant.desktop_pet._screen_bounds_for_target", return_value=(0, 0, 1920, 1080)):
            safe_markers = _build_markers(app)
        self.assertEqual(["3"], [marker.label for marker in safe_markers])

    def test_overlay_freezes_without_live_target_or_for_non_board_phase(self) -> None:
        app = SimpleNamespace(
            current_scene=SimpleNamespace(board=SimpleNamespace()),
            last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
            last_capture_quality={"ok": True},
            recommendations=[],
            last_pet_state=SimpleNamespace(recommendation_target=None),
        )
        self.assertEqual([], _build_markers(app))

        app.last_pet_state = SimpleNamespace(
            recommendation_target=_target(kind=TargetKind.BACKPACK),
        )
        self.assertEqual([], _build_markers(app))


class _FakeWindow:
    def __init__(self) -> None:
        self.visible = False

    def winfo_exists(self) -> bool:
        return True

    def deiconify(self) -> None:
        self.visible = True

    def lift(self) -> None:
        return None

    def attributes(self, *_args) -> None:
        return None

    def state(self) -> str:
        return "normal" if self.visible else "withdrawn"

    def withdraw(self) -> None:
        self.visible = False


class _FakeCanvas:
    def __init__(self) -> None:
        self.delete_count = 0

    def winfo_exists(self) -> bool:
        return True

    def delete(self, *_args) -> None:
        self.delete_count += 1

    def create_line(self, *_args, **_kwargs) -> None:
        return None

    def create_polygon(self, *_args, **_kwargs) -> None:
        return None

    def create_oval(self, *_args, **_kwargs) -> None:
        return None


class _FakeLabel:
    def __init__(self, *, req_width: int = 0, req_height: int = 0) -> None:
        self.req_width = req_width
        self.req_height = req_height

    def configure(self, **_kwargs) -> None:
        return None

    def update_idletasks(self) -> None:
        return None

    def winfo_reqwidth(self) -> int:
        return self.req_width

    def winfo_reqheight(self) -> int:
        return self.req_height


class _FakePhoto:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


if __name__ == "__main__":
    unittest.main()
