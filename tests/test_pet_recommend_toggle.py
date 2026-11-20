from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from huahua_assistant.desktop_pet import (
    PET_ACTIVE_ASSET,
    PET_IDLE_ASSET,
    RECORD_TONE_PALETTES,
    DesktopPetActor,
    _build_markers,
    _record_is_active,
    _record_pet_caption,
    _status_bubble_text,
    pet_recommendations_enabled,
)
from huahua_assistant.guidance import RecommendationTarget, TargetKind
from huahua_assistant.models import Rect


def _target() -> RecommendationTarget:
    return RecommendationTarget(
        key="board:2,3",
        kind=TargetKind.BOARD,
        priority=1,
        screen_point=(960, 540),
        confidence=0.91,
        reason_short="完成匹配",
        source_frame_id="frame-2",
        stable_since=1.0,
        expires_at=time.monotonic() + 10.0,
    )


def _scene():
    board = SimpleNamespace(tile=lambda _row, _col: SimpleNamespace(rect=Rect(0, 0, 100, 100)))
    return SimpleNamespace(board=board)


def _app(**overrides):
    base = dict(
        current_scene=_scene(),
        last_capture_window=SimpleNamespace(left=0, top=0, width=1920, height=1080),
        last_capture_quality={"ok": True},
        last_pet_state=SimpleNamespace(recommendation_target=_target()),
        recommendations=[
            SimpleNamespace(row=0, col=index, explanation=None, completes_match=False)
            for index in range(3)
        ],
        _screen_point_for_board_cell=lambda row, col: (800 + col * 100, 600 + row * 100),
        record_var=SimpleNamespace(get=lambda: "记录 录制中 · 8 步 / 12 帧"),
        record_level="recording",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _Recorder:
    def __init__(self, value: bool) -> None:
        self.value = value

    def get(self) -> bool:
        return self.value


class PetRecommendSwitchTests(unittest.TestCase):
    def test_toolbar_checkbox_wins_over_config(self) -> None:
        with patch("huahua_assistant.desktop_pet.config_value", return_value=True):
            self.assertFalse(pet_recommendations_enabled(_app(desktop_pet_recommend_var=_Recorder(False))))
            self.assertTrue(pet_recommendations_enabled(_app(desktop_pet_recommend_var=_Recorder(True))))

    def test_missing_checkbox_falls_back_to_config_default_off(self) -> None:
        # 2026-11-14 用户定调：两处都没有时默认**关**（桌宠浮层不该默认挡视线）。
        with patch("huahua_assistant.desktop_pet.config_value", return_value=False) as value:
            self.assertFalse(pet_recommendations_enabled(_app()))
        value.assert_called_once_with("pet.recommendations", False)

    def test_markers_disappear_when_switch_is_off(self) -> None:
        live = _app(desktop_pet_recommend_var=_Recorder(True))
        off = _app(desktop_pet_recommend_var=_Recorder(False))

        with patch("huahua_assistant.desktop_pet._screen_bounds_for_target", return_value=(0, 0, 1920, 1080)):
            self.assertEqual(["2", "3"], [m.label for m in _build_markers(live)])
            self.assertEqual([], _build_markers(off))

    def test_actor_drops_pointer_and_guidance_when_recommendations_off(self) -> None:
        actor = DesktopPetActor(SimpleNamespace())
        actor.manual_position = (1700, 800)
        actor.window = _FakeWindow()
        actor.label = _FakeLabel()
        actor.bubble_label = _FakeLabel(req_width=96, req_height=24)
        actor.photo = _FakePhoto(110, 190)
        actor.photo_asset_name = PET_IDLE_ASSET

        with (
            patch("huahua_assistant.desktop_pet._screen_bounds_for_target", return_value=(0, 0, 1920, 1080)),
            patch("huahua_assistant.desktop_pet._clamp_pet_position", return_value=(1700, 800)),
            patch("huahua_assistant.desktop_pet._place_window") as place_window,
            patch("huahua_assistant.desktop_pet._exclude_widget_from_capture"),
            patch.object(actor, "_show_pointer") as show_pointer,
            patch.object(actor, "_hide_pointer") as hide_pointer,
            patch.object(actor, "_apply_input_mode"),
        ):
            actor.show(
                active=False,
                pet_state=SimpleNamespace(recommendation_target=_target(), message="推荐地块"),
                recommendations=False,
                status_message="录制中 · 8 步",
                status_tone="recording",
            )

        show_pointer.assert_not_called()
        hide_pointer.assert_called_once()
        self.assertFalse(actor.guidance_active)
        # 落回手动位置，不再去找「目标上方」
        self.assertEqual((1700, 800), place_window.call_args.args[-2:])
        self.assertEqual("录制中 · 8 步", actor.bubble_label.kwargs["text"])
        self.assertEqual(RECORD_TONE_PALETTES["recording"], (actor.bubble_label.kwargs["bg"], actor.bubble_label.kwargs["fg"]))

    def test_actor_still_guides_when_recommendations_on(self) -> None:
        # 负例：同样的输入，只把开关打开，指针就必须回来。
        actor = DesktopPetActor(SimpleNamespace())
        actor.window = _FakeWindow()
        actor.label = _FakeLabel()
        actor.bubble_label = _FakeLabel(req_width=96, req_height=24)
        actor.photo = _FakePhoto(110, 190)
        actor.photo_asset_name = PET_ACTIVE_ASSET

        with (
            patch("huahua_assistant.desktop_pet._screen_bounds_for_target", return_value=(0, 0, 1920, 1080)),
            patch("huahua_assistant.desktop_pet._place_window"),
            patch("huahua_assistant.desktop_pet._exclude_widget_from_capture"),
            patch.object(actor, "_show_pointer") as show_pointer,
            patch.object(actor, "_apply_input_mode"),
        ):
            actor.show(active=True, pet_state=SimpleNamespace(recommendation_target=_target(), message="推荐地块"))

        show_pointer.assert_called_once()
        self.assertTrue(actor.guidance_active)

    def test_record_caption_strips_prefix_and_long_tail(self) -> None:
        self.assertEqual(("录制中 · 8 步", "recording"), _record_pet_caption(_app()))
        self.assertEqual(("关", "off"), _record_pet_caption(_app(record_var=SimpleNamespace(get=lambda: "记录 关"), record_level="off")))
        # 气泡空白时不能显示成「None」
        self.assertEqual("记录中", _status_bubble_text(""))
        self.assertEqual("记录中", _status_bubble_text(None))

    def test_pet_looks_busy_only_while_actually_recording(self) -> None:
        self.assertTrue(_record_is_active(_app(record_level="recording")))
        self.assertTrue(_record_is_active(_app(record_level="finished")))
        self.assertFalse(_record_is_active(_app(record_level="waiting")))
        self.assertFalse(_record_is_active(_app(record_level="off")))


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


class _FakeLabel:
    def __init__(self, *, req_width: int = 0, req_height: int = 0) -> None:
        self.req_width = req_width
        self.req_height = req_height
        self.kwargs: dict = {}

    def configure(self, **kwargs) -> None:
        self.kwargs.update(kwargs)

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
