"""遮挡判断：什么时候该因为「工具窗口盖住游戏」暂停自动跟随。

这里守的是一个真实踩过的坑：一旦因为遮挡停掉采集线程，就再也没有人去建立
窗口内容抓取的会话；而会话建不起来，遮挡判断又会一直判「被挡住」。自己把
自己锁死，表现就是自动跟随停在覆盖提示上怎么移都不动。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from huahua_assistant import app as app_module
from huahua_assistant.app import HuahuaApp


def _window(captureable: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        handle=1234,
        hwnd=1234,
        left=0,
        top=0,
        right=1920,
        bottom=1020,
        captureable=captureable,
    )


def _app(*, app_rect: tuple[int, int, int, int], probed: bool) -> SimpleNamespace:
    return SimpleNamespace(
        auto_follow_capture_probed=probed,
        winfo_viewable=lambda: True,
        update_idletasks=lambda: None,
        winfo_rootx=lambda: app_rect[0],
        winfo_rooty=lambda: app_rect[1],
        winfo_width=lambda: app_rect[2] - app_rect[0],
        winfo_height=lambda: app_rect[3] - app_rect[1],
    )


def _gate(
    *,
    app_rect: tuple[int, int, int, int],
    probed: bool,
    ready: bool,
    available: bool,
    captureable: bool = True,
) -> bool:
    stub = _app(app_rect=app_rect, probed=probed)
    with (
        patch.object(app_module, "find_window", return_value=_window(captureable)),
        patch.object(app_module, "window_content_capture_ready", return_value=ready),
        patch.object(app_module, "window_content_capture_available", return_value=available),
    ):
        return HuahuaApp._app_overlaps_game_window(stub)


class OverlapGateTests(unittest.TestCase):
    def test_no_overlap_never_pauses(self) -> None:
        self.assertFalse(
            _gate(app_rect=(0, 2000, 800, 2600), probed=True, ready=False, available=False)
        )

    def test_session_ready_means_overlap_does_not_matter(self) -> None:
        self.assertFalse(
            _gate(app_rect=(0, 0, 800, 600), probed=True, ready=True, available=True)
        )

    def test_available_but_sessionless_still_runs(self) -> None:
        """会话还没建好但 WGC 可用：必须放行，否则永远建不出会话。"""

        self.assertFalse(
            _gate(app_rect=(0, 0, 800, 600), probed=True, ready=False, available=True)
        )

    def test_legacy_backend_pauses_when_covered(self) -> None:
        self.assertTrue(
            _gate(app_rect=(0, 0, 800, 600), probed=True, ready=False, available=False)
        )

    def test_first_frame_is_allowed_through_to_probe(self) -> None:
        self.assertFalse(
            _gate(app_rect=(0, 0, 800, 600), probed=False, ready=False, available=False)
        )

    def test_uncaptureable_window_is_not_an_obstacle(self) -> None:
        self.assertFalse(
            _gate(
                app_rect=(0, 0, 800, 600),
                probed=True,
                ready=False,
                available=False,
                captureable=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
