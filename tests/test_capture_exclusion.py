"""工具窗口「对截屏免疫」：让屏幕复制拍不到自己。

实机背景（2026-11-16）：这台机器上 Windows.Graphics.Capture 对**任何**目标都
返回 ``E_ACCESSDENIED``（进程是 LOW 完整性），采集只能退回屏幕复制。屏幕复制
拍的是合成后的桌面，工具窗口压在棋盘上就进帧——用户看到的「软件窗口挡住棋盘」
就是这么来的。

``SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`` 让窗口对截屏 API 不可见，
于是「挡不挡」不再是个问题。实测：洋红满窗覆盖区，截屏洋红占比 0.775 → 0.000。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from huahua_assistant import app as app_module
from huahua_assistant import capture as capture_module
from huahua_assistant.capture import (
    WDA_EXCLUDEFROMCAPTURE,
    WDA_NONE,
    exclude_tool_windows_from_capture,
    windows_excluded_from_capture,
)


class FakeWidget:
    """只实现窗口树需要的两个方法，避免测试依赖真实 Tk。"""

    def __init__(self, handle: int, children: tuple["FakeWidget", ...] = ()) -> None:
        self.handle = handle
        self._children = children

    def winfo_id(self) -> int:  # noqa: D401 - 与 Tk 同名
        return self.handle

    def winfo_children(self):
        return list(self._children)


def _as_root(widget: FakeWidget):
    return patch.object(capture_module, "_top_level_hwnd", lambda item: item.winfo_id())


class ExcludeToolWindowsTests(unittest.TestCase):
    def setUp(self) -> None:
        capture_module._exclusion_cache.update({"key": None, "at": -1e9, "ok": False})
        self.applied: list[int] = []
        self.refused: set[int] = set()

        def fake_set(hwnd, enabled=True):
            if hwnd in self.refused:
                return False
            if enabled:
                self.applied.append(hwnd)
            return True

        self._set_patch = patch.object(capture_module, "set_capture_exclusion", side_effect=fake_set)
        self._read_patch = patch.object(
            capture_module,
            "capture_exclusion",
            side_effect=lambda hwnd: WDA_EXCLUDEFROMCAPTURE if hwnd in self._excluded else None,
        )
        self._excluded: set[int] = set()
        self._set_patch.start()
        self._read_patch.start()

    def tearDown(self) -> None:
        self._set_patch.stop()
        self._read_patch.stop()

    def test_root_and_every_toplevel_are_excluded(self) -> None:
        root = FakeWidget(100, (FakeWidget(101), FakeWidget(102)))
        with _as_root(root):
            applied = exclude_tool_windows_from_capture(root)

        self.assertEqual([100, 101, 102], applied, "浮层/桌宠/对话框都要盖到")

    def test_already_excluded_windows_are_not_touched_again(self) -> None:
        self._excluded = {100, 101}
        root = FakeWidget(100, (FakeWidget(101),))
        with _as_root(root):
            applied = exclude_tool_windows_from_capture(root)

        self.assertEqual([100, 101], applied)
        self.assertEqual([], self.applied, "已经免疫的窗口不该每 2 秒重设一次")

    def test_missing_widget_is_not_a_failure(self) -> None:
        """传进来一个不是窗口的东西时只能返回 False，绝不能抛。"""

        self.assertFalse(windows_excluded_from_capture(object()))

    def test_all_windows_must_be_excluded_before_overlap_stops_mattering(self) -> None:
        self.refused = {101}
        root = FakeWidget(100, (FakeWidget(101),))
        with _as_root(root):
            self.assertFalse(
                windows_excluded_from_capture(root, refresh_seconds=0.0),
                "只免疫了一部分窗口时，剩下那个照样会把画面弄脏",
            )

    def test_success_is_cached_within_the_refresh_window(self) -> None:
        root = FakeWidget(100, (FakeWidget(101),))
        with _as_root(root):
            self.assertTrue(windows_excluded_from_capture(root))
            before = len(self.applied)
            self.assertTrue(windows_excluded_from_capture(root))

        self.assertEqual(before, len(self.applied), "一秒内重复调用不该重新走一遍窗口树")

    def test_cache_expires_so_a_driver_reset_heals(self) -> None:
        root = FakeWidget(100)
        with _as_root(root):
            self.assertTrue(windows_excluded_from_capture(root, refresh_seconds=0.0))
            self._excluded = set()  # 模拟亲和度被系统/驱动重置
            self.assertTrue(windows_excluded_from_capture(root, refresh_seconds=0.0))

        self.assertEqual([100, 100], self.applied, "过期后必须重新设置一次")


class OverlapGateWithExclusionTests(unittest.TestCase):
    def test_excluded_tool_windows_never_pause_auto_follow(self) -> None:
        capture_module._exclusion_cache.update({"key": None, "at": -1e9, "ok": False})
        stub = app_module.HuahuaApp.__new__(app_module.HuahuaApp)
        app_window = __import__("types").SimpleNamespace(
            hwnd=1234,
            left=0,
            top=0,
            right=1920,
            bottom=1020,
            captureable=True,
        )
        with (
            patch.object(app_module, "find_window", return_value=app_window),
            patch.object(app_module, "windows_excluded_from_capture", return_value=True),
        ):
            stub.winfo_viewable = lambda: True
            self.assertFalse(
                app_module.HuahuaApp._app_overlaps_game_window(stub),
                "窗口对截屏免疫之后，压在棋盘上也不再需要暂停",
            )


if __name__ == "__main__":
    unittest.main()
