"""拖窗口之后的自适应：布局缓存必须跟着尺寸走（以及**不该**做什么）。

用户的原话是「游戏窗口变化依旧无法自适应识别」。这里锁两件事：

1. ``_screen_layout`` 的缓存必须跟着尺寸换。原来写成
   ``_LAYOUT_CACHE_KEY not in (None, key)``：先来一次不带尺寸的调用，缓存键就固定
   成 ``(None, None)``，之后换成带尺寸的调用也判成「没变」——拖窗口之后永远按旧框
   识别。这条是实打实的 bug，已修。

2. **不要**改成「按比例外推到任意尺寸」（2026-11-16 实测否掉）。
   本机 ``runtime/layout.json`` 的 7 组标定里，1920x1020 是
   x=0.1755 / y=0.2480 / 边长=0.6127，1920x1080 却是 x=0.1906 / y=0.2102 /
   边长=0.5537：同宽、只差 60px 高度，x 偏移差 8.6%、y 偏移差 15%。
   比例不是尺寸无关的，外推会把框算歪；实测还把一帧从 0.28s 拖到 0.90s
   （多评了三个必错的候选框，最后还得走自愈）。
   换尺寸的正解是 ``vision._recover_board_rect`` 自愈一次、按新尺寸落盘标定。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from huahua_assistant import layout, vision
from huahua_assistant.models import Rect

WIDE_BOARD = Rect(341, 254, 608, 608)  # 1920x1020 实测


class CacheFollowsSizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cache = vision._LAYOUT_CACHE
        self._key = vision._LAYOUT_CACHE_KEY
        vision._LAYOUT_CACHE = None
        vision._LAYOUT_CACHE_KEY = None

    def tearDown(self) -> None:
        vision._LAYOUT_CACHE = self._cache
        vision._LAYOUT_CACHE_KEY = self._key

    def test_sized_call_reloads_after_a_sizeless_one(self) -> None:
        self.assertIsNotNone(vision._screen_layout())
        self.assertEqual((None, None), vision._LAYOUT_CACHE_KEY)

        vision._screen_layout(1765, 965)

        self.assertEqual((1765, 965), vision._LAYOUT_CACHE_KEY, "换了尺寸必须重新读标定")

    def test_same_size_is_cached(self) -> None:
        vision._screen_layout(1765, 965)
        cached = vision._LAYOUT_CACHE
        vision._screen_layout(1765, 965)
        self.assertIs(cached, vision._LAYOUT_CACHE, "同一尺寸不该每帧重读文件")

    def test_size_change_invalidates_the_cache(self) -> None:
        vision._screen_layout(1765, 965)
        first = vision._LAYOUT_CACHE
        vision._screen_layout(1447, 965)
        self.assertIsNot(first, vision._LAYOUT_CACHE)


class NoCrossSizeExtrapolationTests(unittest.TestCase):
    """把「别按比例外推」这条结论钉在测试里，避免以后又有人优化回去。"""

    def test_far_size_still_falls_back_to_builtin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layout.json"
            layout.save_layout(1920, 1020, WIDE_BOARD, path)

            other = layout.load_layout(800, 600, path)
            builtin = layout.ScreenLayout()

        self.assertEqual(
            other.board_rect(800, 600, drawing=True),
            builtin.board_rect(800, 600, drawing=True),
        )

    def test_near_size_still_uses_its_own_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layout.json"
            layout.save_layout(1920, 1020, WIDE_BOARD, path)

            loaded = layout.load_layout(1920, 1020, path)
            rect = loaded.board_rect(1920, 1020, drawing=True)

        self.assertLessEqual(abs(rect.x - WIDE_BOARD.x), 2)
        self.assertLessEqual(abs(rect.w - WIDE_BOARD.w), 2)

    def test_candidate_count_stays_at_two(self) -> None:
        """候选数直接决定每帧的网格评分次数，别悄悄变多。"""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layout.json"
            layout.save_layout(1765, 965, Rect(300, 243, 581, 581), path)
            candidates = layout.load_layout(1765, 965, path).candidates(1765, 965, drawing=True)

        self.assertEqual(2, len(candidates))


if __name__ == "__main__":
    unittest.main()
