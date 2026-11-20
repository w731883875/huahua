"""等待开画布局必须先于「布局自愈」判定。

2026-11-15 事故：``_pick_board_rect(drawing=True)`` 在等待画面上失败后，
旧代码直接跑 ``_recover_board_rect`` 全搜索。它按 active 比例搜，最后只是把
start 框当 active 框交回来，却吃掉 6.003s（交互预算 0.5s），全量回归里
``test_centered_start_scene_transitions_to_active_layout_within_budget`` 因此
稳定失败。等待画面有独立比例 + 独立开画按钮，先问它只要几十毫秒。

复用 ``test_vision_compatibility`` 里现成的等待画面合成工厂，不复制像素逻辑。
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from huahua_assistant import vision
from huahua_assistant.models import RegionKind
from tests.test_vision_compatibility import _synthetic_start_drawing_scene


class WaitingLayoutPriorityTests(unittest.TestCase):
    def setUp(self) -> None:
        vision.reset_vision_caches()

    def test_waiting_scene_is_checked_before_expensive_recovery(self) -> None:
        """关键断言是「没跑自愈」；墙钟只兜「别退化成 6 秒」。

        墙钟在这台机器上抖动很大：同一段代码冷跑 5 次实测
        ``0.99 / 1.89 / 1.96 / 0.48 / 0.50``（后台线程抢 CPU，不是算得慢）。
        所以取 5 次里最快的一次比 0.8s——比最好值(0.48)留 60% 余量，
        仍比当初 6.003s 的事故小一个量级；真正的回归闸门是 ``recovery.call_count``。
        """
        image = _synthetic_start_drawing_scene()
        timings = []
        scene = None
        for _ in range(5):
            vision.reset_vision_caches()
            started = time.perf_counter()
            with patch.object(
                vision,
                "_recover_board_rect",
                side_effect=AssertionError("等待开画画面上不该跑布局自愈"),
            ) as recovery:
                scene = vision.analyze_scene_realtime(image)
            timings.append(time.perf_counter() - started)
            self.assertEqual(0, recovery.call_count)

        self.assertIsNotNone(scene)
        self.assertIsNotNone(scene.region(RegionKind.START_DRAWING))
        self.assertLess(
            min(timings),
            0.8,
            f"最快也要 {min(timings):.2f}s（5 次：{', '.join(f'{t:.2f}' for t in timings)}）",
        )

    def test_active_scene_still_allows_recovery_when_layout_misses(self) -> None:
        """不能为了快把自愈整条删掉：非等待画面仍保留这条路（只校验调用点存在）。"""
        import inspect

        source = inspect.getsource(vision.analyze_scene_realtime)
        self.assertIn("_recover_board_rect", source)
        self.assertLess(source.index("_waiting_scene_if_any"), source.rindex("_recover_board_rect"))


if __name__ == "__main__":
    unittest.main()
