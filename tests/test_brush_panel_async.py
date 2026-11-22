"""异步 OCR 的 ``pending`` 一旦卡住，整局都读不到词缀。

2026-11-22 实机症状：整局读到 17:21 就再也读不到词缀，之后 ``brush_options``
一条都不产生，``affix_card_clicked`` 全部为 0。

根因在 ``brush_panel._async_read_worker``：它的 ``finally`` 里只在
``generation`` 相等时把 ``pending`` 复位。worker 在 ``reset_cache()`` 之前拿到旧
generation、之后才回来，两者不等 → ``pending`` 永久 True。而
``read_brush_panel_async`` 的放行条件要求 ``not pending``，于是彻底停摆。

两条防线，各有测试：
1. worker 按「是不是自己占的位」复位 ``pending``（``inflight_generation``）；
2. 看门狗：``pending`` 挂超过 :data:`PANEL_READ_PATIENCE` 秒就重新放行。
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from huahua_assistant import brush_panel


def _crop() -> np.ndarray:
    return np.full((60, 120, 3), 128, np.uint8)


class PendingNeverSticksTests(unittest.TestCase):
    def setUp(self) -> None:
        brush_panel.reset_cache()

    def tearDown(self) -> None:
        brush_panel.reset_cache()

    def test_worker_clears_its_own_pending_after_reset(self) -> None:
        """worker 抓的是旧 generation，回来时也必须把自己那一笔放掉。"""

        state = brush_panel._ASYNC_STATE
        state["pending"] = True
        state["inflight_generation"] = 7  # 这笔请求属于 generation 7
        state["generation"] = 8  # 期间 reset_cache 过，generation 已经变了

        with mock.patch.object(brush_panel, "read_brush_panel", return_value=None):
            brush_panel._async_read_worker("k", _crop(), 7)

        self.assertFalse(state["pending"], "旧 generation 的 worker 回来必须放行")
        self.assertIsNone(state["inflight_generation"])

    def test_worker_does_not_clear_a_newer_request(self) -> None:
        """旧 worker 回来不能把**新请求**占的位也清掉，否则会并发两笔 OCR。"""

        state = brush_panel._ASYNC_STATE
        state["pending"] = True
        state["inflight_generation"] = 9  # 现在是 generation 9 在飞
        state["generation"] = 9

        with mock.patch.object(brush_panel, "read_brush_panel", return_value=None):
            brush_panel._async_read_worker("k", _crop(), 7)  # 旧的 7 回来

        self.assertTrue(state["pending"], "新请求还该继续占位")

    def test_watchdog_releases_a_hung_request(self) -> None:
        """OCR 子进程卡死时，看门狗必须重新放行，否则整局停摆。

        worker 换成空实现：真跑起来会立刻把 ``pending`` 清掉，那就测不出看门狗。
        """

        state = brush_panel._ASYNC_STATE
        state["pending"] = True
        state["requested_at"] = 0.0  # 很久以前
        state["inflight_generation"] = 0
        state["generation"] = 0
        state["key"] = None
        state["reading"] = None
        state["reading_vector"] = None

        with mock.patch.object(brush_panel, "_async_read_worker", lambda *a, **k: None):
            brush_panel.read_brush_panel_async(_crop())

        # 看门狗放行后，这一帧应该重新把 pending 打开（发起新请求）。
        self.assertTrue(state["pending"], "看门狗放行后应当重新发起请求")

    def test_patience_is_generous_compared_with_normal_read(self) -> None:
        """正常读取远低于 1 秒，忍耐值不能收得太小。"""

        self.assertGreaterEqual(brush_panel.PANEL_READ_PATIENCE, 2.0)
        self.assertGreater(
            brush_panel.PANEL_READ_PATIENCE,
            brush_panel.PANEL_READ_INTERVAL,
            "忍耐值必须大于节流间隔，否则每帧都会被判成卡死",
        )


if __name__ == "__main__":
    unittest.main()
