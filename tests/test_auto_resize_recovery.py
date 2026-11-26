"""拖窗口改尺寸后，自动跟随必须**自己**重新定位并落盘标定。

2026-11-15 23:34 实机：软件刚在 1536x877 上跑得好好的，用户把窗口拖成
1765x965，``paint`` 立刻从 5 掉到 1，而 ``runtime/layout.json`` 的 mtime
纹丝不动。定界发现两段独立问题：

1. ``FramePump`` 的快路径 ``VisionTracker.analyze_quick`` 复用上一帧的棋盘矩形。
   窗口尺寸变了、矩形没变，拿旧框裁新画面逐格全错位，却仍返回一个「成功」的
   scene —— 自愈分支（只在找不到框时进）永远不触发，标定永远落不了盘。
2. ``_locate_board_rect`` 在 1765x965 上要 15.9s（``_best_estimated_board_layout``
   对十几个候选逐个跑全图 Canny + 三重循环 refine 占 11.5s）。就算触发了自愈，
   十几秒不出帧也只会被当成工具死了。

固件是 23:34 拖窗口后的实机帧。
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from huahua_assistant import vision
from huahua_assistant.layout import ScreenLayout
from huahua_assistant.models import Rect, TileState

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "board_resized_1765x965_paints.png"
WIDTH, HEIGHT = 1765, 965
TRUE_RECT = Rect(300, 243, 581, 581)


def _image() -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(FIXTURE), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise unittest.SkipTest(f"读不到固件 {FIXTURE}")
    return image


class AutoResizeRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        vision.set_layout_autosave(False)
        vision.reset_vision_caches()
        vision._RECOVER_ATTEMPTS.clear()

    def tearDown(self) -> None:
        vision.set_layout_autosave(False)
        vision.reset_vision_caches()
        vision._RECOVER_ATTEMPTS.clear()

    def test_size_change_forces_realtime_relocation(self) -> None:
        """尺寸一变必须丢掉旧 scene / 旧框，强制走一次实时定位。"""
        tracker = vision.VisionTracker()
        tracker.frame_size = (1536, 877)
        tracker.last_scene = MagicMock()
        tracker.board_rect = Rect(241, 225, 523, 523)
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        with patch.object(vision, "analyze_scene_realtime", return_value=None) as relocate:
            tracker.analyze_quick(frame)
        relocate.assert_called_once()
        self.assertEqual((WIDTH, HEIGHT), tracker.frame_size)
        self.assertIsNone(tracker.last_scene)
        self.assertIsNone(tracker.board_rect)

    def test_placeholder_frame_does_not_trigger_relocation(self) -> None:
        """游戏最小化时的 1x1 占位帧不能把尺寸记录冲掉。"""
        tracker = vision.VisionTracker()
        tracker.frame_size = (1536, 877)
        with patch.object(vision, "analyze_scene_quick", return_value=None) as quick:
            tracker.analyze_quick(np.zeros((1, 1, 3), dtype=np.uint8))
        self.assertEqual((1536, 877), tracker.frame_size)
        quick.assert_called_once()

    def test_image_geometry_recovers_without_saving_calibration(self) -> None:
        """New sizes are measured from pixels without needing a saved preset."""
        image = _image()
        saved: list[tuple[int, int, Rect]] = []
        with patch.object(vision, "load_layout", return_value=ScreenLayout()), patch.object(
            vision, "save_layout",
            side_effect=lambda w, h, r: saved.append((w, h, r)) or Path("fake"),
        ):
            vision.set_layout_autosave(True)
            vision.reset_vision_caches()
            tracker = vision.VisionTracker()
            tracker.frame_size = (1536, 877)
            tracker.board_rect = Rect(241, 225, 523, 523)   # 旧尺寸的框
            scene = tracker.analyze_quick(image)
        self.assertIsNotNone(scene)
        self.assertLessEqual(abs(scene.board.rect.x - TRUE_RECT.x), 12)
        self.assertLessEqual(abs(scene.board.rect.y - TRUE_RECT.y), 12)
        self.assertEqual(5, sum(1 for t in scene.board.tiles if t.state == TileState.PAINT))
        self.assertEqual([], saved)
        self.assertEqual("image_grid", scene.layout["source"])
        self.assertEqual((WIDTH, HEIGHT), (scene.layout["width"], scene.layout["height"]))

    def test_large_frame_uses_coarse_locate(self) -> None:
        """大窗口必须先在 800 宽缩略图上定位，不能吃满全图搜索。

        只用耗时当判据不可靠：模块缓存热了以后全图搜索也只要 4s（首次冷启动
        才是 15.9s）。所以直接查 ``cv2.resize`` 有没有收到 800 宽这个目标尺寸。
        """
        image = _image()
        real_resize = cv2.resize
        sizes: list[tuple[int, int]] = []

        def spy(src, dsize=None, *args, **kwargs):
            if dsize is not None:
                sizes.append((int(dsize[0]), int(dsize[1])))
            return real_resize(src, dsize, *args, **kwargs)

        with patch.object(vision, "load_layout", return_value=ScreenLayout()), patch.object(
            vision.cv2, "resize", side_effect=spy
        ):
            vision.reset_vision_caches()
            started = time.perf_counter()
            located = vision._locate_board_rect(image)
            elapsed = time.perf_counter() - started
        self.assertIsNotNone(located)
        rect, score = located
        self.assertTrue(
            any(w == 800 for w, _ in sizes),
            f"没有走 800 宽缩略图定位：{sizes[:10]}",
        )
        self.assertLess(elapsed, 10.0, f"定位耗时 {elapsed:.2f}s，超出首帧预算")
        self.assertGreaterEqual(score, 0.68)
        self.assertLessEqual(abs(rect.x - TRUE_RECT.x), 14)
        self.assertLessEqual(abs(rect.y - TRUE_RECT.y), 14)

    def test_stall_budget_covers_first_recovery_frame(self) -> None:
        from huahua_assistant import app

        self.assertGreaterEqual(app.AUTO_FOLLOW_STALL_SECONDS, 12.0)


if __name__ == "__main__":
    unittest.main()
