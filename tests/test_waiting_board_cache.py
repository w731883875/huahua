"""「等待开画」画面的棋盘分类缓存。

2026-11-15 受控实验：``_analyze_start_drawing_scene`` 每次跑满 36 格全分类约
0.27s，而等待画面在玩家点「开画」之前是静止的。实时链路每帧重跑一遍纯属浪费，
外部一抢 CPU 就会把 0.5s 的交互预算吃穿（8 个 CPU 忙循环进程下实测 0.53~0.86s）。
缓存的 key 同时带棋盘矩形和画面指纹，任何一个变了都必须立刻重新分类 —— 否则
发牌之后界面会继续显示上一帧的空棋盘。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from huahua_assistant import vision
from huahua_assistant.models import Rect


def _image(*, variant: int = 0) -> np.ndarray:
    image = np.full((780, 1239, 3), 32, dtype=np.uint8)
    cv2.rectangle(image, (200, 180), (600, 580), (60, 80, 120), 18)
    for offset in range(1, 6):
        cv2.line(image, (200 + offset * 66, 180), (200 + offset * 66, 580), (70, 90, 130), 2)
        cv2.line(image, (200, 180 + offset * 66), (600, 180 + offset * 66), (70, 90, 130), 2)
    if variant:
        cv2.circle(image, (300, 300), 44, (210, 30, 60), -1)
    return image


class WaitingBoardCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        vision.reset_vision_caches()
        self.rect = Rect(200, 180, 400, 400)
        self.button = (0.9, Rect(900, 620, 160, 60))

    def _scene(self, image: np.ndarray, rect: Rect | None = None):
        return vision._analyze_start_drawing_scene(
            image, rect or self.rect, self.button, board_score=0.8
        )

    def test_identical_frame_reuses_classification(self) -> None:
        image = _image()
        with patch.object(vision, "_classify_tiles", wraps=vision._classify_tiles) as spy:
            first = self._scene(image)
            second = self._scene(image)
            third = self._scene(image)
            self.assertEqual(1, spy.call_count)
        self.assertEqual(first.board.tiles, second.board.tiles)
        self.assertEqual(first.board.tiles, third.board.tiles)
        self.assertGreater(len(first.board.tiles), 0)

    def test_changed_board_pixels_invalidate_cache(self) -> None:
        """棋盘里出现一枚新颜料：缓存必须失效，不能继续报上一帧的空板。"""
        with patch.object(vision, "_classify_tiles", wraps=vision._classify_tiles) as spy:
            self._scene(_image())
            self._scene(_image(variant=1))
            self.assertEqual(2, spy.call_count)
        self.assertNotEqual(
            vision._waiting_board_signature(_image(), self.rect),
            vision._waiting_board_signature(_image(variant=1), self.rect),
        )

    def test_moved_board_rect_invalidate_cache(self) -> None:
        with patch.object(vision, "_classify_tiles", wraps=vision._classify_tiles) as spy:
            self._scene(_image())
            self._scene(_image(), Rect(220, 180, 400, 400))
            self.assertEqual(2, spy.call_count)

    def test_reset_vision_caches_clears_waiting_tiles(self) -> None:
        image = _image()
        with patch.object(vision, "_classify_tiles", wraps=vision._classify_tiles) as spy:
            self._scene(image)
            vision.reset_vision_caches()
            self._scene(image)
            self.assertEqual(2, spy.call_count)

    def test_signature_is_cheap_and_deterministic(self) -> None:
        image = _image()
        self.assertEqual(
            vision._waiting_board_signature(image, self.rect),
            vision._waiting_board_signature(image.copy(), self.rect),
        )
        self.assertLess(len(vision._waiting_board_signature(image, self.rect)), 10000)


if __name__ == "__main__":
    unittest.main()
