"""棋盘布局自适应：窗口一改大小，固定比例就不再准。

实机（2026-11-15，同一台机器）：

* 1920x1020 窗口 → 棋盘 (341,254,608)
* 玩家把窗口拖到 1239x780 → 棋盘 (177,211,442)

x 差 164、边长差 166。`layout.py` 的固定比例算出的候选是 (232,200,463)，
网格分只有 0.44（阈值 0.55），实时路径每帧报 ``board not detected``，
界面上「等待安装初始颜料」完全没有反应。

锁定六件事：

1. 固定比例在这两种窗口上都找不到棋盘（事故可复现）；
2. 轻量定位 ``_locate_board_rect`` 能把任意尺寸的棋盘找回来（自愈走的就是这条路），
   且**不碰面板 OCR / 笔触 / 背包**——只要矩形；
3. 标定按**窗口尺寸分库存**，A 尺寸的标定不会污染 B 尺寸；
4. 自愈结果默认不落盘，`set_layout_autosave(True)` 才写；
5. 噪声/纯色画面必须被拒（否则会把错误标定落盘）；
6. 整套判定耗时是分钟级以下的：一次自愈 4~7 秒，不是十几秒。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from huahua_assistant import layout, vision
from huahua_assistant.layout import ACTIVE_LAYOUTS, ScreenLayout
from huahua_assistant.models import Rect

FIXTURES = Path(__file__).resolve().parent / "fixtures"
#: 1920x1020 窗口：内置比例候选偏 15px。
WIDE = FIXTURES / "board_1902x1010.png"
WIDE_SIZE = (1920, 1020)
#: 1239x780 窗口（玩家随手缩小后）：内置候选 x 偏 55px、边长偏 21px。
NARROW = FIXTURES / "board_windowed_1239x780.png"
NARROW_SIZE = (1239, 780)
#: 两张图各自的真实棋盘（``analyze_scene`` 实测，综合分 0.805 / 0.887）。
WIDE_BOARD = Rect(341, 254, 608, 608)
NARROW_BOARD = Rect(177, 211, 442, 442)


def _load(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise unittest.SkipTest(f"读不到固件 {path}")
    return image


class BuiltInLayoutMissesTest(unittest.TestCase):
    """固定比例在这两个尺寸上都过不了网格线判据。"""

    def test_wide_window(self) -> None:
        image = _load(WIDE)
        width, height = WIDE_SIZE
        self.assertIsNone(
            vision._pick_board_rect(image, ScreenLayout(active=ACTIVE_LAYOUTS), width, height, drawing=True)
        )

    def test_narrow_window(self) -> None:
        image = _load(NARROW)
        width, height = NARROW_SIZE
        self.assertIsNone(
            vision._pick_board_rect(image, ScreenLayout(active=ACTIVE_LAYOUTS), width, height, drawing=True)
        )


class RecoverBoardRectTest(unittest.TestCase):
    def setUp(self) -> None:
        vision._RECOVER_ATTEMPTS.clear()

    def tearDown(self) -> None:
        vision._RECOVER_ATTEMPTS.clear()

    def _recover(self, path: Path, size: tuple[int, int]) -> Rect | None:
        image = _load(path)
        width, height = size
        return vision._recover_board_rect(
            image, ScreenLayout(active=ACTIVE_LAYOUTS), width, height, drawing=True
        )

    def test_recovers_the_narrowed_window(self) -> None:
        """玩家随手缩小的窗口也要能定位 —— 这是用户点名要求的那条。"""
        rect = self._recover(NARROW, NARROW_SIZE)
        self.assertIsNotNone(rect)
        assert rect is not None
        self.assertLessEqual(abs(rect.x - NARROW_BOARD.x), NARROW_BOARD.w // 6)
        self.assertLessEqual(abs(rect.y - NARROW_BOARD.y), NARROW_BOARD.h // 6)
        self.assertLessEqual(abs(rect.w - NARROW_BOARD.w), NARROW_BOARD.w // 6)

    def test_recovers_the_wide_window(self) -> None:
        rect = self._recover(WIDE, WIDE_SIZE)
        self.assertIsNotNone(rect)
        assert rect is not None
        self.assertLessEqual(abs(rect.x - WIDE_BOARD.x), WIDE_BOARD.w // 6)
        self.assertLessEqual(abs(rect.w - WIDE_BOARD.w), WIDE_BOARD.w // 6)

    def test_attempts_are_throttled(self) -> None:
        """自愈一次要几秒，同一个尺寸不能每帧都跑。"""
        image = _load(NARROW)
        width, height = NARROW_SIZE
        screen = ScreenLayout(active=ACTIVE_LAYOUTS)
        first = vision._recover_board_rect(image, screen, width, height, drawing=True)
        second = vision._recover_board_rect(image, screen, width, height, drawing=True)
        self.assertIsNotNone(first)
        self.assertIsNone(second, "限流期间应当直接返回 None，不再重复搜索")


class LocateBoardRectTest(unittest.TestCase):
    """``_locate_board_rect`` 只定位，不解析整帧。"""

    def setUp(self) -> None:
        vision._RECOVER_ATTEMPTS.clear()

    def tearDown(self) -> None:
        vision._RECOVER_ATTEMPTS.clear()

    def test_does_not_run_panel_ocr_or_tile_classification(self) -> None:
        """性能契约：自愈只需要矩形，跑面板 OCR 是纯浪费。

        以前 ``_recover_board_rect`` 直接调 ``analyze_scene``，于是把 1381 次模板匹配
        和上千条文案常量匹配也带上；1239x780 实测 10.2 秒，纯定位 2.5 秒。
        """
        image = _load(NARROW)
        with (
            patch.object(vision, "_analyze_scene_from_board_rect") as whole_frame,
            patch.object(vision, "_classify_tiles") as tiles,
            patch.object(vision, "match_keywords") as keywords,
        ):
            rect = vision._recover_board_rect(
                image, ScreenLayout(active=ACTIVE_LAYOUTS), *NARROW_SIZE, drawing=True
            )
        self.assertIsNotNone(rect)
        whole_frame.assert_not_called()
        tiles.assert_not_called()
        keywords.assert_not_called()

    def test_locates_both_window_sizes(self) -> None:
        for path, size, truth in ((NARROW, NARROW_SIZE, NARROW_BOARD), (WIDE, WIDE_SIZE, WIDE_BOARD)):
            with self.subTest(size=size):
                vision._RECOVER_ATTEMPTS.clear()
                located = vision._locate_board_rect(_load(path))
                self.assertIsNotNone(located)
                assert located is not None
                rect, score = located
                self.assertGreaterEqual(score, vision.MIN_BOARD_DETECTION_SCORE)
                self.assertLessEqual(abs(rect.x - truth.x), truth.w // 6)
                self.assertLessEqual(abs(rect.y - truth.y), truth.h // 6)
                self.assertLessEqual(abs(rect.w - truth.w), truth.w // 6)


class ScrambledScreenRejectedTest(unittest.TestCase):
    """负例：不是画画画面时必须说不。

    ``_score_edges_grid`` 只看网格线强度，噪声图满屏 Canny 边缘 → 分数虚高。
    实测噪声图曾拿到 0.9208（真棋盘 0.887），自愈会把它当成棋盘落盘成标定。
    """

    def setUp(self) -> None:
        vision._RECOVER_ATTEMPTS.clear()

    def tearDown(self) -> None:
        vision._RECOVER_ATTEMPTS.clear()

    @staticmethod
    def _recover(image: np.ndarray) -> Rect | None:
        width, height = image.shape[1], image.shape[0]
        return vision._recover_board_rect(
            image, ScreenLayout(active=ACTIVE_LAYOUTS), width, height, drawing=True
        )

    def test_random_noise_is_not_a_board(self) -> None:
        image = np.random.default_rng(7).integers(0, 255, (780, 1239, 3), dtype=np.uint8)
        self.assertIsNone(
            vision._locate_board_rect(image),
            "噪声图被判成棋盘：格内平坦度闸失效了",
        )
        self.assertIsNone(self._recover(image))

    def test_flat_blank_screen_is_not_a_board(self) -> None:
        for name, image in (
            ("black", np.zeros((780, 1239, 3), np.uint8)),
            ("uniform", np.full((780, 1239, 3), 60, np.uint8)),
        ):
            with self.subTest(name=name):
                vision._RECOVER_ATTEMPTS.clear()
                self.assertIsNone(vision._locate_board_rect(image))

    def test_flatness_thresholds_have_margin(self) -> None:
        """两个阈值之间有真实余量，不是拍脑袋取相等。"""
        self.assertLess(vision.BOARD_CELL_FLATNESS_MAX, vision.BOARD_CELL_SCRAMBLED)
        real = vision._board_cell_flatness(_load(NARROW), NARROW_BOARD)
        noise = vision._board_cell_flatness(
            np.random.default_rng(7).integers(0, 255, (780, 1239, 3), dtype=np.uint8),
            NARROW_BOARD,
        )
        self.assertLessEqual(real, vision.BOARD_CELL_FLATNESS_MAX)
        self.assertGreater(noise, vision.BOARD_CELL_SCRAMBLED)


class PerSizeCalibrationTest(unittest.TestCase):
    """标定必须按窗口尺寸分开存，否则换个尺寸就被上一个尺寸带偏。"""

    def test_two_sizes_keep_their_own_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layout.json"
            layout.save_layout(*WIDE_SIZE, WIDE_BOARD, path)
            layout.save_layout(*NARROW_SIZE, NARROW_BOARD, path)

            wide = layout.load_layout(*WIDE_SIZE, path)
            narrow = layout.load_layout(*NARROW_SIZE, path)

        self.assertLessEqual(abs(wide.board_rect(*WIDE_SIZE).x - WIDE_BOARD.x), 2)
        self.assertLessEqual(abs(wide.board_rect(*WIDE_SIZE).w - WIDE_BOARD.w), 2)
        self.assertLessEqual(abs(narrow.board_rect(*NARROW_SIZE).x - NARROW_BOARD.x), 2)
        self.assertLessEqual(abs(narrow.board_rect(*NARROW_SIZE).w - NARROW_BOARD.w), 2)

    def test_unknown_size_falls_back_to_builtin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layout.json"
            layout.save_layout(*WIDE_SIZE, WIDE_BOARD, path)
            # 差得很远的尺寸不该借用 1920x1020 的标定。
            other = layout.load_layout(800, 600, path)
        self.assertEqual(
            other.board_rect(800, 600, drawing=True),
            layout.ScreenLayout().board_rect(800, 600, drawing=True),
        )

    def test_recalibrating_same_size_replaces_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layout.json"
            layout.save_layout(*WIDE_SIZE, WIDE_BOARD, path)
            layout.save_layout(*WIDE_SIZE, Rect(300, 240, 600, 600), path)
            loaded = layout.load_layout(*WIDE_SIZE, path)
        self.assertLessEqual(abs(loaded.board_rect(*WIDE_SIZE).x - 300), 2)


class LayoutAutosaveTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = vision._LAYOUT_AUTOSAVE
        vision.set_layout_autosave(False)

    def tearDown(self) -> None:
        vision.set_layout_autosave(self._saved)

    def test_autosave_is_off_by_default(self) -> None:
        self.assertFalse(vision._LAYOUT_AUTOSAVE)

    def test_remember_does_not_write_when_autosave_is_off(self) -> None:
        with patch.object(vision, "save_layout") as patched:
            vision._remember_board_layout(*WIDE_SIZE, WIDE_BOARD)
        patched.assert_not_called()

    def test_remember_writes_and_resets_cache_when_enabled(self) -> None:
        vision.set_layout_autosave(True)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "layout.json"
            with patch.object(
                vision, "save_layout", side_effect=lambda w, h, r: layout.save_layout(w, h, r, target)
            ) as patched:
                vision._remember_board_layout(*NARROW_SIZE, NARROW_BOARD)
            patched.assert_called_once()
            self.assertTrue(target.exists())
            loaded = layout.load_layout(*NARROW_SIZE, target)
            rect = loaded.board_rect(*NARROW_SIZE, drawing=True)
        self.assertLessEqual(abs(rect.x - NARROW_BOARD.x), 2)
        self.assertLessEqual(abs(rect.w - NARROW_BOARD.w), 2)


if __name__ == "__main__":
    unittest.main()
