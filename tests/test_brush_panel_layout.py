"""笔触面板「这块区域是不是两张卡片」的版式判据回归。

2026-11-20 实机事故：用户新窗口 1607x822 下，两张笔触卡片被切成 404x116 / 390x117，
旧判据要求「宽高比 ≥3.8」，两张都不达标 → ``_detect_brush_options_mode`` 判否 →
BRUSH_OPTIONS 区域根本不建 → 那一局 61 帧的 ``brush.stage1/stage2`` 恒为空。
游戏里明明选过笔触，录制日志里一条都没有。

固定件说明：

- ``brush_panel_user_stage{1,2}.jpg``：用户 2026-11-20 提供的两张实机截图（原尺寸），
  ``BOARD_RECT`` 是在图上量出来的 6x6 盘面外框，只用来算面板 ROI；
- ``roi_*.jpg``：从 ``jietu/`` 老截图（1673x878）按同一套 ROI 规则裁下来的面板区，
  作为第二个窗口尺寸的对照，以及背包/待开画/结束画面的负例。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from unittest import mock

from huahua_assistant import brush_panel, vision
from huahua_assistant.models import Rect, RegionKind, RegionObservation

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "brush_panel_layout"
#: 用户截图里量出来的盘面外框（x, y, w, h），与 1673x878 老截图不是同一套比例。
BOARD_RECT = (173, 128, 639, 650)


def _imread(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise unittest.SkipTest(f"缺少固定件 {path}")
    return image


def _roi(image: np.ndarray, board: Rect) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = vision._clamp_int(board.right + board.w * 0.10, 0, width - 1)
    right = vision._clamp_int(max(x0 + 1, width - board.w * 0.06), 1, width)
    panel_w = max(1, right - x0)
    rect = vision._bounded_rect(
        x0, board.y + board.h * 0.24, panel_w, board.h * 0.45, width, height
    )
    return image[rect.y : rect.bottom, rect.x : rect.right]


class UserScreenshotTest(unittest.TestCase):
    """用户新窗口尺寸：必须认成笔触面板（旧判据在这里全判否）。"""

    def test_stage1_and_stage2_panels_are_detected(self) -> None:
        for name in ("brush_panel_user_stage1.jpg", "brush_panel_user_stage2.jpg"):
            with self.subTest(name=name):
                image = _imread(FIXTURES / name)
                height, width = image.shape[:2]
                board = Rect(*BOARD_RECT)
                x0 = vision._clamp_int(board.right + board.w * 0.10, 0, width - 1)
                right = vision._clamp_int(max(x0 + 1, width - board.w * 0.06), 1, width)
                panel_w = max(1, right - x0)
                self.assertTrue(
                    vision._detect_brush_options_mode(image, x0, board, panel_w),
                    f"{name} 必须被判成笔触面板",
                )
                self.assertTrue(brush_panel.card_bands_ok(_roi(image, board)))

    def test_card_aspect_ratio_here_would_fail_the_old_rule(self) -> None:
        """留一条证据：这张图上的卡片宽高比 <3.8，旧判据必然误杀。

        判据换成亮度分带以后，宽高比不再参与判断——这条用例防止有人把
        「宽高比 ≥3.8」再加回去。
        """
        image = _imread(FIXTURES / "brush_panel_user_stage1.jpg")
        roi = _roi(image, Rect(*BOARD_RECT))
        cards = vision._split_brush_cards(roi)
        ratios = [card.shape[1] / max(1, card.shape[0]) for card in cards]
        self.assertTrue(ratios, "至少要能切出卡片")
        self.assertLess(max(ratios), 3.8)


class RoiFixtureTest(unittest.TestCase):
    """两个窗口尺寸的正例 + 四张负例。"""

    POSITIVE = ("roi_direction_1673.jpg", "roi_effect_1673.jpg")
    NEGATIVE = ("roi_backpack.jpg", "roi_finished.jpg", "roi_waiting.jpg", "roi_start_pack.jpg")

    def test_brush_panels_pass(self) -> None:
        for name in self.POSITIVE:
            with self.subTest(name=name):
                self.assertTrue(brush_panel.card_bands_ok(_imread(FIXTURES / name)), name)

    def test_other_panels_are_rejected(self) -> None:
        for name in self.NEGATIVE:
            with self.subTest(name=name):
                self.assertFalse(brush_panel.card_bands_ok(_imread(FIXTURES / name)), name)

    def test_blank_and_tiny_crops_are_rejected(self) -> None:
        self.assertFalse(brush_panel.card_bands_ok(np.zeros((0, 0, 3), dtype=np.uint8)))
        self.assertFalse(brush_panel.card_bands_ok(np.zeros((30, 100, 3), dtype=np.uint8)))
        black = np.zeros((251, 755, 3), dtype=np.uint8)
        self.assertFalse(brush_panel.card_bands_ok(black))


def _panel_region(image: np.ndarray) -> RegionObservation:
    """按 ``vision`` 里同一套 ROI 规则算出笔触面板区域。"""
    height, width = image.shape[:2]
    board = Rect(*BOARD_RECT)
    x0 = vision._clamp_int(board.right + board.w * 0.10, 0, width - 1)
    right = vision._clamp_int(max(x0 + 1, width - board.w * 0.06), 1, width)
    panel_w = max(1, right - x0)
    rect = vision._bounded_rect(
        x0, board.y + board.h * 0.23, panel_w, board.h * 0.48, width, height
    )
    return RegionObservation(
        kind=RegionKind.BRUSH_OPTIONS, rect=rect, confidence=0.9, label="test"
    )


class PanelToOptionsTest(unittest.TestCase):
    """面板读到了文字之后，阶段/选项必须一路传到 ``BrushOptionsObservation``。

    这是录制链路真正消费的字段：``stage1_options`` 决定点了哪类笔触，
    ``stage2_options`` 决定这一笔带什么词缀。OCR 在测试环境里不可用，按真值打桩。

    打桩目标是 ``read_brush_panel``——``_analyze_brush_options``（完整识别）用它，
    ``_analyze_brush_options_realtime`` 用的是 ``read_brush_panel_async``。打错函数
    时打桩形同不存在，用例会静默掉进字形/光栅兜底，得到一堆与真链路无关的假结论。
    """

    def test_stage1_panel_yields_the_two_brush_types(self) -> None:
        image = _imread(FIXTURES / "brush_panel_user_stage1.jpg")
        reading = brush_panel.parse_lines(
            ["选择笔触方向", "【转锋】转变调色目标和效果", "【堆砌】附加额外的调色效果"]
        )
        self.assertEqual(reading.stage, "direction")
        self.assertEqual(("transform", "stack"), tuple(item.value for item in reading.brush_types))
        with mock.patch.object(brush_panel, "read_brush_panel", return_value=reading):
            observation = vision._analyze_brush_options(image, (_panel_region(image),))
        self.assertIsNotNone(observation)
        self.assertEqual(("transform", "stack"), observation.stage1_options)
        self.assertEqual((), observation.stage2_options)

    def test_realtime_stage1_panel_also_yields_the_two_brush_types(self) -> None:
        """实时帧那条路（录制实际消费的）必须同样认得出来。"""
        image = _imread(FIXTURES / "brush_panel_user_stage1.jpg")
        reading = brush_panel.parse_lines(
            ["选择笔触方向", "【转锋】转变调色目标和效果", "【堆砌】附加额外的调色效果"]
        )
        brush_panel.reset_cache()
        with mock.patch.object(
            brush_panel, "read_brush_panel_async", return_value=reading
        ):
            observation = vision._analyze_brush_options_realtime(
                image, (_panel_region(image),)
            )
        self.assertIsNotNone(observation)
        self.assertEqual(("transform", "stack"), observation.stage1_options)
        self.assertEqual((), observation.stage2_options)

    def test_stage2_panel_yields_the_affix_names(self) -> None:
        """阶段二的词缀来自卡片 OCR，``parse_lines`` 只判阶段不填词缀。

        所以这里直接构造 :class:`BrushPanelReading`——用 ``parse_lines`` 的结果打桩
        会拿到 ``affixes=()``，那条分支直接不成立，测出来的东西和真链路无关。
        """
        image = _imread(FIXTURES / "brush_panel_user_stage2.jpg")
        expected = ("随机改变自身的作用方向", "作用方向改为对角斜向")
        reading = brush_panel.BrushPanelReading(
            stage="effect",
            affixes=expected,
            affix_scores={name: 0.92 for name in expected},
            confidence=0.9,
        )
        with mock.patch.object(brush_panel, "read_brush_panel", return_value=reading):
            observation = vision._analyze_brush_options(image, (_panel_region(image),))
        self.assertIsNotNone(observation)
        self.assertEqual(expected, observation.stage2_options)
        self.assertEqual("transform", observation.chosen_stage1)

    def test_realtime_stage2_panel_yields_the_affix_names(self) -> None:
        image = _imread(FIXTURES / "brush_panel_user_stage2.jpg")
        expected = ("随机改变自身的作用方向", "作用方向改为对角斜向")
        reading = brush_panel.BrushPanelReading(
            stage="effect",
            affixes=expected,
            affix_scores={name: 0.92 for name in expected},
            confidence=0.9,
        )
        brush_panel.reset_cache()
        with mock.patch.object(
            brush_panel, "read_brush_panel_async", return_value=reading
        ):
            observation = vision._analyze_brush_options_realtime(
                image, (_panel_region(image),)
            )
        self.assertIsNotNone(observation)
        self.assertEqual(expected, observation.stage2_options)


class RasterTemplateGeometryGateTest(unittest.TestCase):
    """光栅模板只在「面板形状接近模板拍摄时」可信。

    模板是从 1673x878 窗口截的（面板 760x251，宽高比 3.03），两个文字框是写死的
    像素坐标。窗口一换，同一套裁剪框就压到别的行上——2026-11-20 在 1607x822 的
    截图里，一张「转变自身的作用方向」被读成「作用方向改为横向 0.88」，
    是纯假匹配。宁可这一帧不给词缀，也不能把错词缀写进录制。
    """

    def test_resized_panel_yields_nothing(self) -> None:
        image = _imread(FIXTURES / "brush_panel_user_stage2.jpg")
        crop = image[
            _panel_region(image).rect.y : _panel_region(image).rect.bottom,
            _panel_region(image).rect.x : _panel_region(image).rect.right,
        ]
        self.assertFalse(vision._brush_panel_matches_template_geometry(crop))
        self.assertEqual(((), {}), vision._match_brush_stage2_raster_templates(crop))
        self.assertEqual(((), {}), vision._match_brush_stage2_windows_ocr(crop))

    def test_canonical_shape_and_degenerate_input(self) -> None:
        width, height = vision.BRUSH_STAGE2_CANONICAL_PANEL_SIZE
        self.assertTrue(
            vision._brush_panel_matches_template_geometry(
                np.zeros((height, width, 3), dtype=np.uint8)
            )
        )
        self.assertFalse(
            vision._brush_panel_matches_template_geometry(np.zeros((0, 0, 3), dtype=np.uint8))
        )


if __name__ == "__main__":
    unittest.main()
