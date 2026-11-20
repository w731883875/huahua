"""冷启动「完整识别」两条提速路径的回归。

2026-11-20 实测（1673x878 实机截图，``analyze_scene`` 全流程）：

| 阶段 | 改前 | 改后 | 做法 |
|---|---|---|---|
| 棋盘定位 | 4.3s | 0.15s | 尺寸标定过就直接出框，不再全搜索 + 精修 |
| 当前颜料 | 8.0s | 0.02s | 面板文字读全了就跳过 5.3s 的字形模板匹配 |
| 完成绘制按钮 | 1.4s | 0.0s | 没有红字就不可能是「完成绘制」，先判红字再跑模板 |

这三条都不靠墙钟断言（本机 CPU 争抢会让 wall 抖动 2~3 倍），
断言的是**有没有走那条慢路**。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from huahua_assistant import brush_panel, keyword_match, paint_panel, vision
from huahua_assistant.models import RegionKind, RegionObservation, Rect

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT = ROOT / "jietu" / "初始和背包宝箱.png"
COMPLETE_LINES = (
    "玩法颜料-机械之神LV3",
    "调色目标:在横向地块上放置玩法颜料(0/3)",
    "调色效果:使横向的玩法颜料提升1级",
)


def _image() -> np.ndarray:
    if not SCREENSHOT.exists():
        raise unittest.SkipTest("缺少实机截图")
    image = cv2.imdecode(np.fromfile(str(SCREENSHOT), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise unittest.SkipTest("实机截图读不出来")
    return image


def _current_paint_region(image: np.ndarray) -> tuple[np.ndarray, tuple[RegionObservation, ...]]:
    """把「当前颜料」面板裁出来单独测，别每次都跑整张 1673x878（模板匹配很贵）。"""
    height, width = image.shape[:2]
    rect = Rect(int(width * 0.52), int(height * 0.12), int(width * 0.44), int(height * 0.20))
    crop = image[rect.y : rect.bottom, rect.x : rect.right]
    region = RegionObservation(
        kind=RegionKind.CURRENT_PAINT,
        rect=Rect(0, 0, crop.shape[1], crop.shape[0]),
        confidence=0.9,
        label="test",
    )
    return crop, (region,)


def _no_ocr():
    """把 OCR 打桩成「读不到」：这几条只验证棋盘定位走哪条路，不关 OCR 的事。"""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(mock.patch.object(paint_panel, "read_panel", return_value=None))
    stack.enter_context(mock.patch.object(brush_panel, "read_brush_panel", return_value=None))
    stack.enter_context(
        mock.patch.object(brush_panel, "read_brush_panel_async", return_value=None)
    )
    return stack


class PanelTextCoverageTest(unittest.TestCase):
    def test_all_six_fields_are_required(self) -> None:
        full = {
            "category": object(),
            "level": 3,
            "target_object": object(),
            "direction": object(),
            "upgrade_amount": 1,
            "match_required": 3,
        }
        self.assertTrue(vision._panel_text_covers_paint_fields(full))
        for key in list(full):
            partial = {k: v for k, v in full.items() if k != key}
            with self.subTest(missing=key):
                self.assertFalse(vision._panel_text_covers_paint_fields(partial))

    def test_real_panel_lines_cover_every_field(self) -> None:
        reading = paint_panel.parse_lines(list(COMPLETE_LINES))
        self.assertTrue(
            vision._panel_text_covers_paint_fields(vision._panel_text_overrides(reading)),
            reading,
        )

    def test_missing_target_line_falls_back_to_templates(self) -> None:
        reading = paint_panel.parse_lines(list(COMPLETE_LINES[:1] + COMPLETE_LINES[2:]))
        self.assertFalse(
            vision._panel_text_covers_paint_fields(vision._panel_text_overrides(reading)),
            reading,
        )


class CurrentPaintSkipsGlyphsTest(unittest.TestCase):
    """面板文字读全了就不该再跑字形模板匹配（实测省 5.3s）。"""

    def test_complete_text_skips_keyword_matching(self) -> None:
        image, regions = _current_paint_region(_image())
        reading = paint_panel.parse_lines(list(COMPLETE_LINES))
        with mock.patch.object(paint_panel, "read_panel", return_value=reading), mock.patch.object(
            vision, "_current_paint_keyword_scores"
        ) as glyphs:
            current = vision._analyze_current_paint(image, regions)
        glyphs.assert_not_called()
        self.assertIsNotNone(current)
        self.assertEqual(current.level, 3)
        self.assertEqual(current.match_required, 3)
        self.assertEqual(current.upgrade_amount, 1)

    def test_incomplete_text_still_uses_keyword_matching(self) -> None:
        image, regions = _current_paint_region(_image())
        reading = paint_panel.parse_lines([COMPLETE_LINES[0]])
        with mock.patch.object(paint_panel, "read_panel", return_value=reading), mock.patch.object(
            vision, "_current_paint_keyword_scores", return_value={}
        ) as glyphs:
            vision._analyze_current_paint(image, regions)
        self.assertEqual(1, glyphs.call_count)

    def test_no_panel_text_at_all_still_uses_keyword_matching(self) -> None:
        image, regions = _current_paint_region(_image())
        with mock.patch.object(paint_panel, "read_panel", return_value=None), mock.patch.object(
            vision, "_current_paint_keyword_scores", return_value={}
        ) as glyphs:
            vision._analyze_current_paint(image, regions)
        self.assertEqual(1, glyphs.call_count)


class CalibratedBoardShortcutTest(unittest.TestCase):
    """标定过的尺寸不该再跑全搜索 + 精修。"""

    def test_analyze_scene_uses_the_calibrated_rect(self) -> None:
        image = _image()
        calibrated = (Rect(307, 224, 523, 523), 0.83)
        with mock.patch.object(
            vision, "_calibrated_board_rect", return_value=calibrated
        ), mock.patch.object(
            vision, "_locate_board_rect", side_effect=AssertionError("不该走全搜索")
        ), mock.patch.object(
            vision, "_refine_board_rect", side_effect=AssertionError("标定框不该再精修")
        ), _no_ocr():
            scene = vision.analyze_scene(image)
        self.assertIsNotNone(scene)
        self.assertEqual(Rect(307, 224, 523, 523), scene.board.rect)
        self.assertGreaterEqual(scene.board.score, 0.5)

    def test_falls_back_to_search_when_calibration_misses(self) -> None:
        image = _image()
        with mock.patch.object(
            vision, "_calibrated_board_rect", return_value=None
        ), mock.patch.object(
            vision, "_locate_board_rect", return_value=(Rect(307, 224, 523, 523), 0.83)
        ) as search, _no_ocr():
            scene = vision.analyze_scene(image)
        self.assertEqual(1, search.call_count)
        self.assertIsNotNone(scene)

    def test_calibration_rejects_a_rect_without_a_real_grid(self) -> None:
        """标定框也要过网格关：否则会把「棋盘内部一小块」当成棋盘。"""
        image = _image()
        height, width = image.shape[:2]
        bogus = Rect(int(width * 0.02), int(height * 0.02), int(width * 0.12), int(width * 0.12))
        with mock.patch.object(
            vision, "_pick_board_rect", return_value=bogus
        ):
            self.assertIsNone(vision._calibrated_board_rect(image))


class FinishButtonRedTextGateTest(unittest.TestCase):
    """「完成绘制」是红字按钮：没有红字就不该跑 45 次模板匹配。"""

    def test_blue_button_without_red_text_skips_keyword_score(self) -> None:
        blue = np.zeros((60, 220, 3), dtype=np.uint8)
        blue[:, :] = (150, 90, 40)  # BGR：蓝色底
        with mock.patch.object(vision, "_finish_button_keyword_score") as keywords:
            score = vision._finish_drawing_button_confidence(blue, 0.08, 3.6)
        self.assertEqual(0.0, score)
        keywords.assert_not_called()


class ParallelKeywordMatchTest(unittest.TestCase):
    """关键词匹配改成按关键词并行后，结果必须与串行逐字一致。

    ``match_keywords`` 里每个关键词只取自己的最大值，没有跨关键词归约，
    所以并行是安全的；但「安全」要靠断言证明，不能靠推理。这里的判据是：
    单关键词调用（走串行分支）与多关键词调用（走线程池）对同一个关键词
    必须给出同一个分数。
    """

    KEYWORDS = ("横向", "纵向", "怪物", "玩法", "提升1级", "对角斜向")

    def _panel_like_image(self) -> np.ndarray:
        """画一行真实字形，让分数落在有意义区间（不是全 0 也不是全 1）。"""
        canvas = np.zeros((60, 420, 3), dtype=np.uint8)
        template = keyword_match._keyword_template("横向", 20, False, 1.0)
        patch = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        canvas[12 : 12 + patch.shape[0], 20 : 20 + patch.shape[1]] = patch
        return canvas

    def test_parallel_scores_equal_serial_scores(self) -> None:
        image = self._panel_like_image()
        combined = keyword_match.match_keywords(
            image, self.KEYWORDS, font_size_range=range(16, 26, 4), scale=1.5, threshold=0.0
        )
        self.assertEqual(set(self.KEYWORDS), set(combined))
        for keyword in self.KEYWORDS:
            with self.subTest(keyword=keyword):
                solo = keyword_match.match_keywords(
                    image,
                    (keyword,),
                    font_size_range=range(16, 26, 4),
                    scale=1.5,
                    threshold=0.0,
                )
                self.assertAlmostEqual(solo[keyword], combined[keyword], places=9)

    def test_threshold_still_filters(self) -> None:
        image = self._panel_like_image()
        scores = keyword_match.match_keywords(
            image, self.KEYWORDS, font_size_range=range(16, 26, 4), scale=1.5, threshold=0.5
        )
        self.assertTrue(all(score >= 0.5 for score in scores.values()))
        self.assertIn("横向", scores, "画上去的那行字必须被认出来")

    def test_empty_image_returns_nothing(self) -> None:
        self.assertEqual({}, keyword_match.match_keywords(np.zeros((0, 0, 3), np.uint8), ("横向",)))


if __name__ == "__main__":
    unittest.main()
