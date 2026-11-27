"""笔锋选择面板（选笔触方向 / 选笔触效果）文字解析的回归测试。

这一层是「笔锋阶段 + 可选笔锋类型 + 可选词缀」的唯一可信来源。以前靠字形模板打分，
结果是**「选笔触效果」阶段被当成「选笔触方向」**，报出一堆假的笔锋类型，真正的词缀
又匹配到别的条目上。

真正的判据是文字，而且文字结构非常规整：

- 「选笔触方向」每张卡片是 `【转锋】转变调色目标和效果`，描述部分**固定三条**；
- 「选笔触效果」每张卡片是一句词缀描述，来自官方词缀表。

这里锁定五件事：

1. **解析是纯函数**：给 OCR 出来的行，阶段和笔锋类型必须全对（不依赖 OCR 环境）。
2. **表头读花不影响判据**：OCR 把「选择笔触方向/效果」读成「诜怪竿触赦里」，
   所以阶段只能靠三条固定描述判，不能靠表头。
3. **卡片切分有兜底**：两张卡片贴太近连成一条时，退回固定版式。
4. **词缀匹配不能被子串占便宜**：「范围内随机1个颜料提升1级」是长条目的子串，
   单纯算覆盖率会让短别名顶掉正解，必须用 Dice 相似度。
5. **8 张实拍截图逐字段对真值**（需要 Windows OCR，非 Windows 自动跳过）。
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from huahua_assistant import brush_panel
from huahua_assistant.brush_panel import BrushPanelReading
from huahua_assistant.game_model import BrushType

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "brush_panel_labels.json"

# 直接取自实拍截图里 Windows OCR 的原始输出（含 OCR 塞进来的空格和噪声前缀）。
# 注意表头：「选择笔触方向」被读成「诜 怪 竿 触 方 向」，
# 「选择笔触效果」被读成「诜 怪 竿 触 赦 里」——两者都不可用，只能靠描述判。
OCR_SAMPLES: dict[str, list[str]] = {
    "方向_转锋堆砌": [
        "诜 怪 竿 触 方 向",
        "〗 转 变 调 色 目 标 和 效 果",
        "〖 堆 砌 〗 附 加 额 外 的 调 色 效 果",
    ],
    "方向_堆砌扩散_折行": [
        "诜 怪 竿 触 方 向",
        "〖 堆 砌 〗 附 加 额 外 的 调 色 效 果",
        "一 ． ` ' 0 〖 散 〗 对 特 殊 地 块",
        "产 生 作 用",
    ],
    "效果_升级": [
        "诜 怪 竿 触 赦 里",
        "》 埤 身 等 级 提 升 2 级",
        "围 内 的 随 机 1 个 颜 料 额 外 提 升 1 级",
    ],
    "效果_转锋_碎行": [
        "变 自 身 的",
        "变 自 身 的",
        "选 怪 竿 触 赦 里",
        "， 等 级 不 变",
        "， 内 容 不 变",
    ],
}


def _imread(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _synthetic_panel(
    bands: tuple[tuple[float, float], ...],
    height: int = 250,
    width: int = 757,
) -> np.ndarray:
    """造一张假面板：左侧图标列在 ``bands`` 区间里是亮的，其余全黑。"""
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    left = int(width * 0.02)
    right = int(width * 0.14)
    for top, bottom in bands:
        crop[int(height * top) : int(height * bottom), left:right] = 255
    return crop


class NormalizeTest(unittest.TestCase):
    def test_spaces_and_full_width_spaces_are_dropped(self) -> None:
        self.assertEqual(brush_panel.normalize("转 变\u3000调色 目标"), "转变调色目标")

    def test_non_string_input_is_stringified(self) -> None:
        self.assertEqual(brush_panel.normalize(None), "None")


class SimilarityTest(unittest.TestCase):
    def test_lcs_ratio_ignores_order_gaps(self) -> None:
        self.assertEqual(brush_panel.lcs_ratio("abc", "a1b2c"), 1.0)
        self.assertAlmostEqual(brush_panel.lcs_ratio("abcd", "ab"), 0.5)

    def test_empty_side_scores_zero(self) -> None:
        self.assertEqual(brush_panel.lcs_ratio("", "abc"), 0.0)
        self.assertEqual(brush_panel.dice_similarity("abc", ""), 0.0)

    def test_dice_prefers_the_complete_entry_over_a_substring_alias(self) -> None:
        """短别名是长条目的子串时，覆盖率是 1.00，Dice 会把长度也算进去。"""
        short = "范围内随机1个颜料提升1级"
        long = "对作用范围内的随机1个颜料额外提升1级"
        self.assertEqual(brush_panel.lcs_ratio(short, long), 1.0)
        self.assertLess(
            brush_panel.dice_similarity(short, long),
            brush_panel.dice_similarity(long, long),
        )
        self.assertEqual(brush_panel.dice_similarity(long, long), 1.0)


class MatchAffixTest(unittest.TestCase):
    def test_official_name_matches_itself(self) -> None:
        name, score = brush_panel.match_affix("自身提升2级")
        self.assertEqual(name, "自身提升2级")
        self.assertGreaterEqual(score, 0.99)

    def test_in_game_wording_maps_to_the_official_name(self) -> None:
        cases = {
            "使自身等级提升2级": "自身提升2级",
            "可放置于阻挡地块中": "当前颜料可放置在阻挡地块",
            "重新获得1个随机调色颜料": "重新获得一个随机初始颜料",
            "随机转变自身的等级，内容不变": "当前颜料等级随机改变",
            "随机转变自身的内容，等级不变": "随机改变自身的内容，等级不变",
            "转变自身的作用方向为横向": "作用方向改为横向",
            "将纵向范围中最近1个阻挡地块转变为空白地块": "将纵向范围中最近1个阻挡地块转变为空白地块",
            "对作用范围内的随机1个颜料额外提升1级": "范围内随机1个颜料提升1级",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                name, score = brush_panel.match_affix(text)
                self.assertEqual(name, expected)
                self.assertGreaterEqual(score, brush_panel.AFFIX_THRESHOLD)

    def test_unrelated_text_does_not_match(self) -> None:
        name, score = brush_panel.match_affix("选择笔触方向")
        self.assertLess(score, brush_panel.AFFIX_THRESHOLD)
        self.assertEqual(brush_panel.match_affix(""), ("", 0.0))

    def test_duplicates_collapse_and_order_is_kept(self) -> None:
        names, scores = brush_panel.match_affixes(
            ["使自身等级提升2级", "自身提升2级", "可放置于阻挡地块中"]
        )
        self.assertEqual(names, ("自身提升2级", "当前颜料可放置在阻挡地块"))
        self.assertEqual(set(scores), set(names))

    def test_low_confidence_lines_are_dropped(self) -> None:
        names, _scores = brush_panel.match_affixes(["选择笔触效果", "堆砌"])
        self.assertEqual(names, ())


class ParseLinesTest(unittest.TestCase):
    def test_fixed_descriptions_mean_direction_stage(self) -> None:
        reading = brush_panel.parse_lines(OCR_SAMPLES["方向_转锋堆砌"])
        self.assertEqual(reading.stage, "direction")
        self.assertEqual(
            set(reading.brush_types),
            {BrushType.TRANSFORM, BrushType.STACK},
        )
        self.assertEqual(reading.affixes, ())
        self.assertGreaterEqual(reading.confidence, 0.85)
        self.assertFalse(reading.is_empty)

    def test_wrapped_description_still_matches(self) -> None:
        """「对特殊地块产生作用」被折成两行时，拼起来仍要命中。"""
        reading = brush_panel.parse_lines(OCR_SAMPLES["方向_堆砌扩散_折行"])
        self.assertEqual(reading.stage, "direction")
        self.assertEqual(set(reading.brush_types), {BrushType.STACK, BrushType.SPREAD})

    def test_no_fixed_description_means_effect_stage(self) -> None:
        reading = brush_panel.parse_lines(OCR_SAMPLES["效果_升级"])
        self.assertEqual(reading.stage, "effect")
        self.assertEqual(reading.brush_types, ())
        self.assertEqual(reading.confidence, 0.70)

    def test_garbled_header_alone_never_decides_the_stage(self) -> None:
        """两个阶段的表头都读成「诜怪竿触…」，只有表头时不能冒充方向阶段。"""
        for header in ("诜 怪 竿 触 方 向", "诜 怪 竿 触 赦 里", "选 择 笔 触 方 向"):
            with self.subTest(header=header):
                reading = brush_panel.parse_lines([header])
                self.assertEqual(reading.stage, "effect")

    def test_effect_stage_scores_stay_below_the_threshold(self) -> None:
        reading = brush_panel.parse_lines(OCR_SAMPLES["效果_转锋_碎行"])
        self.assertEqual(reading.stage, "effect")
        for score in reading.description_scores.values():
            self.assertLess(score, brush_panel.DESCRIPTION_THRESHOLD)

    def test_empty_input_is_empty_reading(self) -> None:
        for lines in ([], ["", "   "], ["\u3000"]):
            with self.subTest(lines=lines):
                reading = brush_panel.parse_lines(lines)
                self.assertTrue(reading.is_empty)
                self.assertEqual(reading.stage, "unknown")
        self.assertIsNone(brush_panel.read_brush_panel(np.zeros((0, 0, 3), dtype=np.uint8)))


class GutterBandsTest(unittest.TestCase):
    def test_two_bright_bands_are_split(self) -> None:
        crop = _synthetic_panel(brush_panel.CARD_SLOTS)
        bands = brush_panel.gutter_bands(crop)
        self.assertEqual(len(bands), 2)
        self.assertLess(abs(bands[0][0] / 250 - brush_panel.CARD_SLOTS[0][0]), 0.05)
        self.assertLess(abs(bands[1][0] / 250 - brush_panel.CARD_SLOTS[1][0]), 0.05)

    def test_touching_cards_fall_back_to_the_fixed_layout(self) -> None:
        """两张卡片贴得近、连成一条时，退回固定版式而不是把两段文字读成一坨。"""
        crop = _synthetic_panel(((0.06, 0.94),))
        bands = brush_panel.gutter_bands(crop)
        self.assertEqual(
            bands,
            [
                (int(250 * top), int(250 * bottom))
                for top, bottom in brush_panel.CARD_SLOTS
            ],
        )

    def test_dark_panel_yields_nothing(self) -> None:
        self.assertEqual(brush_panel.gutter_bands(_synthetic_panel(())), [])
        self.assertEqual(brush_panel.gutter_bands(np.zeros((0, 0, 3), dtype=np.uint8)), [])


class PanelCacheTest(unittest.TestCase):
    """面板没换就不要重复 OCR——实时帧每 30ms 一帧，一次面板 OCR 就是 90ms。"""

    def setUp(self) -> None:
        brush_panel.reset_cache()
        self.crop = _synthetic_panel(brush_panel.CARD_SLOTS)

    def tearDown(self) -> None:
        brush_panel.reset_cache()

    def test_signature_is_stable_and_changes_with_content(self) -> None:
        first = brush_panel.panel_signature(self.crop)
        self.assertEqual(first, brush_panel.panel_signature(self.crop.copy()))
        other = self.crop.copy()
        other[0, 0] = (255, 255, 255)
        self.assertNotEqual(first, brush_panel.panel_signature(other))

    def test_second_call_reuses_the_cached_reading(self) -> None:
        lines = OCR_SAMPLES["方向_转锋堆砌"]
        with mock.patch.object(
            brush_panel.windows_ocr, "recognize_chinese_lines", return_value=lines
        ) as reader:
            first = brush_panel.read_brush_panel(self.crop)
            second = brush_panel.read_brush_panel(self.crop)
        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertEqual(reader.call_count, 1)

    def test_stale_reading_is_dropped_after_the_window(self) -> None:
        """面板关掉以后，不能拿几秒前的读数继续解释现在的画面。"""
        lines = OCR_SAMPLES["方向_转锋堆砌"]
        with mock.patch.object(
            brush_panel.windows_ocr, "recognize_chinese_lines", return_value=lines
        ):
            brush_panel.read_brush_panel_async(self.crop)
            self.assertTrue(brush_panel.wait_for_async())
            self.assertIsNotNone(brush_panel.read_brush_panel_async(self.crop))

            with mock.patch.object(brush_panel, "PANEL_READ_STALE", 0.05), mock.patch.object(
                brush_panel.windows_ocr, "recognize_chinese_lines", return_value=()
            ):
                time.sleep(0.12)
                other = self.crop.copy()
                other[:40, :] = (1, 2, 3)  # 面板换成另一张（签名一定会变）
                self.assertIsNone(brush_panel.read_brush_panel_async(other))

    def test_reset_cache_forces_a_fresh_read(self) -> None:
        lines = OCR_SAMPLES["方向_转锋堆砌"]
        with mock.patch.object(
            brush_panel.windows_ocr, "recognize_chinese_lines", return_value=lines
        ) as reader:
            brush_panel.read_brush_panel(self.crop)
            brush_panel.reset_cache()
            brush_panel.read_brush_panel(self.crop)
        self.assertEqual(reader.call_count, 2)

    def test_empty_ocr_result_is_not_cached_as_a_reading(self) -> None:
        with mock.patch.object(
            brush_panel.windows_ocr, "recognize_chinese_lines", return_value=()
        ):
            self.assertIsNone(brush_panel.read_brush_panel(self.crop))

    def test_async_version_returns_none_then_delivers_on_the_next_frame(self) -> None:
        """第一帧只发起后台读取（返回 None），跑完之后沿用这份结果。"""
        lines = OCR_SAMPLES["方向_转锋堆砌"]
        with mock.patch.object(
            brush_panel.windows_ocr, "recognize_chinese_lines", return_value=lines
        ):
            self.assertIsNone(brush_panel.read_brush_panel_async(self.crop))
            self.assertTrue(brush_panel.wait_for_async())
            reading = brush_panel.read_brush_panel_async(self.crop)
            self.assertIsNotNone(reading)
            self.assertEqual(reading.stage, "direction")
            self.assertEqual(
                set(reading.brush_types), {BrushType.TRANSFORM, BrushType.STACK}
            )

    def test_animated_panel_still_delivers_a_reading(self) -> None:
        """卡片在呼吸发光：每帧像素都变，签名每帧都不同。

        旧实现把签名当唯一缓存键，于是每帧都是「新面板」→ 每帧起一个 OCR 线程、
        当帧永远返回 None，实机表现为整局读不到笔触面板（2026-11-20 事故）。
        现在按时间节流 + 沿用上次结果：动画帧必须能拿到读数。
        """
        lines = OCR_SAMPLES["方向_转锋堆砌"]
        readings: list[object] = []
        with mock.patch.object(
            brush_panel.windows_ocr, "recognize_chinese_lines", return_value=lines
        ) as reader:
            with mock.patch.object(brush_panel, "PANEL_READ_INTERVAL", 0.0):
                for frame in range(6):
                    crop = self.crop.copy()
                    crop[frame, :] = (frame * 17) % 255  # 每帧都不一样
                    readings.append(brush_panel.read_brush_panel_async(crop))
                    brush_panel.wait_for_async()
            self.assertTrue(
                any(reading is not None for reading in readings),
                "动画帧里必须至少有一次拿到读数",
            )
            # 6 帧不能是 6 次 OCR：节流后每帧最多一次，且命中缓存后为 0 次
            self.assertLessEqual(reader.call_count, 6)

    def test_panel_switch_drops_the_stale_reading(self) -> None:
        """面板整块换掉之后，不能再把上一次的读数递上来。

        2026-11-21 实机日志：22:08:12 点了第一张卡（转锋），紧接着那帧的
        ``brush_stage1_selected`` 却记成「堆砌」——面板已经从阶段一换成阶段二，
        而「沿用上次结果」还在返回上一笔的 reading。录制里于是写进了没选过的笔触。

        呼吸发光（整体亮度小幅波动）必须继续沿用；换面板（两张卡变一张）必须立刻失效。
        """

        direction = BrushPanelReading(
            stage="direction",
            brush_types=(BrushType.TRANSFORM, BrushType.STACK),
            confidence=0.9,
        )
        effect = BrushPanelReading(
            stage="effect",
            affixes=("自身提升1级",),
            confidence=0.9,
        )
        calls: list[object] = [direction]

        def fake_read(crop):
            return calls.pop(0) if calls else None

        with mock.patch.object(brush_panel, "read_brush_panel", side_effect=fake_read):
            with mock.patch.object(brush_panel, "PANEL_READ_INTERVAL", 0.0):
                brush_panel.reset_cache()
                first = brush_panel.read_brush_panel_async(self.crop)
                self.assertIsNone(first)  # 首帧发起请求
                brush_panel.wait_for_async()
                self.assertEqual("direction", brush_panel.read_brush_panel_async(self.crop).stage)

                # 呼吸：整块面板亮度轻微浮动，必须继续沿用（否则又回到「永远读不到」）
                breathing = self.crop.astype("int16")
                breathing[::3] = np.clip(breathing[::3] + 4, 0, 255)
                breathing = breathing.astype("uint8")
                self.assertIsNotNone(brush_panel.read_brush_panel_async(breathing))

                # 换面板：整块换成另一种底色/结构，必须立刻丢掉旧读数
                switched = np.full_like(self.crop, 0)
                switched[:, : self.crop.shape[1] // 2] = 255
                stale_result = brush_panel.read_brush_panel_async(switched)
                self.assertFalse(
                    stale_result is not None and stale_result.stage == "direction",
                    "面板已经换了，不能还返回旧读数",
                )

                brush_panel.wait_for_async()
                settled = brush_panel.read_brush_panel_async(switched)
                self.assertTrue(settled is None or settled.stage == "effect")


class RealScreenshotTest(unittest.TestCase):
    """8 张实拍截图逐字段对真值（阶段 / 笔锋类型 / 词缀，含顺序）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]

    def test_every_field_matches_truth(self) -> None:
        failures: list[str] = []
        checked = 0
        for case in self.cases:
            image_path = ROOT / case["image"]
            if not image_path.exists():
                self.skipTest(f"缺少截图 {image_path}")
            image = _imread(image_path)
            height, width = image.shape[:2]
            rx, ry, rw, rh = case["rect"]
            crop = image[
                int(ry * height) : int((ry + rh) * height),
                int(rx * width) : int((rx + rw) * width),
            ]
            brush_panel.reset_cache()
            reading = brush_panel.read_brush_panel(crop)
            if reading is None:
                self.skipTest("当前环境没有可用的 Windows OCR")
            checked += 1
            name = Path(case["image"]).name
            if reading.stage != case["stage"]:
                failures.append(f"{name} stage={reading.stage} != {case['stage']}")
            got_types = [item.value for item in reading.brush_types]
            if got_types != case["brush_types"]:
                failures.append(f"{name} brush_types={got_types} != {case['brush_types']}")
            if list(reading.affixes) != case["affixes"]:
                failures.append(f"{name} affixes={list(reading.affixes)} != {case['affixes']}")
        self.assertEqual(failures, [], "\n".join(failures))
        self.assertGreaterEqual(checked, len(self.cases))


if __name__ == "__main__":
    unittest.main()
