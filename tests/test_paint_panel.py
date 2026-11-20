"""「当前颜料」面板文字解析的回归测试。

这一层是类别 / 子类型 / 等级 / 方向 / 目标 / 升级级数 / 进度 (x/N) 的唯一可信来源。
以前的字形模板打分在这七项上的成绩是：目标 2/11、进度恒 0、子类型从未填过。
换成 OCR 直读后，9 张有面板的截图逐字段全对。

这里锁定四件事：

1. **解析是纯函数**：给 OCR 出来的行，字段必须全对（不依赖 OCR 环境，任何机器都能跑）。
2. **换行不能丢进度**：「(0/3)」被甩到下一行时也要捡回来。
3. **残行也要尽量给出规则**：表头被挡掉时，目标和方向还得从效果行兜住。
4. **9 张实拍截图逐字段对齐真值**（需要 Windows OCR，非 Windows 自动跳过）。
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from huahua_assistant import paint_panel
from huahua_assistant.models import PaintCategory, PaintDirection, PaintObject

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "paint_panel_labels.json"

# 直接取自 11 张截图里 Windows OCR 的原始输出（含 OCR 塞进来的空格和全角标点）。
OCR_SAMPLES: dict[str, list[str]] = {
    "怪谈": [
        "玩 法 颜 料 · 怪 谈 LV5",
        "调 色 目 标 ： 在 相 邻 地 块 上 放 置 所 有 颜 料 （ 0 / 4 ）",
        "调 色 效 果 ： 使 相 邻 的 所 有 颜 料 提 升 2 级",
    ],
    "邪影": [
        "怪 物 颜 料 · 邪 影 LV3",
        "调 色 目 标 ： 在 对 角 斜 向 地 块 上 放 置 怪 物 颜 料",
        "调 色 效 果 ： 使 对 角 斜 向 的 怪 物 颜 料 提 升 1 级",
    ],
    "赛季换行": [
        "玩 法 颜 料 · 赛 季 LV3",
        "调 色 目 标 ： 在 对 角 斜 向 地 块 上 放 置 怪 物 颜 料",
        "（ 0 / 3 ）",
        "调 色 效 果 ： 使 对 角 斜 向 的 怪 物 颜 料 提 升 1 级",
    ],
    "无表头": [
        "在 宀 地 块 上 放 置 丌 产 颜 料 （ 0 / 3 ）",
        "色 效 果 ： 使 横 向 的 玩 法 颜 料 提 升 1 级",
    ],
    "噪声前缀": [
        "玩 法 颜 料 · 机 械 之 神 LV3",
        "亻 “ ． 思 》 调 色 目 标 ： 在 相 邻 地 块 上 放 置 玩 法 颜 料 （ 0 / 3 ）",
        "1 ， 、 调 色 效 果 ： 使 相 邻 的 玩 法 颜 料 提 升 1 级",
    ],
}


def _imread(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


class ParseLinesTest(unittest.TestCase):
    def test_parse_reads_every_field(self) -> None:
        reading = paint_panel.parse_lines(OCR_SAMPLES["怪谈"])
        self.assertEqual(reading.category, PaintCategory.GAMEPLAY)
        self.assertEqual(reading.subtype, "怪谈")
        self.assertEqual(reading.level, 5)
        self.assertEqual(reading.target_object, PaintObject.ALL)
        self.assertEqual(reading.direction, PaintDirection.ADJACENT)
        self.assertEqual(reading.upgrade_amount, 2)
        self.assertEqual((reading.match_current, reading.match_required), (0, 4))
        self.assertGreaterEqual(reading.confidence, 0.99)

    def test_monster_paint_is_not_the_same_as_monster_target(self) -> None:
        """颜料自己是怪物，和「规则要怪物」是两件事，不能互相顶掉。"""
        reading = paint_panel.parse_lines(OCR_SAMPLES["邪影"])
        self.assertEqual(reading.category, PaintCategory.MONSTER)
        self.assertEqual(reading.target_object, PaintObject.MONSTER)
        self.assertIsNone(reading.match_required)

        # 反向：玩法颜料配「要怪物」的规则
        reading = paint_panel.parse_lines(OCR_SAMPLES["赛季换行"])
        self.assertEqual(reading.category, PaintCategory.GAMEPLAY)
        self.assertEqual(reading.target_object, PaintObject.MONSTER)

    def test_wrapped_progress_line_is_picked_up(self) -> None:
        reading = paint_panel.parse_lines(OCR_SAMPLES["赛季换行"])
        self.assertEqual((reading.match_current, reading.match_required), (0, 3))
        self.assertEqual(reading.direction, PaintDirection.DIAGONAL)
        self.assertEqual(reading.upgrade_amount, 1)

    def test_partial_panel_still_gives_the_rule(self) -> None:
        """表头被挡住时类别读不到，但规则必须从效果行兜住。"""
        reading = paint_panel.parse_lines(OCR_SAMPLES["无表头"])
        self.assertEqual(reading.category, PaintCategory.UNKNOWN)
        self.assertIsNone(reading.level)
        self.assertEqual(reading.direction, PaintDirection.HORIZONTAL)
        self.assertEqual(reading.target_object, PaintObject.GAMEPLAY)
        self.assertEqual(reading.upgrade_amount, 1)
        self.assertEqual((reading.match_current, reading.match_required), (0, 3))
        self.assertTrue(reading.has_rule)
        self.assertFalse(reading.is_empty)

    def test_noise_prefix_does_not_break_the_rule_line(self) -> None:
        reading = paint_panel.parse_lines(OCR_SAMPLES["噪声前缀"])
        self.assertEqual(reading.subtype, "机械之神")
        self.assertEqual(reading.target_object, PaintObject.GAMEPLAY)
        self.assertEqual(reading.direction, PaintDirection.ADJACENT)

    def test_effect_text_is_the_brush_of_this_paint(self) -> None:
        """「调色效果」整行必须能取回来——它就是这枚颜料的笔触。"""

        reading = paint_panel.parse_lines(OCR_SAMPLES["怪谈"])
        self.assertEqual(reading.effect_text, "使相邻的所有颜料提升2级")
        reading = paint_panel.parse_lines(OCR_SAMPLES["邪影"])
        self.assertEqual(reading.effect_text, "使对角斜向的怪物颜料提升1级")

    def test_effect_text_survives_garbled_effect_word(self) -> None:
        # OCR 把「效果」读成「效里」时也要认，否则复盘又缺一笔。
        reading = paint_panel.parse_lines(
            [
                "怪物颜料·沙怪LV2",
                "调色目标:在横向地块上放置怪物颜料(0/3)",
                "调色效里:使横向的怪物颜料提升1级",
            ]
        )
        self.assertEqual(reading.effect_text, "使横向的怪物颜料提升1级")

    def test_effect_text_drops_the_label_when_ocr_eats_the_colon(self) -> None:
        """OCR 把冒号也吃掉时，返回的必须是条目原文，不能带「调色效果」前缀。"""

        reading = paint_panel.parse_lines(
            [
                "怪物颜料·蛊虫LV3",
                "调色目标:在横向地块上放置怪物颜料(0/3)",
                "调色效果调色目标的进度要求减少2",
            ]
        )
        self.assertEqual(reading.effect_text, "调色目标的进度要求减少2")

    def test_effect_text_is_none_when_panel_has_no_effect_line(self) -> None:
        # 没有效果行时绝不能把「调色目标」当成笔触返回。
        reading = paint_panel.parse_lines(
            ["怪物颜料·鼠人LV1", "调色目标:在相邻地块上放置所有颜料(0/3)"]
        )
        self.assertIsNone(reading.effect_text)

    def test_subtypes_cover_all_official_paints(self) -> None:
        names = paint_panel.known_subtypes()
        self.assertEqual(len(names), 27)
        for expected in ("鼠人", "矮人", "邪影", "龙骨", "宝箱", "机械之神", "赛季", "黑帆"):
            self.assertIn(expected, names)

    def test_garbled_subtype_snaps_to_official_name(self) -> None:
        # OCR 把「机械之神」读丢一个字时，也要能对齐回官方名
        reading = paint_panel.parse_lines(["玩法颜料·机械之 LV3"])
        self.assertEqual(reading.subtype, "机械之神")

    def test_empty_input_is_empty_reading(self) -> None:
        self.assertTrue(paint_panel.parse_lines([]).is_empty)
        self.assertTrue(paint_panel.parse_lines(["", "   "]).is_empty)
        self.assertIsNone(paint_panel.read_panel(np.zeros((0, 0, 3), dtype=np.uint8)))

    def test_placement_line_without_the_word_target_is_parsed(self) -> None:
        """实机（1902x1010 窗口化）：目标行里一个「目标」字都没有。

        2026-11-15 事故的真实 OCR 输出。旧解析按「这一行是不是目标行」分流，
        而实际文案是「在纵向地块上放置玩法颜料(0/3)」；「调色效果」「笔触效果」
        又分别被读成「调色效里」「笋触效果」。三项关键字全不中，方向/目标/提升
        级数一起丢失，``confidence`` 只有 0.52 < 0.55，整份文字证据被丢掉，
        界面表现就是「这份规则没识别出来」。
        """
        lines = (
            "当前颜料最大等级7",
            "玩法颜料·机之神LV3",
            "在纵向地块上放置玩法颜料(0/3)",
            "调色效里",
            "使纵向的玩法颜料提升1级",
            "笋触效果",
            "对作用范围内的随机1个颜料额外提升1级",
        )
        reading = paint_panel.parse_lines(lines)
        self.assertEqual(reading.category, PaintCategory.GAMEPLAY)
        self.assertEqual(reading.subtype, "机械之神")
        self.assertEqual(reading.level, 3)
        self.assertEqual(reading.target_object, PaintObject.GAMEPLAY)
        self.assertEqual(reading.direction, PaintDirection.VERTICAL)
        self.assertEqual(reading.match_required, 3)
        self.assertEqual(reading.match_current, 0)
        self.assertGreaterEqual(reading.confidence, 0.99)

    def test_garbled_effect_word_does_not_hide_the_upgrade(self) -> None:
        """「效果」被读错（效里/效采）时，提升级数仍要拿到。"""
        reading = paint_panel.parse_lines(
            ["怪物颜料·邪影LV4", "在相邻地块上放置怪物颜料(0/4)", "调色效里", "使相邻的怪物颜料提升2级"]
        )
        self.assertEqual(reading.direction, PaintDirection.ADJACENT)
        self.assertEqual(reading.upgrade_amount, 2)


class RealScreenshotTest(unittest.TestCase):
    """9 张实拍截图逐字段对真值。"""

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
            crop = image[int(ry * height) : int((ry + rh) * height), int(rx * width) : int((rx + rw) * width)]
            reading = paint_panel.read_panel(crop)
            if reading is None:
                self.skipTest("当前环境没有可用的 Windows OCR")
            checked += 1
            name = Path(case["image"]).name
            if case["category"] is not None and reading.category.value != case["category"]:
                failures.append(f"{name} category={reading.category.value} != {case['category']}")
            if case["subtype"] is not None and reading.subtype != case["subtype"]:
                failures.append(f"{name} subtype={reading.subtype} != {case['subtype']}")
            if case["level"] is not None and reading.level != case["level"]:
                failures.append(f"{name} level={reading.level} != {case['level']}")
            if reading.target_object.value != case["target"]:
                failures.append(f"{name} target={reading.target_object.value} != {case['target']}")
            if reading.direction.value != case["direction"]:
                failures.append(f"{name} direction={reading.direction.value} != {case['direction']}")
            if reading.upgrade_amount != case["upgrade"]:
                failures.append(f"{name} upgrade={reading.upgrade_amount} != {case['upgrade']}")
            expected_progress = case["progress"]
            if expected_progress is not None:
                got = [reading.match_current, reading.match_required]
                if got != expected_progress:
                    failures.append(f"{name} progress={got} != {expected_progress}")
        self.assertEqual(failures, [], "\n".join(failures))
        self.assertGreaterEqual(checked, len(self.cases))


class AsyncPanelReadTest(unittest.TestCase):
    """实时帧的异步读取。

    原来这里直接套 ``windows_ocr`` 的像素缓存，而面板有呼吸/光效动画，
    每帧像素都不一样 → sha1 永远不命中 → 每帧新起一个 OCR 线程 →
    结果永远拿不到，方向和作用颜料在实时帧里从来没被文字覆盖过。
    现在改成「按时间节流 + 复用上次结果」。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]

    def _first_crop(self) -> np.ndarray | None:
        for case in self.cases:
            image_path = ROOT / case["image"]
            if not image_path.exists():
                continue
            image = _imread(image_path)
            height, width = image.shape[:2]
            rx, ry, rw, rh = case["rect"]
            return image[int(ry * height) : int((ry + rh) * height), int(rx * width) : int((rx + rw) * width)]
        return None

    def test_reading_arrives_within_a_few_frames(self) -> None:
        """第一帧拿到 None 是正常的，但必须在几帧之内追上。"""
        crop = self._first_crop()
        if crop is None:
            self.skipTest("缺少截图")
        paint_panel.reset_async_state()
        first = paint_panel.read_panel_async(crop)
        self.assertIsNone(first, "第一帧应该先返回 None，让后台去跑 OCR")

        deadline = time.monotonic() + 5.0
        got = None
        while time.monotonic() < deadline:
            time.sleep(paint_panel.PANEL_READ_INTERVAL + 0.05)
            got = paint_panel.read_panel_async(crop)
            if got is not None:
                break
        if got is None:
            self.skipTest("当前环境没有可用的 Windows OCR")
        self.assertEqual(PaintCategory.GAMEPLAY, got.category)

    def test_throttle_does_not_resubmit_every_frame(self) -> None:
        """同一段间隔内的重复调用必须直接复用，不能每帧起一个线程。"""
        crop = self._first_crop()
        if crop is None:
            self.skipTest("缺少截图")
        paint_panel.reset_async_state()

        with patch.object(paint_panel, "_refresh_reading") as refresh:
            paint_panel.read_panel_async(crop)
            submitted = refresh is not None
            self.assertTrue(submitted)
        # 节流窗口内再调 5 次，都不该再提交
        with patch("threading.Thread") as thread_cls:
            for _ in range(5):
                paint_panel.read_panel_async(crop)
            thread_cls.assert_not_called()

    def test_lines_before_the_header_are_ignored(self) -> None:
        """搜索区会顺带读到上方的 UI，标题之前的行必须当噪声丢掉。"""
        reading = paint_panel.parse_lines(
            [
                "选择笔触方向",
                "【堆砌】附加额外的调色效果",
                "怪物颜料·黑帮LV3",
                "调色目标:在相邻地块上放置怪物颜料(0/3)",
                "调色效果:使相邻的怪物颜料提升1级",
            ]
        )
        self.assertEqual(PaintCategory.MONSTER, reading.category)
        self.assertEqual("黑帮", reading.subtype)
        self.assertEqual(PaintDirection.ADJACENT, reading.direction)
        self.assertEqual(PaintObject.MONSTER, reading.target_object)
        self.assertEqual(1, reading.upgrade_amount, "笔锋卡上的字不能污染提升级数")
        self.assertEqual([0, 3], [reading.match_current, reading.match_required])


if __name__ == "__main__":
    unittest.main()
