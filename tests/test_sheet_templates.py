"""合成模板表（``runtime/paint_sheet``）并入模板库后，覆盖与识别都得对。

要点不是「多加了几百条模板」，而是**自学习库缺的家族能被认出来**，
并且没有把原本认得出的家族认错。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from huahua_assistant import paint_kind
from huahua_assistant import s8_official_data

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "runtime" / "paint_sheet"
CATALOG = json.loads(
    (ROOT / "huahua_assistant" / "assets" / "paint_catalog.json").read_text(encoding="utf-8")
)


class SheetTemplateTests(unittest.TestCase):
    def _sheet_cell(self, family_key: str, level: int = 5) -> np.ndarray:
        path = SHEET / f"{family_key}_Lv{level}_tile.png"
        self.assertTrue(path.exists(), f"缺合成图：{path}")
        image = paint_kind._read_rgba(path)
        self.assertIsNotNone(image)
        # 压到底色上得到 BGR，和识别器拿到的实机格子同格式。
        return paint_kind._sheet_cell(image, 1.0)

    def test_sheet_covers_every_family_in_catalog(self) -> None:
        templates = paint_kind.sheet_templates()
        self.assertTrue(templates, "合成模板表为空，检查 runtime/paint_sheet 是否存在")
        missing = [name for name in paint_kind.known_kinds() if name not in templates]
        self.assertEqual(missing, [], f"合成模板仍缺家族：{missing}")

    def test_sheet_has_every_level_for_each_family(self) -> None:
        """每个家族要覆盖「这个家族在游戏里真实存在的等级」，一条不能少。

        玩法家族不是 1~7 全覆盖：``g008``~``g014`` 只存在 Lv5~Lv7，
        ``g015 赛季`` 只有 Lv3/Lv4。官方目录的 ``levels`` 字段对 g008~g014
        确实含 Lv3/Lv4，但那两级的官方名是「玩法颜料-赛季」——3~4 级是**赛季
        颜料本体**，而画表时图标取的是 ``entry["icon"]``（家族自己的图标），
        于是会画出「黑潮图标 + Lv3 框」这种游戏里不存在的组合。用户
        2026-11-24 明确要求删除，生成脚本也据此跳过。

        所以期望值要按 ``S8_PARTIAL_LEVEL_FAMILIES`` 过滤，不能直接数目录
        的 ``levels``，也不要硬写 7。
        """
        templates = paint_kind.sheet_templates()
        expected = {}
        for entry in CATALOG.values():
            name = entry.get("name")
            if not name:
                continue
            levels = [int(level) for level in (entry.get("levels") or {})]
            allowed = s8_official_data.S8_PARTIAL_LEVEL_FAMILIES.get(name)
            if allowed is not None:
                levels = [level for level in levels if level in allowed]
            expected[name] = len(levels) * len(paint_kind.SHEET_SCALES)
        short = {
            name: (len(templates.get(name, [])), want)
            for name, want in expected.items()
            if len(templates.get(name, [])) != want
        }
        self.assertEqual(short, {}, f"这些家族的合成模板条数不对（实际, 期望）：{short}")

    def test_impossible_levels_are_not_templated(self) -> None:
        """游戏里不存在的组合不能进模板库，否则会污染圆形/菱形判据。

        用户 2026-11-24 口径：「赛季只有 Lv3/Lv4，黑潮~玩偶只有 Lv5~Lv7才是正确的。」
        """
        templates = paint_kind.sheet_templates()
        for family in ("黑潮", "黑帆", "魔方", "迷城", "梦魇", "怪谈", "玩偶"):
            # 5~7 三级 × 尺度数，多一条都说明低等级被当成了有效模板。
            self.assertEqual(
                len(templates.get(family, [])),
                3 * len(paint_kind.SHEET_SCALES),
                f"{family} 的模板条数不对，3~4 级不该存在",
            )
        self.assertEqual(
            len(templates.get("赛季", [])),
            2 * len(paint_kind.SHEET_SCALES),
            "赛季只有 Lv3/Lv4",
        )

    def test_merged_bank_reports_no_missing_family(self) -> None:
        active = paint_kind.bank()
        self.assertEqual(active.unknown_kinds(), ())
        self.assertEqual(set(active.kinds()), set(paint_kind.known_kinds()))

    def test_family_only_in_sheet_is_recognised(self) -> None:
        """梦魇只在合成模板里有，必须认得出来。

        这里造**空**自学习库，不用 load()：实机识别会往磁盘上的库加样本，
        2026-11-15 实测用户已经在游戏里见过梦魇，load() 回来就带着样本，
        「自学习库里没有梦魇」这个前提被真实游玩数据推翻，测试因此变红。
        要验证的是「合成模板能补上自学习库缺的家族」，前提必须由测试自己造。
        """
        bank_without_sheet = paint_kind.KindTemplateBank()
        self.assertNotIn("梦魇", bank_without_sheet.templates)

        cell = self._sheet_cell("g012")
        without = bank_without_sheet.read(cell)
        self.assertIsNone(without.subtype, "空库不该认出梦魇")

        merged = paint_kind.KindTemplateBank()
        self.assertGreater(merged.merge_sheet(), 0, "合成模板表是空的")
        reading = merged.read(cell)
        self.assertEqual(reading.subtype, "梦魇")
        self.assertGreaterEqual(reading.score, paint_kind.KIND_MATCH_THRESHOLD)

    def test_learned_templates_are_not_displaced(self) -> None:
        """合成模板不能顶掉自学习的家族：每个已学家族都要还在。"""
        merged = paint_kind.bank()
        for name in merged.templates:
            self.assertIn(name, merged.sheet or {name}, f"{name} 学习模板丢了")
            self.assertGreater(len(merged.templates[name]), 0)

    def test_adding_learned_template_still_respects_quota(self) -> None:
        """``add()`` 的配额只看自学习模板，合成模板不能把配额占满。"""
        bank = paint_kind.KindTemplateBank()
        bank.merge_sheet()
        self.assertTrue(bank.sheet, "合成模板没挂上")
        cell = self._sheet_cell("m010")
        self.assertTrue(bank.add("测试家族", cell))
        self.assertEqual(len(bank.templates["测试家族"]), 1)

    def test_cache_round_trip_keeps_scales(self) -> None:
        """缓存里的 float32 不能把倍数判成「变了」——那会导致每次启动重算 6 秒。"""
        first = paint_kind._build_sheet_templates()
        self.assertTrue(first)
        paint_kind._save_sheet_cache(first)
        loaded = paint_kind._load_sheet_cache()
        self.assertIsNotNone(loaded, "缓存读不回来：倍数比较可能又踩了 float32 精度")
        self.assertEqual(sorted(loaded), sorted(first))


if __name__ == "__main__":
    unittest.main()
