"""颜料种类（家族）识别的回归测试。

种类识别踩过的坑，逐条锁死：

1. **同一家族跨等级必须认成同一个**。外框颜色只表示等级，中央图案不变——
   「白发女王」在 Lv1 银框、Lv2 绿框、Lv4 紫框里是同一张图。
2. **不同家族不能混**。实测同族最低 0.802、异族最高 0.547，阈值 0.68 落在空档里。
3. **深色漩涡地块不能被当成颜料**。它们两两相似度能到 1.000，比任何真同族对都高，
   放进来会把模板库整个吃掉。
4. **面板图标要能对回棋盘上的那一格**，这是唯一能自动拿到「图案 ↔ 名字」的路子。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from huahua_assistant import codebook, paint_kind

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "paint_kind_cells.json"


def _imread(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _cells(image: np.ndarray, rect: list[int]) -> dict[tuple[int, int], np.ndarray]:
    """按布局模块的取整规则切 36 格（和 ``vision._classify_tiles`` 一致）。"""
    x, y, w, h = rect
    cell_w = w / 6
    cell_h = h / 6
    result: dict[tuple[int, int], np.ndarray] = {}
    for row in range(6):
        for col in range(6):
            x1 = round(x + col * cell_w)
            y1 = round(y + row * cell_h)
            x2 = round(x + (col + 1) * cell_w)
            y2 = round(y + (row + 1) * cell_h)
            result[(row, col)] = image[y1:y2, x1:x2]
    return result


def _recolor(tile: np.ndarray, hue_shift: int) -> np.ndarray:
    """整体转色相，保持形状不变——用来模拟「同一张图案换了外框等级色」。"""
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_shift) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _scramble(tile: np.ndarray, seed: int) -> np.ndarray:
    """打乱像素位置：颜色直方图和原图**完全相同**，空间结构被彻底破坏。"""
    flat = tile.reshape(-1, 3).copy()
    np.random.default_rng(seed).shuffle(flat)
    return flat.reshape(tile.shape)


class PaintKindFixture(unittest.TestCase):
    """所有用例共用一份棋盘切图，避免每例都重读整张截图。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
        board = cls.spec["board"]
        cls.image = _imread(ROOT / board["image"])
        cls.cells = _cells(cls.image, board["rect"])

    def cell(self, position: tuple[int, int]) -> np.ndarray:
        return self.cells[position]


class DarkTileTest(PaintKindFixture):
    def test_dark_tiles_are_not_paint(self) -> None:
        """16 个深色漩涡地块必须全部被挡掉。"""
        for position in self.spec["dark_tiles"]:
            with self.subTest(position=position):
                self.assertFalse(paint_kind.looks_like_paint_tile(self.cell(tuple(position))))

    def test_real_paint_tiles_are_kept(self) -> None:
        for family in self.spec["families"].values():
            for position in family["cells"]:
                with self.subTest(position=position):
                    self.assertTrue(paint_kind.looks_like_paint_tile(self.cell(tuple(position))))


class SameFamilyAcrossLevelsTest(PaintKindFixture):
    def test_same_art_different_level_still_matches(self) -> None:
        """每个家族只入库一个样本，同族其它等级必须都能认出来。"""
        bank = paint_kind.KindTemplateBank()
        for name, family in self.spec["families"].items():
            bank.add(name, self.cell(tuple(family["cells"][0])))
        for name, family in self.spec["families"].items():
            for position in family["cells"][1:]:
                with self.subTest(family=name, position=position):
                    reading = bank.read(self.cell(tuple(position)))
                    self.assertTrue(reading.present, f"{name} 在 {position} 没认出来")
                    self.assertEqual(reading.subtype, name)
                    self.assertGreaterEqual(reading.score, paint_kind.KIND_MATCH_THRESHOLD)

    def test_other_families_are_rejected(self) -> None:
        """只入库一个家族时，别的家族不能被硬套成它。"""
        bank = paint_kind.KindTemplateBank()
        bank.add("ice", self.cell(tuple(self.spec["families"]["ice"]["cells"][0])))
        for name, family in self.spec["families"].items():
            if name == "ice":
                continue
            for position in family["cells"]:
                with self.subTest(family=name, position=position):
                    self.assertFalse(bank.read(self.cell(tuple(position))).present)

    def test_dark_tile_never_matches(self) -> None:
        bank = paint_kind.KindTemplateBank()
        for name, family in self.spec["families"].items():
            bank.add(name, self.cell(tuple(family["cells"][0])))
        for position in self.spec["dark_tiles"][:4]:
            with self.subTest(position=position):
                self.assertFalse(bank.read(self.cell(tuple(position))).present)


class PrefilterCoverageTest(PaintKindFixture):
    """颜色预筛只能砍冗余模板，不能把整个家族挡在 NCC 评估之外。

    实测事故：征战之神在库里只存了 Lv2（绿框）一份模板，遇到黄框的 Lv5 时，
    颜色直方图（会连菱形外框一起算）给它排到第 31 名，被 top-6 预筛淘汰，
    NCC 一次都没算——``read`` 于是返回「不像」。放开评估后它是 0.699，
    阈值 0.68，亚军才 0.428。正确答案一直在库里，是预筛把证据丢了。

    这里用「换框色 + 同色无结构干扰项」精确复现那个局面：干扰项和查询图
    像素完全相同（直方图相似度 1.0），真家族的模板却换了框色（直方图很低）。
    修复前真家族排在最后、必被淘汰；修复后它照样以大幅优势胜出。
    """

    def _crowded_bank(self) -> paint_kind.KindTemplateBank:
        queen = self.cell(tuple(self.spec["families"]["queen"]["cells"][0]))
        bank = paint_kind.KindTemplateBank()
        bank.add("queen", _recolor(queen, 20))
        for index in range(8):
            bank.add(f"filler{index}", _scramble(queen, index))
        return bank

    def test_family_is_never_dropped_by_the_colour_prefilter(self) -> None:
        bank = self._crowded_bank()
        queen = self.cell(tuple(self.spec["families"]["queen"]["cells"][0]))

        reading = bank.read(queen)

        self.assertGreater(len(bank.kinds()), paint_kind.KIND_PREFILTER_TOP_K, "家族数要超过预筛配额")
        self.assertEqual(
            set(bank.kinds()),
            set(reading.scores),
            "每个家族都该拿到一个分数，预筛不许整族淘汰",
        )

    def test_true_family_still_wins_after_being_prefiltered_out(self) -> None:
        """真家族直方图垫底，但图案证据足以让它反超所有同色干扰项。"""
        bank = self._crowded_bank()
        queen = self.cell(tuple(self.spec["families"]["queen"]["cells"][0]))

        reading = bank.read(queen)
        fillers = {name: score for name, score in reading.scores.items() if name.startswith("filler")}

        self.assertEqual(8, len(fillers), "干扰项也得拿到分数，否则这个对比是空的")
        self.assertGreater(
            reading.scores["queen"],
            max(fillers.values()),
            "同色干扰项只有颜色没有图案，不该压过真家族",
        )


class PersistenceTest(PaintKindFixture):
    def test_round_trip(self) -> None:
        bank = paint_kind.KindTemplateBank()
        for name, family in self.spec["families"].items():
            bank.add(name, self.cell(tuple(family["cells"][0])))
        before = {name: bank.read(self.cell(tuple(family["cells"][0]))).score
                  for name, family in self.spec["families"].items()}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "templates.npz"
            sidecar = Path(directory) / "labels.json"
            bank.save(target, sidecar)
            loaded = paint_kind.KindTemplateBank.load(target, sidecar)
        self.assertEqual(sorted(loaded.kinds()), sorted(bank.kinds()))
        self.assertEqual(len(loaded), len(bank))
        for name, family in self.spec["families"].items():
            reading = loaded.read(self.cell(tuple(family["cells"][0])))
            self.assertEqual(reading.subtype, name)
            self.assertAlmostEqual(reading.score, before[name], places=4)

    def test_empty_bank_returns_nothing(self) -> None:
        bank = paint_kind.KindTemplateBank()
        reading = bank.read(self.cell((0, 2)))
        self.assertFalse(reading.present)
        self.assertIsNone(reading.subtype)

    def test_capacity_cap(self) -> None:
        bank = paint_kind.KindTemplateBank()
        base = self.cell(tuple(self.spec["families"]["ice"]["cells"][0]))
        # 三个两两相似度都低于去重阈值（+40/+80 之间是 0.99，会被当副本挡掉）
        for cell in (base, _recolor(base, 40), _recolor(base, 120)):
            self.assertTrue(bank.add("ice", cell))
        # 第 4 个换成另一族的格子：这里只测配额，不看语义
        extra = self.cell(tuple(self.spec["families"]["queen"]["cells"][0]))
        self.assertFalse(bank.add("ice", extra), "库满之后不该再收")

    def test_duplicate_sample_is_stored_once(self) -> None:
        """棋盘上同族出现 3 格时，harvest 会拿同一张图案来加 3 次。"""
        bank = paint_kind.KindTemplateBank()
        cell = self.cell(tuple(self.spec["families"]["ice"]["cells"][0]))

        self.assertTrue(bank.add("ice", cell))
        self.assertFalse(bank.add("ice", cell), "副本不该再占一个配额")
        self.assertEqual(1, len(bank.templates["ice"]))


class PanelMatchTest(PaintKindFixture):
    def test_panel_icon_maps_back_to_board_cell(self) -> None:
        """「当前颜料」面板图标要能找回棋盘上对应的那一格。"""
        panel = self.spec["panel"]
        image = _imread(ROOT / panel["image"])
        cells = _cells(image, panel["rect"])
        x, y, w, h = panel["icon"]
        icon = image[y : y + h, x : x + w]
        ordered = [cells[(row, col)] for row in range(6) for col in range(6)]
        matched = paint_kind.match_panel_icon(icon, ordered)
        self.assertIsNotNone(matched)
        index, score, margin = matched  # type: ignore[misc]
        expected = panel["cell"][0] * 6 + panel["cell"][1]
        self.assertEqual(index, expected)
        self.assertGreaterEqual(score, paint_kind.LEARN_MIN_SCORE)
        self.assertGreaterEqual(margin, paint_kind.LEARN_MIN_MARGIN)

    def test_unknown_subtype_is_not_learned(self) -> None:
        bank = paint_kind.KindTemplateBank()
        panel = self.spec["panel"]
        image = _imread(ROOT / panel["image"])
        cells = _cells(image, panel["rect"])
        x, y, w, h = panel["icon"]
        icon = image[y : y + h, x : x + w]
        ordered = [cells[(row, col)] for row in range(6) for col in range(6)]
        self.assertFalse(paint_kind.learn_from_panel(icon, "根本不存在的颜料", ordered, active=bank))
        self.assertFalse(paint_kind.learn_from_panel(icon, None, ordered, active=bank))
        self.assertEqual(len(bank), 0)


class CatalogTest(unittest.TestCase):
    def test_family_key_round_trip(self) -> None:
        self.assertEqual(paint_kind.family_key("鼠人"), "m001")
        self.assertEqual(paint_kind.family_key("邪影"), "m012")
        self.assertEqual(paint_kind.family_key("机械之神"), "g004")
        # 赛季是独立家族（只有 Lv3/Lv4），在目录里编为 g015。
        self.assertEqual(paint_kind.family_key("赛季"), "g015")
        self.assertIsNone(paint_kind.family_key("不存在的家族"))
        self.assertIsNone(paint_kind.family_key(None))

    def test_category_of(self) -> None:
        self.assertEqual(paint_kind.category_of("鼠人"), "monster")
        self.assertEqual(paint_kind.category_of("宝箱"), "gameplay")
        self.assertEqual(paint_kind.category_of("赛季"), "gameplay")
        self.assertIsNone(paint_kind.category_of("不存在的家族"))

    def test_catalog_has_all_twenty_six(self) -> None:
        """目录 26 个家族（13 怪物 + 13 玩法），再加赛季这个表外家族。"""
        names = paint_kind.known_kinds()
        self.assertEqual(len(names), 27)
        self.assertEqual(len(set(names)), 27)
        self.assertIn(paint_kind.SEASON_PLACEHOLDER, names)

    def test_season_family_is_standalone(self) -> None:
        """赛季是独立家族：只有 Lv3/Lv4，图标是没被 PlotItem 引用的 ColorPlay15。"""
        entry = paint_kind.catalog()["g015"]
        self.assertEqual(entry["name"], paint_kind.SEASON_PLACEHOLDER)
        self.assertEqual(entry["icon"], paint_kind.SEASON_ICON)
        self.assertEqual(sorted(entry["levels"]), ["3", "4"])
        self.assertEqual(entry["category"], "gameplay")
        # 其余玩法家族都得有 Lv1 或 Lv5，不能也叫赛季。
        others = [e["name"] for k, e in paint_kind.catalog().items() if k != "g015"]
        self.assertNotIn(paint_kind.SEASON_PLACEHOLDER, others)


class SubtypeCodeTest(unittest.TestCase):
    def test_every_family_has_a_unique_code(self) -> None:
        codes = list(codebook._SUBTYPE_INITIALS.values())
        self.assertEqual(len(codes), 27)
        self.assertEqual(len(set(codes)), 27)
        self.assertEqual(set(codebook._SUBTYPE_INITIALS), set(paint_kind.known_kinds()))

    def test_paint_id(self) -> None:
        self.assertEqual(codebook.paint_id("monster", 1, "鼠人"), "sr1")
        self.assertEqual(codebook.paint_id("monster", 3, "邪影"), "xy3")
        self.assertEqual(codebook.paint_id("gameplay", 5, "机械之神"), "ji5")
        self.assertEqual(codebook.paint_id("monster", 2, "幽灵"), "yo2")
        self.assertEqual(codebook.paint_id("monster", 2, "英灵"), "yl2")

    def test_season_code(self) -> None:
        """赛季是独立家族（只有 Lv3/Lv4），照常出两字母编码，不退回 G3。"""
        self.assertEqual(codebook.paint_id("gameplay", 3, "赛季"), "sj3")
        self.assertEqual(codebook.paint_id("gameplay", 4, "赛季"), "sj4")

    def test_unknown_subtype_falls_back(self) -> None:
        self.assertEqual(codebook.paint_id("monster", 2, None), "M2")
        self.assertEqual(codebook.paint_id(None, 2, None), "U2")
        self.assertEqual(codebook.paint_id("monster", 2, "不存在的家族"), "M2")

    def test_treasure_keeps_old_code(self) -> None:
        self.assertEqual(codebook.paint_id(None, 2, None, is_treasure=True), "T2")


if __name__ == "__main__":
    unittest.main()
