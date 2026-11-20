"""「用官方图标认家族」这层的回归测试。

这层解决的是模板库的盲区：模板只能认「见过并入库」的家族（实机截图一局才攒几个），
而官方资源里有 27 张家族图标，一个家族一张图、7 个等级共用——等于一份天然标准照。

锁死这几件事：

1. 图标 ↔ 家族名必须和 ``paint_catalog.json`` 完全对齐（27 个，不多不少）。
2. 同一家族跨等级要认成同一个（外框只表示等级，中央图案不变）。
3. 空白格、阻挡地块这些「没有图案」的格子必须认不出，不能瞎报。
4. 一帧配额（格数 / 秒数）要真的能截断，否则整盘 36 格 × 26ms 会吃穿交互预算。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from huahua_assistant import paint_icon, paint_kind
from huahua_assistant.capture import read_image
from huahua_assistant.vision import analyze_scene_realtime

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_ROOT = ROOT / "jietu"


def _cell_rects(rect, size: int = 6) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """按布局模块的取整规则切 36 格（和 ``vision._classify_tiles`` 一致）。"""
    cell_w = rect.w / size
    cell_h = rect.h / size
    result: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for row in range(size):
        for col in range(size):
            x1 = round(rect.x + col * cell_w)
            y1 = round(rect.y + row * cell_h)
            x2 = round(rect.x + (col + 1) * cell_w)
            y2 = round(rect.y + (row + 1) * cell_h)
            result[(row, col)] = (x1, y1, x2, y2)
    return result


class IconGalleryTests(unittest.TestCase):
    """图标库本身：名字对照、描述子、加载。"""

    def test_icon_names_cover_every_catalog_family_with_an_icon(self) -> None:
        names = paint_icon.icon_names()

        families = {name for name in names.values() if name}
        self.assertEqual(27, len(names), f"图标数应为 27，实际 {len(names)}")
        self.assertIn("黑帮", families)
        self.assertIn("邪影", families)
        self.assertIn("宝箱", families)
        # 名字必须都是 catalog 里真实存在的家族，不能有拼错的野名。
        known = set(paint_kind.known_kinds())
        self.assertEqual(set(), families - known, f"图标里有 catalog 中不存在的家族: {families - known}")

    def test_every_icon_file_on_disk_has_a_family_name(self) -> None:
        names = paint_icon.icon_names()
        files = [path.stem for path in paint_icon.icon_files()]

        self.assertEqual(30, len(files), "assets/paint_icons 下应有 30 张解包图标")
        unmapped = [stem for stem in files if stem not in names]
        # 30 张里 27 张是棋盘家族，另外 3 张在 roles.json 里登记了非家族角色
        # （虚空形态 / 宠物特性 / 待确认），不该有第四种下落。
        self.assertEqual(set(paint_icon.special_roles()), set(unmapped))
        self.assertEqual([], [stem for stem in unmapped if stem not in paint_icon.special_roles()])

    def test_special_icons_never_enter_the_gallery(self) -> None:
        """虚空、宠物特性这些状态图标一旦进识别库，就会把状态报成家族。"""
        gallery = paint_icon.IconGallery(use_cache=False)
        names = paint_icon.icon_names()
        special = set(paint_icon.special_roles())

        self.assertFalse(special & set(names), "状态图标不许出现在家族对照表里")
        self.assertEqual(27, len(names))
        self.assertEqual(set(names.values()), set(gallery.families))

    def test_gallery_loads_descriptors_for_all_icons(self) -> None:
        loaded = paint_icon.IconGallery()

        self.assertGreaterEqual(len(loaded), 27, f"只载入 {len(loaded)} 个家族的描述子")
        for family, descriptors in loaded.families.items():
            self.assertIsNotNone(descriptors, family)
            self.assertGreater(len(descriptors), 0, family)
            self.assertEqual(32, descriptors.shape[1], f"{family} 的 ORB 描述子应为 32 字节")

    def test_descriptor_cache_roundtrips(self) -> None:
        """暖加载必须拿到和冷建一样的家族集合，否则缓存会让识别结果随启动次数漂移。"""
        built = paint_icon.IconGallery(use_cache=False)
        cached = paint_icon.IconGallery(use_cache=True)

        self.assertEqual(set(built.families), set(cached.families))
        for family, descriptors in built.families.items():
            np.testing.assert_array_equal(descriptors, cached.families[family])


class IconIdentifyTests(unittest.TestCase):
    """识别效果：对真实截图上的格子认家族。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.image = read_image(SCREENSHOT_ROOT / "重新获取随机调色颜料之前.png")
        scene = analyze_scene_realtime(cls.image)
        if scene is None:
            raise AssertionError("fixture 截图分析失败，棋盘布局变了")
        cls.rects = _cell_rects(scene.board.rect)

    def crop(self, row: int, col: int) -> np.ndarray:
        x1, y1, x2, y2 = self.rects[(row, col)]
        return self.image[y1:y2, x1:x2]

    def test_lv3_black_gang_tile_is_recognized(self) -> None:
        match = paint_icon.identify(self.crop(3, 2))

        self.assertIsNotNone(match, "Lv3 黑帮格应能靠图标认出来")
        self.assertEqual("黑帮", match.family)
        self.assertGreaterEqual(match.good, paint_icon.ICON_MIN_GOOD)
        self.assertGreaterEqual(match.margin, paint_icon.ICON_MIN_MARGIN)

    def test_lv4_evil_shadow_tile_is_recognized(self) -> None:
        match = paint_icon.identify(self.crop(3, 4))

        self.assertIsNotNone(match, "Lv4 邪影格应能靠图标认出来")
        self.assertEqual("邪影", match.family)

    def test_same_family_across_levels_maps_to_one_name(self) -> None:
        """黑帮是唯一在现有截图里同时出现过 Lv3 和 Lv6/Lv7 的家族。

        外框从金换到别的颜色，中央图案不变，所以必须还是同一个名字。
        """
        cases = [
            ("重新获取随机调色颜料之前.png", 3, 2, "黑帮"),
            ("扩散后选择前.png", 2, 5, "黑帮"),
            ("其他.png", 2, 5, "黑帮"),
        ]
        seen: dict[str, set[int]] = {}
        for filename, row, col, expected in cases:
            with self.subTest(filename=filename, cell=(row, col)):
                image = read_image(SCREENSHOT_ROOT / filename)
                scene = analyze_scene_realtime(image)
                self.assertIsNotNone(scene)
                x1, y1, x2, y2 = _cell_rects(scene.board.rect)[(row, col)]
                match = paint_icon.identify(image[y1:y2, x1:x2])
                self.assertIsNotNone(match, f"{filename} r{row}c{col} 没认出来")
                self.assertEqual(expected, match.family)
                seen.setdefault(match.family, set())

        self.assertEqual({"黑帮"}, set(seen), "三个等级都应归到同一个家族")

    def test_blank_tile_is_not_identified(self) -> None:
        """空格没有中央图案，宁可认不出，也不能报一个家族上去。"""
        blanks = [(0, 0), (0, 1), (5, 5)]
        for row, col in blanks:
            with self.subTest(cell=(row, col)):
                self.assertIsNone(paint_icon.identify(self.crop(row, col)))

    def test_degenerate_input_does_not_raise(self) -> None:
        for image in (
            np.zeros((0, 0, 3), dtype=np.uint8),
            np.zeros((40, 40, 3), dtype=np.uint8),
            np.full((40, 40, 3), 255, dtype=np.uint8),
        ):
            with self.subTest(shape=image.shape, value=int(image.mean()) if image.size else None):
                self.assertIsNone(paint_icon.identify(image))


class IconBudgetTests(unittest.TestCase):
    """一帧配额：格数上限、时间上限、以及 quick 路径的「整帧关闭」。"""

    def test_cell_quota_runs_out(self) -> None:
        budget = paint_icon.IconBudget(cells=3, seconds=10.0)

        budget.reset()
        self.assertEqual([True, True, True, False], [budget.take() for _ in range(4)])

        budget.reset()
        self.assertEqual([True, True, True, False], [budget.take() for _ in range(4)])

    def test_time_quota_runs_out(self) -> None:
        # 用负秒数而不是 0：Windows 上 ``monotonic`` 的分辨率约 15ms，
        # ``now > now + 0`` 在同一个 tick 里不一定成立。
        #
        # 注意计时口径：``reset()`` 之后 deadline 是 0（= **还没开表**），第一格
        # 落地才开始计时（见 ``IconBudget.reset`` 的注释）。所以第一次 ``take()``
        # 一定是放行的、负责开表，负秒数在第二次才立刻过期。
        budget = paint_icon.IconBudget(cells=100, seconds=-1.0)

        budget.reset()
        self.assertTrue(budget.take(), "第一次 take 只负责开表")
        self.assertFalse(budget.take(), "已过期的时间上限应立即截断")

    def test_reset_budget_can_disable_the_whole_frame(self) -> None:
        # analyze_quick 走这里：增量路径一帧要认的格子少，预算再宽也架不住帧率高。
        paint_icon.reset_budget(allow=False)
        self.assertIsNone(
            paint_icon.identify(np.zeros((40, 40, 3), dtype=np.uint8), budgeted=True),
            "allow=False 时必须立刻返回 None，连图标库都不该碰",
        )

        paint_icon.reset_budget(allow=True)
        budget = paint_icon.IconBudget(cells=1, seconds=10.0)
        budget.reset()
        self.assertTrue(budget.take())

    def test_budgeted_false_ignores_the_quota(self) -> None:
        """离线路径（analyze_scene）不受配额约束，否则批量采集会漏格。"""
        paint_icon.reset_budget(allow=False)
        try:
            gallery = paint_icon.gallery()
            self.assertGreaterEqual(len(gallery), 27)
        finally:
            paint_icon.reset_budget(allow=True)


class IconMatchTests(unittest.TestCase):
    def test_margin_survives_zero_runner_up(self) -> None:
        match = paint_icon.IconMatch(
            family="黑帮",
            share=0.25,
            good=50,
            runner_up=None,
            runner_good=0,
        )

        self.assertEqual(50.0, match.margin)

    def test_margin_is_the_leading_ratio(self) -> None:
        match = paint_icon.IconMatch(family="黑帮", share=0.25, good=60, runner_up="邪影", runner_good=20)

        self.assertAlmostEqual(3.0, match.margin)


if __name__ == "__main__":
    unittest.main()
