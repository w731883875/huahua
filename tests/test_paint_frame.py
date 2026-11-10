"""「框」这层的回归测试：14 张官方框素材 + 虚空形态。

锁死四件事：

1. 7 圆形 + 7 菱形、白绿蓝紫橙红彩，一张不缺，文件名和清单对得上；
2. 类别 → 形状的映射是「怪物=圆形、玩法=菱形」，这是解包资源里的硬规则；
3. 虚空形态（放进虚空、只剩等级框）能认出来，而且**等级照读**；
4. 86 个带真值的普通颜料格一个都不能被误判成虚空——阈值两边的余量是 26 倍，
   这条测试就是防止以后有人把阈值调松。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from huahua_assistant import paint_frame, vision
from huahua_assistant.capture import read_image

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_ROOT = ROOT / "jietu"
COVERAGE = ROOT / "runtime" / "paint_coverage.json"

#: 实机里的虚空格：Lv3 蓝环 + 空中心，家族图案已被抹掉。
VOID_CELL = ("斜对角.png", 3, 1)


def _cells(image, scene) -> dict[tuple[int, int], np.ndarray]:
    result: dict[tuple[int, int], np.ndarray] = {}
    for tile in scene.board.tiles:
        rect = tile.rect
        result[(tile.row, tile.col)] = image[rect.y : rect.bottom, rect.x : rect.right]
    return result


class FrameAssetTests(unittest.TestCase):
    def test_all_fourteen_frames_exist(self) -> None:
        found = paint_frame.frames()

        self.assertEqual(14, len(found))
        for (shape, level), asset in found.items():
            with self.subTest(shape=shape, level=level):
                self.assertTrue(asset.path.exists(), asset.path)
                self.assertEqual(shape, asset.shape)
                self.assertEqual(level, asset.level)
                self.assertEqual(paint_frame.LEVEL_COLOR[level], asset.path.stem.split("_")[-1])

    def test_manifest_records_where_each_frame_came_from(self) -> None:
        raw = json.loads(paint_frame.MANIFEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual(14, len(raw))
        for name, entry in raw.items():
            with self.subTest(frame=name):
                self.assertEqual(name, f"{entry['shape']}_Lv{entry['level']}_{paint_frame.LEVEL_COLOR[entry['level']]}.png")
                x, y, w, h = entry["source"]
                self.assertGreater(w, 150)
                self.assertGreater(h, 150)
                self.assertGreater(x, 0)
                self.assertGreater(y, 0)

    def test_frame_path_is_the_single_source_of_truth(self) -> None:
        for (shape, level), asset in paint_frame.frames().items():
            self.assertEqual(asset.path, paint_frame.frame_path(shape, level))

    def test_frame_path_rejects_nonsense(self) -> None:
        with self.assertRaises(ValueError):
            paint_frame.frame_path("triangle", 3)
        with self.assertRaises(ValueError):
            paint_frame.frame_path("circle", 8)

    def test_shape_follows_category(self) -> None:
        self.assertEqual("circle", paint_frame.shape_of_category("monster"))
        self.assertEqual("diamond", paint_frame.shape_of_category("gameplay"))
        self.assertIsNone(paint_frame.shape_of_category(None))
        self.assertIsNone(paint_frame.shape_of_category("unknown"))
        self.assertEqual({"monster": "circle", "gameplay": "diamond"}, paint_frame.SHAPE_OF_CATEGORY)
        self.assertEqual({"circle": "monster", "diamond": "gameplay"}, paint_frame.CATEGORY_OF_SHAPE)

    def test_scale_comes_from_the_real_match_not_from_texture_sizes(self) -> None:
        """两个比例都是实机量出来的，不能退回「贴图尺寸相除」。

        早期按 ``128/200 = 0.64`` 推，错在两头：文件名 ``_128`` 只是逻辑尺寸
        （实际贴图 256×256），框的 200 是切图外接框。
        真正错的是**框没放大**——素材实心环只占画布 0.74，而实机里环贴着格子边。
        """
        self.assertGreater(paint_frame.FRAME_SCALE, 1.15, "框必须放大，否则相对看就是图标过大")
        self.assertLess(paint_frame.FRAME_SCALE, 1.40)
        self.assertGreater(paint_frame.ICON_SCALE, 0.50)
        self.assertLess(paint_frame.ICON_SCALE, 0.70)
        # 合成出来的两个量要分别对上实机，这是「还原游戏内设置」的硬判据
        art_of_cell = paint_frame.FRAME_SCALE * paint_frame.ICON_SCALE * paint_frame.ICON_ART_OF_CANVAS
        self.assertAlmostEqual(paint_frame.TARGET_ART_OF_CELL, art_of_cell, delta=0.02)
        frame_of_cell = paint_frame.FRAME_SCALE * paint_frame.FRAME_RING_OF_CANVAS
        self.assertAlmostEqual(paint_frame.TARGET_FRAME_OF_CELL, frame_of_cell, delta=0.03)

    def test_backpack_and_board_share_one_shape_rule(self) -> None:
        """背包和地块用同一套形状规则：怪物圆形、玩法菱形，不按场景区分。"""
        self.assertFalse(hasattr(paint_frame, "BACKPACK_SHAPE"), "形状不该有「背包专用」特例")
        self.assertEqual("circle", paint_frame.shape_of_category("monster"))
        self.assertEqual("diamond", paint_frame.shape_of_category("gameplay"))


class VoidedTileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cache: dict[str, tuple[np.ndarray, object]] = {}

    def board(self, name: str):
        if name not in self.cache:
            image = read_image(SCREENSHOT_ROOT / name)
            scene = vision.analyze_scene_realtime(image)
            if scene is None or scene.board is None:
                raise AssertionError(f"{name} 分析失败，棋盘布局变了")
            self.cache[name] = (image, scene)
        return self.cache[name]

    def test_void_cell_is_flat(self) -> None:
        name, row, col = VOID_CELL
        image, scene = self.board(name)
        cell = _cells(image, scene)[(row, col)]

        stats = paint_frame.void_center_stats(cell)
        self.assertIsNotNone(stats)
        self.assertTrue(paint_frame.looks_voided(cell), f"中心拉普拉斯方差 {stats[0]:.1f}")

    def test_void_cell_keeps_its_level(self) -> None:
        """图案没了，但框还在——等级必须照读，否则这块地的价值就丢了。"""
        name, row, col = VOID_CELL
        _image, scene = self.board(name)
        tile = next(item for item in scene.board.tiles if (item.row, item.col) == (row, col))

        self.assertEqual(paint_frame.VOIDED, tile.paint_subtype)
        self.assertEqual(3, tile.paint_level)
        self.assertGreaterEqual(tile.paint_level_confidence, 0.70)

    def test_real_paint_tiles_are_never_voided(self) -> None:
        ledger = json.loads(COVERAGE.read_text(encoding="utf-8"))["samples"]
        checked = 0
        for family, levels in ledger.items():
            for level, items in levels.items():
                for name, row, col in items:
                    with self.subTest(family=family, level=level, image=name, cell=(row, col)):
                        image, scene = self.board(name)
                        tile = next(
                            item for item in scene.board.tiles if (item.row, item.col) == (row, col)
                        )
                        self.assertNotEqual(paint_frame.VOIDED, tile.paint_subtype)
                        checked += 1
        self.assertEqual(86, checked, "台账里的真值格数变了，这条测试的前提要重算")

    def test_blank_tile_is_not_voided_paint(self) -> None:
        """空白格中心一样平，但没有 LvN 角标，不能走颜料这条路。"""
        name = "重新获取随机调色颜料之前.png"
        image, scene = self.board(name)
        cell = _cells(image, scene)[(0, 0)]

        self.assertTrue(paint_frame.looks_voided(cell), "空白格中心确实是平的")
        tile = next(item for item in scene.board.tiles if (item.row, item.col) == (0, 0))
        self.assertNotEqual(paint_frame.VOIDED, tile.paint_subtype)

    def test_degenerate_input_does_not_raise(self) -> None:
        for image in (
            np.zeros((0, 0, 3), dtype=np.uint8),
            np.zeros((6, 6, 3), dtype=np.uint8),
            np.full((40, 40, 3), 255, dtype=np.uint8),
        ):
            with self.subTest(shape=image.shape):
                self.assertIsInstance(paint_frame.looks_voided(image), bool)


if __name__ == "__main__":
    unittest.main()
