from __future__ import annotations

import unittest

import numpy as np

from huahua_assistant import tile_art


class TileArtMappingTests(unittest.TestCase):
    def test_family_maps_to_official_icon(self) -> None:
        # 图标名只在 paint_catalog.json 写一次，这里锁住「家族 → 图标」的对应关系。
        self.assertEqual("UI_S8Gameplay_ColorPlay1_128", tile_art.icon_stem("宝箱"))
        self.assertEqual("UI_S8Gameplay_ColorMonster13_128", tile_art.icon_stem("英灵"))
        self.assertEqual("UI_S8Gameplay_ColorPlay4_128", tile_art.icon_stem("机械之神"))

    def test_non_family_subtypes_have_no_icon(self) -> None:
        # 虚空是状态不是家族；随便编的名字更不该被认成某个图标。
        self.assertIsNone(tile_art.icon_stem("虚空"))
        self.assertIsNone(tile_art.icon_stem("并不存在的家族"))
        self.assertIsNone(tile_art.icon_stem(None))

    def test_frame_shape_follows_category(self) -> None:
        self.assertEqual("circle", tile_art.frame_shape("monster"))
        self.assertEqual("diamond", tile_art.frame_shape("gameplay"))
        self.assertIsNone(tile_art.frame_shape("unknown"))
        self.assertIsNone(tile_art.frame_shape(None))


class TileArtComposeTests(unittest.TestCase):
    def setUp(self) -> None:
        tile_art.clear_cache()

    def test_composes_square_rgba_with_painted_center(self) -> None:
        image = tile_art.compose_icon("gameplay", "宝箱", 5, size=88)

        self.assertIsNotNone(image)
        assert image is not None
        self.assertEqual((88, 88), image.size)
        self.assertEqual("RGBA", image.mode)
        alpha = np.array(image)[:, :, 3]
        # 中心必须真有图：透明的话界面上就是「有框没图案」，和虚空分不开了。
        self.assertGreater(float(alpha[30:58, 30:58].mean()), 200.0)

    def test_voided_only_draws_frame(self) -> None:
        normal = tile_art.compose_icon("gameplay", "宝箱", 5, size=88)
        voided = tile_art.compose_icon("gameplay", "宝箱", 5, size=88, voided=True)

        self.assertIsNotNone(voided)
        assert normal is not None and voided is not None
        normal_center = float(np.array(normal)[30:58, 30:58, 3].mean())
        voided_center = float(np.array(voided)[30:58, 30:58, 3].mean())
        # 虚空 = 图案被抹掉，中心应该几乎透明；只留一圈框。
        self.assertLess(voided_center, 60.0)
        self.assertGreater(normal_center - voided_center, 150.0)

    def test_unknown_level_still_draws_icon_without_quality_frame(self) -> None:
        # 等级没读到时宁可不画框，也不能挑个颜色冒充等级。
        image = tile_art.compose_icon("monster", "英灵", None, size=64)

        self.assertIsNotNone(image)
        assert image is not None
        alpha = np.array(image)[:, :, 3]
        self.assertGreater(float(alpha.mean()), 1.0)
        # 只有图案时四周角落是空的（框才会铺到角上）。
        self.assertLess(float(alpha[:4, :4].mean()), 10.0)

    def test_unknown_family_and_category_returns_none(self) -> None:
        self.assertIsNone(tile_art.compose_icon(None, None, None, size=64))
        self.assertIsNone(tile_art.compose_icon("unknown", "并不存在的家族", 3, size=64))

    def test_category_can_be_recovered_from_catalog(self) -> None:
        # 视觉层只给了家族名、类别没认出来时，目录能补出形状。
        image = tile_art.compose_icon(None, "英灵", 4, size=72)
        self.assertIsNotNone(image)
        assert image is not None
        self.assertGreater(float(np.array(image)[:, :, 3].max()), 200.0)

    def test_same_request_is_cached(self) -> None:
        first = tile_art.compose_icon("gameplay", "宝箱", 5, size=88)
        second = tile_art.compose_icon("gameplay", "宝箱", 5, size=88)
        other_size = tile_art.compose_icon("gameplay", "宝箱", 5, size=96)

        self.assertIs(first, second)
        self.assertIsNot(first, other_size)

    def test_too_small_size_is_rejected(self) -> None:
        self.assertIsNone(tile_art.compose_icon("gameplay", "宝箱", 5, size=4))


if __name__ == "__main__":
    unittest.main()
