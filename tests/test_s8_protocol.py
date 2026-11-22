"""``s8_protocol`` 契约测试：枚举值、等级公式、资源余量、笔触家族、难度画像。

这些断言的"真值"全部来自服务端 .proto 导出的描述符
（``proto_pb_all_20260715`` 的 ``gameplays8_data.pb`` / ``gameplays8_api.pb``），
不是从客户端常量池反推的，所以任何一条改动都必须同步改本文件并说明出处。
"""

from __future__ import annotations

import unittest

from huahua_assistant import s8_protocol as proto


class PlotTypeTests(unittest.TestCase):
    def test_enum_values_match_server_descriptor(self):
        self.assertEqual(2001, int(proto.S8PlotType.FREE))
        self.assertEqual(2002, int(proto.S8PlotType.NIHILITY))
        self.assertEqual(2003, int(proto.S8PlotType.RUINS))
        self.assertEqual(3001, int(proto.S8PlotType.INTERNAL_BLOCK))
        self.assertEqual(3002, int(proto.S8PlotType.BORDER_BLOCK))
        self.assertEqual(0, int(proto.S8PlotType.NONE))

    def test_2001_and_3001_are_distinct(self):
        """悬案关闭：2001 空白 ≠ 3001 内部阻挡，是两套东西。"""
        self.assertNotEqual(proto.S8PlotType.FREE, proto.S8PlotType.INTERNAL_BLOCK)

    def test_placeable_and_blocking(self):
        self.assertTrue(proto.S8PlotType.FREE.is_placeable)
        for blocked in (
            proto.S8PlotType.NIHILITY,
            proto.S8PlotType.RUINS,
            proto.S8PlotType.INTERNAL_BLOCK,
            proto.S8PlotType.BORDER_BLOCK,
        ):
            self.assertFalse(blocked.is_placeable)
        # 只有「阻挡」两种能被笔触/拆下改变语义；虚空与失色是另一类，不算阻挡。
        self.assertTrue(proto.S8PlotType.INTERNAL_BLOCK.is_blocking)
        self.assertTrue(proto.S8PlotType.BORDER_BLOCK.is_blocking)
        self.assertFalse(proto.S8PlotType.FREE.is_blocking)
        self.assertFalse(proto.S8PlotType.NIHILITY.is_blocking)
        self.assertFalse(proto.S8PlotType.RUINS.is_blocking)


class OtherEnumTests(unittest.TestCase):
    def test_game_state(self):
        self.assertEqual(8, int(proto.S8GameState.REWARD))
        self.assertEqual(5, int(proto.S8GameState.DRAW_TILE))

    def test_effect_from_type(self):
        self.assertEqual(1, int(proto.S8EffectFromType.ITEM))
        self.assertEqual(2, int(proto.S8EffectFromType.SELECTABLE))
        self.assertEqual(3, int(proto.S8EffectFromType.REVIVE))

    def test_selectable_lib_type(self):
        self.assertEqual(1, int(proto.S8SelectableLibType.TRANSFORM))
        self.assertEqual(2, int(proto.S8SelectableLibType.APPEND))
        self.assertEqual(3, int(proto.S8SelectableLibType.EXPAND))

    def test_changed_field(self):
        self.assertEqual(3, int(proto.S8ChangedField.DRAW_MAP))
        self.assertEqual(4, int(proto.S8ChangedField.OPEN_MAP))


class FinalLevelTests(unittest.TestCase):
    def test_raw_plus_facade(self):
        """Lv2 颜料 + 地块加成 3 → 5（未到上限）。"""
        self.assertEqual(5, proto.final_level(2, 3))

    def test_capped_by_max_plot_level(self):
        """2 + 2 + 笔触额外 3 的实机场景：7 封顶。"""
        self.assertEqual(7, proto.final_level(4, 3))
        self.assertEqual(7, proto.final_level(7, 5))

    def test_cap_is_hardcoded_seven(self):
        """最高 7 级写死：加成再大也只到 7，没有"调上限"的口子。"""
        self.assertEqual(7, proto.MAX_PLOT_LEVEL)
        self.assertEqual(7, proto.final_level(1, 99))
        self.assertEqual(7, proto.final_level(6, 6))
        with self.assertRaises(TypeError):
            proto.final_level(4, 4, max_plot_level=9)  # type: ignore[call-arg]

    def test_empty_plot_is_zero(self):
        self.assertEqual(0, proto.final_level(0, 6))

    def test_negative_inputs_clamped(self):
        self.assertEqual(2, proto.final_level(2, -5))
        self.assertEqual(0, proto.final_level(-3, 5))

    def test_plot_item_level_uses_shared_facade(self):
        item = proto.S8Item(unique_id="u1", item_id=200101, raw_level=4)
        plot = proto.S8Plot(
            no=7,
            raw_plot_type=int(proto.S8PlotType.FREE),
            items=(item,),
            facade_level=2,
        )
        self.assertEqual(6, proto.plot_item_level(item, plot))
        self.assertEqual((6,), plot.item_levels())


class ResourceTests(unittest.TestCase):
    def test_reroll_accounting(self):
        res = proto.S8Resources(remain_reroll=1, max_reroll=2)
        self.assertEqual(1, res.reroll_left)
        self.assertEqual(1, res.reroll_used)
        self.assertTrue(res.can_reroll)

    def test_revive_accounting(self):
        res = proto.S8Resources(remain_revive=0, max_revive=3)
        self.assertEqual(0, res.revive_left)
        self.assertEqual(3, res.revive_used)
        self.assertFalse(res.can_revive)

    def test_uninstall_requires_enable(self):
        self.assertFalse(proto.S8Resources(used_uninstall=0).can_uninstall)
        self.assertTrue(proto.S8Resources(enable_uninstall=True).can_uninstall)

    def test_resources_of_dict_like_message(self):
        class Msg:
            remain_reroll = 2
            max_reroll_times = 5
            remain_revive = 1
            max_revive_times = 1
            used_uninstall = 4
            enable_uninstall = True
            revive_ids = (1001, 2001)

        res = proto.resources_of(Msg())
        self.assertEqual(2, res.remain_reroll)
        self.assertEqual(5, res.max_reroll)
        self.assertEqual(3, res.reroll_used)
        self.assertEqual((1001, 2001), res.revive_ids)
        self.assertTrue(res.can_uninstall)

    def test_resources_of_partial_push_defaults_to_zero(self):
        res = proto.resources_of(object())
        self.assertEqual(0, res.remain_reroll)
        self.assertEqual(0, res.max_reroll)
        self.assertEqual((), res.revive_ids)
        self.assertFalse(res.can_uninstall)


class AffixFamilyTests(unittest.TestCase):
    def test_prefix_families(self):
        self.assertEqual("转锋", proto.affix_family(1001))
        self.assertEqual("堆砌", proto.affix_family(2002))
        self.assertEqual("扩散", proto.affix_family(3005))

    def test_variant_family_needs_catalog(self):
        """4xxx 是权重变体，家族由记录本身决定，不能按千位硬判。

        官方表里 4001~4005 是**转锋**、4006/4007 是**堆砌**——同一个千位下
        两种家族都有，所以只能查表。
        """
        self.assertEqual("转锋", proto.affix_family(4005))
        self.assertEqual("堆砌", proto.affix_family(4006))
        self.assertIsNone(proto.affix_family(4999))

    def test_lib_type_mapping(self):
        self.assertEqual(proto.S8SelectableLibType.TRANSFORM, proto.affix_lib_type(1001))
        self.assertEqual(proto.S8SelectableLibType.APPEND, proto.affix_lib_type(2001))
        self.assertEqual(proto.S8SelectableLibType.EXPAND, proto.affix_lib_type(3001))
        self.assertEqual(proto.S8SelectableLibType.NONE, proto.affix_lib_type(0))


class MapDataTests(unittest.TestCase):
    def test_defaults(self):
        data = proto.S8DrawMapData()
        self.assertEqual(0, data.board_size)
        self.assertEqual(proto.MAX_PLOT_LEVEL, data.max_plot_level)
        self.assertIsNone(data.initial_paint())

    def test_board_size_is_plot_count(self):
        plots = tuple(
            proto.S8Plot(no=index, raw_plot_type=int(proto.S8PlotType.FREE))
            for index in range(36)
        )
        data = proto.S8DrawMapData(plots=plots)
        self.assertEqual(36, data.board_size)
        self.assertIsNotNone(data.plot(35))
        self.assertIsNone(data.plot(36))

    def test_initial_paint_is_first_paint_on_board(self):
        plots = (
            proto.S8Plot(no=0, raw_plot_type=int(proto.S8PlotType.NIHILITY)),
            proto.S8Plot(
                no=1,
                raw_plot_type=int(proto.S8PlotType.FREE),
                items=(proto.S8Item(unique_id="u9", item_id=100301, raw_level=3),),
            ),
        )
        data = proto.S8DrawMapData(plots=plots)
        self.assertEqual("u9", data.initial_paint().unique_id)

    def test_brush_effect_id_reads_task(self):
        item = proto.S8Item(
            unique_id="u1",
            item_id=100301,
            raw_level=3,
            task=proto.S8Task(effect=proto.S8Effect(effect_id=77, selectable_effect_id=2005)),
        )
        self.assertEqual(2005, item.brush_effect_id)
        self.assertEqual(0, proto.S8Item().brush_effect_id)


class DifficultyProfileTests(unittest.TestCase):
    def test_practice_uses_max_difficulty(self):
        self.assertEqual(7, proto.PRACTICE_DIFFICULTY)
        self.assertEqual(7, proto.MAX_DIFFICULTY_PROFILE.difficulty)
        self.assertEqual("绮彩雪纸", proto.MAX_DIFFICULTY_PROFILE.name)

    def test_unknown_values_stay_none(self):
        """服务端数值不在客户端 pak 里：未知必须是 None，不是 0。"""
        profile = proto.difficulty_profile(7)
        self.assertIsNone(profile.block_min)
        self.assertIsNone(profile.block_max)
        self.assertIsNone(profile.init_max_level)
        self.assertFalse(profile.block_range_known)
        self.assertIn("未取到", profile.source)

    def test_unassigned_weight_strings_kept_verbatim(self):
        self.assertEqual(2, len(proto.UNASSIGNED_WEIGHT_STRINGS))
        self.assertTrue(proto.UNASSIGNED_WEIGHT_STRINGS[0].startswith("5:1|9;15:10|19"))
        self.assertEqual(
            proto.UNASSIGNED_WEIGHT_STRINGS, proto.difficulty_profile(7).plot_interval_weights
        )

    def test_teaching_difficulty_name(self):
        self.assertEqual("无色雪纸", proto.difficulty_profile(101).name)

    def test_unknown_difficulty_raises(self):
        with self.assertRaises(KeyError):
            proto.difficulty_profile(999)


if __name__ == "__main__":
    unittest.main()
