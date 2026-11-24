#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""家族与等级绑定规则的回归测试。

用户 2026-11-24 原话：

> 1级迷城是不存在，玩法赛季都是从3-4级赛季颜料在升级到5级才会变成其他玩法颜料，
> 例如迷城、梦魇等最低都是5级！！

权威数据是 ``s8_official_data.S8_PARTIAL_LEVEL_FAMILIES``：

    赛季 3~4
    黑潮 / 黑帆 / 魔方 / 迷城 / 梦魇 / 怪谈 / 玩偶 5~7

**关键是不能按类别一刀切**。``PaintCategory.GAMEPLAY`` 里混着两类完全不同的
家族：宝箱/存活/机械之神等 1~7 级通吃，赛季只有 3~4 级，而迷城等才是 5~7 级。
第一版实现写成「GAMEPLAY 一律 >= 5」，会把**正常的赛季 Lv3/Lv4、宝箱 Lv1**
判成「不可能」并抹掉等级——比原来的识别错误更糟。

所以判据必须落到**具体家族**上，查不到家族时不敢判。
"""

from __future__ import annotations

import unittest

from huahua_assistant import s8_official_data


class PartialLevelFamilyTest(unittest.TestCase):
    """官方表本身：哪些家族被等级段绑定。"""

    def test_table_covers_expected_families(self) -> None:
        table = s8_official_data.S8_PARTIAL_LEVEL_FAMILIES
        self.assertEqual(sorted(table), sorted(
            ["赛季", "黑潮", "黑帆", "魔方", "迷城", "梦魇", "怪谈", "玩偶"]
        ))

    def test_season_paint_is_level_3_to_4(self) -> None:
        # 赛季颜料只有 3~4 级；升到 5 级才变成迷城/梦魇等玩法颜料。
        self.assertEqual(s8_official_data.S8_PARTIAL_LEVEL_FAMILIES["赛季"], [3, 4])

    def test_gameplay_paints_are_level_5_to_7(self) -> None:
        table = s8_official_data.S8_PARTIAL_LEVEL_FAMILIES
        for name in ("黑潮", "黑帆", "魔方", "迷城", "梦魇", "怪谈", "玩偶"):
            self.assertEqual(table[name], [5, 6, 7], name)

    def test_unbound_families_allow_all_levels(self) -> None:
        # 宝箱/存活/各神是 GAMEPLAY 但 1~7 级通吃，不能被当成受绑家族。
        for name in ("宝箱", "存活", "机械之神", "巨力之神", "狩猎之神", "征战之神"):
            self.assertNotIn(name, s8_official_data.S8_PARTIAL_LEVEL_FAMILIES)
            self.assertEqual(
                s8_official_data.S8_PAINT_FAMILIES[name]["levels"],
                [1, 2, 3, 4, 5, 6, 7],
                name,
            )


class ConstraintScopeTest(unittest.TestCase):
    """约束的作用范围：只作废「该家族不存在的等级」。"""

    @staticmethod
    def _verdict(family: str, level: int) -> bool:
        """复刻 vision 里的判据：返回 True 表示这个等级合法（放行）。"""
        allowed = s8_official_data.S8_PARTIAL_LEVEL_FAMILIES.get(family)
        return allowed is None or level in allowed

    def test_season_paint_low_levels_survive(self) -> None:
        # 这是第一版按类别一刀切会破坏的正常数据。
        self.assertTrue(self._verdict("赛季", 3))
        self.assertTrue(self._verdict("赛季", 4))
        # 赛季不存在 5 级以上。
        self.assertFalse(self._verdict("赛季", 5))

    def test_normal_gameplay_families_survive_low_levels(self) -> None:
        # 宝箱/存活在 1 级完全合法，绝不能被抹掉。
        self.assertTrue(self._verdict("宝箱", 1))
        self.assertTrue(self._verdict("宝箱", 4))
        self.assertTrue(self._verdict("存活", 1))

    def test_monster_families_unaffected(self) -> None:
        for family in ("鼠人", "幽灵", "矮人", "沙怪"):
            for level in range(1, 8):
                self.assertTrue(self._verdict(family, level), f"{family} Lv{level}")

    def test_impossible_combinations_rejected(self) -> None:
        # 用户点名的「1 级迷城」以及同类。
        self.assertFalse(self._verdict("迷城", 1))
        self.assertFalse(self._verdict("梦魇", 4))
        self.assertFalse(self._verdict("黑帆", 2))
        # 合法的高等级照常放行。
        self.assertTrue(self._verdict("迷城", 5))
        self.assertTrue(self._verdict("梦魇", 7))


if __name__ == "__main__":
    unittest.main()
