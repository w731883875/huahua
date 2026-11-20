from __future__ import annotations

import unittest
from collections import Counter

from huahua_assistant.codes import (
    affix_code,
    affix_name_from_code,
    brush_payload_codes,
    canonical_affix_id,
    compact_paint_with_code,
    compact_rule_with_code,
    resolve_affix,
)
from huahua_assistant.game_model import (
    BRUSH_TYPE_ORDER,
    BrushType,
    CurrentPaintRule,
    Paint,
    build_affix_catalog,
    normalize_brush_options,
)
from huahua_assistant.models import PaintCategory, PaintDirection, PaintObject


class RuleCatalogBaselineTests(unittest.TestCase):
    def test_catalog_covers_each_brush_type_with_unique_legacy_codes(self) -> None:
        catalog = build_affix_catalog()
        codes = [affix_code(affix) for affix in catalog]

        self.assertGreater(len(catalog), 0)
        self.assertEqual(set(BRUSH_TYPE_ORDER), {affix.brush_type for affix in catalog})
        self.assertNotIn(None, codes)
        self.assertEqual(len(codes), len(set(codes)))

    def test_each_catalog_entry_round_trips_through_legacy_code(self) -> None:
        for affix in build_affix_catalog():
            with self.subTest(affix=affix.name, brush=affix.brush_type.value):
                self.assertTrue(affix.legacy_codes)
                for code in affix.legacy_codes:
                    self.assertEqual(affix.name, affix_name_from_code(code))

    def test_catalog_has_one_identity_per_semantic_rule_and_all_legacy_codes(self) -> None:
        catalog = build_affix_catalog()
        identities = [affix.canonical_affix_id for affix in catalog]
        legacy_codes = [code for affix in catalog for code in affix.legacy_codes]

        # 61 = 早期 48 条语义 + 依据官方 S8_Gameplay_SelectableEffect 补齐的 13 条
        self.assertEqual(61, len(catalog))
        self.assertEqual(len(identities), len(set(identities)))
        self.assertNotIn("", identities)
        self.assertEqual(68, len(legacy_codes))
        self.assertEqual(len(legacy_codes), len(set(legacy_codes)))

    def test_official_records_anchor_every_semantic_rule(self) -> None:
        catalog = build_affix_catalog()
        official_ids = [official_id for affix in catalog for official_id in affix.official_ids]

        # 官方 59 条记录收敛到 52 种语义（4001~4007 与 1xxx/2xxx 完全重复）
        self.assertEqual(59, len(official_ids))
        self.assertEqual(len(official_ids), len(set(official_ids)))
        self.assertEqual(52, sum(1 for affix in catalog if affix.verified))
        self.assertEqual(52, sum(1 for affix in catalog if affix.official_text))

    def test_bonus_flag_matches_official_extra_upgrade_wording(self) -> None:
        """官方文案说「额外提升」的笔触必须按**地块加成**记账，反之亦然。

        2026-11-19 实机复盘：2006/2007/2014/2015 四条与 2008/2016 同族（都是
        「额外提升N级」），却被标成 ``grants_bonus=False``——因为
        ``_affix_semantic_key`` 不含该字段，同键的 legacy 条目先到先得，把官方条目
        并进了普通提升条目。这一条用全目录扫描把两个方向都钉住，以后新增条目也会被检查。
        """
        catalog = build_affix_catalog()
        for affix in catalog:
            text = affix.official_text or affix.name
            with self.subTest(affix=affix.canonical_affix_id):
                self.assertEqual(
                    "额外提升" in text,
                    affix.grants_bonus,
                    f"{affix.canonical_affix_id} 的 grants_bonus 与文案「{text}」不一致",
                )

    def test_bonus_flag_participates_in_semantic_identity(self) -> None:
        """仅差 ``grants_bonus`` 的两条规则必须有不同的合并键，不能被并成一条。"""
        from huahua_assistant.game_model import AffixEffect, AffixRule, _affix_semantic_key

        base = {"brush_type": BrushType.STACK, "effect": AffixEffect.UPGRADE_IN_SCOPE, "amount": 1}
        plain = AffixRule(name="范围内所有颜料提升1级", grants_bonus=False, **base)
        bonus = AffixRule(name="范围内所有颜料额外提升1级", grants_bonus=True, **base)
        self.assertNotEqual(_affix_semantic_key(plain), _affix_semantic_key(bonus))

    def test_unverified_entries_are_the_known_legacy_guesses(self) -> None:
        unverified = {affix.canonical_affix_id for affix in build_affix_catalog() if not affix.verified}

        self.assertEqual(
            {
                "transform.direction.global",
                "transform.randomize_scope.any",
                "transform.randomize_scope.gameplay",
                "transform.randomize_scope.monster",
                "stack.upgrade_scope.same_paint.lv1",
                "stack.upgrade_scope.same_paint.lv2",
                "stack.upgrade_global.gameplay.lv2",
                "stack.upgrade_global.monster.lv2",
                "spread.clear_global.faded.random_1",
            },
            unverified,
        )

    def test_official_ids_and_text_resolve_to_the_same_semantic_rule(self) -> None:
        # 官方 4005 与 1008 是同一条（随机等级），必须收敛到同一个规范化 ID
        self.assertEqual(canonical_affix_id("4005"), canonical_affix_id("1008"))
        self.assertEqual(canonical_affix_id("1002"), canonical_affix_id("4001"))

        for official_id in (1012, 2008, 2013, 2016, 2018, 2019, 2022, 3008, 3009, 3011, 3014, 3015, 3016, 3018):
            with self.subTest(official_id=official_id):
                resolved = resolve_affix(str(official_id))
                self.assertIsNotNone(resolved, f"官方记录 {official_id} 未接入词缀目录")
                self.assertTrue(resolved.verified)
                self.assertIn(official_id, resolved.official_ids)

    def test_3015_uses_official_two_void_semantics(self) -> None:
        resolved = resolve_affix("3015")

        self.assertIsNotNone(resolved)
        self.assertEqual("spread.clear_global.void.random_2", resolved.canonical_affix_id)
        self.assertEqual(2, resolved.count)
        self.assertIn("2个", resolved.official_text)
        self.assertEqual(resolved.canonical_affix_id, canonical_affix_id("3-10"))

    def test_official_text_aliases_are_resolvable(self) -> None:
        # 游戏内显示的是官方文案，必须能按原文反查到同一条目
        for official_text, legacy_code in (
            ("将全图所有阻挡地块转变为空白地块", "3-16"),
            ("随机转变全图所有阻挡地块的位置至其他空白地块", "3-17"),
            ("将全图中所有颜料的内容替换为自身", "2-27"),
            ("调色目标的进度要求减少3", "1-18"),
        ):
            with self.subTest(text=official_text):
                resolved = resolve_affix(official_text)
                self.assertIsNotNone(resolved)
                self.assertEqual(canonical_affix_id(legacy_code), resolved.canonical_affix_id)
                # 官方文案要么就是条目名，要么进 aliases，保证按游戏内原文可识别
                self.assertIn(official_text, (resolved.name, *resolved.aliases))
                self.assertEqual(official_text, resolved.official_text)

    def test_equivalent_display_texts_and_legacy_codes_share_identity(self) -> None:
        short = resolve_affix("范围内随机1个颜料提升1级")
        long = resolve_affix("对调色范围内的随机1个颜料额外提升1级")

        self.assertIsNotNone(short)
        self.assertEqual(short, long)
        self.assertEqual(canonical_affix_id("2-2"), canonical_affix_id("2-6"))
        self.assertEqual(short.canonical_affix_id, canonical_affix_id(long))
        self.assertIn("对调色范围内的随机1个颜料额外提升1级", short.aliases)

    def test_canonical_ids_are_explicit_and_stable(self) -> None:
        self.assertEqual(
            "stack.upgrade_scope.random_1.lv1",
            canonical_affix_id("2-2"),
        )
        self.assertEqual(
            "stack.upgrade_scope.random_1.lv1",
            canonical_affix_id("2-6"),
        )

    def test_alias_lookup_normalizes_spacing_punctuation_and_brush_prefix(self) -> None:
        resolved = resolve_affix(
            "stack: 对调色范围内的随机 1 个颜料，额外提升 1 级"
        )

        self.assertIsNotNone(resolved)
        self.assertEqual("stack.upgrade_scope.random_1.lv1", resolved.canonical_affix_id)

    def test_unambiguous_names_keep_string_lookup_compatibility(self) -> None:
        catalog = build_affix_catalog()
        name_counts = Counter(affix.name for affix in catalog)

        for affix in catalog:
            if name_counts[affix.name] != 1:
                continue
            with self.subTest(affix=affix.name):
                self.assertEqual(affix_code(affix), affix_code(affix.name))

    def test_stage_one_options_use_documented_game_order(self) -> None:
        unordered = [BrushType.SPREAD, BrushType.TRANSFORM, BrushType.STACK]

        self.assertEqual(tuple(BRUSH_TYPE_ORDER), normalize_brush_options(unordered))
        self.assertEqual(
            (BrushType.TRANSFORM, BrushType.SPREAD),
            normalize_brush_options([BrushType.SPREAD, BrushType.TRANSFORM]),
        )

    def test_compact_codes_remain_stable_for_existing_payloads(self) -> None:
        paint = Paint(
            category=PaintCategory.GAMEPLAY,
            level=6,
            subtype="宝箱",
            is_treasure=True,
        )
        rule = CurrentPaintRule(
            paint=paint,
            direction=PaintDirection.ADJACENT,
            target_object=PaintObject.GAMEPLAY,
            match_required=4,
            upgrade_amount=2,
            match_current=1,
        )

        self.assertEqual("T6-宝箱", compact_paint_with_code(paint)["code"])
        self.assertEqual("I:T6-宝箱:A:G:4:2", compact_rule_with_code(rule)["code"])

    def test_unknown_brush_and_affix_names_are_not_serialized_as_codes(self) -> None:
        payload = brush_payload_codes(
            [BrushType.TRANSFORM.value, "future-brush"],
            "future-brush",
            ["future-affix"],
        )

        self.assertEqual(["1"], payload["stage1_codes"])
        self.assertIsNone(payload["chosen_code"])
        self.assertEqual([], payload["stage2_codes"])
        self.assertEqual([], payload["stage2_ids"])

    def test_brush_payload_keeps_legacy_codes_and_adds_canonical_ids(self) -> None:
        payload = brush_payload_codes(
            [BrushType.STACK.value],
            BrushType.STACK.value,
            ["对调色范围内的随机1个颜料额外提升1级"],
        )

        self.assertEqual(["2-2"], payload["stage2_codes"])
        self.assertEqual([canonical_affix_id("2-2")], payload["stage2_ids"])


if __name__ == "__main__":
    unittest.main()
