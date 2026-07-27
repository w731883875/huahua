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

        self.assertEqual(48, len(catalog))
        self.assertEqual(len(identities), len(set(identities)))
        self.assertNotIn("", identities)
        self.assertEqual(55, len(legacy_codes))
        self.assertEqual(len(legacy_codes), len(set(legacy_codes)))

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
