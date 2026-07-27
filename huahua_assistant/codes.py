from __future__ import annotations

import unicodedata
from functools import lru_cache

from .game_model import AffixRule, BrushType, CurrentPaintRule, Paint, build_affix_catalog
from .models import PaintCategory, PaintDirection, PaintObject


BRUSH_CODES: dict[BrushType, str] = {
    BrushType.TRANSFORM: "1",
    BrushType.STACK: "2",
    BrushType.SPREAD: "3",
}

BRUSH_NAMES: dict[BrushType, str] = {
    BrushType.TRANSFORM: "转锋",
    BrushType.STACK: "堆砌",
    BrushType.SPREAD: "扩散",
}

PAINT_CATEGORY_CODES: dict[PaintCategory, str] = {
    PaintCategory.MONSTER: "M",
    PaintCategory.GAMEPLAY: "G",
    PaintCategory.UNKNOWN: "U",
}

DIRECTION_CODES: dict[PaintDirection, str] = {
    PaintDirection.HORIZONTAL: "H",
    PaintDirection.VERTICAL: "V",
    PaintDirection.DIAGONAL: "D",
    PaintDirection.ADJACENT: "A",
    PaintDirection.GLOBAL: "O",
    PaintDirection.UNKNOWN: "U",
}

TARGET_CODES: dict[PaintObject, str] = {
    PaintObject.MONSTER: "M",
    PaintObject.GAMEPLAY: "G",
    PaintObject.ALL: "A",
    PaintObject.BLOCKED_TILE: "K",
    PaintObject.BLANK_TILE: "B",
    PaintObject.VOID_TILE: "V",
    PaintObject.UNKNOWN: "U",
}


def brush_code(brush_type: BrushType | str | None) -> str | None:
    parsed = _parse_brush_type(brush_type)
    if parsed is None:
        return None
    return BRUSH_CODES[parsed]


def affix_code(affix: AffixRule | str | None) -> str | None:
    resolved = resolve_affix(affix)
    if resolved is None or not resolved.legacy_codes:
        return None
    return resolved.legacy_codes[0]


def canonical_affix_id(affix: AffixRule | str | None) -> str | None:
    resolved = resolve_affix(affix)
    return resolved.canonical_affix_id if resolved is not None else None


def resolve_affix(
    affix: AffixRule | str | None,
    brush_type: BrushType | str | None = None,
) -> AffixRule | None:
    if affix is None:
        return None
    parsed_brush = _parse_brush_type(brush_type)
    if isinstance(affix, AffixRule):
        if affix.canonical_affix_id:
            resolved = _affix_by_id().get(affix.canonical_affix_id)
            if resolved is not None:
                return resolved
        parsed_brush = affix.brush_type
        value = affix.name
    else:
        value = str(affix).strip()
    if not value:
        return None
    if value in _affix_by_id():
        return _affix_by_id()[value]
    if value in _affix_by_legacy_code():
        return _affix_by_legacy_code()[value]
    if ":" in value:
        brush_value, alias_value = value.split(":", 1)
        prefixed_brush = _parse_brush_type(brush_value)
        if prefixed_brush is not None:
            return _affix_by_brush_alias().get((prefixed_brush, _normalize_affix_text(alias_value)))
    key = _normalize_affix_text(value)
    if parsed_brush is not None:
        return _affix_by_brush_alias().get((parsed_brush, key))
    return _affix_by_unambiguous_alias().get(key)


def affix_aliases(affix: AffixRule | str | None) -> tuple[str, ...]:
    resolved = resolve_affix(affix)
    if resolved is None:
        return ()
    return (resolved.display_name, *resolved.aliases)


def affix_label_with_code(affix: AffixRule) -> str:
    code = affix_code(affix)
    return f"{code} {affix.name}" if code else affix.name


def affix_name_from_code(code: str | None) -> str | None:
    resolved = resolve_affix(code)
    return resolved.display_name if resolved is not None else None


def affix_display_from_code(code: str | None) -> str | None:
    if not code:
        return None
    name = affix_name_from_code(code)
    return f"{code} {name}" if name else code


def paint_code(paint: Paint | None) -> str | None:
    if paint is None:
        return None
    category = "T" if paint.is_treasure else PAINT_CATEGORY_CODES.get(paint.category, "U")
    subtype = _subtype_code(paint.subtype)
    return f"{category}{paint.level}{subtype}"


def rule_code(rule: CurrentPaintRule | None) -> str | None:
    if rule is None:
        return None
    paint = paint_code(rule.paint) or "U0"
    direction = DIRECTION_CODES.get(rule.direction, "U")
    target = TARGET_CODES.get(rule.target_object, "U")
    return f"I:{paint}:{direction}:{target}:{rule.match_required}:{rule.upgrade_amount}"


def compact_paint_with_code(paint: Paint) -> dict:
    return {
        "code": paint_code(paint),
        "c": paint.category.value,
        "l": paint.level,
        "t": 1 if paint.is_treasure else 0,
        "s": paint.subtype,
    }


def compact_rule_with_code(rule: CurrentPaintRule) -> dict:
    return {
        "code": rule_code(rule),
        "paint": compact_paint_with_code(rule.paint),
        "direction": rule.direction.value,
        "direction_code": DIRECTION_CODES.get(rule.direction, "U"),
        "target": rule.target_object.value,
        "target_code": TARGET_CODES.get(rule.target_object, "U"),
        "match": [rule.match_current, rule.match_required],
        "upgrade": rule.upgrade_amount,
        "extra": [item.value for item in rule.extra_directions],
        "extra_code": [DIRECTION_CODES.get(item, "U") for item in rule.extra_directions],
    }


def brush_payload_codes(stage1: list[str], chosen: str | None, stage2: list[str]) -> dict:
    return {
        "stage1_codes": [code for item in stage1 if (code := brush_code(item))],
        "chosen_code": brush_code(chosen),
        "stage2_codes": [code for item in stage2 if (code := affix_code(item))],
        "stage2_ids": [identity for item in stage2 if (identity := canonical_affix_id(item))],
    }


@lru_cache(maxsize=1)
def _affix_by_id() -> dict[str, AffixRule]:
    return {affix.canonical_affix_id: affix for affix in build_affix_catalog()}


@lru_cache(maxsize=1)
def _affix_by_legacy_code() -> dict[str, AffixRule]:
    result: dict[str, AffixRule] = {}
    for affix in build_affix_catalog():
        for code in affix.legacy_codes:
            result[code] = affix
    return result


@lru_cache(maxsize=1)
def _affix_by_brush_alias() -> dict[tuple[BrushType, str], AffixRule]:
    result: dict[tuple[BrushType, str], AffixRule] = {}
    for affix in build_affix_catalog():
        for alias in (affix.display_name, *affix.aliases):
            result[(affix.brush_type, _normalize_affix_text(alias))] = affix
    return result


@lru_cache(maxsize=1)
def _affix_by_unambiguous_alias() -> dict[str, AffixRule]:
    candidates: dict[str, list[AffixRule]] = {}
    for affix in build_affix_catalog():
        for alias in (affix.display_name, *affix.aliases):
            candidates.setdefault(_normalize_affix_text(alias), []).append(affix)
    return {
        key: matches[0]
        for key, matches in candidates.items()
        if len({item.canonical_affix_id for item in matches}) == 1
    }


def _parse_brush_type(value: BrushType | str | None) -> BrushType | None:
    if value is None:
        return None
    if isinstance(value, BrushType):
        return value
    try:
        return BrushType(value)
    except ValueError:
        for brush_type, code in BRUSH_CODES.items():
            if value == code:
                return brush_type
    return None


def _subtype_code(subtype: str | None) -> str:
    if not subtype:
        return ""
    normalized = "".join(ch for ch in subtype.upper() if ch.isalnum())
    if not normalized:
        return ""
    return f"-{normalized[:4]}"


def _normalize_affix_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(ch for ch in normalized if ch.isalnum())
