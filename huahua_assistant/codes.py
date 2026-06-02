from __future__ import annotations

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
    if affix is None:
        return None
    if isinstance(affix, AffixRule):
        return _affix_code_by_key().get((affix.brush_type, affix.name))
    return _affix_code_by_name().get(affix)


def affix_label_with_code(affix: AffixRule) -> str:
    code = affix_code(affix)
    return f"{code} {affix.name}" if code else affix.name


def affix_name_from_code(code: str | None) -> str | None:
    if not code:
        return None
    return _affix_name_by_code().get(code)


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
    }


@lru_cache(maxsize=1)
def _affix_code_by_key() -> dict[tuple[BrushType, str], str]:
    counters: dict[BrushType, int] = {brush_type: 0 for brush_type in BrushType}
    result: dict[tuple[BrushType, str], str] = {}
    for affix in build_affix_catalog():
        counters[affix.brush_type] += 1
        result[(affix.brush_type, affix.name)] = f"{BRUSH_CODES[affix.brush_type]}-{counters[affix.brush_type]}"
    return result


@lru_cache(maxsize=1)
def _affix_code_by_name() -> dict[str, str]:
    result: dict[str, str] = {}
    for (brush_type, name), code in _affix_code_by_key().items():
        result[name] = code
        result[f"{brush_type.value}:{name}"] = code
    return result


@lru_cache(maxsize=1)
def _affix_name_by_code() -> dict[str, str]:
    return {code: name for (_brush_type, name), code in _affix_code_by_key().items()}


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
