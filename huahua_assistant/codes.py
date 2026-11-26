from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
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
    PaintObject.FADED_TILE: "F",
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


def brush_effect_identity(affix: AffixRule | str | None,
                          brush_type: BrushType | str | None = None) -> dict:
    """Serialize selectable-effect identity without inventing a unique table row.

    Recognition yields card text, not a protocol ID. Some identical effects have
    several official rows; preserve all of them and leave the scalar ID null.
    Exact catalog aliases only: no fuzzy text match in saved choice identity.
    """
    resolved = resolve_affix(affix, brush_type)
    ids = sorted(set(resolved.official_ids)) if resolved is not None else []
    if isinstance(affix, str) and affix.strip().isdigit() and int(affix.strip()) in ids:
        ids = [int(affix.strip())]
    return {
        "canonical_affix_id": resolved.canonical_affix_id if resolved else None,
        "brush_effect_id": ids[0] if len(ids) == 1 else None,
        "brush_effect_ids": ids,
        "brush_effect_id_status": "unique" if len(ids) == 1 else "ambiguous" if ids else "unknown",
        "brush_effect_id_source": "catalog_mapping" if ids else None,
        "brush_effect_table": "S8_Gameplay_SelectableEffect",
    }


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
    if value.isdigit() and int(value) in _affix_by_official_id():
        return _affix_by_official_id()[int(value)]
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


@lru_cache(maxsize=1)
def _official_effect_index() -> dict[str, tuple[int, ...]]:
    """官方**笔触**文案（归一化）→ 官方记录 ID。同一文案可能对应多条记录（1008/4005）。"""
    from .s8_official_data import S8_SELECTABLE_EFFECTS

    index: dict[str, list[int]] = {}
    for item in S8_SELECTABLE_EFFECTS:
        index.setdefault(_normalize_affix_text(item.text), []).append(int(item.id))
    return {key: tuple(sorted(ids)) for key, ids in index.items()}


@lru_cache(maxsize=1)
def _official_plot_effect_index() -> dict[str, tuple[int, ...]]:
    """官方**调色效果**文案（归一化）→ 官方记录 ID（``s8_plot_effect.PLOT_EFFECTS``）。"""
    from .s8_plot_effect import PLOT_EFFECTS

    index: dict[str, list[int]] = {}
    for item in PLOT_EFFECTS:
        index.setdefault(_normalize_affix_text(item.effect), []).append(int(item.id))
    return {key: tuple(sorted(ids)) for key, ids in index.items()}


def _lookup_official_ids(index: dict[str, tuple[int, ...]], text: str | None) -> tuple[int, ...]:
    """归一化精确匹配，失败按整体相似度兜底（阈值 0.85），低于阈值一律返回空。"""
    if not text:
        return ()
    key = _normalize_affix_text(text)
    if not key:
        return ()
    exact = index.get(key)
    if exact:
        return exact
    best_score = 0.0
    best_ids: tuple[int, ...] = ()
    for candidate, ids in index.items():
        score = _similarity(key, candidate)
        if score > best_score:
            best_score = score
            best_ids = ids
    return best_ids if best_score >= 0.85 else ()


def official_effect_ids(text: str | None) -> tuple[int, ...]:
    """面板「调色效果」原文 → 官方**笔触**记录 ID，例如「调色目标的进度要求减少2」→ ``(1011,)``。

    官方 ID 自带语义：首位 ``1``=转锋 / ``2``=堆砌 / ``3``=扩散，后三位是条目。
    所以复盘时只留 ID 就够（用户 2026-11-18 明确要求按 ID 记录）。

    ⚠️ 这是**笔触表**（``S8_Gameplay_SelectableEffect``，59 条）。地块自身的
    「调色效果」（``S8_Gameplay_ItemEffect``，27 条，1101~1108 等）ID 空间与它重叠
    但语义完全不同，必须用 :func:`plot_effect_ids` 查，不要混用。
    """

    return _lookup_official_ids(_official_effect_index(), text)


def plot_effect_ids(text: str | None) -> tuple[int, ...]:
    """面板「调色效果」原文 → 官方**地块调色效果**记录 ID，例如「使纵向的玩法颜料提升2级」→ ``(2006,)``。

    与 :func:`official_effect_ids` 分属两张官方表，ID 会撞号（1101 在调色表里是
    「使横向的怪物颜料提升1级」，在笔触表里是「转变自身的作用方向为横向」），
    所以复盘时必须分开记成两个字段。
    """

    return _lookup_official_ids(_official_plot_effect_index(), text)


#: 语义更直白的别名：谁也别再猜 ``effect_id`` 到底指哪张表。
brush_effect_ids = official_effect_ids


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


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
        # Fused paints can repeat an effect or target different paint categories.
        "effects": [
            {"direction": item.direction.value, "target": item.target_object.value,
             "upgrade": item.upgrade_amount}
            for item in rule.effects
        ],
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
def _affix_by_official_id() -> dict[int, AffixRule]:
    """官方 S8_Gameplay_SelectableEffect 记录 ID -> 词缀（用于按游戏内 id 反查）。"""
    result: dict[int, AffixRule] = {}
    for affix in build_affix_catalog():
        for official_id in affix.official_ids:
            result[official_id] = affix
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


def normalize_affix_text(value: str | None) -> str:
    """词缀文本归一化（NFKC + 只留字母数字），比对卡片名/别名时用。

    单独暴露出来是为了让"匹配卡片"这类判断不必各自抄一遍归一化规则——
    抄歪一处，别名就匹配不上了。
    """
    if not value:
        return ""
    return _normalize_affix_text(str(value))


def _normalize_affix_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(ch for ch in normalized if ch.isalnum())
