"""Read recorded effects and level changes without simulating gameplay."""
from __future__ import annotations

from .game_model import _scope_cells
from .models import PaintDirection


def rule_effects(rule: dict) -> list[dict]:
    if isinstance(rule.get("effects"), list) and rule["effects"]:
        return [dict(effect) for effect in rule["effects"]]
    # Older archives stored one target with optional extra directions.
    return [{"direction": direction, "target": rule.get("target"),
             "upgrade": rule.get("upgrade")}
            for direction in [rule.get("direction"), *(rule.get("extra") or [])]]


def selection_evidence(step: dict) -> dict:
    """Expose a choice with provenance; never treat temporal proximity as proof."""
    bundle = step.get("recording_evidence") or {}
    modern = bundle.get("brush_choices")
    if isinstance(modern, dict) and modern.get("schema") == "brush-evidence-v1":
        choices = modern.get("choices") or []
        record = choices[-1] if choices else None
        if record is None or record.get("stage") != "effect" or record.get("status") != "confirmed":
            return {"status": "unknown", "selected": None, "source": "retained_card_ocr",
                    "choice_id": (record or {}).get("choice_id"),
                    "note": "卡片记录尚未取得选择、确认和文字的完整证据；不采用旧版倒推结论。"}
        result = dict(record)
        if not result.get("selected"):
            from .codes import resolve_affix
            rule = resolve_affix(result.get("canonical_affix_id"))
            result["selected"] = rule.display_name if rule is not None else None
        result["status"] = "unlinked"
        result["note"] = "已记录卡片选择和确认；对应颜料及落点仍需独立证据。"
        paint = (step.get("current") or {}).get("paint") or {}
        observed = record.get("paint_observation") or {}
        placement = record.get("placement") or {}
        if (placement.get("status") == "associated" and step.get("anchor") == [placement.get("row"), placement.get("col")]
                and observed.get("agreement") is True and observed.get("subtype") == paint.get("s")
                and observed.get("category") == paint.get("c")):
            result["status"] = "associated"
            result["note"] = "选卡、确认、笔触正文及落点读数一致；视觉关联，不等于协议确认。"
        return result
    events = bundle.get("events") or []
    selected = None
    for event in events:
        payload = event.get("payload") or {}
        kind = event.get("event")
        if kind in ("affix_options_observed", "affix_options_refreshed") or (
                kind == "brush_stage1_options_observed" and not payload.get("stage2_visible")
                and len(payload.get("options") or []) > 1):
            selected = None
        elif kind == "affix_choice_resolved":
            selected = event
        elif kind == "current_paint_changed":
            left, right = payload.get("before") or {}, payload.get("after") or {}
            if any(left.get(k) is not None and right.get(k) is not None and left[k] != right[k]
                   for k in ("category", "subtype", "treasure", "direction", "target", "match_required", "effect")):
                selected = None
    result = {"status": "unknown", "selected": None, "source": None,
              "note": "未保存可关联的具体词缀选择。"}
    if selected is None:
        return result
    payload = selected.get("payload") or {}
    result.update(selected=payload.get("selected"), source=payload.get("source"),
                  event={"session": selected.get("session"), "sequence": selected.get("sequence"),
                         "ts": selected.get("ts")}, confidence=payload.get("confidence"))
    # Old saves remain unknown here; do not silently turn a present-day lookup
    # into an ID allegedly recorded during that historical game.
    for key in ("canonical_affix_id", "brush_effect_id", "brush_effect_ids",
                "brush_effect_id_status", "brush_effect_id_source", "brush_effect_table"):
        result[key] = payload.get(key)
    observed = payload.get("observed_current") or {}
    rule = step.get("current") or {}
    paint = rule.get("paint") or {}
    # Matching readings cannot prove identity across two identical draws.
    # Missing identity, a new panel, or a different anchor prevents even this association.
    compatible = (observed.get("subtype") and observed.get("subtype") == paint.get("s")
                  and observed.get("category") == paint.get("c")
                  and observed.get("level") is not None and observed.get("level") == paint.get("l")
                  and observed.get("direction") == rule.get("direction")
                  and observed.get("target") == rule.get("target"))
    anchor = payload.get("observed_anchor")
    if anchor is not None and anchor != step.get("anchor"):
        compatible = False
    if (step.get("brush") or {}).get("stage2"):
        compatible = False
    if compatible:
        result["status"] = "associated"
        result["note"] = "与本帧颜料读数相符；按面板前后关联，不能单凭此证明是同一枚。"
    else:
        result["status"] = "unlinked"
        result["note"] = "记录了这条选择，但无法确认对应本帧初始颜料。"
    if result["source"] not in ("captured_click", "buffered_click"):
        result["note"] += "该选择来自效果推断，未记录点击。"
    return result


def _readings(step: dict, observed: bool) -> list[dict]:
    if observed:
        return step["observed_tiles"]
    states = (step.get("board") or {}).get("states") or ""
    tiles = step.get("tiles") or []
    return [{**(tiles[i] or {}), "state": states[i]} for i in range(min(len(states), len(tiles)))]


def scope_comparison(before: dict | None, after: dict, gap: float) -> dict:
    """Each effect keeps its own scope and target, including exact duplicates."""
    report = {"available": False, "effects": [], "anchor": None,
              "note": "首帧之前没有可比较的局面。", "causal": False}
    if before is None:
        return report
    anchor = before.get("anchor")
    placed_now = False
    if before.get("pending") is True and after.get("pending") is False:
        left, right = before.get("current") or {}, after.get("current") or {}
        paint = left.get("paint") or {}
        if paint.get("s") and paint == right.get("paint") and rule_effects(left) == rule_effects(right):
            anchor = after.get("anchor")
            placed_now = True
    if ((before.get("pending") is True and not placed_now) or not isinstance(anchor, (list, tuple)) or len(anchor) != 2
            or any(type(n) is not int or not 0 <= n < 6 for n in anchor)):
        report["note"] = "上一帧的初始颜料位置未确认，暂不能核对范围。"
        return report
    rule = before.get("current") or {}
    if not rule:
        report["note"] = "上一帧未保存调色规则。"
        return report
    observed = all(len(step.get("observed_tiles") or []) == 36 for step in (before, after))
    readings_before, readings_after = _readings(before, observed), _readings(after, observed)
    uncertain_frame = (gap < 0 or gap > 8 or
                       any((s.get("quality") or {}).get("ok") is not True for s in (before, after)))
    report.update(available=True, anchor=list(anchor), anchor_source="placement_frame" if placed_now else "previous_frame",
                  level_source="displayed_observation" if observed else "legacy_model",
                  note=("比较上一帧初始颜料的调色范围与本帧等级读数；变化不等于这条效果造成。"
                        + ("间隔或采集质量不足，全部变化待核对。" if uncertain_frame else "")
                        + ("旧档只保存模型基础等级，不含推算加成。" if not observed else "")))
    for ordinal, effect in enumerate(rule_effects(rule), 1):
        entry = {**effect, "ordinal": ordinal, "cells": [], "targets": [], "uncertain_cells": [],
                 "available": False, "increased": 0, "unchanged": 0, "decreased": 0}
        report["effects"].append(entry)
        try:
            direction = PaintDirection(effect.get("direction"))
        except (TypeError, ValueError):
            continue
        if direction == PaintDirection.UNKNOWN:
            continue
        scope = [r * 6 + c for r, c in _scope_cells(*anchor, direction) if (r, c) != tuple(anchor)]
        entry["cells"] = scope
        target = effect.get("target")
        if target not in ("monster", "gameplay", "all"):
            continue
        entry["available"] = True
        for i in scope:
            left = readings_before[i] if i < len(readings_before) else {}
            right = readings_after[i] if i < len(readings_after) else {}
            if left.get("state") == "?" or not left:
                entry["uncertain_cells"].append(i)
                continue
            if left.get("state") != "P":
                # Conflicting legacy terrain and paint records cannot be trusted.
                if left.get("c") in ("monster", "gameplay"):
                    entry["uncertain_cells"].append(i)
                continue
            if left.get("c") not in ("monster", "gameplay"):
                entry["uncertain_cells"].append(i)
                continue
            if target != "all" and left.get("c") != target:
                continue
            same = (right.get("state") == "P" and left.get("c") == right.get("c")
                    and bool(left.get("s")) and left.get("s") == right.get("s")
                    and bool(left.get("t")) == bool(right.get("t")))
            levels = type(left.get("l")) is int and type(right.get("l")) is int
            confidence_ok = not observed or all(
                (p.get("confidence") or 0) >= .65 and (p.get("level_confidence") or 0) >= .65
                for p in (left, right))
            status = "uncertain"
            delta = right["l"] - left["l"] if levels else None
            if same and levels and confidence_ok and not uncertain_frame:
                status = "increased" if delta > 0 else "decreased" if delta < 0 else "unchanged"
                entry[status] += 1
            else:
                entry["uncertain_cells"].append(i)
            entry["targets"].append({"cell": i, "name": left.get("s"), "category": left.get("c"),
                                     "before": left.get("l"), "after": right.get("l"),
                                     "delta": delta, "status": status})
    return report
