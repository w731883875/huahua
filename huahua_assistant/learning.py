from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


RUNTIME_DIR = Path("runtime")
OBSERVATION_LOG = RUNTIME_DIR / "observations.jsonl"
STATS_PATH = RUNTIME_DIR / "stats.json"


def record_observation(event_type: str, payload: dict[str, Any], *, source: str) -> None:
    if source != "real_capture":
        return
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": time.time(),
        "event": event_type,
        "source": source,
        "payload": _jsonable(payload),
    }
    with OBSERVATION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    _write_stats(_load_events())


def load_learning_stats() -> dict[str, Any]:
    # Rebuild from raw events so old stat files cannot keep pre-filter data.
    return _build_stats(_load_events())


def _load_events() -> list[dict[str, Any]]:
    if not OBSERVATION_LOG.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in OBSERVATION_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _write_stats(events: list[dict[str, Any]]) -> None:
    STATS_PATH.write_text(
        json.dumps(_build_stats(events), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    events = [event for event in events if event.get("source") == "real_capture"]
    event_counts = Counter(event.get("event", "unknown") for event in events)
    affix_seen = Counter()
    stage1_seen = Counter()
    current_seen = Counter()
    mismatch_seen = Counter()
    backpack_levels = Counter()
    backpack_categories = Counter()
    treasure_count = 0

    for event in events:
        payload = event.get("payload") or {}
        event_type = event.get("event")
        if event_type == "brush_options":
            for item in payload.get("stage1_options") or []:
                stage1_seen[item] += 1
            for item in payload.get("stage2_options") or []:
                affix_seen[item] += 1
        elif event_type == "current_paint":
            key = "|".join(
                str(payload.get(name))
                for name in ("category", "level", "direction", "target", "match_required", "upgrade_amount")
            )
            current_seen[key] += 1
        elif event_type == "backpack_capture":
            for paint in payload.get("paints") or []:
                backpack_levels[str(paint.get("level"))] += 1
                backpack_categories[str(paint.get("category"))] += 1
                treasure_count += 1 if paint.get("is_treasure") else 0
        elif event_type == "rule_mismatch":
            mismatch_seen[str(payload.get("kind", "unknown"))] += 1

    total_backpack = sum(backpack_levels.values())
    return {
        "total_events": len(events),
        "event_counts": dict(event_counts),
        "stage1_seen": dict(stage1_seen),
        "affix_seen": dict(affix_seen),
        "current_paint_seen": dict(current_seen),
        "rule_mismatches": dict(mismatch_seen),
        "backpack_levels": dict(backpack_levels),
        "backpack_categories": dict(backpack_categories),
        "treasure_ratio": treasure_count / total_backpack if total_backpack else 0.0,
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value
