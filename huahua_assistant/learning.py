from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .codes import affix_code, brush_code
from .config import config_value


RUNTIME_DIR = Path("runtime")
OBSERVATION_LOG = RUNTIME_DIR / "observations.jsonl"
CORRECTION_LOG = RUNTIME_DIR / "corrected_observations.jsonl"
TRAJECTORY_LOG = RUNTIME_DIR / "high_score_trajectories.jsonl"
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
    _write_stats(_load_events(), _load_corrections(), _load_trajectories())


def record_correction(payload: dict[str, Any], *, source: str) -> None:
    if source != "real_capture":
        return
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": time.time(),
        "event": "corrected_observation",
        "source": source,
        "payload": _jsonable(payload),
    }
    with CORRECTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    _write_stats(_load_events(), _load_corrections(), _load_trajectories())


def record_high_score_trajectory(payload: dict[str, Any], *, source: str) -> bool:
    if source != "real_capture":
        return False
    if not _trajectory_payload_is_trusted(payload):
        return False
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload_hash = _payload_hash(payload)
    existing_hashes = {_payload_hash(event.get("payload") or {}) for event in _load_trajectories()}
    if payload_hash in existing_hashes:
        return False
    event = {
        "ts": time.time(),
        "event": "high_score_trajectory",
        "source": source,
        "payload": _jsonable(payload),
        "payload_hash": payload_hash,
    }
    with TRAJECTORY_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    _write_stats(_load_events(), _load_corrections(), _load_trajectories())
    return True


def _trajectory_payload_is_trusted(payload: dict[str, Any]) -> bool:
    gate = payload.get("high_score_gate")
    if isinstance(gate, dict) and gate.get("passed") is not True:
        return False
    steps = payload.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        if not isinstance(step, dict):
            return False
        quality = step.get("quality") or {}
        if quality.get("ok") is not True:
            return False
    return True


def export_learning_bundle(path: str | Path) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    observations = _dedupe_events(_load_events())
    trajectories = _dedupe_events(_load_trajectories())
    corrections = _dedupe_events(_load_corrections())
    bundle = {
        "schema": "huahua-learning-bundle-v1",
        "exported_at": time.time(),
        "stats": load_learning_stats(),
        "observations": observations,
        "trajectories": trajectories,
        "corrections": corrections,
    }
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "path": str(output),
        "observations": len(observations),
        "trajectories": len(trajectories),
        "corrections": len(corrections),
    }


def merge_learning_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    observations = data.get("observations") or []
    trajectories = data.get("trajectories") or []
    corrections = data.get("corrections") or []
    added_observations = _append_unique_events(OBSERVATION_LOG, observations, event_name="observation")
    added_trajectories = _append_unique_events(TRAJECTORY_LOG, trajectories, event_name="high_score_trajectory")
    added_corrections = _append_unique_events(CORRECTION_LOG, corrections, event_name="corrected_observation")
    _write_stats(_load_events(), _load_corrections(), _load_trajectories())
    return {
        "path": str(bundle_path),
        "added_observations": added_observations,
        "added_trajectories": added_trajectories,
        "added_corrections": added_corrections,
    }


def load_learning_stats() -> dict[str, Any]:
    # Rebuild from raw events so old stat files cannot keep pre-filter data.
    return _build_stats(_load_events(), _load_corrections(), _load_trajectories())


def _load_events() -> list[dict[str, Any]]:
    return _load_jsonl(OBSERVATION_LOG)


def _load_corrections() -> list[dict[str, Any]]:
    return _load_jsonl(CORRECTION_LOG)


def _load_trajectories() -> list[dict[str, Any]]:
    return _load_jsonl(TRAJECTORY_LOG)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for event in events:
        if event.get("source") != "real_capture":
            continue
        if event.get("event") == "high_score_trajectory" and not _trajectory_payload_is_trusted(event.get("payload") or {}):
            continue
        key = event.get("payload_hash") or _payload_hash(event.get("payload") or {})
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return result


def _append_unique_events(path: Path, incoming: list[dict[str, Any]], *, event_name: str) -> int:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    existing_hashes = {
        event.get("payload_hash") or _payload_hash(event.get("payload") or {})
        for event in _load_jsonl(path)
        if event.get("source") == "real_capture"
    }
    added = 0
    with path.open("a", encoding="utf-8") as handle:
        for event in incoming:
            if event.get("source") != "real_capture":
                continue
            payload = event.get("payload") or {}
            if event_name == "high_score_trajectory" and not _trajectory_payload_is_trusted(payload):
                continue
            payload_hash = event.get("payload_hash") or _payload_hash(payload)
            if payload_hash in existing_hashes:
                continue
            merged = {
                "ts": event.get("ts", time.time()),
                "event": event.get("event") or event_name,
                "source": "real_capture",
                "payload": _jsonable(payload),
                "payload_hash": payload_hash,
            }
            handle.write(json.dumps(merged, ensure_ascii=False, sort_keys=True) + "\n")
            existing_hashes.add(payload_hash)
            added += 1
    return added


def _write_stats(
    events: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    trajectories: list[dict[str, Any]] | None = None,
) -> None:
    STATS_PATH.write_text(
        json.dumps(_build_stats(events, corrections, trajectories), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_stats(
    events: list[dict[str, Any]],
    corrections: list[dict[str, Any]] | None = None,
    trajectories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    events = [event for event in events if event.get("source") == "real_capture"]
    corrections = [event for event in (corrections or []) if event.get("source") == "real_capture"]
    trajectories = [
        event
        for event in (trajectories or [])
        if event.get("source") == "real_capture" and _trajectory_payload_is_trusted(event.get("payload") or {})
    ]
    event_counts = Counter(event.get("event", "unknown") for event in events)
    affix_seen = Counter()
    affix_code_seen = Counter()
    stage1_seen = Counter()
    stage1_code_seen = Counter()
    unknown_stage1_seen = Counter()
    unknown_affix_seen = Counter()
    unknown_affix_samples: dict[str, list[dict[str, Any]]] = {}
    current_seen = Counter()
    current_code_seen = Counter()
    mismatch_seen = Counter()
    anomaly_seen = Counter()
    anomaly_severity = Counter()
    backpack_levels = Counter()
    backpack_categories = Counter()
    correction_fields = Counter()
    correction_samples = Counter()
    trajectory_actions = Counter()
    inferred_actions = Counter()
    inferred_action_codes = Counter()
    inferred_action_details = Counter()
    placement_cells = Counter()
    placement_zones = Counter()
    recommendation_cells = Counter()
    recommendation_shorts = Counter()
    recommendation_tags = Counter()
    high_value_recommendation_shorts = Counter()
    high_value_recommendation_tags = Counter()
    upgrade_events = 0
    forced_events = 0
    refresh_events = Counter()
    trajectory_brushes = Counter()
    trajectory_brush_codes = Counter()
    trajectory_affixes = Counter()
    trajectory_affix_codes = Counter()
    high_score_gate_reasons = Counter()
    high_score_gate_primary = Counter()
    trajectory_scores: list[float] = []
    imitation_actions = Counter()
    imitation_action_codes = Counter()
    imitation_action_score_delta = Counter()
    imitation_action_score_samples = Counter()
    imitation_recommendation_hits = Counter()
    imitation_recommendation_shorts = Counter()
    imitation_recommendation_tags = Counter()
    imitation_brush_choices = Counter()
    imitation_brush_choice_codes = Counter()
    imitation_affix_choices = Counter()
    imitation_affix_choice_codes = Counter()
    treasure_count = 0
    quick_steps = 0
    quick_actions = Counter()
    quick_elapsed: list[float] = []
    finished_runs = 0
    finished_scores: list[float] = []
    ignored_quick_steps = 0
    ignored_trajectory_steps = 0
    skipped_finished_runs = 0
    high_value_seen = 0
    high_value_best_score = 0.0
    high_value_counts = Counter()

    for event in events:
        payload = event.get("payload") or {}
        event_type = event.get("event")
        if event_type == "brush_options":
            for item in payload.get("stage1_options") or []:
                stage1_seen[item] += 1
                if brush_code(item) is None and (item_text := _unknown_text(item)):
                    unknown_stage1_seen[item_text] += 1
            for item in payload.get("stage1_codes") or []:
                stage1_code_seen[str(item)] += 1
            for item in payload.get("stage2_options") or []:
                affix_seen[item] += 1
                if affix_code(item) is None and (item_text := _unknown_text(item)):
                    unknown_affix_seen[item_text] += 1
                    _append_unknown_affix_sample(unknown_affix_samples, item_text, payload, event)
            for item in payload.get("stage2_codes") or []:
                affix_code_seen[str(item)] += 1
        elif event_type == "current_paint":
            key = "|".join(
                str(payload.get(name))
                for name in ("category", "level", "direction", "target", "match_required", "upgrade_amount")
            )
            current_seen[key] += 1
            if payload.get("rule_code"):
                current_code_seen[str(payload.get("rule_code"))] += 1
            elif payload.get("paint_code"):
                current_code_seen[str(payload.get("paint_code"))] += 1
        elif event_type == "backpack_capture":
            for paint in payload.get("paints") or []:
                backpack_levels[str(paint.get("level"))] += 1
                backpack_categories[str(paint.get("category"))] += 1
                treasure_count += 1 if paint.get("is_treasure") else 0
        elif event_type == "rule_mismatch":
            mismatch_seen[str(payload.get("kind", "unknown"))] += 1
        elif event_type == "rule_anomaly":
            anomaly_seen[str(payload.get("kind", "unknown"))] += 1
            anomaly_severity[str(payload.get("severity", "unknown"))] += 1
        elif event_type == "quick_capture_step":
            quick_steps += 1
            inferred = payload.get("inferred_action") or {}
            quick_actions[str(inferred.get("type", "unknown"))] += 1
            if inferred.get("code"):
                inferred_action_codes[str(inferred.get("code"))] += 1
            elapsed = (payload.get("quality") or {}).get("elapsed")
            if isinstance(elapsed, (int, float)):
                quick_elapsed.append(float(elapsed))
            _accumulate_recommendation_stats(
                payload.get("top") or (),
                cells=recommendation_cells,
                shorts=recommendation_shorts,
                tags=recommendation_tags,
            )
        elif event_type == "run_session_finished":
            finished_runs += 1
            final_score = payload.get("internal_score", payload.get("final_score"))
            if isinstance(final_score, (int, float)):
                finished_scores.append(float(final_score))
        elif event_type == "quick_capture_ignored":
            ignored_quick_steps += 1
        elif event_type in {"capture_step_ignored", "trajectory_step_ignored"}:
            ignored_trajectory_steps += 1
        elif event_type == "run_session_finish_skipped":
            skipped_finished_runs += 1
        elif event_type == "high_value_seen":
            high_value_seen += 1
            known = payload.get("known") or {}
            score = payload.get("score")
            if isinstance(score, (int, float)):
                high_value_best_score = max(high_value_best_score, float(score))
            for key in ("treasure7", "treasure6", "treasure5", "gameplay7", "monster7", "lv7"):
                high_value_counts[key] += int(known.get(key, 0) or 0)
            _accumulate_recommendation_stats(
                payload.get("recommendations") or (),
                cells=recommendation_cells,
                shorts=recommendation_shorts,
                tags=recommendation_tags,
                high_value_shorts=high_value_recommendation_shorts,
                high_value_tags=high_value_recommendation_tags,
            )

    for event in corrections:
        payload = event.get("payload") or {}
        original = payload.get("original") or {}
        corrected = payload.get("corrected") or {}
        fields = set(original) | {key for key in corrected if key != "note"}
        changed_fields = set(payload.get("changed_fields") or [])
        for field in fields:
            correction_samples[str(field)] += 1
        for field in changed_fields:
            correction_fields[str(field)] += 1

    for event in trajectories:
        payload = event.get("payload") or {}
        gate = payload.get("high_score_gate") if isinstance(payload.get("high_score_gate"), dict) else {}
        primary_reason = gate.get("primary_reason") or payload.get("reason")
        if primary_reason:
            high_score_gate_primary[str(primary_reason)] += 1
        for reason in gate.get("reasons") or ():
            high_score_gate_reasons[str(reason)] += 1
        final_score = payload.get("internal_score", payload.get("final_score"))
        if isinstance(final_score, (int, float)):
            trajectory_scores.append(float(final_score))
        previous_brush_choice: str | None = None
        previous_affix_choice: str | None = None
        for step in payload.get("steps") or []:
            action = step.get("action") or step.get("type") or "unknown"
            inferred = step.get("inferred_action") or {}
            trajectory_actions[str(action)] += 1
            inferred_type = str(inferred.get("type", "unknown"))
            inferred_actions[inferred_type] += 1
            if inferred.get("code"):
                inferred_action_codes[str(inferred.get("code"))] += 1
            if inferred_type not in {"session_start", "capture_update", "unknown"}:
                imitation_actions[inferred_type] += 1
                if inferred.get("code"):
                    imitation_action_codes[str(inferred.get("code"))] += 1
                outcome = step.get("outcome") if isinstance(step.get("outcome"), dict) else {}
                score_delta = outcome.get("score_delta")
                if isinstance(score_delta, (int, float)):
                    imitation_action_score_delta[inferred_type] += float(score_delta)
                    imitation_action_score_samples[inferred_type] += 1
            detail = inferred.get("detail") if isinstance(inferred.get("detail"), dict) else {}
            if detail.get("code"):
                inferred_action_details[str(detail.get("code"))] += 1
            summary = inferred.get("summary") if isinstance(inferred.get("summary"), dict) else {}
            upgrade_events += int(summary.get("upgrade_count", 0) or 0)
            if inferred_type == "force_complete":
                forced_events += 1
            if inferred_type == "refresh_affix":
                refresh_events[str(inferred.get("stage", "unknown"))] += 1
            if inferred_type in {"install_paint", "place_current_paint", "new_current_paint"}:
                for cell in _placement_cells_from_inferred(inferred):
                    row = cell.get("r")
                    col = cell.get("c")
                    if not isinstance(row, int) or not isinstance(col, int):
                        continue
                    placement_cells[f"{row},{col}"] += 1
                    placement_zones[_cell_zone(row, col)] += 1
            brush = step.get("brush") or {}
            chosen_brush = brush.get("chosen")
            if chosen_brush:
                trajectory_brushes[str(chosen_brush)] += 1
            chosen_brush_code = brush.get("chosen_code")
            if chosen_brush_code:
                trajectory_brush_codes[str(chosen_brush_code)] += 1
            for affix_name in brush.get("stage2") or []:
                trajectory_affixes[str(affix_name)] += 1
            for code in brush.get("stage2_codes") or []:
                trajectory_affix_codes[str(code)] += 1
            if step.get("brush_type"):
                trajectory_brushes[str(step.get("brush_type"))] += 1
            if step.get("brush_code"):
                trajectory_brush_codes[str(step.get("brush_code"))] += 1
            if step.get("affix"):
                trajectory_affixes[str(step.get("affix"))] += 1
            if step.get("affix_code"):
                trajectory_affix_codes[str(step.get("affix_code"))] += 1
            chosen_brush_key = str(chosen_brush_code or chosen_brush or "")
            if chosen_brush_key and chosen_brush_key != previous_brush_choice:
                if chosen_brush:
                    imitation_brush_choices[str(chosen_brush)] += 1
                if chosen_brush_code:
                    imitation_brush_choice_codes[str(chosen_brush_code)] += 1
                previous_brush_choice = chosen_brush_key
            active_affix = step.get("active_affix") if isinstance(step.get("active_affix"), dict) else {}
            chosen_affix = active_affix.get("name") or step.get("affix")
            chosen_affix_code = active_affix.get("code") or step.get("affix_code")
            chosen_affix_key = str(chosen_affix_code or chosen_affix or "")
            if chosen_affix_key and chosen_affix_key != previous_affix_choice:
                if chosen_affix:
                    imitation_affix_choices[str(chosen_affix)] += 1
                if chosen_affix_code:
                    imitation_affix_choice_codes[str(chosen_affix_code)] += 1
                previous_affix_choice = chosen_affix_key
            chosen_recommendation = step.get("chosen_recommendation")
            if isinstance(chosen_recommendation, dict):
                imitation_recommendation_hits["decisions"] += 1
                if chosen_recommendation.get("matched") is True:
                    imitation_recommendation_hits["hit"] += 1
                    rank = chosen_recommendation.get("rank")
                    if isinstance(rank, int):
                        imitation_recommendation_hits[f"rank_{rank}"] += 1
                else:
                    imitation_recommendation_hits["miss"] += 1
                rec = chosen_recommendation.get("recommendation")
                if isinstance(rec, dict):
                    explanation = rec.get("explanation") if isinstance(rec.get("explanation"), dict) else {}
                    short = explanation.get("short")
                    if short:
                        imitation_recommendation_shorts[str(short)] += 1
                    for tag in explanation.get("tags") or ():
                        imitation_recommendation_tags[str(tag)] += 1
            _accumulate_recommendation_stats(
                step.get("top") or (),
                cells=recommendation_cells,
                shorts=recommendation_shorts,
                tags=recommendation_tags,
            )

    total_backpack = sum(backpack_levels.values())
    field_accuracy = {}
    risky_fields = []
    for field, samples in correction_samples.items():
        errors = correction_fields[field]
        error_rate = errors / samples if samples else 0.0
        field_accuracy[field] = {
            "samples": samples,
            "errors": errors,
            "accuracy": 1.0 - error_rate,
            "error_rate": error_rate,
        }
        if samples >= 3 and error_rate >= 0.25:
            risky_fields.append(field)
    risky_fields.sort(key=lambda field: (field_accuracy[field]["error_rate"], field_accuracy[field]["errors"]), reverse=True)
    policy_prior_samples = {
        "placements": sum(placement_cells.values()),
        "brushes": max(sum(trajectory_brushes.values()), sum(trajectory_brush_codes.values())),
        "affixes": max(sum(trajectory_affixes.values()), sum(trajectory_affix_codes.values())),
    }
    imitation_policy = _imitation_policy_summary(
        trajectory_count=len(trajectories),
        trajectory_scores=trajectory_scores,
        action_counts=imitation_actions,
        action_codes=imitation_action_codes,
        action_score_delta=imitation_action_score_delta,
        action_score_samples=imitation_action_score_samples,
        placement_cells=placement_cells,
        placement_zones=placement_zones,
        brush_choices=imitation_brush_choices,
        brush_choice_codes=imitation_brush_choice_codes,
        affix_choices=imitation_affix_choices,
        affix_choice_codes=imitation_affix_choice_codes,
        recommendation_hits=imitation_recommendation_hits,
        recommendation_shorts=imitation_recommendation_shorts,
        recommendation_tags=imitation_recommendation_tags,
    )
    return {
        "total_events": len(events),
        "event_counts": dict(event_counts),
        "stage1_seen": dict(stage1_seen),
        "stage1_code_seen": dict(stage1_code_seen),
        "stage1_seen_total": sum(stage1_seen.values()),
        "stage1_code_seen_total": sum(stage1_code_seen.values()),
        "stage1_seen_pct": _counter_percentages(stage1_seen),
        "stage1_code_seen_pct": _counter_percentages(stage1_code_seen),
        "unknown_stage1_seen": dict(unknown_stage1_seen),
        "unknown_stage1_seen_total": sum(unknown_stage1_seen.values()),
        "unknown_stage1_seen_pct": _counter_percentages(unknown_stage1_seen),
        "affix_seen": dict(affix_seen),
        "affix_code_seen": dict(affix_code_seen),
        "affix_seen_total": sum(affix_seen.values()),
        "affix_code_seen_total": sum(affix_code_seen.values()),
        "affix_seen_pct": _counter_percentages(affix_seen),
        "affix_code_seen_pct": _counter_percentages(affix_code_seen),
        "unknown_affix_seen": dict(unknown_affix_seen),
        "unknown_affix_seen_total": sum(unknown_affix_seen.values()),
        "unknown_affix_seen_pct": _counter_percentages(unknown_affix_seen),
        "unknown_affix_samples": unknown_affix_samples,
        "unknown_affix_appearance_rates": _counter_distribution(unknown_affix_seen),
        "new_affix_candidates": _unknown_affix_candidates(unknown_affix_seen, unknown_affix_samples),
        "affix_appearance_rates": {
            "stage1": _counter_distribution(stage1_seen),
            "stage1_codes": _counter_distribution(stage1_code_seen),
            "stage2": _counter_distribution(affix_seen),
            "stage2_codes": _counter_distribution(affix_code_seen),
            "unknown_stage1": _counter_distribution(unknown_stage1_seen),
            "unknown_stage2": _counter_distribution(unknown_affix_seen),
        },
        "current_paint_seen": dict(current_seen),
        "current_paint_code_seen": dict(current_code_seen),
        "rule_mismatches": dict(mismatch_seen),
        "rule_anomalies": dict(anomaly_seen),
        "rule_anomaly_severity": dict(anomaly_severity),
        "backpack_levels": dict(backpack_levels),
        "backpack_categories": dict(backpack_categories),
        "corrections": len(corrections),
        "correction_fields": dict(correction_fields),
        "correction_samples": dict(correction_samples),
        "field_accuracy": field_accuracy,
        "risky_fields": risky_fields,
        "trajectories": len(trajectories),
        "trajectory_actions": dict(trajectory_actions),
        "inferred_actions": dict(inferred_actions),
        "inferred_action_codes": dict(inferred_action_codes),
        "inferred_action_details": dict(inferred_action_details),
        "placement_cells": dict(placement_cells),
        "placement_zones": dict(placement_zones),
        "recommendation_cells": dict(recommendation_cells),
        "recommendation_shorts": dict(recommendation_shorts),
        "recommendation_tags": dict(recommendation_tags),
        "high_value_recommendation_shorts": dict(high_value_recommendation_shorts),
        "high_value_recommendation_tags": dict(high_value_recommendation_tags),
        "upgrade_events": upgrade_events,
        "forced_events": forced_events,
        "refresh_events": dict(refresh_events),
        "trajectory_brushes": dict(trajectory_brushes),
        "trajectory_brush_codes": dict(trajectory_brush_codes),
        "trajectory_affixes": dict(trajectory_affixes),
        "trajectory_affix_codes": dict(trajectory_affix_codes),
        "high_score_gate_primary": dict(high_score_gate_primary),
        "high_score_gate_reasons": dict(high_score_gate_reasons),
        "policy_prior_samples": policy_prior_samples,
        "imitation_samples": imitation_policy["samples"],
        "imitation_ready": imitation_policy["ready"],
        "imitation_policy": imitation_policy,
        "trajectory_best_score": max(trajectory_scores) if trajectory_scores else None,
        "trajectory_avg_score": (sum(trajectory_scores) / len(trajectory_scores)) if trajectory_scores else None,
        "treasure_ratio": treasure_count / total_backpack if total_backpack else 0.0,
        "quick_capture_steps": quick_steps,
        "quick_capture_ignored": ignored_quick_steps,
        "trajectory_step_ignored": ignored_trajectory_steps,
        "quick_capture_actions": dict(quick_actions),
        "quick_capture_avg_elapsed": (sum(quick_elapsed) / len(quick_elapsed)) if quick_elapsed else None,
        "finished_runs": finished_runs,
        "finished_skipped": skipped_finished_runs,
        "finished_best_score": max(finished_scores) if finished_scores else None,
        "finished_avg_score": (sum(finished_scores) / len(finished_scores)) if finished_scores else None,
        "high_value_seen": high_value_seen,
        "high_value_best_score": high_value_best_score if high_value_seen else None,
        "high_value_counts": dict(high_value_counts),
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


def _counter_percentages(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {
        str(key): round(float(count) * 100.0 / float(total), 3)
        for key, count in counter.items()
    }


def _counter_distribution(counter: Counter) -> dict[str, Any]:
    total = sum(counter.values())
    items = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        rate = float(count) / float(total) if total else 0.0
        items.append(
            {
                "key": str(key),
                "count": int(count),
                "rate": round(rate, 6),
                "pct": round(rate * 100.0, 3),
            }
        )
    return {
        "total": int(total),
        "items": items,
    }


def _unknown_text(value: Any) -> str | None:
    text = str(value).strip()
    if not text or text.casefold() in {"none", "null", "unknown", "?"}:
        return None
    return text


def _append_unknown_affix_sample(
    samples: dict[str, list[dict[str, Any]]],
    name: str,
    payload: dict[str, Any],
    event: dict[str, Any],
) -> None:
    bucket = samples.setdefault(name, [])
    if len(bucket) >= 5:
        return
    scores = payload.get("stage2_scores") if isinstance(payload.get("stage2_scores"), dict) else {}
    stage1_scores = payload.get("stage1_scores") if isinstance(payload.get("stage1_scores"), dict) else {}
    sample = {
        "ts": event.get("ts"),
        "confidence": payload.get("confidence"),
        "score": scores.get(name),
        "chosen_stage1": payload.get("chosen_stage1"),
        "chosen_stage1_code": payload.get("chosen_stage1_code"),
        "stage1_options": list(payload.get("stage1_options") or [])[:3],
        "stage1_codes": list(payload.get("stage1_codes") or [])[:3],
    }
    if stage1_scores:
        sample["stage1_scores"] = {
            str(key): round(float(value), 3)
            for key, value in stage1_scores.items()
            if isinstance(value, (int, float))
        }
    bucket.append(sample)


def _unknown_affix_candidates(
    counter: Counter,
    samples: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    total = sum(counter.values())
    candidates: list[dict[str, Any]] = []
    for name, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        sample_list = samples.get(str(name), [])
        score_values = [
            float(sample.get("score"))
            for sample in sample_list
            if isinstance(sample.get("score"), (int, float))
        ]
        confidence_values = [
            float(sample.get("confidence"))
            for sample in sample_list
            if isinstance(sample.get("confidence"), (int, float))
        ]
        rate = float(count) / float(total) if total else 0.0
        candidates.append(
            {
                "name": str(name),
                "count": int(count),
                "rate": round(rate, 6),
                "pct": round(rate * 100.0, 3),
                "avg_score": round(sum(score_values) / len(score_values), 3) if score_values else None,
                "avg_confidence": round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else None,
                "samples": sample_list,
            }
        )
    return candidates


def _imitation_policy_summary(
    *,
    trajectory_count: int,
    trajectory_scores: list[float],
    action_counts: Counter,
    action_codes: Counter,
    action_score_delta: Counter,
    action_score_samples: Counter,
    placement_cells: Counter,
    placement_zones: Counter,
    brush_choices: Counter,
    brush_choice_codes: Counter,
    affix_choices: Counter,
    affix_choice_codes: Counter,
    recommendation_hits: Counter,
    recommendation_shorts: Counter,
    recommendation_tags: Counter,
) -> dict[str, Any]:
    samples = {
        "trajectories": int(trajectory_count),
        "actions": int(sum(action_counts.values())),
        "placements": int(sum(placement_cells.values())),
        "brushes": int(max(sum(brush_choices.values()), sum(brush_choice_codes.values()))),
        "affixes": int(max(sum(affix_choices.values()), sum(affix_choice_codes.values()))),
        "recommendation_decisions": int(recommendation_hits.get("decisions", 0)),
    }
    thresholds = {
        "actions": _config_int("learning.action_min_samples", 30),
        "placements": _config_int("learning.placement_min_samples", 30),
        "brushes": _config_int("learning.brush_min_samples", 20),
        "affixes": _config_int("learning.affix_min_samples", 20),
    }
    ready = {
        key: samples.get(key, 0) >= threshold
        for key, threshold in thresholds.items()
    }
    ready["recommendations"] = samples["recommendation_decisions"] >= thresholds["placements"]
    return {
        "samples": samples,
        "thresholds": thresholds,
        "ready": ready,
        "top_actions": _counter_distribution(action_counts),
        "top_action_codes": _counter_distribution(action_codes),
        "action_score_delta": _average_delta_distribution(action_score_delta, action_score_samples),
        "top_placement_cells": _counter_distribution(placement_cells),
        "top_placement_zones": _counter_distribution(placement_zones),
        "top_brush_choices": _counter_distribution(brush_choices),
        "top_brush_codes": _counter_distribution(brush_choice_codes),
        "top_affix_choices": _counter_distribution(affix_choices),
        "top_affix_codes": _counter_distribution(affix_choice_codes),
        "recommendation_hits": _recommendation_hit_summary(recommendation_hits),
        "recommendation_shorts": _counter_distribution(recommendation_shorts),
        "recommendation_tags": _counter_distribution(recommendation_tags),
        "score": {
            "best": max(trajectory_scores) if trajectory_scores else None,
            "avg": round(sum(trajectory_scores) / len(trajectory_scores), 3) if trajectory_scores else None,
        },
    }


def _config_int(path: str, default: int) -> int:
    try:
        return int(config_value(path, default))
    except (TypeError, ValueError):
        return int(default)


def _average_delta_distribution(sums: Counter, counts: Counter) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))):
        if count <= 0:
            continue
        avg = float(sums.get(key, 0.0)) / float(count)
        items.append(
            {
                "key": str(key),
                "count": int(count),
                "avg_score_delta": round(avg, 3),
            }
        )
    return {
        "total": int(sum(counts.values())),
        "items": items,
    }


def _recommendation_hit_summary(counter: Counter) -> dict[str, Any]:
    decisions = int(counter.get("decisions", 0))
    hits = int(counter.get("hit", 0))
    misses = int(counter.get("miss", 0))
    top1 = int(counter.get("rank_1", 0))
    return {
        "decisions": decisions,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(float(hits) / float(decisions), 6) if decisions else 0.0,
        "hit_pct": round(float(hits) * 100.0 / float(decisions), 3) if decisions else 0.0,
        "top1": top1,
        "top1_rate": round(float(top1) / float(decisions), 6) if decisions else 0.0,
        "top1_pct": round(float(top1) * 100.0 / float(decisions), 3) if decisions else 0.0,
        "rank_counts": {
            key: int(counter.get(key, 0))
            for key in sorted((key for key in counter if str(key).startswith("rank_")), key=str)
        },
    }


def _placement_cells_from_inferred(inferred: dict[str, Any]) -> list[dict[str, Any]]:
    cells = [cell for cell in inferred.get("changed_cells") or [] if isinstance(cell, dict)]
    installed = [
        cell
        for cell in cells
        if cell.get("to_state") == "P"
        and (cell.get("from_state") != "P" or cell.get("from_paint") is None)
    ]
    if installed:
        return installed[:3]
    if inferred.get("type") not in {"install_paint", "place_current_paint"}:
        return []
    # Fallback for old trajectory payloads that did not include state transitions.
    return cells[:1]


def _accumulate_recommendation_stats(
    recommendations: Any,
    *,
    cells: Counter,
    shorts: Counter,
    tags: Counter,
    high_value_shorts: Counter | None = None,
    high_value_tags: Counter | None = None,
) -> None:
    if not isinstance(recommendations, (list, tuple)):
        return
    for rec in recommendations:
        if not isinstance(rec, dict):
            continue
        row = rec.get("r")
        col = rec.get("c")
        if isinstance(row, int) and isinstance(col, int):
            cells[f"{row},{col}"] += 1
        explanation = rec.get("explanation") if isinstance(rec.get("explanation"), dict) else {}
        short = explanation.get("short")
        if short:
            shorts[str(short)] += 1
            if high_value_shorts is not None:
                high_value_shorts[str(short)] += 1
        for tag in explanation.get("tags") or ():
            tags[str(tag)] += 1
            if high_value_tags is not None:
                high_value_tags[str(tag)] += 1


def _cell_zone(row: int, col: int) -> str:
    if row in {0, 5} and col in {0, 5}:
        return "corner"
    if row in {0, 5} or col in {0, 5}:
        return "edge"
    if row in {2, 3} and col in {2, 3}:
        return "center"
    return "inner"


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
