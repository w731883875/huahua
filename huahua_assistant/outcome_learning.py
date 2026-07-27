from __future__ import annotations

import hashlib
import json
import math
import statistics
from copy import deepcopy
from typing import Any, Mapping, Sequence


DECISION_SCHEMA = "huahua-decision-v2"
RUN_SCHEMA = "huahua-run-outcome-v2"
RISK_SCHEMA = "huahua-risk-summary-v1"
DEFAULT_MIN_COMPARABLE_RUNS = 20
STRATEGY_MODES = ("stable", "balanced", "gamble")

_RUN_EVENT_PRIORITY = {
    "run_session_finished": 40,
    "compressed_run_summary": 30,
    "run_session_failed": 25,
    "run_session_finish_skipped": 20,
    "high_score_trajectory": 10,
}


def normalize_observation_payload(
    event_type: str,
    payload: Mapping[str, Any],
    *,
    session_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Add the versioned learning contract while preserving legacy fields."""
    if event_type == "recommendation_decision":
        return normalize_decision_payload(payload)
    if event_type in _RUN_EVENT_PRIORITY:
        return normalize_run_payload(payload, event_type=event_type, session_events=session_events)
    return deepcopy(dict(payload))


def normalize_trajectory_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_run_payload(payload, event_type="high_score_trajectory")


def decision_records_from_steps(steps: Sequence[Mapping[str, Any]], *, session: Any = None) -> list[dict[str, Any]]:
    return _decisions_from_run({"session": session, "steps": list(steps)})


def normalize_run_payload(
    payload: Mapping[str, Any],
    *,
    event_type: str,
    session_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    normalized = deepcopy(dict(payload))
    normalized["schema"] = RUN_SCHEMA
    normalized["versions"] = _version_contract(payload)
    normalized["strategy_mode"] = _strategy_mode(payload)
    normalized["observed_outcome"] = _observed_outcome(payload, event_type=event_type)
    normalized["proxy_score"] = _proxy_score(payload)
    normalized["resource_usage"] = _resource_usage(payload)

    decisions = _decisions_from_run(payload)
    if not decisions:
        decisions = _decisions_from_events(payload.get("session"), session_events)
    normalized["trajectory_trust"] = _trajectory_trust(payload, decisions)

    retention = "trusted_full_trajectory" if event_type == "high_score_trajectory" else "lightweight_summary"
    normalized["retention"] = retention
    if retention == "trusted_full_trajectory":
        normalized["decision_records"] = decisions
    else:
        normalized.pop("decision_records", None)
        normalized.pop("decisions", None)
        normalized["ordinary_summary"] = _ordinary_summary(payload, decisions, event_type=event_type)
    return normalized


def normalize_decision_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = deepcopy(dict(payload))
    candidates_raw = source.get("candidates")
    if not isinstance(candidates_raw, list):
        candidates_raw = source.get("top") if isinstance(source.get("top"), list) else []
    candidates = [
        normalize_candidate(candidate, rank=index)
        for index, candidate in enumerate(candidates_raw, start=1)
        if isinstance(candidate, Mapping)
    ]

    choice_raw = source.get("actual_choice")
    if not isinstance(choice_raw, Mapping):
        choice_raw = source.get("chosen_recommendation")
    actual_choice = _actual_choice(choice_raw, candidates)
    rank = actual_choice.get("rank") if isinstance(actual_choice, dict) else None
    quality = source.get("recognition_quality")
    if not isinstance(quality, Mapping):
        quality = source.get("quality") if isinstance(source.get("quality"), Mapping) else {}
    transition = _transition_contract(source.get("transition", source.get("outcome")))
    result_quality = source.get("result_quality") if isinstance(source.get("result_quality"), Mapping) else {}
    trusted = _quality_is_trusted(quality) and (not result_quality or _quality_is_trusted(result_quality))
    action = source.get("action") if isinstance(source.get("action"), Mapping) else {}
    action_type = str(action.get("type") or source.get("action_type") or "")
    action_confidence = _finite_number(action.get("confidence", source.get("action_confidence")))
    action_eligible = not action_type or action_type in {"install_paint", "place_current_paint", "placement"}
    confidence_eligible = action_confidence is None or action_confidence >= 0.65
    attributable = bool(
        trusted
        and action_eligible
        and confidence_eligible
        and actual_choice
        and actual_choice.get("matched") is True
    )

    state_signature = source.get("state_signature") or _state_signature(source)
    session = source.get("session")
    decision_id = source.get("decision_id") or _stable_id(
        {
            "session": session,
            "state": state_signature,
            "index": source.get("transition_index", source.get("step_index")),
            "choice": actual_choice,
        }
    )
    return {
        "schema": DECISION_SCHEMA,
        "decision_id": decision_id,
        "session": session,
        "state_signature": state_signature,
        "phase": _decision_phase(source),
        "recognition_quality": deepcopy(dict(quality)),
        "candidates": candidates,
        "actual_choice": actual_choice,
        "top1_hit": rank == 1 if isinstance(rank, int) else (False if actual_choice else None),
        "top3_hit": rank <= 3 if isinstance(rank, int) else (False if actual_choice else None),
        "transition": transition,
        "resources": deepcopy(dict(source.get("resources") or {})),
        "attribution": {
            "trusted": trusted,
            "action_type": action_type or None,
            "action_confidence": action_confidence,
            "outcome_attributable": attributable,
            "eligible_for_policy_evaluation": attributable,
        },
    }


def normalize_candidate(candidate: Mapping[str, Any], *, rank: int) -> dict[str, Any]:
    source = deepcopy(dict(candidate))
    row = source.get("row", source.get("r"))
    col = source.get("col", source.get("c"))
    paint = source.get("paint") if isinstance(source.get("paint"), Mapping) else None
    explicit_policy = source.get("policy_score")
    if isinstance(explicit_policy, Mapping):
        value = _finite_number(explicit_policy.get("value"))
        components = explicit_policy.get("components")
    else:
        value = _finite_number(explicit_policy)
        if value is None:
            value = _finite_number(source.get("score"))
        explanation = source.get("explanation") if isinstance(source.get("explanation"), Mapping) else {}
        components = explanation.get("layers")
    components = dict(components) if isinstance(components, Mapping) else {}
    numeric_components = {
        str(key): number
        for key, raw in components.items()
        if (number := _finite_number(raw)) is not None
    }
    risk_value = numeric_components.get("risk")

    candidate_id = str(source.get("candidate_id") or _candidate_id(row, col, paint))
    action_id = str(source.get("action_id") or _action_id(row, col))
    paint_id = _paint_id(paint)
    return {
        "candidate_id": candidate_id,
        "canonical_ids": {"action": action_id, "paint": paint_id},
        "rank": int(source.get("rank") or rank),
        "legal": source.get("legal") is not False,
        "target": {"row": row, "col": col},
        "paint": deepcopy(dict(paint)) if paint is not None else None,
        "completes_match": bool(source.get("complete", source.get("completes_match", False))),
        "policy_score": {"value": value, "components": numeric_components},
        "risk": {"value": risk_value, "components": {"aggregate": risk_value} if risk_value is not None else {}},
    }


def build_outcome_risk_report(
    events: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]] = (),
    *,
    min_samples: int = DEFAULT_MIN_COMPARABLE_RUNS,
) -> dict[str, Any]:
    minimum = max(1, int(min_samples))
    runs = _deduplicated_runs(events, trajectories)
    mode_reports = {
        mode: _mode_report([run for run in runs if run.get("strategy_mode") == mode], minimum)
        for mode in STRATEGY_MODES
    }
    unclassified = [run for run in runs if run.get("strategy_mode") not in STRATEGY_MODES]
    return {
        "schema": RISK_SCHEMA,
        "minimum_comparable_runs": minimum,
        "runs_total": len(runs),
        "deduplicated_sessions": len(runs),
        "unclassified_runs": len(unclassified),
        "unclassified": _mode_report(unclassified, minimum),
        "modes": mode_reports,
        "personalization_ready": any(report["personalization_ready"] for report in mode_reports.values()),
        "style_label": None,
    }


def _deduplicated_runs(
    events: Sequence[Mapping[str, Any]], trajectories: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    selected: dict[str, tuple[int, float, dict[str, Any]]] = {}
    for event in (*events, *trajectories):
        if not isinstance(event, Mapping) or event.get("source") != "real_capture":
            continue
        event_type = str(event.get("event") or "")
        if event_type not in _RUN_EVENT_PRIORITY:
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        run = normalize_run_payload(payload, event_type=event_type)
        session = str(run.get("session") or f"anonymous:{_stable_id(payload)}")
        priority = _RUN_EVENT_PRIORITY[event_type]
        timestamp = float(_finite_number(event.get("ts")) or 0.0)
        previous = selected.get(session)
        if previous is None or (priority, timestamp) >= (previous[0], previous[1]):
            selected[session] = (priority, timestamp, run)
    return [selected[key][2] for key in sorted(selected)]


def _mode_report(runs: list[dict[str, Any]], minimum: int) -> dict[str, Any]:
    actual_scores = [
        score
        for run in runs
        if (score := _finite_number((run.get("observed_outcome") or {}).get("actual_score"))) is not None
    ]
    proxy_scores = [
        score
        for run in runs
        if (score := _finite_number((run.get("proxy_score") or {}).get("value"))) is not None
    ]
    if actual_scores:
        basis = "observed_outcome"
        scores = actual_scores
        outcome_status = "observed"
    elif proxy_scores:
        basis = "proxy_score"
        scores = proxy_scores
        outcome_status = "proxy_only"
    else:
        basis = "unavailable"
        scores = []
        outcome_status = "unavailable"

    groups: dict[str, int] = {}
    for run in runs:
        key = _comparable_group_key(run)
        groups[key] = groups.get(key, 0) + 1
    largest_group = max(groups.values(), default=0)
    enough = len(scores) >= minimum and largest_group >= minimum
    status = "ready" if enough and basis == "observed_outcome" else "observing"
    return {
        "status": status,
        "outcome_status": outcome_status,
        "metric_basis": basis,
        "samples": len(runs),
        "scored_samples": len(scores),
        "minimum_samples": minimum,
        "trend_visible": status == "ready",
        "personalization_ready": status == "ready",
        "style_label": None,
        "largest_comparable_group": largest_group,
        "comparable_groups": dict(sorted(groups.items())),
        "metrics": _risk_metrics(runs, scores),
    }


def _risk_metrics(runs: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    ordered = sorted(scores)
    count = len(ordered)
    bottom_count = max(1, math.ceil(count * 0.2)) if count else 0
    high = sum(1 for run in runs if _high_score_hit(run))
    failures = sum(1 for run in runs if _run_failed(run))
    low_value = sum(1 for run in runs if _low_value_run(run))
    failure_or_low = sum(1 for run in runs if _run_failed(run) or _low_value_run(run))
    refresh = [_finite_number((run.get("resource_usage") or {}).get("refresh_used")) or 0.0 for run in runs]
    force = [_finite_number((run.get("resource_usage") or {}).get("force_used")) or 0.0 for run in runs]
    denominator = len(runs)
    return {
        "mean": _rounded(statistics.fmean(ordered)) if ordered else None,
        "median": _rounded(statistics.median(ordered)) if ordered else None,
        "stddev": _rounded(statistics.pstdev(ordered)) if ordered else None,
        "iqr": _rounded(_percentile(ordered, 0.75) - _percentile(ordered, 0.25)) if ordered else None,
        "p10": _rounded(_percentile(ordered, 0.10)) if ordered else None,
        "p20": _rounded(_percentile(ordered, 0.20)) if ordered else None,
        "p80": _rounded(_percentile(ordered, 0.80)) if ordered else None,
        "p90": _rounded(_percentile(ordered, 0.90)) if ordered else None,
        "bottom_20_mean": _rounded(statistics.fmean(ordered[:bottom_count])) if bottom_count else None,
        "high_score_rate": _rate(high, denominator),
        "failure_rate": _rate(failures, denominator),
        "low_value_rate": _rate(low_value, denominator),
        "failure_or_low_value_rate": _rate(failure_or_low, denominator),
        "refresh_used_mean": _rounded(statistics.fmean(refresh)) if refresh else None,
        "force_used_mean": _rounded(statistics.fmean(force)) if force else None,
    }


def _decisions_from_run(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = payload.get("decision_records", payload.get("decisions"))
    if isinstance(explicit, list):
        return [normalize_decision_payload(item) for item in explicit if isinstance(item, Mapping)]
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return []
    decisions: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        action = step.get("inferred_action") if isinstance(step.get("inferred_action"), Mapping) else {}
        if previous is not None and action.get("type") not in {None, "session_start", "no_material_change"}:
            context = step.get("decision_context") if isinstance(step.get("decision_context"), Mapping) else {}
            decision_source = dict(context)
            candidates = previous.get("candidates", previous.get("top", []))
            chosen = step.get("chosen_recommendation")
            if not isinstance(chosen, Mapping):
                chosen = _choice_from_action(action, candidates)
            outcome = step.get("outcome")
            if not isinstance(outcome, Mapping):
                outcome = {
                    "changed_cells": action.get("changed_cells") or [],
                    "summary": action.get("summary") or {},
                }
            decision_source.update(
                {
                    "session": payload.get("session"),
                    "transition_index": step.get("transition_index", step.get("step_index")),
                    "quality": previous.get("quality") or {},
                    "result_quality": step.get("quality") or {},
                    "resources": previous.get("resources") or {},
                    "board": previous.get("board") or {},
                    "current": previous.get("current"),
                    "candidates": candidates,
                    "chosen_recommendation": chosen,
                    "outcome": outcome,
                    "action": action,
                }
            )
            decisions.append(normalize_decision_payload(decision_source))
        previous = step
    return decisions


def _decisions_from_events(session: Any, events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not session:
        return []
    decisions: list[dict[str, Any]] = []
    steps: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or payload.get("session") != session:
            continue
        if event.get("event") == "recommendation_decision":
            decisions.append(normalize_decision_payload(payload))
        elif event.get("event") in {"quick_capture_step", "capture_step"}:
            steps.append(payload)
    decisions.extend(decision_records_from_steps(steps, session=session))
    return decisions


def _choice_from_action(action: Mapping[str, Any], candidates: Any) -> dict[str, Any] | None:
    if not isinstance(candidates, list):
        return None
    summary = action.get("summary") if isinstance(action.get("summary"), Mapping) else {}
    raw_cells = summary.get("placed_cells") if isinstance(summary.get("placed_cells"), list) else []
    cells: list[list[Any]] = [list(cell[:2]) for cell in raw_cells if isinstance(cell, (list, tuple)) and len(cell) >= 2]
    if not cells:
        for changed in action.get("changed_cells") or []:
            if not isinstance(changed, Mapping) or changed.get("to_state") != "P":
                continue
            cells.append([changed.get("r"), changed.get("c")])
    if not cells:
        return None
    for cell in cells:
        for rank, candidate in enumerate(candidates, start=1):
            if isinstance(candidate, Mapping) and candidate.get("r", candidate.get("row")) == cell[0] and candidate.get("c", candidate.get("col")) == cell[1]:
                return {"matched": True, "rank": rank, "cell": cell, "recommendation": candidate}
    return {"matched": False, "cells": cells[:3]}


def _observed_outcome(payload: Mapping[str, Any], *, event_type: str) -> dict[str, Any]:
    existing = payload.get("observed_outcome") if isinstance(payload.get("observed_outcome"), Mapping) else {}
    actual_score = _first_finite(
        existing.get("actual_score"),
        payload.get("actual_score"),
        payload.get("observed_score"),
        payload.get("game_score"),
    )
    finish_seen = bool(existing.get("finish_seen", payload.get("finish_seen", False)))
    terminal = existing.get("terminal")
    if not isinstance(terminal, Mapping):
        terminal = payload.get("terminal") if isinstance(payload.get("terminal"), Mapping) else {}
    status = existing.get("status")
    if not status:
        if event_type == "run_session_failed":
            status = "failure"
        elif event_type in {"run_session_finished", "compressed_run_summary"} or finish_seen:
            status = "observed_terminal"
        elif event_type == "run_session_finish_skipped":
            status = "partial"
        else:
            status = "unknown"
    quality = existing.get("quality", payload.get("quality"))
    if quality is None and isinstance(terminal, Mapping):
        quality = terminal.get("quality")
    return {
        "status": str(status),
        "actual_score": actual_score,
        "terminal": deepcopy(dict(terminal)),
        "finish_seen": finish_seen,
        "quality": deepcopy(quality),
        "failure_reason": payload.get("failure_reason", payload.get("skip_reason")),
    }


def _proxy_score(payload: Mapping[str, Any]) -> dict[str, Any]:
    existing = payload.get("proxy_score")
    if isinstance(existing, Mapping):
        value = _finite_number(existing.get("value"))
        model = existing.get("model")
        breakdown = existing.get("breakdown")
    else:
        value = _finite_number(existing)
        model = None
        breakdown = None
    if value is None:
        value = _first_finite(payload.get("internal_score"), payload.get("final_score"))
    if model is None:
        model = payload.get("score_model")
    if not isinstance(breakdown, Mapping):
        breakdown = payload.get("score_breakdown") if isinstance(payload.get("score_breakdown"), Mapping) else {}
    return {"value": value, "model": model or "unknown", "breakdown": deepcopy(dict(breakdown))}


def _ordinary_summary(
    payload: Mapping[str, Any], decisions: list[dict[str, Any]], *, event_type: str
) -> dict[str, Any]:
    terminal = payload.get("terminal") if isinstance(payload.get("terminal"), Mapping) else {}
    initial = payload.get("initial_snapshot") if isinstance(payload.get("initial_snapshot"), Mapping) else {}
    quality = payload.get("quality")
    if quality is None and isinstance(terminal, Mapping):
        quality = terminal.get("quality")
    major = [
        {
            "decision_id": decision.get("decision_id"),
            "phase": decision.get("phase"),
            "actual_choice": decision.get("actual_choice"),
            "top1_hit": decision.get("top1_hit"),
            "top3_hit": decision.get("top3_hit"),
            "attribution": decision.get("attribution"),
        }
        for decision in decisions
        if decision.get("actual_choice") is not None
    ][-12:]
    return {
        "initial_map_bucket": _initial_bucket(initial),
        "major_decisions": major,
        "resource_usage": _resource_usage(payload),
        "terminal_distribution": deepcopy(dict(terminal)),
        "failure_reason": payload.get("failure_reason", payload.get("skip_reason")),
        "recognition_quality": deepcopy(quality),
        "source_event": event_type,
    }


def _resource_usage(payload: Mapping[str, Any]) -> dict[str, Any]:
    existing = payload.get("resource_usage")
    if isinstance(existing, Mapping):
        return {
            "force_used": _finite_number(existing.get("force_used")) or 0.0,
            "refresh_used": _finite_number(existing.get("refresh_used")) or 0.0,
        }
    steps = payload.get("steps")
    resources: list[Mapping[str, Any]] = []
    if isinstance(steps, list):
        resources = [step.get("resources") for step in steps if isinstance(step, Mapping) and isinstance(step.get("resources"), Mapping)]
    direct = payload.get("resources") if isinstance(payload.get("resources"), Mapping) else {}
    force_used = max((_finite_number(item.get("force_used")) or 0.0 for item in resources), default=0.0)
    force_used = max(force_used, _finite_number(direct.get("force_used")) or 0.0)
    refresh_used = _finite_number(direct.get("refresh_used")) or 0.0
    refresh_remaining = [_finite_number(item.get("refresh_remaining")) for item in resources]
    refresh_remaining = [value for value in refresh_remaining if value is not None]
    if len(refresh_remaining) >= 2:
        refresh_used = max(refresh_used, max(0.0, refresh_remaining[0] - refresh_remaining[-1]))
    action_counts = payload.get("trajectory_summary")
    if isinstance(action_counts, Mapping):
        action_counts = action_counts.get("action_counts")
    if isinstance(action_counts, Mapping):
        refresh_used = max(refresh_used, float(action_counts.get("refresh", 0) or 0))
    return {"force_used": force_used, "refresh_used": refresh_used}


def _trajectory_trust(payload: Mapping[str, Any], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    steps = payload.get("steps")
    trusted_steps = 0
    step_count = 0
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            step_count += 1
            if _quality_is_trusted(step.get("quality") if isinstance(step.get("quality"), Mapping) else {}):
                trusted_steps += 1
    gate = payload.get("high_score_gate") if isinstance(payload.get("high_score_gate"), Mapping) else {}
    trusted = bool(step_count and trusted_steps == step_count and gate.get("passed", True) is True)
    attributable = sum(1 for decision in decisions if (decision.get("attribution") or {}).get("outcome_attributable") is True)
    return {
        "trusted": trusted,
        "step_count": step_count,
        "trusted_step_count": trusted_steps,
        "attributable_decisions": attributable,
    }


def _transition_contract(value: Any) -> dict[str, Any]:
    source = dict(value) if isinstance(value, Mapping) else {}
    proxy_delta = _first_finite(source.get("proxy_score_delta"), source.get("score_delta"))
    proxy_to_final = _first_finite(source.get("proxy_score_to_final"), source.get("score_to_final"))
    return {
        "before": deepcopy(source.get("before")),
        "after": deepcopy(source.get("after")),
        "changed_cells": deepcopy(source.get("changed_cells") or []),
        "result_summary": deepcopy(source.get("summary") or {}),
        "proxy_score_delta": proxy_delta,
        "proxy_score_to_final": proxy_to_final,
        "terminal": deepcopy(source.get("terminal")),
    }


def _actual_choice(value: Any, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    source = dict(value)
    rank = source.get("rank")
    try:
        rank = int(rank) if rank is not None else None
    except (TypeError, ValueError):
        rank = None
    matched = source.get("matched") is True
    selected = next((candidate for candidate in candidates if rank is not None and candidate.get("rank") == rank), None)
    recommendation = source.get("recommendation")
    if selected is None and isinstance(recommendation, Mapping):
        selected = normalize_candidate(recommendation, rank=rank or len(candidates) + 1)
    cells = source.get("cells") or ([source.get("cell")] if source.get("cell") else [])
    return {
        "matched": matched,
        "rank": rank,
        "candidate_id": selected.get("candidate_id") if selected else None,
        "canonical_ids": selected.get("canonical_ids") if selected else {},
        "cells": deepcopy(cells),
    }


def _version_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    versions = payload.get("versions") if isinstance(payload.get("versions"), Mapping) else {}
    return {
        "schema": RUN_SCHEMA,
        "game": versions.get("game", payload.get("game_version", "unknown")),
        "recognizer": versions.get("recognizer", payload.get("recognizer_version", "unknown")),
        "policy": versions.get("policy", payload.get("policy_version", "builtin-rules-v1")),
    }


def _strategy_mode(payload: Mapping[str, Any]) -> str:
    value = str(payload.get("strategy_mode", payload.get("risk_mode", "unknown"))).strip().casefold()
    aliases = {
        "stable": "stable",
        "safe": "stable",
        "稳健": "stable",
        "balanced": "balanced",
        "balance": "balanced",
        "均衡": "balanced",
        "gamble": "gamble",
        "high_score": "gamble",
        "搏高分": "gamble",
    }
    return aliases.get(value, "unknown")


def _decision_phase(source: Mapping[str, Any]) -> str:
    phase = source.get("phase")
    if phase:
        return str(phase)
    if source.get("candidates") or source.get("top"):
        return "BOARD_PLACE"
    action = source.get("action") if isinstance(source.get("action"), Mapping) else {}
    action_type = str(action.get("type") or source.get("action_type") or "")
    if "brush" in action_type:
        return "BRUSH_STAGE2"
    if "backpack" in action_type:
        return "BACKPACK_SELECT"
    return "UNSTABLE"


def _state_signature(source: Mapping[str, Any]) -> str:
    context = source.get("decision_context") if isinstance(source.get("decision_context"), Mapping) else source
    explicit = context.get("state_signature") or context.get("signature")
    if explicit:
        return str(explicit)
    return _stable_id(
        {
            "board": context.get("board"),
            "current": context.get("current", context.get("current_code")),
            "anchor": context.get("anchor"),
            "pending": context.get("pending"),
            "active_affix": context.get("active_affix"),
            "resources": context.get("resources"),
        }
    )


def _candidate_id(row: Any, col: Any, paint: Mapping[str, Any] | None) -> str:
    paint_id = _paint_id(paint) or "unknown"
    return f"placement:{row}:{col}:{paint_id}"


def _action_id(row: Any, col: Any) -> str:
    return f"board.place:{row}:{col}"


def _paint_id(paint: Mapping[str, Any] | None) -> str | None:
    if not isinstance(paint, Mapping):
        return None
    value = paint.get("canonical_paint_id", paint.get("code"))
    return str(value) if value is not None else None


def _initial_bucket(initial: Mapping[str, Any]) -> str:
    board = initial.get("board") if isinstance(initial.get("board"), Mapping) else initial
    blank = int(board.get("blank", 0) or 0)
    paint = int(board.get("paint", 0) or 0)
    blocked = int(board.get("blocked", 0) or 0)
    void = int(board.get("void", 0) or 0)
    states = str(board.get("states") or "")
    center_open = sum(1 for index in (14, 15, 20, 21) if len(states) > index and states[index] in {"B", "P"})
    return "|".join(
        (
            f"open:{_bucket(blank + paint, 16, 22)}",
            f"block:{_bucket(blocked, 4, 8)}",
            f"void:{_bucket(void, 8, 13)}",
            f"center:{_bucket(center_open, 1, 3)}",
        )
    )


def _bucket(value: int, low: int, high: int) -> str:
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def _comparable_group_key(run: Mapping[str, Any]) -> str:
    summary = run.get("ordinary_summary") if isinstance(run.get("ordinary_summary"), Mapping) else {}
    bucket = summary.get("initial_map_bucket")
    if not bucket:
        initial = run.get("initial_snapshot") if isinstance(run.get("initial_snapshot"), Mapping) else {}
        bucket = _initial_bucket(initial)
    versions = run.get("versions") if isinstance(run.get("versions"), Mapping) else {}
    return f"{bucket}|policy:{versions.get('policy', 'unknown')}"


def _quality_is_trusted(quality: Mapping[str, Any]) -> bool:
    if quality.get("ok") is not None:
        return quality.get("ok") is True
    score = _finite_number(quality.get("score", quality.get("overall")))
    return bool(score is not None and score >= 0.58)


def _high_score_hit(run: Mapping[str, Any]) -> bool:
    gate = run.get("high_score_gate") if isinstance(run.get("high_score_gate"), Mapping) else {}
    return gate.get("passed") is True


def _run_failed(run: Mapping[str, Any]) -> bool:
    observed = run.get("observed_outcome") if isinstance(run.get("observed_outcome"), Mapping) else {}
    return observed.get("status") == "failure"


def _low_value_run(run: Mapping[str, Any]) -> bool:
    if str(run.get("kind") or "").startswith("low_value"):
        return True
    gate = run.get("high_score_gate") if isinstance(run.get("high_score_gate"), Mapping) else {}
    return gate.get("passed") is False


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower]) * (1.0 - weight) + float(ordered[upper]) * weight


def _rate(count: int, total: int) -> float | None:
    return _rounded(count / total) if total else None


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _first_finite(*values: Any) -> float | None:
    for value in values:
        number = _finite_number(value)
        if number is not None:
            return number
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _stable_id(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]
