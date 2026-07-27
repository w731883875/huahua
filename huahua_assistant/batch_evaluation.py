from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import __version__
from .config import get_strategy_config
from .game_model import (
    AffixRule,
    BrushChoiceScenario,
    Paint,
    PlacementRecommendation,
    PuzzleState,
    TileState,
    build_affix_catalog,
    force_complete_current_rule,
    install_backpack_paint,
    random_brush_choice_scenario,
    random_state,
    recommend_affixes,
    recommend_brush_types,
    recommend_placements,
)


BATCH_REPORT_SCHEMA = "huahua-batch-evaluation-v1"
STRATEGY_MODES = ("stable", "balanced", "gamble")
HIGH_SCORE_PROXY_THRESHOLD = 1800.0
_SOURCE_FILES = ("batch_evaluation.py", "game_model.py", "config.py")


def build_batch_evaluation_report(
    seeds: Iterable[int],
    *,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic current-rule policy and paired resource proxies.

    This is deliberately not a gameplay-outcome report. Every score is labelled
    ``simulation_proxy`` and is derived from the rule model without writing any
    observation, trajectory, or learning file.
    """

    seed_list = _normalize_seeds(seeds)
    config = _json_copy(get_strategy_config())
    policy_samples: dict[str, list[dict[str, Any]]] = {mode: [] for mode in STRATEGY_MODES}
    force_pairs: list[dict[str, Any]] = []
    refresh1_pairs: list[dict[str, Any]] = []
    refresh2_pairs: list[dict[str, Any]] = []

    for seed in seed_list:
        state = random_state(seed=seed)
        for mode in STRATEGY_MODES:
            policy_samples[mode].append(_evaluate_policy_run(state, seed, mode))

        force_pairs.append(_force_pair(state, seed))
        scenario = random_brush_choice_scenario(seed=_branch_seed(seed, "brush-original"))
        rerolled = random_brush_choice_scenario(seed=_branch_seed(seed, "brush-refresh-stage1"))
        refresh1_pairs.append(_refresh_stage1_pair(state, seed, scenario, rerolled))
        refresh2_pairs.append(_refresh_stage2_pair(state, seed, scenario))

    report = {
        "schema": BATCH_REPORT_SCHEMA,
        "score_type": "simulation_proxy",
        "observed_outcome": None,
        "deterministic": True,
        "seed_set": {
            "count": len(seed_list),
            "values": seed_list,
            "sha256": _stable_hash(seed_list),
        },
        "configuration": {
            "snapshot": config,
            "sha256": _stable_hash(config),
        },
        "code": {
            "package_version": __version__,
            "source_sha256": _source_hash(),
            "revision": code_revision or "unknown",
        },
        "definitions": {
            "evaluation_unit": "one deterministic current-rule episode per initial state",
            "success": "policy completes the current match before it exhausts legal candidates or backpack paint",
            "high_score": f"episode board-value delta plus completion proxy >= {HIGH_SCORE_PROXY_THRESHOLD:.0f}",
            "realized_proxy_formula": "post-action board proxy delta + 900 completion bonus",
            "candidate_lookahead": "disabled inside each step because the episode runner supplies multi-step replay",
            "top_hit_oracle": "highest realized proxy among the complete legal candidate set",
            "strategy_formulas": {
                "stable": "model_score + 0.75*risk + 0.35*min(terrain,0)",
                "balanced": "model_score",
                "gamble": "model_score - 0.45*risk + 0.25*max(value+expected,0)",
            },
            "resource_ab": "paired branches share the same initial state; random branches use stable derived seeds",
        },
        "policies": {mode: _policy_summary(policy_samples[mode]) for mode in STRATEGY_MODES},
        "resource_ab": {
            "force_use_vs_keep": _pair_summary(force_pairs),
            "refresh_stage1_use_vs_keep": _pair_summary(refresh1_pairs),
            "refresh_stage2_use_vs_keep": _pair_summary(refresh2_pairs),
        },
    }
    report["report_sha256"] = _stable_hash(report)
    return report


def _evaluate_policy_run(state: PuzzleState, seed: int, mode: str) -> dict[str, Any]:
    initial_proxy = _board_proxy(state)
    current = state
    decisions: list[dict[str, Any]] = []
    success = False
    max_steps = min(len(state.backpack), 36)

    for step in range(max_steps):
        candidates = recommend_placements(current, limit=4096, include_lookahead=False)
        if not candidates:
            break
        outcomes = {_candidate_key(item): _candidate_outcome_proxy(current, item) for item in candidates}
        oracle = max(candidates, key=lambda item: (outcomes[_candidate_key(item)], _candidate_key(item)))
        ranked = sorted(
            candidates,
            key=lambda item: (_policy_score(item, mode), _candidate_key(item)),
            reverse=True,
        )
        selected = ranked[0]
        top_keys = [_candidate_key(item) for item in ranked[:3]]
        oracle_key = _candidate_key(oracle)
        decisions.append(
            {
                "step": step + 1,
                "candidate_count": len(candidates),
                "selected_key": _candidate_key(selected),
                "policy_score": _round(_policy_score(selected, mode)),
                "realized_proxy": _round(outcomes[_candidate_key(selected)]),
                "top1_hit": top_keys[0] == oracle_key,
                "top3_hit": oracle_key in top_keys,
            }
        )
        backpack_index = next(
            (index for index, paint in enumerate(current.backpack) if paint == selected.paint),
            None,
        )
        if backpack_index is None:
            break
        result = install_backpack_paint(current, backpack_index, selected.row, selected.col)
        if not result.success:
            break
        current = result.state
        if result.completed_match:
            success = True
            break

    episode_proxy = _board_proxy(current) - initial_proxy + (900.0 if success else 0.0)
    return {
        "seed": seed,
        "decision_count": len(decisions),
        "simulation_proxy": _round(episode_proxy),
        "success": success,
        "top1_hits": sum(bool(item["top1_hit"]) for item in decisions),
        "top3_hits": sum(bool(item["top3_hit"]) for item in decisions),
        "decisions": decisions,
    }


def _normalize_seeds(seeds: Iterable[int]) -> list[int]:
    values = list(seeds)
    if not values:
        raise ValueError("at least one seed is required")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("seeds must be integers")
    if len(set(values)) != len(values):
        raise ValueError("seeds must be unique")
    return values


def _policy_score(item: PlacementRecommendation | None, mode: str) -> float:
    if item is None:
        return 0.0
    explanation = item.explanation
    if explanation is None or mode == "balanced":
        return float(item.score)
    if mode == "stable":
        return float(item.score) + 0.75 * explanation.risk + 0.35 * min(explanation.terrain, 0.0)
    if mode == "gamble":
        return (
            float(item.score)
            - 0.45 * explanation.risk
            + 0.25 * max(explanation.value + explanation.expected, 0.0)
        )
    raise ValueError(f"unknown strategy mode: {mode}")


def _candidate_outcome_proxy(state: PuzzleState, item: PlacementRecommendation) -> float:
    backpack_index = next(
        (index for index, paint in enumerate(state.backpack) if paint == item.paint),
        None,
    )
    if backpack_index is None:
        return -math.inf
    result = install_backpack_paint(state, backpack_index, item.row, item.col)
    if not result.success:
        return -math.inf
    return _board_proxy(result.state) - _board_proxy(state) + (900.0 if result.completed_match else 0.0)


def _candidate_key(item: PlacementRecommendation | None) -> str:
    if item is None:
        return ""
    paint = item.paint
    paint_key = "none" if paint is None else f"{paint.category.value}:{paint.level}:{int(paint.is_treasure)}"
    return f"{item.row},{item.col}:{paint_key}"


def _policy_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(item["simulation_proxy"]) for item in samples if item["simulation_proxy"] is not None]
    total = len(samples)
    decisions = sum(int(item["decision_count"]) for item in samples)
    return {
        "total_runs": total,
        "scored_runs": len(scores),
        "success_rate": _rate(sum(bool(item["success"]) for item in samples), total),
        "high_score_rate": _rate(sum(score >= HIGH_SCORE_PROXY_THRESHOLD for score in scores), len(scores)),
        "decision_count": decisions,
        "top1_hit_rate": _rate(sum(int(item["top1_hits"]) for item in samples), decisions),
        "top3_hit_rate": _rate(sum(int(item["top3_hits"]) for item in samples), decisions),
        "distribution": _distribution(scores),
        "runs": samples,
    }


def _force_pair(state: PuzzleState, seed: int) -> dict[str, Any]:
    keep = _board_proxy(state)
    result = force_complete_current_rule(state, force_used=0, seed=_branch_seed(seed, "force-use"))
    use = _board_proxy(result.state) if result.success else keep
    if result.success and state.current_rule is not None:
        use += 900.0 + _paint_value(state.current_rule.paint) * 0.20
    return _pair(seed, keep, use, {"destroyed_cells": len(result.destroyed_cells), "use_success": result.success})


def _refresh_stage1_pair(
    state: PuzzleState,
    seed: int,
    original: BrushChoiceScenario,
    rerolled: BrushChoiceScenario,
) -> dict[str, Any]:
    keep_recs = recommend_brush_types(state, original.stage1_options)
    use_recs = recommend_brush_types(state, rerolled.stage1_options)
    keep = keep_recs[0].score if keep_recs else 0.0
    use = use_recs[0].score if use_recs else 0.0
    return _pair(
        seed,
        keep,
        use,
        {
            "keep_options": [item.value for item in original.stage1_options],
            "use_options": [item.value for item in rerolled.stage1_options],
        },
    )


def _refresh_stage2_pair(state: PuzzleState, seed: int, original: BrushChoiceScenario) -> dict[str, Any]:
    keep_recs = recommend_affixes(state, original.stage2_options)
    rerolled_options = _sample_affixes(
        original.chosen_stage1.value,
        _branch_seed(seed, "brush-refresh-stage2"),
        len(original.stage2_options),
    )
    use_recs = recommend_affixes(state, rerolled_options)
    keep = keep_recs[0].score if keep_recs else 0.0
    use = use_recs[0].score if use_recs else 0.0
    return _pair(
        seed,
        keep,
        use,
        {
            "brush_type": original.chosen_stage1.value,
            "keep_options": [_affix_id(item) for item in original.stage2_options],
            "use_options": [_affix_id(item) for item in rerolled_options],
        },
    )


def _sample_affixes(brush_type: str, seed: int, count: int) -> tuple[AffixRule, ...]:
    catalog = [item for item in build_affix_catalog() if item.brush_type.value == brush_type]
    return tuple(random.Random(seed).sample(catalog, min(count, len(catalog))))


def _affix_id(item: AffixRule) -> str:
    return item.canonical_affix_id or (item.legacy_codes[0] if item.legacy_codes else item.name)


def _pair(seed: int, keep: float, use: float, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "keep_proxy": _round(keep),
        "use_proxy": _round(use),
        "delta": _round(use - keep),
        **details,
    }


def _pair_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(item["delta"]) for item in pairs]
    return {
        "paired_runs": len(pairs),
        "same_seed_set": True,
        "use_win_rate": _rate(sum(delta > 0 for delta in deltas), len(deltas)),
        "keep_win_rate": _rate(sum(delta < 0 for delta in deltas), len(deltas)),
        "tie_rate": _rate(sum(delta == 0 for delta in deltas), len(deltas)),
        "delta_distribution": _distribution(deltas),
        "pairs": pairs,
    }


def _board_proxy(state: PuzzleState) -> float:
    total = 0.0
    for tile in state.tiles:
        if tile.paint is not None:
            total += _paint_value(tile.paint)
        elif tile.state == TileState.BLANK:
            total += 18.0
        elif tile.state == TileState.BLOCKED:
            total -= 22.0
        elif tile.state == TileState.VOID:
            total -= 10.0
        elif tile.state == TileState.FADED:
            total -= 35.0
    return total


def _paint_value(paint: Paint) -> float:
    level = max(1, min(7, paint.level))
    if paint.is_treasure:
        return {1: 260.0, 2: 420.0, 3: 620.0, 4: 820.0, 5: 1000.0, 6: 3000.0, 7: 5000.0}[level]
    if paint.category.value == "gameplay":
        return {1: 25.0, 2: 60.0, 3: 120.0, 4: 220.0, 5: 380.0, 6: 560.0, 7: 1000.0}[level]
    if paint.category.value == "monster":
        return {1: 18.0, 2: 44.0, 3: 88.0, 4: 160.0, 5: 280.0, 6: 410.0, 7: 1000.0 / 1.4}[level]
    return level * 45.0


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {
            "count": 0, "mean": None, "median": None, "stddev": None,
            "p10": None, "p20": None, "p80": None, "p90": None, "bottom20_mean": None,
        }
    bottom_count = max(1, math.ceil(len(ordered) * 0.20))
    return {
        "count": len(ordered),
        "mean": _round(statistics.fmean(ordered)),
        "median": _round(statistics.median(ordered)),
        "stddev": _round(statistics.pstdev(ordered)),
        "p10": _round(_percentile(ordered, 0.10)),
        "p20": _round(_percentile(ordered, 0.20)),
        "p80": _round(_percentile(ordered, 0.80)),
        "p90": _round(_percentile(ordered, 0.90)),
        "bottom20_mean": _round(statistics.fmean(ordered[:bottom_count])),
    }


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower]) * (1.0 - weight) + float(ordered[upper]) * weight


def _branch_seed(seed: int, branch: str) -> int:
    digest = hashlib.sha256(f"{seed}:{branch}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _source_hash() -> str:
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _SOURCE_FILES:
        path = directory / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_copy(value: Any) -> Any:
    return json.loads(_stable_json(value))


def _rate(count: int, total: int) -> float | None:
    return _round(count / total) if total else None


def _round(value: float) -> float:
    return round(float(value), 6)
