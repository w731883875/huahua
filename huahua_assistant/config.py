from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("runtime/strategy_config.json")

DEFAULT_STRATEGY_CONFIG: dict[str, Any] = {
    "quality": {
        "min_overall": 0.58,
        "min_board": 0.48,
        "min_current": 0.45,
        "min_backpack": 0.35,
        "min_brush": 0.35,
        "max_unknown_tiles": 1,
        "quick_max_unknown_tiles": 1,
        "board_paint_min_category_confidence": 0.46,
        "board_paint_min_level_confidence": 0.36,
    },
    "autosave": {
        "min_internal_score": 3000.0,
        "min_treasure7": 1,
        "min_treasure6": 1,
        "min_high_score_treasure6": 1,
        "min_high_score_lv7": 1,
        "min_lv7": 3,
        "min_gameplay7_mixed": 2,
        "min_monster7_mixed": 1,
        "min_treasure5_with_lv7": 1,
        "min_lv7_with_treasure5": 2,
    },
    "values": {
        "gameplay7": 1000.0,
        "monster7": 714.0,
        "treasure5": 1000.0,
        "treasure6": 3000.0,
        "treasure7": 5000.0,
    },
    "refresh": {
        "stage1_threshold": 120.0,
        "stage2_threshold": 120.0,
        "treasure_missing_stack_bonus": 420.0,
        "treasure_missing_premium_stage2_bonus": 520.0,
        "treasure_has_premium_stage2_penalty": 760.0,
    },
    "force": {
        "recommend_threshold": 350.0,
        "protected_treasure_penalty": 260.0,
        "natural_completion_penalty": 900.0,
        "no_natural_completion_bonus": 450.0,
        "high_treasure_rescue_bonus": 520.0,
    },
    "placement": {
        "terrain_delta_weight": 1.15,
        "bad_terrain_extra_weight": 0.65,
        "overwrite_value_weight": 0.72,
        "overwrite_treasure_base_penalty": 900.0,
        "overwrite_treasure_with_non_treasure_penalty": 950.0,
        "low_treasure_over_high_treasure_penalty": 1400.0,
        "overwrite_high_treasure_penalty": 2200.0,
        "overwrite_level7_penalty": 1000.0,
        "overwrite_level6_penalty": 420.0,
        "treasure_overwrite_discount": 0.65,
    },
    "learning": {
        "placement_min_samples": 30,
        "brush_min_samples": 20,
        "affix_min_samples": 20,
        "placement_max_bonus": 180.0,
        "brush_max_bonus": 130.0,
        "affix_max_bonus": 150.0,
    },
    "debug": {
        "save_fast_captures": False,
    },
    "capture": {
        "backend": "auto",
        "highspeed_enabled": False,
        "allow_composited_highspeed": False,
    },
}

_CACHE: dict[str, Any] | None = None


def get_strategy_config() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_strategy_config()
    return _CACHE


def reload_strategy_config() -> dict[str, Any]:
    global _CACHE
    _CACHE = _load_strategy_config()
    return _CACHE


def config_value(path: str, default: Any | None = None) -> Any:
    current: Any = get_strategy_config()
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _load_strategy_config() -> dict[str, Any]:
    config = deepcopy(DEFAULT_STRATEGY_CONFIG)
    if not CONFIG_PATH.exists():
        return config
    try:
        overrides = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return config
    if isinstance(overrides, dict):
        _deep_update(config, overrides)
    return config


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
