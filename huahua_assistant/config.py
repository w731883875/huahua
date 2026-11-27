from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("runtime/strategy_config.json")

DEFAULT_STRATEGY_CONFIG: dict[str, Any] = {
    "brush_recording": {
        "mode": "validation",
        "max_image_mb": 128,
    },
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
        # 入库口径（用户 2026-11-15 定，两轮确认）：
        #   1. 库里不满 ``max_sessions`` 局 → **全部入库**，先攒样本；
        #   2. 满了 → 至少一枚 7 级颜料的局直接收；
        #   3. 满了且没有 7 级 → 只有内部评分高于库内最低分才写入（挤掉最低那局）。
        # ``require_level7=false`` 表示库满后不再给 7 级局开直通，一律按分数比。
        "require_level7": True,
        "max_sessions": 500,
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
        "good_bucket_bonus": 80.0,
        "bad_bucket_penalty": 130.0,
    },
    "force": {
        "recommend_threshold": 350.0,
        "protected_treasure_penalty": 260.0,
        "natural_completion_penalty": 900.0,
        "no_natural_completion_bonus": 450.0,
        "high_treasure_rescue_bonus": 520.0,
        "good_bucket_bonus": 90.0,
        "bad_bucket_penalty": 160.0,
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
        "action_min_samples": 30,
        "placement_min_samples": 30,
        "brush_min_samples": 20,
        "affix_min_samples": 20,
        "terrain_min_samples": 20,
        "terrain_bucket_min_samples": 8,
        "placement_max_bonus": 180.0,
        "brush_max_bonus": 130.0,
        "affix_max_bonus": 150.0,
        "terrain_prior_max_bonus": 95.0,
        "terrain_prior_max_penalty": 85.0,
    },
    # 桌宠：屏幕上那只小宠物的行为开关。
    "pet": {
        # 桌宠推荐。关掉后桌宠只剩「记录状态」这一件事：不画地块标记、不画指向箭头、
        # 气泡也只报录制进度。专为「打一局录数据」准备——录数据时建议不但没用，
        # 还会盖在棋盘上挡住真正要看的东西。
        # 2026-11-14 用户定调：**默认关**。推荐本身在软件主界面上照样有，桌宠这层
        # 只是叠在游戏画面上的浮层，默认开等于默认挡视线。
        "recommendations": False,
    },
    # 记录模式：一局一局打、只把数据录下来，不需要任何建议。
    #
    # 2026-11-22 用户明确要求：「现在是记录模式，不是推荐模式，任何推荐的都需要
    # 关掉！不要浪费不必要的时间」。开成 True 后主采集路径**不再计算**
    # recommend_placements / recommend_install_plan / recommend_affixes，
    # 这些每帧要花几十到几百毫秒，纯粹白烧。
    # 注意和 ``pet.recommendations`` 的区别：那个只管桌宠浮层画不画标记，
    # 管不到主路径的计算；这个才是总闸门。
    "recording_mode": True,
    "debug": {
        "save_fast_captures": False,
    },
    "capture": {
        "backend": "auto",
        # Windows.Graphics.Capture：直接读窗口自己渲染的表面，别的窗口盖在上面
        # 不会进帧。DirectX 游戏用不了 PrintWindow，这条是唯一能「被遮挡也读对」的路。
        # 失败会自动退回原来的截屏方式，所以默认开着。
        "wgc_enabled": True,
        "highspeed_enabled": False,
        "allow_composited_highspeed": False,
    },
    # 游戏日志：按「画面变了」触发识别，而不是按固定节拍盲截图。
    # 日志里记录着每一次落子/开笔锋/结算/入画，时间戳精确到毫秒。
    "game_log": {
        # 留空 = 自动从游戏进程反查（<安装目录>\UE_game\TorchLight\Saved\Logs\UE_game.log）
        "path": "",
        # 没有事件时的节拍（秒）。棋盘在两笔之间不会动，没必要高频重读。
        "idle_interval": 1.5,
    },
    # 大模型托管：把「看画面 → 给动作」整体交给视觉语言模型。
    # 云端（智谱 / 通义 / DeepSeek / OpenAI）和本地 Ollama 都走 OpenAI 兼容协议，
    # 换 provider 只要改 base_url 和 model。
    "llm": {
        "enabled": False,
        # openai_compat：云端；ollama：本地
        "provider": "openai_compat",
        # 智谱 GLM：https://open.bigmodel.cn/api/paas/v4  （glm-4.6v-flash 输入输出全免费）
        # 通义 Qwen-VL：https://dashscope.aliyuncs.com/compatible-mode/v1
        # 本地 Ollama：http://127.0.0.1:11434/v1
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": "",
        "model": "glm-4.6v-flash",
        "timeout": 30.0,
        # 两次调用至少隔这么久（秒）。大模型单次 2~5 秒，节流防止排队堆积。
        "min_interval": 2.0,
        # 送进模型的图先缩到这个边长，省 token 也省时间。
        "max_image_side": 1280,
        "temperature": 0.1,
        # 输出上限。推理型模型（DeepSeek 等）若思考没关掉会把它全烧在思考链上，
        # 正文一个字都不剩，所以别给太小。
        "max_tokens": 600,
        # 原样并进请求体的额外字段，各家私有开关走这里。
        # 智谱视觉模型默认会「思考」，托管场景只要结论，关掉能快一倍：
        #   "extra_body": {"thinking": {"type": "disabled"}}
        "extra_body": {},
        # 托管模式下是否还并行跑本地链路做降级（强烈建议开）
        "keep_local_fallback": True,
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


def save_strategy_config(updates: dict[str, Any]) -> Path:
    """把改动写回 ``runtime/strategy_config.json`` 并刷新缓存。

    只写 ``updates`` 里出现的键，文件里已有的其他内容原样保留——用户手写的注释性
    字段不该被程序的默认值覆盖掉。
    """
    payload: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    _deep_update(payload, updates)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    reload_strategy_config()
    return CONFIG_PATH


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
