from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .codes import affix_code, brush_code, canonical_affix_id
from .config import config_value
from .outcome_learning import (
    build_outcome_risk_report,
    normalize_observation_payload,
    normalize_trajectory_payload,
)


RUNTIME_DIR = Path("runtime")
OBSERVATION_LOG = RUNTIME_DIR / "observations.jsonl"
CORRECTION_LOG = RUNTIME_DIR / "corrected_observations.jsonl"
TRAJECTORY_LOG = RUNTIME_DIR / "high_score_trajectories.jsonl"
STATS_PATH = RUNTIME_DIR / "stats.json"
_RUN_CONTEXT_EVENTS = {
    "run_session_finished",
    "compressed_run_summary",
    "run_session_failed",
    "run_session_finish_skipped",
}
#: 诊断类事件：不参与学习，只记录「刚才发生了什么」的现场。
#: 它们必须允许来自任何来源——游戏日志、自动跟随状态机——否则出了事
#: 只能靠猜。学习样本仍然只收 ``real_capture``。
_DIAGNOSTIC_EVENTS = {
    "game_log_attached",
    "game_log_event",
    "auto_follow_pause",
    "auto_follow_restart",
    # 工具窗口什么时候开始「对截屏不可见」。用户报「软件窗口挡住棋盘」时，
    # 第一件事就是确认这条有没有出现、以及那之后的帧是不是就不脏了。
    "capture_exclusion",
    # 识别侧慢路径。用户报「开局识别卡十秒」时，日志里原来没有任何路径信息：
    # 分不清是标定没命中、全搜索太慢，还是 OCR 卡住。这两条把耗时和结果写下来。
    "vision_diagnostic",
    "slow_frame",
}

#: 最近一次落盘各步骤的耗时（秒）。收尾卡顿时用它定位到底哪一步吃掉了时间。
#:
#: 为什么放在模块级而不是直接写一条观测：``record_observation`` 自己就是被埋点
#: 的对象，从里面再写一条观测会递归。所以这里只记内存，由 ``app.py`` 在写
#: ``run_session_finished`` 时把快照塞进 payload。
LAST_PERSIST_TIMING: dict[str, float] = {}


def record_observation(event_type: str, payload: dict[str, Any], *, source: str) -> None:
    if source != "real_capture" and event_type not in _DIAGNOSTIC_EVENTS:
        return
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # 浅拷贝一份：会话上下文只读，但缓存里的那个列表归缓存所有。
    session_events = list(_load_events()) if event_type in _RUN_CONTEXT_EVENTS else ()
    loaded = time.perf_counter()
    normalized_payload = normalize_observation_payload(
        event_type,
        _jsonable(payload),
        session_events=session_events,
    )
    normalized = time.perf_counter()
    event = {
        "ts": time.time(),
        "event": event_type,
        "source": source,
        "payload": normalized_payload,
    }
    _append_jsonl_cached(OBSERVATION_LOG, event)
    written = time.perf_counter()
    LAST_PERSIST_TIMING.update(
        {
            "observation_load": round(loaded - started, 3),
            "observation_normalize": round(normalized - loaded, 3),
            "observation_write": round(written - normalized, 3),
        }
    )


def record_correction(payload: dict[str, Any], *, source: str) -> None:
    if source != "real_capture":
        return
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    event = {
        "ts": time.time(),
        "event": "corrected_observation",
        "source": source,
        "payload": _jsonable(payload),
    }
    _append_jsonl_cached(CORRECTION_LOG, event)
    appended = time.perf_counter()
    _write_stats(_load_events(), _load_corrections(), _load_trajectories())
    finished = time.perf_counter()
    LAST_PERSIST_TIMING.update(
        {
            "correction_append": round(appended - started, 3),
            "correction_stats": round(finished - appended, 3),
            "correction_total": round(finished - started, 3),
        }
    )


def record_high_score_trajectory(payload: dict[str, Any], *, source: str) -> bool:
    if source != "real_capture":
        return False
    if not _trajectory_payload_is_trusted(payload):
        return False
    started = time.perf_counter()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    legacy_payload_hash = _payload_hash(payload)
    normalized_payload = normalize_trajectory_payload(_jsonable(payload))
    payload_hash = _payload_hash(normalized_payload)
    existing = _load_trajectories()
    existing_hashes: set[str] = set()
    for entry in existing:
        existing_payload = entry.get("payload") or {}
        # 新记录里 ``payload_hash`` **就是**规范化后载荷的哈希（见下面写入处），
        # 直接拿来用即可；只有老记录缺这个字段时才重算。
        # 重算一条要遍历最多 240 步做 normalize + sha256，32 局实测 4.75 秒——
        # 这是收尾卡顿的第二大头，不能再无条件跑。
        stored_hash = str(entry.get("payload_hash") or "")
        if stored_hash:
            existing_hashes.add(stored_hash)
        else:
            existing_hashes.add(_payload_hash(normalize_trajectory_payload(existing_payload)))
        existing_hashes.add(str(entry.get("legacy_payload_hash") or _payload_hash(existing_payload)))
    hashed = time.perf_counter()
    if payload_hash in existing_hashes or legacy_payload_hash in existing_hashes:
        LAST_PERSIST_TIMING.update(
            {
                "trajectory_hash": round(hashed - started, 3),
                "trajectory_total": round(hashed - started, 3),
            }
        )
        return False
    event = {
        "ts": time.time(),
        "event": "high_score_trajectory",
        "source": source,
        "payload": normalized_payload,
        "payload_hash": payload_hash,
        "legacy_payload_hash": legacy_payload_hash,
    }
    _append_trajectory_event(event)
    written = time.perf_counter()
    # 直接读缓存那一份（刚追加的那条已经在里面了，不能再手动 + [event]，
    # 否则同一条会被算两次）。缓存若因签名不符被丢弃，这里会重读一次，结果仍正确。
    _prune_trajectories(events=_load_trajectories())
    pruned = time.perf_counter()
    _write_stats(_load_events(), _load_corrections(), _load_trajectories())
    stats_done = time.perf_counter()
    LAST_PERSIST_TIMING.update(
        {
            "trajectory_hash": round(hashed - started, 3),
            "trajectory_write": round(written - hashed, 3),
            "trajectory_prune": round(pruned - written, 3),
            "trajectory_stats": round(stats_done - pruned, 3),
            "trajectory_total": round(stats_done - started, 3),
        }
    )
    return True


#: 高分轨迹最多保留多少局。超出后按**内部评分降序**只留前 N 局。
#: 500 局是用户定下的上限（2026-11-15）：满铺会自动存档，一晚上就能攒几十局，
#: 不设上限文件会无界增长（当时 14 局 3.5 MB，只因为每局都带 240 步细节）。
MAX_TRAJECTORY_SESSIONS = 500


def trajectory_count() -> int:
    """高分库里现在有几局。

    只**数行**、不解析 JSON：500 局的文件里每局都带最多 240 步细节（实测 14 局
    就有 3.5 MB），满库时逐行 ``json.loads`` 要几秒，而调用方（自动存档的门槛
    判断）只需要一个数字。
    """
    if not TRAJECTORY_LOG.exists():
        return 0
    count = 0
    try:
        with TRAJECTORY_LOG.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except OSError:
        return 0
    return count


def lowest_trajectory_score() -> float | None:
    """库内**最低**的内部评分；空库返回 ``None``。

    满库后「能否替换掉一局」就看它——只有分数更高的局才值得写盘。
    """
    scores = [_trajectory_sort_key(event)[0] for event in _load_trajectories()]
    return min(scores) if scores else None


def _trajectory_sort_key(event: dict[str, Any]) -> tuple[float, float]:
    payload = event.get("payload") or {}
    try:
        score = float(payload.get("internal_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        ts = float(event.get("ts") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    # 同分时新的排前面：同样的分数更近的一局信息更新。
    return score, ts


def _prune_trajectories(
    limit: int = MAX_TRAJECTORY_SESSIONS,
    *,
    events: list[dict[str, Any]] | None = None,
) -> int:
    """按内部评分降序只保留前 ``limit`` 局，返回丢掉的条数。

    只在**超限**时才重写文件：正常一局一次 append，不额外付出 IO。
    先写 ``.tmp`` 再 ``replace``，避免写到一半崩溃把整个高分库搞坏。

    ``events`` 允许调用方把已经读出来的那份传进来，省掉一次整文件解析
    （13.9 MB 实测 0.7 秒，而它在收尾路径上会被调用两次）。
    """
    if limit <= 0:
        return 0
    if events is None:
        events = _load_trajectories()
    if len(events) <= limit:
        return 0
    keep = sorted(events, key=_trajectory_sort_key, reverse=True)[:limit]
    temp = TRAJECTORY_LOG.with_name(TRAJECTORY_LOG.name + ".tmp")
    try:
        temp.write_text(
            "".join(
                json.dumps(_compact_trajectory_event(event), ensure_ascii=False, sort_keys=True) + "\n"
                for event in keep
            ),
            encoding="utf-8",
        )
        temp.replace(TRAJECTORY_LOG)
    except OSError:
        # 裁剪失败不该影响「这一局已经存下来了」这个事实。
        if temp.exists():
            temp.unlink(missing_ok=True)
        return 0
    # 整份重写后缓存必然过期。签名检查也能发现，但那要多一次 stat 才发现，
    # 而且留着一份随时会膨胀的旧列表没意义。
    _invalidate_jsonl_cache(TRAJECTORY_LOG)
    return len(events) - limit


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


_LEARNING_STATS_CACHE_SIGNATURE: tuple[tuple[bool, int, int], ...] | None = None
_LEARNING_STATS_CACHE: dict[str, Any] | None = None


def _logs_signature() -> tuple[tuple[bool, int, int], ...]:
    """三个学习日志的 ``(存在, 字节数, mtime_ns)`` 组合签名。"""
    return tuple(
        _file_signature(path) for path in (OBSERVATION_LOG, CORRECTION_LOG, TRAJECTORY_LOG)
    )


def load_learning_stats() -> dict[str, Any]:
    # Rebuild only when a source log changes. Re-parsing all JSONL files for
    # every realtime frame consumed most of the 500 ms response budget.
    global _LEARNING_STATS_CACHE_SIGNATURE, _LEARNING_STATS_CACHE
    frozen_signature = _logs_signature()
    if _LEARNING_STATS_CACHE is not None and frozen_signature == _LEARNING_STATS_CACHE_SIGNATURE:
        return _LEARNING_STATS_CACHE
    stats = _build_complete_stats(_load_events(), _load_corrections(), _load_trajectories())
    _LEARNING_STATS_CACHE_SIGNATURE = frozen_signature
    _LEARNING_STATS_CACHE = stats
    return stats


def cached_learning_stats() -> dict[str, Any] | None:
    """**只取现成的**统计，绝不现算；没有就是 ``None``。

    界面刷新用这个：后台线程算完会顺手把结果放进缓存，界面拿着用即可，不必在主线程
    上再等 1.8 秒。
    """
    return _LEARNING_STATS_CACHE


def _load_events() -> list[dict[str, Any]]:
    return _load_jsonl_cached(OBSERVATION_LOG)


def _load_corrections() -> list[dict[str, Any]]:
    return _load_jsonl_cached(CORRECTION_LOG)


def _load_trajectories() -> list[dict[str, Any]]:
    """读高分轨迹库，**返回展开后的**事件（磁盘上可能存的是压缩形式）。

    所有调用方（统计、推荐、模仿分析、最低分门槛）看到的都是和压缩前一模一样的
    字段，压缩对它们完全透明。展开结果按文件签名单独缓存，所以只有文件变了才会
    重新展开。
    """
    global _TRAJECTORY_EXPANDED
    signature = _file_signature(TRAJECTORY_LOG)
    cached = _TRAJECTORY_EXPANDED
    if cached is not None and cached[0] == signature:
        return cached[1]
    expanded = [_expand_trajectory_event(event) for event in _load_jsonl_cached(TRAJECTORY_LOG)]
    _TRAJECTORY_EXPANDED = (signature, expanded)
    return expanded


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


#: 轨迹里**同一局内高度重复**的步骤字段：一批一模一样的快照会被原样存几十遍。
#:
#: 实测某局 63 步里 ``backpack_cumulative`` 只有 37 种、``backpack_pages`` 只有
#: 31 种、``tiles`` 只有 43 种，而这三项（加上 ``backpack_visible``）占了整局
#: 体积的 49%。压缩方式是**引用去重**：同样的内容只存一份放进 ``snapshots``
#: 表，步骤里留一个整数下标。内容一个字节不少，只是不再重复存。
#:
#: 为什么不直接把字段删掉：这些字段是复盘/训练的证据（背包当时长什么样、
#: 推荐候选是什么、这一手改了什么）。去重是无损的，删字段是有损的。
_DEDUPE_STEP_FIELDS = (
    "tiles",
    "line_capacity",
    "observed_tiles",
    "recording_evidence",
    "candidates",
    "top",
    "backpack_cumulative",
    "backpack_pages",
    "backpack_visible",
    "decision_context",
    "inferred_action",
    "outcome",
)

#: 压缩结构版本。读取时只认这个标记；老轨迹没有它，按原样读。
_TRAJECTORY_COMPACTION_VERSION = 2

#: 背包格里那个 96 字符的图标指纹串：整库出现 11789 次只有 1935 种，
#: 单这一项就占 10.4% 体积。做一张字符串池，格子里的值换成池下标。
_POOLED_FIELD = "fingerprint"


def _pool_fingerprints(node: Any, pool: list[str], index: dict[str, int]) -> None:
    """递归把 ``fingerprint`` 字符串换成字符串池下标（原地改 ``node``）。"""
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key == _POOLED_FIELD and isinstance(value, str):
                slot = index.get(value)
                if slot is None:
                    slot = len(pool)
                    pool.append(value)
                    index[value] = slot
                node[key] = slot
            else:
                _pool_fingerprints(value, pool, index)
    elif isinstance(node, list):
        for item in node:
            _pool_fingerprints(item, pool, index)


def _unpool_fingerprints(node: Any, pool: list[str]) -> None:
    """``_pool_fingerprints`` 的逆操作。下标越界就原样留着，绝不抛。"""
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key == _POOLED_FIELD and isinstance(value, int) and not isinstance(value, bool):
                if 0 <= value < len(pool):
                    node[key] = pool[value]
            else:
                _unpool_fingerprints(value, pool)
    elif isinstance(node, list):
        for item in node:
            _unpool_fingerprints(item, pool)


def compact_trajectory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """把一局轨迹里重复的步骤字段抽成 ``snapshots`` 表，步骤里只留下标。

    完全无损：``expand_trajectory_payload`` 能还原出逐字节等价的结构。
    序列化不了的值（理论上不该有）原样留在步骤里，绝不丢。
    """
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        return payload
    tables: dict[str, list[Any]] = {}
    indexes: dict[str, dict[str, int]] = {}
    compacted_steps: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            compacted_steps.append(step)
            continue
        new_step = dict(step)
        for field in _DEDUPE_STEP_FIELDS:
            if field not in new_step:
                continue
            value = new_step.pop(field)
            try:
                key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                new_step[field] = value
                continue
            table = tables.setdefault(field, [])
            index = indexes.setdefault(field, {})
            slot = index.get(key)
            if slot is None:
                slot = len(table)
                table.append(value)
                index[key] = slot
            # ``#字段名`` = 该字段在这张表里的下标。
            new_step["#" + field] = slot
        compacted_steps.append(new_step)
    if not tables:
        return payload
    # 第二层：图标指纹串池化。必须放在字段去重**之后**——先去重再池化，
    # 池里不会留下「只被丢弃的重复快照」引用到的废串。
    strings: list[str] = []
    _pool_fingerprints(tables, strings, {})
    compacted = dict(payload)
    compacted["steps"] = compacted_steps
    compacted["snapshots"] = tables
    if strings:
        compacted["strings"] = strings
    compacted["compaction"] = {"version": _TRAJECTORY_COMPACTION_VERSION}
    return compacted


def expand_trajectory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """把 ``snapshots`` 下标还原成完整字段，结果与压缩前逐字段等价。

    对没有 ``compaction`` 标记的老轨迹原样返回，所以新旧文件混在一个库里没问题。
    """
    meta = payload.get("compaction")
    tables = payload.get("snapshots")
    if not isinstance(meta, dict) or not isinstance(tables, dict):
        return payload
    strings = payload.get("strings")
    if isinstance(strings, list) and strings:
        _unpool_fingerprints(tables, [str(item) for item in strings])
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return payload
    expanded_steps: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            expanded_steps.append(step)
            continue
        new_step = dict(step)
        for key in [k for k in new_step if isinstance(k, str) and k.startswith("#")]:
            field = key[1:]
            slot = new_step.pop(key)
            table = tables.get(field) or []
            if isinstance(slot, int) and 0 <= slot < len(table):
                new_step[field] = table[slot]
        expanded_steps.append(new_step)
    expanded = dict(payload)
    expanded["steps"] = expanded_steps
    expanded.pop("snapshots", None)
    expanded.pop("strings", None)
    expanded.pop("compaction", None)
    return expanded


def _compact_trajectory_event(event: dict[str, Any]) -> dict[str, Any]:
    """事件外壳不变，只压 ``payload``；已经压过的原样返回（幂等）。"""
    if not isinstance(event, dict):
        return event
    payload = event.get("payload")
    if not isinstance(payload, dict) or "compaction" in payload:
        return event
    compacted = dict(event)
    compacted["payload"] = compact_trajectory_payload(payload)
    return compacted


def _expand_trajectory_event(event: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(event, dict):
        return event
    payload = event.get("payload")
    if not isinstance(payload, dict) or "compaction" not in payload:
        return event
    expanded = dict(event)
    expanded["payload"] = expand_trajectory_payload(payload)
    return expanded


#: JSONL 读取缓存：``路径 -> (文件签名, 事件列表)``。
#:
#: 为什么必须有：``observations.jsonl`` 实测 35.7 MB / 8000+ 条，``_load_jsonl``
#: 解析一次 1.9~2.4 秒；``high_score_trajectories.jsonl`` 13.9 MB 解析一次 0.7 秒。
#: 而一次收尾要读好几遍（``record_observation`` 取会话上下文 1 遍、
#: ``record_high_score_trajectory`` 去重/裁剪/回写统计各 1 遍）——
#: 实测合计 11.5 秒，全压在 Tk 主线程上，玩家看到的就是「点完结束本局，界面没反应」。
#:
#: 缓存命中只花一次 ``stat()``；写盘走 :func:`_append_jsonl_cached` 原地追加，
#: 所以下一帧不必重读整个文件。
_JSONL_CACHE: dict[Path, tuple[tuple[bool, int, int], list[dict[str, Any]]]] = {}


def _file_signature(path: Path) -> tuple[bool, int, int]:
    """``(存在, 字节数, mtime_ns)``——够用且比内容比对便宜得多。"""
    try:
        stat = path.stat()
        return (True, int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        return (False, 0, 0)


def _load_jsonl_cached(path: Path) -> list[dict[str, Any]]:
    """带签名的读取。**返回的列表归缓存所有，调用方不得原地修改。**

    需要改内容的调用方请自己 ``list(...)`` 一份（浅拷贝只是复制指针，不解析 JSON）。
    """
    signature = _file_signature(path)
    cached = _JSONL_CACHE.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    events = _load_jsonl(path)
    _JSONL_CACHE[path] = (signature, events)
    return events


def _append_jsonl_cached(path: Path, event: dict[str, Any]) -> None:
    """追加一条 JSONL，并把缓存同步到最新。

    只在「缓存与磁盘一致」时才增量追加；万一别人改过文件（签名对不上），
    就把缓存丢掉，下次老老实实重读——宁可慢一次，也不能给出错的数据。
    """
    before = _file_signature(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    cached = _JSONL_CACHE.get(path)
    if cached is None:
        return
    signature, events = cached
    if signature != before:
        _JSONL_CACHE.pop(path, None)
        return
    events.append(event)
    _JSONL_CACHE[path] = (_file_signature(path), events)


def _invalidate_jsonl_cache(path: Path) -> None:
    """让某个日志的读取缓存作废。

    轨迹库多一层「展开缓存」，重写后必须一起丢，否则会读到展开前的旧快照。
    """
    global _TRAJECTORY_EXPANDED
    _JSONL_CACHE.pop(path, None)
    if path == TRAJECTORY_LOG:
        _TRAJECTORY_EXPANDED = None


#: ``(文件签名, 展开后的事件列表)``。磁盘上存的是压缩形式，展开一次就够，
#: 之后所有读取复用这一份（32 局展开实测几十毫秒，比重新解析 JSON 便宜得多）。
_TRAJECTORY_EXPANDED: tuple[tuple[bool, int, int], list[dict[str, Any]]] | None = None


def _append_trajectory_event(event: dict[str, Any]) -> None:
    """追加一局轨迹：**磁盘写压缩形式，内存缓存存展开形式**。

    两个缓存必须各存各的：磁盘那份要小，内存那份要给调用方看完整字段。
    """
    global _TRAJECTORY_EXPANDED
    disk_event = _compact_trajectory_event(event)
    before = _file_signature(TRAJECTORY_LOG)
    with TRAJECTORY_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(disk_event, ensure_ascii=False, sort_keys=True) + "\n")
    signature = _file_signature(TRAJECTORY_LOG)
    raw = _JSONL_CACHE.get(TRAJECTORY_LOG)
    if raw is not None and raw[0] == before:
        raw[1].append(disk_event)
        _JSONL_CACHE[TRAJECTORY_LOG] = (signature, raw[1])
    else:
        _JSONL_CACHE.pop(TRAJECTORY_LOG, None)
    expanded = _TRAJECTORY_EXPANDED
    if expanded is not None and expanded[0] == before:
        expanded[1].append(event)
        _TRAJECTORY_EXPANDED = (signature, expanded[1])
    else:
        _TRAJECTORY_EXPANDED = None


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
    # 这里绕过了 :func:`_append_jsonl_cached`（一次写多条 + 去重），
    # 所以事后统一让缓存作废，别让后续读到旧快照。
    _invalidate_jsonl_cache(path)
    return added


#: 统计摘要的后台写盘：``_write_stats`` 只登记「最新一份快照」，由这个线程串行落盘。
#:
#: 统计是派生摘要，继续留在后台计算并发布内存缓存，界面不等它。
#: 仅移到线程不能隔离 CPU 开销：旧风险报告深复制整段历史曾把同时运行的识别
#: 拖到 6 秒。现在风险报告先去重、只生成统计摘要，避免重建完整轨迹决策。
_STATS_PENDING: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None = None
_STATS_LOCK = threading.Lock()
_STATS_WORKER: threading.Thread | None = None


def _write_stats(
    events: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    trajectories: list[dict[str, Any]] | None = None,
) -> None:
    """登记一份统计快照，由后台线程算完写 ``stats.json``（同一时刻只有一个线程）。

    传进来的三个列表会被浅拷贝：调用方拿着的是 JSONL 缓存里的列表，缓存随时可能被
    原地追加，后台线程不能在它变动的时候遍历。
    """
    global _STATS_PENDING, _STATS_WORKER
    snapshot = (list(events), list(corrections), list(trajectories or []))
    with _STATS_LOCK:
        _STATS_PENDING = snapshot
        if _STATS_WORKER is not None and _STATS_WORKER.is_alive():
            # 正在算的那一份已经过期了，让它算完自己再取一次最新的。
            return
        _STATS_WORKER = threading.Thread(
            target=_stats_worker_loop,
            name="HuahuaStatsWriter",
            daemon=True,
        )
        _STATS_WORKER.start()


def _stats_worker_loop() -> None:
    """``latest wins``：有更新就再算一遍，没有就退出。"""
    global _STATS_PENDING, _STATS_WORKER
    while True:
        with _STATS_LOCK:
            payload = _STATS_PENDING
            _STATS_PENDING = None
        if payload is None:
            with _STATS_LOCK:
                # 必须在锁内宣布退出，否则下一次 _write_stats 会以为还有人在跑而干等。
                if _STATS_PENDING is None:
                    _STATS_WORKER = None
                    return
            continue
        try:
            stats = _build_complete_stats(payload[0], payload[1], payload[2])
            # 先写 .tmp 再 replace：后台写也不能让外部读到半截 JSON。
            temp = STATS_PATH.with_name(STATS_PATH.name + ".tmp")
            temp.write_text(
                json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temp.replace(STATS_PATH)
            _publish_stats_cache(stats)
        except Exception:  # noqa: BLE001 - 统计写不出来不该影响已存下的这一局
            pass


def flush_pending_stats(timeout: float = 60.0) -> bool:
    """等后台统计写盘跑完，返回「是否已无待写任务」。

    给测试和「关闭前收尾」用：``_write_stats`` 变成异步之后，想立刻读到
    ``stats.json`` 就必须先等它一下。
    """
    worker = _STATS_WORKER
    if worker is None:
        return True
    worker.join(timeout)
    return not worker.is_alive()


def _publish_stats_cache(stats: dict[str, Any]) -> None:
    """把后台算好的统计塞回内存缓存，界面下次取用就不用再算 1.8 秒。

    签名取「此刻三个日志的签名」：万一算的这段时间里又追加了新记录，签名会对不上，
    下次 :func:`load_learning_stats` 自然会重算——宁可多算一次，也不能显示旧数据。
    """
    global _LEARNING_STATS_CACHE, _LEARNING_STATS_CACHE_SIGNATURE
    with _STATS_LOCK:
        _LEARNING_STATS_CACHE = stats
        _LEARNING_STATS_CACHE_SIGNATURE = _logs_signature()


def _build_complete_stats(
    events: list[dict[str, Any]],
    corrections: list[dict[str, Any]] | None = None,
    trajectories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stats = _build_stats(events, corrections, trajectories)
    stats["outcome_risk"] = build_outcome_risk_report(
        events,
        trajectories or [],
        min_samples=int(config_value("learning.outcome_min_samples", 20) or 20),
    )
    return stats


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
    affix_id_seen = Counter()
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
    trajectory_affix_ids = Counter()
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
    imitation_affix_choice_ids = Counter()
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
    compressed_run_summaries = 0
    compressed_summary_kinds = Counter()
    compressed_summary_skip_reasons = Counter()
    compressed_summary_scores: list[float] = []
    compressed_summary_weight_total = 0.0
    compressed_summary_weighted_score = 0.0
    compressed_initial_terrain = Counter()
    compressed_initial_open_cells = Counter()
    compressed_initial_signatures = Counter()
    compressed_terrain_buckets = Counter()
    compressed_terrain_bucket_scores: dict[str, list[float]] = {}
    compressed_terminal_counts = Counter()
    trajectory_initial_snapshots = 0
    trajectory_initial_terrain = Counter()
    trajectory_initial_open_cells = Counter()
    trajectory_terrain_buckets = Counter()
    trajectory_terrain_bucket_scores: dict[str, list[float]] = {}

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
            for identity in _normalized_affix_ids(
                names=payload.get("stage2_options"),
                codes=payload.get("stage2_codes"),
                identities=payload.get("stage2_ids"),
            ):
                affix_id_seen[identity] += 1
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
        elif event_type == "compressed_run_summary":
            compressed_run_summaries += 1
            kind = str(payload.get("kind") or "unknown")
            compressed_summary_kinds[kind] += 1
            skip_reason = payload.get("skip_reason") or payload.get("reason")
            if skip_reason:
                compressed_summary_skip_reasons[str(skip_reason)] += 1
            score = payload.get("internal_score")
            if isinstance(score, (int, float)):
                score_value = float(score)
                compressed_summary_scores.append(score_value)
                weight = payload.get("weight")
                if isinstance(weight, (int, float)) and weight > 0:
                    compressed_summary_weight_total += float(weight)
                    compressed_summary_weighted_score += score_value * float(weight)
            initial_snapshot = payload.get("initial_snapshot") if isinstance(payload.get("initial_snapshot"), dict) else {}
            _accumulate_initial_terrain(
                initial_snapshot,
                terrain=compressed_initial_terrain,
                open_cells=compressed_initial_open_cells,
                signatures=compressed_initial_signatures,
            )
            bucket = _terrain_bucket_key(initial_snapshot)
            compressed_terrain_buckets[bucket] += 1
            if isinstance(score, (int, float)):
                compressed_terrain_bucket_scores.setdefault(bucket, []).append(float(score))
            terminal = payload.get("terminal") if isinstance(payload.get("terminal"), dict) else {}
            known = terminal.get("known") if isinstance(terminal.get("known"), dict) else {}
            for key in ("total", "lv7", "gameplay7", "monster7", "treasure5", "treasure6", "treasure7", "max_treasure"):
                compressed_terminal_counts[key] += _safe_int(known.get(key))

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
        initial_snapshot = payload.get("initial_snapshot") if isinstance(payload.get("initial_snapshot"), dict) else {}
        if initial_snapshot:
            trajectory_initial_snapshots += 1
            _accumulate_initial_terrain(
                initial_snapshot,
                terrain=trajectory_initial_terrain,
                open_cells=trajectory_initial_open_cells,
            )
            bucket = _terrain_bucket_key(initial_snapshot)
            trajectory_terrain_buckets[bucket] += 1
            final_score = payload.get("internal_score", payload.get("final_score"))
            if isinstance(final_score, (int, float)):
                trajectory_terrain_bucket_scores.setdefault(bucket, []).append(float(final_score))
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
            for identity in _normalized_affix_ids(
                names=brush.get("stage2"),
                codes=brush.get("stage2_codes"),
                identities=brush.get("stage2_ids"),
            ):
                trajectory_affix_ids[identity] += 1
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
            chosen_affix_id = (
                active_affix.get("canonical_affix_id")
                or step.get("canonical_affix_id")
                or canonical_affix_id(chosen_affix_code or chosen_affix)
            )
            chosen_affix_key = str(chosen_affix_id or chosen_affix_code or chosen_affix or "")
            if chosen_affix_key and chosen_affix_key != previous_affix_choice:
                if chosen_affix:
                    imitation_affix_choices[str(chosen_affix)] += 1
                if chosen_affix_code:
                    imitation_affix_choice_codes[str(chosen_affix_code)] += 1
                if chosen_affix_id:
                    imitation_affix_choice_ids[str(chosen_affix_id)] += 1
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
        "affixes": max(
            sum(trajectory_affix_ids.values()),
            sum(trajectory_affixes.values()),
            sum(trajectory_affix_codes.values()),
        ),
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
        affix_choice_ids=imitation_affix_choice_ids,
        recommendation_hits=imitation_recommendation_hits,
        recommendation_shorts=imitation_recommendation_shorts,
        recommendation_tags=imitation_recommendation_tags,
    )
    expert_sample_stage = _expert_sample_stage(
        trajectory_count=len(trajectories),
        trajectory_scores=trajectory_scores,
        high_score_gate_reasons=high_score_gate_reasons,
        compressed_count=compressed_run_summaries,
    )
    terrain_prior = _terrain_prior_summary(
        high_terrain=trajectory_initial_terrain,
        high_open_cells=trajectory_initial_open_cells,
        high_samples=trajectory_initial_snapshots,
        compressed_terrain=compressed_initial_terrain,
        compressed_open_cells=compressed_initial_open_cells,
        compressed_samples=compressed_run_summaries,
        compressed_scores=compressed_summary_scores,
        high_buckets=trajectory_terrain_buckets,
        high_bucket_scores=trajectory_terrain_bucket_scores,
        compressed_buckets=compressed_terrain_buckets,
        compressed_bucket_scores=compressed_terrain_bucket_scores,
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
        "affix_id_seen": dict(affix_id_seen),
        "affix_seen_total": sum(affix_seen.values()),
        "affix_code_seen_total": sum(affix_code_seen.values()),
        "affix_id_seen_total": sum(affix_id_seen.values()),
        "affix_seen_pct": _counter_percentages(affix_seen),
        "affix_code_seen_pct": _counter_percentages(affix_code_seen),
        "affix_id_seen_pct": _counter_percentages(affix_id_seen),
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
            "stage2_ids": _counter_distribution(affix_id_seen),
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
        "trajectory_affix_ids": dict(trajectory_affix_ids),
        "high_score_gate_primary": dict(high_score_gate_primary),
        "high_score_gate_reasons": dict(high_score_gate_reasons),
        "policy_prior_samples": policy_prior_samples,
        "imitation_samples": imitation_policy["samples"],
        "imitation_ready": imitation_policy["ready"],
        "imitation_policy": imitation_policy,
        "expert_sample_stage": expert_sample_stage,
        "terrain_prior": terrain_prior,
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
        "trajectory_initial_snapshots": trajectory_initial_snapshots,
        "trajectory_initial_terrain": _terrain_counter_summary(trajectory_initial_terrain, trajectory_initial_snapshots),
        "trajectory_initial_open_cells": _counter_distribution(trajectory_initial_open_cells),
        "compressed_run_summaries": compressed_run_summaries,
        "compressed_summary_kinds": dict(compressed_summary_kinds),
        "compressed_summary_kind_pct": _counter_distribution(compressed_summary_kinds),
        "compressed_summary_skip_reasons": dict(compressed_summary_skip_reasons),
        "compressed_summary_avg_score": (
            sum(compressed_summary_scores) / len(compressed_summary_scores)
        ) if compressed_summary_scores else None,
        "compressed_summary_weighted_avg_score": (
            compressed_summary_weighted_score / compressed_summary_weight_total
        ) if compressed_summary_weight_total else None,
        "compressed_summary_best_score": max(compressed_summary_scores) if compressed_summary_scores else None,
        "compressed_initial_terrain": _terrain_counter_summary(compressed_initial_terrain, compressed_run_summaries),
        "compressed_initial_open_cells": _counter_distribution(compressed_initial_open_cells),
        "compressed_initial_signatures": _counter_distribution(compressed_initial_signatures),
        "compressed_terminal_counts": dict(compressed_terminal_counts),
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


def _accumulate_initial_terrain(
    snapshot: dict[str, Any],
    *,
    terrain: Counter,
    open_cells: Counter | None = None,
    signatures: Counter | None = None,
) -> None:
    if not isinstance(snapshot, dict):
        return
    board = snapshot.get("board") if isinstance(snapshot.get("board"), dict) else {}
    values = {
        "blank": _safe_int(board.get("blank")),
        "blocked": _safe_int(board.get("blocked")),
        "void": _safe_int(board.get("void")),
        "faded": _safe_int(board.get("faded")),
        "paint": _safe_int(board.get("paint")),
    }
    values["known_cells"] = sum(values.values())
    values["open_cells"] = values["blank"] + values["paint"]
    for key, value in values.items():
        terrain[key] += value
    if open_cells is not None:
        open_cells[str(values["open_cells"])] += 1
    if signatures is not None and snapshot.get("signature"):
        signatures[str(snapshot.get("signature"))] += 1


def _terrain_counter_summary(counter: Counter, samples: int) -> dict[str, Any]:
    sample_count = int(samples or 0)
    totals = {str(key): int(value) for key, value in counter.items()}
    averages = {
        str(key): round(float(value) / float(sample_count), 3)
        for key, value in counter.items()
        if sample_count > 0
    }
    return {
        "samples": sample_count,
        "totals": totals,
        "avg": averages,
    }


def _expert_sample_stage(
    *,
    trajectory_count: int,
    trajectory_scores: list[float],
    high_score_gate_reasons: Counter,
    compressed_count: int,
) -> dict[str, Any]:
    treasure7 = int(high_score_gate_reasons.get("treasure7_plus", 0))
    many_lv7 = int(high_score_gate_reasons.get("many_level7", 0))
    treasure6 = int(high_score_gate_reasons.get("treasure6_plus", 0))
    lv7 = int(high_score_gate_reasons.get("level7_paint", 0))
    stage = 0
    label = "样本不足"
    if treasure6 > 0 or lv7 > 0:
        stage = 1
        label = "阶段1：6级箱子/任意7级颜料"
    if treasure7 > 0 or many_lv7 > 0 or (treasure6 > 0 and lv7 >= 2):
        stage = 2
        label = "阶段2：7级箱子/多个7级/6级箱子+多个7级"
    if trajectory_count >= 80 and trajectory_scores:
        stage = 3
        label = "阶段3：可筛选前20%专家样本"
    sorted_scores = sorted((float(score) for score in trajectory_scores), reverse=True)
    top20_count = max(1, int(len(sorted_scores) * 0.2)) if sorted_scores else 0
    top20_threshold = sorted_scores[top20_count - 1] if top20_count else None
    return {
        "stage": stage,
        "label": label,
        "trajectory_count": int(trajectory_count),
        "compressed_count": int(compressed_count),
        "top20_threshold": round(top20_threshold, 3) if top20_threshold is not None else None,
        "ready_for_prior": stage >= 1,
        "ready_for_expert_filter": stage >= 3,
    }


def _terrain_prior_summary(
    *,
    high_terrain: Counter,
    high_open_cells: Counter,
    high_samples: int,
    high_buckets: Counter,
    high_bucket_scores: dict[str, list[float]],
    compressed_terrain: Counter,
    compressed_open_cells: Counter,
    compressed_samples: int,
    compressed_scores: list[float],
    compressed_buckets: Counter,
    compressed_bucket_scores: dict[str, list[float]],
) -> dict[str, Any]:
    high_avg = _terrain_avg(high_terrain, high_samples)
    compressed_avg = _terrain_avg(compressed_terrain, compressed_samples)
    open_bias = _open_cell_bias(high_open_cells, compressed_open_cells)
    enough_samples = high_samples >= _config_int("learning.terrain_min_samples", 20)
    if compressed_samples >= _config_int("learning.terrain_min_samples", 20):
        enough_samples = True
    deltas = {
        key: round(float(high_avg.get(key, 0.0)) - float(compressed_avg.get(key, 0.0)), 3)
        for key in {"blank", "paint", "blocked", "void", "faded", "open_cells"}
    }
    return {
        "samples": {
            "high": int(high_samples),
            "compressed": int(compressed_samples),
        },
        "ready": bool(enough_samples and (high_samples > 0 or compressed_samples > 0)),
        "high_avg": high_avg,
        "compressed_avg": compressed_avg,
        "delta_high_minus_compressed": deltas,
        "open_cell_bias": open_bias,
        "terrain_buckets": _terrain_bucket_summary(
            high_buckets=high_buckets,
            high_bucket_scores=high_bucket_scores,
            compressed_buckets=compressed_buckets,
            compressed_bucket_scores=compressed_bucket_scores,
        ),
        "compressed_avg_score": round(sum(compressed_scores) / len(compressed_scores), 3) if compressed_scores else None,
    }


def _terrain_avg(counter: Counter, samples: int) -> dict[str, float]:
    if samples <= 0:
        return {}
    return {
        str(key): round(float(value) / float(samples), 3)
        for key, value in counter.items()
    }


def _open_cell_bias(high_open_cells: Counter, compressed_open_cells: Counter) -> dict[str, Any]:
    keys = sorted({*high_open_cells.keys(), *compressed_open_cells.keys()}, key=lambda item: int(item) if str(item).isdigit() else 0)
    high_total = sum(high_open_cells.values())
    compressed_total = sum(compressed_open_cells.values())
    items: list[dict[str, Any]] = []
    for key in keys:
        high_rate = float(high_open_cells.get(key, 0)) / float(high_total) if high_total else 0.0
        compressed_rate = float(compressed_open_cells.get(key, 0)) / float(compressed_total) if compressed_total else 0.0
        items.append(
            {
                "key": str(key),
                "high_count": int(high_open_cells.get(key, 0)),
                "compressed_count": int(compressed_open_cells.get(key, 0)),
                "bias": round(high_rate - compressed_rate, 6),
                "bias_pct": round((high_rate - compressed_rate) * 100.0, 3),
            }
        )
    items.sort(key=lambda item: (item["bias"], item["high_count"]), reverse=True)
    return {"items": items, "high_total": int(high_total), "compressed_total": int(compressed_total)}


def _terrain_bucket_key(snapshot: dict[str, Any]) -> str:
    board = snapshot.get("board") if isinstance(snapshot.get("board"), dict) else {}
    blank = _safe_int(board.get("blank"))
    paint = _safe_int(board.get("paint"))
    blocked = _safe_int(board.get("blocked"))
    void = _safe_int(board.get("void"))
    states = str(board.get("states") or "")
    open_cells = blank + paint
    center_open = _center_open_count(states)
    return "|".join(
        (
            f"open:{_bucket_range(open_cells, (16, 22))}",
            f"block:{_bucket_range(blocked, (4, 8))}",
            f"void:{_bucket_range(void, (8, 13))}",
            f"center:{_bucket_range(center_open, (1, 3))}",
        )
    )


def _center_open_count(states: str) -> int:
    if len(states) < 36:
        return 0
    count = 0
    for index in (14, 15, 20, 21):
        if states[index] in {"B", "P"}:
            count += 1
    return count


def _bucket_range(value: int, cuts: tuple[int, int]) -> str:
    low, high = cuts
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def _terrain_bucket_summary(
    *,
    high_buckets: Counter,
    high_bucket_scores: dict[str, list[float]],
    compressed_buckets: Counter,
    compressed_bucket_scores: dict[str, list[float]],
) -> dict[str, Any]:
    keys = set(high_buckets) | set(compressed_buckets)
    items: list[dict[str, Any]] = []
    for key in keys:
        high_count = int(high_buckets.get(key, 0))
        compressed_count = int(compressed_buckets.get(key, 0))
        total = high_count + compressed_count
        success_rate = float(high_count) / float(total) if total else 0.0
        high_avg = _avg(high_bucket_scores.get(key, []))
        compressed_avg = _avg(compressed_bucket_scores.get(key, []))
        score_gap = None
        if high_avg is not None and compressed_avg is not None:
            score_gap = round(high_avg - compressed_avg, 3)
        items.append(
            {
                "key": str(key),
                "high": high_count,
                "compressed": compressed_count,
                "total": total,
                "success_rate": round(success_rate, 6),
                "success_pct": round(success_rate * 100.0, 3),
                "high_avg_score": high_avg,
                "compressed_avg_score": compressed_avg,
                "score_gap": score_gap,
            }
        )
    items.sort(key=lambda item: (item["success_rate"], item["high"], item.get("score_gap") or 0.0), reverse=True)
    return {"items": items, "total": int(sum(item["total"] for item in items))}


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _unknown_text(value: Any) -> str | None:
    text = str(value).strip()
    if not text or text.casefold() in {"none", "null", "unknown", "?"}:
        return None
    return text


def _normalized_affix_ids(
    *,
    names: Any = None,
    codes: Any = None,
    identities: Any = None,
) -> tuple[str, ...]:
    for values in (identities, codes, names):
        if not isinstance(values, (list, tuple)) or not values:
            continue
        normalized = tuple(
            identity
            for value in values
            if (identity := canonical_affix_id(str(value))) is not None
        )
        if normalized:
            return normalized
    return ()


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
    affix_choice_ids: Counter,
    recommendation_hits: Counter,
    recommendation_shorts: Counter,
    recommendation_tags: Counter,
) -> dict[str, Any]:
    samples = {
        "trajectories": int(trajectory_count),
        "actions": int(sum(action_counts.values())),
        "placements": int(sum(placement_cells.values())),
        "brushes": int(max(sum(brush_choices.values()), sum(brush_choice_codes.values()))),
        "affixes": int(max(
            sum(affix_choice_ids.values()),
            sum(affix_choices.values()),
            sum(affix_choice_codes.values()),
        )),
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
        "top_affix_ids": _counter_distribution(affix_choice_ids),
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
    top3 = sum(int(counter.get(f"rank_{rank}", 0)) for rank in range(1, 4))
    return {
        "decisions": decisions,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(float(hits) / float(decisions), 6) if decisions else 0.0,
        "hit_pct": round(float(hits) * 100.0 / float(decisions), 3) if decisions else 0.0,
        "top1": top1,
        "top1_rate": round(float(top1) / float(decisions), 6) if decisions else 0.0,
        "top1_pct": round(float(top1) * 100.0 / float(decisions), 3) if decisions else 0.0,
        "top3": top3,
        "top3_rate": round(float(top3) / float(decisions), 6) if decisions else 0.0,
        "top3_pct": round(float(top3) * 100.0 / float(decisions), 3) if decisions else 0.0,
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
