"""「当前颜料」面板的文字解析。

游戏右侧面板固定三行，内容全在文字里：

```
玩法颜料·怪谈LV5
调色目标:在相邻地块上放置所有颜料(0/4)
调色效果:使相邻的所有颜料提升2级
```

以前这些信息是靠「字形模板逐个打分 + 阈值」猜出来的，结果很脆：11 张截图里
``target_object`` 只有 2 张读对，``match_current`` 永远是 0，``match_required``
靠 ``_empirical_current_defaults`` 按等级瞎填，``subtype`` 从来没填过。

改用 Windows OCR 直接读这三行后（11 张截图实测）：

| 字段 | 命中 |
|---|---|
| 类别（怪物/玩法） | 11/11 |
| 子类型（邪影/怪谈/赛季…） | 10/11 |
| 等级 | 10/11 |
| 方向（横向/纵向/对角斜向/相邻） | 11/11 |
| 目标（所有/怪物/玩法） | 10/11 |
| 升级级数 | 11/11 |
| 进度 (x/N) | 9/11 |

OCR 单次 70~120ms，进不了每帧路径，所以分两条路：

- :func:`read_panel` 同步版，给冷启动 / 单张截图分析用；
- :func:`read_panel_async` 异步版，给实时帧用——缓存命中就立即返回，
  没命中就丢后台线程，这一帧先沿用上一帧的结果。

解析本身是纯函数（:func:`parse_lines`），不依赖 OCR，方便单测。
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import windows_ocr
from .models import PaintCategory, PaintDirection, PaintObject

# 面板文字很小（正文约 14px），放大 3 倍再喂 OCR 最稳。
OCR_SCALE = 3.0

# 实时帧读面板的最小间隔。面板有动画，像素哈希必然每帧都变，
# 只能按时间节流：0.35 秒一次，其余帧复用上次结果。
PANEL_READ_INTERVAL = 0.35
PANEL_RESULT_MAX_AGE = 1.5

_reading_lock = threading.Lock()
_last_reading: "PanelReading | None" = None
_last_reading_at = 0.0
_read_inflight = False
_reading_generation = 0
_input_signature: np.ndarray | None = None
_last_completed_at = 0.0

# 方向词按长度降序，避免「对角斜向」被「横向」截胡。
_DIRECTION_KEYWORDS: tuple[tuple[str, PaintDirection], ...] = (
    ("对角斜向", PaintDirection.DIAGONAL),
    ("相邻", PaintDirection.ADJACENT),
    ("横向", PaintDirection.HORIZONTAL),
    ("纵向", PaintDirection.VERTICAL),
    ("全图", PaintDirection.GLOBAL),
)

_OBJECT_KEYWORDS: tuple[tuple[str, PaintObject], ...] = (
    ("阻挡地块", PaintObject.BLOCKED_TILE),
    ("空白地块", PaintObject.BLANK_TILE),
    ("虚空地块", PaintObject.VOID_TILE),
    ("所有", PaintObject.ALL),
    ("怪物", PaintObject.MONSTER),
    ("玩法", PaintObject.GAMEPLAY),
)

# Windows OCR renders the game's separator as '-', '·', '．' or '.'.
# Category + official subtype + LV are the header anchor, not that punctuation.
_HEADER_RE = re.compile(r"(怪物|玩法)颜料[-·・.．:：]?(.*?)LV([1-7])(?!\d)", re.IGNORECASE)
_PROGRESS_RE = re.compile(r"\((\d+)/(\d+)\)")
_UPGRADE_RE = re.compile(r"提升(\d)级")

# 官方 27 种颜料的子类型名，用来从 OCR 噪声里挑出最像的那个。
_SUBTYPE_CACHE: tuple[str, ...] | None = None


def known_subtypes() -> tuple[str, ...]:
    """官方配置里的全部颜料子类型名（13 怪物 + 14 玩法）。"""
    global _SUBTYPE_CACHE
    if _SUBTYPE_CACHE is None:
        from .s8_official_data import S8_PLOT_ITEMS

        names: list[str] = []
        for item in S8_PLOT_ITEMS:
            prefix = "怪物颜料-" if item.category == "MONSTER" else "玩法颜料-"
            if item.name.startswith(prefix):
                name = item.name[len(prefix) :]
                if name and name not in names:
                    names.append(name)
        _SUBTYPE_CACHE = tuple(names)
    return _SUBTYPE_CACHE


def normalize_line(text: str) -> str:
    """去掉 OCR 塞进来的空格，全角标点转半角。"""
    cleaned = "".join(str(text).split())
    table = str.maketrans({"（": "(", "）": ")", "：": ":", "／": "/"})
    return cleaned.translate(table)


@dataclass(frozen=True)
class PanelReading:
    """一屏「当前颜料」面板读出来的全部结构化信息。"""

    category: PaintCategory = PaintCategory.UNKNOWN
    subtype: str | None = None
    level: int | None = None
    target_object: PaintObject = PaintObject.UNKNOWN
    direction: PaintDirection = PaintDirection.UNKNOWN
    upgrade_amount: int | None = None
    match_current: int | None = None
    match_required: int | None = None
    lines: tuple[str, ...] = ()
    confidence: float = 0.0

    @property
    def has_rule(self) -> bool:
        return self.direction != PaintDirection.UNKNOWN or self.target_object != PaintObject.UNKNOWN

    @property
    def is_empty(self) -> bool:
        return (
            self.category == PaintCategory.UNKNOWN
            and self.level is None
            and self.match_required is None
            and not self.has_rule
        )

    @property
    def effect_text(self) -> str | None:
        """调色效果正文；与用户选中的笔触效果是两个独立字段。"""
        return _section_text(self.lines, "tint")

    @property
    def brush_effect_text(self) -> str | None:
        """笔触效果正文，保留换行内容，不从调色效果猜测。"""
        return _section_text(self.lines, "brush")


_SECTION_RE = re.compile(r"(调?色效[果里杲采]|[笔笋]触效[果里杲采]|颜料内容|调色目标(?!的))")


def _section(line: str) -> tuple[str | None, str]:
    match = _SECTION_RE.search(line)
    if match is None:
        return None, line
    label = match.group(1)
    kind = ("brush" if "触" in label else "tint" if "效" in label
            else "target" if "目标" in label else "content")
    return kind, line[match.end():].lstrip(":：")


def _section_text(lines, wanted):
    active = None
    parts = []
    for raw in lines:
        line = normalize_line(raw)
        kind, tail = _section(line)
        if kind is not None:
            if active == wanted:
                break
            active = kind
            if active == wanted and tail:
                parts.append(tail)
        elif active == wanted:
            if _HEADER_RE.search(line):
                break
            parts.append(line)
    return "".join(parts) or None


def _pick_subtype(raw: str) -> str | None:
    """把 OCR 读到的子类型对齐到官方名字。"""
    if not raw:
        return None
    names = known_subtypes()
    if raw in names:
        return raw
    # OCR 常把某个字读错一位，按「最长公共子串占比」挑最接近的官方名。
    best: tuple[float, str] | None = None
    for name in names:
        if name in raw or raw in name:
            score = 1.0
        else:
            score = _lcs_ratio(raw, name)
        if best is None or score > best[0]:
            best = (score, name)
    if best is not None and best[0] >= 0.60:
        return best[1]
    return None


def _lcs_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for ch in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if ch == other else max(previous[index], current[index - 1]))
        previous = current
    return previous[-1] / max(len(left), len(right))


def _match_direction(text: str) -> PaintDirection:
    for keyword, direction in _DIRECTION_KEYWORDS:
        if keyword in text:
            return direction
    return PaintDirection.UNKNOWN


def _match_object(text: str) -> PaintObject:
    for keyword, target in _OBJECT_KEYWORDS:
        if keyword in text:
            return target
    return PaintObject.UNKNOWN


def parse_lines(lines: tuple[str, ...] | list[str]) -> PanelReading:
    """把 OCR 出来的若干行解析成 :class:`PanelReading`（纯函数，无 IO）。"""
    normalized = [normalize_line(line) for line in lines]
    normalized = [line for line in normalized if line]
    # Tooltips render the paint title and level on separate rows.
    for index in range(len(normalized)-1):
        if (re.fullmatch(r"(怪物|玩法)颜料[-·・.．:：]?.+", normalized[index])
                and "LV" not in normalized[index].upper()
                and re.fullmatch(r"LV[1-7]", normalized[index+1], re.IGNORECASE)):
            normalized[index] += normalized[index+1]
            normalized[index+1] = ""
    normalized = [line for line in normalized if line]

    # 标题行（「怪物颜料·黑帮LV3」）是面板里唯一的锚点。
    # 搜索区为了兼容窗口化布局会向上多留一截，可能顺带读到上方的其它 UI；
    # 从标题行开始解析，标题之前的一律当噪声丢掉。
    start = 0
    for index, line in enumerate(normalized):
        if _HEADER_RE.search(line) is not None:
            start = index
            break
    body = normalized[start:]

    category = PaintCategory.UNKNOWN
    subtype: str | None = None
    level: int | None = None
    target_object = PaintObject.UNKNOWN
    direction = PaintDirection.UNKNOWN
    upgrade_amount: int | None = None
    match_current: int | None = None
    match_required: int | None = None

    active_section = None
    for line in body:
        header = _HEADER_RE.search(line)
        if header is not None:
            category = PaintCategory.MONSTER if header.group(1) == "怪物" else PaintCategory.GAMEPLAY
            subtype = _pick_subtype(header.group(2))
            level = int(header.group(3))
            active_section = None
            continue

        section, _ = _section(line)
        if section is not None:
            active_section = section
        if active_section in {"brush", "content"}:
            continue

        # Missing headings still allow rule recovery from text. A recognized
        # brush/content section, however, must never contaminate the tint rule.
        if direction == PaintDirection.UNKNOWN:
            direction = _match_direction(line)
        if target_object == PaintObject.UNKNOWN:
            target_object = _match_object(line)
        upgrade = _UPGRADE_RE.search(line)
        if upgrade is not None and upgrade_amount is None:
            upgrade_amount = int(upgrade.group(1))
        # 目标行换行时，「(0/3)」会被单独甩到下一行
        progress = _PROGRESS_RE.search(line)
        if progress is not None and match_required is None:
            match_current = int(progress.group(1))
            match_required = int(progress.group(2))

    confidence = _score_reading(category, level, target_object, direction, match_required, upgrade_amount)
    return PanelReading(
        category=category,
        subtype=subtype,
        level=level,
        target_object=target_object,
        direction=direction,
        upgrade_amount=upgrade_amount,
        match_current=match_current,
        match_required=match_required,
        lines=tuple(normalized),
        confidence=confidence,
    )


def _score_reading(
    category: PaintCategory,
    level: int | None,
    target_object: PaintObject,
    direction: PaintDirection,
    match_required: int | None,
    upgrade_amount: int | None,
) -> float:
    """按读到的字段数给个置信度，0~1。"""
    score = 0.0
    if category != PaintCategory.UNKNOWN:
        score += 0.24
    if level is not None:
        score += 0.16
    if direction != PaintDirection.UNKNOWN:
        score += 0.22
    if target_object != PaintObject.UNKNOWN:
        score += 0.18
    if match_required is not None:
        score += 0.12
    if upgrade_amount is not None:
        score += 0.08
    return round(min(1.0, score), 3)


def prepare_for_ocr(text_crop: np.ndarray) -> np.ndarray:
    """放大到 OCR 舒服的尺寸。太小的图 OCR 会整行丢掉。"""
    if text_crop.size == 0:
        return text_crop
    height, width = text_crop.shape[:2]
    return cv2.resize(
        text_crop,
        (max(1, int(width * OCR_SCALE)), max(1, int(height * OCR_SCALE))),
        interpolation=cv2.INTER_CUBIC,
    )


def read_panel(text_crop: np.ndarray) -> PanelReading | None:
    """同步读一屏面板文字。空图 / OCR 不可用时返回 ``None``。"""
    if text_crop is None or text_crop.size == 0:
        return None
    lines = windows_ocr.recognize_chinese_lines(prepare_for_ocr(text_crop))
    if not lines:
        return None
    reading = parse_lines(lines)
    return None if reading.is_empty else reading


def read_panel_async(text_crop: np.ndarray) -> PanelReading | None:
    """异步读面板：按时间节流，其余帧复用上一次成功的结果。

    这里原来直接用 ``windows_ocr`` 的像素缓存，但面板有呼吸/光效动画，
    每帧像素都不一样，sha1 永远不命中 → 每帧都新起一个 OCR 线程 →
    线程堆积、结果永远拿不到，上层只能一直沿用「方向未知」的旧值。
    （实测：实时帧里方向/目标从来没被文字覆盖过。）

    每 ``PANEL_READ_INTERVAL`` 秒最多提交一次 OCR。文字区域明显变化时，旧结果
    和旧请求立即失效；细微进度变化在下一次读取后更新。连续读取失败时，结果
    最多保留 ``PANEL_RESULT_MAX_AGE`` 秒，避免一直显示上一个颜料。
    """
    global _last_reading, _last_reading_at, _read_inflight, _reading_generation, _input_signature

    if text_crop is None or text_crop.size == 0:
        return None
    now = time.monotonic()
    signature = _panel_signature(text_crop)
    with _reading_lock:
        if _input_signature is not None and not _same_panel(_input_signature, signature):
            # A reroll/new paint can keep the same window and panel rectangle.
            # Discard both cached text and the outstanding result from the old UI.
            _reading_generation += 1
            _last_reading = None
            _last_reading_at = 0.0
            _read_inflight = False
            _input_signature = signature
        elif _input_signature is None:
            _input_signature = signature
        cached = _last_reading if now - _last_completed_at <= PANEL_RESULT_MAX_AGE else None
        if _read_inflight:
            return cached
        if now - _last_reading_at < PANEL_READ_INTERVAL:
            return cached
        _read_inflight = True
        generation = _reading_generation
        # 提交时就记时间：万一这一轮 OCR 失败，也别每帧重试把线程打爆。
        _last_reading_at = now
    payload = prepare_for_ocr(text_crop).copy()
    threading.Thread(
        target=_refresh_reading,
        args=(payload, generation),
        name="huahua-panel-ocr",
        daemon=True,
    ).start()
    return cached


def _panel_signature(crop: np.ndarray) -> np.ndarray:
    """Dark lettering, with aspect preserved so crop-edge jitter can be aligned."""
    return windows_ocr.text_signature(crop)


def _same_panel(left: np.ndarray, right: np.ndarray) -> bool:
    # Small progress/word edits are refreshed by the regular 350 ms OCR. This
    # gate rejects a different panel without repeatedly cancelling animated UI.
    return windows_ocr.same_text(left, right)


def reading_generation() -> int:
    return _reading_generation


def _refresh_reading(payload: np.ndarray, generation: int) -> None:
    global _last_reading, _read_inflight, _last_completed_at
    try:
        lines = windows_ocr.recognize_chinese_lines(payload)
        if not lines:
            return
        reading = parse_lines(lines)
        if reading.is_empty:
            return
        with _reading_lock:
            if generation == _reading_generation:
                _last_reading = reading
                _last_completed_at = time.monotonic()
    except Exception:
        return
    finally:
        with _reading_lock:
            if generation == _reading_generation:
                _read_inflight = False


def last_reading() -> PanelReading | None:
    """最近一次成功读到的面板（调试/提示用）。"""
    return _last_reading


def reset_async_state() -> None:
    global _last_reading, _last_reading_at, _read_inflight, _reading_generation, _input_signature, _last_completed_at
    with _reading_lock:
        _reading_generation += 1
        _last_reading = None
        _last_reading_at = 0.0
        _read_inflight = False
        _input_signature = None
        _last_completed_at = 0.0
