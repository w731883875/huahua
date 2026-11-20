"""笔锋选择面板（选笔触方向 / 选笔触效果）的文字解析。

面板是左侧一列卡片，每张卡片长得几乎一样，只差文字。原来靠「字形模板打分 + 阈值」
硬猜，结果很糟：**「选笔触效果」阶段被当成「选笔触方向」**，报出一堆假的笔锋类型，
而真正的词缀描述又匹配到别的条目上。

真正的判据是文字，而且文字结构非常规整：

- 「选笔触方向」每张卡片是 `【转锋】转变调色目标和效果`，
  描述部分是**固定三条**（转锋 / 堆砌 / 扩散），长且互不重叠；
- 「选笔触效果」每张卡片是一句词缀描述，来自官方词缀表。

所以先判断「面板里有没有出现那三条固定描述」——有就是方向阶段，没有就是效果阶段。
11 张截图实测这条判据干净利落：

| 阶段 | 三条描述的最高匹配度 |
|---|---|
| 选笔触方向（4 张） | 0.89 ~ 1.00 |
| 选笔触效果（4 张） | 0.11 ~ 0.33 |

中间隔了 0.56，取 0.60 作阈值。
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import windows_ocr
from .game_model import BrushType, build_affix_catalog

# 三条固定的笔触方向描述。OCR 会把「【转锋】」的方括号读花，但描述本体读得很稳。
BRUSH_DESCRIPTIONS: tuple[tuple[BrushType, str], ...] = (
    (BrushType.TRANSFORM, "转变调色目标和效果"),
    (BrushType.STACK, "附加额外的调色效果"),
    (BrushType.SPREAD, "对特殊地块产生作用"),
)

# 描述匹配度阈值。实测方向阶段 ≥0.89、效果阶段 ≤0.33，取中间偏下。
DESCRIPTION_THRESHOLD = 0.60

# 词缀描述匹配度阈值。OCR 少一两个字仍能对上，取 0.55。
AFFIX_THRESHOLD = 0.55

# 面板整体放大倍数（正文约 13px，3 倍最稳）。
PANEL_SCALE = 3.0
# 单张卡片放大倍数（只看文字区，跳过左侧图标）。
CARD_SCALE = 4.0

# 卡片文字区左边界：跳过左侧图标/留白。实测 0.12 时 OCR 最干净。
CARD_TEXT_LEFT = 0.12
CARD_TEXT_RIGHT = 0.80

_SPACES = re.compile(r"[\s\u3000]+")


def normalize(text: str) -> str:
    return _SPACES.sub("", str(text))


def lcs_ratio(needle: str, haystack: str) -> float:
    """``needle`` 有多少比例按顺序出现在 ``haystack`` 里。"""
    if not needle or not haystack:
        return 0.0
    previous = [0] * (len(haystack) + 1)
    for ch in needle:
        current = [0]
        for index, other in enumerate(haystack, start=1):
            current.append(
                previous[index - 1] + 1 if ch == other else max(previous[index], current[index - 1])
            )
        previous = current
    return previous[-1] / len(needle)


def dice_similarity(left: str, right: str) -> float:
    """2·LCS / (|left| + |right|)。

    单纯用「needle 覆盖率」会让短别名占便宜——「范围内随机1个颜料提升1级」是
    「对作用范围内的随机1个颜料额外提升1级」的子串，覆盖率 1.00，会顶掉正确答案。
    除以两边长度之和就把「长度对得上」也算进去，长而完整的条目自然胜出。
    """
    if not left or not right:
        return 0.0
    length = lcs_length(left, right)
    return 2.0 * length / (len(left) + len(right))


def lcs_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for ch in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1 if ch == other else max(previous[index], current[index - 1])
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class BrushPanelReading:
    """一屏笔锋面板读出来的东西。"""

    #: ``"direction"``（选笔触方向）/ ``"effect"``（选笔触效果）/ ``"unknown"``
    stage: str = "unknown"
    brush_types: tuple[BrushType, ...] = ()
    affixes: tuple[str, ...] = ()
    affix_scores: dict[str, float] = field(default_factory=dict)
    description_scores: dict[str, float] = field(default_factory=dict)
    lines: tuple[str, ...] = ()
    confidence: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.stage == "unknown" and not self.brush_types and not self.affixes


# 卡片版式（相对面板高度的比例）。实测 8 张截图里两张卡片都落在这一组位置上，
# 差别不到 2%。亮度剖面切不开时用它兜底。
CARD_SLOTS: tuple[tuple[float, float], ...] = ((0.108, 0.488), (0.520, 0.900))


def gutter_bands(crop: np.ndarray) -> list[tuple[int, int]]:
    """切出每张卡片的上下边界。

    主判据是「卡片左侧图标列」的亮度起伏：卡片内部明显亮于卡片之间的缝隙。
    但有两张卡片贴得很近、缝隙只有 10 来个亮度差的时候，单靠阈值会连成一条，
    这时退回固定版式（:data:`CARD_SLOTS`）。
    """
    if crop.size == 0:
        return []
    height, width = crop.shape[:2]
    gutter = crop[:, int(width * 0.02) : max(int(width * 0.02) + 1, int(width * 0.14))]
    profile = cv2.cvtColor(gutter, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32).mean(axis=1)
    kernel = max(3, int(height * 0.02) | 1)
    smooth = np.convolve(profile, np.ones(kernel, dtype=np.float32) / kernel, mode="same")

    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y, value in enumerate(smooth > 160.0):
        if value and start is None:
            start = y
        elif not value and start is not None:
            if y - start > height * 0.14:
                bands.append((start, y))
            start = None
    if start is not None and height - start > height * 0.14:
        bands.append((start, height))

    if len(bands) == 2:
        return bands
    if not bands:
        return []
    y0, y1 = bands[0]
    if y1 - y0 > height * 0.62:
        # 一整条横跨面板：两张卡片被连成一条，按固定版式切开
        return [(int(height * top), int(height * bottom)) for top, bottom in CARD_SLOTS]
    return bands


def _affix_catalog_index() -> tuple[tuple[str, str], ...]:
    """(官方名, 展示名) 对，供模糊匹配。"""
    pairs: list[tuple[str, str]] = []
    for affix in build_affix_catalog():
        names = (affix.display_name, *affix.aliases)
        for name in names:
            cleaned = normalize(name)
            if cleaned:
                pairs.append((affix.display_name, cleaned))
    return tuple(pairs)


def match_affix(text: str) -> tuple[str, float]:
    """把一段 OCR 文本对到官方词缀名上，返回 (名字, 匹配度)。"""
    cleaned = normalize(text)
    if not cleaned:
        return "", 0.0
    best_name = ""
    best_score = 0.0
    for display_name, alias in _affix_catalog_index():
        score = dice_similarity(alias, cleaned)
        if score > best_score:
            best_score = score
            best_name = display_name
    return best_name, round(best_score, 3)


def match_affixes(texts: list[str]) -> tuple[tuple[str, ...], dict[str, float]]:
    """多条文本 → 去重后的词缀名列表（保持出现顺序）。"""
    names: list[str] = []
    scores: dict[str, float] = {}
    for text in texts:
        name, score = match_affix(text)
        if not name or score < AFFIX_THRESHOLD:
            continue
        scores[name] = max(scores.get(name, 0.0), score)
        if name not in names:
            names.append(name)
    return tuple(names), scores


def parse_lines(lines: tuple[str, ...] | list[str]) -> BrushPanelReading:
    """纯函数：面板 OCR 行 → 结构化结果（只看阶段和笔锋类型）。

    词缀要按卡片分别读，所以不在这里做，交给 :func:`read_brush_panel`。
    """
    normalized = [normalize(line) for line in lines]
    normalized = [line for line in normalized if line]
    joined = "".join(normalized)
    if not joined:
        return BrushPanelReading()

    description_scores = {
        brush_type.value: round(lcs_ratio(description, joined), 3)
        for brush_type, description in BRUSH_DESCRIPTIONS
    }
    matched = [
        brush_type
        for brush_type, _description in BRUSH_DESCRIPTIONS
        if description_scores[brush_type.value] >= DESCRIPTION_THRESHOLD
    ]
    if matched:
        return BrushPanelReading(
            stage="direction",
            brush_types=tuple(matched),
            description_scores=description_scores,
            lines=tuple(normalized),
            confidence=max(description_scores[item.value] for item in matched),
        )
    return BrushPanelReading(
        stage="effect",
        description_scores=description_scores,
        lines=tuple(normalized),
        confidence=0.70,
    )


def _card_text(crop: np.ndarray, y0: int, y1: int) -> np.ndarray:
    height, width = crop.shape[:2]
    band = crop[max(0, y0 - 2) : min(height, y1 + 2), int(width * CARD_TEXT_LEFT) : int(width * CARD_TEXT_RIGHT)]
    if band.size == 0:
        return band
    return cv2.resize(band, None, fx=CARD_SCALE, fy=CARD_SCALE, interpolation=cv2.INTER_CUBIC)


def _card_texts(crop: np.ndarray, reader) -> tuple[list[str], bool]:
    """逐卡片读文字。``reader`` 是同步/异步的 OCR 入口。

    返回 (卡片文本列表, 是否还有卡片没读完)。异步版拿到 ``None`` 时把 pending 置真，
    调用方可以先返回阶段结果，词缀下一帧再补。
    """
    height, width = crop.shape[:2]
    bands = gutter_bands(crop)
    if not bands:
        # 分不出卡片时退化成整块面板
        band = crop[:, int(width * CARD_TEXT_LEFT) : int(width * CARD_TEXT_RIGHT)]
        if band.size == 0:
            return [], False
        scaled = cv2.resize(band, None, fx=CARD_SCALE, fy=CARD_SCALE, interpolation=cv2.INTER_CUBIC)
        lines = reader(scaled)
        if lines is None:
            return [], True
        return (["".join(normalize(line) for line in lines)] if lines else []), False

    texts: list[str] = []
    pending = False
    for y0, y1 in bands:
        card = _card_text(crop, y0, y1)
        if card.size == 0:
            continue
        lines = reader(card)
        if lines is None:
            pending = True
            continue
        if lines:
            texts.append("".join(normalize(line) for line in lines))
    return texts, pending


# 面板签名缓存。整块面板放大 3 倍再编码要 20~40ms，每帧都算受不了，
# 所以先用一个 32×12 的小图哈希当键：面板没换就直接返回上次的解析结果。
_SIGNATURE_CACHE: OrderedDict[str, "BrushPanelReading"] = OrderedDict()
_LINES_CACHE: OrderedDict[str, tuple[str, ...]] = OrderedDict()
_CACHE_LIMIT = 16


def panel_signature(crop: np.ndarray) -> str:
    small = cv2.resize(crop, (32, 12), interpolation=cv2.INTER_AREA)
    return hashlib.sha1(small.tobytes()).hexdigest()


def _cache_get(cache: OrderedDict, key: str):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key: str, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def reset_cache() -> None:
    _SIGNATURE_CACHE.clear()
    _LINES_CACHE.clear()


def _build_reading(lines: tuple[str, ...], crop: np.ndarray, reader) -> tuple[BrushPanelReading, bool]:
    """由面板 OCR 行 + 逐卡片 OCR 拼出完整结果。第二个返回值表示「还没读完」。"""
    reading = parse_lines(lines)
    if reading.stage != "effect":
        return reading, False
    texts, pending = _card_texts(crop, reader)
    if not texts:
        if pending:
            return reading, True
        texts = list(reading.lines[1:])  # 退化成整面板的逐行
    affixes, affix_scores = match_affixes(texts)
    return (
        BrushPanelReading(
            stage="effect",
            affixes=affixes,
            affix_scores=affix_scores,
            description_scores=reading.description_scores,
            lines=reading.lines,
            confidence=max(0.70, max(affix_scores.values(), default=0.0)),
        ),
        False,
    )


def read_brush_panel(crop: np.ndarray) -> BrushPanelReading | None:
    """同步读一屏笔锋面板。"""
    if crop is None or crop.size == 0:
        return None
    key = panel_signature(crop)
    cached = _cache_get(_SIGNATURE_CACHE, key)
    if cached is not None:
        return cached

    lines = _cache_get(_LINES_CACHE, key)
    if lines is None:
        panel = cv2.resize(crop, None, fx=PANEL_SCALE, fy=PANEL_SCALE, interpolation=cv2.INTER_CUBIC)
        lines = windows_ocr.recognize_chinese_lines(panel)
        if not lines:
            return None
        _cache_put(_LINES_CACHE, key, lines)

    reading, pending = _build_reading(lines, crop, windows_ocr.recognize_chinese_lines)
    if reading.is_empty:
        return None
    if not pending:
        _cache_put(_SIGNATURE_CACHE, key, reading)
    return reading


def read_brush_panel_async(crop: np.ndarray) -> BrushPanelReading | None:
    """异步版：缓存命中立即返回，否则丢后台线程，这一帧返回 ``None``。

    实时帧用这个。逐卡片读词缀那一步也走异步——第一次返回时可能只有阶段结果、
    没有词缀，下一帧补上；面板和卡片文本都进缓存，稳定状态下几乎零成本。
    """
    if crop is None or crop.size == 0:
        return None
    key = panel_signature(crop)
    cached = _cache_get(_SIGNATURE_CACHE, key)
    if cached is not None:
        return cached

    lines = _cache_get(_LINES_CACHE, key)
    if lines is None:
        panel = cv2.resize(crop, None, fx=PANEL_SCALE, fy=PANEL_SCALE, interpolation=cv2.INTER_CUBIC)
        lines = windows_ocr.recognize_chinese_lines_async(panel)
        if not lines:
            return None
        _cache_put(_LINES_CACHE, key, lines)

    reading, pending = _build_reading(lines, crop, windows_ocr.recognize_chinese_lines_async)
    if not pending:
        _cache_put(_SIGNATURE_CACHE, key, reading)
    return reading


