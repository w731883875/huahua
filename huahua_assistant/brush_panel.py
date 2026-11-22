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
import threading
import time
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

# 「这块区域是不是两张笔触卡片」的分带判据（都是相对面板高度的比例）。
#
# 2026-11-20 实机事故：原来的判据是「切出 ≥2 张宽卡片，宽高比 ≥3.8」。
# 用户新窗口 1607x822 下两张卡片切成 404x116 / 390x117，宽高比 3.48 / 3.33，
# 一个都过不了 —— 于是 `_detect_brush_options_mode` 判否、BRUSH_OPTIONS 区域根本
# 没建，整局 `brush.stage1/stage2` 恒为空（那一局 61 帧全空）。
#
# 换成「左侧图标列的亮度分带」：正例（8 张 1673x878 实拍 + 2 张 1607x822 新截图）
# 一律恰好 2 条带、高度 0.24~0.40、第二条顶边 0.44~0.56、缝 0.01~0.17；
# 负例（背包页 / 待开画 / 结束画面）一条带都切不出来。宽高比与宽度不再参与判断。
BAND_MIN_HEIGHT = 0.15
BAND_MAX_HEIGHT = 0.50
BAND_SECOND_TOP_RANGE = (0.35, 0.75)
BAND_MAX_GAP = 0.20
#: 卡片本体的最大平均亮度。笔触卡片是一块深青色半透明底板（实测 V 92~113），
#: 而背包格/空面板的同一位置是更亮的冰蓝背景（V 130+）。
BAND_BODY_MAX_VALUE = 145.0


def card_bands_ok(crop: np.ndarray) -> bool:
    """这块区域是不是「上下两张宽卡片」的笔触面板。

    只看左侧图标列的亮度分带，不看宽高比：窗口尺寸一变，卡片宽高比就从 4.9
    掉到 3.3，而分带结构完全不变（见 :data:`BAND_MIN_HEIGHT` 的注释）。
    """
    if crop is None or crop.size == 0:
        return False
    height, width = crop.shape[:2]
    if height < 40 or width < 120:
        return False
    bands = gutter_bands(crop)
    if len(bands) != 2:
        return False
    height_fractions = [(bottom - top) / height for top, bottom in bands]
    if not all(BAND_MIN_HEIGHT <= value <= BAND_MAX_HEIGHT for value in height_fractions):
        return False
    second_top = bands[1][0] / height
    if not BAND_SECOND_TOP_RANGE[0] <= second_top <= BAND_SECOND_TOP_RANGE[1]:
        return False
    gap = (bands[1][0] - bands[0][1]) / height
    if gap < -0.02 or gap > BAND_MAX_GAP:
        return False
    value = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
    body_left = int(width * 0.25)
    body_right = int(width * 0.75)
    for top, bottom in bands:
        body = value[top:bottom, body_left:body_right]
        if body.size and float(body.mean()) > BAND_BODY_MAX_VALUE:
            return False
    return True


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

#: 两次面板 OCR 的最小间隔（秒）。面板要发光呼吸，签名每帧都变；按时间节流
#: 才不会每帧起一个线程。0.35s 的滞后对提示来说完全够——比一轮推荐的计算还短。
PANEL_READ_INTERVAL = 0.35
#: 上一次结果最多沿用到这么久（秒）。超过就当作「没有读数」，别拿几秒前的
#: 词缀去解释当前这一笔。必须大于采集泵的间隔：自动跟随实测 ``pump_interval``
#: 是 1.5s，取 1.2s 会让「上一帧刚读到、这一帧刚好过期」反复发生，
#: 读到的面板又变成空——那是 2026-11-20 之前的老毛病。3.0s ≈ 两个泵周期。
PANEL_READ_STALE = 3.0

#: 一笔异步 OCR 最多允许在飞这么久（秒）。超过就当它卡死，重新放行请求。
#:
#: 2026-11-22 实机：OCR 子进程卡住后 ``pending`` 永久为 True，而
#: :func:`read_brush_panel_async` 的放行条件要求 ``not pending``，于是整局再也不
#: 请求 OCR——阶段二永远是空，``affix_card_clicked`` 一条都没有。一次正常读取远
#: 低于 1 秒，给到 6 秒足够宽松。
PANEL_READ_PATIENCE = 6.0

_ASYNC_LOCK = threading.Lock()
_ASYNC_STATE: dict[str, object] = {
    "key": None,
    "reading": None,
    "reading_at": 0.0,
    "requested_at": 0.0,
    "pending": False,
    "thread": None,
    "generation": 0,
    #: 当前占着 ``pending`` 的那笔请求是哪个 generation。worker 回来时靠它判断
    #: 「这是我占的位吗」，只清自己那一笔，不清后来者的。
    "inflight_generation": None,
    #: 上一次成功读取时的 32x12 灰度缩略图，用来判断「面板换了没有」。
    "reading_vector": None,
}


def panel_signature(crop: np.ndarray) -> str:
    small = cv2.resize(crop, (32, 12), interpolation=cv2.INTER_AREA)
    return hashlib.sha1(small.tobytes()).hexdigest()


def panel_signature_vector(crop: np.ndarray) -> np.ndarray:
    """32x12 灰度缩略图（float32，0~255）。

    用来判断「面板到底换没换」：呼吸发光让摘要每帧都变，但**整体分布**几乎不动；
    换面板（两张卡 → 一张卡、阶段一 → 阶段二）才会大幅变化。沿用上次结果前
    先比一下这个，避免把上一笔的「堆砌」当成这一笔的选择写进录制。
    """

    small = cv2.resize(crop, (32, 12), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small.astype(np.float32)


#: 沿用上次结果的相似度上限（32x12 灰度平均绝对差，0~255）。
#: 实测卡片呼吸 3~8；整块换掉（两张卡变一张、或内容换成别的卡）> 25。
PANEL_REUSE_MAX_DISTANCE = 14.0


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
    with _ASYNC_LOCK:
        _ASYNC_STATE.update(
            {
                "key": None,
                "reading": None,
                "reading_at": 0.0,
                "requested_at": 0.0,
                "pending": False,
                "inflight_generation": None,
                "generation": int(_ASYNC_STATE["generation"]) + 1,
                "reading_vector": None,
            }
        )


def wait_for_async(timeout: float = 2.0) -> bool:
    """等后台那次面板 OCR 跑完（测试与「点开面板后立即取一次结果」用）。"""
    with _ASYNC_LOCK:
        thread = _ASYNC_STATE.get("thread")
    if isinstance(thread, threading.Thread):
        thread.join(timeout)
        return not thread.is_alive()
    return True


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
    """异步版：缓存命中立即返回，否则丢后台线程，这一帧返回上一次的结果。

    2026-11-20 重写。原来这里是「面板签名（32x12 缩略图 sha1）命中就复用，不命中就
    起线程」，而笔触卡片一直在呼吸发光：每帧像素都变 → 签名每帧都是新的 →
    ① ``_SIGNATURE_CACHE`` 永远不命中，线程每帧新起一个；② OCR 走的是同一个管道，
    排队越来越长；③ 当帧永远返回 ``None``。实机表现就是**整局笔触面板一条都读不到**
    （同一局「当前颜料」面板是好的，因为那边早就改成了按时间节流 + 沿用上次结果）。

    现在与 ``paint_panel.read_panel_async`` 同口径：
    - 同一张签名命中 → 零成本返回；
    - 面板没换（签名变了但内容还是那两张卡片）→ 沿用最近一次成功结果；
    - 距上次请求超过 :data:`PANEL_READ_INTERVAL` 才起新线程，一次只允许一个在飞。
    """
    if crop is None or crop.size == 0:
        return None
    key = panel_signature(crop)
    vector = panel_signature_vector(crop)
    cached = _cache_get(_SIGNATURE_CACHE, key)
    if cached is not None:
        with _ASYNC_LOCK:
            _remember_async(key, cached)
            _ASYNC_STATE["reading_vector"] = vector
        return cached

    now = time.monotonic()
    with _ASYNC_LOCK:
        if (
            _ASYNC_STATE["reading"] is not None
            and now - float(_ASYNC_STATE["reading_at"]) > PANEL_READ_STALE
        ):
            _ASYNC_STATE["reading"] = None
            _ASYNC_STATE["reading_vector"] = None
        if _ASYNC_STATE["key"] == key and _ASYNC_STATE["reading"] is not None:
            return _ASYNC_STATE["reading"]
        stale = _ASYNC_STATE["reading"]
        previous_vector = _ASYNC_STATE.get("reading_vector")
        if stale is not None and previous_vector is not None:
            # 沿用上一次结果之前先确认「还是同一块面板」。
            # 2026-11-21 实机日志：22:08:12 点了第一张卡（转锋），下一帧却记成
            # 「堆砌」——面板已经换成阶段二，缓存仍把上一笔的 reading 递了上来。
            distance = float(np.mean(np.abs(np.asarray(previous_vector) - vector)))
            if distance > PANEL_REUSE_MAX_DISTANCE:
                # 只把局部变量置空是不够的：``_ASYNC_STATE["reading"]`` 还留着旧读数，
                # 下一次 ``_ASYNC_STATE["key"] == key`` 那条分支又会把它递出来。
                stale = None
                _ASYNC_STATE["reading"] = None
                _ASYNC_STATE["reading_vector"] = None
            else:
                _ASYNC_STATE["reading_vector"] = vector
        # 看门狗：OCR 子进程卡死时 ``pending`` 会一直挂着，而放行条件要求
        # ``not pending`` —— 结果就是**整局再也不请求 OCR**，阶段二永远为空。
        # 2026-11-22 实机就是读到 17:21 之后彻底停摆。超过
        # :data:`PANEL_READ_PATIENCE` 秒就当那笔已经死了，重新放行。
        if _ASYNC_STATE["pending"] and (
            now - float(_ASYNC_STATE["requested_at"]) > PANEL_READ_PATIENCE
        ):
            _ASYNC_STATE["pending"] = False
            _ASYNC_STATE["inflight_generation"] = None
        should_request = not _ASYNC_STATE["pending"] and (
            now - float(_ASYNC_STATE["requested_at"]) >= PANEL_READ_INTERVAL
        )
        if should_request:
            _ASYNC_STATE["pending"] = True
            _ASYNC_STATE["requested_at"] = now
            _ASYNC_STATE["key"] = key
            _ASYNC_STATE["inflight_generation"] = int(_ASYNC_STATE["generation"])
    if should_request:
        thread = threading.Thread(
            target=_async_read_worker,
            args=(key, crop.copy(), int(_ASYNC_STATE["generation"])),
            name="huahua-brush-panel-ocr",
            daemon=True,
        )
        with _ASYNC_LOCK:
            _ASYNC_STATE["thread"] = thread
        thread.start()
    return stale


def _remember_async(key: str, reading: "BrushPanelReading") -> None:
    _ASYNC_STATE["key"] = key
    _ASYNC_STATE["reading"] = reading
    _ASYNC_STATE["reading_at"] = time.monotonic()


def _async_read_worker(key: str, crop: np.ndarray, generation: int) -> None:
    reading: BrushPanelReading | None = None
    try:
        reading = read_brush_panel(crop)
    finally:
        with _ASYNC_LOCK:
            # ``pending`` 必须能复位，否则**整局再也不请求 OCR**。
            #
            # 2026-11-22 实机：整局读到 17:21 就再也读不到词缀，``brush_options``
            # 之后一条都不产生。原实现只在 generation 相等时复位 ``pending``：
            # worker 在 ``reset_cache()`` 之前拿到旧 generation、之后才回来，两者
            # 不等，``pending`` 就永久停在 True。而放行条件要求 ``not pending``，
            # 于是彻底停摆，阶段二永远是空、``affix_card_clicked`` 一条都没有。
            #
            # 现在按「是不是自己占的位」来判：只清自己那一笔，不清后来者的。
            # 读数是否采用仍然只看 generation，旧面板的结果不会串到新面板上。
            if _ASYNC_STATE.get("inflight_generation") == generation:
                _ASYNC_STATE["pending"] = False
                _ASYNC_STATE["inflight_generation"] = None
            if generation == int(_ASYNC_STATE["generation"]):
                if reading is not None:
                    _remember_async(key, reading)
                    _ASYNC_STATE["reading_vector"] = panel_signature_vector(crop)


