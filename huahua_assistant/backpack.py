"""背包颜料：一屏读不全，落一枚之后再识别一次才读全。

实机背包是 3 行 × 5 列 = 15 格，容量 16，所以单帧一定有一个被挤到屏外
（面板右下角显示 `1/16`）。落一枚颜料后整列前移，第二帧就能看到它——
这就是"识别 2 次即可把背包的颜料识别完全"。

由此定下三条规则：

1. **以最新一帧为准**，不跨帧累加。累加会算错：背包里本来就常有 3 枚一样的颜料。
2. **按枚数保留**，不按种类去重，也不按槽位去重。槽位会移动，同种颜料会有多枚。
3. **没有落笔的那一帧如果少了颜料，按上一帧补回来**——那是单帧漏识别，不是被用掉了。
   落笔之后的帧才是新的真相。

背包还按当前初始颜料排序（需求怪物就把怪物排前面，需求玩法就排玩法），
``matching`` 用来数"这一手能用的颜料有几个"。

## 排序还能反过来校验识别

同样的排序规则（用户 2026-11-16 确认：**目标类别优先，然后等级从高到低**）
可以拿来验一帧读数对不对：类别最多切换一次、每段内等级单调不增、第一格必须
能用于当前目标。违反这三条的位置就是可疑格，见 :func:`order_violations`。
只报可疑、不改值——识别是概率判断，排序是硬约束，冲突时该由用户看大图定夺。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import game_model as gm

# 一屏能显示的槽位数（3 行 × 5 列）。实机容量是 16，所以满屏即可能被截断。
PAGE_CAPACITY = 15


def paint_signature(paint: gm.Paint) -> str:
    """同种颜料的判据：类别 + 等级 + 子类型 + 是否宝箱。"""
    return f"{paint.category.value}|{int(paint.level)}|{paint.subtype or ''}|{int(bool(paint.is_treasure))}"


def paint_from_observation(observed: Any) -> gm.Paint:
    """把视觉层的观测对象转成 ``Paint``（不依赖 vision 的导入）。"""
    level = getattr(observed, "level", 1)
    try:
        level = int(level)
    except (TypeError, ValueError):
        level = 1
    return gm.Paint(
        category=getattr(observed, "category", gm.PaintCategory.UNKNOWN),
        level=max(1, min(gm.MAX_LEVEL, level)),
        subtype=getattr(observed, "subtype", None),
        is_treasure=bool(getattr(observed, "is_treasure", False)),
    )


def _confidence_of(observed: Any) -> float:
    category = float(getattr(observed, "category_confidence", 0.5) or 0.5)
    level = float(getattr(observed, "level_confidence", 0.5) or 0.5)
    return category * 0.55 + level * 0.45


def order_violations(
    paints: Sequence[gm.Paint],
    target: gm.PaintObject | None = None,
) -> tuple[int, ...]:
    """按背包排序规则检查一帧读数，返回「疑似认错」的槽位序号（1 基）。

    规则（实机确认过）：背包按**当前初始颜料的目标**排序——目标是玩法就把玩法
    颜料排在前面、怪物排在后面，然后**等级从高到低**。所以一帧里：

    * 类别最多切换一次（前段就是目标类别，后段是另一类）；
    * 每段内等级单调不增；
    * 给了 ``target`` 时，第一格必须能用于这个目标。

    违反这三条的位置就是可疑格，通常是家族或等级认错了。

    **只报可疑、不改值**：识别是概率判断，排序是硬约束，两者冲突时该由用户看
    大图定夺。悄悄按排序"纠正"识别结果会造出一个看起来自洽、实则伪造的局面。
    """
    flags: list[int] = []
    if not paints:
        return ()
    if target is not None and not gm._paint_matches_target(paints[0], target):
        flags.append(1)
    seen: list[str] = []
    previous: gm.Paint | None = None
    for index, paint in enumerate(paints, start=1):
        category = paint.category.value
        if not seen or seen[-1] != category:
            if category in seen:
                # 类别反复切换：A → B → A，第二个 A 以前的那格大概率认错了。
                flags.append(index)
            else:
                seen.append(category)
        if (
            previous is not None
            and previous.category == paint.category
            and int(paint.level) > int(previous.level)
        ):
            # 同类别内等级该降不该升。
            flags.append(index)
        previous = paint
    return tuple(sorted(set(flags)))


@dataclass
class BackpackEntry:
    paint: gm.Paint
    count: int
    confidence: float
    observed: Any = None
    #: 这个读数连续出现了几帧。翻页时只把「稳住过」的读数并进历史，
    #: 单帧抖动（2↔3 那种）不会污染跨页累计。
    hits: int = 1
    #: 这一格换过读数（而且还没连续稳定 ≥2 帧）。翻页时不收录。
    unstable: bool = False


def _observed_slot_key(observed: Any) -> tuple[int, int, int, int] | None:
    """观测里的格子坐标，拿来做「同一格」的键。视觉层不提供就不参与定位跟踪。"""
    rect = getattr(observed, "slot_rect", None)
    if rect is None:
        return None
    try:
        return (int(rect.x), int(rect.y), int(rect.w), int(rect.h))
    except AttributeError:
        try:
            x, y, w, h = rect
        except (TypeError, ValueError):
            return None
        return (int(x), int(y), int(w), int(h))


#: 坐标对得上、但**内容**（类别 + 等级 + 宝箱，不含家族）变了超过这个比例的格子，
#: 当成「翻页」。翻页时整页内容会同时换掉，而读数抖动只会让一两格变。
PAGE_TURN_RATIO = 0.6


def _content_key(paint: gm.Paint) -> tuple[str, int, bool]:
    """格子内容的粗键：类别 + 等级 + 是否宝箱。

    **故意不含家族**：家族是后台线程后补的（先 None、后 ``矮人``），
    把那一步当"内容变了"会让首帧整页被当成翻页。
    """
    return (paint.category.value, int(paint.level), bool(paint.is_treasure))


@dataclass
class BackpackAccumulator:
    """背包颜料的当前状态。会话（= 一条规则）内持续跟踪。

    **按格子位置跟踪，不按「颜料签名 + 枚数」累加。** 这是 2026-11-17 的实机教训：

    旧实现把每一帧的 ``(类别, 等级, 子类型, 宝箱)`` 当键累加，带来两个必现的膨胀：

    * 家族识别是后台上线程跑的：第一帧 ``subtype=None``、第二帧认出 ``矮人``，
      同一个颜料就成了两个签名 → 15 格显示成 30 格。
    * 等级读在 2 / 3 之间抖一下，就凭空多出「矮人 Lv3」这种不存在的颜料。

    实机验证：3 格会从 3 → 6 → 9 这样长。用户看到的正是「多出两枚」和
    「拖动时冒出一大堆」。现在一帧里**每个格子坐标只认一枚**：读数变了就是这格
    换了颜料（或认错了），旧读数当场作废，绝不留两份。

    **按「排序目标」分页记录，每页 15 格，合计最多 30 格。** 2026-11-18 用户定调：

    * 背包按**当前初始颜料的目标**排序——怪物初始颜料就怪物在前，玩法初始颜料就
      玩法在前。所以一页 15 格只能看到「目标类别那一批」。
    * 目标从怪物换成玩法时，界面会重排，**再记一页 15 格玩法颜料**。
    * 两页各自保留、合计最多 30 格，**允许重复**（同一枚颜料两页都在是正常的）。
    * 同一页内翻页/拖动不累计：那是同一批颜料换了位置，多记就会凭空翻倍
      （实机报过「多出两枚」「拖动时冒出一大堆」「几个 5 级机械之神」）。

    没有 ``slot_rect`` 的调用方（离线脚本、单元测试）仍走原来的签名累加路径。
    """

    session_key: str = ""
    page_capacity: int = PAGE_CAPACITY
    #: 当前页的排序目标（``PaintObject.value``，如 ``monster`` / ``gameplay``）。
    page_key: str = ""
    _entries: dict[str, BackpackEntry] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _slots: dict[tuple[int, int, int, int], BackpackEntry] = field(default_factory=dict)
    #: 每个排序目标各一页 15 格。``_slots`` 永远是其中某一页的**同一个字典对象**，
    #: 所以原地改动会自动同步到这张表里。
    _pages: dict[str, dict[tuple[int, int, int, int], BackpackEntry]] = field(default_factory=dict)
    last_visible: int = 0

    def reset(self, session_key: str = "") -> None:
        self.session_key = session_key
        self.page_key = ""
        self._entries = {}
        self._order = []
        self._slots = {}
        self._pages = {}
        self.last_visible = 0

    def _select_page(self, page_key: str) -> None:
        """切到某个排序目标页。

        关键：``_slots`` 必须指向 ``_pages`` 里**同一个字典对象**，后续的原地
        改动（补格、替换读数）才会自动同步进分页表。重建字典会让旧页停在快照上。
        """
        page_key = str(page_key or "").strip()
        if page_key in ("", "unknown"):
            # 目标还没读出来（选色面板刚弹出、颜料认不出）：这不是一页，沿用当前页。
            # 放进来的话快照会多出一页 15 格空读数（实机 2026-11-18 抓到过）。
            return
        if page_key == self.page_key and self._slots is self._pages.get(page_key):
            return
        if self.page_key:
            self._pages[self.page_key] = self._slots
        self.page_key = page_key
        self._slots = self._pages.setdefault(page_key, {})

    # ---- 合并 --------------------------------------------------------
    def update(
        self,
        observed_paints: Iterable[Any],
        session_key: str = "",
        *,
        consumed: bool = False,
        page_key: str | None = None,
    ) -> tuple[gm.Paint, ...]:
        """合并一帧观测。

        ``consumed`` 表示上一帧之后真的落过一枚颜料（背包少了一枚）。
        此时消失的格子就真的空了；否则按上一帧补回来（单帧漏识别）。

        ``page_key`` 是这一帧的**排序目标**（怪物/玩法）。换了目标就是换了一页：
        旧页原样保留，新页从空开始记。同一页内的翻页/拖动仍然只保留最新一页。
        """
        if session_key != self.session_key:
            self.reset(session_key)
        if page_key:
            self._select_page(str(page_key))
        items = list(observed_paints)
        self.last_visible = len(items)

        frame: dict[tuple[int, int, int, int], BackpackEntry] = {}
        legacy: dict[str, tuple[gm.Paint, int, float, Any]] = {}
        for observed in items:
            paint = paint_from_observation(observed)
            confidence = _confidence_of(observed)
            slot_key = _observed_slot_key(observed)
            if slot_key is None:
                signature = paint_signature(paint)
                current = legacy.get(signature)
                if current is None:
                    legacy[signature] = (paint, 1, confidence, observed)
                    continue
                paint_kept, count, best, best_observed = current
                legacy[signature] = (
                    paint if confidence >= best else paint_kept,
                    count + 1,
                    max(best, confidence),
                    observed if confidence >= best else best_observed,
                )
                continue
            current = frame.get(slot_key)
            if current is None or confidence >= current.confidence:
                frame[slot_key] = BackpackEntry(
                    paint=paint, count=1, confidence=confidence, observed=observed
                )

        if frame:
            self._merge_slots(frame, consumed=consumed)
        if legacy:
            self._merge_legacy(legacy, consumed=consumed)

        # 位置模式与签名模式不会同时使用；真的混用时以位置为准，
        # 免得同一枚颜料在两边各算一次。
        return self.paints()

    def _merge_slots(
        self,
        frame: dict[tuple[int, int, int, int], BackpackEntry],
        *,
        consumed: bool,
    ) -> None:
        old_keys = set(self._slots)
        new_keys = set(frame)
        turned = False
        if old_keys and new_keys:
            common = old_keys & new_keys
            if common:
                changed = sum(
                    1
                    for key in common
                    if _content_key(self._slots[key].paint) != _content_key(frame[key].paint)
                )
                turned = changed / len(common) >= PAGE_TURN_RATIO
            elif len(old_keys) >= 3:
                turned = True
        if turned:
            # 同一页内整页换血：这是翻页/拖动，不是抖动。**整页替换**，旧读数丢掉——
            # 同一页里再往后翻也只是换位置，累计就会凭空翻倍（实机报过
            # 「多出两枚」「拖动时冒出一大堆」「几个 5 级机械之神」）。
            # 注意必须原地改：``_slots`` 是 ``_pages`` 里那一页的字典对象，
            # 重建字典会让分页表停在旧内容上。
            self._slots.clear()
            self._slots.update(frame)
            self._entries = {}
            self._order = []
            return
        for key in old_keys - new_keys:
            if consumed:
                self._slots.pop(key, None)
            # 没落笔却消失了 = 单帧漏识别，保留上一帧读数
        for key, entry in frame.items():
            previous = self._slots.get(key)
            if previous is not None and _content_key(previous.paint) == _content_key(entry.paint):
                entry.hits = previous.hits + 1
                entry.unstable = previous.unstable and entry.hits < 2
            else:
                entry.hits = 1
                # 首帧没有「上一帧」，不算抖动；之后换过读数先标不稳定，
                # 连续稳定两帧再销案。
                entry.unstable = previous is not None
            self._slots[key] = entry
        self._entries = {}
        self._order = []

    def _merge_legacy(
        self,
        frame: dict[str, tuple[gm.Paint, int, float, Any]],
        *,
        consumed: bool,
    ) -> None:
        if not consumed:
            for signature, entry in self._entries.items():
                existing = frame.get(signature)
                if existing is None:
                    frame[signature] = (entry.paint, entry.count, entry.confidence, entry.observed)
                elif existing[1] < entry.count:
                    frame[signature] = (existing[0], entry.count, existing[2], existing[3])
        self._entries = {
            signature: BackpackEntry(paint=paint, count=count, confidence=confidence, observed=observed)
            for signature, (paint, count, confidence, observed) in frame.items()
        }
        self._order = list(frame.keys())

    # ---- 读取 --------------------------------------------------------
    def _ordered_slots(self) -> list[BackpackEntry]:
        return [
            entry for _, entry in sorted(self._slots.items(), key=lambda item: (item[0][1], item[0][0]))
        ]

    # ---- 分页读取 ----------------------------------------------------
    #: 排序目标页的固定展示顺序：怪物页在前、玩法页在后，未知兜底。
    PAGE_ORDER = ("monster", "gameplay")

    def page_keys(self) -> tuple[str, ...]:
        """已经记过的排序目标页，按固定顺序（怪物 → 玩法 → 其它）。"""
        keys = [key for key in self.PAGE_ORDER if key in self._pages]
        extra = [key for key in self._pages if key not in self.PAGE_ORDER]
        return tuple(keys + sorted(extra))

    def page_entries(self, page_key: str) -> tuple[BackpackEntry, ...]:
        """某一页的 15 格（按屏幕位置：先上后下、先左后右）。"""
        slots = self._pages.get(page_key) or {}
        return tuple(entry for _, entry in sorted(slots.items(), key=lambda item: (item[0][1], item[0][0])))

    def pages(self) -> tuple[tuple[str, tuple[BackpackEntry, ...]], ...]:
        """所有排序目标页：``(page_key, 15 格)``。两页合计最多 30 格，允许重复。"""
        return tuple((key, self.page_entries(key)) for key in self.page_keys())

    def all_paints(self) -> tuple[gm.Paint, ...]:
        """两页合起来的颜料（怪物目标页 + 玩法目标页，最多 30 枚）。"""
        return tuple(entry.paint for _, entries in self.pages() for entry in entries)

    def paints(self) -> tuple[gm.Paint, ...]:
        """**当前页**的颜料——落子/推荐用它，索引不能跨页错位。"""
        if self._slots:
            return tuple(entry.paint for entry in self._ordered_slots())
        paints: list[gm.Paint] = []
        for signature in self._order:
            entry = self._entries[signature]
            paints.extend(entry.paint for _ in range(max(1, entry.count)))
        return tuple(paints)

    def entries(self) -> tuple[BackpackEntry, ...]:
        if self._slots:
            return tuple(self._ordered_slots())
        return tuple(self._entries[signature] for signature in self._order)

    @property
    def total(self) -> int:
        """当前页的枚数（重复的算多枚）。"""
        if self._slots:
            return len(self._slots)
        return sum(max(1, entry.count) for entry in self._entries.values())

    @property
    def all_total(self) -> int:
        """所有排序目标页合计的枚数（怪物页 + 玩法页，最多 30）。"""
        return sum(len(entries) for _, entries in self.pages())

    @property
    def distinct(self) -> int:
        if self._slots:
            return len({paint_signature(entry.paint) for entry in self._slots.values()})
        return len(self._entries)

    @property
    def visible_total(self) -> int:
        """当前这一页真正看到的枚数（不含历史页补的那部分）。"""
        if self._slots:
            return len(self._slots)
        return self.total

    @property
    def may_be_truncated(self) -> bool:
        """上一帧顶满了整屏。顶满说明可能还有一格被挤在屏外，值得再识别一次。"""
        return self.last_visible >= self.page_capacity

    def matching(self, target: gm.PaintObject | None) -> tuple[gm.Paint, ...]:
        """能用于当前目标的颜料。背包按目标排序，所以这些就是排在前面那批。"""
        if target is None:
            return self.paints()
        return tuple(paint for paint in self.paints() if gm._paint_matches_target(paint, target))


#: 背包快照的落盘结构版本。复现脚本按这个字段挑解析分支。
#: v2 起按「排序目标」分页：怪物页 15 格 + 玩法页 15 格，合计最多 30 格（允许重复）。
SNAPSHOT_SCHEMA = "huahua-backpack-snapshot-v2"


def _slot_payload(index: int, entry: BackpackEntry) -> dict:
    paint = entry.paint
    rect = getattr(entry.observed, "slot_rect", None)
    slot = None
    if rect is not None:
        try:
            slot = [int(rect.x), int(rect.y), int(rect.w), int(rect.h)]
        except AttributeError:
            try:
                slot = [int(value) for value in rect]
            except (TypeError, ValueError):
                slot = None
    return {
        "i": index,
        "slot": slot,
        "category": paint.category.value,
        "level": int(paint.level),
        "subtype": paint.subtype,
        "treasure": bool(paint.is_treasure),
        "confidence": round(float(entry.confidence), 3),
    }


def snapshot_payload(
    pages: Sequence[tuple[str, Sequence[BackpackEntry]]],
    *,
    session: str = "",
    target: str | None = None,
    page_capacity: int = PAGE_CAPACITY,
    now: float | None = None,
) -> dict:
    """把背包读数整理成可落盘的快照。

    用户要求（2026-11-18）：**背包按初始颜料的目标排序，所以要记两页**——
    怪物初始颜料时记 3 排 15 格怪物排序的，等玩法初始颜料出现再记 3 排 15 格
    玩法排序的，两页合计最多 30 格，**允许重复**（同一枚颜料两页都在是正常的）。
    每格带家族、类别、等级、宝箱和格子坐标，这局打完后复现时据此还原背包。

    ``pages`` 传 :meth:`BackpackAccumulator.pages` 的结果。
    """
    import time as _time

    page_payloads: list[dict] = []
    for page_key, entries in pages:
        slots = [_slot_payload(index, entry) for index, entry in enumerate(entries)]
        page_payloads.append(
            {
                "target": page_key,
                "visible_total": len(slots),
                "may_be_truncated": len(slots) >= page_capacity,
                "slots": slots,
            }
        )
    return {
        "schema": SNAPSHOT_SCHEMA,
        "ts": float(now if now is not None else _time.time()),
        "session": session,
        "current_target": target,
        "page_capacity": int(page_capacity),
        "page_count": len(page_payloads),
        "total_slots": sum(page["visible_total"] for page in page_payloads),
        "pages": page_payloads,
    }
