"""S8「画中雪原 · 作画」服务端协议契约（只固化，不决策、不改现有行为）。

为什么不猜
----------
这个模块里的每个枚举值、每个字段名，都来自**服务端 .proto 导出的描述符**
（``FileDescriptorSet``，2026-07-15 抓取），而不是从客户端字节码常量池反推：

    ...\\12345\\_归并_逆向归档\\抓包与协议\\proto_pb_all_20260715\\
        UE_game\\Content\\Script\\Proto\\proto\\gameplays8_data.pb
        UE_game\\Content\\Script\\Proto\\proto\\gameplays8_api.pb

描述符内部头为 ``package rpc;  syntax = "proto3";  源路径 haidao/backend/core/rpcb``，
即服务端那一份。字段序号与类型是权威值，原文可逐条复核。

本模块只做三件事：

1. **枚举固化**：地块类型、玩法状态、笔触来源、变更字段（数值与服务端一致）；
2. **算术关系**：等级公式 ``最终等级 = min(7, 原始等级 + 等级加成)``（7 级写死）、
   资源余量（刷新 / 拆下 / 复活）的换算；
3. **难度画像骨架**：练习模式按**最大难度**取画像，但具体随机数值
   （``BlockMin`` / ``BlockMax`` / ``InitMaxLevel`` 等）**不在客户端 pak 里**，
   只有字段名，所以这里一律留 ``None`` 并标注来源，等真值到手再填。

刻意不做的事
------------
* 不生成棋盘、不抽颜料、不参与推荐 —— 这些仍在 ``game_model`` 里，
  本模块只提供"官方口径"，不改变任何现有行为；
* 不解析 ``UNASSIGNED_WEIGHT_STRINGS`` 那两条难度区间串：它们的语义
  （下界 / 权重 / 上界的排列顺序）在客户端没有任何佐证，写死一个解析器
  等于把猜测固化成代码。

字段对照（服务端 ``gameplays8_data.proto``）::

    S8DrawMapDataAck  plots items remainReroll remainRevive reviveIds
                      fixedEffectLibTypes effectLibTypes effectIds willReviveId
                      maxPlotLevel maxReviveTimes usedUninstall maxRerollTimes
                      enableUninstall
    S8Plot            no rawPlotType items effectFroms facadeLevel facadeItemId
                      facadePlotType
    S8Item            uniqueId itemId rawLevel installSort task maxLvSort
    S8Effect          effectId selectableEffectType selectableEffectId
                      selectableEffectParams selectableFromOptIds
    S8DrawPushChangeNotify  changePlots changeItems removeItems + 与 ack 同名的资源字段
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

#: 颜料最高 **7** 级，写死（用户 2026-11-22 确认）。
#:
#: 服务端 ack 里确实有 ``maxPlotLevel`` 字段，但它是**冗余信息**：实机恒为 7，
#: 而技艺「色彩研究」改的是"获得颜料的最大可能**初始**等级"，不动 7 级封顶。
#: 所以这里不给调用方留"调上限"的口子，封顶一律按这个常量算。
MAX_PLOT_LEVEL = 7

#: 难度表里的教学档（``config_s8_gameplay_difficulty`` 记录 101「无色雪纸」）。
TEACHING_DIFFICULTY = 101


# ---------------------------------------------------------------------------
# 枚举：数值＝服务端描述符里的值，逐条照抄
# ---------------------------------------------------------------------------


class S8PlotType(IntEnum):
    """地块类型（服务端 ``enum S8PlotType``）。

    ``2001`` 与 ``3001`` 是**两个不同的东西**（"2001↔3001 对应关系未确认"
    这个悬案到此关闭）：

    * ``S8_Free``            2001  空白地块：可放任意颜料
    * ``S8_Nihility``        2002  虚空地块：不能放，可被转成空白
    * ``S8_Ruins``           2003  失色地块：被摧毁过的地块
    * ``S8_Internal_Block``  3001  内部阻挡
    * ``S8_Border_Block``    3002  边界阻挡
    """

    NONE = 0
    FREE = 2001
    NIHILITY = 2002
    RUINS = 2003
    INTERNAL_BLOCK = 3001
    BORDER_BLOCK = 3002

    @property
    def is_placeable(self) -> bool:
        """只有空白地块可以直接落笔（其余要靠笔触转换）。"""
        return self is S8PlotType.FREE

    @property
    def is_blocking(self) -> bool:
        """两种阻挡（内部 / 边界）都算阻挡。"""
        return self in (S8PlotType.INTERNAL_BLOCK, S8PlotType.BORDER_BLOCK)


class S8GameState(IntEnum):
    """玩法状态机（服务端 ``enum S8GameState``）。"""

    NONE = 0
    NOT_OPEN = 1
    OPEN = 2
    TILE_DROP = 3
    TILE_DROP_DONE = 4
    DRAW_TILE = 5
    BATTLE = 6
    BATTLE_DONE = 7
    REWARD = 8


class S8EffectFromType(IntEnum):
    """笔触来源（服务端 ``enum S8EffectFromType``）。"""

    NONE = 0
    ITEM = 1          # 颜料自带的笔触
    SELECTABLE = 2    # 面板里选的笔触
    REVIVE = 3        # 复活/强制调色带来的笔触


class S8SelectableLibType(IntEnum):
    """可选笔触库类型（服务端 ``enum S8SelectableLibType``）。

    对应 ``S8DrawSelectEffectLibReq.selectLibType``：客户端**向服务器要候选**，
    不是在本地 59 条表里抽。
    """

    NONE = 0
    TRANSFORM = 1     # 转锋
    APPEND = 2        # 堆砌
    EXPAND = 3        # 扩散


class S8ChangedField(IntEnum):
    """增量推送里"哪一块变了"（服务端 ``enum S8ChangedField``）。"""

    INVALID = 0
    CURRENT_STATE = 1
    TILE_COLLECT = 2
    DRAW_MAP = 3
    OPEN_MAP = 4


# ---------------------------------------------------------------------------
# 等级公式：rawLevel + facadeLevel，按 7 封顶（写死）
# ---------------------------------------------------------------------------


def final_level(raw_level: int, facade_level: int) -> int:
    """``最终等级 = min(7, 原始等级 + 等级加成)``——**7 级写死**。

    官方字段语义（服务端 pb）：

    * ``S8Item.rawLevel``     —— 颜料自身等级（"等级"）
    * ``S8Plot.facadeLevel``  —— 挂在**地块**上的加成（"等级加成"）

    两层相加后统一按 :data:`MAX_PLOT_LEVEL` 封顶，且都不允许负数。
    空地块（``raw_level = 0``）返回 0。

    实机验证过的两条极端路径都落在 7：

    * ``Lv2 颜料 + 调色效果 +2 + 笔触额外提升 +3`` → 7；
    * ``Lv1 颜料放在已叠 +6 加成的格子上`` → 7。
    """
    raw = max(0, int(raw_level))
    if raw <= 0:
        return 0
    bonus = max(0, int(facade_level))
    return min(MAX_PLOT_LEVEL, raw + bonus)


def plot_item_level(item: "S8Item", plot: "S8Plot") -> int:
    """一格上某枚颜料的最终等级（该格的 ``facadeLevel`` 是共享的）。"""
    return final_level(item.raw_level, plot.facade_level)


# ---------------------------------------------------------------------------
# 资源：刷新（reroll）/ 拆下（uninstall）/ 复活（revive）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class S8Resources:
    """一局里的三种资源余量，字段名与 ack 完全一致。

    * ``remain_reroll`` / ``max_reroll``       —— 刷新笔触候选（``S8DrawRerollEffectId``）
    * ``remain_revive`` / ``max_revive``      —— 复活 / 强制调色（``S8DrawRevive``）
    * ``used_uninstall`` / ``enable_uninstall`` —— 拆下（``S8DrawUninstallAll``）

    "刷新的具体次数由宠物等级给"是本项目此前的实机结论；协议层的真值就是
    ``maxRerollTimes``，两者不冲突：宠物等级只是决定服务器给多少。
    """

    remain_reroll: int = 0
    max_reroll: int = 0
    remain_revive: int = 0
    max_revive: int = 0
    used_uninstall: int = 0
    enable_uninstall: bool = False
    revive_ids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def reroll_left(self) -> int:
        return max(0, self.remain_reroll)

    @property
    def reroll_used(self) -> int:
        """已用刷新次数＝上限 − 剩余（负数钳到 0）。"""
        return max(0, self.max_reroll - self.remain_reroll)

    @property
    def revive_left(self) -> int:
        return max(0, self.remain_revive)

    @property
    def revive_used(self) -> int:
        return max(0, self.max_revive - self.remain_revive)

    @property
    def can_reroll(self) -> bool:
        return self.reroll_left > 0

    @property
    def can_revive(self) -> bool:
        return self.revive_left > 0

    @property
    def can_uninstall(self) -> bool:
        return bool(self.enable_uninstall)


def resources_of(message: object) -> S8Resources:
    """从 ack / 增量推送里取资源。

    ``S8DrawMapDataAck`` 与 ``S8DrawPushChangeNotify`` 用的是同一套字段名，
    所以这里按属性名鸭子取值，缺字段就取默认值（推送里可以只带变化项）。
    """

    def _int(name: str) -> int:
        return max(0, int(getattr(message, name, 0) or 0))

    return S8Resources(
        remain_reroll=_int("remain_reroll"),
        max_reroll=_int("max_reroll_times"),
        remain_revive=_int("remain_revive"),
        max_revive=_int("max_revive_times"),
        used_uninstall=_int("used_uninstall"),
        enable_uninstall=bool(getattr(message, "enable_uninstall", False)),
        revive_ids=tuple(int(value) for value in getattr(message, "revive_ids", ()) or ()),
    )


# ---------------------------------------------------------------------------
# 消息结构（只用于解析/断言，不参与决策）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class S8Effect:
    effect_id: int = 0
    selectable_effect_type: int = 0
    selectable_effect_id: int = 0
    selectable_effect_params: str = ""
    selectable_from_opt_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class S8Task:
    effect: S8Effect | None = None
    raw_need_progress: int = 0
    need_progress: int = 0
    cur_progress: int = 0
    is_finish: bool = False


@dataclass(frozen=True)
class S8Item:
    """一枚颜料（``S8Item``）。"""

    unique_id: str = ""
    item_id: int = 0
    raw_level: int = 0
    install_sort: int = 0
    task: S8Task | None = None
    max_lv_sort: int = 0

    @property
    def brush_effect_id(self) -> int:
        """这枚颜料身上的笔触 ID（``task.effect.selectableEffectId``，没有就是 0）。"""
        if self.task is None or self.task.effect is None:
            return 0
        return int(self.task.effect.selectable_effect_id)


@dataclass(frozen=True)
class S8EffectFrom:
    effect_id: int = 0
    item_unique_id: str = ""
    effect_from_type: int = 0


@dataclass(frozen=True)
class S8Plot:
    """一格地块（``S8Plot``）。``no`` 是服务器给的格号。"""

    no: int = 0
    raw_plot_type: int = 0
    items: tuple[S8Item, ...] = ()
    effect_froms: tuple[S8EffectFrom, ...] = ()
    facade_level: int = 0
    facade_item_id: int = 0
    facade_plot_type: int = 0

    @property
    def plot_type(self) -> S8PlotType:
        try:
            return S8PlotType(int(self.raw_plot_type))
        except ValueError:
            return S8PlotType.NONE

    @property
    def has_paint(self) -> bool:
        return bool(self.items)

    def item_levels(self) -> tuple[int, ...]:
        return tuple(plot_item_level(item, self) for item in self.items)


@dataclass(frozen=True)
class S8DrawMapData:
    """``S8DrawMapDataAck``（整盘数据）。

    **棋盘格数 = ``len(plots)``**：客户端没有固定 36 格表（``S8SnowField_``
    是动态建格），所以尺寸只能从服务器来。
    """

    plots: tuple[S8Plot, ...] = ()
    items: tuple[S8Item, ...] = ()
    remain_reroll: int = 0
    remain_revive: int = 0
    revive_ids: tuple[int, ...] = ()
    fixed_effect_lib_types: tuple[int, ...] = ()
    effect_lib_types: tuple[int, ...] = ()
    effect_ids: tuple[int, ...] = ()
    will_revive_id: int = 0
    #: 服务端字段，仅保留 schema 形状；封顶一律用 :data:`MAX_PLOT_LEVEL`（恒 7）。
    max_plot_level: int = MAX_PLOT_LEVEL
    max_revive_times: int = 0
    used_uninstall: int = 0
    max_reroll_times: int = 0
    enable_uninstall: bool = False

    @property
    def board_size(self) -> int:
        """地块总数（服务器下发的格数）。"""
        return len(self.plots)

    @property
    def resources(self) -> S8Resources:
        return resources_of(self)

    def plot(self, no: int) -> S8Plot | None:
        for plot in self.plots:
            if plot.no == no:
                return plot
        return None

    def initial_paint(self) -> S8Item | None:
        """初始颜料＝开局就落在盘上的那枚。

        服务器把它随 ``plots[].items[]`` 一起下发（客户端拿不到"哪一枚是初始"
        的标记位，实机左上角灰白感叹号是**视图**画的），所以这里只能给
        "第一个带颜料的格子里的第一枚"，调用方要自己判断可信度。
        """
        for plot in self.plots:
            if plot.items:
                return plot.items[0]
        return None


# ---------------------------------------------------------------------------
# 笔触 ID：本项目与官方一致，用 10xx / 20xx / 30xx 表示家族
# ---------------------------------------------------------------------------

#: 官方 59 条可选笔触的家族前缀（``S8_Gameplay_SelectableEffect`` 记录 ID 千位）：
#: 1xxx 转锋 / 2xxx 堆砌 / 3xxx 扩散；4xxx 是"权重变体"，家族由记录本身决定。
AFFIX_FAMILY_PREFIX: dict[int, str] = {
    1: "转锋",
    2: "堆砌",
    3: "扩散",
}

#: 4xxx 变体里既有转锋也有堆砌（4001~4007 与 1xxx/2xxx 语义重复），
#: 所以 4 段不能按前缀硬判，必须查官方表。
_AFFIX_VARIANT_PREFIX = 4


def affix_family(effect_id: int) -> str | None:
    """笔触家族：优先查官方表（含 4xxx 变体），查不到再退回千位前缀。"""
    from .s8_official_data import S8_SELECTABLE_EFFECTS

    for record in S8_SELECTABLE_EFFECTS:
        if record.id == int(effect_id):
            return record.library_name
    prefix = int(effect_id) // 1000
    if prefix == _AFFIX_VARIANT_PREFIX:
        return None
    return AFFIX_FAMILY_PREFIX.get(prefix)


def affix_lib_type(effect_id: int) -> S8SelectableLibType:
    """笔触家族 → 服务端 ``S8SelectableLibType``（要候选时用的就是它）。"""
    family = affix_family(effect_id)
    return {
        "转锋": S8SelectableLibType.TRANSFORM,
        "堆砌": S8SelectableLibType.APPEND,
        "扩散": S8SelectableLibType.EXPAND,
    }.get(family or "", S8SelectableLibType.NONE)


# ---------------------------------------------------------------------------
# 难度画像（练习模式：按最大难度）
# ---------------------------------------------------------------------------

#: 难度表里**语义未确认**的两条区间权重串（原样保留，不做解释）：
#:
#:     5:1|9;15:10|19;25:20|29;35:30|39;45:40|49;55:50|1000
#:     5:1|9;15:10|19;25:20|29;35:30|39;45:40|49;55:50;59:65:60|69;75:70|100
#:
#: 它们出现在 ``config_s8_gameplay_difficulty`` 的常量池里，但**归属字段不明**
#: （去重 + 顺序错位）。不要写解析器，等抓包真值。
UNASSIGNED_WEIGHT_STRINGS: tuple[str, ...] = (
    "5:1|9;15:10|19;25:20|29;35:30|39;45:40|49;55:50|1000",
    "5:1|9;15:10|19;25:20|29;35:30|39;45:40|49;55:50;59:65:60|69;75:70|100",
)

#: 练习模式默认难度：**最大难度**（用户 2026-11-22 指定）。
#: 7 = 绮彩雪纸（战斗难度等同于异界深空，掉落 +200%）。
PRACTICE_DIFFICULTY = 7


@dataclass(frozen=True)
class DifficultyProfile:
    """一个难度下、服务端会下发的随机参数。

    除 ``difficulty`` / ``name`` 外**全部可能为 ``None``**：``BlockMin`` /
    ``BlockMax`` / ``InitMaxLevel`` 在客户端只有字段名没有值（``__index``
    元表取，即服务端下发）。``None`` 表示"未知"，**不是 0**。
    """

    difficulty: int
    name: str
    block_min: int | None = None
    block_max: int | None = None
    init_max_level: int | None = None
    plot_item_source: int | None = None
    available_dir: tuple[int, ...] = ()
    uninstall_cost: tuple[int, ...] = ()
    fixed_effect_lib_cost: tuple[int, ...] = ()
    plot_interval_weights: tuple[str, ...] = UNASSIGNED_WEIGHT_STRINGS
    source: str = "server（客户端 pak 只有字段名，值未取到）"

    @property
    def block_range_known(self) -> bool:
        return self.block_min is not None and self.block_max is not None


def difficulty_profile(difficulty: int = PRACTICE_DIFFICULTY) -> DifficultyProfile:
    """按难度取画像；名字来自官方多语言表，其余留空等真值。"""
    from .s8_official_data import S8_DIFFICULTIES

    for record_id, name, _battle, _unlock in S8_DIFFICULTIES:
        if int(record_id) == int(difficulty):
            return DifficultyProfile(difficulty=int(record_id), name=name)
    raise KeyError(f"未知难度 {difficulty!r}")


#: 练习模式用的那份画像（最大难度）。
MAX_DIFFICULTY_PROFILE = difficulty_profile(PRACTICE_DIFFICULTY)
