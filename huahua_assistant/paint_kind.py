"""颜料「种类（家族）」图案识别。

棋盘上每一枚颜料的中央都画着**所属家族的图案**（鼠人、矮人、龙骨、邪影……），
而外圈菱形框的颜色只表示等级。两者完全正交：同一张「白发女王」会同时出现在
Lv1 银框、Lv2 绿框、Lv4 紫框里。

解包配置也印证了这一点——``config_s8_gameplay_plotitem`` 每条记录都带一个
``Icon`` 字段，值是 ``UI_S8Gameplay_ColorMonster{种类号}_128`` /
``ColorPlay{种类号}_128``：**一个家族一个图标，七个等级共用同一个**。
``S8MapRes`` 才是按等级变的（``Icon_S8map_Yanliaoqiu{等级}``）。
所以种类模板只需按家族建一套，不用按「家族 × 等级」建 182 套。

## 怎么认

不猜颜色语义，走「特征 + 模板匹配」：

1. 取格子中央 **54%** 的方形区域（外框和右下角 ``LvN`` 角标都在外面），
   灰度化后归一化成 44×44 的搜索窗；
2. 再从中央 43% 取 30×30 当模板——模板比搜索窗小一圈，
   让 ``matchTemplate`` 自己找最佳对齐位置，容忍 ±3px 抖动和轻微缩放；
3. 另外算一张 H-S 二维直方图（16×8），补上「颜色分布」这条与形状正交的证据；
4. 分数 = ``NCC × 0.70 + 直方图余弦 × 0.30``。

## 参数是量出来的，不是拍的

``结束--地块全部无空白和入画按钮.png`` 的满盘有 20 枚真颜料、15 个家族，
其中 4 个家族出现了重复样本（白发女王 ×3、蓝冰晶 ×2、紫尖晶 ×2、铜金箱 ×2），
构成 6 组同族对。实测：

===========  ==========  ==========
指标          同族最低    异族最高
===========  ==========  ==========
纯 NCC        0.763       0.708
综合分        0.802       0.547
===========  ==========  ==========

NCC 单独用只有 0.055 的余量，一抖动就翻车；加上直方图后余量 0.255，
阈值取中点 0.68。

## 为什么必须先把「非颜料格」挡掉

同一张满盘图里 16 个深色漩涡地块全是 ``meanV≈82、sat_ratio≈0.00``，
两两相似度能到 1.000——比任何真同族对都高。放任它们进模板库，
整个库会被一团黑影占据。所以 :func:`read_kind` 先过一道
:func:`looks_like_paint_tile`（真颜料一定有明亮的菱形框，整格平均亮度压不到 100）。

还有一类**不是颜料、但长得像颜料**的东西：通关后的「宠物效果」展示图。
``结束--地块全部无空白和入画按钮.png`` 是结束态截图，其中 ``r0c3`` 就是这张
白发尖耳女性人像（紫红渐变底、Lv4 紫框）。它在棋盘进行中不会出现，
所以**不要给它建模板、也不要给它起家族名**——聚类时它会单独成一簇，
命名那一步跳过即可（识别分达不到阈值，不会被误判成某个家族）。

## 名字从哪来

图案和名字的对应**没法从解包资源里读**（美术资源没解出来，只有图标名）。
唯一可靠的自动来源是右侧「当前颜料」面板：那里同时渲染着**家族图标**和
**``怪物颜料·邪影LV3`` 这行字**，OCR 命中 10/11。:func:`learn_from_panel`
就是拿这两样自动配对入库，玩几局就能把 26 个家族攒齐。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

#: 搜索窗：格子中央 54% 的方形。外框（约 0.80~1.02 环带）和右下角 LvN 角标都在外面。
KIND_WINDOW_RATIO = 0.54
#: 搜索窗归一化尺寸。
KIND_WINDOW_SIZE = (44, 44)
#: 模板取景框比例。比搜索窗小一圈，留给 matchTemplate 做平移搜索。
KIND_TEMPLATE_RATIO = 0.43
#: 模板归一化尺寸。
KIND_TEMPLATE_SIZE = (30, 30)
#: 算颜色直方图前的归一化尺寸。
KIND_HIST_SIZE = (48, 48)
#: H-S 二维直方图分箱。
KIND_HIST_BINS = (16, 8)
#: 综合分里 NCC 的权重（剩下给直方图）。
KIND_NCC_WEIGHT = 0.70
#: 判定「认出种类」的阈值。实测同族最低 0.802、异族最高 0.547，取中点偏上。
KIND_MATCH_THRESHOLD = 0.68
#: 先按直方图筛候选，只对前 K 个跑 matchTemplate（36 格 × 全库逐次匹配太慢）。
#:
#: 预筛**只砍同家族内的冗余模板，不砍整个家族**——每个家族至少留一个代表进
#: NCC 评估（见 :meth:`KindTemplateBank.read`）。原因是颜色直方图会把菱形外框
#: 一起算进去，而外框色正是等级的编码：绿=低、黄=中、紫=高。一个家族如果只
#: 存过单一等级的模板，遇到别的等级就会在颜色通道上被误判成「不像」，被预筛
#: 提前淘汰。实测征战之神只存了 Lv2（绿框）一份模板，对黄框的 Lv5 直方图排到
#: 第 31 名，NCC 一律没算；放开评估后它是 0.699（阈值 0.68），亚军才 0.428。
#: 正确结果一直在库里，是预筛把证据丢了。
KIND_PREFILTER_TOP_K = 6
#: 每个家族最多留几个模板。多了每格要多跑 matchTemplate，收益递减。
MAX_TEMPLATES_PER_KIND = 3
#: 两份模板相似度超过这个值就算同一份，不再重复入库。
#:
#: 棋盘上同一家族常同时出现 3 格，harvest 会把这 3 个**同一张图案**的格子
#: 各存一遍。实测「实验体」「机械之神」「邪影」三家各 3 份模板两两 ncc = 1.000，
#: 完全是副本：配额被占满、每格白算 3 倍 NCC，而统计里「3 个模板」看着像
#: 有冗余证据，其实只有 1 份。
TEMPLATE_DUPLICATE_NCC = 0.99
#: 深色地块判据：亮度低于此值且几乎没有饱和像素 → 不是颜料格。
DARK_MEAN_VALUE_MAX = 100.0
DARK_SAT_RATIO_MAX = 0.05

BANK_VERSION = 1
_BANK_DIR = Path(__file__).resolve().parent / "assets" / "paint_kinds"
_BANK_PATH = _BANK_DIR / "templates.npz"
_LABELS_PATH = _BANK_DIR / "labels.json"
_CATALOG_PATH = Path(__file__).resolve().parent / "assets" / "paint_catalog.json"

#: 合成模板表：官方图标 + 官方品质框拼出来的 **27 家族 × 7 等级**（``runtime/paint_sheet``）。
#:
#: 它和自学习库（``templates.npz``，从实机截图里学来）的关系是**同一个库的两个来源**，
#: 不是「主用 + 兜底」：:class:`KindTemplateBank` 把它们摊平进同一张表，同一个家族
#: 取两边更高的分。自学习模板来自真实画面，分数天然更高（同族 0.80+），所以在已覆盖的
#: 17 个家族上仍然说了算；合成模板补的是自学习库没有的那 10 个家族
#: （黑潮/黑帆/魔方/迷城/梦魇/玩偶/觉醒者/龙骨/沙怪/幽灵）。
SHEET_DIR = Path(__file__).resolve().parent.parent / "runtime" / "paint_sheet"
#: 合成模板的二进制缓存。189 张 PNG 每次启动都解一遍要 1 秒以上，存成特征数组后是几十毫秒。
SHEET_CACHE_PATH = _BANK_DIR / "sheet_templates.npz"
#: 合成模板要放的倍数。
#:
#: 为什么不是 1.0：合成图里框只占画布的 **75%**（官方 ``frame SizeBox=192`` / 画布 256），
#: 而游戏里框几乎顶满格子（``paint_frame.TARGET_FRAME_OF_CELL = 0.977``）。两边中心 54%
#: 的搜索窗因此截到的不是同一块内容——合成图截到「整个头 + 一圈留白」，游戏格截到「放大的头部」。
#: 实机 6 格实测（玩偶/梦魇/伊吉/矮人/鼠人/存活）：
#:
#: ====  ==============
#: 倍数  6 格认对数
#: ====  ==============
#: 1.0   3/6
#: 1.15  4/6
#: 1.3   6/6
#: 1.45  6/6
#: ====  ==============
#:
#: 1.3~1.45 之间是平台，取多档并存（``max`` 取最高分）比押单一常数稳：
#: 不同分辨率下格子的实际边长不一样（87px / 104px 都见过），最优倍数会左右漂。
SHEET_SCALES = (1.0, 1.15, 1.3, 1.45)
#: 合成模板压平用的底色。用黑色而不是灰色：直方图那一项对底色不敏感（实测换底色分数一字不差），
#: 灰度 NCC 才是主导项，而黑色与游戏深色地块（``meanV≈82``）最接近。
SHEET_BACKGROUND = (0, 0, 0)



@dataclass(frozen=True)
class KindFeature:
    """一个格子的种类特征。"""

    #: 44×44 灰度搜索窗（已模糊）。
    window: np.ndarray
    #: 30×30 灰度模板（已模糊）。
    template: np.ndarray
    #: 128 维 H-S 直方图（L2 归一）。
    hist: np.ndarray


@dataclass(frozen=True)
class KindReading:
    """一次种类读取的结果。"""

    #: 认出的家族名（``邪影``）；没把握时为 ``None``。
    subtype: str | None
    #: 综合分（0~1）。
    score: float
    #: 是否达到阈值。
    present: bool
    #: 各家族的最高分，调试用。
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str | None:
        return family_key(self.subtype) if self.subtype else None


# ---------------------------------------------------------------- 家族目录


#: 「赛季」是**独立的玩法家族**，只有 Lv3/Lv4 两级，有自己的一张图标。
#:
#: 解包配置里 ``ColorPlay`` 图标一共 **16** 张（``ColorPlay1`` ~ ``ColorPlay16``），
#: 而玩法家族只有 14 个（``g001/g003~g014``）。没被任何 PlotItem 记录引用的是
#: ``ColorPlay2`` 和 ``ColorPlay16``：前者是「放进虚空、只剩等级框」的状态图标，
#: 后者是通关后才出现的宠物特性头像——都不是棋盘家族，见
#: ``assets/paint_icons/roles.json``。
#:
#: 赛季的图标是 ``ColorPlay15``。168 条记录里只有 201103（ItemName「玩法颜料-赛季」）
#: 的 ``buckets.after_ItemName`` 出现过它一次，但那个桶里同时混进了 Lua 报错字符串，
#: 属于解析残留，所以「引用次数」这一条不能单独当证据。
#:
#: 真正的证据在实机：``扩散后选择前.png``、``堆砌选择前.png``、``转锋后选择前.png``
#: 三张截图的面板 OCR 都读到「赛季」，与满盘图第 14 号图案匹配 0.730；
#: 棋盘上三格赛季的识别分 0.862~0.877，亚军只有 0.42~0.44——图案是稳定的同一张，
#: 不存在「7 个家族各画一张但都叫赛季」这回事。
#:
#: 它在目录里编为 ``g015``（:func:`catalog` 里唯一一条手工补的条目，带 ``note``）。
SEASON_PLACEHOLDER = "赛季"
#: 赛季家族在解包里的图标名，记录用。
SEASON_ICON = "UI_S8Gameplay_ColorPlay15_128"

_CATALOG_CACHE: dict[str, dict] | None = None


def catalog() -> dict[str, dict]:
    """读 ``paint_catalog.json``：家族编号 → {category, kind, name, levels, level_ids}。"""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        try:
            payload = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        _CATALOG_CACHE = payload if isinstance(payload, dict) else {}
    return _CATALOG_CACHE


def family_key(subtype: str | None) -> str | None:
    """家族名 → 目录编号（``邪影`` → ``m012``）。不在目录里返回 ``None``。"""
    if not subtype:
        return None
    for key, entry in catalog().items():
        if entry.get("name") == subtype:
            return key
    return None


def known_kinds() -> tuple[str, ...]:
    """官方目录里的全部家族名（13 怪物 + 14 玩法，含只有 Lv3/Lv4 的赛季）。"""
    return tuple(entry["name"] for entry in catalog().values() if entry.get("name"))


def category_of(subtype: str | None) -> str | None:
    """家族名 → ``monster`` / ``gameplay``。认对种类也就顺带认对了类别。"""
    key = family_key(subtype)
    if key is None:
        return None
    entry = catalog().get(key) or {}
    return entry.get("category")


# ---------------------------------------------------------------- 特征


def _center_square(tile_bgr: np.ndarray, ratio: float) -> np.ndarray:
    height, width = tile_bgr.shape[:2]
    side = int(round(min(height, width) * ratio))
    if side < 4:
        return tile_bgr
    top = (height - side) // 2
    left = (width - side) // 2
    return tile_bgr[top : top + side, left : left + side]


def _gray_patch(tile_bgr: np.ndarray, ratio: float, size: tuple[int, int]) -> np.ndarray:
    crop = _center_square(tile_bgr, ratio)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(resized, (3, 3), 0).astype(np.float32)


def color_hist(tile_bgr: np.ndarray) -> np.ndarray:
    """H-S 二维直方图，L2 归一。颜色分布和图案形状是两条正交证据。"""
    crop = cv2.resize(_center_square(tile_bgr, KIND_WINDOW_RATIO), KIND_HIST_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(KIND_HIST_BINS), [0, 180, 0, 256]).reshape(-1)
    hist = hist.astype(np.float32)
    return hist / (np.linalg.norm(hist) + 1e-6)


def kind_feature(tile_bgr: np.ndarray, *, window_ratio: float | None = None) -> KindFeature | None:
    """从一个格子里抽出种类特征。图太小 / 太空时返回 ``None``。"""
    if tile_bgr is None or tile_bgr.size == 0 or tile_bgr.ndim != 3:
        return None
    if min(tile_bgr.shape[:2]) < 12:
        return None
    window = _gray_patch(tile_bgr, KIND_WINDOW_RATIO if window_ratio is None else window_ratio, KIND_WINDOW_SIZE)
    template_ratio = KIND_TEMPLATE_RATIO if window_ratio is None else window_ratio * (KIND_TEMPLATE_RATIO / KIND_WINDOW_RATIO)
    template = _gray_patch(tile_bgr, template_ratio, KIND_TEMPLATE_SIZE)
    return KindFeature(window=window, template=template, hist=color_hist(tile_bgr))


def looks_like_paint_tile(tile_bgr: np.ndarray) -> bool:
    """挡掉深色漩涡 / 虚空地块——它们两两相似度能到 1.000，会污染模板库。

    实机满盘 36 格实测：16 个深色地块全是 ``meanV≈82、饱和像素占比≈0.00``，
    20 个真颜料格 ``meanV`` 最低 136.6。所以「整格平均亮度 < 100」这一刀就够了，
    不需要去算菱形环带（那要建两张 ``mgrid``，比这贵一个数量级）。
    """
    if tile_bgr is None or tile_bgr.size == 0 or tile_bgr.ndim != 3:
        return False
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    mean_value = float(hsv[:, :, 2].mean())
    if mean_value >= DARK_MEAN_VALUE_MAX:
        return True
    return float((hsv[:, :, 1] > 110).mean()) >= DARK_SAT_RATIO_MAX


def _ncc(image: np.ndarray, template: np.ndarray) -> float:
    """带平移容忍的归一化互相关。

    两个数组尺寸不同才能搜位置：大的一侧当图、小的一侧当模板，
    ``matchTemplate`` 会在图上滑一圈取最高分，于是 ±3px 抖动和轻微缩放都不怕。
    尺寸相同时退化成 1×1 的点积，那对像素级偏移毫无容忍度——所以这里
    统一由调用方传「窗口 vs 模板」，不要传两个窗口。
    """
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        image, template = template, image
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return float(result.max()) if result.size else 0.0


def _is_same_template(left: KindFeature, right: KindFeature) -> bool:
    """两份特征是不是同一张图案的副本（窗口和中心裁剪都高度一致）。"""
    return (
        _ncc(left.window, right.window) >= TEMPLATE_DUPLICATE_NCC
        and _ncc(left.template, right.template) >= TEMPLATE_DUPLICATE_NCC
    )


# ---------------------------------------------------------------- 合成模板表


def _read_rgba(path: Path) -> np.ndarray | None:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError):
        return None
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        return None
    return image


def _sheet_cell(image: np.ndarray, scale: float) -> np.ndarray:
    """把 256 画布按 ``scale`` 缩放后居中裁回 256：等价于「把框整体放大」。"""
    side = image.shape[0]
    if scale != 1.0:
        edge = max(8, int(round(side * scale)))
        resized = cv2.resize(image, (edge, edge), interpolation=cv2.INTER_AREA)
        if edge >= side:
            offset = (edge - side) // 2
            image = resized[offset : offset + side, offset : offset + side]
        else:
            canvas = np.zeros((side, side, 4), np.uint8)
            offset = (side - edge) // 2
            canvas[offset : offset + edge, offset : offset + edge] = resized
            image = canvas
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    base = np.zeros(image.shape[:2] + (3,), np.float32)
    base[:, :] = SHEET_BACKGROUND
    flat = image[:, :, :3].astype(np.float32) * alpha + base * (1.0 - alpha)
    return np.clip(flat, 0, 255).astype(np.uint8)


def _build_sheet_templates() -> dict[str, list[KindFeature]]:
    """从 ``runtime/paint_sheet`` 现算一份合成模板。

    文件名是 ``{家族编号}_Lv{等级}_tile.png``（``build_paint_sheet.py`` 生成，还有一份
    ``_backpack`` 形态给背包用）。家族编号查 :func:`catalog` 换成中文名，与自学习库同一套键。
    """
    if not SHEET_DIR.is_dir():
        return {}
    keys = catalog()
    out: dict[str, list[KindFeature]] = {}
    for path in sorted(SHEET_DIR.glob("*_tile.png")):
        # 文件名形如 ``g014_Lv5_tile.png``：家族编号 + 等级 + 形态后缀。
        parts = path.name[: -len("_tile.png")].split("_Lv")
        if len(parts) != 2:
            continue
        family_key, level_text = parts
        name = (keys.get(family_key) or {}).get("name")
        if not name or not level_text.isdigit():
            continue
        image = _read_rgba(path)
        if image is None:
            continue
        for scale in SHEET_SCALES:
            feature = kind_feature(_sheet_cell(image, scale))
            if feature is not None:
                out.setdefault(str(name), []).append(feature)
    return out


def _load_sheet_cache() -> dict[str, list[KindFeature]] | None:
    if not SHEET_CACHE_PATH.exists():
        return None
    try:
        with np.load(SHEET_CACHE_PATH) as data:
            version = int(data["version"])
            scales = tuple(float(value) for value in data["scales"])
            owners = [str(value) for value in data["owners"]]
            hists = data["hists"]
            windows = data["windows"]
            templates = data["templates"]
    except (OSError, KeyError, ValueError):
        return None
    # 倍数改了就必须重算：旧缓存里的尺度组合对不上，分数会整体偏低。
    # 比较时按 3 位小数归一——缓存是 float32，``1.15`` 存进去读出来是 ``1.1499999…``，
    # 直接比元组永远不相等（踩过一次：缓存写成了却每次启动都重算 6 秒）。
    if version != BANK_VERSION or tuple(round(value, 3) for value in scales) != tuple(
        round(value, 3) for value in SHEET_SCALES
    ):
        return None
    if not (len(owners) == len(hists) == len(windows) == len(templates)):
        return None
    out: dict[str, list[KindFeature]] = {}
    for index, owner in enumerate(owners):
        out.setdefault(owner, []).append(
            KindFeature(
                window=np.asarray(windows[index], dtype=np.float32),
                template=np.asarray(templates[index], dtype=np.float32),
                hist=np.asarray(hists[index], dtype=np.float32),
            )
        )
    return out


def _save_sheet_cache(payload: dict[str, list[KindFeature]]) -> None:
    owners: list[str] = []
    windows: list[np.ndarray] = []
    templates: list[np.ndarray] = []
    hists: list[np.ndarray] = []
    for name in sorted(payload):
        for feature in payload[name]:
            owners.append(name)
            windows.append(feature.window)
            templates.append(feature.template)
            hists.append(feature.hist)
    if not owners:
        return
    try:
        SHEET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            SHEET_CACHE_PATH,
            version=np.int64(BANK_VERSION),
            scales=np.asarray(SHEET_SCALES, dtype=np.float32),
            owners=np.asarray(owners),
            windows=np.stack(windows).astype(np.float32),
            templates=np.stack(templates).astype(np.float32),
            hists=np.stack(hists).astype(np.float32),
        )
    except OSError:
        return


_SHEET_MEMORY: dict[str, list[KindFeature]] | None = None


def sheet_templates() -> dict[str, list[KindFeature]]:
    """合成模板表（进程内缓存；磁盘缓存坏了就现算）。"""
    global _SHEET_MEMORY
    if _SHEET_MEMORY is not None:
        return _SHEET_MEMORY
    cached = _load_sheet_cache()
    if cached:
        _SHEET_MEMORY = cached
        return cached
    built = _build_sheet_templates()
    if built:
        _save_sheet_cache(built)
    _SHEET_MEMORY = built
    return built


# ---------------------------------------------------------------- 模板库


@dataclass
class KindTemplateBank:
    """家族图案模板库。每个家族可以存多个样本，取最高分。

    两个来源放进同一张表：

    * ``templates`` —— 从**实机截图**里学来的（:meth:`add` / :func:`learn_from_panel`），
      分数天然高（同族 0.80+），每个家族最多 :data:`MAX_TEMPLATES_PER_KIND` 份；
    * ``sheet`` —— 由官方图标 + 官方品质框**合成**的（:func:`sheet_templates`），
      27 家族 × 7 等级 × 多尺度，用来补齐自学习库没有的家族。
    """

    #: 家族名 → 特征列表（自学习）
    templates: dict[str, list[KindFeature]] = field(default_factory=dict)
    #: 家族名 → 特征列表（官方合成模板）
    sheet: dict[str, list[KindFeature]] = field(default_factory=dict)
    #: 摊平后的大数组缓存。``_stack()`` 每次重建 78 个数组要花掉大半时间，
    #: 库不变时直接复用——单格识别从 2.2ms 降到 0.3ms。
    _stacked: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None = field(
        default=None, repr=False
    )

    # ------------------------------------------------------------ 构建

    def merge_sheet(self, payload: dict[str, list[KindFeature]] | None = None) -> int:
        """把合成模板挂进同一个库。返回挂了几条。

        ``sheet`` 与 ``templates`` 分开存是刻意的：``add()`` 的配额（每个家族 3 份）
        只约束自学习模板，否则合成模板一挂上来，面板自学习就再也加不进去了。
        """
        source = sheet_templates() if payload is None else payload
        if not source:
            return 0
        added = 0
        for name, features in source.items():
            if not features:
                continue
            self.sheet.setdefault(name, []).extend(features)
            added += len(features)
        self._stacked = None
        return added

    def add(self, subtype: str, tile_bgr: np.ndarray, *, replace: bool = False) -> bool:
        """往库里加一个样本。库满或图不合格时返回 ``False``。"""
        if not subtype:
            return False
        feature = kind_feature(tile_bgr)
        if feature is None:
            return False
        items = self.templates.setdefault(subtype, [])
        if replace and items:
            items[-1] = feature
            self._stacked = None
            return True
        if len(items) >= MAX_TEMPLATES_PER_KIND:
            return False
        # 同一张图案重复入库不产生新信息，只会把配额占满、把每格的 NCC 计算
        # 量翻倍（见 :data:`TEMPLATE_DUPLICATE_NCC`）。
        if any(_is_same_template(feature, item) for item in items):
            return False
        items.append(feature)
        self._stacked = None
        return True

    def add_from_cells(self, subtype: str, cells: list[np.ndarray]) -> int:
        added = 0
        for cell in cells:
            if self.add(subtype, cell):
                added += 1
        return added

    def __len__(self) -> int:
        """自学习模板条数。合成模板单独用 :meth:`sheet_count` 报，两者性质不同。"""
        return sum(len(items) for items in self.templates.values())

    def sheet_count(self) -> int:
        return sum(len(items) for items in self.sheet.values())

    def kinds(self) -> tuple[str, ...]:
        """两个来源合起来覆盖的家族。"""
        return tuple(sorted(set(self.templates) | set(self.sheet)))

    def unknown_kinds(self) -> tuple[str, ...]:
        """官方目录里有、两个来源都没有的家族。"""
        covered = set(self.templates) | set(self.sheet)
        return tuple(name for name in known_kinds() if name not in covered)

    def stats_text(self) -> str:
        missing = self.unknown_kinds()
        total = len(self.kinds())
        lines = [f"{total}/{len(known_kinds())} 个家族, {len(self)} 个学习模板 + {self.sheet_count()} 个合成模板"]
        if self.templates:
            detail = "，".join(f"{name}×{len(items)}" for name, items in sorted(self.templates.items()))
            lines.append(f"  已学: {detail}")
        if self.sheet:
            detail = "，".join(f"{name}×{len(items)}" for name, items in sorted(self.sheet.items()))
            lines.append(f"  合成: {detail}")
        if missing:
            lines.append(f"  还缺: {'，'.join(missing)}")
        return "\n".join(lines)

    # ------------------------------------------------------------ 读取

    def _stack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """把模板摊平成四个大数组，方便一次算完直方图相似度。"""
        if self._stacked is not None:
            return self._stacked
        hists: list[np.ndarray] = []
        windows: list[np.ndarray] = []
        templates: list[np.ndarray] = []
        owners: list[str] = []
        for subtype in self.kinds():
            for feature in self.templates.get(subtype, ()):
                hists.append(feature.hist)
                windows.append(feature.window)
                templates.append(feature.template)
                owners.append(subtype)
            for feature in self.sheet.get(subtype, ()):
                hists.append(feature.hist)
                windows.append(feature.window)
                templates.append(feature.template)
                owners.append(subtype)
        if not hists:
            empty = np.zeros((0, 1), np.float32)
            self._stacked = (empty, empty, empty, [])
            return self._stacked
        self._stacked = (np.stack(hists), np.stack(windows), np.stack(templates), owners)
        return self._stacked

    def read(self, tile_bgr: np.ndarray) -> KindReading:
        if not self.templates and not self.sheet:
            return KindReading(subtype=None, score=0.0, present=False)
        if not looks_like_paint_tile(tile_bgr):
            return KindReading(subtype=None, score=0.0, present=False)
        feature = kind_feature(tile_bgr)
        if feature is None:
            return KindReading(subtype=None, score=0.0, present=False)

        hist_matrix, windows, templates, owners = self._stack()
        if hist_matrix.size == 0:
            return KindReading(subtype=None, score=0.0, present=False)

        hist_scores = hist_matrix @ feature.hist
        order = np.argsort(-hist_scores)
        # 每个家族先认领一个「代表作」（该家族直方图分最高的那份），再把直方图
        # 前 K 名并进来。这样预筛只能删掉同家族内的冗余模板，绝不会把某个家族
        # 整个挡在 NCC 外面——家族名才是我们要的结论，而直方图这个启发式对
        # 「外框变色」天生敏感（见 :data:`KIND_PREFILTER_TOP_K` 的说明）。
        per_kind: dict[str, int] = {}
        for index in order:
            per_kind.setdefault(owners[index], int(index))
        candidates = {int(index) for index in order[: max(1, KIND_PREFILTER_TOP_K)]}
        candidates.update(per_kind.values())
        best_by_kind: dict[str, float] = {}
        for index in sorted(candidates):
            # 双向各算一次：查询当图 / 模板当图，谁都不吃亏。
            ncc = (
                _ncc(feature.window, templates[index]) + _ncc(windows[index], feature.template)
            ) / 2.0
            score = ncc * KIND_NCC_WEIGHT + float(hist_scores[index]) * (1.0 - KIND_NCC_WEIGHT)
            owner = owners[index]
            if score > best_by_kind.get(owner, 0.0):
                best_by_kind[owner] = score
        # 到这里每个家族都至少有一份模板被真正评估过，``scores`` 对全库完整。
        if best_by_kind:
            subtype, score = max(best_by_kind.items(), key=lambda item: item[1])
        else:
            subtype, score = None, 0.0
        present = subtype is not None and score >= KIND_MATCH_THRESHOLD
        return KindReading(
            subtype=subtype if present else None,
            score=score,
            present=present,
            scores=best_by_kind,
        )

    # ------------------------------------------------------------ 持久化

    def save(self, path: Path | str | None = None, labels_path: Path | str | None = None) -> Path:
        target = Path(path) if path is not None else _BANK_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        hists, _windows, templates, owners = self._stack()
        payload: dict[str, np.ndarray] = {}
        if owners:
            payload["hists"] = hists.astype(np.float32)
            payload["windows"] = _windows.astype(np.float32)
            payload["templates"] = templates.astype(np.float32)
        np.savez_compressed(target, **payload)
        sidecar = Path(labels_path) if labels_path is not None else _LABELS_PATH
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps({"version": BANK_VERSION, "labels": owners}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | str | None = None, labels_path: Path | str | None = None) -> "KindTemplateBank":
        target = Path(path) if path is not None else _BANK_PATH
        sidecar = Path(labels_path) if labels_path is not None else _LABELS_PATH
        bank = cls()
        if not target.exists() or not sidecar.exists():
            return bank
        try:
            labels = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return bank
        if not isinstance(labels, dict) or labels.get("version") != BANK_VERSION:
            return bank
        owners = list(labels.get("labels") or [])
        if not owners:
            return bank
        try:
            with np.load(target) as data:
                hists = data["hists"]
                windows = data["windows"]
                templates = data["templates"]
        except (OSError, KeyError, ValueError):
            return bank
        if not (len(owners) == len(hists) == len(windows) == len(templates)):
            return bank
        for index, owner in enumerate(owners):
            bank.templates.setdefault(owner, []).append(
                KindFeature(
                    window=np.asarray(windows[index], dtype=np.float32),
                    template=np.asarray(templates[index], dtype=np.float32),
                    hist=np.asarray(hists[index], dtype=np.float32),
                )
            )
        return bank


_BANK_CACHE: KindTemplateBank | None = None


def bank() -> KindTemplateBank:
    """进程内共享的模板库（首次调用时从磁盘加载，并挂上官方合成模板）。"""
    global _BANK_CACHE
    if _BANK_CACHE is None:
        _BANK_CACHE = KindTemplateBank.load()
        _BANK_CACHE.merge_sheet()
    return _BANK_CACHE


def reload_bank() -> KindTemplateBank:
    global _BANK_CACHE
    _BANK_CACHE = KindTemplateBank.load()
    _BANK_CACHE.merge_sheet()
    return _BANK_CACHE


def read_kind(tile_bgr: np.ndarray) -> KindReading:
    """认一个棋盘格子的颜料种类。库为空时返回「没认出来」。"""
    active = bank()
    if not active.templates and not active.sheet:
        return KindReading(subtype=None, score=0.0, present=False)
    return active.read(tile_bgr)


# ---------------------------------------------------------------- 从面板自学习

#: 面板图标和棋盘格的取景比例不同（面板图标里图案几乎顶满 90px，棋盘格里只占约 60%），
#: 所以配对时要在面板图标这一侧扫一遍比例，找到最贴的那个尺度。
PANEL_SCALE_CANDIDATES = (0.44, 0.52, 0.60, 0.68, 0.76, 0.84, 0.92)
#: 自学习的最低分。宁可少学几个，也不要学错——错一个模板会持续污染这个家族。
#:
#: 这个阈值比 :data:`KIND_MATCH_THRESHOLD` 高，因为「面板图标 ↔ 棋盘格」比
#: 「棋盘格 ↔ 棋盘格」难：面板图标里图案顶满 90px、底色是圆形徽章，棋盘格里
#: 图案只占约 60%、底色是菱形。11 张实机截图实测：真正对得上的两张是
#: 邪影 0.769、机械之神 0.768；对不上的最高 0.686（而且差距只有 0.011）。
#: 所以 0.72 这一刀正好卡在中间。
LEARN_MIN_SCORE = 0.72
#: 最佳匹配还要领先亚军这么多，才认为「确实是这一格」。
LEARN_MIN_MARGIN = 0.12


def match_panel_icon(panel_icon: np.ndarray, cells: list[np.ndarray]) -> tuple[int, float, float] | None:
    """把「当前颜料」面板图标对到棋盘上的一格。

    返回 ``(格子下标, 分数, 与亚军的差距)``；对不上返回 ``None``。
    """
    if not cells:
        return None
    hist_query = color_hist(panel_icon)
    best: tuple[int, float] | None = None
    for index, cell in enumerate(cells):
        if not looks_like_paint_tile(cell):
            continue
        hist_score = float(np.dot(hist_query, color_hist(cell)))
        cell_score = 0.0
        for ratio in PANEL_SCALE_CANDIDATES:
            feature = kind_feature(panel_icon, window_ratio=ratio)
            if feature is None:
                continue
            window = _gray_patch(cell, KIND_WINDOW_RATIO, KIND_WINDOW_SIZE)
            ncc = _ncc(window, feature.template)
            cell_score = max(cell_score, ncc * KIND_NCC_WEIGHT + hist_score * (1.0 - KIND_NCC_WEIGHT))
        if best is None or cell_score > best[1]:
            best = (index, cell_score)
    if best is None:
        return None
    runner_up = 0.0
    for index, cell in enumerate(cells):
        if index == best[0] or not looks_like_paint_tile(cell):
            continue
        hist_score = float(np.dot(hist_query, color_hist(cell)))
        window = _gray_patch(cell, KIND_WINDOW_RATIO, KIND_WINDOW_SIZE)
        cell_score = 0.0
        for ratio in PANEL_SCALE_CANDIDATES:
            feature = kind_feature(panel_icon, window_ratio=ratio)
            if feature is None:
                continue
            ncc = _ncc(window, feature.template)
            cell_score = max(cell_score, ncc * KIND_NCC_WEIGHT + hist_score * (1.0 - KIND_NCC_WEIGHT))
        runner_up = max(runner_up, cell_score)
    return best[0], best[1], best[1] - runner_up


def learn_from_panel(
    panel_icon: np.ndarray,
    subtype: str | None,
    cells: list[np.ndarray],
    *,
    active: KindTemplateBank | None = None,
) -> bool:
    """面板上读到家族名时，顺手把棋盘上对应的那一格收进模板库。

    这是唯一能自动拿到「图案 ↔ 名字」对应关系的路子：解包资源里只有图标名
    （``UI_S8Gameplay_ColorMonster2_128``），没有图，也没有名字对照表；
    而右侧面板同时渲染着图标和 ``怪物颜料·邪影LV3`` 这行字。
    """
    if not subtype or family_key(subtype) is None:
        return False
    target = active if active is not None else bank()
    if len(target.templates.get(subtype, [])) >= MAX_TEMPLATES_PER_KIND:
        return False
    matched = match_panel_icon(panel_icon, cells)
    if matched is None:
        return False
    index, score, margin = matched
    if score < LEARN_MIN_SCORE or margin < LEARN_MIN_MARGIN:
        return False
    return target.add(subtype, cells[index])


# ---------------------------------------------------------------- 命令行


def _read_image(path: Path) -> np.ndarray | None:
    """Windows 上 ``cv2.imread`` 读不了中文路径，统一走 ``imdecode``。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if ok:
        buffer.tofile(str(path))


def _scene_cells(image: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]], np.ndarray | None, str | None]:
    """跑一遍场景分析，取棋盘 36 格 + 当前颜料面板。"""
    from . import vision

    scene = vision.analyze_scene(image)
    if scene is None:
        return [], [], None, None
    cells: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    for tile in scene.board.tiles:
        rect = tile.rect
        cells.append(image[rect.y : rect.bottom, rect.x : rect.right])
        positions.append((tile.row, tile.col))
    panel_icon = None
    subtype = None
    if scene.current_paint is not None:
        icon = scene.current_paint.icon_rect
        panel_icon = image[icon.y : icon.bottom, icon.x : icon.right]
        subtype = scene.current_paint.subtype
    return cells, positions, panel_icon, subtype


def harvest_images(
    paths: list[Path],
    *,
    save: bool = True,
    verbose: bool = True,
    skip: set[tuple[str, int, int]] | None = None,
) -> dict[str, int]:
    """扫一批截图，用「面板图标 + 面板文字」自动给模板库打标。

    返回 ``{家族名: 新增样本数}``。凑不齐「分数够高 + 领先亚军够多」的样本会被丢掉，
    宁可少学也不学错——一个错模板会持续污染这个家族。

    ``skip`` 是 ``{(图名, 行, 列)}``：人工在覆盖台账里划掉的格子不许入库。
    这条口子非有不可——实机里颜料**放进虚空之后图案会整个换掉**（邪影变成一颗
    蓝色能量球），而右侧面板照样显示「邪影」，于是面板图标会以 0.769 的高分、
    0.206 的大差距匹配到那颗球。阈值拦不住它，只有人工划线能拦住。
    """
    active = reload_bank() if save else bank()
    excluded = skip or set()
    learned: dict[str, int] = {}
    for path in paths:
        image = _read_image(path)
        if image is None:
            if verbose:
                print(f"  跳过（读不出）: {path.name}")
            continue
        cells, positions, panel_icon, subtype = _scene_cells(image)
        if panel_icon is None or not subtype:
            if verbose:
                print(f"  跳过（面板没读到种类）: {path.name}")
            continue
        # 被划掉的格子要从候选里剔掉再匹配，否则它照样可能当上最佳格。
        keep = [
            index
            for index, position in enumerate(positions)
            if (path.name, position[0], position[1]) not in excluded
        ]
        if not keep:
            continue
        matched = match_panel_icon(panel_icon, [cells[index] for index in keep])
        if matched is None:
            continue
        short_index, score, margin = matched
        index = keep[short_index]
        ok = score >= LEARN_MIN_SCORE and margin >= LEARN_MIN_MARGIN
        if verbose:
            flag = "收录" if ok else "放弃"
            print(f"  {path.name}: 面板={subtype} 最佳格={index} 分数={score:.3f} 差距={margin:.3f} → {flag}")
        if not ok:
            continue
        if active.add(subtype, cells[index]):
            learned[subtype] = learned.get(subtype, 0) + 1
    if save and learned:
        active.save()
    return learned


def _pair_score(left: KindFeature, right: KindFeature) -> float:
    ncc = (_ncc(left.window, right.template) + _ncc(right.window, left.template)) / 2.0
    return ncc * KIND_NCC_WEIGHT + float(np.dot(left.hist, right.hist)) * (1.0 - KIND_NCC_WEIGHT)


#: 归组阈值。实测同族最低 0.802、异族最高 0.547，0.70 落在中间的空档里。
FAMILY_CLUSTER_THRESHOLD = 0.70


def cluster_cells(cells: list[np.ndarray], threshold: float = FAMILY_CLUSTER_THRESHOLD) -> list[list[int]]:
    """把一屏的颜料格按家族归组，返回每组的格子下标（组内第一项当代表）。

    只用来「生成一张待命名的拼图」，不参与运行时识别——运行时走模板库。
    """
    candidates = [index for index, cell in enumerate(cells) if looks_like_paint_tile(cell)]
    features = {index: kind_feature(cells[index]) for index in candidates}
    features = {index: feature for index, feature in features.items() if feature is not None}
    groups: list[list[int]] = []
    assigned: set[int] = set()
    for index in candidates:
        if index in assigned or index not in features:
            continue
        group = [index]
        assigned.add(index)
        for other in candidates:
            if other in assigned or other not in features:
                continue
            if _pair_score(features[index], features[other]) >= threshold:
                group.append(other)
                assigned.add(other)
        groups.append(group)
    return groups


def build_montage(image_path: Path, out_png: Path, out_json: Path, *, unique: bool = True) -> dict:
    """把一张截图里的颜料格排成带编号的拼图，供人工命名。

    解包资源里只有图标名、没有图，图案 ↔ 名字这张表只能人工填一次。
    默认按家族去重（``unique=True``）：同一家族的多个格子只出一张代表图，
    命名一次就能覆盖全组——15 个家族只需要填 15 行。

    拼图旁边落一个 JSON，填好每格的 ``name`` 再跑 ``apply`` 就能入库。
    """
    image = _read_image(image_path)
    if image is None:
        raise SystemExit(f"读不出图片: {image_path}")
    cells, positions, _panel, _subtype = _scene_cells(image)
    clusters = cluster_cells(cells)
    if not clusters:
        raise SystemExit("这张图里没找到颜料格")
    if not unique:
        clusters = [[index] for group in clusters for index in group]
    active = bank()
    size = 150
    #: 每格下面单独留一条字幕带。名字必须画在格子**外面**——格子右下角自带
    #: ``LvN`` 角标，四字家族名（征战之神、狩猎之神）画在格子内会被角标压掉最后一个字。
    caption_height = 26
    columns = 7
    rows = (len(clusters) + columns - 1) // columns
    row_height = size + caption_height
    canvas = np.zeros((rows * row_height, columns * size, 3), dtype=np.uint8)
    entries: list[dict] = []
    for slot, group in enumerate(clusters):
        representative = group[0]
        top = (slot // columns) * row_height
        left = (slot % columns) * size
        canvas[top : top + size, left : left + size] = cv2.resize(
            cells[representative], (size, size), interpolation=cv2.INTER_CUBIC
        )
        cv2.putText(
            canvas, str(slot), (left + 4, top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        # 库里已经认识的家族直接预填，人工只需要补剩下的。
        guessed = active.read(cells[representative]) if active.templates else None
        preset = guessed.subtype if guessed is not None and guessed.present else None
        caption = preset or "?"
        cv2.putText(
            canvas,
            caption,
            (left + 6, top + size + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (120, 255, 120) if preset else (110, 110, 110),
            2,
        )
        if len(group) > 1:
            cv2.putText(
                canvas,
                f"x{len(group)}",
                (left + size - 40, top + size + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (120, 255, 120),
                2,
            )
        entries.append(
            {
                "slot": slot,
                "cell": representative,
                "row": positions[representative][0],
                "col": positions[representative][1],
                "also": group[1:],
                "name": preset,
                "auto": preset is not None,
            }
        )
    _write_image(out_png, canvas)
    payload = {"image": str(image_path), "cells": entries}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def apply_labels(json_path: Path, *, save: bool = True, verbose: bool = True) -> dict[str, int]:
    """把人工填好的 ``names`` 收进模板库。

    每格只填一个家族名；同组的其它格子（``also``）会一起入库，
    每个家族最多留 :data:`MAX_TEMPLATES_PER_KIND` 个样本。
    """
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit(f"读不出标注文件: {json_path}")
    image = _read_image(Path(payload.get("image") or ""))
    if image is None:
        raise SystemExit("标注文件里的 image 路径读不出图片")
    cells, _positions, _panel, _subtype = _scene_cells(image)
    active = reload_bank() if save else bank()
    added: dict[str, int] = {}
    for entry in payload.get("cells") or []:
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        if family_key(name) is None:
            if verbose:
                print(f"  跳过（不在官方目录里）: {name}")
            continue
        indices = [entry.get("cell")] + list(entry.get("also") or [])
        for index in indices:
            if not isinstance(index, int) or not (0 <= index < len(cells)):
                continue
            if active.add(name, cells[index]):
                added[name] = added.get(name, 0) + 1
    if save and added:
        active.save()
    return added


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="paint_kind", description="颜料种类模板库工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_harvest = sub.add_parser("harvest", help="用「面板图标 + 面板文字」自动入库")
    p_harvest.add_argument("paths", nargs="+", help="截图或截图目录")

    p_montage = sub.add_parser("montage", help="生成带编号的拼图，供人工命名")
    p_montage.add_argument("image", help="截图路径")
    p_montage.add_argument("--png", default="runtime/paint_kinds_montage.png")
    p_montage.add_argument("--json", default="runtime/paint_kinds_montage.json")

    p_apply = sub.add_parser("apply", help="把填好 names 的标注文件入库")
    p_apply.add_argument("json", help="标注文件路径")

    sub.add_parser("stats", help="看模板库现状")

    args = parser.parse_args(argv)

    if args.command == "harvest":
        targets: list[Path] = []
        for raw in args.paths:
            path = Path(raw)
            if path.is_dir():
                targets.extend(sorted(path.glob("*.png")))
            elif path.is_file():
                targets.append(path)
        if not targets:
            print("没找到图片")
            return 1
        print(f"扫描 {len(targets)} 张图")
        learned = harvest_images(targets)
        total = sum(learned.values())
        print(f"新收录 {total} 个样本，覆盖 {len(learned)} 个家族: {learned}")
        print(f"模板库: {bank().stats_text()}")
        return 0

    if args.command == "montage":
        payload = build_montage(Path(args.image), Path(args.png), Path(args.json))
        print(f"拼图: {args.png}  标注文件: {args.json}  共 {len(payload['cells'])} 格")
        print("把每格的 name 填成官方家族名（如 鼠人 / 邪影 / 宝箱），再跑 apply。")
        return 0

    if args.command == "apply":
        added = apply_labels(Path(args.json))
        print(f"入库 {sum(added.values())} 个样本: {added}")
        print(f"模板库: {bank().stats_text()}")
        return 0

    if args.command == "stats":
        print(bank().stats_text())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
