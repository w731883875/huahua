"""颜料外框：14 张官方框素材，以及「只剩框」的虚空形态。

## 素材从哪来

S8 的美术资源在 PAK 里，图集 ``Atlas_Atlas``（4096² BC7）把 14 个框排在一起。
按连通块扫出来、再按颜色/形状分类，正好是 **7 菱形 + 7 圆形**，与
「白绿蓝紫橙红彩 = Lv1~Lv7」完全对应：

===========  ==================  ==================
形状          谁用                依据
===========  ==================  ==================
圆形框        怪物家族（13 个）    ``m00N`` ↔ ``ColorMonster{N}``
菱形框        玩法家族（14 个）    ``g00N`` ↔ ``ColorPlay{N}``
===========  ==================  ==================

这份素材有两个用途：

1. **出核对表**（``scripts/build_paint_sheet.py``）：图标 + 框合成背包形态与地块形态；
2. **认虚空形态**（:func:`looks_voided`）：颜料放进虚空后图案被整个抹掉，
   只剩一圈彩色框——实机 ``斜对角.png`` r3c1 就是 Lv3 蓝环 + 空中心。

## 虚空地块的框 = 初始颜料的等级框

这一点容易搞反：虚空格上的框**不是**「这格颜料自己的等级」，也不是固定的，
而是**本局初始颜料的等级**——初始颜料升到 Lv5，虚空格的框就跟着变橙。
所以：

* 虚空格的框色只能当「初始颜料现在几级」来读，不能当这格颜料的价值；
* 反过来，读任意一个虚空格就能拿到初始颜料等级，比翻面板 OCR 稳。

实机样本（``斜对角.png`` r3c1）：Lv3 蓝环 + 圆形，中心全平。

## 背包和地块是同一套形状规则

**怪物 = 圆形，玩法 = 菱形，背包和棋盘都一样。**

解包资源里能对上：``Img_Paint_0N``（圆形底盘）与 ``Img_Paint_0N_Quality``（菱形框）
各 7 张按等级，``Img_Icon_Monster`` / ``Img_Icon_NonMonster`` 各 1 张做形状遮罩。
形状由**类别**决定，与「在背包里还是在地上」无关。

## 图标占框多大：直接取解包出来的官方设置

**这一版不再拟合，而是从 UI 蓝图里读出来的原始设置。**

``Content/BluePrint/UI/S8Gameplay/S8Paint.uexp`` （从 ``pakchunk1`` 解出）
里，颜料控件只有两个系统性出现的尺寸：

* ``frame = 192`` —— 品质框（``Quality1`` / ``Quality2``）的 SizeBox；
* ``icon  = 128`` —— 图标（``Icon`` / ``BonusIcon``）的 SizeBox。

两者之比 **128 / 192 = 0.6667**，就是图标相对框的官方比例。
``S8ChoosePaint``、``S8DrawMap`` 里是同一对数字（192 出现 15/63 次、128 出现 2/10 次），
说明这是整个 S8 玩法界面通用的尺寸，不是某个面板的特例。

所以 :data:`ICON_SCALE` 的含义被重新定义为 **图标 ÷ 框（逻辑尺寸）**，
不再需要 ``ICON_ART_OF_CANVAS`` 这层换算：

* ``ICON_SCALE = 128 / 192 = 0.6667``
* ``FRAME_SCALE`` 只负责把「框素材画布」放成「一个逻辑格」：
  ``FRAME_LOGICAL / 框素材画布边长``。因为切图带辉光留白，画布 198~233，
  所以这个系数在 0.82~0.97 之间，**不是放大**。

### 之前为什么会算成 1.30

一直在从实机截图**反推**两个未知量（框多大、图标多大），但格子和逻辑格之间
还夹着一个未知的 UI 缩放比，两个未知数搅在一起，怎么拟合都能自圆其说：

* ``FRAME_RING_OF_CANVAS = 0.74`` 量的是**实心环**，而视觉框外沿是**含辉光**那圈；
  拿小的去除大的，倍数就被高估；
* 蒙版 IoU 会奖励「填满」而不是「对齐」，于是框越大分越高，把错误方向当成正解。

现在官方尺寸是硬证据：**192 与 128 是同一份 uexp 里成对出现的 SizeBox 值**，
不需要再猜。实机截图只用来做最后一道校验（见 ``TARGET_*``）。


## 为什么形状不做运行时判据

试过三种从像素反推形状的办法，在 86 个带真值的棋盘格上都不够稳：

============================  ==========
判据                          形状准确率
============================  ==========
饱和环剖面峰值                 63/86
外圈高饱和像素质心集中度        57/86
框素材掩码模板匹配（饱和度）    47/86
============================  ==========

格子只有 87px，框外还有一圈很亮的辉光，把细环的形状信息糊掉了。与其硬接一个
六成准确率的判据，不如把形状当**已知规则**用：家族认出来了，类别就知道，
形状也就定了（:func:`shape_of_category`）。所以这里只保留素材和虚空检测。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: 14 个框的目录：``{circle,diamond}_Lv{1..7}_{白绿蓝紫橙红彩}.png``。
FRAME_DIR = Path(__file__).resolve().parent / "assets" / "paint_frames"
#: 框的清单（形状、等级、在图集里的源矩形）。
MANIFEST_PATH = FRAME_DIR / "frames.json"

#: 等级 → 框颜色名。只用于拼文件名，判据本身是颜色。
LEVEL_COLOR = {1: "白", 2: "绿", 3: "蓝", 4: "紫", 5: "橙", 6: "红", 7: "彩"}
#: 类别 → 框形状。解包资源里的硬规则。
SHAPE_OF_CATEGORY = {"monster": "circle", "gameplay": "diamond"}
#: 形状 → 类别（反查）。
CATEGORY_OF_SHAPE = {value: key for key, value in SHAPE_OF_CATEGORY.items()}

#: 「只剩框」的判据：中心 52% 缩到 48×48 后的拉普拉斯方差。
#:
#: 86 个带真值的颜料格最低 677.8（邪影 Lv4，深色图案），虚空格只有 25.6 ——
#: 差 26 倍。阈值取 150，两边都留着大余量。
#: 必须先缩放到固定边长再算：拉普拉斯方差随分辨率变化，不归一化换个分辨率就失效。
VOID_CENTER_LAPLACIAN_MAX = 150.0
#: 取中心多大一块。太小会漏掉「图案偏一边」的情况，太大会把框的辉光算进来。
VOID_CENTER_RATIO = 0.52
#: 归一化边长。
VOID_CENTER_SIZE = 48

#: 框素材画布边长（切出来的 198~233，含辉光与外沿）。
FRAME_CANVAS_NATIVE = 200
#: 图标贴图的实际像素边长。文件名 ``_128`` 是逻辑尺寸，贴图本身 256×256。
ICON_CANVAS_NATIVE = 256

#: S8Paint.uexp 里的品质框 SizeBox（``Quality1`` / ``Quality2``）。
FRAME_LOGICAL = 192
#: S8Paint.uexp 里的图标 SizeBox（``Icon`` / ``BonusIcon``）。
ICON_LOGICAL = 128

#: 实机目标（86 格样本中位）：框环外径 / 格子边长。
TARGET_FRAME_OF_CELL = 0.977
#: 实机目标（86 格样本中位）：图案边长 / 格子边长。
TARGET_ART_OF_CELL = 0.656

#: 框素材的**实心彩色环**占其画布多少（alpha>0.85 且高饱和的外接框）。
#: 圆形实测 0.68~0.75、菱形 0.74~0.80；取两类中位。画布其余部分是渐隐辉光。
#: 只用于文档说明与历史对照，**不再参与任何比例推导**。
FRAME_RING_OF_CANVAS = 0.74
#: 图标画布里图案有效区占多少（30 张图标 alpha 外接框中位）。
ICON_ART_OF_CANVAS = 0.8965

#: 图标该画多大：图标 ÷ 框，**官方逻辑尺寸之比**。
#:
#: ``128 / 192 = 0.6667``，直接来自 ``S8Paint.uexp`` 的 SizeBox，
#: 不是从截图反推的，也不是「图案占画布」的换算值。
ICON_SCALE = ICON_LOGICAL / FRAME_LOGICAL

#: 框该画多大：框素材画布 ÷ 一个逻辑格。
#:
#: 切图带辉光留白，画布 198~233（见 ``frames.json`` 的 ``source``），
#: 所以这个系数**略小于 1**——框素材本身比逻辑格大，要缩一点才铺满一格。
#: 早期这里被写成 1.30（「框要放大」），是从截图反推时把实心环和含辉光外沿
#: 搞混了导致的，见模块文档。
FRAME_SCALE = FRAME_LOGICAL / FRAME_CANVAS_NATIVE

#: 虚空形态在界面上的名字。不是家族，``paint_kind.family_key`` 查不到它。
VOIDED = "虚空"


@dataclass(frozen=True)
class FrameAsset:
    """一个框素材。"""

    shape: str
    level: int
    path: Path
    #: 在图集里的源矩形 ``(x, y, w, h)``；没有记录时为 ``None``。
    source: tuple[int, int, int, int] | None


def manifest() -> dict[str, dict]:
    """``frames.json``：文件名 → {shape, level, source}。"""
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def frames() -> dict[tuple[str, int], FrameAsset]:
    """``(形状, 等级)`` → 框素材。缺文件时抛 ``FileNotFoundError``。"""
    result: dict[tuple[str, int], FrameAsset] = {}
    for shape in SHAPE_OF_CATEGORY.values():
        for level in LEVEL_COLOR:
            path = FRAME_DIR / f"{shape}_Lv{level}_{LEVEL_COLOR[level]}.png"
            if not path.exists():
                raise FileNotFoundError(f"框素材缺失: {path}")
            raw = manifest().get(path.name) or {}
            source = raw.get("source")
            result[(shape, level)] = FrameAsset(
                shape=shape,
                level=level,
                path=path,
                source=tuple(source) if isinstance(source, list) and len(source) == 4 else None,
            )
    return result


def frame_path(shape: str, level: int) -> Path:
    """按形状和等级取框素材路径。"""
    if shape not in SHAPE_OF_CATEGORY.values():
        raise ValueError(f"未知框形状: {shape}")
    if level not in LEVEL_COLOR:
        raise ValueError(f"未知等级: {level}")
    return FRAME_DIR / f"{shape}_Lv{level}_{LEVEL_COLOR[level]}.png"


def shape_of_category(category: str | None) -> str | None:
    """``monster`` → ``circle``；``gameplay`` → ``diamond``；其它返回 ``None``。"""
    return SHAPE_OF_CATEGORY.get(str(category)) if category else None


def void_center_stats(tile_bgr: np.ndarray) -> tuple[float, float, float] | None:
    """中心块的 ``(拉普拉斯方差, 边缘像素占比, 灰度标准差)``。

    先裁中心 :data:`VOID_CENTER_RATIO`，再缩到 :data:`VOID_CENTER_SIZE` 见方，
    这样换分辨率、换棋盘大小都不影响阈值。
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return None
    height, width = tile_bgr.shape[:2]
    side = int(round(min(height, width) * VOID_CENTER_RATIO))
    if side < 8:
        return None
    top = (height - side) // 2
    left = (width - side) // 2
    crop = tile_bgr[top : top + side, left : left + side]
    crop = cv2.resize(crop, (VOID_CENTER_SIZE, VOID_CENTER_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edge = float((np.hypot(gx, gy) > 40).mean())
    return laplacian, edge, float(gray.std())


def looks_voided(tile_bgr: np.ndarray) -> bool:
    """这一格是不是「放进虚空、只剩框」的颜料。

    只在**确认有颜料**（右下角有 ``LvN`` 角标）之后调用——空白格、阻挡地块
    的中心同样是平的，但它们本来就不该走颜料这条路。

    命中之后，框的颜色仍然可以读等级，但那个等级属于**初始颜料**，
    不是这格颜料自己的等级（见模块文档）。
    """
    stats = void_center_stats(tile_bgr)
    if stats is None:
        return False
    return stats[0] < VOID_CENTER_LAPLACIAN_MAX
