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

## 图标占框多大、框占格子多大

这两个量都是**从实机截图上量出来的**，不是从贴图尺寸推的（推错过两次）。

量法一（``scripts/measure_art_scale.py``）：把图标带 alpha 蒙版在实机格子上做
多尺度 ``TM_CCORR_NORMED``，只比图案本身的像素，框和辉光不参与；
分数峰所在的比例就是图案真实尺寸。86 格样本、分数 0.90~1.00：

====================  ==========  ==========
量                    中位数      说明
====================  ==========  ==========
框环外径 / 格子边长   0.977       框几乎撑满格子
图案边长 / 格子边长   0.656       图案只占格子约三分之二
====================  ==========  ==========

量法二（``scripts/fit_composite.py``）：按候选参数合成，与实机格子**逐像素比对**
（只比中心 62% 的图案区，避开框边与角标），78 格取中位 NCC 扫网格。
两条曲线都是单峰，峰位 **FRAME=1.24、ICON=0.62**，和量法一独立吻合。

素材自身的占比：框素材的有色环只占其画布的 0.845，图标画布里的图案占 0.8965。
所以：

* ``FRAME_SCALE = 1.234`` —— 框素材要放大 23% 才和实机一样大（0.977 / 0.845）；
* ``ICON_SCALE  = 0.62`` —— 图标相对**框画布**的比例。

早期按 ``128/200 = 0.64`` 推导（把 200 当成框的真实边长）是错的：
文件名 ``_128`` 只是逻辑尺寸，实际贴图 256×256，框的 200 是切图外接框含辉光。
**框没放大**时图案相对框就显得过满，这才是「图标很大」的根因。


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

#: 框素材的**有色环**占其画布多少。7 张框实测 0.840~0.858，取中位 0.845。
FRAME_RING_OF_CANVAS = 0.845
#: 图标画布里图案有效区占多少（30 张图标 alpha 外接框中位）。
ICON_ART_OF_CANVAS = 0.8965

#: 框该画多大：框素材边长 ÷ 格子边长。
#:
#: **这个值要放大到超过 1**，是最反直觉的一点：框素材自带一圈辉光，
#: 真正的彩色环只占画布 84.5%，而实机里环是撑着格子画的（0.977）。
#: 不放大就等于把框画小了，图案相对框就显得过满——这就是「图标很大」的根因。
#:
#: 两个独立标定给同一个答案：几何量 0.977/0.845 = 1.156（下限），
#: 逐像素 NCC 拟合 78 格单峰在 1.24。取 1.24。
FRAME_SCALE = 1.24
#: 图标该画多大：图标画布边长 ÷ 框画布边长。
#:
#: 几何量 0.656 / 1.234 / 0.8965 ≈ 0.59，NCC 拟合单峰 0.62。取 0.62，
#: 此时图案在格子里占 0.62 × 1.24 × 0.8965 = 0.689，实机是 0.656（差 0.03）。
ICON_SCALE = 0.62

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
