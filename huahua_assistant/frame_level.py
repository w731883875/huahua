"""按菱形边框颜色读取棋盘格子等级。

游戏里颜料外框是一圈**菱形描边**，颜色按等级固定：

===========  ==================  =============
等级          边框              实测色相(HSV)
===========  ==================  =============
Lv1           银白（低饱和）      —
Lv2           绿                 71~72
Lv3           青蓝               103~107
Lv4           紫                 144~146
Lv5           橙                 15~17
Lv6           红                 179
===========  ==================  =============

实测 32 个带真值的棋盘格子（``结束`` 20 个 + ``斜对角`` 12 个）：
银色框 ``sat_ratio`` 0.06~0.09，其余等级 0.46~0.93 —— 中间是空的，
所以「银 = Lv1」这一刀极稳；彩色框的色相窗口之间也全都有空隙。

这条比读 ``LvN`` 角标（留一法 63.6%）可靠得多，所以作为等级的第一来源，
角标降级为交叉校验和兜底。

两个必须注意的坑
----------------
1. **淡金色生效范围地块**。笔触/初始颜料的作用范围会给整格盖一层淡金，
   色相 27~30，正好压在 Lv5 橙（15~17）旁边。但淡金是**低饱和**的
   （带内平均饱和度 38~45），真边框是 107~178。用饱和度绝对值就能分开。
2. **只对「确实有颜料」的格子生效**。空白、冰块、漩涡地块没有菱形框，
   强行读会读出噪声。调用方应先确认角标存在。

已知缺口（已补）：**Lv7 的边框是圆形彩虹环**，不像 Lv1~Lv6 那样是菱形描边。
圆环的轴点落在 ``|dx|/cx + |dy|/cy ≈ 0.47``，整圈都在菱形带（0.80~1.02）**内侧**，
所以菱形带只从环上漏进一点点饱和像素：``sat_ratio`` 0.135~0.220。

光看饱和会误伤 Lv1——蓝冰晶这类图案也会往带内漏色（``sat_ratio`` 0.118）。
补上第二条证据就分开了：彩虹环是**多色**的，色相峰值占比只有 0.038~0.042，
而单色图案漏色有明显主色相（0.079~0.086）。

两条一起，在 41 个带真值的格子和 107 个生产格子上零误报，只命中那 2 枚 Lv7。
置信度给 0.72 而不是 0.88——样本只有 2 枚，够越过上层 0.70 的采用门槛即可。
上层 ``vision._detect_board_tile_level`` 会在角标也读出 7 时改采信角标更高的分数。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: 边框带：按菱形度量 ``|dx|/cx + |dy|/cy`` 取环带，天然贴合菱形描边。
FRAME_BAND = (0.80, 1.02)
#: 带内「饱和像素」占比低于此值 → 银色框 → Lv1。实测银色 0.06~0.09，彩色 0.46+。
SILVER_SAT_RATIO_MAX = 0.30
#: 带内平均饱和度低于此值 → 银色框 → Lv1。实测银色 52~59，彩色 107+，
#: 淡金色生效范围 38~45（这一条专门用来挡它）。
SILVER_MEAN_SAT_MAX = 85.0
#: 彩框（Lv7）判据。圆环是**圆形**的，整圈落在菱形带（0.80~1.02）**内侧**，
#: 只从轴点方向漏进一点点饱和，所以 ``sat_ratio`` 卡在夹缝里：
#: 实测样本 0.080 / 0.121 / 0.135 / 0.220，而银框 ≤ 0.118、彩色菱形框 ≥ 0.36。
#: 光看饱和还不够——Lv1 的图案（比如蓝冰晶）也会往带内漏色（sat_ratio 0.118、
#: 色相峰值占比 0.082）。彩虹环是**多色**的，没有主色相，峰值占比只有 0.038/0.042，
#: 而单色图案漏色有明显主色相（0.079~0.086）。两个条件一起才分得干净。
#:
#: 下限从 0.10 压到 0.05（2026-11-24 实机）：彩虹环转到某些相位时整圈饱和都会掉，
#: c3r5(格27) 实测 sat_ratio 只有 0.080、share 0.0476，被银框判据抢走读成 Lv1，
#: 玩家看到「7 级识别成 1 级」。虚空格的 sat_ratio 只有 0.001~0.009，离 0.05 很远，
#: 不会被误伤；全场 36 格里只有两个真 Lv7 落进这一档。
RAINBOW_SAT_RATIO_MIN = 0.05
RAINBOW_HUE_SHARE_MAX = 0.06
#: **几何判据**（比颜色抗动画）：Lv7 的菱形描边是**白色细线 + 外圈彩色辉光**，
#: 白线本身不饱和，所以菱形外圈（菱形度量 0.86~1.0）里几乎没有「亮且饱和」的像素。
#: 实测：Lv7 该比值 0.263，Lv2~Lv6 最低 0.526（都是彩色菱形描边），
#: 银框 Lv1 只有 0.085~0.132（但它饱和度低，走银框判据）。
#: 彩虹环一转，主色相会偏橙（判 Lv5）、饱和会掉（判 Lv1），**但形状不会变**，
#: 所以这条判据不受动画相位影响。
DIAMOND_LIT_RATIO_MIN = 0.34
#: 走几何判据时要求的带内饱和占比下限：把「银白框」排除掉（真银框 ≤0.132）。
RING_SAT_RATIO_MIN = 0.16
#: 几何判据的第二道：软饱和上限。真 Lv7 实测 81，被面板下沿裁掉边框的格子 98~100。
RAINBOW_MEAN_SAT_MAX = 92.0
#: 色相窗口（OpenCV 的 H 是 0~179）。窗口之间留了空隙，落在空隙里就返回 None。
HUE_WINDOWS: dict[int, tuple[int, int]] = {
    2: (60, 88),
    3: (95, 122),
    4: (132, 162),
    5: (10, 24),
    6: (168, 180),
}
#: 色相峰值的圆周平滑核，避免红色跨 0/180 边界被切开。
_HUE_SMOOTH_KERNEL = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=np.float64)
#: 红色跨 0/180 边界：Lv6 的纯红实测色相**同时**落在 168~180 和 0~8。
#:
#: 只在 ``HUE_WINDOWS[6]`` 里写 ``(168, 180)`` 是错的，会把一半纯红判成
#: 「不在任何窗口」——2026-11-24 实机 c5r4 真 Lv6 就是这样：边框像素里
#: hue=179 出现 1020 次、hue=0 出现 1009 次，两边几乎对半，平滑后的峰值
#: 落到 0 → ``_level_for_hue`` 返回 None → 退到角标那一路给了 4，
#: 玩家看到的就是「6 级识别成 4 级」。
RED_HUE_WRAP_MAX = 8


def hue_matches_level(hue: np.ndarray | int, level: int) -> np.ndarray | bool:
    """``hue`` 是否落在 ``level`` 的色相窗口里（Lv6 红色按跨 0/180 处理）。"""
    window = HUE_WINDOWS.get(level)
    if window is None:
        return np.zeros(np.shape(hue), dtype=bool) if isinstance(hue, np.ndarray) else False
    low, high = window
    if level == 6:
        # 红：168~180 与 0~8 是同一段色。
        return (hue >= low) | (hue <= RED_HUE_WRAP_MAX)
    return (hue >= low) & (hue <= high)

#: 带内饱和像素少于此数就不敢报色相。
MIN_SATURATED_PIXELS = 20


@dataclass(frozen=True)
class FrameReading:
    """一次边框颜色读取的结果。"""

    level: int | None
    confidence: float
    #: 带内平均饱和度，用来区分银色框和淡金色覆盖。
    mean_saturation: float
    #: 带内饱和像素占比。
    sat_ratio: float
    #: 色相峰值；读不到时为 -1。
    hue: int
    #: 菱形外圈（度量 0.86~1.0）里「亮且饱和」像素的占比。
    #: 彩色菱形描边 ≥0.5；Lv7 的白细线 + 彩色辉光只有 ~0.26；银框 ~0.1。
    diamond_lit_ratio: float = 1.0


def frame_stats(tile_bgr: np.ndarray) -> tuple[float, float, int, float, float] | None:
    """返回 ``(mean_saturation, sat_ratio, hue, hue_share, diamond_lit_ratio)``。"""
    if tile_bgr.size == 0:
        return None
    h, w = tile_bgr.shape[:2]
    if h < 12 or w < 12:
        return None
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    hue_map = hsv[:, :, 0].astype(np.int32)

    ys, xs = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    diamond = np.abs(ys - cy) / max(cy, 1.0) + np.abs(xs - cx) / max(cx, 1.0)
    band = (diamond >= FRAME_BAND[0]) & (diamond <= FRAME_BAND[1])
    if int(band.sum()) < 40:
        return None

    mean_saturation = float(np.mean(saturation[band]))
    sat_ratio = float(np.mean(saturation[band] > 110))

    outer = (diamond >= 0.86) & (diamond <= 1.0)
    lit = (saturation > 100) & (value > 105)
    diamond_lit_ratio = float(np.mean(lit[outer])) if int(outer.sum()) else 1.0

    saturated = band & (saturation > 110) & (value > 90)
    if int(saturated.sum()) < MIN_SATURATED_PIXELS:
        return mean_saturation, sat_ratio, -1, 0.0, diamond_lit_ratio

    histogram = np.bincount(hue_map[saturated], minlength=180).astype(np.float64)
    padded = np.concatenate([histogram[-9:], histogram, histogram[:9]])
    kernel = _HUE_SMOOTH_KERNEL / _HUE_SMOOTH_KERNEL.sum()
    smoothed = np.convolve(padded, kernel, mode="same")[9:189]
    peak = int(np.argmax(smoothed))
    share = float(smoothed[peak] / max(1.0, smoothed.sum()))
    return mean_saturation, sat_ratio, peak, share, diamond_lit_ratio


def _level_for_hue(hue: int) -> tuple[int, int] | None:
    """返回 ``(level, 距窗口边缘的最小距离)``；不在任何窗口内时返回 None。

    Lv6 的红色跨 0/180 边界，``HUE_WINDOWS[6] = (168, 180)`` 只覆盖了一半，
    另一半（0~8）在这里补上，否则纯红会被判成「不在任何窗口」。
    """
    for level, (low, high) in HUE_WINDOWS.items():
        if low <= hue <= high:
            return level, min(hue - low, high - hue)
    if hue <= RED_HUE_WRAP_MAX:
        # 红色绕回 0 侧：按 Lv6 处理，距离取「离 180 有多远」。
        return 6, 180 - 168 + hue
    return None


def read_frame_level(tile_bgr: np.ndarray) -> FrameReading:
    """读取一个棋盘格子的边框等级。

    调用方必须先确认这格确实有颜料（有 ``LvN`` 角标）。
    """
    stats = frame_stats(tile_bgr)
    if stats is None:
        return FrameReading(level=None, confidence=0.0, mean_saturation=0.0, sat_ratio=0.0, hue=-1)
    mean_saturation, sat_ratio, hue, share, diamond_lit_ratio = stats

    # 彩框（Lv7）第一判据：**几何**。白细线菱形 + 彩色辉光 → 菱形外圈几乎没有
    # 亮且饱和的像素。形状不随动画变，所以这一刀比色相稳。第二道用「柔和不刺眼」
    # 或「没有主色相」二者之一兜住——被面板下沿裁掉边框的格子是 dlit 0.356/0.378
    # 且 mean 98~100、share 0.087~0.096，两道都过不去。
    if (
        diamond_lit_ratio < DIAMOND_LIT_RATIO_MIN
        and sat_ratio >= RING_SAT_RATIO_MIN
        and (mean_saturation < RAINBOW_MEAN_SAT_MAX or share < RAINBOW_HUE_SHARE_MAX)
    ):
        return FrameReading(
            level=7,
            confidence=0.80,
            mean_saturation=mean_saturation,
            sat_ratio=sat_ratio,
            hue=hue,
            diamond_lit_ratio=diamond_lit_ratio,
        )

    # 彩框（Lv7）第二判据：圆形彩虹环，整圈在菱形带内侧，只漏进一点饱和；
    # 而且没有主色相。这一刀必须排在银框判据**前面**——彩虹环的平均饱和度 72~81
    # 也满足「< SILVER_MEAN_SAT_MAX」，放后面就永远读不到 Lv7。
    #
    # ``RAINBOW_SAT_RATIO_MIN`` 从 0.10 放宽到 0.05（2026-11-24 实机 36 格实测）：
    # 彩虹环转到某些相位时整圈饱和都会掉下去，sat_ratio 只剩 0.080（c3r5 格27），
    # 于是被下面那条银框判据抢走，读成 Lv1——玩家看到「7 级识别成 1 级」。
    # 与会掉进这一档的格子只有 c5r2(格11) 和 c3r5(格27) 两个真 Lv7：
    #     格11 sat_ratio 0.121 share 0.0375 dlit 0.169  -> 7
    #     格27 sat_ratio 0.080 share 0.0476 dlit 0.082  -> 1（本次要修的就是它）
    # 虚空格不会被误伤：它们的 sat_ratio 只有 0.001~0.009（远低于 0.05），
    # 而且 share 是 0.0000——单靠 share 判会连虚空一起吃掉，所以下限必须保留，
    # 只是压到 0.05。全场其余 34 格都不落在 [0.05, 0.30) 且 share<0.06 这一档。
    if RAINBOW_SAT_RATIO_MIN <= sat_ratio < SILVER_SAT_RATIO_MAX and share < RAINBOW_HUE_SHARE_MAX:
        return FrameReading(
            level=7,
            confidence=0.72,
            mean_saturation=mean_saturation,
            sat_ratio=sat_ratio,
            hue=hue,
            diamond_lit_ratio=diamond_lit_ratio,
        )

    # 银白框：低饱和。这一刀同时挡掉淡金色生效范围（低饱和的淡金）。
    if sat_ratio < SILVER_SAT_RATIO_MAX and mean_saturation < SILVER_MEAN_SAT_MAX:
        confidence = 0.90 if sat_ratio < 0.15 else 0.78
        return FrameReading(
            level=1,
            confidence=confidence,
            mean_saturation=mean_saturation,
            sat_ratio=sat_ratio,
            hue=hue,
            diamond_lit_ratio=diamond_lit_ratio,
        )
    if hue < 0:
        return FrameReading(
            level=None,
            confidence=0.0,
            mean_saturation=mean_saturation,
            sat_ratio=sat_ratio,
            hue=hue,
            diamond_lit_ratio=diamond_lit_ratio,
        )

    matched = _level_for_hue(hue)
    if matched is None:
        # 有饱和边框但色相不在任何窗口内 —— 要么是还没见过的等级（Lv7），
        # 要么是覆盖层。不猜，交回 None。
        return FrameReading(
            level=None,
            confidence=0.0,
            mean_saturation=mean_saturation,
            sat_ratio=sat_ratio,
            hue=hue,
            diamond_lit_ratio=diamond_lit_ratio,
        )

    level, margin = matched
    confidence = 0.88 if margin >= 3 else 0.72
    if share < 0.06:
        confidence = min(confidence, 0.62)
    return FrameReading(
        level=level,
        confidence=confidence,
        mean_saturation=mean_saturation,
        sat_ratio=sat_ratio,
        hue=hue,
        diamond_lit_ratio=diamond_lit_ratio,
    )
