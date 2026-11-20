"""棋盘 / 背包格子画成游戏里那个样子：官方品质框 + 官方家族图案。

## 为什么不再用文字

以前格子里写 ``M5`` / ``G3`` / ``T2``：能算，但**没法核**。用户要的是「一眼看出
识别错没错」——图案对不对、等级对不对，只有把游戏里那张图放回去才能比。
文字标签天生做不到这件事，还占掉了格子里最值钱的那块面积。

## 素材与比例都是官方的，不是拟合的

* 图案：``assets/paint_icons/UI_S8Gameplay_{ColorMonster,ColorPlay}N_128.png``
  （30 张 256×256，从 S8 的 PAK 解出，与 tlidb 图标逐像素比过）；
* 品质框：``assets/paint_frames/{circle,diamond}_Lv{1..7}_{色}.png``
  （圆形=怪物、菱形=玩法，见 :mod:`paint_frame`）；
* 图案占框的比例：``paint_frame.ICON_SCALE = 128/192``，来自 ``S8Paint.uexp``
  里成对出现的两个 SizeBox，不是从截图反推的。

## 「虚空」是只有框、没有图案

颜料放进虚空后图案会被整个抹掉，只剩一圈彩色框（:func:`paint_frame.looks_voided`）。
所以这种格子 :func:`compose_icon` 只画框、中心留空——用户一眼就能认出是虚空形态，
而不是「图案识别失败」。框色此时代表**初始颜料**的等级，不是这格自己的等级，
这一点由调用方在角标上体现（见 ``app._draw_grid``）。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from . import paint_frame, paint_kind

ICON_DIR = Path(__file__).resolve().parent / "assets" / "paint_icons"

#: 「放进虚空」状态用的图标名（``roles.json`` 里 ``role=voided`` 的那张）。
#: 只在需要画「虚空形态长什么样」时用；:func:`compose_icon` 的虚空分支**不贴图案**，
#: 因为游戏里那一格中心本来就是空的。
VOIDED_ICON = "UI_S8Gameplay_ColorPlay2_128"

#: 「放进虚空」的颜料在视觉层返回的子类型名（``paint_frame.VOIDED``）。
#: 它不是家族，目录里查不到图标；这种格子只画框、中心留空。
VOIDED_SUBTYPE = paint_frame.VOIDED

#: 图案相对品质框的边长比（官方 128 / 192）。
ICON_OF_FRAME = paint_frame.ICON_SCALE

#: 合成缓存上限。棋盘 36 格 + 背包 16 格，同一盘里家族/等级组合远少于这个数；
#: 换局后旧组合留着也不亏，攒满就丢最早的那批。
CACHE_LIMIT = 512

_CACHE: dict[tuple, Image.Image] = {}


def icon_stem(subtype: str | None) -> str | None:
    """家族名 → 图标文件名（不含扩展名）。不在官方目录里返回 ``None``。

    ``paint_catalog.json`` 是唯一对照表——图标名只在那里写一次，
    别在这里另抄一份（抄了就会跟目录变更脱节）。
    """
    key = paint_kind.family_key(subtype)
    if key is None:
        return None
    entry = paint_kind.catalog().get(key) or {}
    icon = entry.get("icon")
    return str(icon) if icon else None


def frame_shape(category: str | None) -> str | None:
    """类别 → 框形状（``monster`` → 圆形，``gameplay`` → 菱形）。"""
    return paint_frame.shape_of_category(category)


def family_name(subtype: str | None) -> str | None:
    """家族名归一（目录里的规范名）；不在目录里返回 ``None``。"""
    key = paint_kind.family_key(subtype)
    if key is None:
        return None
    entry = paint_kind.catalog().get(key) or {}
    name = entry.get("name")
    return str(name) if name else None


def _open_rgba(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGBA")
    except (OSError, ValueError):
        return None


def _sanitize_level(level: int | None) -> int | None:
    if level is None:
        return None
    try:
        value = int(level)
    except (TypeError, ValueError):
        return None
    return value if value in paint_frame.LEVEL_COLOR else None


def compose_icon(
    category: str | None,
    subtype: str | None,
    level: int | None,
    *,
    size: int,
    voided: bool = False,
) -> Image.Image | None:
    """合成一格图：品质框 + 家族图案。返回 ``size×size`` 的 RGBA，画不出时 ``None``。

    ``voided=True`` 只画框（图案已被虚空抹掉）。``level`` 不在 1~7 时只画图案不画框——
    宁可不给品质框，也不要随手挑一个颜色冒充等级。
    """
    size = int(size)
    if size < 8:
        return None
    shape = frame_shape(category)
    stem = icon_stem(subtype)
    level = _sanitize_level(level)
    if shape is None and level is not None:
        # 类别未知但有等级：形状判不出来，只能靠目录里的 category 反查一次。
        category_from_catalog = paint_kind.category_of(subtype)
        shape = frame_shape(category_from_catalog)
    if voided:
        stem = None
    if shape is None and stem is None:
        return None

    key = (shape, level, stem, size, bool(voided))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if shape is not None and level is not None:
        frame = _open_rgba(paint_frame.frame_path(shape, level))
        if frame is not None:
            frame = frame.resize((size, size), Image.LANCZOS)
            canvas.alpha_composite(frame)
    if stem is not None:
        icon = _open_rgba(ICON_DIR / f"{stem}.png")
        if icon is not None:
            side = max(6, int(round(size * ICON_OF_FRAME)))
            icon = icon.resize((side, side), Image.LANCZOS)
            offset = (size - side) // 2
            canvas.alpha_composite(icon, (offset, offset))
    if shape is None and stem is None:
        return None
    if len(_CACHE) >= CACHE_LIMIT:
        for stale in list(_CACHE)[: CACHE_LIMIT // 4]:
            _CACHE.pop(stale, None)
    _CACHE[key] = canvas
    return canvas


def clear_cache() -> None:
    """测试与热更新用：把合成缓存清空。"""
    _CACHE.clear()
