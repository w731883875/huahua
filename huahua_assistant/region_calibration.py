"""识别区域标定：把棋盘 / 调色范围 / 笔触卡片画成可拖动的绿框。

为什么要有这个模块
------------------
``layout.json`` 用**窗口像素尺寸**当键（1922x1030、1602x939、800x469 …）。
同一局里两条抓图路径拿到的尺寸并不一致（完整识别 1602x939、实时快路径
800x469，正好差 2 倍），于是总有一条永远匹配不上，退回内置默认比例。

2026-11-24 实测证据（用户当夜连开两局、中途切过窗口大小）：

    layout.json 里 7 → 8 条，新增 1602x939（完整识别写入，23:21:56）
    但实时帧一直是 800x469：matched=False fallback=True
    用户看到的现象 =「第一个初始颜料的等级和范围都是错的」

默认比例歪一点也能过 presence 检查，于是**裁偏了还读得很自信**——这正是
"安静地给出错误答案"。用户要的解决办法是：把识别范围画在屏上，自己拖到
正确位置，固定下来。

设计要点
--------
* 区域一律存**相对比例**（相对窗口宽/高），换分辨率自动跟着走；
* 绿框跟随游戏窗口，坐标系与识别器一致（窗口物理像素）；
* 拖动框体平移、拖右下角缩放，都有最小尺寸保护；
* 标定结果单独存 ``runtime/calibration.json``，不碰 ``layout.json``，
  避免和自动标定互相污染。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = ROOT / "runtime" / "calibration.json"

#: 允许标定的区域。键名同时用于 JSON 与界面标签。
REGION_LABELS: dict[str, str] = {
    "board": "棋盘 6x6",
    "scope": "调色范围（初始颜料的生效范围）",
    "brush_stage1": "笔触卡片 1",
    "brush_stage2": "笔触卡片 2",
}

#: 最小像素尺寸，防止误拖成一个点。
MIN_SIDE_PX = 24

#: 必须是正方形的区域。棋盘是 6x6 正方格，比例存成 ``w_ratio`` / ``h_ratio``
#: 会毁掉这个语义：1602x939 下 w=h=558 只是巧合，换到 1536x864 就变成
#: 535x513 的长方形（2026-11-24 端到端测试实测到）。这些键改用「相对短边」
#: 的单一 ``size_ratio``，与 :class:`huahua_assistant.layout.LayoutRect` 一致。
SQUARE_KEYS = frozenset({"board"})


@dataclass(frozen=True)
class Region:
    """一块识别区域，用相对窗口的比例描述。

    与 :class:`huahua_assistant.layout.LayoutRect` 的区别：这里的宽高可以不同
    （笔触卡片是长方形），所以分开存 ``w_ratio`` / ``h_ratio``。
    """

    key: str
    x_ratio: float
    y_ratio: float
    w_ratio: float
    h_ratio: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """换算成窗口像素 ``(x, y, w, h)``。

        正方形区域取 ``min(w, h)``，保证换任何窗口比例都还是正方形。
        """
        x = int(round(self.x_ratio * width))
        y = int(round(self.y_ratio * height))
        if self.is_square:
            # 正方形按短边缩放（与 layout.LayoutRect 一致），否则窗口比例一变，
            # 存的是「相对宽」的边长就会跟着宽度乱涨。
            short = max(1, min(width, height))
            side = max(MIN_SIDE_PX, int(round(self.w_ratio * short)))
            return x, y, side, side
        w = max(MIN_SIDE_PX, int(round(self.w_ratio * width)))
        h = max(MIN_SIDE_PX, int(round(self.h_ratio * height)))
        return x, y, w, h

    @property
    def is_square(self) -> bool:
        return self.key in SQUARE_KEYS

    @classmethod
    def from_pixels(
        cls,
        key: str,
        x: float,
        y: float,
        w: float,
        h: float,
        width: int,
        height: int,
    ) -> "Region":
        """从窗口像素构造，自动夹在窗口内、并保证最小尺寸。"""
        width = max(1, int(width))
        height = max(1, int(height))
        w = max(MIN_SIDE_PX, int(round(w)))
        h = max(MIN_SIDE_PX, int(round(h)))
        x = max(0, min(int(round(x)), width - w))
        y = max(0, min(int(round(y)), height - h))
        if key in SQUARE_KEYS:
            # 正方形：边长统一按短边存，避免 w_ratio / h_ratio 各说各话。
            short = max(1, min(width, height))
            side = max(MIN_SIDE_PX, min(w, h))
            return cls(
                key=key,
                x_ratio=x / width,
                y_ratio=y / height,
                w_ratio=side / short,
                h_ratio=side / short,
            )
        return cls(
            key=key,
            x_ratio=x / width,
            y_ratio=y / height,
            w_ratio=w / width,
            h_ratio=h / height,
        )

    def moved_by(self, dx: float, dy: float, width: int, height: int) -> "Region":
        x, y, w, h = self.to_pixels(width, height)
        return Region.from_pixels(self.key, x + dx, y + dy, w, h, width, height)

    def resized_by(self, dw: float, dh: float, width: int, height: int) -> "Region":
        x, y, w, h = self.to_pixels(width, height)
        if self.is_square:
            # 正方形只认一个边长：取两个方向里变化更大的那个，跟着手走。
            delta = dw if abs(dw) >= abs(dh) else dh
            return Region.from_pixels(self.key, x, y, w + delta, h + delta, width, height)
        return Region.from_pixels(self.key, x, y, w + dw, h + dh, width, height)

    def to_payload(self) -> dict[str, float | str]:
        return {
            "key": self.key,
            "x_ratio": round(self.x_ratio, 6),
            "y_ratio": round(self.y_ratio, 6),
            "w_ratio": round(self.w_ratio, 6),
            "h_ratio": round(self.h_ratio, 6),
        }


def default_regions(width: int, height: int) -> dict[str, Region]:
    """按工程内置默认比例给出四块区域（棋盘用 ``ACTIVE_LAYOUTS[0]``）。"""
    from .layout import ACTIVE_LAYOUTS

    base = ACTIVE_LAYOUTS[0]
    short = max(1, min(width, height))
    board_side = max(MIN_SIDE_PX, int(round(base.size_ratio * short)))
    board = Region.from_pixels(
        "board",
        base.x_ratio * width,
        base.y_ratio * height,
        board_side,
        board_side,
        width,
        height,
    )
    # 调色范围默认与棋盘同框：初始颜料的生效范围就是棋盘里的一片格子。
    scope = replace(board, key="scope")
    # 笔触卡片在背包面板下方，先用「棋盘右下角外一块」占位，等用户拖。
    card_w = max(MIN_SIDE_PX, int(round(board_side * 0.30)))
    card_h = max(MIN_SIDE_PX, int(round(board_side * 0.42)))
    stage1 = Region.from_pixels(
        "brush_stage1",
        base.x_ratio * width + board_side * 0.05,
        base.y_ratio * height + board_side * 1.05,
        card_w,
        card_h,
        width,
        height,
    )
    stage2 = Region.from_pixels(
        "brush_stage2",
        base.x_ratio * width + board_side * 0.40,
        base.y_ratio * height + board_side * 1.05,
        card_w,
        card_h,
        width,
        height,
    )
    return {region.key: region for region in (board, scope, stage1, stage2)}


def load_regions() -> dict[str, Region]:
    """读已固定的标定；文件不存在或坏掉就返回空字典。"""
    if not CALIBRATION_PATH.exists():
        return {}
    try:
        payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, Region] = {}
    for entry in payload.get("regions") or []:
        try:
            key = str(entry["key"])
            if key not in REGION_LABELS:
                continue
            out[key] = Region(
                key=key,
                x_ratio=float(entry["x_ratio"]),
                y_ratio=float(entry["y_ratio"]),
                w_ratio=float(entry["w_ratio"]),
                h_ratio=float(entry["h_ratio"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def save_regions(regions: dict[str, Region], *, width: int, height: int) -> Path:
    """把标定固定下来。同时记下当时的窗口尺寸，便于事后核对。"""
    payload = {
        "note": "识别区域标定（相对窗口比例）。由界面里的『区域标定』保存。",
        "width": int(width),
        "height": int(height),
        "regions": [regions[key].to_payload() for key in REGION_LABELS if key in regions],
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return CALIBRATION_PATH


def resolved(regions: dict[str, Region], width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    """四块区域的窗口像素矩形，缺的用默认值补上。"""
    merged = default_regions(width, height)
    merged.update(regions)
    return {key: merged[key].to_pixels(width, height) for key in REGION_LABELS}


def cells_of(board_px: tuple[int, int, int, int]) -> Iterator[tuple[int, int, int, int]]:
    """把棋盘矩形切成 6x6，逐格产出窗口像素矩形。"""
    x, y, w, h = board_px
    for row in range(6):
        for col in range(6):
            yield (
                x + int(round(col * w / 6.0)),
                y + int(round(row * h / 6.0)),
                max(1, int(round(w / 6.0))),
                max(1, int(round(h / 6.0))),
            )