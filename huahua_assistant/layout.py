"""界面布局：把"棋盘在哪"变成一次坐标计算，而不是每帧做图像搜索。

对应 TartarusSolver 的 ``ScreenLayout.cs``：它先把游戏 UI 量好，
运行时只按窗口宽高换算，不再去找棋盘。

本项目原先把这个换算写死在 ``vision.py`` 的实时链路里（0.187/0.256/0.594 三个
魔数），这里把它提出来，并支持把实测结果存进 ``runtime/layout.json``，
换分辨率时只要标定一次，不用改代码。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from .models import Rect

# 虚拟画布：和 TartarusSolver 一样用一个固定参考尺寸来描述布局。
# 实际换算时棋盘边长按 min(width, height) 缩放，所以画布只用来表达"比例"。
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900

BOARD_SIZE = 6


@dataclass(frozen=True)
class LayoutRect:
    """用比例描述的矩形。

    * ``x_ratio`` 相对窗口宽
    * ``y_ratio`` 相对窗口高
    * ``size_ratio`` 相对 ``min(窗口宽, 窗口高)`` —— 棋盘是正方形，跟短边走
    """

    x_ratio: float
    y_ratio: float
    size_ratio: float

    def resolve(self, width: int, height: int) -> Rect:
        short = max(1, min(width, height))
        size = max(1, int(round(self.size_ratio * short)))
        return Rect(
            x=int(round(self.x_ratio * width)),
            y=int(round(self.y_ratio * height)),
            w=size,
            h=size,
        )


# 实测标定值（1673x878 实机截图，6x6 网格线与格子边界完全重合）。
# 第二组是同一界面在 DPI 缩放 / 窗口裁剪后的偏移补偿：窗口被裁掉一部分时，
# 相对宽度会变，但棋盘相对高度的位置几乎不变，所以留一组备用比例。
ACTIVE_LAYOUTS: tuple[LayoutRect, ...] = (
    LayoutRect(x_ratio=0.187, y_ratio=0.256, size_ratio=0.594),
    LayoutRect(x_ratio=0.174, y_ratio=0.244, size_ratio=0.607),
)
# 等待开画时棋盘居中，比进行中偏右。
START_LAYOUTS: tuple[LayoutRect, ...] = (
    LayoutRect(x_ratio=0.354, y_ratio=0.256, size_ratio=0.594),
    LayoutRect(x_ratio=0.335, y_ratio=0.244, size_ratio=0.607),
)


@dataclass(frozen=True)
class ScreenLayout:
    active: tuple[LayoutRect, ...] = ACTIVE_LAYOUTS
    start: tuple[LayoutRect, ...] = START_LAYOUTS

    def board_rect(self, width: int, height: int, *, drawing: bool = True) -> Rect:
        return self.candidates(width, height, drawing=drawing)[0]

    def candidates(self, width: int, height: int, *, drawing: bool = True) -> tuple[Rect, ...]:
        """按可信度排好序的候选棋盘位置。位置全部算出来，不做搜索。"""
        layouts = self.active if drawing else self.start
        rects: list[Rect] = []
        for layout in layouts:
            rect = layout.resolve(width, height)
            if rect not in rects:
                rects.append(rect)
        return tuple(rects)

    def cell_rects(self, board: Rect) -> tuple[Rect, ...]:
        """把棋盘切成 36 格。位置全部由棋盘矩形算出，不做任何检测。"""
        cell_w = board.w / BOARD_SIZE
        cell_h = board.h / BOARD_SIZE
        rects: list[Rect] = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x1 = int(round(board.x + col * cell_w))
                y1 = int(round(board.y + row * cell_h))
                x2 = int(round(board.x + (col + 1) * cell_w))
                y2 = int(round(board.y + (row + 1) * cell_h))
                rects.append(Rect(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1)))
        return tuple(rects)

    def presence_rect(self, board: Rect) -> Rect:
        """取棋盘中央区域做"是不是目标界面"的特征比对。"""
        inset_x = int(round(board.w * 0.10))
        inset_y = int(round(board.h * 0.10))
        return Rect(
            x=board.x + inset_x,
            y=board.y + inset_y,
            w=max(1, board.w - 2 * inset_x),
            h=max(1, board.h - 2 * inset_y),
        )


#: 最多保留几组「窗口尺寸 → 棋盘位置」的标定。
#: 用户会拖窗口、最大化、换分辨率、改 DPI 缩放，每组各不相同；
#: 只存一组的话换个尺寸就整片失效（2026-11-15 实机事故）。
MAX_CALIBRATIONS = 8
#: 窗口尺寸匹配容差（相对）。2% 能吃掉 DPI 取整差，又不至于把 1920x1080
#: 和 1600x900 认成同一块窗口。
WINDOW_MATCH_TOLERANCE = 0.02

#: 落盘/读取标定时的合理性闸门。
#:
#: 2026-11-19 实机事故：`runtime/layout.json` 里出现了一组
#: ``1922x1030 -> x_ratio=0.0, y_ratio=0.0, size_ratio=0.532``，
#: 也就是「棋盘 = 窗口左上角 (0,0,548,548)」。写入时刻 23:27:51 正好是本局
#: 开局后 4 秒——那会儿画面还在转场，自愈在转场画面上把左上角当成了棋盘。
#: 之后每一帧都按这个框去裁格子：真盘面是 空18/阻挡6/虚空11/颜料1，
#: 程序读到的是 空3/阻挡10/虚空17/颜料6，整个开局全错。
#:
#: 棋盘永远是「窗口中间偏左的一块正方形」，x/y 贴到 0、边长过小或过大的
#: 框只可能是误判。所以读写两边都挡：坏条目读不进来，也写不出去。
MIN_CALIBRATION_X_RATIO = 0.02
MIN_CALIBRATION_Y_RATIO = 0.05
MIN_CALIBRATION_SIZE_RATIO = 0.35
MAX_CALIBRATION_SIZE_RATIO = 0.85


def aspect_key(width: int, height: int) -> str:
    return f"{width / max(1, height):.2f}"


def load_layout(
    width: int | None = None,
    height: int | None = None,
    path: str | Path | None = None,
) -> ScreenLayout:
    """读取标定结果；没有匹配的标定就用内置实测值。

    传了 ``width``/``height`` 就按窗口尺寸挑最接近的那组标定——
    不同尺寸的窗口里棋盘相对位置并不一样，拿 A 尺寸的标定去算 B 尺寸会整体偏十几像素。

    **别改成「按比例外推到任意尺寸」**（2026-11-16 实测否掉）：这个游戏的比例
    不是尺寸无关的。同一台机器的标定里，1920x1020 是 x=0.1755/y=0.2480/边长=0.6127，
    1920x1080 却是 x=0.1906/y=0.2102/边长=0.5537——只差 60px 高度，x 偏移差 8.6%、
    y 偏移差 15%。硬外推会把框算歪，还会让快路径每次多评几个候选框（实测
    一帧 0.28s → 0.90s）。换尺寸的正解是 ``vision._recover_board_rect`` 自愈
    一次然后按新尺寸落盘。
    """
    file = Path(path) if path is not None else _default_path()
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ScreenLayout()
    entry = _pick_calibration(payload, width, height)
    if entry is None:
        return ScreenLayout()
    return ScreenLayout(active=(entry, *ACTIVE_LAYOUTS[1:]), start=START_LAYOUTS)


def save_layout(
    width: int,
    height: int,
    board_rect: Rect,
    path: str | Path | None = None,
) -> Path:
    """把一次可信的实测棋盘位置记下来（按窗口尺寸分开存）。

    同一个窗口尺寸重复标定就覆盖那一组；不同尺寸各自保留，
    所以用户拖窗口/最大化之后不必手动重新标定。
    """
    file = Path(path) if path is not None else _default_path()
    existing = _read_calibrations(file)
    entry = LayoutRect(
        x_ratio=round(board_rect.x / max(1, width), 6),
        y_ratio=round(board_rect.y / max(1, height), 6),
        size_ratio=round(board_rect.w / max(1, min(width, height)), 6),
    )
    if not calibration_is_plausible(entry):
        # 一眼假的框不落盘，也不改动已有内容：落一次错后面每帧都错。
        return file
    kept = [
        item
        for item in existing
        if not _same_window((item[0], item[1]), (width, height))
    ]
    kept.insert(0, (int(width), int(height), entry))
    payload = {
        "canvas": [CANVAS_WIDTH, CANVAS_HEIGHT],
        "calibrations": [
            {
                "width": item_width,
                "height": item_height,
                "x_ratio": item.x_ratio,
                "y_ratio": item.y_ratio,
                "size_ratio": item.size_ratio,
            }
            for item_width, item_height, item in kept[:MAX_CALIBRATIONS]
        ],
    }
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file


def _read_calibrations(file: Path) -> list[tuple[int, int, LayoutRect]]:
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return _payload_entries(payload)


def _payload_entries(payload: object) -> list[tuple[int, int, LayoutRect]]:
    """把文件内容读成 ``[(宽, 高, 比例), ...]``。

    兼容两种历史格式：早期只有一组的 ``{"active": {...}}``（没有尺寸，
    当作「任意窗口」用，排最后），以及现在按尺寸分组的 ``calibrations``。
    """
    if not isinstance(payload, dict):
        return []
    entries: list[tuple[int, int, LayoutRect]] = []
    for item in payload.get("calibrations") or ():
        if not isinstance(item, dict):
            continue
        try:
            rect = LayoutRect(
                x_ratio=float(item["x_ratio"]),
                y_ratio=float(item["y_ratio"]),
                size_ratio=float(item["size_ratio"]),
            )
            width = int(item["width"])
            height = int(item["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if not calibration_is_plausible(rect):
            # 落地时拦一道：历史文件里可能已经躺着「左上角当棋盘」的坏条目。
            continue
        entries.append((width, height, rect))
    legacy = payload.get("active")
    if isinstance(legacy, dict):
        try:
            rect = LayoutRect(
                x_ratio=float(legacy["x_ratio"]),
                y_ratio=float(legacy["y_ratio"]),
                size_ratio=float(legacy["size_ratio"]),
            )
        except (KeyError, TypeError, ValueError):
            rect = None
        if rect is not None and calibration_is_plausible(rect):
            entries.append((0, 0, rect))
    return entries


def _same_window(first: tuple[int, int], second: tuple[int, int]) -> bool:
    if first[0] <= 0 or first[1] <= 0:
        return False
    return (
        abs(first[0] - second[0]) <= max(8, second[0] * WINDOW_MATCH_TOLERANCE)
        and abs(first[1] - second[1]) <= max(8, second[1] * WINDOW_MATCH_TOLERANCE)
    )


def calibration_is_plausible(rect: LayoutRect) -> bool:
    """这块标定像不像「棋盘在窗口里的位置」。

    只挡物理上不可能的：贴到窗口左上角、小到不像棋盘、大到超出窗口一半多。
    真正的对错仍然靠网格分（见 ``vision._recover_board_rect``），这里只是不让
    一眼假的框污染 ``runtime/layout.json``。
    """

    try:
        values = (float(rect.x_ratio), float(rect.y_ratio), float(rect.size_ratio))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in values):
        return False
    x_ratio, y_ratio, size_ratio = values
    if x_ratio < MIN_CALIBRATION_X_RATIO or y_ratio < MIN_CALIBRATION_Y_RATIO:
        return False
    if x_ratio > 1.0 or y_ratio > 1.0:
        return False
    return MIN_CALIBRATION_SIZE_RATIO <= size_ratio <= MAX_CALIBRATION_SIZE_RATIO


def _pick_calibration(
    payload: object,
    width: int | None,
    height: int | None,
) -> LayoutRect | None:
    entries = _payload_entries(payload)
    if not entries:
        return None
    if width is None or height is None:
        return entries[0][2]
    best: tuple[float, LayoutRect] | None = None
    for item_width, item_height, rect in entries:
        if item_width <= 0 or item_height <= 0:
            # 旧格式没尺寸：只在没有更精确的匹配时才用。
            score = 0.5
        else:
            score = abs(item_width - width) / max(1, width) + abs(item_height - height) / max(1, height)
            if score > WINDOW_MATCH_TOLERANCE * 4:
                continue
        if best is None or score < best[0]:
            best = (score, rect)
    return None if best is None else best[1]


def calibrated_layout(width: int, height: int, board_rect: Rect) -> ScreenLayout:
    """不落盘，直接用一次实测结果构造布局（用于测试与即时校正）。"""
    short = max(1, min(width, height))
    active = replace(
        ACTIVE_LAYOUTS[0],
        x_ratio=board_rect.x / max(1, width),
        y_ratio=board_rect.y / max(1, height),
        size_ratio=board_rect.w / short,
    )
    return ScreenLayout(active=(active, *ACTIVE_LAYOUTS[1:]), start=START_LAYOUTS)


def _rect_from_payload(payload: object, fallback: LayoutRect) -> LayoutRect | None:
    if not isinstance(payload, dict):
        return fallback
    try:
        return LayoutRect(
            x_ratio=float(payload["x_ratio"]),
            y_ratio=float(payload["y_ratio"]),
            size_ratio=float(payload["size_ratio"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _default_path() -> Path:
    return Path(__file__).resolve().parent.parent / "runtime" / "layout.json"
