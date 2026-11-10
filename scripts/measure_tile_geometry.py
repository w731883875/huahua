"""直接从实机截图量「框占格子多大」「图案占框多大」。

## 思路

不拟合、不猜。实机格子里有两样东西可以量：

* **框**：一圈高饱和、按等级定色的环（白/绿/蓝/紫/橙/红），
  取它的外接框 → 框外径 ``frame_px``；
* **图案**：框内沿以内的有色细节，取它的外接框 → 图案边长 ``art_px``。

格子边长从识别出的棋盘矩形除以 6 得到 → ``cell_px``。

于是：

* ``FRAME_SCALE = frame_px / cell_px``
* ``ICON_SCALE = art_px / frame_px``

## 为什么不用 NCC 拟合

试过，坏在两点：实机图案自带光照/描边，与素材不是同一张；
样本里同一比例下不同家族分数差异远大于比例本身的差异，
于是曲面是平的、最优解贴在搜索边界（0.17 的分，属于噪声）。
几何量取不需要「像」，只需要边界，稳得多。

用法::

    python scripts/measure_tile_geometry.py           # 打印测量结果
    python scripts/measure_tile_geometry.py --apply   # 写回 paint_frame.py
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huahua_assistant import paint_frame, vision  # noqa: E402
from huahua_assistant.capture import read_image  # noqa: E402

SHOTS = ROOT / "jietu"

#: 框环的 HSV 判据。白色框饱和度低，单独用亮度+低饱和处理。
#: (h_lo, h_hi) 单位是 OpenCV 的 0~179。
FRAME_HUES: dict[str, tuple[int, int]] = {
    "绿": (40, 80),
    "蓝": (85, 125),
    "紫": (125, 165),
    "橙": (10, 28),
    "红": (0, 10),
}
#: 环像素的最低饱和度/亮度。辉光会拉低饱和度，所以不能太严。
RING_SAT_MIN = 120
RING_VAL_MIN = 90
#: 图案判据：框内沿以内，亮度或饱和度明显区别于背景。
ART_VAL_MIN = 60
#: 有效样本最少要多少像素，滤掉零星噪声。
MIN_RING_PIXELS = 120
MIN_ART_PIXELS = 160


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _side(box: tuple[int, int, int, int] | None) -> int:
    if box is None:
        return 0
    return max(box[2] - box[0] + 1, box[3] - box[1] + 1)


def _ring_mask(tile: np.ndarray, level: int) -> np.ndarray:
    """按等级取框环。"""
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)
    if level == 1:
        # 白框：低饱和、高亮度。边框是灰白，底色通常更暗。
        return (((s < 60) & (v > 150)).astype(np.uint8) * 255)
    name = paint_frame.LEVEL_COLOR.get(level)
    lo_hi = FRAME_HUES.get(str(name))
    if lo_hi is None:
        # 彩框（Lv7）是多色，任何高饱和环都算
        return (((s > RING_SAT_MIN) & (v > RING_VAL_MIN)).astype(np.uint8) * 255)
    lo, hi = lo_hi
    mask = (h >= lo) & (h <= hi) & (s > RING_SAT_MIN) & (v > RING_VAL_MIN)
    return (mask.astype(np.uint8) * 255)


def _art_mask(tile: np.ndarray, inner_box: tuple[int, int, int, int]) -> np.ndarray:
    """框内沿以内的图案。"""
    x0, y0, x1, y1 = inner_box
    sub = tile[y0 : y1 + 1, x0 : x1 + 1]
    if sub.size == 0:
        return np.zeros(tile.shape[:2], np.uint8)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    mask = ((v > ART_VAL_MIN) | (s > 60)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    full = np.zeros(tile.shape[:2], np.uint8)
    full[y0 : y1 + 1, x0 : x1 + 1] = mask
    return full


def measure_tile(tile: np.ndarray, level: int) -> dict | None:
    """量一个格子。返回 ``{"frame_px", "art_px", "cell_px"}``。"""
    if tile is None or tile.size == 0:
        return None
    cell_px = min(tile.shape[:2])
    ring = _ring_mask(tile, level)
    if int((ring > 0).sum()) < MIN_RING_PIXELS:
        return None
    frame_box = _bbox(ring > 0)
    frame_px = _side(frame_box)
    if frame_px < cell_px * 0.45:
        # 环没连着，只量到一小段
        return None
    # 框内沿：环的外接框再往里收一圈（环宽约 6%，取 0.80 求稳）
    cx = (frame_box[0] + frame_box[2]) / 2.0
    cy = (frame_box[1] + frame_box[3]) / 2.0
    half = frame_px * 0.80 / 2.0
    inner = (
        int(round(cx - half)),
        int(round(cy - half)),
        int(round(cx + half)),
        int(round(cy + half)),
    )
    inner = (
        max(0, inner[0]),
        max(0, inner[1]),
        min(tile.shape[1] - 1, inner[2]),
        min(tile.shape[0] - 1, inner[3]),
    )
    art = _art_mask(tile, inner)
    if int((art > 0).sum()) < MIN_ART_PIXELS:
        return None
    art_box = _bbox(art > 0)
    art_px = _side(art_box)
    if art_px <= 0 or art_px > frame_px:
        return None
    return {
        "frame_px": frame_px,
        "art_px": art_px,
        "cell_px": cell_px,
        "frame_scale": frame_px / float(cell_px),
        "icon_scale": art_px / float(frame_px),
    }


def collect() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(SHOTS.glob("*.png")):
        try:
            image = read_image(path)
        except Exception:
            continue
        if image is None or image.size == 0:
            continue
        try:
            scene = vision.analyze_scene_realtime(image)
        except Exception:
            continue
        board = getattr(scene, "board", None)
        rect = getattr(board, "rect", None)
        if rect is None:
            continue
        for tile_obs in getattr(board, "tiles", ()):
            family = getattr(tile_obs, "paint_subtype", None)
            level = getattr(tile_obs, "paint_level", None)
            if not family or not level or str(family) == paint_frame.VOIDED:
                continue
            tr = tile_obs.rect
            tile = image[tr.y : tr.bottom, tr.x : tr.right]
            result = measure_tile(tile, int(level))
            if result is None:
                continue
            result["file"] = path.name
            result["family"] = str(family)
            result["level"] = int(level)
            rows.append(result)
    return rows


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "median": round(float(statistics.median(values)), 4),
        "mean": round(float(statistics.fmean(values)), 4),
        "stdev": round(float(statistics.pstdev(values)), 4) if len(values) > 1 else 0.0,
    }


def apply_scale(frame_scale: float, icon_scale: float) -> None:
    path = ROOT / "huahua_assistant" / "paint_frame.py"
    text = path.read_text(encoding="utf-8")
    if re.search(r"^FRAME_SCALE = ", text, flags=re.M):
        text = re.sub(r"^FRAME_SCALE = .*$", f"FRAME_SCALE = {frame_scale}", text, flags=re.M)
    else:
        text = re.sub(
            r"^ICON_SCALE = .*$",
            f"FRAME_SCALE = {frame_scale}\nICON_SCALE = {icon_scale}",
            text,
            flags=re.M,
        )
        return path.write_text(text, encoding="utf-8")
    text = re.sub(r"^ICON_SCALE = .*$", f"ICON_SCALE = {icon_scale}", text, flags=re.M)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--by-level", action="store_true", help="按等级分组看")
    args = parser.parse_args()

    rows = collect()
    if not rows:
        raise SystemExit("没量到任何格子")
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0

    frame_stats = _stats([row["frame_scale"] for row in rows])
    icon_stats = _stats([row["icon_scale"] for row in rows])
    print(f"样本 {len(rows)} 格")
    print(f"框/格子  {frame_stats}")
    print(f"图案/框  {icon_stats}")

    if args.by_level:
        print("\n按等级:")
        levels = sorted({row["level"] for row in rows})
        for level in levels:
            subset = [row for row in rows if row["level"] == level]
            if len(subset) < 2:
                continue
            print(
                f"  Lv{level} n={len(subset):>2} "
                f"fr={statistics.median([s['frame_scale'] for s in subset]):.3f} "
                f"is={statistics.median([s['icon_scale'] for s in subset]):.3f}"
            )

    print("\n对照：素材原生比 128/200 = 0.640；背包实测 51/80 = 0.638")
    if args.apply:
        apply_scale(frame_stats["median"], icon_stats["median"])
        print(f"已写回 FRAME_SCALE={frame_stats['median']} ICON_SCALE={icon_stats['median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
