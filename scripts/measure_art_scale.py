"""直接量：图案在实机格子里多大。

## 做法

同一张图案，在实机截图里一定出现一次。于是把图标（RGBA）缩放到候选尺寸，
**带 alpha 蒙版**在实机格子上做 ``TM_CCORR_NORMED`` 匹配：

* 只比图案本身的像素（alpha 之外的透明区不参与），所以格子的底色、
  框的辉光都影响不到分数；
* 对每个候选尺寸取全格最高分，分数的峰就是图案的真实尺寸。

再除以同一格的框外径，得到 ``图案 / 框``。

## 和前几次的区别

前面两次拟合都失败在「拿合成图去比实机」——合成图里框是画上去的，
框画多大直接影响分数，两个未知量搅在一起，曲面就平了。
这次**只比图案**，框完全不参与，是一维搜索。

用法::

    python scripts/measure_art_scale.py           # 打印曲线与结论
    python scripts/measure_art_scale.py --apply   # 写回 paint_frame.py
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

ICONS = ROOT / "huahua_assistant" / "assets" / "paint_icons"
CATALOG = ROOT / "huahua_assistant" / "assets" / "paint_catalog.json"
SHOTS = ROOT / "jietu"

#: 图案边长占格子边长的候选。实机格子 87px，图案约 50~80px。
RATIOS = tuple(round(0.40 + 0.0125 * i, 4) for i in range(41))
#: 匹配时把格子放大到这么大再搜，避免小尺寸下采样把峰抹平。
UPSCALE = 3


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"读不出图片: {path}")
    return image


def _art_of(icon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """裁掉图标画布四周的空白，返回 ``(图案 BGR, 蒙版)``。

    画布 256×256，图案有效区只占约 91%，直接拿去匹配会把空白也算进去。
    """
    alpha = icon[:, :, 3]
    ys, xs = np.where(alpha > 24)
    if xs.size == 0:
        return icon[:, :, :3], np.full(alpha.shape, 255, np.uint8)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return icon[y0:y1, x0:x1, :3], alpha[y0:y1, x0:x1]


def _catalog_names() -> set[str]:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    return {entry["icon"] for entry in payload.values()}


def _samples(limit: int = 60) -> list[dict]:
    """取「家族已确认」的实机格子。"""
    by_name = {entry["name"]: entry for entry in json.loads(CATALOG.read_text(encoding="utf-8")).values()}
    out: list[dict] = []
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
        if board is None or getattr(board, "rect", None) is None:
            continue
        for tile_obs in getattr(board, "tiles", ()):
            family = getattr(tile_obs, "paint_subtype", None)
            level = getattr(tile_obs, "paint_level", None)
            if not family or not level or str(family) == paint_frame.VOIDED:
                continue
            if float(getattr(tile_obs, "paint_category_confidence", 0.0) or 0.0) < 0.80:
                continue
            entry = by_name.get(str(family))
            if entry is None:
                continue
            rect = tile_obs.rect
            tile = image[rect.y : rect.bottom, rect.x : rect.right]
            if tile.size == 0 or min(tile.shape[:2]) < 40:
                continue
            out.append(
                {
                    "file": path.name,
                    "family": str(family),
                    "level": int(level),
                    "icon": entry["icon"],
                    "category": entry["category"],
                    "tile": tile,
                }
            )
            if len(out) >= limit:
                return out
    return out


def _frame_outer_px(tile: np.ndarray, level: int) -> float:
    """框外径（像素）。按等级取环色，取外接框。"""
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)
    if level == 1:
        mask = (s < 60) & (v > 150)
    else:
        name = str(paint_frame.LEVEL_COLOR.get(level, "彩"))
        hue = {"绿": (40, 80), "蓝": (85, 125), "紫": (125, 165), "橙": (10, 28), "红": (0, 10)}.get(name)
        if hue is None:
            mask = (s > 120) & (v > 90)
        else:
            mask = (h >= hue[0]) & (h <= hue[1]) & (s > 120) & (v > 90)
    # 只留外圈：把中心 55% 挖掉，避免图案里的同色像素被算进来
    height, width = tile.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    inner = (np.abs(xx - cx) / cx <= 0.55) & (np.abs(yy - cy) / cy <= 0.55)
    mask = mask & ~inner
    ys, xs = np.where(mask)
    if xs.size < 60:
        return 0.0
    return float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def best_ratio(tile: np.ndarray, art_bgr: np.ndarray, art_mask: np.ndarray) -> tuple[float, float]:
    """返回 ``(图案边长/格子边长, 该比例下的匹配分)``。"""
    cell = min(tile.shape[:2])
    big = cv2.resize(tile, (cell * UPSCALE, cell * UPSCALE), interpolation=cv2.INTER_CUBIC)
    big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32)
    best = (0.0, -1.0)
    for ratio in RATIOS:
        side = int(round(cell * ratio * UPSCALE))
        if side < 12 or side >= big_gray.shape[0]:
            continue
        template = cv2.resize(art_bgr, (side, side), interpolation=cv2.INTER_AREA)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mask = cv2.resize(art_mask, (side, side), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        if float((mask > 0.5).mean()) < 0.10:
            continue
        try:
            result = cv2.matchTemplate(big_gray, template_gray, cv2.TM_CCORR_NORMED, mask=mask)
        except cv2.error:
            continue
        result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
        score = float(result.max())
        if score > best[1]:
            best = (ratio, score)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    samples = _samples()
    if not samples:
        raise SystemExit("没取到实机格子")
    print(f"样本 {len(samples)} 格")

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict] = []
    for item in samples:
        name = item["icon"]
        if name not in cache:
            cache[name] = _art_of(_read(ICONS / f"{name}.png"))
        art_bgr, art_mask = cache[name]
        ratio, score = best_ratio(item["tile"], art_bgr, art_mask)
        if ratio <= 0 or score < 0.5:
            continue
        frame_px = _frame_outer_px(item["tile"], item["level"])
        cell_px = float(min(item["tile"].shape[:2]))
        rows.append(
            {
                "file": item["file"],
                "family": item["family"],
                "level": item["level"],
                "ratio": ratio,
                "score": round(score, 4),
                "frame_scale": round(frame_px / cell_px, 4) if frame_px else None,
                "art_over_frame": round(ratio * cell_px / frame_px, 4) if frame_px else None,
            }
        )

    if not rows:
        raise SystemExit("没有一条通过阈值")
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0

    ratios = [row["ratio"] for row in rows]
    print(f"图案/格子  median={statistics.median(ratios):.4f} mean={statistics.fmean(ratios):.4f} n={len(ratios)}")
    over = [row["art_over_frame"] for row in rows if row["art_over_frame"]]
    if over:
        print(f"图案/框    median={statistics.median(over):.4f} mean={statistics.fmean(over):.4f} n={len(over)}")
    frames = [row["frame_scale"] for row in rows if row["frame_scale"]]
    if frames:
        print(f"框/格子    median={statistics.median(frames):.4f} n={len(frames)}")

    print("\n按等级:")
    for level in sorted({row["level"] for row in rows}):
        subset = [row for row in rows if row["level"] == level]
        if len(subset) < 2:
            continue
        print(
            f"  Lv{level} n={len(subset):>2} "
            f"图案/格子={statistics.median([r['ratio'] for r in subset]):.3f} "
            f"图案/框={statistics.median([r['art_over_frame'] for r in subset if r['art_over_frame']] or [0]):.3f}"
        )
    print("\n分数分布:", f"min={min(r['score'] for r in rows):.3f}", f"max={max(r['score'] for r in rows):.3f}")

    if args.apply and over:
        # 合成时图标画布缩到 scale，图案在画布内占 0.91（见 _art_of 说明），
        # 所以「图案/框」目标值要除掉 0.91 才是画布比。
        target_art = statistics.median(over)
        canvas_ratio = target_art / 0.9102
        path = ROOT / "huahua_assistant" / "paint_frame.py"
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^ICON_SCALE = .*$", f"ICON_SCALE = {round(canvas_ratio, 4)}", text, flags=re.M)
        path.write_text(text, encoding="utf-8")
        print(f"\n已写回 ICON_SCALE={round(canvas_ratio, 4)}（图案/框 {target_art:.4f} ÷ 画布占比 0.9102）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
