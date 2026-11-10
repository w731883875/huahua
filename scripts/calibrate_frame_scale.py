"""二维标定：框在格子里多大（fr）× 图标相对框多大（is）。

## 为什么是二维

合成一张格子的图，有两个互相独立的量：

* **fr** —— 框素材的边长 ÷ 格子边长。实机里框是撑着格子画的，
  而素材本身带辉光、外沿不满边，所以 fr 不等于 1。
* **is** —— 图标边长 ÷ 框素材边长。

只搜一个必然错：把 fr 固定成 1.0 时，为了让图案显得够大，就只能把 is 顶到 0.75；
真实情况是框该放大、is 该留在原生比。

## 判据

只在**框内沿的实心区域**里算零均值 NCC（:func:`_inner_mask`）。框环本体、
外圈辉光、右下角 LvN 角标都不参与——它们跟图标比例无关，掺进来只会把
「框画多大」的信号盖过去。

## 用法::

    python scripts/calibrate_frame_scale.py            # 打印网格
    python scripts/calibrate_frame_scale.py --apply    # 写回 paint_frame.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huahua_assistant import paint_frame  # noqa: E402
from huahua_assistant.capture import read_image  # noqa: E402

ICONS = ROOT / "huahua_assistant" / "assets" / "paint_icons"
CATALOG = ROOT / "huahua_assistant" / "assets" / "paint_catalog.json"

#: 比对边长。
SIDE = 128
#: 框边长候选。0.85~1.25。
FR_CANDIDATES = tuple(round(0.85 + 0.025 * i, 3) for i in range(17))
#: 图标相对框的候选。0.50~0.90。
IS_CANDIDATES = tuple(round(0.50 + 0.025 * i, 3) for i in range(17))


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"读不出图片: {path}")
    return image


def _to_bgra(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2] + (1,), 255, np.uint8)
        return np.concatenate([image, alpha], axis=2)
    return image


def _over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    a = top[:, :, 3:4].astype(np.float32) / 255.0
    out = top[:, :, :3].astype(np.float32) * a + bottom[:, :, :3].astype(np.float32) * (1.0 - a)
    alpha = top[:, :, 3:4].astype(np.float32) + bottom[:, :, 3:4].astype(np.float32) * (1.0 - a)
    return np.concatenate([out, np.clip(alpha, 0, 255)], axis=2).astype(np.uint8)


def _gray(image: np.ndarray) -> np.ndarray:
    bgr = image[:, :, :3] if image.ndim == 3 and image.shape[2] == 4 else image
    if bgr.ndim == 3:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr.astype(np.float32)


def _paste(canvas: np.ndarray, image: np.ndarray, scale: float) -> np.ndarray:
    side = max(4, int(round(SIDE * scale)))
    resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
    offset = (SIDE - side) // 2
    if offset >= 0:
        canvas[offset : offset + side, offset : offset + side] = resized
    else:
        cut = -offset
        canvas[:, :] = resized[cut : cut + SIDE, cut : cut + SIDE]
    return canvas


def _composite(icon: np.ndarray, frame: np.ndarray, fr: float, is_: float) -> np.ndarray:
    frame_canvas = _paste(np.zeros((SIDE, SIDE, 4), np.uint8), frame, fr)
    icon_canvas = _paste(np.zeros((SIDE, SIDE, 4), np.uint8), icon, fr * is_)
    return _over(icon_canvas, frame_canvas)


def _inner_mask(frame: np.ndarray) -> np.ndarray:
    """框内沿的实心区域。

    框素材中心是透明的（图案就画在那儿），所以 ``alpha < 0.35`` 就是内沿以内。
    再往里收 6%，避开内沿那圈抗锯齿边——那里一半是框、一半是图案，
    比例一动它就跟着动，留着会把信号变成噪声。
    """
    alpha = frame[:, :, 3].astype(np.float32) / 255.0
    inner = (alpha < 0.35).astype(np.uint8)
    # 只保留最大的连通块，去掉零散的抗锯齿孔
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inner, 8)
    if count > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        inner = (labels == biggest).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner = cv2.erode(inner, kernel, iterations=max(1, SIDE // 40))
    return inner.astype(np.float32)


def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 0.5
    if int(sel.sum()) < 64:
        return -1.0
    x = a[sel].astype(np.float64) - float(a[sel].mean())
    y = b[sel].astype(np.float64) - float(b[sel].mean())
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator < 1e-6:
        return -1.0
    return float((x * y).sum() / denominator)


def _truth_grid(path: Path) -> list[tuple[int, int, str, int]]:
    """从识别结果里取「家族已确认」的格子：``(row, col, family, level)``。"""
    image = read_image(path)
    if image is None or image.size == 0:
        return []
    try:
        board = vision_board(image)
    except Exception:
        return []
    if board is None:
        return []
    rect, cells = board
    out: list[tuple[int, int, str, int]] = []
    for (row, col), (family, level) in cells.items():
        if family:
            out.append((row, col, family, level))
    return out, rect, image


def vision_board(image: np.ndarray):
    """识别一次，取「家族已确认」的格子。

    家族名在 ``paint_subtype``、等级在 ``paint_level``（虚空格只有等级、
    没有家族，正好被跳过——它本来就没有图案可比）。
    """
    from huahua_assistant import vision

    scene = vision.analyze_scene_realtime(image)
    board = getattr(scene, "board", None)
    rect = getattr(board, "rect", None)
    if rect is None or rect.w < 200 or rect.h < 200:
        return None
    cells: dict[tuple[int, int], tuple[str, int]] = {}
    for tile in getattr(board, "tiles", ()):
        family = getattr(tile, "paint_subtype", None)
        level = getattr(tile, "paint_level", None)
        if not family or not level:
            continue
        if str(family) == paint_frame.VOIDED:
            continue
        confidence = float(getattr(tile, "paint_category_confidence", 0.0) or 0.0)
        if confidence < 0.75:
            continue
        cells[(int(tile.row), int(tile.col))] = (str(family), int(level))
    return rect, cells


def _catalog_by_icon() -> dict[str, dict]:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    return {entry["icon"]: entry for entry in payload.values()}


def _sample_tiles(limit: int = 80) -> list[tuple[np.ndarray, dict, int]]:
    """实机格子 + 它对应的家族条目 + 等级。"""
    by_name = {entry["name"]: entry for entry in _catalog_by_icon().values()}
    shots = ROOT / "jietu"
    samples: list[tuple[np.ndarray, dict, int]] = []
    for path in sorted(shots.glob("*.png")):
        try:
            image = read_image(path)
        except Exception:
            continue
        if image is None or image.size == 0:
            continue
        board = vision_board(image)
        if board is None:
            continue
        rect, cells = board
        cw, ch = rect.w / 6.0, rect.h / 6.0
        for (row, col), (family, level) in cells.items():
            entry = by_name.get(family)
            if entry is None:
                continue
            y0 = int(round(rect.y + row * ch))
            x0 = int(round(rect.x + col * cw))
            tile = image[y0 : int(round(rect.y + (row + 1) * ch)), x0 : int(round(rect.x + (col + 1) * cw))]
            if tile.size == 0 or min(tile.shape[:2]) < 40:
                continue
            samples.append((tile, entry, level))
            if len(samples) >= limit:
                return samples
    return samples


def build_grid() -> tuple[list[dict], list[dict]]:
    samples = _sample_tiles()
    if not samples:
        raise SystemExit("没取到实机格子")
    print(f"实机格子 {len(samples)} 个（家族已确认）")

    frames = paint_frame.frames()
    icon_cache: dict[str, np.ndarray] = {}
    grid: list[dict] = []
    for shape in ("circle", "diamond"):
        shape_samples = [s for s in samples if paint_frame.shape_of_category(s[1]["category"]) == shape]
        if not shape_samples:
            continue
        frame = cv2.resize(
            _to_bgra(_read(frames[(shape, 3)].path)), (SIDE, SIDE), interpolation=cv2.INTER_AREA
        )
        mask = _inner_mask(frame)
        frame_gray = _gray(frame)
        for fr in FR_CANDIDATES:
            for is_ in IS_CANDIDATES:
                if fr * is_ > 1.0:
                    continue
                scores: list[float] = []
                for tile, entry, _level in shape_samples[:40]:
                    name = entry["icon"]
                    if name not in icon_cache:
                        icon_cache[name] = _to_bgra(_read(ICONS / f"{name}.png"))
                    composite = _composite(icon_cache[name], frame, fr, is_)
                    tile_norm = cv2.resize(tile, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
                    scores.append(_masked_ncc(_gray(composite), _gray(tile_norm), mask))
                grid.append(
                    {
                        "shape": shape,
                        "fr": fr,
                        "is": is_,
                        "ncc": round(float(np.mean(scores)), 5),
                        "n": len(shape_samples),
                    }
                )
    return grid, samples


def summarize(grid: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for shape in ("circle", "diamond"):
        subset = [row for row in grid if row["shape"] == shape]
        if not subset:
            continue
        best[shape] = max(subset, key=lambda row: row["ncc"])
    return best


def apply_scale(frame_ratio: float, icon_ratio: float) -> None:
    path = ROOT / "huahua_assistant" / "paint_frame.py"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^FRAME_SCALE = .*$", f"FRAME_SCALE = {frame_ratio}", text, flags=re.M)
    text = re.sub(r"^ICON_SCALE = .*$", f"ICON_SCALE = {icon_ratio}", text, flags=re.M)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    grid, _samples = build_grid()
    if args.json:
        print(json.dumps(grid, ensure_ascii=False, indent=1))
    else:
        for shape in ("circle", "diamond"):
            subset = [row for row in grid if row["shape"] == shape]
            if not subset:
                continue
            print(f"\n=== {shape} 前 8 ===")
            for row in sorted(subset, key=lambda r: -r["ncc"])[:8]:
                print(f"  fr={row['fr']:.3f} is={row['is']:.3f} ncc={row['ncc']:.4f}")
    best = summarize(grid)
    print("\n最优:", json.dumps(best, ensure_ascii=False))
    if args.apply and best:
        fr = round(float(np.mean([b["fr"] for b in best.values()])), 3)
        is_ = round(float(np.mean([b["is"] for b in best.values()])), 3)
        apply_scale(fr, is_)
        print(f"已写回 FRAME_SCALE={fr} ICON_SCALE={is_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
