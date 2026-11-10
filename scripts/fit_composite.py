"""定稿标定：整格 NCC 直接评判「合成得像不像实机」。

## 为什么还要这一步

前面两个量（框/格子、图案/格子）是分开量的，各自都有饱和与裁切的问题：
框撑满格子时外径会被格子边界切住，量出来永远约 0.98，真实值可能更大。
分离量法没法解决这个，只能靠**整格比对**——把合成图与实机格子逐像素对齐后算
NCC，谁分高就是谁。这也是最终目的：蒙版要能通过。

## 做法

对每个实机格子（家族+等级已知）：

1. 按候选 ``(FRAME_SCALE, ICON_SCALE)`` 合成；
2. 缩到格子边长，与实机格子算零均值 NCC（全格，含框与辉光）；
3. 全部格子取中位，扫网格取最高。

用法::

    python scripts/fit_composite.py                # 扫网格
    python scripts/fit_composite.py --apply        # 写回 paint_frame.py
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

ASSETS = ROOT / "huahua_assistant" / "assets"
ICONS = ASSETS / "paint_icons"
FRAMES = ASSETS / "paint_frames"
CATALOG = ASSETS / "paint_catalog.json"
SHOTS = ROOT / "jietu"

SIDE = 128
FRAME_CANDIDATES = tuple(round(0.95 + 0.02 * i, 3) for i in range(21))   # 0.95~1.35
ICON_CANDIDATES = tuple(round(0.52 + 0.01 * i, 3) for i in range(25))    # 0.52~0.76


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"读不出图片: {path}")
    return image


def _rgba(image: np.ndarray) -> np.ndarray:
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


def _paste(canvas: np.ndarray, image: np.ndarray, ratio: float) -> np.ndarray:
    side = max(4, int(round(SIDE * ratio)))
    resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
    offset = (SIDE - side) // 2
    if offset >= 0:
        canvas[offset : offset + side, offset : offset + side] = resized
    else:
        cut = -offset
        canvas[:, :] = resized[cut : cut + SIDE, cut : cut + SIDE]
    return canvas


def _composite(icon: np.ndarray, frame: np.ndarray, fr: float, is_: float, bg: tuple[int, int, int]) -> np.ndarray:
    frame_layer = _paste(np.zeros((SIDE, SIDE, 4), np.uint8), frame, fr)
    icon_layer = _paste(np.zeros((SIDE, SIDE, 4), np.uint8), icon, fr * is_)
    composed = _over(icon_layer, frame_layer)
    base = np.zeros((SIDE, SIDE, 4), np.uint8)
    base[:, :, :3] = bg
    base[:, :, 3] = 255
    return _over(base, composed)[:, :, :3]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator < 1e-6:
        return -1.0
    return float((x * y).sum() / denominator)


def _samples(limit: int = 90) -> list[dict]:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in payload.values()}
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
            if float(getattr(tile_obs, "paint_category_confidence", 0.0) or 0.0) < 0.82:
                continue
            entry = by_name.get(str(family))
            if entry is None:
                continue
            rect = tile_obs.rect
            tile = image[rect.y : rect.bottom, rect.x : rect.right]
            if tile.size == 0 or min(tile.shape[:2]) < 40:
                continue
            shape = paint_frame.shape_of_category(entry["category"])
            frame_path = FRAMES / f"{shape}_Lv{int(level)}_{paint_frame.LEVEL_COLOR[int(level)]}.png"
            icon_path = ICONS / f"{entry['icon']}.png"
            if not frame_path.exists() or not icon_path.exists():
                continue
            # 背景色取格子四角的均值，抵消不同截图的底色差异
            corners = np.concatenate(
                [
                    tile[:6, :6].reshape(-1, 3),
                    tile[:6, -6:].reshape(-1, 3),
                    tile[-6:, :6].reshape(-1, 3),
                    tile[-6:, -6:].reshape(-1, 3),
                ]
            )
            out.append(
                {
                    "family": str(family),
                    "level": int(level),
                    "tile": cv2.resize(tile, (SIDE, SIDE), interpolation=cv2.INTER_AREA),
                    "icon": _rgba(_read(icon_path)),
                    "frame": _rgba(_read(frame_path)),
                    "bg": tuple(int(v) for v in corners.mean(axis=0)),
                }
            )
            if len(out) >= limit:
                return out
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    samples = _samples()
    if not samples:
        raise SystemExit("没取到实机格子")
    print(f"样本 {len(samples)} 格")

    results: list[dict] = []
    for fr in FRAME_CANDIDATES:
        for is_ in ICON_CANDIDATES:
            scores = [
                _ncc(_composite(s["icon"], s["frame"], fr, is_, s["bg"]), s["tile"])
                for s in samples
            ]
            results.append(
                {"frame_scale": fr, "icon_scale": is_, "ncc": round(float(statistics.median(scores)), 5)}
            )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
        return 0

    top = sorted(results, key=lambda row: -row["ncc"])[:10]
    print(f"{'FRAME':>7}{'ICON':>7}{'整格NCC':>10}")
    for row in top:
        print(f"{row['frame_scale']:>7.3f}{row['icon_scale']:>7.3f}{row['ncc']:>10.4f}")

    # 分别固定一个量看另一个的单峰，确认不是噪声
    best = top[0]
    print("\n当前常量整格 NCC:", end=" ")
    current = [
        _ncc(_composite(s["icon"], s["frame"], paint_frame.FRAME_SCALE, paint_frame.ICON_SCALE, s["bg"]), s["tile"])
        for s in samples
    ]
    print(f"{statistics.median(current):.4f}  (FRAME={paint_frame.FRAME_SCALE}, ICON={paint_frame.ICON_SCALE})")

    if args.apply:
        path = ROOT / "huahua_assistant" / "paint_frame.py"
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^FRAME_SCALE = .*$", f"FRAME_SCALE = {best['frame_scale']}", text, flags=re.M)
        text = re.sub(r"^ICON_SCALE = .*$", f"ICON_SCALE = {best['icon_scale']}", text, flags=re.M)
        path.write_text(text, encoding="utf-8")
        print(f"已写回 FRAME_SCALE={best['frame_scale']} ICON_SCALE={best['icon_scale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
