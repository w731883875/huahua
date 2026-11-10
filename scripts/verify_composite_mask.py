"""校验合成图与实机格子**像素级重合**（蒙版校验）。

## 判的是重合，不是「像」

把合成图和实机格子对齐到同一尺度后逐像素比较，两种指标：

* ``IoU`` —— 两侧都把「显著像素」（图案 + 框，去掉背景）二值化，交集 / 并集。
  这是最直接的「蒙版有没有对上」；1.0 是完全重合。
* ``NCC`` —— 灰度零均值相关，容忍整体色调差异（实机有光照与描边），
  只看结构位置。合成图不可能有实机的光晕纹理，所以不追求 1.0。

期望值：IoU 因描边、辉光、抗锯齿会略低；NCC 应在 0.75 以上。
若尺寸或位置错了，这两项会立刻掉到 0.4 以下——这正是要拦住的。

用法::

    python scripts/verify_composite_mask.py
    python scripts/verify_composite_mask.py --detail
"""

from __future__ import annotations

import argparse
import json
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
CATALOG = ASSETS / "paint_catalog.json"
SHEET = ROOT / "runtime" / "paint_sheet"
SHOTS = ROOT / "jietu"

#: 对齐边长。
SIDE = 160


def _load(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"读不出图片: {path}")
    return image


def _flatten(image: np.ndarray, background: int = 238) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        a = image[:, :, 3:4].astype(np.float32) / 255.0
        base = np.full(image.shape[:2] + (3,), background, np.float32)
        return np.clip(image[:, :, :3] * a + base * (1 - a), 0, 255).astype(np.uint8)
    return image


def _prepare(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (SIDE, SIDE), interpolation=cv2.INTER_AREA)


def _salient(image: np.ndarray) -> np.ndarray:
    """显著像素：既不是背景也不接近背景的暗白。

    用「与格子四角中位色的距离」判断，避免不同截图背景色略有差异导致误判。
    """
    bgr = image.astype(np.float32)
    corners = np.concatenate(
        [
            bgr[:5, :5].reshape(-1, 3),
            bgr[:5, -5:].reshape(-1, 3),
            bgr[-5:, :5].reshape(-1, 3),
            bgr[-5:, -5:].reshape(-1, 3),
        ]
    )
    background = np.median(corners, axis=0)
    distance = np.abs(bgr - background).sum(axis=2)
    return (distance > 60).astype(np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a > 0, b > 0).sum())
    union = int(np.logical_or(a > 0, b > 0).sum())
    return inter / union if union else 0.0


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float((x * y).sum() / denominator) if denominator > 1e-6 else -1.0


def _samples(limit: int = 90) -> list[dict]:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in payload.values()}
    key_of = {entry["name"]: key for key, entry in payload.items()}
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
            sheet_path = SHEET / f"{key_of[str(family)]}_Lv{int(level)}_tile.png"
            if not sheet_path.exists():
                continue
            rect = tile_obs.rect
            tile = image[rect.y : rect.bottom, rect.x : rect.right]
            if tile.size == 0 or min(tile.shape[:2]) < 40:
                continue
            out.append(
                {
                    "family": str(family),
                    "level": int(level),
                    "file": path.name,
                    "generated": _flatten(_load(sheet_path)),
                    "real": tile,
                }
            )
            if len(out) >= limit:
                return out
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", action="store_true", help="逐格列出")
    args = parser.parse_args()

    samples = _samples()
    if not samples:
        raise SystemExit("没取到可比对的格子（先跑 scripts/build_paint_sheet.py）")
    print(f"样本 {len(samples)} 格，对齐到 {SIDE}×{SIDE}")

    ious: list[float] = []
    nccs: list[float] = []
    rows: list[tuple[str, int, float, float]] = []
    for sample in samples:
        gen = _prepare(sample["generated"])
        real = _prepare(sample["real"])
        iou = _iou(_salient(gen), _salient(real))
        ncc = _ncc(gen, real)
        ious.append(iou)
        nccs.append(ncc)
        rows.append((sample["family"], sample["level"], iou, ncc))

    print(f"\nIoU   中位={statistics.median(ious):.4f} 均值={statistics.fmean(ious):.4f} "
          f"最低={min(ious):.4f} 最高={max(ious):.4f}")
    print(f"NCC   中位={statistics.median(nccs):.4f} 均值={statistics.fmean(nccs):.4f} "
          f"最低={min(nccs):.4f} 最高={max(nccs):.4f}")

    bad = [row for row in rows if row[2] < 0.45]
    print(f"\nIoU < 0.45 的 {len(bad)} 格" + (f"：{' / '.join(f'{r[0]}Lv{r[1]}({r[2]:.2f})' for r in bad[:8])}" if bad else "（无）"))

    if args.detail:
        print()
        for family, level, iou, ncc in sorted(rows, key=lambda r: r[2]):
            print(f"  {family:<6} Lv{level}  IoU={iou:.3f}  NCC={ncc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
