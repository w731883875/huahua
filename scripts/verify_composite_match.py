"""验证「合成模板能不能用」：拿实机格子去查**生产用的那个模板库**。

## 两段结果，只有第一段算数

**【生产口径】** 直接构造 :class:`paint_kind.KindTemplateBank` 并挂上合成模板
（``merge_sheet()``），然后对每个实机格子调 ``bank.read()`` ——
跟软件里跑的**是同一行代码、同一批模板、同一套参数**
（:data:`paint_kind.SHEET_SCALES` 多尺度 1.0/1.15/1.3/1.45、
:data:`paint_kind.SHEET_BACKGROUND` 黑底、双向 NCC、按家族取最高分）。
这一段报的命中率才代表实机表现。

**【旧口径对照】** 早期版本是「浅色底 + 单一尺度 + 只拿自己家族的合成图来打」，
报出来只有 39.7% 通过率。这个数字**不代表现在的能力**，留着只为说明当初为什么
判定「合成图没用」是错的：合成图里框只占画布 75%（官方 ``frame SizeBox=192`` /
画布 256），而游戏里框顶满格子（``paint_frame.TARGET_FRAME_OF_CELL``），
两边中心 54% 搜索窗截到的根本不是同一块内容。放大 1.3 倍才对得上。

用法::

    python scripts/verify_composite_match.py
    python scripts/verify_composite_match.py --threshold 0.68 --legacy
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

from huahua_assistant import paint_frame, paint_kind, vision  # noqa: E402
from huahua_assistant.capture import read_image  # noqa: E402

ASSETS = ROOT / "huahua_assistant" / "assets"
ICONS = ASSETS / "paint_icons"
FRAMES = ASSETS / "paint_frames"
CATALOG = ASSETS / "paint_catalog.json"
SHEET = ROOT / "runtime" / "paint_sheet"
SHOTS = ROOT / "jietu"

#: 合成模板的边长。与 :data:`huahua_assistant.paint_frame.ICON_CANVAS_NATIVE` 同量级。
TEMPLATE_SIDE = 256


def _read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"读不出图片: {path}")
    return image


def _flatten(image: np.ndarray, background: tuple[int, int, int]) -> np.ndarray:
    """把 RGBA 压到指定底色上，得到 BGR。"""
    if image.ndim == 3 and image.shape[2] == 4:
        a = image[:, :, 3:4].astype(np.float32) / 255.0
        base = np.full(image.shape[:2] + (3,), 0, np.float32)
        base[:, :] = background
        return np.clip(image[:, :, :3] * a + base * (1 - a), 0, 255).astype(np.uint8)
    return image


def _score(left: paint_kind.KindFeature, right: paint_kind.KindFeature) -> float:
    """与识别器同款的综合分。"""
    ncc = float(np.clip(paint_kind._ncc(left.window, right.window), 0.0, 1.0))
    hist = float(np.clip(np.dot(left.hist, right.hist), 0.0, 1.0))
    return paint_kind.KIND_NCC_WEIGHT * ncc + (1.0 - paint_kind.KIND_NCC_WEIGHT) * hist


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
            out.append(
                {
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


def _composite(icon: str, category: str, level: int) -> np.ndarray:
    """与 ``build_paint_sheet.compose`` 同一套参数，返回 RGBA。"""
    icon_img = _read(ICONS / f"{icon}.png")
    shape = paint_frame.shape_of_category(category)
    frame_img = _read(FRAMES / f"{shape}_Lv{level}_{paint_frame.LEVEL_COLOR[level]}.png")

    def paste(image: np.ndarray, ratio: float) -> np.ndarray:
        canvas = np.zeros((TEMPLATE_SIDE, TEMPLATE_SIDE, 4), np.uint8)
        side = max(4, int(round(TEMPLATE_SIDE * ratio)))
        resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
        offset = (TEMPLATE_SIDE - side) // 2
        if offset >= 0:
            canvas[offset : offset + side, offset : offset + side] = resized
        else:
            cut = -offset
            canvas[:, :] = resized[cut : cut + TEMPLATE_SIDE, cut : cut + TEMPLATE_SIDE]
        return canvas

    frame_layer = paste(_rgba(frame_img), paint_frame.FRAME_SCALE)
    icon_layer = paste(_rgba(icon_img), paint_frame.FRAME_SCALE * paint_frame.ICON_SCALE)
    a = icon_layer[:, :, 3:4].astype(np.float32) / 255.0
    out = icon_layer[:, :, :3] * a + frame_layer[:, :, :3] * (1 - a)
    alpha = icon_layer[:, :, 3:4].astype(np.float32) + frame_layer[:, :, 3:4] * (1 - a)
    return np.concatenate([out, np.clip(alpha, 0, 255)], axis=2).astype(np.uint8)


def _rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return np.concatenate([image, np.full(image.shape[:2] + (1,), 255, np.uint8)], axis=2)
    return image


def _production(samples: list[dict], threshold: float) -> int:
    """生产口径：用软件里那个库（自学习 + 合成模板都挂着），走它自己的 ``read()``。"""
    bank = paint_kind.bank()
    print(f"\n【生产口径 · 与软件同一行代码】")
    print(
        f"  {len(bank.kinds())}/{len(paint_kind.known_kinds())} 个家族；"
        f"合成模板 {bank.sheet_count()} 条，自学习 {len(bank)} 条（{len(bank.templates)} 家族）"
    )
    print(f"  缺 {bank.unknown_kinds() or '无'}")
    print(f"  尺度 {paint_kind.SHEET_SCALES}；底色 {paint_kind.SHEET_BACKGROUND}")

    hits = 0
    misses: list[str] = []
    wrong: list[str] = []
    own_scores: list[float] = []
    total = 0
    for sample in samples:
        reading = bank.read(sample["tile"])
        if not reading.scores:
            continue
        total += 1
        own = float(reading.scores.get(sample["family"], 0.0))
        own_scores.append(own)
        if reading.subtype == sample["family"]:
            hits += 1
        elif reading.subtype is not None:
            wrong.append(f"{sample['family']}→{reading.subtype}({reading.score:.2f})")
        else:
            misses.append(f"{sample['family']} Lv{sample['level']} 本家 {own:.3f}")
    own_scores.sort()
    print(f"\n  认对 {hits}/{total}  ({hits / max(1, total) * 100:.1f}%)")
    if own_scores:
        print(
            f"  本家分 中位={statistics.median(own_scores):.4f}"
            f" 最低={own_scores[0]:.4f} 最高={own_scores[-1]:.4f}"
        )
    if wrong:
        print(f"  认错 {len(wrong)} 格：{' / '.join(wrong[:10])}")
    if misses:
        print(f"  未过线 {len(misses)} 格：{' / '.join(misses[:10])}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=paint_kind.KIND_MATCH_THRESHOLD)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="额外跑一遍旧的浅色底+单尺度对照（39.7% 那个数字的出处，不代表现在能力）",
    )
    args = parser.parse_args()

    samples = _samples()
    if not samples:
        raise SystemExit("没取到实机格子")
    print(f"样本 {len(samples)} 格；阈值 {args.threshold}")

    _production(samples, args.threshold)
    if not args.legacy:
        return 0

    print("\n【旧口径对照 · 浅色底 + 单尺度 + 只打自己家族】")
    print("  注意：这一段是历史参数，数字偏低不代表现在识别不了。")

    # 预生成合成模板（同一家族+等级只做一次）
    cache: dict[tuple[str, int], paint_kind.KindFeature] = {}

    def feature(icon: str, category: str, level: int) -> paint_kind.KindFeature | None:
        key = (icon, level)
        if key not in cache:
            rgba = _composite(icon, category, level)
            # 合成图是透明的，压到实机地块的浅色底上，与运行时一致
            bgr = _flatten(rgba, (238, 242, 247))
            made = paint_kind.kind_feature(bgr)
            if made is None:
                return None
            cache[key] = made
        return cache[key]

    own_scores: list[float] = []
    misses: list[str] = []
    for sample in samples:
        tile_feature = paint_kind.kind_feature(sample["tile"])
        if tile_feature is None:
            continue
        template = feature(sample["icon"], sample["category"], sample["level"])
        if template is None:
            continue
        score = _score(template, tile_feature)
        own_scores.append(score)
        if score < args.threshold:
            misses.append(f"{sample['family']} Lv{sample['level']} = {score:.3f}")

    own_scores.sort()
    passed = sum(1 for s in own_scores if s >= args.threshold)
    print(f"\n【自家模板打实机格】")
    print(f"  通过 {passed}/{len(own_scores)}  ({passed / max(1, len(own_scores)) * 100:.1f}%)")
    print(f"  分数 中位={statistics.median(own_scores):.4f} 最低={own_scores[0]:.4f} 最高={own_scores[-1]:.4f}")
    if misses:
        print(f"  未过线 {len(misses)} 个：{' / '.join(misses[:8])}")

    # 全库对打：27 家族 × 两个常见等级当模板库
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    bank: dict[str, paint_kind.KindFeature] = {}
    for entry in payload.values():
        for level in (3, 4):
            made = feature(entry["icon"], entry["category"], level)
            if made is not None:
                bank.setdefault(entry["name"], made)
    hits = 0
    total = 0
    wrong: list[str] = []
    for sample in samples:
        tile_feature = paint_kind.kind_feature(sample["tile"])
        if tile_feature is None:
            continue
        best_name, best_score = "", 0.0
        for name, template in bank.items():
            score = _score(template, tile_feature)
            if score > best_score:
                best_name, best_score = name, score
        total += 1
        if best_name == sample["family"] and best_score >= args.threshold:
            hits += 1
        elif best_name != sample["family"]:
            wrong.append(f"{sample['family']}→{best_name}({best_score:.2f})")
    print(f"\n【全库对打 {len(bank)} 家族】")
    print(f"  认对 {hits}/{total}  ({hits / max(1, total) * 100:.1f}%)")
    if wrong:
        print(f"  认错 {len(wrong)} 个：{' / '.join(wrong[:10])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
