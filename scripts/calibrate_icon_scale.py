"""标定「图标占框多大」——以实机蒙版匹配分数为判据。

## 为什么需要这个脚本

``paint_frame.ICON_SCALE`` 决定合成图里图标相对框的大小。这个值直接决定
模板匹配能不能过：图标画大了，蒙版压不住图案边缘；画小了，图案细节又不够。

之前用的是贴图原生比 ``128/200 = 0.64``，那是**推导值**，不是实机值——
200 是从切出来的框尺寸（198~233，含辉光）反推的，本身就带误差。

这个脚本换一条路：拿实机截图里的真格子当靶子，把候选比例逐个画出来，
算**带蒙版的 NCC**（只在框内沿的圆/菱形区域里比，框与辉光不参与），
取分数最高的那个。分数最高的比例就是「蒙版能通过」的比例。

## 判据为什么用蒙版

棋盘格 87px，框外一圈辉光很亮、且随背景变化，直接全图 NCC 会被辉光主导，
比例差异被淹没。只比框内沿的实心区域，比的就纯粹是「图案在这块区域里
占多大、落在哪」，正是比例的作用面。

用法::

    python scripts/calibrate_icon_scale.py            # 打印标定表
    python scripts/calibrate_icon_scale.py --apply    # 顺便写回 paint_frame.py
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

from huahua_assistant import paint_frame, vision  # noqa: E402
from huahua_assistant.capture import read_image  # noqa: E402

ICONS = ROOT / "huahua_assistant" / "assets" / "paint_icons"
SHOTS = ROOT / "jietu"
#: 候选比例。0.55~0.95 覆盖「明显偏小」到「几乎铺满」。
CANDIDATES = tuple(round(0.55 + 0.025 * i, 3) for i in range(17))
#: 比对边长。和 paint_kind 的搜索窗同量级，且能被框素材整除。
SIDE = 96


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


def _icon_files() -> list[Path]:
    return sorted(ICONS.glob("UI_S8Gameplay_Color*_128.png"))


def _inner_mask(frame: np.ndarray) -> np.ndarray:
    """框内沿的实心区域（不含框本身和辉光）。"""
    alpha = frame[:, :, 3].astype(np.float32) / 255.0
    # 框外沿的 alpha 通常 > 0.55，内沿里面是 0（透明）
    return (alpha < 0.35).astype(np.float32)


def _composite(icon: np.ndarray, frame: np.ndarray, scale: float) -> np.ndarray:
    side = max(6, int(round(SIDE * scale)))
    offset = (SIDE - side) // 2
    canvas = np.zeros((SIDE, SIDE, 4), np.uint8)
    canvas[offset : offset + side, offset : offset + side] = cv2.resize(
        icon, (side, side), interpolation=cv2.INTER_AREA
    )
    return _over(canvas, frame)


def _gray(image: np.ndarray) -> np.ndarray:
    bgr = image[:, :, :3] if image.shape[2] == 4 else image
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """只在 mask>0.5 的像素上算零均值归一化相关。"""
    sel = mask > 0.5
    if sel.sum() < 32:
        return -1.0
    x = a[sel] - a[sel].mean()
    y = b[sel] - b[sel].mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator < 1e-6:
        return -1.0
    return float((x * y).sum() / denominator)


def _tiles() -> list[np.ndarray]:
    """从截图里按识别出的棋盘切格子，只留「有图案」的格子。"""
    out: list[np.ndarray] = []
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
        if rect is None or rect.w < 200 or rect.h < 200:
            continue
        cw, ch = rect.w / 6.0, rect.h / 6.0
        for row in range(6):
            for col in range(6):
                x0 = int(round(rect.x + col * cw))
                y0 = int(round(rect.y + row * ch))
                x1 = int(round(rect.x + (col + 1) * cw))
                y1 = int(round(rect.y + (row + 1) * ch))
                tile = image[y0:y1, x0:x1]
                if tile.size == 0 or min(tile.shape[:2]) < 40:
                    continue
                # 空白/阻挡地块整体很平，不要
                gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
                if float(cv2.Laplacian(gray, cv2.CV_32F).var()) < 300:
                    continue
                out.append(tile)
    return out


def build_table() -> list[dict]:
    frames = paint_frame.frames()
    icons = [(path, _to_bgra(_read(path))) for path in _icon_files()]
    tiles = _tiles()
    if not tiles:
        raise SystemExit("没有切出任何实机格子，检查 jietu/ 下的截图")
    print(f"实机格子 {len(tiles)} 个，图标 {len(icons)} 张")

    # 每个格子先缩到统一边长；用 (框形状, 等级) 无关的平均分作为该比例的整体表现
    normalized = [
        cv2.resize(tile, (SIDE, SIDE), interpolation=cv2.INTER_AREA) for tile in tiles
    ]

    rows: list[dict] = []
    for shape in ("circle", "diamond"):
        frame_asset = frames[(shape, 3)]
        frame = cv2.resize(_to_bgra(_read(frame_asset.path)), (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        mask = _inner_mask(frame)
        frame_gray = _gray(frame)
        for scale in CANDIDATES:
            # 同比例下，每张图标的平均分（不挑最优，避免「随便一张撞高分」）
            scores: list[float] = []
            for _path, icon in icons:
                composite = _composite(icon, frame, scale)
                scores.append(_masked_ncc(_gray(composite), frame_gray, mask))
            composite_mean = float(np.mean(scores))
            # 实机格子上，这个比例合成的模板最高能拿多少分
            best = 0.0
            for _path, icon in icons:
                composite = _composite(icon, frame, scale)
                template = _gray(composite)
                template = template[int(SIDE * 0.22) : int(SIDE * 0.78), int(SIDE * 0.22) : int(SIDE * 0.78)]
                for tile in normalized:
                    tile_gray = _gray(tile)
                    if template.shape[0] > tile_gray.shape[0] or template.shape[1] > tile_gray.shape[1]:
                        continue
                    result = cv2.matchTemplate(tile_gray, template, cv2.TM_CCOEFF_NORMED)
                    best = max(best, float(cv2.minMaxLoc(result)[1]))
            rows.append(
                {"shape": shape, "scale": scale, "mean_self_ncc": composite_mean, "best_tile_ncc": best}
            )
    return rows


def summarize(rows: list[dict]) -> dict[str, float]:
    """两种形状各自的最优比例，再取平均。"""
    best: dict[str, float] = {}
    for shape in ("circle", "diamond"):
        subset = [row for row in rows if row["shape"] == shape]
        winner = max(subset, key=lambda row: row["best_tile_ncc"])
        best[shape] = winner["scale"]
    best["mean"] = round((best["circle"] + best["diamond"]) / 2.0, 4)
    return best


def apply_scale(scale: float) -> None:
    path = ROOT / "huahua_assistant" / "paint_frame.py"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^ICON_SCALE = .*$", f"ICON_SCALE = {scale}", text, flags=re.M)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="把标定结果写回 paint_frame.py")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    rows = build_table()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        print(f"{'形状':<8}{'比例':>6}{'自比NCC':>10}{'实机最高NCC':>12}")
        for row in rows:
            print(
                f"{row['shape']:<8}{row['scale']:>6.3f}"
                f"{row['mean_self_ncc']:>10.4f}{row['best_tile_ncc']:>12.4f}"
            )
    best = summarize(rows)
    print("\n标定结果:", best)
    if args.apply:
        apply_scale(best["mean"])
        print(f"已写回 ICON_SCALE = {best['mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
