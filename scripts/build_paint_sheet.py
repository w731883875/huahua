"""把「图标 + 框」合成为一张核对表：背包（无背景）与地块（白底）两种形态。

数据全部来自解包资源，脚本本身不含任何美术素材：

* 图标：``huahua_assistant/assets/paint_icons/UI_S8Gameplay_Color*_128.png``
  （从 S8 PAK 里解出来的 256×256 BC7 贴图，与 tlidb 网页图标逐像素比对过，MAE<6）；
* 框：``huahua_assistant/assets/paint_frames/{circle,diamond}_Lv{1..7}_*.png``
  （从 S8 图集里切出来的 14 个框，白绿蓝紫橙红彩 × 圆形/菱形）；
* 家族、等级、形状规则：``paint_catalog.json``（怪物=圆形框，玩法=菱形框）。

用法::

    python scripts/build_paint_sheet.py            # 生成 runtime/paint_sheet/
    python scripts/build_paint_sheet.py --open     # 顺手打开 index.html

产物：

* ``runtime/paint_sheet/<家族编号>_Lv<等级>_<bg>.png`` —— 单格合成图；
  ``bg=backpack`` 保留透明通道，``bg=tile`` 压在浅色地块底上；
* ``runtime/paint_sheet/index.html`` —— 27 个家族 × 7 级 × 2 形态的核对表。
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from huahua_assistant import paint_frame  # noqa: E402

ASSETS = ROOT / "huahua_assistant" / "assets"
ICON_DIR = ASSETS / "paint_icons"
FRAME_DIR = ASSETS / "paint_frames"
CATALOG = ASSETS / "paint_catalog.json"
OUT_DIR = ROOT / "runtime" / "paint_sheet"

#: 棋盘上地块的底色。实机空白地块是偏冷的白（带画笔水印），这里取它的平均色。
TILE_BG = (238, 242, 247)  # BGR
#: 合成边长。图标 256、框按同尺寸缩放，不引入额外重采样。
SIZE = 256
#: 家族类别 → 框形状。这是解包资源里的硬规则，不是拟合出来的。
SHAPE_OF_CATEGORY = {"monster": "circle", "gameplay": "diamond"}
#: 等级 → 框颜色名（只用于文件命名与表头，判据是颜色本身）。
LEVEL_COLOR = {1: "白", 2: "绿", 3: "蓝", 4: "紫", 5: "橙", 6: "红", 7: "彩"}

BACKGROUNDS = (("backpack", "背包（无背景）"), ("tile", "白色地块（白底）"))


def _read(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Windows 上 ``cv2.imread`` 读不了中文路径，统一走 ``imdecode``。"""
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise SystemExit(f"读不出图片: {path}")
    return image


def _write(path: Path, image: np.ndarray) -> None:
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise SystemExit(f"编码失败: {path}")
    buffer.tofile(str(path))


def _rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2] + (1,), 255, np.uint8)
        return np.concatenate([image, alpha], axis=2)
    return image


def _over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """标准 alpha 合成（两者都是 BGRA，返回 BGRA）。"""
    a = top[:, :, 3:4].astype(np.float32) / 255.0
    out = top[:, :, :3].astype(np.float32) * a + bottom[:, :, :3].astype(np.float32) * (1.0 - a)
    alpha = top[:, :, 3:4].astype(np.float32) + bottom[:, :, 3:4].astype(np.float32) * (1.0 - a)
    return np.concatenate([out, np.clip(alpha, 0, 255)], axis=2).astype(np.uint8)


def _paste_center(canvas: np.ndarray, image: np.ndarray, ratio: float) -> np.ndarray:
    """把 ``image`` 缩放成 ``边长 × ratio`` 后居中贴到 ``canvas``。

    ``ratio`` 可以大于 1（框要放大到超出画布），超出部分居中裁掉——
    实机里框外沿本来就是切着格子边缘画的。
    """
    side = max(8, int(round(SIZE * ratio)))
    resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
    offset = (SIZE - side) // 2
    if offset >= 0:
        canvas[offset : offset + side, offset : offset + side] = resized
    else:
        cut = -offset
        canvas[:, :] = resized[cut : cut + SIZE, cut : cut + SIZE]
    return canvas


def compose(icon_path: Path, frame_path: Path) -> np.ndarray:
    """图标与框都按实机比例画：框 :data:`paint_frame.FRAME_SCALE`、图标再乘 ``ICON_SCALE``。

    合成参数是从实机截图量出来的（见 :mod:`huahua_assistant.paint_frame` 文档）：
    框要放大到 1.234 才和游戏里一样大，图标相对框 0.62。
    框放大后会超出画布，居中裁掉——实机的框外沿也是切在格子边上的。
    """
    icon = _rgba(_read(icon_path))
    frame = _rgba(_read(frame_path))
    frame_layer = _paste_center(np.zeros((SIZE, SIZE, 4), np.uint8), frame, paint_frame.FRAME_SCALE)
    icon_layer = _paste_center(
        np.zeros((SIZE, SIZE, 4), np.uint8),
        icon,
        paint_frame.FRAME_SCALE * paint_frame.ICON_SCALE,
    )
    return _over(icon_layer, frame_layer)


def _frame_path(shape: str, level: int) -> Path:
    return FRAME_DIR / f"{shape}_Lv{level}_{LEVEL_COLOR[level]}.png"


def _load_catalog() -> dict[str, dict]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def build(out_dir: Path = OUT_DIR) -> tuple[Path, int]:
    catalog = _load_catalog()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    rows: list[dict] = []

    for key in sorted(catalog):
        entry = catalog[key]
        shape = SHAPE_OF_CATEGORY.get(str(entry.get("category")))
        icon_name = entry.get("icon")
        levels = sorted(int(level) for level in (entry.get("levels") or {}))
        if not shape or not icon_name or not levels:
            continue
        icon_path = ICON_DIR / f"{icon_name}.png"
        if not icon_path.exists():
            raise SystemExit(f"{key} 的图标缺失: {icon_path}")
        cells: dict[int, dict[str, str]] = {}
        for level in levels:
            # 背包和白色地块同一套规则：怪物圆形、玩法菱形。
            frame_paths = {bg: _frame_path(shape, level) for bg, _label in BACKGROUNDS}
            for bg, path in frame_paths.items():
                if not path.exists():
                    raise SystemExit(f"{key} Lv{level} 的框缺失: {path}")
            composed = {bg: compose(icon_path, path) for bg, path in frame_paths.items()}
            cell: dict[str, str] = {}
            for bg, _label in BACKGROUNDS:
                image = composed[bg]
                if bg == "tile":
                    base = np.zeros((SIZE, SIZE, 4), np.uint8)
                    base[:, :, :3] = TILE_BG
                    base[:, :, 3] = 255
                    image = _over(base, image)
                name = f"{key}_Lv{level}_{bg}.png"
                _write(out_dir / name, image)
                cell[bg] = name
                written += 1
            cells[level] = cell
        rows.append(
            {
                "key": key,
                "name": entry.get("name") or key,
                "category": entry.get("category"),
                "shape": shape,
                "icon": icon_name,
                "levels": levels,
                "cells": cells,
            }
        )

    page = out_dir / "index.html"
    page.write_text(_render_html(rows, _special_rows(out_dir)), encoding="utf-8")
    return page, written


def _special_rows(out_dir: Path) -> list[dict]:
    """非棋盘图标（虚空 / 宠物特性 / 待确认），合成到浅底上便于核对。"""
    roles_path = ICON_DIR / "roles.json"
    try:
        roles = json.loads(roles_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    result: list[dict] = []
    for icon_name, entry in roles.items():
        path = ICON_DIR / f"{icon_name}.png"
        if not path.exists():
            continue
        icon = cv2.resize(_rgba(_read(path)), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        base = np.zeros((SIZE, SIZE, 4), np.uint8)
        base[:, :, :3] = TILE_BG
        base[:, :, 3] = 255
        name = f"special_{icon_name}.png"
        _write(out_dir / name, _over(base, icon))
        result.append(
            {
                "icon": icon_name,
                "file": name,
                "role": entry.get("role") or "",
                "name": entry.get("name") or "待确认",
                "note": entry.get("note") or "",
            }
        )
    return result


def _render_html(rows: list[dict], specials: list[dict] | None = None) -> str:
    head_cells = "".join(
        f'<th class="lv">Lv{level}<span>{LEVEL_COLOR[level]}</span></th>' for level in range(1, 8)
    )
    body: list[str] = []
    for row in rows:
        shape = "圆形" if row["shape"] == "circle" else "菱形"
        category = "怪物" if row["category"] == "monster" else "玩法"
        head = (
            f'<th class="fam" rowspan="2"><b>{html.escape(row["name"])}</b>'
            f'<span>{row["key"]} · {category} · {shape}框</span></th>'
        )
        for index, (bg, label) in enumerate(BACKGROUNDS):
            cells = []
            for level in range(1, 8):
                cell = row["cells"].get(level, {}).get(bg)
                if cell is None:
                    cells.append('<td class="na">—</td>')
                    continue
                cls = "shot transparent" if bg == "backpack" else "shot"
                cells.append(f'<td class="{cls}"><img src="{cell}" alt="{row["key"]} Lv{level}"></td>')
            prefix = head if index == 0 else ""
            css = ' class="start"' if index == 0 else ""
            body.append(f'<tr{css}>{prefix}<th class="bg">{label}</th>{"".join(cells)}</tr>')

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>画中雪原 · 颜料图标 + 框 核对表</title>
<style>
  :root {{
    --ink: #16202b; --muted: #5b6b7c; --line: #d6dee8; --bg: #f7f9fc;
    --monster: #b4442f; --gameplay: #1f6f8b;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 28px 32px 64px; background: var(--bg); color: var(--ink);
         font: 14px/1.5 "Microsoft YaHei", "PingFang SC", system-ui, sans-serif; }}
  h1 {{ margin: 0 0 6px; font-size: 22px; }}
  .lede {{ margin: 0 0 18px; color: var(--muted); max-width: 980px; }}
  .lede code {{ background: #eef2f7; padding: 1px 5px; border-radius: 4px; }}
  table {{ border-collapse: separate; border-spacing: 0; background: #fff;
           border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
  thead th {{ position: sticky; top: 0; z-index: 3; background: #eef2f7; padding: 8px 6px;
              font-weight: 600; border-bottom: 1px solid var(--line); }}
  thead th.lv span {{ display: block; font-weight: 400; color: var(--muted); font-size: 11px; }}
  th.fam {{ position: sticky; left: 0; z-index: 2; background: #fbfcfe; text-align: left;
            padding: 8px 12px; min-width: 168px; border-right: 1px solid var(--line);
            border-top: 1px solid var(--line); }}
  th.fam b {{ display: block; font-size: 15px; }}
  th.fam span {{ display: block; color: var(--muted); font-size: 11px; font-weight: 400; }}
  th.bg {{ background: #fbfcfe; color: var(--muted); font-size: 11px; font-weight: 500;
           text-align: left; padding: 0 8px; border-top: 1px solid var(--line); white-space: nowrap; }}
  td {{ padding: 4px; border-top: 1px solid #edf1f6; border-left: 1px solid #edf1f6; }}
  td.shot img {{ display: block; width: 92px; height: 92px; image-rendering: auto; }}
  td.transparent {{ background-color: #fff;
    background-image: linear-gradient(45deg, #e7ecf3 25%, transparent 25%, transparent 75%, #e7ecf3 75%),
                      linear-gradient(45deg, #e7ecf3 25%, transparent 25%, transparent 75%, #e7ecf3 75%);
    background-size: 12px 12px; background-position: 0 0, 6px 6px; }}
  td.na {{ color: #b9c3cf; text-align: center; }}
  tr.start td, tr.start th.bg {{ border-top: 2px solid var(--line); }}
  h2 {{ margin: 34px 0 8px; font-size: 17px; }}
  .special {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .special figure {{ margin: 0; width: 240px; background: #fff; border: 1px solid var(--line);
                     border-radius: 10px; padding: 12px; }}
  .special img {{ display: block; width: 96px; height: 96px; margin: 0 auto 8px; }}
  .special b {{ display: block; text-align: center; }}
  .special .role {{ display: block; text-align: center; color: var(--muted); font-size: 11px; margin-bottom: 6px; }}
  .special p {{ margin: 0; font-size: 12px; color: var(--muted); line-height: 1.55; }}
</style>
</head>
<body>
<h1>画中雪原 · 颜料「图标 + 框」核对表</h1>
<p class="lede">
  每个家族两行：<b>背包</b>透明底、<b>白色地块</b>压在实机空白可落子地块的浅色底上。
  <b>两处形状规则相同</b>：<b style="color:var(--monster)">怪物 = 圆形</b>、
  <b style="color:var(--gameplay)">玩法 = 菱形</b>。
  框的颜色按等级固定：白绿蓝紫橙红彩 = Lv1~Lv7。
  <b>图标按原生尺寸比 0.64 缩放</b>（图标 <code>_128</code> ÷ 品质框 ≈200），
  不是铺满框——之前画成 1.0 所以显得过大。
  图标来自 PAK 解包（<code>UI_S8Gameplay_Color*_128</code>），框来自 S8 图集切片，均未从网页下载。
  共 {len(rows)} 个家族；赛季只有 Lv3/Lv4，黑潮~玩偶只有 Lv3~Lv7，缺的等级显示「—」。
</p>
<table>
<thead><tr><th class="fam">家族</th><th class="bg">形态</th>{head_cells}</tr></thead>
<tbody>
{''.join(body)}
</tbody>
</table>
{_render_special(specials or [])}
</body>
</html>
"""


def _render_special(specials: list[dict]) -> str:
    if not specials:
        return ""
    cards = []
    for item in specials:
        cards.append(
            '<figure>'
            f'<img src="{item["file"]}" alt="{html.escape(item["icon"])}">'
            f'<b>{html.escape(item["name"])}</b>'
            f'<span class="role">{html.escape(item["role"])} · {html.escape(item["icon"])}</span>'
            f'<p>{html.escape(item["note"])}</p>'
            "</figure>"
        )
    return (
        "<h2>非棋盘图标（不是孤儿资源）</h2>"
        '<p class="lede">这三张在 PlotItem 表里查不到家族记录，但都有明确用途。'
        "它们不进识别库，避免把状态认成家族。</p>"
        f'<div class="special">{"".join(cards)}</div>'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="合成颜料「图标+框」核对表")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="输出目录")
    parser.add_argument("--open", action="store_true", help="生成后打开 index.html")
    args = parser.parse_args(argv)

    page, count = build(args.out)
    print(f"合成 {count} 张，核对表: {page}")
    if args.open:
        import os

        os.startfile(str(page))  # noqa: S606 - 本地查看用
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
