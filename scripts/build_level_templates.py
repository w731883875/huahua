"""用真值标注构建/评估棋盘等级角标模板库。

用法::

    .venv/Scripts/python.exe scripts/build_level_templates.py          # 构建并保存
    .venv/Scripts/python.exe scripts/build_level_templates.py --report # 只评估，不写盘

真值来自 ``tests/fixtures/board_level_labels.json``。评估采用**留一法**：
用同一张截图里其它格子的样本当模板，这样分数不会因为「拿自己匹配自己」而虚高。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huahua_assistant import tile_reader  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "board_level_labels.json"


def imread(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def parse_position(text: str) -> tuple[int, int]:
    row, col = text.split(",")
    return int(row), int(col)


def load_cells(entry: dict) -> dict[tuple[int, int], np.ndarray]:
    image = imread(ROOT / entry["image"])
    if image is None:
        raise SystemExit(f"读不到截图: {entry['image']}")
    x0, y0 = entry["origin"]
    cell = int(entry["cell"])
    cells: dict[tuple[int, int], np.ndarray] = {}
    for row in range(6):
        for col in range(6):
            top = y0 + row * cell
            left = x0 + col * cell
            cells[(row, col)] = image[top : top + cell, left : left + cell]
    return cells


def build(report_only: bool) -> int:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    bank = tile_reader.LevelTemplateBank()

    positives: list[tuple[str, tuple[int, int], int, np.ndarray]] = []
    negatives: list[tuple[str, tuple[int, int], np.ndarray]] = []

    for entry in payload["screenshots"]:
        cells = load_cells(entry)
        labels = {parse_position(key): int(value) for key, value in entry["labels"].items()}
        for position, level in labels.items():
            bank.add(level, cells[position])
            positives.append((entry["image"], position, level, cells[position]))
        for key in entry.get("empty", []):
            position = parse_position(key)
            negatives.append((entry["image"], position, cells[position]))

    print(f"模板库: {len(bank)} 个样本，等级 {bank.levels()}")

    # ---------------------------------------------------------------- 存在性
    # 留一法：评估某个格子时把它自己的样本从库里摘掉，否则「自己匹配自己」恒为 1.0。
    def loo_scores(tile: np.ndarray, skip: tuple[str, tuple[int, int]] | None) -> tuple[float, float]:
        window = tile_reader.label_window(tile)
        x0, y0, x1, y1 = tile_reader.LABEL_TEMPLATE_BOX
        patch = window[y0:y1, x0:x1]
        density = float(patch.mean()) if patch.size else 0.0
        best = 0.0
        for image, position, _level, other in positives:
            if skip is not None and (image, position) == skip:
                continue
            other_patch = tile_reader.label_template(other)
            if other_patch is None:
                continue
            best = max(best, tile_reader._match_score(window, other_patch))
        return best, density

    threshold = tile_reader.PRESENCE_THRESHOLD
    density_min = tile_reader.PRESENCE_DENSITY_MIN
    pos_hits = 0
    pos_density = []
    for image, position, level, tile in positives:
        score, density = loo_scores(tile, (image, position))
        pos_density.append(density)
        if density >= density_min and score >= threshold:
            pos_hits += 1
    neg_hits = 0
    neg_density = []
    for image, position, tile in negatives:
        score, density = loo_scores(tile, None)
        neg_density.append(density)
        if density >= density_min and score >= threshold:
            neg_hits += 1
    print(
        f"存在性（留一法）: 有标签 {pos_hits}/{len(positives)} 命中；"
        f"无标签误报 {neg_hits}/{len(negatives)}"
    )
    print(
        f"  能量密度: 有标签 {min(pos_density):.1f}~{max(pos_density):.1f}，"
        f"无标签 {min(neg_density):.1f}~{max(neg_density):.1f}"
        f"（阈值 {density_min}）"
    )

    # ---------------------------------------------------------------- 等级（留一法）
    correct = 0
    total = 0
    failures = []
    for image, position, level, tile in positives:
        window = tile_reader.label_window(tile)
        best_score = 0.0
        best_level = None
        for other_image, other_position, other_level, other_tile in positives:
            if other_level == level and other_image == image and other_position == position:
                continue
            patch = tile_reader.label_template(other_tile)
            if patch is None:
                continue
            score = tile_reader._match_score(window, patch)
            if score > best_score:
                best_score = score
                best_level = other_level
        total += 1
        if best_level == level:
            correct += 1
        else:
            failures.append((image, position, level, best_level, round(best_score, 3)))

    print(f"等级（留一法）: {correct}/{total} = {correct / total:.1%}")
    for image, position, level, best_level, score in failures:
        print(f"  错 {Path(image).name} {position} 真值 Lv{level} → Lv{best_level} ({score})")

    if report_only:
        return 0

    target = bank.save()
    print(f"已写入 {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="只评估，不写模板文件")
    args = parser.parse_args()
    return build(report_only=args.report)


if __name__ == "__main__":
    raise SystemExit(main())
