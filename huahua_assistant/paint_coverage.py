"""颜料「家族 × 等级」覆盖台账。

棋盘上的颜料一共 27 个家族名（26 个官方 + 手工补的「赛季」），
但它们**不是都齐 7 级**：13 个怪物家族 + 6 个基础玩法家族有 Lv1~Lv7；
7 个赛季轮换玩法（黑潮、黑帆、魔方、迷城、梦魇、怪谈、玩偶）在 PlotItem
表里占了 5 条记录，但其中 Lv3/Lv4 的名字其实是「赛季」、图标也是
``ColorPlay15``，**真正属于自己的只有 Lv5~Lv7**；赛季本身只有 Lv3/Lv4。
**合计 156 个组合**（19×7 + 7×3 + 2），对上解包 168 条记录里
「13×7 + 6×7 + 7×(2 条赛季 + 3 条专属)」的分布。

识别链要回答两个不同的问题，别混在一起：

- **认不认得这个家族**——归 :class:`paint_kind.KindTemplateBank` 管。
  一个家族只要 1~3 张样张就能认全 7 个等级，因为解包图标是按家族共用的
  （``ColorMonster{N}_128`` 一张图配 7 个等级）。
- **见没见过这个组合**——归这里管。Lv1 的鼠人长什么样、Lv6 的龙骨有没有
  真样本，只有**实机截图**能回答。而截图是一次性的：代码扫完就把结论丢了，
  下次再问「还差哪几个组合」又得从头扫 11 张图。

所以这里把扫描结果落成一份**可增量的台账**：

    {"version": 1,
     "samples": {"鼠人": {"3": [["斜对角.png", 2, 4]], ...}},
     "unknown": {"4": [["其他.png", 2, 2]]}}

同一张图的同一格重复扫只算一次（以 ``image + row + col`` 去重），
所以新截图往 ``jietu/`` 一丢、重跑一遍 ``scan`` 就行，台账只增不改。

``unknown`` 单独一栏，专门记「看着像颜料、但 27 个家族谁都没对上」的格子。
它们是补模板的第一顺位线索：要么是还没入库的家族，要么是通关宠物特效图
这类非颜料（见 :mod:`paint_kind` 的说明），人工看一眼就能分流。

命令行::

    python -m huahua_assistant.paint_coverage scan      # 扫 jietu/ 并更新台账
    python -m huahua_assistant.paint_coverage report    # 打印矩阵 + 缺哪些组合
    python -m huahua_assistant.paint_coverage export    # 把每个样本格裁成 家族/LvN 归档
    python -m huahua_assistant.paint_coverage verify    # 用官方图标复核归档结果
    python -m huahua_assistant.paint_coverage ignore    <图名> <行> <列> [原因]
    python -m huahua_assistant.paint_coverage relabel   <图名> <行> <列> <家族> <等级>

改判这一条是给人用的：自动识别会认错家族（例如把征战之神认成宝箱），
``relabel`` 把格子整体搬到新家族下，``ignore`` 把不该入库的格子（虚空化
形态、通关展示图）永久划出，两者都不改动原始截图，随时可回溯。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import paint_icon, paint_kind

#: 台账路径。放在 ``runtime/`` 下，跟其它运行产物一起，不进版本库。
DEFAULT_LEDGER = Path(__file__).resolve().parent.parent / "runtime" / "paint_coverage.json"

#: 全部等级。
LEVELS = (1, 2, 3, 4, 5, 6, 7)


def expected_levels(family: str) -> tuple[int, ...]:
    """这个家族理论上该有几个等级——按条目本身的名字算，不按块的 ID 算。

    黑潮那一块（``g008``）在 PlotItem 表里占 5 条记录（200803~200807），
    但 200803/200804 的名字是「玩法颜料-赛季」、图标是 ``ColorPlay15``，
    真正属于黑潮的只有 200805~200807。用「条目名末尾是不是这个家族」
    过滤一遍，5 条就自动收敛成 3 条，不用再手写等级范围。
    """
    key = paint_kind.family_key(family)
    entry = paint_kind.catalog().get(key or "") or {}
    levels = entry.get("levels") or {}
    if not levels:
        return LEVELS
    own = [
        int(level)
        for level, label in levels.items()
        if str(label).split("-")[-1].strip() == family
    ]
    return tuple(sorted(own)) or LEVELS


def expected_total() -> int:
    """156：19 个齐 7 级的家族（91+42）+ 7 个只有 Lv5~Lv7 的赛季玩法 + 赛季两级。"""
    return sum(len(expected_levels(name)) for name in paint_kind.known_kinds())


def load(path: Path | None = None) -> dict:
    """读台账；不存在或读坏时返回空台账（扫描会从头补齐）。"""
    target = Path(path) if path is not None else DEFAULT_LEDGER
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    samples = payload.get("samples")
    unknown = payload.get("unknown")
    ignored = payload.get("ignored")
    return {
        "version": 1,
        "ignored": ignored if isinstance(ignored, list) else [],
        "samples": samples if isinstance(samples, dict) else {},
        "unknown": unknown if isinstance(unknown, dict) else {},
    }


def _prune_empty(store: dict) -> None:
    """删掉没有样本的空桶，让「有桶」始终等于「有样本」。

    :func:`matrix` 是按桶建表的：只要等级键还在，它就会如实列出这个等级、
    计数为 0。人工改判把某个家族唯一的一格摘走之后，如果空桶留着，
    ``matrix`` 就会报出一个「属于该家族但一条样本都没有」的等级，
    看上去像覆盖率还在。内存态和落盘态必须同一口径，否则
    「跑一遍脚本看结果」和「读 json 看结果」会得出不同结论。
    """
    store["unknown"] = {level: items for level, items in store["unknown"].items() if items}
    store["samples"] = {
        family: {level: items for level, items in levels.items() if items}
        for family, levels in store["samples"].items()
    }
    store["samples"] = {family: levels for family, levels in store["samples"].items() if levels}


def save(store: dict, path: Path | None = None) -> Path:
    target = Path(path) if path is not None else DEFAULT_LEDGER
    target.parent.mkdir(parents=True, exist_ok=True)
    _prune_empty(store)
    target.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def _bucket(store: dict, family: str | None, level: int | None) -> list:
    table = store["unknown"] if family is None else store["samples"].setdefault(family, {})
    return table.setdefault(str(level) if level is not None else "?", [])


def record(store: dict, samples: list[tuple[str, int, int, str | None, int | None]]) -> int:
    """把一批 ``(image, row, col, family, level)`` 收进台账，返回新增条数。

    去重键是 ``image + row + col``：同一张截图重扫不会把计数翻倍，
    但换一张截图拍到同一家族的同一等级会照常累加——这是
    「一个组合有几份独立样本」的唯一口径。
    """
    added = 0
    for image_name, row, col, family, level in samples:
        if _is_ignored(store, image_name, row, col):
            continue
        # 同一格之前在 unknown 里挂过（那时还没认出来），现在认出来了就迁走，
        # 否则「未认名」会越积越多，明明已经进门了还赖在待办里。
        for bucket in store["unknown"].values():
            for item in list(bucket):
                if item[0] == image_name and item[1] == row and item[2] == col:
                    bucket.remove(item)
        bucket = _bucket(store, family, level)
        if any(item[0] == image_name and item[1] == row and item[2] == col for item in bucket):
            continue
        bucket.append([image_name, row, col])
        added += 1
    return added


def forget(store: dict, image_name: str, row: int, col: int) -> list[tuple[str, int | None]]:
    """把某格从所有家族里摘掉，返回它原本待过的 ``(家族, 等级)`` 列表。

    台账的键是「家族 → 等级 → 格子」，一个格子没有独立身份，只能挂在某个
    桶里。人工复核发现认错了家族（例如 ``宝箱 Lv5`` 实为征战之神），就必须
    先把它从原桶摘走再重记，否则同一格会同时挂在两个家族下，模板库学歪。
    """
    removed: list[tuple[str, int | None]] = []
    for family, levels in store["samples"].items():
        for level_text, items in levels.items():
            for item in list(items):
                if item[0] == image_name and item[1] == row and item[2] == col:
                    items.remove(item)
                    try:
                        level = int(level_text)
                    except ValueError:
                        level = None
                    removed.append((family, level))
    _prune_empty(store)
    return removed


def relabel(
    store: dict,
    image_name: str,
    row: int,
    col: int,
    family: str,
    level: int,
) -> list[tuple[str, int | None]]:
    """把某格改判到 ``(family, level)``，返回它原来的位置（空列表说明本来就不在册）。

    只改归属、不新造样本：格子来自哪张截图的哪一行哪一列这件事不动，
    所以导出时的文件名依旧可回溯到原图。
    """
    moved_from = forget(store, image_name, row, col)
    record(store, [(image_name, row, col, family, level)])
    return moved_from


def unignore(store: dict, image_name: str, row: int, col: int) -> bool:
    """撤销 :func:`ignore`：早先误判成「非颜料」的格子可以捞回来重认。

    与 :func:`ignore` 对称存在。之所以需要它，是因为 :func:`record` 会跳过
    ignore 名单里的格子——一张图看走眼划掉了，没有撤销口子就再也进不来。
    """
    kept = [item for item in store["ignored"] if item[:3] != [image_name, row, col]]
    if len(kept) == len(store["ignored"]):
        return False
    store["ignored"] = kept
    return True


def _read_image(path: Path):
    return paint_kind._read_image(path)


def scan_image(path: Path) -> list[tuple[str, int, int, str | None, int | None]]:
    """扫一张截图，返回其中每个 PAINT 格的 ``(图名, 行, 列, 家族, 等级)``。"""
    from . import vision
    from .models import TileState

    image = _read_image(path)
    if image is None:
        return []
    scene = vision.analyze_scene(image)
    if scene is None or scene.board is None:
        return []
    result = []
    for tile in scene.board.tiles:
        if tile.state != TileState.PAINT:
            continue
        result.append((path.name, tile.row, tile.col, tile.paint_subtype, tile.paint_level))
    return result


def harvest_icons(
    folder: Path = Path("jietu"),
    *,
    store: dict | None = None,
    save_ledger: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """扫一遍截图，**用官方图标**给每个颜料格定家族，顺手填模板库。

    以前模板只能来自右侧「当前颜料」面板（OCR 读到名字 + 面板图标配到棋盘格），
    一局才攒一个家族。有了 27 张官方图标，任何一格都能直接认，于是：

    - 模板库一次补齐所有见过的家族（每个家族最多留 3 张，见 ``MAX_TEMPLATES_PER_KIND``）；
    - 台账跟着记样本，未认名栏里那些格子迁进正册。

    只收图标判据过得硬的格子（见 :mod:`paint_icon` 的三条阈值），
    认不准的一律不进——模板错了比缺模板更麻烦。
    """
    from . import paint_icon
    from .models import TileState

    folder = Path(folder)
    active = store if store is not None else load()
    bank = paint_kind.reload_bank()
    gallery = paint_icon.gallery()
    added: dict[str, int] = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        image = _read_image(path)
        if image is None:
            continue
        from . import vision

        scene = vision.analyze_scene(image)
        if scene is None or scene.board is None:
            continue
        samples = []
        for tile in scene.board.tiles:
            if tile.state != TileState.PAINT:
                continue
            # 人工划掉的格子不许进模板库：虚空化形态、通关展示图这类东西
            # 图案已经不是家族本身了，学进去会让这个家族永远认错。
            if _is_ignored(active, path.name, tile.row, tile.col):
                continue
            rect = tile.rect
            cell = image[rect.y : rect.bottom, rect.x : rect.right]
            match = gallery.identify(cell)
            if match is None:
                continue
            if bank.add(match.family, cell):
                added[match.family] = added.get(match.family, 0) + 1
            samples.append((path.name, tile.row, tile.col, match.family, tile.paint_level))
        if samples:
            record(active, samples)
    if added:
        bank.save()
    if save_ledger:
        save(active)
    if verbose:
        print(f"模板新增: {added or '无（都满了或不认识）'}")
        print(bank.stats_text())
    return added


def scan_folder(folder: Path, *, store: dict | None = None, verbose: bool = True) -> tuple[dict, int]:
    """扫一个目录下的全部图片，增量更新台账。"""
    active = store if store is not None else load()
    paths = sorted(
        path
        for path in Path(folder).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    added = 0
    for path in paths:
        samples = scan_image(path)
        if not samples:
            if verbose:
                print(f"  跳过（没识别到棋盘）: {path.name}")
            continue
        new = record(active, samples)
        added += new
        if verbose:
            named = sum(1 for item in samples if item[3])
            print(f"  {path.name}: {len(samples)} 格，认名 {named}，新增 {new}")
    return active, added


def matrix(store: dict) -> dict[str, dict[int, int]]:
    """家族 → 等级 → 样本数。没有样本的等级不出现，方便直接 ``.get`` 判空。"""
    result: dict[str, dict[int, int]] = {}
    for family, table in store["samples"].items():
        row: dict[int, int] = {}
        for level_text, items in table.items():
            try:
                level = int(level_text)
            except ValueError:
                continue
            row[level] = len(items)
        result[family] = row
    return result


def coverage(store: dict) -> tuple[int, int, list[tuple[str, int]]]:
    """返回 ``(已覆盖组合数, 应有组合数, 缺口列表)``。"""
    table = matrix(store)
    have = 0
    missing: list[tuple[str, int]] = []
    for family in paint_kind.known_kinds():
        row = table.get(family, {})
        for level in expected_levels(family):
            if row.get(level):
                have += 1
            else:
                missing.append((family, level))
    return have, expected_total(), missing


def report_text(store: dict) -> str:
    """人看的矩阵报告：一行一个家族，一列一个等级。"""
    table = matrix(store)
    families = list(paint_kind.known_kinds())
    # 只列已知家族里出现过的；顺序按目录（怪物 13 → 玩法 14），方便对照解包。
    header = "家族".ljust(10) + "".join(f"Lv{level}".rjust(6) for level in LEVELS)
    lines = [header, "-" * len(header.encode("gbk", errors="replace"))]
    for family in families:
        row = table.get(family, {})
        cells = []
        for level in LEVELS:
            if level not in expected_levels(family):
                cells.append("—".rjust(6))  # 赛季没有这一级
            elif row.get(level):
                cells.append(str(row[level]).rjust(6))
            else:
                cells.append("·".rjust(6))
        lines.append(family.ljust(10) + "".join(cells))

    have, total, missing = coverage(store)
    lines.append("")
    lines.append(f"覆盖 {have}/{total} 组合（{have * 100.0 / max(total, 1):.1f}%）")

    per_level = []
    for level in LEVELS:
        families_at_level = [name for name in families if level in expected_levels(name)]
        hit = sum(1 for name in families_at_level if table.get(name, {}).get(level))
        per_level.append(f"Lv{level} {hit}/{len(families_at_level)}")
    lines.append("按等级: " + "  ".join(per_level))

    if missing:
        by_family: dict[str, list[int]] = {}
        for family, level in missing:
            by_family.setdefault(family, []).append(level)
        lines.append("")
        lines.append("缺口（按家族）:")
        for family, levels in by_family.items():
            lines.append(f"  {family}: " + " ".join(f"Lv{level}" for level in levels))

    unknown = store["unknown"]
    if unknown:
        total_unknown = sum(len(items) for items in unknown.values())
        detail = " ".join(
            f"Lv{level}×{len(items)}" for level, items in sorted(unknown.items(), key=lambda kv: kv[0])
        )
        lines.append("")
        lines.append(f"未认名格 {total_unknown} 个（待补模板/待排除）: {detail}")
    return "\n".join(lines)


def draw_matrix(store: dict, out_png: Path) -> Path:
    """把矩阵画成图：绿=有样本、灰=缺、虚线格=该家族没有这一级。

    用 Pillow 画而不是 OpenCV——OpenCV 的 ``putText`` 不支持中文，
    家族名会全变成问号。
    """
    from PIL import Image, ImageDraw, ImageFont

    families = list(paint_kind.known_kinds())
    cell_w, cell_h = 64, 34
    name_w, head_h = 132, 42
    width = name_w + cell_w * len(LEVELS) + 12
    height = head_h + cell_h * len(families) + 44
    canvas = Image.new("RGB", (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    font_path = "C:/Windows/Fonts/msyh.ttc"
    try:
        font = ImageFont.truetype(font_path, 15)
        small = ImageFont.truetype(font_path, 13)
    except OSError:  # 非 Windows：退回默认字体，中文会掉成方块但不至于崩
        font = ImageFont.load_default()
        small = font

    table = matrix(store)
    draw.text((10, 12), "颜料覆盖矩阵", font=font, fill=(240, 240, 240))
    for index, level in enumerate(LEVELS):
        x = name_w + index * cell_w + cell_w // 2
        draw.text((x, 14), f"Lv{level}", font=small, fill=(200, 200, 200), anchor="mm")

    for row_index, family in enumerate(families):
        top = head_h + row_index * cell_h
        draw.text((10, top + cell_h // 2), family, font=font, fill=(225, 225, 225), anchor="lm")
        row = table.get(family, {})
        for index, level in enumerate(LEVELS):
            left = name_w + index * cell_w
            box = (left + 3, top + 3, left + cell_w - 3, top + cell_h - 3)
            if level not in expected_levels(family):
                draw.rectangle(box, outline=(60, 60, 66))
                continue
            count = row.get(level, 0)
            if count:
                shade = min(90 + count * 40, 210)
                fill = (46, shade, 96)
            else:
                fill = (46, 46, 52)
            draw.rectangle(box, fill=fill)
            if count:
                draw.text(
                    (left + cell_w // 2, top + cell_h // 2), str(count), font=small,
                    fill=(250, 250, 250), anchor="mm",
                )

    have, total, _missing = coverage(store)
    draw.text(
        (10, height - 30),
        f"覆盖 {have}/{total} 组合（{have * 100.0 / max(total, 1):.1f}%）  绿=有样本  灰=缺  空框=该家族无此级",
        font=small,
        fill=(190, 190, 190),
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png


def _tile_crops(image_path: Path) -> dict[tuple[int, int], Any]:
    """重新分析一张图，取 ``(行, 列) → 格子图``。"""
    from . import vision

    image = _read_image(image_path)
    if image is None:
        return {}
    scene = vision.analyze_scene(image)
    if scene is None or scene.board is None:
        return {}
    crops = {}
    for tile in scene.board.tiles:
        rect = tile.rect
        crops[(tile.row, tile.col)] = image[rect.y : rect.bottom, rect.x : rect.right]
    return crops


#: 单格样本图的落盘位置：``runtime/paint_samples/<家族>/LvN__<来源>_rXcY.png``。
#: 放 runtime 而不是 assets，是因为它是「从截图里裁出来的产物」，随截图增删而变，
#: 不该像 assets 那样当成随包发布的只读资产。
DEFAULT_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "runtime" / "paint_samples"


def _safe_stem(text: str) -> str:
    """把图名变成能当文件名的片段：去掉扩展名和 Windows 禁用字符，中文原样保留。"""
    stem = Path(text).stem
    for bad in '<>:"/\\|?*':
        stem = stem.replace(bad, "_")
    return stem.strip() or "sample"


def export_samples(
    store: dict | None = None,
    out_dir: Path | None = None,
    *,
    clean: bool = False,
    verbose: bool = True,
) -> dict:
    """把台账里的每个样本格裁成独立 PNG，按 ``家族 / LvN`` 归档。

    台账只记「图名 + 行 + 列」，要核对「这格到底是不是黑帮 Lv3」就得回原图数格子；
    模板库那边存的是压缩后的裸灰度数组，更没法用眼睛看。所以这里按组合落成文件。

    命名 ``Lv3__重新获取随机调色颜料之前_r3c2.png``：等级在前（同目录内天然按等级
    排序），后面跟来源图和棋盘坐标，任何一张都能一路追回原始截图。

    返回 ``{"dir", "written", "skipped", "combinations", "index"}``。
    """
    target = Path(out_dir) if out_dir is not None else DEFAULT_SAMPLES_DIR
    ledger = store if store is not None else load()

    if clean and target.exists():
        # 只清自己命名模式的文件，不递归删目录——万一 out_dir 传错也不会伤到别的东西。
        for stale in target.glob("*/Lv*__*.png"):
            try:
                stale.unlink()
            except OSError:
                pass

    # 先把「要裁哪些格」按图归拢，这样每张截图只重新分析一次。
    wanted: dict[str, list[tuple[str, int, int, int]]] = {}
    for family, table in ledger["samples"].items():
        for level_text, items in table.items():
            try:
                level = int(level_text)
            except ValueError:
                continue
            for image_name, row, col in items:
                wanted.setdefault(image_name, []).append((family, row, col, level))

    written = 0
    skipped = 0
    index: dict[str, dict[str, list[str]]] = {}
    for image_name in sorted(wanted):
        image_path = Path("jietu") / image_name
        if not image_path.is_file():
            image_path = Path(image_name)
        crops = _tile_crops(image_path) if image_path.is_file() else {}
        if not crops:
            skipped += len(wanted[image_name])
            if verbose:
                print(f"  跳过 {image_name}：读不到或认不出棋盘")
            continue
        for family, row, col, level in wanted[image_name]:
            crop = crops.get((row, col))
            if crop is None or crop.size == 0:
                skipped += 1
                continue
            folder = target / family
            folder.mkdir(parents=True, exist_ok=True)
            filename = f"Lv{level}__{_safe_stem(image_name)}_r{row}c{col}.png"
            destination = folder / filename
            ok, buffer = cv2.imencode(".png", crop)
            if not ok:
                skipped += 1
                continue
            try:
                destination.write_bytes(buffer.tobytes())
            except OSError:
                skipped += 1
                continue
            written += 1
            index.setdefault(family, {}).setdefault(str(level), []).append(
                str(destination.relative_to(target)).replace("\\", "/")
            )

    payload = {
        "version": 1,
        "note": "家族/LvN__来源图_r行c列.png；由 paint_coverage export 生成",
        "dir": str(target),
        "written": written,
        "skipped": skipped,
        "combinations": sum(len(levels) for levels in index.values()),
        "index": index,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    draw_sample_index(target, index, target / "index.png")

    if verbose:
        print(
            f"导出 {written} 张单格图，覆盖 {payload['combinations']} 个组合，跳过 {skipped} 格\n"
            f"目录 {target}\n清单 {target / 'index.json'}\n总览 {target / 'index.png'}"
        )
    return payload


def draw_sample_index(target: Path, index: dict[str, dict[str, list[str]]], out_png: Path) -> Path:
    """画一张「家族 × 等级」总览，每格贴第一张样本的缩略图。

    和 ``draw_matrix`` 的区别：那张只看「有没有」，这张能直接看出「长得对不对」——
    同一行七格的图案应该高度相似（外框变色、中央图案不变），一眼就能发现串族。
    """
    from PIL import Image, ImageDraw, ImageFont

    families = list(paint_kind.known_kinds())
    thumb = 92
    cell_w, cell_h = 96, 96
    name_w, head_h = 150, 46
    width = name_w + cell_w * len(LEVELS) + 12
    height = head_h + cell_h * len(families) + 40
    canvas = Image.new("RGB", (width, height), (22, 22, 26))
    draw = ImageDraw.Draw(canvas)
    font_path = "C:/Windows/Fonts/msyh.ttc"
    try:
        font = ImageFont.truetype(font_path, 15)
        small = ImageFont.truetype(font_path, 12)
    except OSError:
        font = ImageFont.load_default()
        small = font

    draw.text((10, 14), "样本总览（家族 × 等级）", font=font, fill=(240, 240, 240))
    for index_level, level in enumerate(LEVELS):
        x = name_w + index_level * cell_w + cell_w // 2
        draw.text((x, 16), f"Lv{level}", font=small, fill=(205, 205, 205), anchor="mm")

    for row_index, family in enumerate(families):
        top = head_h + row_index * cell_h
        draw.text((10, top + cell_h // 2), family, font=font, fill=(228, 228, 228), anchor="lm")
        levels = index.get(family, {})
        for index_level, level in enumerate(LEVELS):
            left = name_w + index_level * cell_w
            box = (left + 2, top + 2, left + cell_w - 2, top + cell_h - 2)
            if level not in expected_levels(family):
                draw.rectangle(box, outline=(56, 56, 62))
                continue
            files = levels.get(str(level), [])
            if not files:
                draw.rectangle(box, fill=(40, 40, 46))
                continue
            draw.rectangle(box, fill=(34, 60, 52))
            sample_path = target / files[0]
            try:
                tile = Image.open(sample_path).convert("RGB")
                tile.thumbnail((thumb, thumb), Image.LANCZOS)
                offset_x = left + (cell_w - tile.width) // 2
                offset_y = top + (cell_h - tile.height) // 2
                canvas.paste(tile, (offset_x, offset_y))
            except (OSError, ValueError):
                draw.text(
                    (left + cell_w // 2, top + cell_h // 2), "?", font=small,
                    fill=(230, 120, 120), anchor="mm",
                )
            if len(files) > 1:
                draw.text(
                    (left + cell_w - 6, top + cell_h - 8), f"×{len(files)}", font=small,
                    fill=(250, 250, 250), anchor="rb",
                )

    draw.text(
        (10, height - 28),
        "行内图案应高度相似（外框变色、中央图案不变）；×N = 该组合的样本数",
        font=small,
        fill=(185, 185, 185),
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png


def verify_samples(
    out_dir: Path | None = None,
    *,
    verbose: bool = True,
) -> dict:
    """用官方图标复核导出的样本，挑出「目录名和图标结论对不上」的格子。

    这条检查的意义在于**独立来源**：模板库本身就是拿这些样本学出来的，用它验证
    自己是循环论证；27 张官方图标则是完全独立的标准照。

    图标认不出的样本（细节少的齿轮、狮头那类 good 上不去）不算错，单独计数——
    它们只是「没证据」，不是「反证」。
    """
    target = Path(out_dir) if out_dir is not None else DEFAULT_SAMPLES_DIR
    consistent = 0
    conflicts: list[dict[str, Any]] = []
    unclear = 0
    for folder in sorted(p for p in target.iterdir() if p.is_dir()) if target.exists() else []:
        family = folder.name
        for sample in sorted(folder.glob("Lv*.png")):
            buffer = np.fromfile(str(sample), dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
            if image is None:
                continue
            match = paint_icon.identify(image)
            if match is None:
                unclear += 1
                continue
            if match.family == family:
                consistent += 1
            else:
                conflicts.append(
                    {
                        "expected": family,
                        "actual": match.family,
                        "file": str(sample.relative_to(target)).replace("\\", "/"),
                        "good": match.good,
                        "share": round(match.share, 3),
                    }
                )

    payload = {
        "consistent": consistent,
        "conflicts": conflicts,
        "unclear": unclear,
        "checked": consistent + len(conflicts) + unclear,
    }
    if verbose:
        print(
            f"复核 {payload['checked']} 张：一致 {consistent}，冲突 {len(conflicts)}，"
            f"图标给不出结论 {unclear}"
        )
        for item in conflicts:
            print(
                f"  冲突 {item['expected']} → {item['actual']}  {item['file']}  "
                f"good={item['good']} share={item['share']}"
            )
    return payload


def ignore(store: dict, image_name: str, row: int, col: int, reason: str = "非颜料") -> bool:
    """把某格永久划出「待认名」：它是通关展示图、空白地块这类不是颜料的东西。

    记下来而不是删掉，是为了下次重扫时不再把它捞回来——``record`` 和
    :func:`unknown_cells` 都会看这份名单。
    """
    key = [image_name, row, col, reason]
    if any(item[:3] == key[:3] for item in store["ignored"]):
        return False
    store["ignored"].append(key)
    for bucket in store["unknown"].values():
        for item in list(bucket):
            if item[0] == image_name and item[1] == row and item[2] == col:
                bucket.remove(item)
    return True


def _is_ignored(store: dict, image_name: str, row: int, col: int) -> bool:
    return any(item[0] == image_name and item[1] == row and item[2] == col for item in store["ignored"])


def unknown_cells(store: dict) -> list[tuple[str, int, int, int | None]]:
    """台账里所有未认名的格子，``(图名, 行, 列, 等级)``。"""
    cells: list[tuple[str, int, int, int | None]] = []
    for level_text, items in store["unknown"].items():
        try:
            level = int(level_text)
        except ValueError:
            level = None
        for image_name, row, col in items:
            if _is_ignored(store, image_name, row, col):
                continue
            cells.append((image_name, row, col, level))
    return cells


def unknown_montage(
    store: dict,
    out_png: Path,
    out_json: Path,
    *,
    folder: Path = Path("jietu"),
) -> dict:
    """把「未认名」的格子拼成一张带编号的图，供一次性命名。

    只收**真像颜料**的格子（过 :func:`paint_kind.looks_like_paint_tile`）：
    结束态截图里 17 个空白格会被判成 PAINT 但没有等级，它们不该出现在
    命名清单里，否则每次都要人肉跳过一遍。

    按图案聚类去重：同一家族的多个格子只占一行，填一次名覆盖全组。
    每个槽位下面写 ``编号 图名 r,c``，方便对照原图复核。
    """
    from PIL import Image, ImageDraw, ImageFont

    folder = Path(folder)
    cache: dict[str, dict[tuple[int, int], Any]] = {}
    entries: list[dict] = []
    skipped: list[dict] = []
    cells: list[np.ndarray] = []
    for image_name, row, col, level in unknown_cells(store):
        crops = cache.get(image_name)
        if crops is None:
            crops = _tile_crops(folder / image_name)
            cache[image_name] = crops
        cell = crops.get((row, col))
        if cell is None or cell.size == 0:
            continue
        if not paint_kind.looks_like_paint_tile(cell):
            skipped.append({"image": image_name, "row": row, "col": col, "level": level, "why": "不像颜料"})
            ignore(store, image_name, row, col, "不像颜料")
            continue
        entries.append({"image": image_name, "row": row, "col": col, "level": level, "cell": len(cells)})
        cells.append(cell)

    groups = paint_kind.cluster_cells(cells) if cells else []
    # 按簇重排，簇内保持原顺序；同时把簇号写回条目。
    cluster_of: dict[int, int] = {}
    for cluster_index, group in enumerate(groups):
        for cell_index in group:
            cluster_of[cell_index] = cluster_index
    slots = sorted(entries, key=lambda item: (cluster_of.get(item["cell"], -1), item["cell"]))
    for entry in slots:
        entry["cluster"] = cluster_of.get(entry["cell"], -1)

    size, caption_h, columns = 150, 30, 7
    rows = max(1, (len(slots) + columns - 1) // columns)
    width, height = columns * size, rows * (size + caption_h) + 46
    canvas = Image.new("RGB", (width, height), (22, 22, 26))
    draw = ImageDraw.Draw(canvas)
    font_path = "C:/Windows/Fonts/msyh.ttc"
    try:
        font = ImageFont.truetype(font_path, 15)
        small = ImageFont.truetype(font_path, 11)
    except OSError:
        font = ImageFont.load_default()
        small = font
    draw.text((8, 12), "未认名颜料格（同一编号=同一簇，填一次名覆盖全组）", font=font, fill=(235, 235, 235))

    for slot, entry in enumerate(slots):
        top = 40 + (slot // columns) * (size + caption_h)
        left = (slot % columns) * size
        crop = cv2.resize(cells[entry["cell"]], (size, size), interpolation=cv2.INTER_CUBIC)
        canvas.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (left, top))
        draw.rectangle((left, top, left + size - 1, top + size - 1), outline=(70, 70, 78))
        draw.text((left + 5, top + 4), str(slot), font=font, fill=(255, 220, 80))
        level_text = f"Lv{entry['level']}" if entry["level"] else "Lv?"
        draw.text(
            (left + 5, top + size + 5),
            f"{level_text}  {entry['image'][:8]} r{entry['row']}c{entry['col']}",
            font=small,
            fill=(170, 220, 170),
        )
        entry["slot"] = slot

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    payload = {
        "note": "给每个 slot 填 name（家族名，留空=跳过）；同一 slot 组内格子会一起入库",
        "slots": slots,
        "skipped": skipped,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def apply_unknown(
    json_path: Path,
    *,
    store: dict | None = None,
    folder: Path = Path("jietu"),
    save_ledger: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """把人工填好的未知格命名收尾：入模板库 + 台账从 ``unknown`` 迁到 ``samples``。

    ``slots[i].name`` 填家族名即可。同一簇的其它格子（同 cluster 的其它 slot）
    会一起入库、一起迁移——它们图案相同，只是等级和位置不同。
    """
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    slots = payload.get("slots") or []
    names_by_cluster: dict[int | None, str] = {}
    for slot in slots:
        name = (slot.get("name") or "").strip()
        if slot.get("cluster") is not None and name:
            names_by_cluster.setdefault(slot["cluster"], name)

    active = store if store is not None else load()
    bank = paint_kind.reload_bank()
    added: dict[str, int] = {}
    cache: dict[str, dict[tuple[int, int], Any]] = {}
    for slot in slots:
        cluster = slot.get("cluster")
        name = (slot.get("name") or "").strip() or names_by_cluster.get(cluster) or ""
        if not name:
            continue
        if paint_kind.family_key(name) is None:
            if verbose:
                print(f"  跳过：{name} 不在官方目录里")
            continue
        image_name = slot["image"]
        crops = cache.get(image_name)
        if crops is None:
            crops = _tile_crops(Path(folder) / image_name)
            cache[image_name] = crops
        cell = crops.get((slot["row"], slot["col"]))
        if cell is None:
            continue
        if bank.add(name, cell):
            added[name] = added.get(name, 0) + 1
        # 台账迁移：从 unknown 里摘掉，挂到 samples 的这个家族/等级下。
        level = slot.get("level")
        level_text = str(level) if level is not None else "?"
        bucket = active["unknown"].get(level_text)
        record_key = [image_name, slot["row"], slot["col"]]
        if bucket and record_key in bucket:
            bucket.remove(record_key)
        record(active, [(image_name, slot["row"], slot["col"], name, level)])

    bank.save()
    if save_ledger:
        save(active)
    if verbose:
        print(f"入库: {added or '无新增'}")
        print(bank.stats_text())
    return added


def _split_flags(values: list[str]) -> tuple[list[str], list[str]]:
    """把手写 CLI 的开关和位置参数分开，返回 ``(位置参数, 开关)``。

    这个模块的 ``main`` 不走 argparse，直接按位置取 ``args[1]`` 当目录。于是
    ``export --clean`` 会把 ``--clean`` 当成目录名，在仓库根目录建出一个真叫
    ``--clean`` 的文件夹——看着像成功，其实样本全导到了错地方，而且还因为
    ``clean`` 开关没生效，旧样本一张都没删。开关必须先从位置参数里摘出去。
    """
    positionals = [item for item in values if not item.startswith("--")]
    flags = [item for item in values if item.startswith("--")]
    return positionals, flags


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "report"
    rest, flags = _split_flags(args[1:])
    ledger = DEFAULT_LEDGER

    if command == "scan":
        folder = Path(rest[0]) if rest else Path("jietu")
        store, added = scan_folder(folder)
        save(store, ledger)
        have, total, _missing = coverage(store)
        print(f"\n新增 {added} 条样本，台账 {ledger}")
        print(f"覆盖 {have}/{total} 组合")
        print(report_text(store))
        draw_matrix(store, ledger.with_suffix(".png"))
        print(f"矩阵图 {ledger.with_suffix('.png')}")
        return 0

    if command == "harvest":
        folder = Path(rest[0]) if rest else Path("jietu")
        harvest_icons(folder)
        store = load(ledger)
        print(report_text(store))
        draw_matrix(store, ledger.with_suffix(".png"))
        return 0

    if command == "report":
        store = load(ledger)
        print(report_text(store))
        draw_matrix(store, ledger.with_suffix(".png"))
        print(f"\n矩阵图 {ledger.with_suffix('.png')}")
        return 0

    if command == "export":
        out_dir = Path(rest[0]) if rest else DEFAULT_SAMPLES_DIR
        export_samples(out_dir=out_dir, clean="--clean" in flags)
        return 0

    if command == "verify":
        out_dir = Path(rest[0]) if rest else DEFAULT_SAMPLES_DIR
        payload = verify_samples(out_dir)
        return 0 if not payload["conflicts"] else 1

    if command == "unknown":
        store = load(ledger)
        out_png = ledger.with_name("paint_unknown.png")
        out_json = ledger.with_name("paint_unknown.json")
        payload = unknown_montage(store, out_png, out_json)
        save(store, ledger)
        print(f"{len(payload['slots'])} 个待命名槽位（跳过 {len(payload['skipped'])} 个非颜料格）")
        print(f"拼图 {out_png}\n清单 {out_json}")
        return 0

    if command == "apply":
        if not rest:
            print("用法: python -m huahua_assistant.paint_coverage apply <json>")
            return 2
        apply_unknown(Path(rest[0]))
        store = load(ledger)
        draw_matrix(store, ledger.with_suffix(".png"))
        print(report_text(store))
        return 0

    if command == "ignore":
        if len(rest) < 3:
            print("用法: python -m huahua_assistant.paint_coverage ignore <图名> <行> <列> [原因]")
            return 2
        store = load(ledger)
        reason = rest[3] if len(rest) > 3 else "非颜料"
        ok = ignore(store, rest[0], int(rest[1]), int(rest[2]), reason)
        save(store, ledger)
        print(f"{'已划出' if ok else '本来就在名单里'}: {rest[0]} r{rest[1]}c{rest[2]}（{reason}）")
        return 0

    if command == "relabel":
        if len(rest) < 5:
            print("用法: python -m huahua_assistant.paint_coverage relabel <图名> <行> <列> <家族> <等级>")
            return 2
        store = load(ledger)
        moved = relabel(store, rest[0], int(rest[1]), int(rest[2]), rest[3], int(rest[4]))
        save(store, ledger)
        where = "、".join(f"{family} Lv{level}" for family, level in moved) or "（原本不在册）"
        print(f"已改判: {rest[0]} r{rest[1]}c{rest[2]} {where} -> {rest[3]} Lv{rest[4]}")
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
