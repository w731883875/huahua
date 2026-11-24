from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from difflib import SequenceMatcher
import ctypes
import queue
import re
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .capture import write_image
from .game_model import BRUSH_TYPE_ORDER, BrushType, build_affix_catalog
from .keyword_match import match_keywords
from .layout import (
    ScreenLayout,
    calibrated_layout,
    last_layout_match,
    load_layout,
    save_layout,
)
from . import frame_level
from . import brush_panel
from . import paint_frame
from . import paint_icon
from . import paint_kind
from . import paint_panel
from . import presence
from . import tile_reader
from .models import (
    BackpackObservation,
    BackpackPaintObservation,
    BoardObservation,
    BrushOptionsObservation,
    CurrentPaintObservation,
    PaintCategory,
    PaintDirection,
    PaintObject,
    Rect,
    RegionKind,
    RegionObservation,
    SceneObservation,
    TileObservation,
    TileState,
)
from .windows_ocr import recognize_chinese_lines_async, warm_windows_ocr_async


BOARD_SIZE = 6
MIN_CAPTURE_SIDE = 320
MIN_BOARD_DETECTION_SCORE = 0.70
#: 银框兜底时，角标的最高等级分要比 Lv1 高出这么多才敢翻案。
#: 实测：真 Lv1 的 Lv1 分 1.00、次高 0.84（差 -0.15）；Lv7 宝箱的 Lv1 只有 0.73、
#: Lv7 有 0.91~0.94（差 +0.15~+0.21）。取 0.10 两边都留了余量。
SILVER_LABEL_MARGIN = 0.10
CURRENT_PAINT_TEMPLATE_DIR = Path(__file__).resolve().parent / "assets" / "current_paint"
CURRENT_DIRECTION_TEMPLATE_FILES = {
    "横向": "direction_horizontal.png",
    "相邻": "direction_adjacent.png",
    "纵向": "direction_vertical.png",
    "对角斜向": "direction_diagonal.png",
}
CURRENT_TARGET_TEMPLATE_FILES = {
    "玩法": ("target_gameplay.png", "target_gameplay_red.png"),
    "怪物": ("target_monster.png",),
    "所有": ("target_all.png",),
}
CURRENT_PAINT_CANONICAL_TEXT_SIZE = (725, 97)
CURRENT_PAINT_KEYWORD_ROI = (180, 25, 520, 70)
BRUSH_STAGE2_TEMPLATE_DIR = Path(__file__).resolve().parent / "assets" / "brush_stage2"
BRUSH_STAGE2_TEMPLATE_FILES = {
    "随机改变自身的内容，等级不变": "transform_random_content.png",
    "当前颜料等级随机改变": "transform_random_level.png",
    "作用方向改为横向": "transform_direction_horizontal.png",
    "重新获得一个随机初始颜料": "transform_reroll_current.png",
    "自身提升2级": "stack_upgrade_self_2.png",
    "范围内随机1个颜料提升1级": "stack_upgrade_scope_random_1.png",
    "将距离最近的1个阻挡地块转变为空白地块": "spread_clear_nearest_blocked.png",
    "当前颜料可放置在阻挡地块": "spread_place_blocked.png",
}
BRUSH_STAGE2_CANONICAL_PANEL_SIZE = (755, 251)
BRUSH_STAGE2_TEXT_RECTS = (
    (85, 42, 555, 76),
    (85, 137, 555, 82),
)
#: 光栅模板只在「面板形状接近模板拍摄时」才可信。模板是从 1673x878 窗口上截的
#: （面板 760x251，宽高比 3.03），文字框是写死的像素坐标；窗口尺寸一换，
#: 同样的裁剪框就压到别的行上——2026-11-20 在 1607x822 的截图上实测，一张
#: 明明是「转变自身的作用方向」的卡片被读成「作用方向改为横向 0.88」，是纯粹的
#: 假匹配。宁可不用模板，也不能把错词缀写进录制。
BRUSH_STAGE2_CANONICAL_ASPECT = BRUSH_STAGE2_CANONICAL_PANEL_SIZE[0] / BRUSH_STAGE2_CANONICAL_PANEL_SIZE[1]
BRUSH_STAGE2_ASPECT_TOLERANCE = 0.18


def _brush_panel_matches_template_geometry(crop: np.ndarray) -> bool:
    if crop.size == 0:
        return False
    aspect = crop.shape[1] / max(1, crop.shape[0])
    low = BRUSH_STAGE2_CANONICAL_ASPECT * (1.0 - BRUSH_STAGE2_ASPECT_TOLERANCE)
    high = BRUSH_STAGE2_CANONICAL_ASPECT * (1.0 + BRUSH_STAGE2_ASPECT_TOLERANCE)
    return low <= aspect <= high


def _warm_numpy_median() -> None:
    """预加载 ``numpy.ma``，避免首次 ``np.median`` 落在实时路径上。

    ``np.median`` 在 numpy 里是懒加载入口：第一次调用才会 import ``numpy.ma``，
    实测 0.048s。实时首帧预算只有 0.20s，这一下就能占掉四分之一。模块加载阶段
    一次性付掉，之后刷色板色相、背包格子间距等处的 ``np.median`` 都走热路径。
    """

    np.median(np.zeros(2, dtype=np.float64))


_warm_numpy_median()


class VisionTracker:
    def __init__(self) -> None:
        self.board_rect: Rect | None = None
        self.last_scene: SceneObservation | None = None
        self.frame_size: tuple[int, int] | None = None

    def _frame_size_changed(self, image: np.ndarray) -> bool:
        """窗口尺寸变了吗（拖窗口 / 最大化 / 换分辨率 / 改 DPI）。

        1x1 的 ``paused:game-not-foreground`` 占位帧不算：游戏最小化时采集会退回它，
        不能让它把尺寸记录冲掉、也不能触发一次无意义的重新定位。
        """
        height, width = image.shape[:2]
        if width < 100 or height < 100:
            return False
        size = (int(width), int(height))
        changed = self.frame_size is not None and self.frame_size != size
        self.frame_size = size
        return changed

    def analyze(self, image: np.ndarray) -> SceneObservation | None:
        scene = analyze_scene(image, previous_board_rect=self.board_rect)
        if scene is not None:
            self.board_rect = scene.board.rect
            self.last_scene = scene
        return scene

    def analyze_quick(self, image: np.ndarray) -> SceneObservation | None:
        if self._frame_size_changed(image):
            # 快路径会复用上一帧的棋盘矩形。窗口一拖，那个矩形属于旧尺寸，拿它
            # 裁新画面逐格全错位，却仍返回一个「看起来成功」的 scene：自愈分支
            # 永远进不去、标定永远落不了盘（2026-11-15 23:34 实机：1536x877 →
            # 1765x965 后 paint 从 5 掉到 1，layout.json 的 mtime 纹丝不动）。
            # 尺寸一变必须强制重新定位一次，让自愈按新尺寸落盘标定。
            self.board_rect = None
            self.last_scene = None
            return self.analyze_realtime(image)
        previous_scene = self.last_scene
        scene = analyze_scene_quick(
            image,
            previous_scene=previous_scene,
            previous_board_rect=self.board_rect,
        )
        if (
            scene is not None
            and scene.current_paint is None
            and scene.backpack is not None
            and len(scene.backpack.paints) >= 5
        ):
            recovered = analyze_scene_realtime(image)
            if recovered is not None and recovered.current_paint is not None:
                scene = recovered
        if scene is not None:
            self.board_rect = scene.board.rect
            self.last_scene = scene
        return scene

    def analyze_realtime(self, image: np.ndarray) -> SceneObservation | None:
        scene = analyze_scene_realtime(
            image,
            previous_scene=self.last_scene,
            previous_board_rect=self.board_rect,
        )
        if scene is not None:
            self.board_rect = scene.board.rect
            self.last_scene = scene
        return scene

    def reset(self) -> None:
        self.board_rect = None
        self.last_scene = None
        self.frame_size = None


def analyze_frame(image: np.ndarray) -> BoardObservation | None:
    scene = analyze_scene(image)
    return scene.board if scene is not None else None


def analyze_scene(
    image: np.ndarray,
    previous_board_rect: Rect | None = None,
    *,
    icon_fallback: bool = True,
) -> SceneObservation | None:
    """单张截图分析（离线/冷启动）。默认开官方图标兜底，见 ``_classify_tiles``。"""
    if not _is_supported_scene_image(image):
        return None
    if previous_board_rect is not None:
        scene = _analyze_scene_from_board_rect(
            image, previous_board_rect, fast=True, icon_fallback=icon_fallback
        )
        # A wrong-but-grid-like candidate can score well if it lands on the
        # backpack/debug grid. Only reuse the tracked board when it is still a
        # strong match; otherwise do a full search and recover.
        if scene is not None and scene.board.score >= 0.76:
            return scene

    # 标定过的窗口尺寸：直接按标定出框，不再精修、不再搜索。
    # 2026-11-20 实测（1673x878）：这条 0.15s，走下面的全搜索 4.3s，
    # 而全搜索最后交出来的还是同一个框——冷启动「完整识别」的一半时间在这里。
    calibrated = _calibrated_board_rect(image)
    if calibrated is not None:
        rect, score = calibrated
        return _analyze_scene_from_board_rect(
            image, rect, fast=True, board_score=score, icon_fallback=icon_fallback
        )

    located = _locate_board_rect(image)
    if located is None:
        return None
    rect, score = located
    return _analyze_scene_from_board_rect(
        image, rect, fast=False, board_score=score, icon_fallback=icon_fallback
    )


#: 棋盘 36 格「格内色彩标准差中位数」的上限。
#: 真实棋盘每格是一块纯色（颜料/空白/虚空/阻挡）；噪声与花哨 UI 的格内是随机纹理。
#: 实测（2026-11-15）：真实截图 19.4 / 22.8，纯噪声 73.6 —— 取 45 留两倍余量。
BOARD_CELL_FLATNESS_MAX = 45.0
#: 「整张画面根本不像棋盘」的下限。只用于在**全搜索之前**提前收手，不参与候选筛选。
#: 必须比 ``BOARD_CELL_FLATNESS_MAX`` 松得多：未 refine 的候选框会跨到格子线上，
#: 真实棋盘的候选实测能到 49.0，拿 45 去筛会误杀；噪声则一律 73 以上。
BOARD_CELL_SCRAMBLED = 60.0


def _board_cell_flatness(image: np.ndarray, rect: Rect) -> float:
    """36 格各取中心 60% 区域，返回「每格色彩标准差」的中位数。

    用中位数而不是均值：真棋盘上总有一两格压着带细节的图标（实测最大 66~71），
    均值会被它拉高；噪声图则是**每一格**都乱，中位数直接顶到 73 以上。
    """
    height, width = image.shape[:2]
    cell_w = rect.w / BOARD_SIZE
    cell_h = rect.h / BOARD_SIZE
    half_w = cell_w * 0.30
    half_h = cell_h * 0.30
    values: list[float] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cx = rect.x + cell_w * (col + 0.5)
            cy = rect.y + cell_h * (row + 0.5)
            x0 = int(max(0, cx - half_w))
            x1 = int(min(width, cx + half_w))
            y0 = int(max(0, cy - half_h))
            y1 = int(min(height, cy + half_h))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            patch = image[y0:y1, x0:x1].reshape(-1, image.shape[2])
            values.append(float(patch.std(axis=0).mean()))
    if not values:
        return 0.0
    values.sort()
    return values[len(values) // 2]


def _board_cells_look_flat(image: np.ndarray, rect: Rect) -> bool:
    return _board_cell_flatness(image, rect) <= BOARD_CELL_FLATNESS_MAX


def _locate_board_rect(image: np.ndarray) -> tuple[Rect, float] | None:
    """定位棋盘，并在慢的时候把这次走的路径记进诊断日志。

    只包一层：真正的实现在 :func:`_locate_board_rect_impl`（它会递归处理缩略图，
    递归那层不重复埋点）。
    """

    started = time.perf_counter()
    result = _locate_board_rect_impl(image)
    elapsed = time.perf_counter() - started
    if elapsed >= LOCATE_DIAGNOSTIC_SECONDS:
        height, width = image.shape[:2]
        _report_diagnostic(
            {
                "kind": "slow_locate",
                "elapsed": round(elapsed, 2),
                "width": int(width),
                "height": int(height),
                "found": result is not None,
                "rect": None if result is None else [result[0].x, result[0].y, result[0].w, result[0].h],
                "score": None if result is None else round(float(result[1]), 4),
            }
        )
    if result is None:
        _dump_locate_failure(image)
    return result


#: 定位失败时最多多久落一张原图。玩家报「识别不到棋盘」时，日志里只有
#: "found=False" 这几个字，看不出当时窗口是黑屏、弹窗盖住、还是缩放变了——
#: 留一张原图才有可能事后还原现场。但一秒一帧全存会塞满磁盘，所以限流。
LOCATE_FAIL_DUMP_INTERVAL = 60.0
_LAST_LOCATE_FAIL_DUMP = 0.0


def _dump_locate_failure(image: np.ndarray) -> None:
    """定位失败时把当前帧落一张 PNG（限流 :data:`LOCATE_FAIL_DUMP_INTERVAL`）。

    只写 ``runtime/_locate_fail.png`` 一张，覆盖写——需要的是「最近一次失败长什么样」，
    不是所有失败的历史。任何异常都吞掉：存诊断图绝不能让识别路径崩。
    """
    global _LAST_LOCATE_FAIL_DUMP
    now = time.monotonic()
    if now - _LAST_LOCATE_FAIL_DUMP < LOCATE_FAIL_DUMP_INTERVAL:
        return
    _LAST_LOCATE_FAIL_DUMP = now
    try:
        target = Path("runtime") / "_locate_fail.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_image(target, image)
    except Exception:  # noqa: BLE001 - 存诊断图失败不算识别失败
        pass


def _locate_board_rect_impl(image: np.ndarray) -> tuple[Rect, float] | None:
    """只回答「棋盘在哪、有多像」，不碰面板 OCR / 笔触 / 背包 / 逐格分类。

    两个调用方都只需要矩形：``analyze_scene``（离线冷启动）和布局自愈
    （``_recover_board_rect``）。后者原来直接调 ``analyze_scene``，于是窗口一改尺寸
    就把一万多条文案常量匹配和 1381 次模板匹配也跑一遍——1239x780 实测 10.2 秒，
    而纯定位只要 3.2 秒。实时采集跑在后台线程，界不冻，但用户要干等十几秒才恢复
    推荐；这段等待没有任何收益。

    门槛与 ``analyze_scene`` 原逻辑逐字一致，不在此处另行加减。
    """
    height, width = image.shape[:2]
    # 先按 layout.json 里已经标定过的窗口尺寸直接取框：一次十几毫秒，
    # 而下面「缩略图定位 + 版式候选逐个精修」实测 3~4s，全搜索更久。
    # 冷启动只要这个尺寸标定过，就没必要再搜一遍。
    calibrated = _calibrated_board_rect(image)
    if calibrated is not None:
        return calibrated
    # 大窗口上 ``_best_estimated_board_layout`` 要对十几个候选逐个跑
    # ``_refine_board_rect``（全图 Canny + 三重循环）：1765x965 实测 11.5s，
    # 加全搜索共 15.9s。拖一次窗口十几秒不出帧，玩家只会以为工具死了。
    # 棋盘是按 UI 比例缩放的固定结构，先在 800 宽缩略图上定位、映射回原图，
    # 再用**原图**重新过一遍全部判据；实测 3.9s，5 枚颜料与等级全对。
    if width > 1400:
        scale = 800.0 / width
        small = cv2.resize(
            image, (800, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA
        )
        small_located = _locate_board_rect_impl(small)
        if small_located is not None:
            small_rect, _ = small_located
            mapped = Rect(
                x=int(round(small_rect.x / scale)),
                y=int(round(small_rect.y / scale)),
                w=int(round(small_rect.w / scale)),
                h=int(round(small_rect.h / scale)),
            )
            score = _score_board_rect(image, mapped)
            if (
                score >= 0.68
                and _board_size_score(image.shape, mapped) >= 0.85
                and _board_cells_look_flat(image, mapped)
            ):
                return mapped, score
        # 缩略图没定位到、或原图复核不过：完整走下面的原流程，行为不变。

    layout_best = _best_estimated_board_layout(image)
    if layout_best is not None:
        refined_layout, layout_score = layout_best
        # 布局候选是「按 UI 版式推算」出来的，比全搜索的纯边缘匹配可靠得多。
        # 但边缘分数会随分辨率掉：2K 下同一块棋盘 layout 只有 0.712、全搜索更是
        # 掉到 0.64，两个门槛都过不去，直接判「无棋盘」。所以 layout 这条路单独
        # 放宽到 0.68，同时要求尺寸先验过关，避免又把「棋盘内部一圈」的小框放进来。
        layout_size_ok = _board_size_score(image.shape, refined_layout) >= 0.85
        if layout_score >= 0.68 and layout_size_ok:
            if _board_cells_look_flat(image, refined_layout):
                return refined_layout, layout_score
            # 版式对得上、分数还很高，但格内一塌糊涂 —— 这不是画画界面。
            # 直接判否：这里再跑一遍全搜索，只会用十几秒换回同样一个错框。
            return None
        if _board_cell_flatness(image, refined_layout) > BOARD_CELL_SCRAMBLED:
            # 分数不够格、格内又是一片杂乱：整张画面不是画画界面。
            # 提前收手，省掉随后的全搜索（噪声图上实测 12 秒，注定被同一个判据否掉）。
            return None

    rects = _candidate_board_rects(image)
    best: tuple[float, Rect] | None = None
    for rect in rects:
        score = _score_board_rect(image, rect)
        if best is None or score > best[0]:
            best = (score, rect)

    if best is None or best[0] < MIN_BOARD_DETECTION_SCORE:
        return None

    rect = _choose_board_rect(image, best[1], best[0])
    if not _board_cells_look_flat(image, rect):
        return None
    return rect, _score_board_rect(image, rect)


def _calibrated_board_rect(image: np.ndarray) -> tuple[Rect, float] | None:
    """按已标定的布局直接给出棋盘框，不做任何搜索。

    判据与实时路径一致（网格分 ≥ :data:`GRID_PRESENCE_SCORE`，再由
    ``_pick_board_rect`` 用棋盘自身的分数复核），所以不会因为「快」而放松：
    窗口尺寸没标定过、或棋盘不在标定位置时，老实返回 ``None`` 让调用方去搜。
    """
    height, width = image.shape[:2]
    screen = _screen_layout(width, height)
    rect = _pick_board_rect(image, screen, width, height, drawing=True)
    if rect is None:
        return None
    if _board_grid_score(image, rect, outer_penalty=False) < GRID_PRESENCE_SCORE:
        return None
    score = _score_board_rect(image, rect)
    if score < MIN_BOARD_DETECTION_SCORE:
        return None
    if not _board_cells_look_flat(image, rect):
        return None
    return rect, score


def _choose_board_rect(image: np.ndarray, best_rect: Rect, best_score: float) -> Rect:
    refined_best = _refine_board_rect(image, best_rect)
    refined_best_score = _score_board_rect(image, refined_best)
    layout_best = _best_estimated_board_layout(image)
    if layout_best is None:
        return refined_best
    refined_layout, layout_score = layout_best
    if layout_score >= max(best_score, refined_best_score) * 0.94:
        return refined_layout
    return refined_best


def analyze_scene_quick(
    image: np.ndarray,
    previous_scene: SceneObservation | None = None,
    previous_board_rect: Rect | None = None,
) -> SceneObservation | None:
    if not _is_supported_scene_image(image):
        return None
    if previous_scene is not None and previous_scene.region(RegionKind.START_DRAWING) is not None:
        return analyze_scene_realtime(
            image,
            previous_scene=previous_scene,
            previous_board_rect=previous_board_rect,
        )
    if previous_scene is not None and _find_start_drawing_button_fast(image) is not None:
        return analyze_scene_realtime(
            image,
            previous_scene=previous_scene,
            previous_board_rect=previous_board_rect,
        )
    rect = previous_scene.board.rect if previous_scene is not None else previous_board_rect
    if rect is None:
        return analyze_scene_realtime(image)
    if previous_scene is not None and not _tracked_board_visual_guard(image, rect, previous_scene.board):
        return None
    cached_board_score = previous_scene.board.score if previous_scene is not None else None
    scene = _analyze_scene_from_board_rect(
        image,
        rect,
        fast=True,
        quick=True,
        previous_scene=previous_scene,
        board_score=cached_board_score,
    )
    if scene is not None and scene.board.score >= MIN_BOARD_DETECTION_SCORE:
        return scene
    return None


def analyze_scene_realtime(
    image: np.ndarray,
    previous_scene: SceneObservation | None = None,
    previous_board_rect: Rect | None = None,
) -> SceneObservation | None:
    """Low-latency scene analysis for the interactive capture path.

    棋盘位置由布局直接算出（``layout.py``），不再每帧做多候选搜索；
    "这块区域是不是棋盘"先用 6x6 网格线判断（内容无关、约 5ms），
    再用梯度模板兜底（``presence.py``）。比整块评分便宜一个数量级。
    """

    if not _is_supported_scene_image(image):
        return None
    height, width = image.shape[:2]
    screen = _screen_layout(width, height)

    # 这个窗口尺寸在标定表里**没有任何匹配**时，固定比例候选往往「歪得刚好能过
    # presence」——于是被当成有效棋盘一路接受，每格都裁偏却读得很自信。
    #
    # 2026-11-24 实测：游戏窗口 1536x792，layout.json 里最近的一条是 1522x885，
    # 偏差分 0.1265 远超阈值 0.08，整盘用内置默认比例（x=0.187/y=0.256/
    # size=0.594）。自评分 0.83、unknown_tiles 0，但每格读数全错。这就是
    # 「昨晚稳定、今天全错」的真身：和代码改动无关，是窗口尺寸变了没标定。
    #
    # 所以只要没匹配就先自愈一次并落盘。自愈约 1.6 秒，之后同尺寸都走快路径。
    match = last_layout_match()
    if match is not None and not match.get("matched"):
        key = (width, height)
        # 同尺寸只自愈一次：自愈约 1.5 秒，如果每帧都跑，实时路径直接废掉
        # （实测 1.47s/帧）。落盘只在界面进程生效（``_LAYOUT_AUTOSAVE``），
        # 测试和子进程里存不下，所以必须另有进程内缓存兜底。
        cached = _RECOVERED_LAYOUTS.get(key)
        if cached is not None:
            screen = cached
        else:
            recovered = _recover_board_rect(image, screen, width, height, drawing=True)
            if recovered is not None:
                _remember_board_layout(width, height, recovered)
                screen = calibrated_layout(width, height, recovered)
                _RECOVERED_LAYOUTS[key] = screen

    active_rect = _pick_board_rect(image, screen, width, height, drawing=True)
    if active_rect is None:
        # 「等待开画」是另一套布局（棋盘居中），必须先试它再自愈。
        # 自愈是按 active 比例做的全搜索，在等待画面上会选中 start 框并吃掉
        # 6 秒（1239x780 实测，预算是 0.5 秒）；start 布局 + 开画按钮 50ms 就够。
        waiting = _waiting_scene_if_any(image, screen, width, height, previous_scene)
        if waiting is not None:
            return waiting
        # 布局比例对不上（窗口被最大化/改尺寸/换 DPI）时，固定比例算出的候选会整体
        # 偏十几像素，网格线全错位。这里在候选附近把棋盘找回来并落盘标定，
        # 于是「窗口一变就整片不识别」不会重演。
        recovered = _recover_board_rect(image, screen, width, height, drawing=True)
        if recovered is not None:
            _remember_board_layout(width, height, recovered)
            active_rect = recovered
    if active_rect is not None:
        board_score = _score_board_rect(image, active_rect)
        tracked = previous_scene.board.rect if previous_scene is not None else previous_board_rect
        accepted = board_score >= MIN_BOARD_DETECTION_SCORE
        if not accepted:
            # 布局可信但分数偏低：上一帧就在跟踪同一块棋盘，或网格线足够硬。
            if tracked is not None and _rect_is_near(tracked, active_rect):
                board_score = max(board_score, previous_scene.board.score if previous_scene is not None else 0.0)
                accepted = board_score >= MIN_BOARD_DETECTION_SCORE
            elif _grid_confirms_board(image, active_rect):
                # 新画纸（上半还没落子）完整度天然低，综合分比阈值只差千分之几，
                # 但 6x6 网格线成立——见函数注释。board_score 仍报真实值。
                accepted = True
        if not accepted:
            # 布局勉强被 presence 放过、但棋盘自身不成立，等价于「这个窗口尺寸
            # 没有标定」。原实现在这里直接 return None，自愈永远没机会跑，于是
            # 拖过窗口之后每帧都错、点完整识别也一直失败。补一次自愈并落盘：
            # 限流在 _recover_board_rect 内（失败指数退避），非游戏画面不会
            # 反复吃 CPU；成功则同一尺寸之后都走快路径。
            recovered = _recover_board_rect(image, screen, width, height, drawing=True)
            if recovered is None:
                return None
            _remember_board_layout(width, height, recovered)
            active_rect = recovered
            board_score = _score_board_rect(image, active_rect)
            if board_score < MIN_BOARD_DETECTION_SCORE:
                return None
        scene = _analyze_scene_from_board_rect(
            image,
            active_rect,
            fast=True,
            quick=True,
            previous_scene=previous_scene,
            board_score=board_score,
        )
        return scene

    # 等待开画时棋盘居中。上面那条「active 不成立」的分支已经试过一次，
    # 走到这里说明 active 成立过又被否掉；再试一次代价只有几十毫秒。
    waiting = _waiting_scene_if_any(image, screen, width, height, previous_scene)
    if waiting is not None:
        return waiting

    tracked = previous_scene.board.rect if previous_scene is not None else previous_board_rect
    if tracked is None:
        return None
    if any(_rect_is_near(tracked, rect) for rect in screen.candidates(width, height, drawing=True)):
        return None
    if any(_rect_is_near(tracked, rect) for rect in screen.candidates(width, height, drawing=False)):
        return None
    if not _board_present(image, tracked, screen):
        return None
    tracked_score = _score_board_rect(image, tracked)
    if tracked_score < MIN_BOARD_DETECTION_SCORE:
        return None
    scene = _analyze_scene_from_board_rect(
        image,
        tracked,
        fast=True,
        quick=True,
        previous_scene=previous_scene,
        board_score=tracked_score,
    )
    if scene is not None and scene.board.score >= MIN_BOARD_DETECTION_SCORE:
        return scene
    # Realtime capture must reject non-drawing game screens immediately. A
    # full-frame fallback can find incidental 6x6-like UI and take many seconds.
    return None


def _waiting_scene_if_any(
    image: np.ndarray,
    screen: ScreenLayout,
    width: int,
    height: int,
    previous_scene: SceneObservation | None,
) -> SceneObservation | None:
    """「等待开画」画面：棋盘居中 + 独立的开画按钮。不是就返回 None。

    start 布局的棋盘位置比 active 偏右得多（0.354 vs 0.187），单看网格线不够，
    必须同时认出那个开画按钮 —— 它才是「这一帧是在等开局」的硬证据。
    """
    start_rect = _pick_board_rect(image, screen, width, height, drawing=False)
    if start_rect is None:
        return None
    start_button = _find_start_drawing_button_fast(image)
    if start_button is None:
        return None
    start_score = _score_board_rect(image, start_rect)
    # The centered waiting layout receives a lower position prior than the
    # active left layout. The independently detected action button keeps
    # this relaxed threshold specific to the waiting scene.
    if start_score < 0.56:
        return None
    return _analyze_start_drawing_scene(
        image,
        start_rect,
        start_button,
        previous_scene=previous_scene,
        board_score=start_score,
    )


GRID_PRESENCE_SCORE = 0.55
#: 布局自愈的限流：同一窗口尺寸最多每这么久尝试一次。
#: ``_recover_board_rect`` 一次要跑七八十次网格评分（1902x1010 上约 1.6s），
#: 在根本不是游戏画面的时候不能每帧都试。
RECOVER_ATTEMPT_INTERVAL = 5.0
#: 成功一次之后的静默期。成功会落盘标定，下一帧候选就该对上了；
#: 再撞进这里说明标定没生效（写失败/别的尺寸），给足时间让它别每帧重跑。
RECOVER_SUCCESS_INTERVAL = 30.0
#: 窗口尺寸 → (上次尝试时刻, 下次允许的间隔)。失败指数退避，上限 120s。
_RECOVER_ATTEMPTS: dict[tuple[int, int], tuple[float, float]] = {}
#: 布局自愈是否允许写 ``runtime/layout.json``。默认关，界面进程启动时打开；
#: 测试必须保持关闭，否则跑完一轮就把测试截图的坐标留在文件里。
_LAYOUT_AUTOSAVE = False


def set_layout_autosave(enabled: bool) -> None:
    """界面进程启动时打开自动标定（见 :func:`_remember_board_layout`）。"""
    global _LAYOUT_AUTOSAVE
    _LAYOUT_AUTOSAVE = bool(enabled)


#: 定位路径的诊断回调。参数是一个 dict：``path``（calibrated/tracked/layout/
#: search/none）、``elapsed``、``width``/``height``、``rect``/``score``。
#: 界面进程接上它，把慢帧写进 ``observations.jsonl`` —— 「开局卡十秒」这类问题
#: 原来只能靠猜是哪条分支，日志里没有路径信息。
_DIAGNOSTIC_HOOK: Callable[[dict[str, object]], None] | None = None


def set_diagnostic_hook(hook: Callable[[dict[str, object]], None] | None) -> None:
    global _DIAGNOSTIC_HOOK
    _DIAGNOSTIC_HOOK = hook


#: 定位耗时超过这个秒数才写诊断记录。正常（标定命中）是 0.1~0.2s，
#: 走全搜索是 2~4s，只有后者值得记 —— 否则自动跟随每 1.5 秒一条，把日志淹了。
LOCATE_DIAGNOSTIC_SECONDS = 1.0


def _report_diagnostic(payload: dict[str, object]) -> None:
    hook = _DIAGNOSTIC_HOOK
    if hook is None:
        return
    try:
        hook(payload)
    except Exception:  # noqa: BLE001 - 诊断绝不能影响识别
        pass


def _pick_board_rect(
    image: np.ndarray,
    screen: ScreenLayout,
    width: int,
    height: int,
    *,
    drawing: bool,
) -> Rect | None:
    """在布局给出的少数候选里挑一个网格线成立的；都立不住就返回 None。

    实机 11 张截图：棋盘区网格分 0.573~0.673，其他区域最高 0.504。
    """
    best: tuple[Rect | None, float] = (None, 0.0)
    for rect in screen.candidates(width, height, drawing=drawing):
        score = _board_grid_score(image, rect, outer_penalty=False)
        if score >= GRID_PRESENCE_SCORE:
            return rect
        if score > best[1]:
            best = (rect, score)
    if best[0] is None:
        return None
    # presence 只回答「这是不是画画界面」，**不回答这个框对不对**：它的阈值只有
    # 0.15，而模板文件缺失时 load_template() 干脆返回 True（见 presence.py:94）。
    # 2026-11-15 把窗口拖成 1536x877 后，库里没有这组标定，内置比例算出的候选
    # (287,225,521) 网格分仅 0.416、棋盘综合分 0.146，却因 presence 相似度
    # 0.249 被当成真棋盘收下。实时路径拿错框算分不达标就直接 return None，
    # 自愈分支永远进不去、标定永远落不了盘 —— 玩家看到的就是「改了窗口大小
    # 之后整片识别错误」，而且点完整识别也救不回来（它走的也是实时路径）。
    # 所以让 presence 说话之前，必须再用棋盘自身的两条证据复核。
    if best[1] < GRID_PRESENCE_SCORE - 0.10:
        return None
    if _score_board_rect(image, best[0]) < MIN_BOARD_DETECTION_SCORE * 0.8:
        return None
    present, _ = presence.is_drawing_screen(
        image, screen.presence_rect(best[0]), template=_presence_template()
    )
    return best[0] if present else None


def _recover_board_rect(
    image: np.ndarray,
    screen: ScreenLayout,
    width: int,
    height: int,
    *,
    drawing: bool = True,
) -> Rect | None:
    """布局候选全都不成立时，把棋盘找回来。

    为什么不用「在候选附近微调」：窗口比例一变，偏差不是十几像素而是在**几十像素**
    且方向无规律——2026-11-15 同一台机器上，1920x1020 的窗口棋盘在 (341,254,608)，
    拖到 1239x780 后落在 (177,211,442)：x 差 164、边长差 166。局部搜索
    （±20px、±16px 边长）根本够不着，实测直接返回 None。

    ``layout.py`` 的比例只是「大概率在这附近」的先验，游戏 UI 的缩放/居中规则既然
    不透明，就只能用完整搜索兜底。这条路只在**布局失效时跑一次**：成功后调用方
    把结果按窗口尺寸落盘，同一尺寸之后都走快路径；失败则指数退避，避免在非游戏
    画面上反复吃 CPU。

    这里调 :func:`_locate_board_rect` 而不是 ``analyze_scene``：只需要矩形，没必要
    把面板 OCR、笔触、背包和逐格分类也跑一遍（那一段在 1239x780 上就占 6 秒）。
    """
    key = (width, height)
    now = time.monotonic()
    prev = _RECOVER_ATTEMPTS.get(key)
    if prev is not None and now - prev[0] < prev[1]:
        return None
    # 失败就退避（5s → 10s → 20s … 上限 120s）：一次要跑十几秒，
    # 在「根本不是游戏画面」的时候不能每 5 秒就吃一遍。
    _RECOVER_ATTEMPTS[key] = (now, min(RECOVER_ATTEMPT_INTERVAL * 2, 120.0))

    located = _locate_board_rect(image)
    if located is None:
        return None
    rect, score = located
    # layout 那条路刻意放宽到 0.68（2K 下同一块棋盘 layout 只有 0.712，用 0.70
    # 会整片判不出）。但自愈要把结果**落盘成标定**，落错一次后面全错，
    # 所以这里按正式门槛复核。实测两张实机图 0.760 / 0.777，都过。
    if score < MIN_BOARD_DETECTION_SCORE:
        return None
    # 全搜索给的是**格子级粗对齐**，实测会差到 ±8px：1239x780 上自愈落盘过
    # (181,203,439)，而真框是 (177,211,442)。8px 约等于 6x6 棋盘的 11% 格高，
    # 实时路径不做 refine，逐格裁剪整体错位 —— 画面上 5 枚颜料只认出 2 枚、
    # 等级全部丢失，玩家看到的就是「识别不了全部颜料」（2026-11-15 22:50 实机）。
    # 自愈结果要落盘成标定，错一次后面每帧都错，所以这里必须做亚像素级精修，
    # 且只在综合分不降时采用（防止精修自己跑偏）。自愈本来就是几秒一次的冷路径，
    # 这 0.4s 值得花。
    refined = _refine_board_rect(image, rect)
    if _score_board_rect(image, refined) + 1e-6 >= _score_board_rect(image, rect):
        rect = refined
    # 落盘前最后一道：网格线必须真的成立。
    # 2026-11-19 实机事故——开局那一帧还在转场，自愈在画面左上角找到一块
    # 「像棋盘」的东西（综合分过了门槛），于是把 (0,0,548) 当成标定写进
    # layout.json，之后整局每一帧都按窗口左上角裁格子，识别全错。
    # 网格线是内容无关的硬证据，转场画面/装饰边框给不出来。
    if _board_grid_score(image, rect) < GRID_PRESENCE_SCORE:
        return None
    _RECOVER_ATTEMPTS[key] = (now, RECOVER_SUCCESS_INTERVAL)
    return rect


def _remember_board_layout(width: int, height: int, rect: Rect) -> None:
    """把一次可信的棋盘位置落盘，并让下一次识别立刻用上。

    没有这一步，窗口比例一变（最大化、分辨率、DPI）实时路径就整片失效，
    而完整搜索在 1902x1010 上要 28 秒，绝不能每帧跑。

    **默认不落盘**：只有界面进程显式调用 :func:`set_layout_autosave` 打开它。
    测试里跑几十帧不同尺寸的截图，如果自动标定生效，最后一次的结果会留在
    ``runtime/layout.json`` 里污染后续所有测试与真实运行——实测就是这么让
    两条背包等级用例变红的。
    """
    if not _LAYOUT_AUTOSAVE:
        return
    from .layout import calibration_is_plausible
    from .layout import LayoutRect

    plausible = calibration_is_plausible(
        LayoutRect(
            x_ratio=rect.x / max(1, width),
            y_ratio=rect.y / max(1, height),
            size_ratio=rect.w / max(1, min(width, height)),
        )
    )
    try:
        save_layout(width, height, rect)
    except OSError:
        return
    # 落盘是否真的生效，决定「下一局还要不要重搜」——原来这件事只能靠读
    # 磁盘上的 layout.json 反推，而用户那边的现象正是「每局都慢」。
    _report_diagnostic(
        {
            "kind": "layout_saved",
            "width": int(width),
            "height": int(height),
            "rect": [rect.x, rect.y, rect.w, rect.h],
            "plausible": bool(plausible),
        }
    )
    reset_vision_caches()


def remember_board_calibration(image: np.ndarray, scene: SceneObservation) -> None:
    """把一次可信识别的棋盘位置按窗口尺寸落盘（供完整识别的冷启动兜底调用）。

    ``_remember_board_layout`` 默认不写盘，只有界面进程显式打开
    :func:`set_layout_autosave` 才生效，所以测试里跑再多帧也不会污染标定文件。
    """
    height, width = image.shape[:2]
    _remember_board_layout(width, height, scene.board.rect)


def _grid_confirms_board(image: np.ndarray, rect: Rect) -> bool:
    """6x6 网格线足够硬 → 这块区域就是棋盘，哪怕综合分差一点。

    为什么需要这条：``_score_board_rect`` 的 ``completeness`` 分量按「格内有内容」
    算，新画纸（上半还没落子）天然只有 0.43，实测综合分 **0.6985**，比阈值 0.70
    只差 0.0015 —— 差这么一点就整帧被判「看不到棋盘」，
    于是 auto 链路既不更新界面、也没机会认新局，日志只剩每秒一条
    「低置信度」，用户看到的就是「收尾后一直挂着上一局」。

    网格线是内容无关的硬证据：同一帧软网格分 0.668，比满铺局的 0.593 还高。
    这里要求「明显高于 ``GRID_PRESENCE_SCORE``」是为了不把随便一块带纹理的
    画面当成棋盘——布局候选本来就只有在网格成立（或 presence 模板成立）时才会返回。
    """

    return _board_grid_score(image, rect, outer_penalty=False) >= GRID_PRESENCE_SCORE + 0.08


def _board_present(image: np.ndarray, rect: Rect, screen: ScreenLayout) -> bool:
    """这块区域像不像棋盘：先看 6x6 网格线（内容无关），再用梯度模板兜底。"""
    if _board_grid_score(image, rect) >= GRID_PRESENCE_SCORE:
        return True
    present, _ = presence.is_drawing_screen(
        image, screen.presence_rect(rect), template=_presence_template()
    )
    return present


_LAYOUT_CACHE: ScreenLayout | None = None
#: 上面这份缓存是按哪个窗口尺寸算的。尺寸一变就要重读标定。
_LAYOUT_CACHE_KEY: tuple[int | None, int | None] | None = None
#: 本进程内**自愈**出来的布局，按窗口尺寸缓存。
#:
#: 为什么需要：某个尺寸在 ``layout.json`` 里没有标定时（换分辨率/拖窗口/改 DPI
#: 之后很常见），就要跑一次棋盘自愈，而自愈约 1.5 秒。落盘只在界面进程生效
#: （``_LAYOUT_AUTOSAVE``），测试与子进程存不下，于是「下一帧又发现没标定→
#: 再自愈一次」，实时路径直接被拖垮（实测 1.47s/帧）。这里兜住同尺寸的重复自愈。
_RECOVERED_LAYOUTS: dict[tuple[int, int], ScreenLayout] = {}
_PRESENCE_LOADED = False
_PRESENCE_TEMPLATE: np.ndarray | None = None


def _screen_layout(width: int | None = None, height: int | None = None) -> ScreenLayout:
    """取布局。

    传了窗口尺寸就按尺寸挑对应的标定：不同大小的窗口里棋盘相对位置并不一样，
    拿 A 尺寸的标定去算 B 尺寸会整体偏十几像素（2026-11-15 实机事故：
    1673x878 的标定用在 1920x1020 上，棋盘 6x6 网格线全部错位）。
    没传尺寸时用文件里第一组（兼容老调用）。
    """
    global _LAYOUT_CACHE, _LAYOUT_CACHE_KEY
    key = (width, height)
    # 缓存只对「同一个窗口尺寸」有效。原来写成
    # ``_LAYOUT_CACHE_KEY not in (None, key)``：一旦先来一次不带尺寸的调用，
    # 缓存键就固定成 (None, None)，之后换成带尺寸的调用也判成「没变」，
    # 于是拖窗口后再也不重新读标定——典型的「改了窗口大小照样按旧框识别」。
    if _LAYOUT_CACHE is None or (width is not None and _LAYOUT_CACHE_KEY != key):
        _LAYOUT_CACHE = load_layout(width, height)
        _LAYOUT_CACHE_KEY = key
    return _LAYOUT_CACHE


def _presence_template() -> np.ndarray | None:
    global _PRESENCE_LOADED, _PRESENCE_TEMPLATE
    if not _PRESENCE_LOADED:
        _PRESENCE_TEMPLATE = presence.load_template()
        _PRESENCE_LOADED = True
    return _PRESENCE_TEMPLATE


def reset_vision_caches() -> None:
    """布局文件或模板被替换后调用，让下一次识别重新读取。"""
    global _LAYOUT_CACHE, _LAYOUT_CACHE_KEY, _PRESENCE_LOADED, _PRESENCE_TEMPLATE
    global _WAITING_BOARD_CACHE
    _LAYOUT_CACHE = None
    _LAYOUT_CACHE_KEY = None
    # 自愈缓存也要清：布局文件被替换（用户点了重新标定 / 换了标定）之后，
    # 继续用旧的实测框就是错的。
    _RECOVERED_LAYOUTS.clear()
    _PRESENCE_LOADED = False
    _PRESENCE_TEMPLATE = None
    _WAITING_BOARD_CACHE = None
    # 背包家族缓存只在「游戏/窗口/模板变了」时才该清：它省的是 40~60ms/次
    # 的 ORB 匹配，平时必须留着；测试靠它隔离用例。
    reset_family_cache()


def _rect_is_near(first: Rect, second: Rect) -> bool:
    tolerance = max(8, int(round(second.w * 0.08)))
    return (
        abs(first.x - second.x) <= tolerance
        and abs(first.y - second.y) <= tolerance
        and abs(first.w - second.w) <= tolerance
    )


def _median(values: list[float]) -> float:
    """36 元素级别的小数组中位数，比 ``np.median`` 快一个数量级。

    ``np.median`` 每次调用要过一层 numpy 调度，在 36 个元素上开销远大于计算本身
    （实测 0.06ms vs 0.002ms）。它还只在首次调用时才懒加载 ``numpy.ma``（0.048s），
    若落在首帧实时路径上会直接吃掉整个增量预算，因此热路径一律走这里。
    """

    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _tracked_board_visual_guard(
    image: np.ndarray,
    rect: Rect,
    previous_board: BoardObservation,
) -> bool:
    """Reject a vanished board using 4x4 cell signatures instead of a full board rescore."""

    differences: list[float] = []
    cell_w = rect.w / BOARD_SIZE
    cell_h = rect.h / BOARD_SIZE
    for tile in previous_board.tiles:
        previous_signature = _tile_signature_from_features(tile.features)
        if previous_signature is None:
            continue
        x1 = round(rect.x + tile.col * cell_w)
        y1 = round(rect.y + tile.row * cell_h)
        x2 = round(rect.x + (tile.col + 1) * cell_w)
        y2 = round(rect.y + (tile.row + 1) * cell_h)
        current_signature = _tile_visual_signature(image[y1:y2, x1:x2])
        differences.append(
            float(np.mean(np.abs(np.asarray(current_signature) - np.asarray(previous_signature))))
        )
    if len(differences) < BOARD_SIZE:
        return True
    return _median(differences) <= 0.26


#: 「等待开画」画面在玩家点按钮之前是静止的，而 ``_classify_tiles`` 跑满 36 格
#: 约 0.27s（1239x780 实测）。实时链路每帧都重跑这一遍纯属浪费：2026-11-15
#: 受控负载实验里，光这一步就把 0.5s 的交互预算吃掉大半，外部一抢 CPU 就破线。
#: key 里带棋盘矩形和画面签名，任一变化立即失效，绝不把旧格子状态带到发牌后。
_WAITING_BOARD_CACHE: tuple[tuple[int, int, int, int, bytes], tuple] | None = None
_WAITING_SIGNATURE_SIDE = 48


def _waiting_board_signature(image: np.ndarray, rect: Rect) -> bytes:
    """棋盘画面的廉价指纹：缩到 48x48 的 BGR 字节，局部图标变化也会改变它。"""
    patch = image[rect.y : rect.bottom, rect.x : rect.right]
    if patch.size == 0:
        return b""
    small = cv2.resize(patch, (_WAITING_SIGNATURE_SIDE, _WAITING_SIGNATURE_SIDE), interpolation=cv2.INTER_AREA)
    return small.tobytes()


def _analyze_start_drawing_scene(
    image: np.ndarray,
    rect: Rect,
    start_button: tuple[float, Rect],
    *,
    previous_scene: SceneObservation | None = None,
    board_score: float | None = None,
) -> SceneObservation:
    global _WAITING_BOARD_CACHE
    bounded = _bounded_rect(rect.x, rect.y, rect.w, rect.h, image.shape[1], image.shape[0])
    key = (
        bounded.x, bounded.y, bounded.w, bounded.h,
        _waiting_board_signature(image, bounded),
    )
    cached = _WAITING_BOARD_CACHE
    if cached is not None and cached[0] == key:
        tiles = cached[1]
    else:
        previous_board = previous_scene.board if previous_scene is not None and _rect_is_near(previous_scene.board.rect, bounded) else None
        tiles = tuple(
            _classify_tiles_quick(image, bounded, previous_board)
            if previous_board is not None
            else _classify_tiles(image, bounded)
        )
        _WAITING_BOARD_CACHE = (key, tiles)
    raw_board_score = _score_board_rect(image, bounded) if board_score is None else board_score
    position_score = _board_position_score(image.shape, bounded)
    normalized_board_score = float(np.clip(raw_board_score / max(0.55, position_score), 0.0, 0.96))
    board = BoardObservation(
        rect=bounded,
        score=normalized_board_score,
        tiles=tiles,
    )
    region = RegionObservation(
        kind=RegionKind.START_DRAWING,
        rect=start_button[1],
        confidence=start_button[0],
        label="start drawing button fast",
    )
    return SceneObservation(board=board, regions=(region,))


def _is_supported_scene_image(image: np.ndarray) -> bool:
    return bool(
        isinstance(image, np.ndarray)
        and image.ndim == 3
        and image.shape[2] == 3
        and image.size
        and min(image.shape[:2]) >= MIN_CAPTURE_SIDE
    )


def _analyze_scene_from_board_rect(
    image: np.ndarray,
    rect: Rect,
    fast: bool,
    *,
    quick: bool = False,
    previous_scene: SceneObservation | None = None,
    board_score: float | None = None,
    icon_fallback: bool = False,
) -> SceneObservation | None:
    if not fast:
        rect = _refine_board_rect(image, rect)
    rect = _bounded_rect(rect.x, rect.y, rect.w, rect.h, image.shape[1], image.shape[0])
    score = _score_board_rect(image, rect) if board_score is None else board_score
    if score < 0.20:
        return None
    tiles = tuple(
        _classify_tiles_quick(image, rect, previous_scene.board)
        if quick and previous_scene is not None
        else _classify_tiles(image, rect, icon_fallback=icon_fallback)
    )
    board = BoardObservation(rect=rect, score=score, tiles=tiles)
    regions = tuple(_estimate_regions_fast(image, board) if quick else _estimate_regions(image, board))
    if quick:
        current_paint = _analyze_current_paint_fast(image, regions, previous_scene.current_paint if previous_scene is not None else None)
        current_paint = _correct_current_paint_from_board_anchor(image, current_paint, tiles)
        current_paint = _correct_current_paint_from_scope(current_paint, tiles)
        board = _sync_single_anchor_tile_from_current_paint(board, current_paint)
        backpack = _analyze_backpack_fast(image, regions, previous_scene.backpack if previous_scene is not None else None)
        current_paint = _correct_current_target_from_backpack(current_paint, backpack)
        brush_region = next((item for item in regions if item.kind == RegionKind.BRUSH_OPTIONS), None)
        brush_options = (
            _analyze_brush_options_realtime(
                image,
                regions,
                previous=previous_scene.brush_options if previous_scene is not None else None,
            )
            if brush_region is not None
            else None
        )
        return SceneObservation(
            board=board,
            regions=regions,
            current_paint=current_paint,
            brush_options=brush_options,
            backpack=backpack,
        )
    current_paint = _analyze_current_paint(image, regions)
    current_paint = _correct_current_paint_from_board_anchor(image, current_paint, tiles)
    current_paint = _correct_current_paint_from_scope(current_paint, tiles)
    board = _sync_single_anchor_tile_from_current_paint(board, current_paint)
    _learn_kind_from_scene(image, board, current_paint)
    brush_options = _analyze_brush_options(image, regions)
    backpack = _analyze_backpack(image, regions)
    current_paint = _correct_current_target_from_backpack(current_paint, backpack)
    return SceneObservation(
        board=board,
        regions=regions,
        current_paint=current_paint,
        brush_options=brush_options,
        backpack=backpack,
    )


def _correct_current_target_from_backpack(
    current_paint: CurrentPaintObservation | None,
    backpack: BackpackObservation | None,
) -> CurrentPaintObservation | None:
    """用背包排序反推当前初始颜料的目标类别。

    实机背包按当前初始颜料排序，所以靠前那批就是匹配颜料。但这个判据只在
    **类别识别真的有把握**时才有意义：`_classify_backpack_frame_shape` 的
    置信度下限是 0.64，实机 15 格常年全部读成 0.64（等于"我触发了，但几乎没证据"）。
    那种情况下 5 格全是 GAMEPLAY 是识别退化，不是排序证据——照它纠正会把
    怪物规则一律翻成玩法。所以这里要求 0.70 以上（明显高于下限）才采信。
    """
    if current_paint is None or backpack is None or len(backpack.paints) < 5:
        return current_paint
    # 目标类别已经由面板文字直接读出来了，就不要再让「背包前 5 格投票」翻案。
    # 框形状分不出类别（背包 15 格外框全是菱形），这条旁证只在没读到文字时兜底。
    if float(current_paint.keyword_scores.get("panel_text:target", 0.0)) >= 0.5:
        return current_paint
    leading = [
        paint for paint in backpack.paints[:5] if paint.category_confidence >= 0.70
    ]
    if len(leading) < 4:
        return current_paint
    counts = {
        category: sum(1 for paint in leading if paint.category == category)
        for category in (PaintCategory.MONSTER, PaintCategory.GAMEPLAY)
    }
    category, count = max(counts.items(), key=lambda item: item[1])
    if count < 4:
        return current_paint
    target = PaintObject.MONSTER if category == PaintCategory.MONSTER else PaintObject.GAMEPLAY
    if current_paint.target_object == target:
        return current_paint
    evidence = sum(
        paint.category_confidence
        for paint in leading
        if paint.category == category
    ) / count
    scores = dict(current_paint.keyword_scores)
    scores[f"backpack_order:target:{target.value}"] = float(evidence)
    return replace(
        current_paint,
        target_object=target,
        keyword_scores=scores,
        raw_text_hint=f"{current_paint.raw_text_hint} backpack-order-corrected",
    )


def _correct_current_paint_from_board_anchor(
    image: np.ndarray,
    current_paint: CurrentPaintObservation | None,
    tiles: tuple[TileObservation, ...],
) -> CurrentPaintObservation | None:
    if current_paint is None:
        return None
    marked_tiles = [tile for tile in tiles if tile.is_current_anchor]
    if len(marked_tiles) == 1:
        anchor = marked_tiles[0]
        if (
            anchor.paint_category != PaintCategory.UNKNOWN
            and anchor.paint_category_confidence >= 0.62
            and anchor.paint_level is not None
            and anchor.paint_level_confidence >= 0.62
        ):
            scores = dict(current_paint.keyword_scores)
            scores[f"board_anchor:category:{anchor.paint_category.value}"] = float(
                anchor.paint_category_confidence
            )
            return replace(
                current_paint,
                category=anchor.paint_category,
                category_confidence=max(
                    float(current_paint.category_confidence),
                    float(anchor.paint_category_confidence),
                ),
                level=anchor.paint_level,
                level_confidence=max(
                    float(current_paint.level_confidence),
                    float(anchor.paint_level_confidence),
                ),
                subtype=anchor.paint_subtype,
                is_treasure=anchor.paint_is_treasure,
                keyword_scores=scores,
                raw_text_hint=f"{current_paint.raw_text_hint} board-anchor-corrected",
            )
    if current_paint.category != PaintCategory.UNKNOWN and current_paint.category_confidence >= 0.58:
        return current_paint
    paint_tiles = marked_tiles or [tile for tile in tiles if tile.state == TileState.PAINT]
    if len(paint_tiles) != 1:
        return current_paint
    category, confidence = _classify_current_paint_category(image, paint_tiles[0].rect)
    if category == PaintCategory.UNKNOWN or confidence < current_paint.category_confidence:
        return current_paint
    return CurrentPaintObservation(
        panel_rect=current_paint.panel_rect,
        icon_rect=current_paint.icon_rect,
        text_rect=current_paint.text_rect,
        category=category,
        category_confidence=confidence,
        subtype=current_paint.subtype,
        is_treasure=current_paint.is_treasure,
        level=current_paint.level,
        level_confidence=current_paint.level_confidence,
        match_current=current_paint.match_current,
        match_required=current_paint.match_required,
        target_object=current_paint.target_object,
        direction=current_paint.direction,
        upgrade_amount=current_paint.upgrade_amount,
        keyword_scores=current_paint.keyword_scores,
        raw_text_hint=current_paint.raw_text_hint,
        effect_text=current_paint.effect_text,
    )


def _correct_current_paint_from_scope(
    current_paint: CurrentPaintObservation | None,
    tiles: tuple[TileObservation, ...],
) -> CurrentPaintObservation | None:
    if current_paint is None:
        return None
    highlighted = [tile for tile in tiles if tile.is_scope_highlight]
    if len(highlighted) < 2:
        return current_paint

    rows = {tile.row for tile in highlighted}
    cols = {tile.col for tile in highlighted}
    direction = PaintDirection.UNKNOWN
    if len(cols) == 1 and len(rows) >= 2:
        direction = PaintDirection.VERTICAL
    elif len(rows) == 1 and len(cols) >= 2:
        direction = PaintDirection.HORIZONTAL
    if direction == PaintDirection.UNKNOWN or direction == current_paint.direction:
        return current_paint

    scores = dict(current_paint.keyword_scores)
    scores[f"board_scope:direction:{direction.value}"] = max(
        tile.scope_highlight_confidence for tile in highlighted
    )
    return replace(
        current_paint,
        direction=direction,
        keyword_scores=scores,
        raw_text_hint=f"{current_paint.raw_text_hint} scope-corrected",
    )


#: 运行时自学习的节流：同一个家族两次尝试之间至少隔这么久。
#: 面板一换颜料就会重算一遍「面板图标 ↔ 36 格」的配对，36 次多尺度匹配约 30ms，
#: 不能每帧都跑。
KIND_LEARN_INTERVAL = 3.0

#: 官方图标兜底给出的置信度。比模板命中（0.68~0.9）低一档：图标证明「画的是谁」，
#: 但没经过本机渲染的像素对齐，所以不拿它去压等级/类别的其它证据。
ICON_FALLBACK_CONFIDENCE = 0.72

_KIND_LEARN_AT: dict[str, float] = {}


def _learn_kind_from_scene(
    image: np.ndarray,
    board: BoardObservation,
    current_paint: CurrentPaintObservation | None,
) -> None:
    """面板上读到家族名时，顺手把这个家族的图案收进模板库。

    这是唯一能自动拿到「图案 ↔ 名字」对应关系的路子：解包资源里只有图标名
    （``UI_S8Gameplay_ColorMonster2_128``），既没有图也没有名字对照表；
    而右侧「当前颜料」面板同时渲染着图标和 ``怪物颜料·邪影LV3`` 这行字。

    只在慢路径（冷启动 / 单张截图分析）里调用。凑不齐「分数够高 + 领先亚军够多」
    就不学——一个错模板会持续污染这个家族。
    """
    if current_paint is None or not current_paint.subtype:
        return
    subtype = current_paint.subtype
    if paint_kind.family_key(subtype) is None:
        return
    active = paint_kind.bank()
    if len(active.templates.get(subtype, [])) >= paint_kind.MAX_TEMPLATES_PER_KIND:
        return
    now = time.monotonic()
    if now - _KIND_LEARN_AT.get(subtype, 0.0) < KIND_LEARN_INTERVAL:
        return
    _KIND_LEARN_AT[subtype] = now
    icon = current_paint.icon_rect
    panel_icon = image[icon.y : icon.bottom, icon.x : icon.right]
    if panel_icon.size == 0:
        return
    cells = [image[tile.rect.y : tile.rect.bottom, tile.rect.x : tile.rect.right] for tile in board.tiles]
    if paint_kind.learn_from_panel(panel_icon, subtype, cells, active=active):
        try:
            active.save()
        except OSError:
            pass


def _sync_single_anchor_tile_from_current_paint(
    board: BoardObservation,
    current_paint: CurrentPaintObservation | None,
) -> BoardObservation:
    if current_paint is None:
        return board
    if current_paint.category == PaintCategory.UNKNOWN or current_paint.level is None:
        return board
    if current_paint.category_confidence < 0.55 or current_paint.level_confidence < 0.55:
        return board

    # 盘上不止一枚颜料 = 这局的初始颜料早落完了，面板上这枚是**下一枚**，
    # 不归它管。原来的 ``paint_indices = marked_indices or [...]`` 一旦判出锚点
    # 就只看那一格，「唯一一枚」的前提检查被整体绕过：2026-11-15 实机盘上 5 枚
    # 颜料、锚点指向 r2c2、面板读「机械之神 Lv3」，把 r2c2 的「赛季 Lv4」盖成
    # 「机械之神 Lv3」。而没有金框的那一帧锚点消失、又走回 ``len != 1`` 不动，
    # 于是同一格在两帧之间来回翻 —— 玩家看到的就是「4 级赛季未识别」。
    if sum(1 for tile in board.tiles if tile.state == TileState.PAINT) > 1:
        return board

    marked_indices = [index for index, tile in enumerate(board.tiles) if tile.is_current_anchor]
    paint_indices = marked_indices or [index for index, tile in enumerate(board.tiles) if tile.state == TileState.PAINT]
    if len(paint_indices) != 1:
        return board

    index = paint_indices[0]
    tile = board.tiles[index]
    # 「盘上只有一枚颜料」不等于「这枚就是本局的初始颜料」。
    #
    # 2026-11-15 实机事故：新画纸开局自带一枚**赛季 Lv4**，而面板上读到的还是
    # 初始颜料规则「怪物 Lv3」。这里无条件拿面板读数盖上去，结果那一格变成
    # 「怪物 Lv3、家族空」——家族直接被抹掉，界面上看就是「4 级赛季未识别」。
    #
    # 判据：没有锚点高亮、而这一格自己认出了家族、家族类别又和面板规则类别冲突
    # → 它们是两枚不同的颜料，面板读数没有资格覆盖它。
    # 面板读出了家族、而和棋盘这格不是同一个 → 两枚不同的颜料，面板没资格覆盖。
    # 这一条比「类别冲突」严格：赛季与机械之神**同属玩法类别**，只看类别会放过它，
    # 而实机上正是「赛季 Lv4 被机械之神 Lv3 盖掉」。
    if tile.paint_subtype and current_paint.subtype and current_paint.subtype != tile.paint_subtype:
        return board
    has_initial_marker = float(tile.features.get("initial_marker_score", 0.0)) >= 0.5
    if tile.paint_subtype and (not marked_indices or has_initial_marker):
        tile_category = paint_kind.category_of(tile.paint_subtype)
        if tile_category and tile_category != current_paint.category.value:
            return board
    # 面板没读出家族时，别把棋盘已经认出来的家族抹成空。
    subtype = current_paint.subtype or tile.paint_subtype
    # 等级：边框色读得够硬（>=0.80）就不拿面板改。
    # ``frame_level`` 在 32 个带真值格子上实测 100%，而这个函数的前提
    # 「唯一颜料就是初始颜料」本身就可能不成立（新画纸自带预置颜料）。
    # 边框勉强能读时才退回面板——那正是面板等级存在的价值。
    frame_level_is_solid = tile.paint_level is not None and tile.paint_level_confidence >= 0.80
    level = tile.paint_level if frame_level_is_solid else current_paint.level
    level_confidence = (
        tile.paint_level_confidence
        if frame_level_is_solid
        else max(tile.paint_level_confidence, current_paint.level_confidence)
    )
    synced = replace(
        tile,
        paint_category=current_paint.category,
        paint_category_confidence=max(tile.paint_category_confidence, current_paint.category_confidence),
        paint_level=level,
        paint_level_confidence=level_confidence,
        paint_subtype=subtype,
        paint_is_treasure=current_paint.is_treasure or subtype == "宝箱",
    )
    tiles = list(board.tiles)
    tiles[index] = synced
    return BoardObservation(rect=board.rect, score=board.score, tiles=tuple(tiles))


def make_debug_image(image: np.ndarray, scene: SceneObservation | BoardObservation | None) -> np.ndarray:
    debug = image.copy()
    board = scene.board if isinstance(scene, SceneObservation) else scene
    if board is None:
        cv2.putText(
            debug,
            "board not detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return debug

    regions = scene.regions if isinstance(scene, SceneObservation) else ()

    colors = {
        TileState.BLANK: (230, 230, 230),
        TileState.BLOCKED: (255, 190, 0),
        TileState.VOID: (35, 35, 35),
        TileState.FADED: (120, 120, 120),
        TileState.PAINT: (0, 220, 255),
        TileState.UNKNOWN: (0, 0, 255),
    }

    rect = board.rect
    cv2.rectangle(debug, (rect.x, rect.y), (rect.right, rect.bottom), (0, 255, 0), 3)
    cv2.putText(
        debug,
        f"board score={board.score:.2f}",
        (rect.x, max(25, rect.y - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    for tile in board.tiles:
        tr = tile.rect
        color = colors[tile.state]
        cv2.rectangle(debug, (tr.x, tr.y), (tr.right, tr.bottom), color, 2)
        label = f"{tile.state.value[:1]} {tile.confidence:.2f}"
        cv2.putText(
            debug,
            label,
            (tr.x + 4, tr.y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    region_colors = {
        RegionKind.CURRENT_PAINT: (0, 200, 255),
        RegionKind.BACKPACK: (255, 120, 0),
        RegionKind.BRUSH_OPTIONS: (255, 80, 255),
        RegionKind.CONFIRM_BUTTON: (80, 255, 120),
        RegionKind.START_DRAWING: (255, 210, 80),
        RegionKind.FINISH_DRAWING: (80, 255, 255),
        RegionKind.FORCE_COMPLETE: (0, 140, 255),
        RegionKind.REFRESH_AFFIX: (0, 255, 255),
    }
    for region in regions:
        rr = region.rect
        color = region_colors[region.kind]
        cv2.rectangle(debug, (rr.x, rr.y), (rr.right, rr.bottom), color, 3)
        cv2.putText(
            debug,
            region.label,
            (rr.x, max(22, rr.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    if isinstance(scene, SceneObservation) and scene.current_paint is not None:
        paint = scene.current_paint
        cv2.rectangle(
            debug,
            (paint.icon_rect.x, paint.icon_rect.y),
            (paint.icon_rect.right, paint.icon_rect.bottom),
            (0, 255, 255),
            2,
        )
        cv2.rectangle(
            debug,
            (paint.text_rect.x, paint.text_rect.y),
            (paint.text_rect.right, paint.text_rect.bottom),
            (255, 255, 0),
            2,
        )
    if isinstance(scene, SceneObservation) and scene.brush_options is not None:
        brush = scene.brush_options
        rr = brush.panel_rect
        label = " / ".join(brush.stage1_options) or "brush?"
        if brush.stage2_options:
            label = f"{label} | {brush.stage2_options[0][:12]}"
        cv2.putText(
            debug,
            label,
            (rr.x + 4, rr.y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 80, 255),
            2,
            cv2.LINE_AA,
        )
    if isinstance(scene, SceneObservation) and scene.backpack is not None:
        for index, paint in enumerate(scene.backpack.paints, start=1):
            rr = paint.slot_rect
            cv2.rectangle(debug, (rr.x, rr.y), (rr.right, rr.bottom), (255, 180, 40), 2)
            label = f"{index}:{paint.category.value[:1]}{paint.level}{'T' if paint.is_treasure else ''}"
            cv2.putText(
                debug,
                label,
                (rr.x + 3, rr.y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 220, 80),
                1,
                cv2.LINE_AA,
            )
    return debug


def analyze_and_save_debug(
    image: np.ndarray,
    debug_path: str | Path | None = None,
    tracker: VisionTracker | None = None,
    save_crops: bool = True,
    realtime: bool = False,
) -> tuple[SceneObservation | None, Path | None]:
    if tracker is not None:
        scene = tracker.analyze_realtime(image) if realtime else tracker.analyze(image)
    else:
        scene = analyze_scene_realtime(image) if realtime else analyze_scene(image)
    if save_crops:
        _save_current_paint_crops(image, scene)
        _save_brush_options_crops(image, scene)
        _save_backpack_slot_crops(image, scene)
    if debug_path is None:
        return scene, None
    debug = make_debug_image(image, scene)
    return scene, write_image(debug_path, debug)


# 面板文字读到几个字段才算可信。低于这个值就走原来的字形模板路径。
_PANEL_TEXT_MIN_CONFIDENCE = 0.55


def _panel_text_overrides(reading: paint_panel.PanelReading | None) -> dict[str, object]:
    """把「当前颜料」面板读到的字段整理成覆盖项，两条分析路径共用。"""
    if reading is None or reading.confidence < _PANEL_TEXT_MIN_CONFIDENCE:
        return {}
    overrides: dict[str, object] = {}
    if reading.category != PaintCategory.UNKNOWN:
        overrides["category"] = reading.category
        overrides["category_confidence"] = 0.88
    if reading.subtype:
        overrides["subtype"] = reading.subtype
        # 官方配置里「玩法颜料-宝箱」就是宝箱玩法，7 级即通关目标。
        overrides["is_treasure"] = reading.subtype == "宝箱"
    if reading.level is not None:
        overrides["level"] = reading.level
        overrides["level_confidence"] = 0.90
    if reading.target_object != PaintObject.UNKNOWN:
        overrides["target_object"] = reading.target_object
    if reading.direction != PaintDirection.UNKNOWN:
        overrides["direction"] = reading.direction
    if reading.upgrade_amount is not None:
        overrides["upgrade_amount"] = reading.upgrade_amount
    if reading.match_required is not None:
        overrides["match_current"] = int(reading.match_current or 0)
        overrides["match_required"] = int(reading.match_required)
    return overrides


def _panel_text_covers_paint_fields(overrides: dict[str, object]) -> bool:
    """面板文字是否把「当前颜料」的全部字段都读出来了。

    只有全读出来才敢跳过字形模板匹配：少一个字段（比如界面换了版式、OCR 只读到
    标题）就得让模板去补，那时省时间等于拿错数值换速度。
    """
    required = (
        "category",
        "level",
        "target_object",
        "direction",
        "upgrade_amount",
        "match_required",
    )
    return all(key in overrides for key in required)


def _current_paint_text_crop(image: np.ndarray, panel: Rect) -> np.ndarray:
    """给 OCR 用的「当前颜料」文字区，比模板用的裁剪框更宽也更高。

    全屏时这个面板和棋盘顶部基本齐平，窗口化后它会往上挪一大截
    （实测面板顶比棋盘顶还高 0.15h 以上）。原来只按 ``panel.y + 0.08h`` 取，
    窗口化后就只剩「调色效果」那一行，OCR 读不出标题和目标，方向和
    作用颜料全成 UNKNOWN。所以向上多留一截，宁可多读几行——
    :func:`paint_panel.parse_lines` 从标题行开始解析，多余的行会被丢掉。

    **下边界同样放过**（2026-11-15 实机事故）：窗口 1902x1010 下，面板标题落在
    ``panel.y + 0.83h``、放置目标在 ``+1.1h``、调色效果在 ``+1.4h``、笔触效果在
    ``+2.4h``，而原来只取到 ``+1.0h``，OCR 只读到标题上半行 → ``reading.confidence``
    只有 0.40（= 类别 0.24 + 等级 0.16），低于 ``_PANEL_TEXT_MIN_CONFIDENCE`` 0.55，
    整份覆盖被丢掉，于是「玩法颜料-机械之神 LV3 / 纵向」被认成
    「怪物 / 方向未知 / 家族空」。现在取到 ``+3.4h``，标题、放置目标、调色效果、
    笔触效果四行都在框内（实测 151px 的面板高正好卡到笔触效果行）。
    """
    width = image.shape[1]
    height = image.shape[0]
    rect = _bounded_rect(
        panel.x + panel.w * 0.02,
        panel.y - panel.h * 1.20,
        panel.w * 0.96,
        panel.h * 4.60,
        width,
        height,
    )
    return image[rect.y : rect.bottom, rect.x : rect.right]


def _analyze_current_paint(
    image: np.ndarray,
    regions: tuple[RegionObservation, ...],
) -> CurrentPaintObservation | None:
    region = next((item for item in regions if item.kind == RegionKind.CURRENT_PAINT), None)
    if region is None:
        return None

    width = image.shape[1]
    height = image.shape[0]
    panel = region.rect
    icon = _locate_current_paint_icon(image, panel)
    text_x = panel.x + panel.w * 0.02
    text = _bounded_rect(
        text_x,
        panel.y + panel.h * 0.08,
        panel.right - text_x - panel.w * 0.02,
        panel.h * 0.78,
        width,
        height,
    )
    category, category_confidence = _classify_current_paint_category(image, icon)
    subtype = None
    is_treasure = False
    level, level_confidence = _detect_level_by_color(image, icon)
    text_crop = image[text.y : text.bottom, text.x : text.right]

    # 面板文字是最终真值，所以**先** OCR，再决定要不要跑字形模板。
    #
    # 2026-11-20：原来顺序是反的——先跑 `_current_paint_keyword_scores`（整套字形
    # 模板匹配，实测 5.3s），读出来之后再让 OCR 覆盖一遍。既然结果一定是文字说了算，
    # 文字读全了就完全没必要跑那 5.3s。冷启动「完整识别」15.8s 里这是一半。
    panel_text = paint_panel.read_panel(_current_paint_text_crop(image, panel))
    overrides = _panel_text_overrides(panel_text)
    text_is_complete = _panel_text_covers_paint_fields(overrides)

    keyword_scores: dict[str, float] = (
        {} if text_is_complete else _current_paint_keyword_scores(text_crop)
    )
    _store_current_panel_signature(keyword_scores, _current_panel_signature(text_crop))
    match_current, match_required = _infer_match_count(keyword_scores)
    if match_required is None and not text_is_complete:
        match_current, match_required = _detect_match_count_hint(text_crop)
    target_object = _infer_target_object(keyword_scores)
    direction = _infer_direction(keyword_scores)
    upgrade_amount = _infer_upgrade_amount(keyword_scores)
    if match_required is None or upgrade_amount is None:
        default_match_required, default_upgrade_amount = _empirical_current_defaults(level)
        if match_required is None:
            match_current, match_required = 0, default_match_required
        if upgrade_amount is None:
            upgrade_amount = default_upgrade_amount

    if overrides:
        category = overrides.get("category", category)  # type: ignore[assignment]
        category_confidence = overrides.get("category_confidence", category_confidence)  # type: ignore[assignment]
        subtype = overrides.get("subtype", subtype)  # type: ignore[assignment]
        is_treasure = overrides.get("is_treasure", is_treasure)  # type: ignore[assignment]
        level = overrides.get("level", level)  # type: ignore[assignment]
        level_confidence = overrides.get("level_confidence", level_confidence)  # type: ignore[assignment]
        target_object = overrides.get("target_object", target_object)  # type: ignore[assignment]
        direction = overrides.get("direction", direction)  # type: ignore[assignment]
        upgrade_amount = overrides.get("upgrade_amount", upgrade_amount)  # type: ignore[assignment]
        match_current = overrides.get("match_current", match_current)  # type: ignore[assignment]
        match_required = overrides.get("match_required", match_required)  # type: ignore[assignment]
        # 记下「目标类别是文字读出来的」，好让背包排序那条旁证不要再翻案。
        keyword_scores["panel_text:confidence"] = float(panel_text.confidence)
        if "target_object" in overrides:
            keyword_scores["panel_text:target"] = 1.0

    raw_text_hint = _build_raw_text_hint(
        category,
        level,
        match_current,
        match_required,
        target_object,
        direction,
        upgrade_amount,
        getattr(panel_text, "effect_text", None),
    )

    return CurrentPaintObservation(
        panel_rect=panel,
        icon_rect=icon,
        text_rect=text,
        category=category,
        category_confidence=category_confidence,
        subtype=subtype,
        is_treasure=is_treasure,
        level=level,
        level_confidence=level_confidence,
        match_current=match_current,
        match_required=match_required,
        target_object=target_object,
        direction=direction,
        upgrade_amount=upgrade_amount,
        keyword_scores=keyword_scores,
        raw_text_hint=raw_text_hint,
        effect_text=getattr(panel_text, "effect_text", None),
    )


def _carry_confirmed_target(
    previous: CurrentPaintObservation,
    keyword_scores: dict[str, float],
    detected: PaintObject,
) -> PaintObject:
    """上一帧文字确认过「作用对象」时，别让栅格模板把它翻掉。

    实机证据（2026-11-14）：同一块面板，文字 OCR 读出「所有」
    （``panel_text:target = 1.0``），栅格模板却给「玩法」0.90~0.96、
    「所有」只有 0.28——差得很远，于是 ``_infer_target_object`` 的 0.88 短路口
    每次都采信栅格，作用对象就在「所有 / 玩法」之间来回翻。
    文字才是最终真值（见 ``_analyze_current_paint_fast`` 末尾的 override）；
    这一帧只是**还没读到文字**（异步 OCR 有节流），所以沿用上一帧的文字结论。
    """
    if float(previous.keyword_scores.get("panel_text:target", 0.0)) < 0.5:
        return detected if detected != PaintObject.UNKNOWN else previous.target_object
    if float(keyword_scores.get("panel_text:target", 0.0)) >= 0.5:
        # 本帧自己也读到文字了，交给 override 定论，这里不抢。
        return detected if detected != PaintObject.UNKNOWN else previous.target_object
    return previous.target_object


def _analyze_current_paint_fast(
    image: np.ndarray,
    regions: tuple[RegionObservation, ...],
    previous: CurrentPaintObservation | None = None,
) -> CurrentPaintObservation | None:
    region = next((item for item in regions if item.kind == RegionKind.CURRENT_PAINT), None)
    if region is None:
        return previous

    width = image.shape[1]
    height = image.shape[0]
    panel = region.rect
    icon = _locate_current_paint_icon(image, panel)
    text_x = panel.x + panel.w * 0.02
    text = _bounded_rect(
        text_x,
        panel.y + panel.h * 0.08,
        panel.right - text_x - panel.w * 0.02,
        panel.h * 0.78,
        width,
        height,
    )
    category, category_confidence = _classify_current_paint_category(image, icon)
    level, level_confidence = _detect_level_by_color(image, icon)
    text_crop = image[text.y : text.bottom, text.x : text.right]
    if level is None or level_confidence < 0.38:
        hint_level, hint_confidence = _detect_level_hint(text_crop)
        if hint_level is not None and hint_confidence > level_confidence:
            level, level_confidence = hint_level, hint_confidence

    keyword_scores = _current_paint_raster_direction_scores(text_crop)
    keyword_scores.update(_current_paint_raster_target_scores(text_crop))
    signature = _current_panel_signature(text_crop)
    previous_signature = _current_panel_signature_from_scores(previous.keyword_scores) if previous is not None else None
    panel_changed = (
        previous_signature is not None
        and float(np.mean(np.abs(np.asarray(signature) - np.asarray(previous_signature)))) > 0.045
    )
    _store_current_panel_signature(keyword_scores, signature)
    detected_target = _infer_target_object(keyword_scores)
    detected_direction = _infer_direction(keyword_scores)

    if previous is not None and not panel_changed:
        if category == PaintCategory.UNKNOWN or category_confidence < 0.46:
            category = previous.category
            category_confidence = max(0.42, previous.category_confidence * 0.72)
        if level is None or level_confidence < 0.40:
            level = previous.level
            level_confidence = max(0.36, previous.level_confidence * 0.72)
        target_object = _carry_confirmed_target(previous, keyword_scores, detected_target)
        direction = detected_direction if detected_direction != PaintDirection.UNKNOWN else previous.direction
        match_current = previous.match_current
        match_required = previous.match_required
        upgrade_amount = previous.upgrade_amount
        subtype = previous.subtype
        is_treasure = previous.is_treasure
    else:
        target_object = detected_target
        direction = detected_direction
        match_current = 0
        match_required = None
        upgrade_amount = None
        subtype = None
        is_treasure = False

    if match_required is None or upgrade_amount is None:
        default_match_required, default_upgrade_amount = _empirical_current_defaults(level)
        match_current = 0 if match_current is None else match_current
        match_required = default_match_required if match_required is None else match_required
        upgrade_amount = default_upgrade_amount if upgrade_amount is None else upgrade_amount

    # 实时帧不能等 OCR：按时间节流，其余帧复用上一次成功的结果。
    panel_text = paint_panel.read_panel_async(_current_paint_text_crop(image, panel))
    overrides = _panel_text_overrides(panel_text)
    if overrides:
        category = overrides.get("category", category)  # type: ignore[assignment]
        category_confidence = overrides.get("category_confidence", category_confidence)  # type: ignore[assignment]
        subtype = overrides.get("subtype", subtype)  # type: ignore[assignment]
        is_treasure = overrides.get("is_treasure", is_treasure)  # type: ignore[assignment]
        level = overrides.get("level", level)  # type: ignore[assignment]
        level_confidence = overrides.get("level_confidence", level_confidence)  # type: ignore[assignment]
        target_object = overrides.get("target_object", target_object)  # type: ignore[assignment]
        direction = overrides.get("direction", direction)  # type: ignore[assignment]
        upgrade_amount = overrides.get("upgrade_amount", upgrade_amount)  # type: ignore[assignment]
        match_current = overrides.get("match_current", match_current)  # type: ignore[assignment]
        match_required = overrides.get("match_required", match_required)  # type: ignore[assignment]
        # 记下「目标类别是文字读出来的」，好让背包排序那条旁证不要再翻案。
        keyword_scores["panel_text:confidence"] = float(panel_text.confidence)
        if "target_object" in overrides:
            keyword_scores["panel_text:target"] = 1.0

    raw_text_hint = _build_raw_text_hint(
        category,
        level,
        match_current,
        match_required,
        target_object,
        direction,
        upgrade_amount,
        getattr(panel_text, "effect_text", None),
    )
    if previous is not None and not panel_changed:
        raw_text_hint = f"{raw_text_hint} fast"

    return CurrentPaintObservation(
        panel_rect=panel,
        icon_rect=icon,
        text_rect=text,
        category=category,
        category_confidence=category_confidence,
        subtype=subtype,
        is_treasure=is_treasure,
        level=level,
        level_confidence=level_confidence,
        match_current=match_current,
        match_required=match_required,
        target_object=target_object,
        direction=direction,
        upgrade_amount=upgrade_amount,
        keyword_scores=keyword_scores,
        raw_text_hint=raw_text_hint,
        effect_text=getattr(panel_text, "effect_text", None),
    )


def _current_panel_signature(crop: np.ndarray) -> tuple[float, ...]:
    if crop.size == 0:
        return (0.0,) * 48
    small = cv2.resize(crop, (8, 2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return tuple(float(value) for value in small.reshape(-1))


def _store_current_panel_signature(scores: dict[str, float], signature: tuple[float, ...]) -> None:
    for index, value in enumerate(signature):
        scores[f"current_visual_{index}"] = value


def _current_panel_signature_from_scores(scores: dict[str, float]) -> tuple[float, ...] | None:
    values: list[float] = []
    for index in range(48):
        key = f"current_visual_{index}"
        if key not in scores:
            return None
        values.append(float(scores[key]))
    return tuple(values)


def _empirical_current_defaults(level: int | None) -> tuple[int, int]:
    if level is None:
        return 3, 1
    if level >= 5:
        return 4, 2
    if level == 4:
        return 4, 1
    return 3, 1


def _locate_current_paint_icon(image: np.ndarray, panel: Rect) -> Rect:
    width = image.shape[1]
    height = image.shape[0]
    search = _bounded_rect(
        panel.x,
        panel.y,
        panel.w * 0.34,
        panel.h,
        width,
        height,
    )
    roi = image[search.y : search.bottom, search.x : search.right]
    if roi.size == 0:
        side = max(1, int(panel.h * 0.72))
        return _bounded_rect(panel.x, panel.y, side, side, width, height)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s > 70) & (v > 100) & (((h >= 120) & (h <= 170)) | ((h >= 20) & (h <= 45)))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, w, h_box = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < max(12.0, roi.shape[0] * roi.shape[1] * 0.001):
            continue
        aspect = w / max(1, h_box)
        if not 0.28 <= aspect <= 2.20:
            continue
        side = max(w, h_box)
        if side < panel.h * 0.16 or side > panel.h * 0.62:
            continue
        left_bonus = max(0.0, 1.0 - x / max(1, search.w * 0.42))
        shape_score = 1.0 - min(1.0, abs(np.log(max(0.1, aspect))) / 1.4)
        size_score = min(1.0, side / max(1.0, panel.h * 0.34))
        candidates.append((area * (0.55 + left_bonus) * (0.55 + shape_score) * (0.75 + size_score), (x, y, w, h_box)))

    if candidates:
        _, (x, y, w, h_box) = max(candidates, key=lambda item: item[0])
        side = int(np.clip(max(w, h_box) * 1.45, panel.h * 0.24, panel.h * 0.56))
        cx = search.x + x + w / 2
        cy = search.y + y + h_box / 2
        return _bounded_rect(cx - side / 2, cy - side / 2, side, side, width, height)

    side = max(1, int(panel.h * 0.72))
    return _bounded_rect(panel.x + panel.w * 0.05, panel.y + panel.h * 0.14, side, side, width, height)


def _classify_current_paint_category(image: np.ndarray, icon: Rect) -> tuple[PaintCategory, float]:
    crop = image[icon.y : icon.bottom, icon.x : icon.right]
    if crop.size == 0:
        return PaintCategory.UNKNOWN, 0.0

    frame_category, frame_confidence = _classify_backpack_frame_shape(crop)
    if frame_category != PaintCategory.UNKNOWN and frame_confidence >= 0.68:
        return frame_category, frame_confidence

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, min(gray.shape[:2]) // 2),
        param1=80,
        param2=18,
        minRadius=max(8, min(gray.shape[:2]) // 5),
        maxRadius=max(12, min(gray.shape[:2]) // 2),
    )
    circle_score = 0.0 if circles is None else min(1.0, circles.shape[1] / 2.0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diamond_score = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < crop.shape[0] * crop.shape[1] * 0.08:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / max(1, h)
            if 0.75 <= aspect <= 1.25:
                diamond_score = max(diamond_score, min(1.0, area / (crop.shape[0] * crop.shape[1] * 0.45)))

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    saturated = ((s > 55) & (v > 75)).astype(np.uint8) * 255
    saturated = cv2.morphologyEx(
        saturated,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    color_contours, _ = cv2.findContours(saturated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in color_contours:
        area = cv2.contourArea(contour)
        if area < crop.shape[0] * crop.shape[1] * 0.06:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.075 * peri, True)
        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect = w_box / max(1, h_box)
        circularity = 4.0 * np.pi * area / max(1.0, peri * peri)
        touches_crop_edge = x <= 1 or y <= 1 or x + w_box >= crop.shape[1] - 1 or y + h_box >= crop.shape[0] - 1
        if len(approx) == 4 and 0.68 <= aspect <= 1.35 and (not touches_crop_edge or circularity <= 0.62):
            diamond_score = max(diamond_score, min(1.0, area / (crop.shape[0] * crop.shape[1] * 0.36)))

    # Gameplay paints use a diamond/rhombus icon. Hough circles often fire on
    # highlights inside that diamond, but a strong outer circular frame must
    # still win for monster icons.
    if frame_category == PaintCategory.MONSTER and frame_confidence >= 0.58:
        return PaintCategory.MONSTER, max(0.62, frame_confidence)
    if diamond_score >= 0.24:
        return PaintCategory.GAMEPLAY, max(0.62, min(0.92, diamond_score))
    if circle_score >= 0.5:
        return PaintCategory.MONSTER, 0.72
    if frame_category != PaintCategory.UNKNOWN:
        return frame_category, frame_confidence
    return PaintCategory.UNKNOWN, 0.35


def _detect_level_by_color(image: np.ndarray, icon: Rect) -> tuple[int | None, float]:
    crop = image[icon.y : icon.bottom, icon.x : icon.right]
    if crop.size == 0:
        return None, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    yy, xx = np.indices(h.shape)
    cx = (h.shape[1] - 1) / 2.0
    cy = (h.shape[0] - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(1.0, min(h.shape[:2]) / 2.0)
    ring = (radius >= 0.45) & (radius <= 0.90) & (s > 55) & (v > 70)

    if not np.any(ring):
        return None, 0.0

    scores = {
        2: float(np.mean(ring & (h >= 45) & (h <= 85))),   # green
        3: float(np.mean(ring & (h >= 90) & (h <= 115))),  # blue
        4: float(np.mean(ring & (h >= 125) & (h <= 165))), # purple
        5: float(np.mean(ring & (h >= 8) & (h <= 24))),    # orange
        6: float(np.mean(ring & (((h >= 0) & (h <= 8)) | (h >= 170)))), # red
    }
    best_level, best_score = max(scores.items(), key=lambda item: item[1])
    ring_ratio = float(np.mean(ring))
    if best_score < max(0.015, ring_ratio * 0.08):
        return None, 0.0
    return best_level, float(np.clip(best_score / max(0.001, ring_ratio * 0.55), 0.45, 0.98))


def _detect_level_hint(text_crop: np.ndarray) -> tuple[int | None, float]:
    if text_crop.size == 0:
        return None, 0.0

    title = text_crop[: max(1, int(text_crop.shape[0] * 0.34)), :]
    gray = cv2.cvtColor(title, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < binary.shape[0] * 0.30 or w < 5:
            continue
        if x < binary.shape[1] * 0.35:
            continue
        candidates.append((x, y, w, h))

    # This is a lightweight hint until OCR lands. In the provided UI the title
    # line contains "LV4"; the final bright digit is usually the rightmost
    # compact glyph in the title row.
    if not candidates:
        return None, 0.0

    right_side = sorted(candidates, key=lambda item: item[0] + item[2], reverse=True)[:3]
    for x, y, w, h in right_side:
        digit_crop = binary[y : y + h, x : x + w]
        digit = _match_digit_template(digit_crop)
        if digit is not None:
            return digit, 0.55
    return None, 0.25


def _detect_match_count_hint(text_crop: np.ndarray) -> tuple[int | None, int | None]:
    if text_crop.size == 0:
        return None, None
    blue_line = _blue_text_mask(text_crop)
    if not np.any(blue_line):
        return None, None

    # The completion text ends with "(0/N)". We only infer the common required
    # value from the count of compact bright glyphs near the right edge.
    h, w = blue_line.shape
    right = blue_line[:, int(w * 0.72) :]
    contours, _ = cv2.findContours(right.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh >= h * 0.12 and bw >= 2:
            boxes.append((x, y, bw, bh))
    if len(boxes) >= 2:
        return 0, 4
    return None, None


def _current_paint_keyword_scores(text_crop: np.ndarray) -> dict[str, float]:
    target_line = _crop_current_target_line(text_crop)
    effect_line = _crop_current_effect_line(text_crop)
    scores: dict[str, float] = {}
    target_keywords = (
        "横向",
        "相邻",
        "纵向",
        "对角斜向",
        "全图",
        "所有",
        "怪物",
        "玩法",
        "阻挡地块",
        "空白地块",
        "虚空地块",
        "宝箱",
        "(0/3)",
        "(0/4)",
    )
    effect_keywords = (
        "提升1级",
        "提升2级",
        "升1级",
        "升2级",
        "横向",
        "相邻",
        "纵向",
        "对角斜向",
        "全图",
        "所有",
        "怪物",
        "玩法",
    )
    scores.update(
        {
            f"target:{key}": value
            for key, value in match_keywords(
                target_line,
                target_keywords,
                font_size_range=range(12, 30, 2),
                scale=1.8,
                include_binary=True,
                threshold=0.0,
            ).items()
        }
    )
    scores.update(
        {
            f"effect:{key}": value
            for key, value in match_keywords(
                effect_line,
                effect_keywords,
                font_size_range=range(12, 30, 3),
                scale=1.5,
                include_binary=False,
                threshold=0.0,
            ).items()
        }
    )
    scores.update(_current_paint_color_hint_scores(target_line))
    scores.update(_current_paint_raster_direction_scores(text_crop))
    return scores


def _current_paint_raster_direction_scores(text_crop: np.ndarray) -> dict[str, float]:
    return {
        f"raster:direction:{keyword}": score
        for keyword, score in _match_current_paint_raster_templates(
            text_crop,
            {keyword: (filename,) for keyword, filename in CURRENT_DIRECTION_TEMPLATE_FILES.items()},
        ).items()
    }


def _current_paint_raster_target_scores(text_crop: np.ndarray) -> dict[str, float]:
    return {
        f"raster:target:{keyword}": score
        for keyword, score in _match_current_paint_raster_templates(
            text_crop,
            CURRENT_TARGET_TEMPLATE_FILES,
        ).items()
    }


def _match_current_paint_raster_templates(
    text_crop: np.ndarray,
    template_files: dict[str, tuple[str, ...]],
) -> dict[str, float]:
    if text_crop.size == 0:
        return {}
    canonical_w, canonical_h = CURRENT_PAINT_CANONICAL_TEXT_SIZE
    normalized = cv2.resize(text_crop, (canonical_w, canonical_h), interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = CURRENT_PAINT_KEYWORD_ROI
    roi = normalized[y1:y2, x1:x2]
    scores: dict[str, float] = {}
    for keyword, filenames in template_files.items():
        best = 0.0
        for filename in filenames:
            template = _load_current_paint_template(filename)
            if template is None or template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
                continue
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, _ = cv2.minMaxLoc(result)
            best = max(best, float(max_value))
        scores[keyword] = best
    return scores


@lru_cache(maxsize=16)
def _load_current_paint_template(filename: str) -> np.ndarray | None:
    path = CURRENT_PAINT_TEMPLATE_DIR / filename
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _current_paint_color_hint_scores(target_line: np.ndarray) -> dict[str, float]:
    if target_line.size == 0:
        return {}

    hsv = cv2.cvtColor(target_line, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    scores: dict[str, float] = {}
    red_mask = (((h <= 18) | (h >= 165)) & (s > 50) & (v > 75)).astype(np.uint8) * 255
    blue_mask = ((h >= 86) & (h <= 114) & (s > 45) & (v > 75)).astype(np.uint8) * 255
    white_mask = ((s < 105) & (v > 130)).astype(np.uint8) * 255

    direction_scores = _keyword_scores_from_mask(
        red_mask,
        ("横向", "纵向", "对角斜向", "相邻", "全图"),
        target_line.shape,
        font_size_range=range(12, 30, 2),
        scale=2.0,
    )
    for key, value in direction_scores.items():
        scores[f"color:direction:{key}"] = value
    long_red_width = _largest_text_component_width(red_mask)
    if long_red_width >= max(70, int(target_line.shape[1] * 0.10)):
        scores["color:direction:long_diagonal"] = 0.93

    object_scores = _keyword_scores_from_mask(
        blue_mask | white_mask,
        ("怪物", "玩法", "所有", "宝箱", "(0/3)", "(0/4)"),
        target_line.shape,
        font_size_range=range(12, 30, 2),
        scale=2.0,
    )
    for key, value in object_scores.items():
        scores[f"color:target:{key}"] = value
    return scores


def _keyword_scores_from_mask(
    mask: np.ndarray,
    keywords: tuple[str, ...],
    shape: tuple[int, ...],
    *,
    font_size_range: range,
    scale: float,
) -> dict[str, float]:
    if mask.size == 0 or float(np.mean(mask > 0)) < 0.002:
        return {}
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    text_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    return match_keywords(
        text_bgr,
        keywords,
        font_size_range=font_size_range,
        scale=scale,
        include_binary=True,
        threshold=0.0,
    )


def _infer_target_object(scores: dict[str, float]) -> PaintObject:
    raster_mapping = {
        "玩法": PaintObject.GAMEPLAY,
        "怪物": PaintObject.MONSTER,
        "所有": PaintObject.ALL,
    }
    raster_ranked = sorted(
        ((key, scores.get(f"raster:target:{key}", 0.0)) for key in raster_mapping),
        key=lambda item: item[1],
        reverse=True,
    )
    if raster_ranked and raster_ranked[0][1] >= 0.88:
        runner_up = raster_ranked[1][1] if len(raster_ranked) > 1 else 0.0
        if raster_ranked[0][1] - runner_up >= 0.08:
            return raster_mapping[raster_ranked[0][0]]

    object_scores = {
        PaintObject.MONSTER: max(scores.get("target:怪物", 0.0), scores.get("color:target:怪物", 0.0), scores.get("effect:怪物", 0.0)),
        PaintObject.GAMEPLAY: max(scores.get("target:玩法", 0.0), scores.get("color:target:玩法", 0.0), scores.get("effect:玩法", 0.0)),
        PaintObject.ALL: max(scores.get("target:所有", 0.0), scores.get("color:target:所有", 0.0), scores.get("effect:所有", 0.0)),
        PaintObject.BLOCKED_TILE: scores.get("target:阻挡地块", 0.0),
        PaintObject.BLANK_TILE: scores.get("target:空白地块", 0.0),
        PaintObject.VOID_TILE: scores.get("target:虚空地块", 0.0),
    }
    target, score = max(object_scores.items(), key=lambda item: item[1])
    sorted_scores = sorted(object_scores.values(), reverse=True)
    if score < 0.42 or (len(sorted_scores) > 1 and score - sorted_scores[1] < 0.035):
        return PaintObject.UNKNOWN
    return target


def _infer_direction(scores: dict[str, float]) -> PaintDirection:
    mapping = {
        "横向": PaintDirection.HORIZONTAL,
        "纵向": PaintDirection.VERTICAL,
        "对角斜向": PaintDirection.DIAGONAL,
        "相邻": PaintDirection.ADJACENT,
        "全图": PaintDirection.GLOBAL,
    }
    raster_scores = {
        keyword: scores.get(f"raster:direction:{keyword}", 0.0)
        for keyword in CURRENT_DIRECTION_TEMPLATE_FILES
    }
    raster_ranked = sorted(raster_scores.items(), key=lambda item: item[1], reverse=True)
    if raster_ranked and raster_ranked[0][1] >= 0.78:
        runner_up = raster_ranked[1][1] if len(raster_ranked) > 1 else 0.0
        if raster_ranked[0][1] - runner_up >= 0.08:
            return mapping[raster_ranked[0][0]]

    direction_scores = {
        "横向": max(scores.get("target:横向", 0.0), scores.get("color:direction:横向", 0.0), scores.get("effect:横向", 0.0)),
        "纵向": max(scores.get("target:纵向", 0.0), scores.get("color:direction:纵向", 0.0), scores.get("effect:纵向", 0.0)),
        "对角斜向": max(
            scores.get("target:对角斜向", 0.0),
            scores.get("color:direction:diagonal", 0.0),
            scores.get("color:direction:对角斜向", 0.0),
            scores.get("color:direction:long_diagonal", 0.0),
            scores.get("effect:对角斜向", 0.0),
        ),
        "相邻": max(scores.get("target:相邻", 0.0), scores.get("color:direction:相邻", 0.0), scores.get("effect:相邻", 0.0)),
        "全图": max(scores.get("target:全图", 0.0), scores.get("color:direction:全图", 0.0), scores.get("effect:全图", 0.0)),
    }
    # Long direction labels contain short direction glyphs. If the full
    # "对角斜向" phrase is reasonably strong, do not let a partial "横向/纵向"
    # template steal the result.
    diagonal_color = max(
        scores.get("color:direction:diagonal", 0.0),
        scores.get("color:direction:对角斜向", 0.0),
        scores.get("color:direction:long_diagonal", 0.0),
    )
    if diagonal_color >= 0.60:
        direction_scores["横向"] *= 0.72
        direction_scores["纵向"] *= 0.78
    keyword, score = max(direction_scores.items(), key=lambda item: item[1])
    sorted_scores = sorted(direction_scores.values(), reverse=True)
    if score < 0.42 or (len(sorted_scores) > 1 and score - sorted_scores[1] < 0.035):
        return PaintDirection.UNKNOWN
    return mapping[keyword]


def _infer_match_count(scores: dict[str, float]) -> tuple[int | None, int | None]:
    three = max(scores.get("target:(0/3)", 0.0), scores.get("color:target:(0/3)", 0.0))
    four = max(scores.get("target:(0/4)", 0.0), scores.get("color:target:(0/4)", 0.0))
    if max(three, four) < 0.42 or abs(three - four) < 0.05:
        return None, None
    return (0, 3) if three > four else (0, 4)


def _crop_current_target_line(text_crop: np.ndarray) -> np.ndarray:
    if text_crop.size == 0:
        return text_crop
    target_band = _locate_current_text_band(text_crop, prefer_blue=True)
    if target_band is not None:
        top, bottom = target_band
        return text_crop[top:bottom, :]
    top = int(text_crop.shape[0] * 0.18)
    bottom = int(text_crop.shape[0] * 0.58)
    return text_crop[top:max(top + 1, bottom), :]


def _crop_current_effect_line(text_crop: np.ndarray) -> np.ndarray:
    if text_crop.size == 0:
        return text_crop
    effect_band = _locate_current_text_band(text_crop, prefer_blue=False)
    if effect_band is not None:
        top, bottom = effect_band
        return text_crop[top:bottom, :]
    return text_crop[int(text_crop.shape[0] * 0.52) :, :]


def _largest_text_component_width(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2)), iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    widths: list[int] = []
    min_area = max(8, int(mask.shape[0] * mask.shape[1] * 0.00015))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area or h < 5:
            continue
        widths.append(w)
    return max(widths, default=0)


def _locate_current_text_band(text_crop: np.ndarray, *, prefer_blue: bool) -> tuple[int, int] | None:
    hsv = cv2.cvtColor(text_crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    blue = (h >= 86) & (h <= 114) & (s > 45) & (v > 75)
    yellow = (h >= 18) & (h <= 45) & (s > 45) & (v > 95)
    red = (((h <= 18) | (h >= 165)) & (s > 50) & (v > 75))
    white = (s < 115) & (v > 135)
    if prefer_blue:
        mask = blue | red | white
        y_scores = np.mean(mask, axis=1)
        search_start = int(text_crop.shape[0] * 0.12)
        search_end = int(text_crop.shape[0] * 0.68)
    else:
        mask = yellow | white
        y_scores = np.mean(mask, axis=1)
        search_start = int(text_crop.shape[0] * 0.42)
        search_end = text_crop.shape[0]
    if search_end <= search_start:
        return None
    window = y_scores[search_start:search_end]
    if window.size == 0 or float(np.max(window)) < 0.006:
        return None
    peak = search_start + int(np.argmax(window))
    threshold = max(0.004, float(np.max(window)) * 0.30)
    top = peak
    while top > search_start and y_scores[top - 1] >= threshold:
        top -= 1
    bottom = peak + 1
    while bottom < search_end and y_scores[bottom] >= threshold:
        bottom += 1
    pad = max(3, text_crop.shape[0] // 18)
    top = max(0, top - pad)
    bottom = min(text_crop.shape[0], bottom + pad)
    if bottom - top < max(8, text_crop.shape[0] * 0.10):
        return None
    return top, bottom


def _infer_upgrade_amount(scores: dict[str, float]) -> int | None:
    one = max(scores.get("effect:提升1级", 0.0), scores.get("effect:升1级", 0.0))
    two = max(scores.get("effect:提升2级", 0.0), scores.get("effect:升2级", 0.0))
    if max(one, two) < 0.44 or abs(one - two) < 0.025:
        return None
    if two > one:
        return 2
    return 1


def _blue_text_mask(text_crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(text_crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    return (h >= 88) & (h <= 112) & (s > 55) & (v > 80)


def _bright_text_mask(text_crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(text_crop, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    return (s < 120) & (v > 125)


def _match_digit_template(digit_crop: np.ndarray) -> int | None:
    if digit_crop.size == 0:
        return None
    norm = cv2.resize(digit_crop, (14, 22), interpolation=cv2.INTER_AREA)
    _, norm = cv2.threshold(norm, 127, 255, cv2.THRESH_BINARY)
    templates = _digit_templates()
    scores = []
    for digit, template in templates.items():
        diff = float(np.mean(np.abs(norm.astype(np.float32) - template.astype(np.float32))) / 255.0)
        scores.append((diff, digit))
    scores.sort()
    if scores[0][0] > 0.48:
        return None
    return scores[0][1]


def _digit_templates() -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for digit in range(1, 8):
        canvas = np.zeros((22, 14), dtype=np.uint8)
        cv2.putText(
            canvas,
            str(digit),
            (1, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            255,
            2,
            cv2.LINE_AA,
        )
        _, canvas = cv2.threshold(canvas, 20, 255, cv2.THRESH_BINARY)
        templates[digit] = canvas
    return templates


def _build_raw_text_hint(
    category: PaintCategory,
    level: int | None,
    match_current: int | None,
    match_required: int | None,
    target_object: PaintObject,
    direction: PaintDirection,
    upgrade_amount: int | None,
    effect_text: str | None = None,
) -> str:
    parts = [f"category={category.value}"]
    if level is not None:
        parts.append(f"level={level}")
    if match_current is not None and match_required is not None:
        parts.append(f"match={match_current}/{match_required}")
    if target_object != PaintObject.UNKNOWN:
        parts.append(f"target={target_object.value}")
    if direction != PaintDirection.UNKNOWN:
        parts.append(f"direction={direction.value}")
    if upgrade_amount is not None:
        parts.append(f"upgrade={upgrade_amount}")
    # 笔触（调色效果）原文。以前这一行解析完就扔了，复盘时查不到「这笔吃的是哪条效果」。
    if effect_text:
        parts.append(f"effect={effect_text}")
    return " ".join(parts)


def _analyze_brush_options(
    image: np.ndarray,
    regions: tuple[RegionObservation, ...],
) -> BrushOptionsObservation | None:
    region = next((item for item in regions if item.kind == RegionKind.BRUSH_OPTIONS), None)
    if region is None:
        return None

    panel = region.rect
    crop = image[panel.y : panel.bottom, panel.x : panel.right]
    if crop.size == 0:
        return None
    crop = _trim_blank_right(crop)

    # 面板文字是最终真值，先走它。以前靠「≥2 张卡片就算方向阶段」区分阶段，
    # 结果「选笔触效果」阶段被当成「选笔触方向」，报出一堆假的笔锋类型。
    # 现在用「三条固定描述有没有出现」判定，8 张实拍截图全对。
    panel_reading = brush_panel.read_brush_panel(crop)
    if panel_reading is not None:
        if panel_reading.stage == "direction" and panel_reading.brush_types:
            options = tuple(item.value for item in panel_reading.brush_types)
            _ignored, selected_by_card = _brush_stage1_options_from_cards(crop)
            if selected_by_card not in options:
                selected_by_card = None
            return BrushOptionsObservation(
                panel_rect=panel,
                stage1_options=options,
                stage1_scores={option: 0.90 for option in options},
                chosen_stage1=selected_by_card,
                stage2_options=(),
                stage2_scores={},
                keyword_scores={},
                confidence=float(panel_reading.confidence),
            )
        if panel_reading.stage == "effect" and panel_reading.affixes:
            chosen_stage1 = _infer_chosen_stage1((), panel_reading.affixes)
            return BrushOptionsObservation(
                panel_rect=panel,
                stage1_options=(chosen_stage1,) if chosen_stage1 is not None else (),
                stage1_scores={chosen_stage1: 0.80} if chosen_stage1 is not None else {},
                chosen_stage1=chosen_stage1,
                stage2_options=panel_reading.affixes,
                stage2_scores=dict(panel_reading.affix_scores),
                keyword_scores={},
                confidence=float(panel_reading.confidence),
            )

    stage1_crop = _brush_stage1_crop(crop)
    stage2_crop = _brush_stage2_crop(crop)
    stage1_scores = _brush_stage1_scores(stage1_crop)
    pattern_scores = _brush_card_pattern_scores(crop)
    for brush_type, score in pattern_scores.items():
        stage1_scores[brush_type] = max(stage1_scores.get(brush_type, 0.0), score)
    stage1_options = _detected_stage1_options(stage1_scores)
    ordered_options, selected_by_card = _brush_stage1_options_from_cards(crop)
    if ordered_options:
        stage1_options = ordered_options
        for option in ordered_options:
            stage1_scores[option] = max(stage1_scores.get(option, 0.0), 0.76)
    if len(ordered_options) >= 2:
        # Two distinct brush-pattern cards prove this is stage 1. Running all
        # affix templates against those descriptive cards creates convincing
        # but false stage-2 matches.
        keyword_scores: dict[str, float] = {}
        stage2_scores: dict[str, float] = {}
        stage2_options: tuple[str, ...] = ()
    else:
        keyword_scores = _brush_keyword_scores(stage2_crop, stage1_options)
        stage2_scores = _brush_stage2_scores(keyword_scores, stage1_scores, stage1_options)
        stage2_options = _detected_stage2_options(stage2_scores)
    chosen_stage1 = _infer_chosen_stage1(stage1_options, stage2_options)
    if selected_by_card is not None:
        chosen_stage1 = selected_by_card
    if not stage1_options and chosen_stage1 is not None:
        stage1_options = (chosen_stage1,)

    confidence_parts = list(stage1_scores.values()) + list(stage2_scores.values())
    confidence = max(confidence_parts) if confidence_parts else 0.0
    return BrushOptionsObservation(
        panel_rect=panel,
        stage1_options=stage1_options,
        stage1_scores=stage1_scores,
        chosen_stage1=chosen_stage1,
        stage2_options=stage2_options,
        stage2_scores=stage2_scores,
        keyword_scores=keyword_scores,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def _trim_blank_right(crop: np.ndarray) -> np.ndarray:
    """裁掉右侧整列无内容的空白，再交给分带判据。

    为什么需要：笔触面板 ROI 的右边界是靠 ``b.right + 1.04·b.w`` 估的，估宽了就会
    把右侧大片空白雪地圈进来。而 :func:`brush_panel._split_brush_cards` 用**整幅宽度**
    求行活动度均值（``active.mean(axis=1)``），多出来的空白列会把它稀释到阈值以下，
    于是「上下两张卡」判不出来、面板直接检测失败——2026-11-22 17:30 之后
    ``brush_options`` 归零就是这个现象。

    这里不依赖任何面板常数：从右侧往里找第一列有内容的，再留一点边距。
    """

    if crop.size == 0:
        return crop
    height, width = crop.shape[:2]
    if width < 40:
        return crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 「有内容」= 足够饱和（卡片底色/图标）或明显暗（文字与阴影）。
    interesting = (hsv[:, :, 1] > 24) | (hsv[:, :, 2] < 110)
    column_ratio = interesting.mean(axis=0)
    active = np.where(column_ratio > 0.05)[0]
    if len(active) == 0:
        return crop
    right = min(width, int(active[-1]) + max(4, int(width * 0.02)) + 1)
    if right >= width:
        return crop
    # 裁得太狠会切掉卡片正文，至少要保留 55% 宽度。
    if right < width * 0.55:
        return crop
    return crop[:, :right]


def _analyze_brush_options_realtime(
    image: np.ndarray,
    regions: tuple[RegionObservation, ...],
    previous: BrushOptionsObservation | None = None,
) -> BrushOptionsObservation | None:
    """Recognize brush cards without the expensive general text matcher.

    ``previous`` 是上一帧的结果。面板 OCR 走异步，第一次调用只会在后台起线程、
    当帧返回 ``None``；这时如果直接掉进下面的字形模板兜底，一帧要多花 150ms，
    而面板压根没换。所以面板区没动过就沿用上一帧，下一帧 OCR 结果自然补上。
    """

    region = next((item for item in regions if item.kind == RegionKind.BRUSH_OPTIONS), None)
    if region is None:
        return None
    panel = region.rect
    crop = image[panel.y : panel.bottom, panel.x : panel.right]
    if crop.size == 0:
        return None
    crop = _trim_blank_right(crop)

    # 实时帧：面板文字缓存命中就立即用，没命中丢后台线程，这一帧走原来的兜底。
    panel_reading = brush_panel.read_brush_panel_async(crop)
    if panel_reading is None and previous is not None and previous.panel_rect == panel:
        return previous
    if panel_reading is not None:
        if panel_reading.stage == "direction" and panel_reading.brush_types:
            options = tuple(item.value for item in panel_reading.brush_types)
            _ignored, selected_by_card = _brush_stage1_options_from_cards(crop)
            if selected_by_card not in options:
                selected_by_card = None
            return BrushOptionsObservation(
                panel_rect=panel,
                stage1_options=options,
                stage1_scores={option: 0.90 for option in options},
                chosen_stage1=selected_by_card,
                stage2_options=(),
                stage2_scores={},
                keyword_scores={},
                confidence=float(panel_reading.confidence),
            )
        if panel_reading.stage == "effect" and panel_reading.affixes:
            chosen_stage1 = _infer_chosen_stage1((), panel_reading.affixes)
            return BrushOptionsObservation(
                panel_rect=panel,
                stage1_options=(chosen_stage1,) if chosen_stage1 is not None else (),
                stage1_scores={chosen_stage1: 0.80} if chosen_stage1 is not None else {},
                chosen_stage1=chosen_stage1,
                stage2_options=panel_reading.affixes,
                stage2_scores=dict(panel_reading.affix_scores),
                keyword_scores={},
                confidence=float(panel_reading.confidence),
            )

    stage2_options, stage2_scores = _match_brush_stage2_raster_templates(crop)
    if len(stage2_options) == 2:
        chosen_stage1 = _infer_chosen_stage1((), stage2_options)
        stage1_scores = {chosen_stage1: max(stage2_scores.values())} if chosen_stage1 is not None else {}
        return BrushOptionsObservation(
            panel_rect=panel,
            stage1_options=(chosen_stage1,) if chosen_stage1 is not None else (),
            stage1_scores=stage1_scores,
            chosen_stage1=chosen_stage1,
            stage2_options=stage2_options,
            stage2_scores=stage2_scores,
            keyword_scores={},
            confidence=max(stage2_scores.values(), default=0.0),
        )

    ordered_options, selected_by_card = _brush_stage1_options_from_cards(crop)
    if len(ordered_options) < 2:
        stage2_options, stage2_scores = _match_brush_stage2_windows_ocr(crop)
        if len(stage2_options) == 2:
            chosen_stage1 = _infer_chosen_stage1((), stage2_options)
            stage1_scores = {chosen_stage1: max(stage2_scores.values())} if chosen_stage1 is not None else {}
            return BrushOptionsObservation(
                panel_rect=panel,
                stage1_options=(chosen_stage1,) if chosen_stage1 is not None else (),
                stage1_scores=stage1_scores,
                chosen_stage1=chosen_stage1,
                stage2_options=stage2_options,
                stage2_scores=stage2_scores,
                keyword_scores={},
                confidence=max(stage2_scores.values(), default=0.0),
            )

    if len(ordered_options) >= 2:
        warm_windows_ocr_async()

    pattern_scores = _brush_card_pattern_scores(crop)
    stage1_options = ordered_options or _detected_stage1_options(pattern_scores)
    for option in stage1_options:
        pattern_scores[option] = max(pattern_scores.get(option, 0.0), 0.76)
    confidence = max(pattern_scores.values(), default=0.0)
    return BrushOptionsObservation(
        panel_rect=panel,
        stage1_options=stage1_options,
        stage1_scores=pattern_scores,
        chosen_stage1=selected_by_card,
        stage2_options=(),
        stage2_scores={},
        keyword_scores={},
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def _match_brush_stage2_raster_templates(crop: np.ndarray) -> tuple[tuple[str, ...], dict[str, float]]:
    if crop.size == 0 or not _brush_panel_matches_template_geometry(crop):
        return (), {}
    panel = cv2.resize(crop, BRUSH_STAGE2_CANONICAL_PANEL_SIZE, interpolation=cv2.INTER_AREA)
    templates = _brush_stage2_raster_templates()
    if not templates:
        return (), {}

    options: list[str] = []
    scores: dict[str, float] = {}
    for x, y, w, h in BRUSH_STAGE2_TEXT_RECTS:
        text_crop = panel[y : y + h, x : x + w]
        target = _brush_stage2_text_mask(text_crop)
        ranked = sorted(
            (
                (_aligned_binary_template_score(target, template), name)
                for name, template in templates.items()
                if template.shape == target.shape
            ),
            reverse=True,
        )
        if not ranked:
            continue
        score, name = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if score < 0.48 or score - second_score < 0.045 or name in options:
            continue
        options.append(name)
        scores[name] = score
    return tuple(options), scores


_BRUSH_STAGE2_UI_ALIASES = {
    "范围内随机1个颜料提升1级": ("对作用范围内的随机1个颜料额外提升1级",),
    "范围内随机2个颜料提升1级": ("对作用范围内的随机2个颜料额外提升1级",),
    "全图和自身相同的颜料提升1级": ("全图中与自身内容相同的颜料额外提升1级",),
    "全图和自身相同的颜料提升2级": ("全图中与自身内容相同的颜料额外提升2级",),
}


def _match_brush_stage2_windows_ocr(crop: np.ndarray) -> tuple[tuple[str, ...], dict[str, float]]:
    if crop.size == 0 or not _brush_panel_matches_template_geometry(crop):
        return (), {}
    panel = cv2.resize(crop, BRUSH_STAGE2_CANONICAL_PANEL_SIZE, interpolation=cv2.INTER_AREA)
    text_crops = [panel[y : y + h, x : x + w] for x, y, w, h in BRUSH_STAGE2_TEXT_RECTS]
    combined = np.vstack(text_crops)
    lines = recognize_chinese_lines_async(combined)
    if lines is None:
        return (), {}
    return _match_brush_stage2_ocr_lines(lines)


def _match_brush_stage2_ocr_lines(lines: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, float]]:
    normalized_lines = tuple(
        normalized
        for line in lines
        if len(normalized := _normalize_brush_ocr_text(line)) >= 4
    )
    if not normalized_lines:
        return (), {}

    candidates: list[tuple[str, str, str]] = []
    for affix in build_affix_catalog():
        aliases = (
            affix.display_name,
            *affix.aliases,
            *_BRUSH_STAGE2_UI_ALIASES.get(affix.name, ()),
        )
        for alias in aliases:
            candidates.append((affix.name, affix.brush_type.value, _normalize_brush_ocr_text(alias)))

    matched: list[tuple[str, str, float]] = []
    for line in normalized_lines:
        ranked = sorted(
            (
                (_brush_ocr_similarity(line, alias), name, brush_type)
                for name, brush_type, alias in candidates
                if alias
            ),
            reverse=True,
        )
        if not ranked:
            continue
        score, name, brush_type = ranked[0]
        if score < 0.66:
            continue
        existing_index = next((index for index, item in enumerate(matched) if item[0] == name), None)
        if existing_index is None:
            matched.append((name, brush_type, score))
        elif score > matched[existing_index][2]:
            matched[existing_index] = (name, brush_type, score)

    if len(matched) < 2:
        return (), {}
    for index, first in enumerate(matched):
        for second in matched[index + 1 :]:
            if first[1] == second[1]:
                options = (first[0], second[0])
                return options, {first[0]: first[2], second[0]: second[2]}
    return (), {}


def _normalize_brush_ocr_text(value: str) -> str:
    return "".join(character for character in value if character.isalnum())


def _brush_ocr_similarity(observed: str, expected: str) -> float:
    score = SequenceMatcher(None, observed, expected).ratio()
    observed_digits = re.findall(r"\d+", observed)
    expected_digits = re.findall(r"\d+", expected)
    if observed_digits and expected_digits and observed_digits != expected_digits:
        score -= 0.22
    if observed in expected or expected in observed:
        score = max(score, min(len(observed), len(expected)) / max(len(observed), len(expected)))
    return float(np.clip(score, 0.0, 1.0))


@lru_cache(maxsize=1)
def _brush_stage2_raster_templates() -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for name, filename in BRUSH_STAGE2_TEMPLATE_FILES.items():
        template = cv2.imread(str(BRUSH_STAGE2_TEMPLATE_DIR / filename), cv2.IMREAD_GRAYSCALE)
        if template is not None and template.size:
            templates[name] = template
    return templates


def _brush_stage2_text_mask(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros(crop.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    white = (value > 150) & (saturation < 155)
    cyan = (hue >= 82) & (hue <= 112) & (saturation >= 65) & (value > 135)
    mask = np.where(white | cyan, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))


def _aligned_binary_template_score(target: np.ndarray, template: np.ndarray) -> float:
    if target.size == 0 or template.size == 0 or target.shape != template.shape:
        return 0.0
    pad = 3
    padded = cv2.copyMakeBorder(target, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    result = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(result)
    return float(max(0.0, score))


#: 背包面板「自证存在」的三条阈值。
#:
#: BACKPACK 区域不是检测出来的，是按棋盘尺寸推出的相对矩形（右侧面板中部）。
#: 没开背包时这块正好压在「选择笔触方向」的两张宽卡片上，`_calibrated_backpack_slots`
#: 会照样切出 3×5 个 88px 方框，于是凭空产出 15 枚「怪物 Lv3」——实机表现为
#: 「背包里根本不是这些颜料」。
#:
#: 阈值来自实测（6 张真背包截图 vs 笔触选择面板误判）：
#:   真背包   V<85 占比 0.316~0.357，15 格指纹两两距离均值 0.555~0.674
#:   笔触卡片 0.135 / 0.265
#: 取偏严的一侧：漏判只丢一次背包读数（界面会显示「本帧未看到背包」），
#: 误判却会把假颜料喂进估值与录制。
BACKPACK_PANEL_MIN_DARK85 = 0.26
BACKPACK_PANEL_MIN_FP_MEAN = 0.40
BACKPACK_PANEL_MIN_SLOTS = 5


def backpack_panel_evidence(
    image: np.ndarray,
    panel: Rect,
    slots: tuple[Rect, ...],
) -> dict[str, float]:
    """量这块矩形像不像真的背包面板。

    ``dark85``：背包格是深色底，面板内应有大比例暗像素。
    ``fp_mean``：真背包 15 格装的是不同颜料，指纹两两距离大；笔触卡片被切成的
    15 个假格是同一块连续画面的切片，彼此高度相似。
    """
    crop = image[panel.y : panel.bottom, panel.x : panel.right]
    if crop.size == 0:
        return {"dark85": 0.0, "fp_mean": 0.0, "slots": 0.0}
    value = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
    dark85 = float(np.mean(value < 85))
    fingerprints = [
        _backpack_slot_fingerprint(image[slot.y : slot.bottom, slot.x : slot.right])
        for slot in slots
    ]
    distances = [
        _backpack_fingerprint_distance(fingerprints[i], fingerprints[j])
        for i in range(len(fingerprints))
        for j in range(i + 1, len(fingerprints))
    ]
    return {
        "dark85": dark85,
        "fp_mean": float(np.mean(distances)) if distances else 0.0,
        "slots": float(len(slots)),
    }


def backpack_panel_present(image: np.ndarray, panel: Rect, slots: tuple[Rect, ...]) -> bool:
    evidence = backpack_panel_evidence(image, panel, slots)
    return (
        evidence["slots"] >= BACKPACK_PANEL_MIN_SLOTS
        and evidence["dark85"] >= BACKPACK_PANEL_MIN_DARK85
        and evidence["fp_mean"] >= BACKPACK_PANEL_MIN_FP_MEAN
    )


def _analyze_backpack(
    image: np.ndarray,
    regions: tuple[RegionObservation, ...],
) -> BackpackObservation | None:
    region = next((item for item in regions if item.kind == RegionKind.BACKPACK), None)
    if region is None:
        return None

    panel = region.rect
    slots = _locate_backpack_slots(image, panel)
    if not backpack_panel_present(image, panel, slots):
        return None
    _reset_family_sync_budget()
    paints = tuple(
        observation
        for slot in slots
        if (observation := _analyze_backpack_slot(image, slot)) is not None
    )
    if not paints:
        return None
    confidence = float(np.mean([paint.category_confidence * 0.55 + paint.level_confidence * 0.45 for paint in paints]))
    return BackpackObservation(panel_rect=panel, paints=paints, confidence=float(np.clip(confidence, 0.0, 1.0)))


def _analyze_backpack_fast(
    image: np.ndarray,
    regions: tuple[RegionObservation, ...],
    previous: BackpackObservation | None = None,
) -> BackpackObservation | None:
    region = next((item for item in regions if item.kind == RegionKind.BACKPACK), None)
    if region is None:
        return None

    panel = region.rect
    slots = _locate_backpack_slots_fast(image, panel, previous)
    if not backpack_panel_present(image, panel, slots):
        return None
    _reset_family_sync_budget()
    previous_by_rect = {
        (paint.slot_rect.x, paint.slot_rect.y, paint.slot_rect.w, paint.slot_rect.h): paint
        for paint in (previous.paints if previous is not None else ())
    }
    paints: list[BackpackPaintObservation] = []
    for slot in slots:
        crop = image[slot.y : slot.bottom, slot.x : slot.right]
        if crop.size == 0:
            continue
        fingerprint = _backpack_slot_fingerprint(crop)
        cached = previous_by_rect.get((slot.x, slot.y, slot.w, slot.h))
        unchanged = (
            cached is not None
            and _backpack_fingerprint_distance(cached.fingerprint, fingerprint)
            <= FAMILY_MATCH_DISTANCE
        )
        # 内容没变、家族也已认出来的话整格沿用上一帧。
        # 但家族可能是「后台刚认出来」的：这时要补算一次，否则空框永远不会变成图案。
        if unchanged and not (cached.subtype is None and _family_lookup(fingerprint) is not None):
            is_selected, selection_confidence = _detect_selected_backpack_slot(crop)
            paints.append(
                replace(
                    cached,
                    slot_rect=slot,
                    fingerprint=fingerprint,
                    is_selected=is_selected,
                    selection_confidence=selection_confidence,
                )
            )
            continue
        if unchanged:
            observation = _analyze_backpack_slot_fast(
                image,
                slot,
                assume_visible=True,
                fingerprint=fingerprint,
                identify_family=True,
            )
            if observation is not None:
                paints.append(observation)
            continue
        # 长得一样的格子就是同一款颜料（同家族、同等级）：认一次就够。
        # 指纹是 8×8 的 HSV 量化，同款两格实测距离 0.06~0.16，不同款都在 0.25 以上。
        # 背包里「3 枚存活 Lv1」「4 枚狩猎之神 Lv1」这种重复很常见。
        twin = next(
            (
                item
                for item in paints
                if item.subtype is not None
                and _backpack_fingerprint_distance(item.fingerprint, fingerprint)
                <= FAMILY_MATCH_DISTANCE
            ),
            None,
        )
        if twin is not None:
            is_selected, selection_confidence = _detect_selected_backpack_slot(crop)
            paints.append(
                replace(
                    twin,
                    slot_rect=slot,
                    fingerprint=fingerprint,
                    is_selected=is_selected,
                    selection_confidence=selection_confidence,
                )
            )
            continue
        observation = _analyze_backpack_slot_fast(
            image,
            slot,
            assume_visible=True,
            fingerprint=fingerprint,
            identify_family=True,
        )
        if observation is not None:
            paints.append(observation)
    if paints:
        confidence = float(np.mean([paint.category_confidence * 0.55 + paint.level_confidence * 0.45 for paint in paints]))
        return BackpackObservation(panel_rect=panel, paints=tuple(paints), confidence=float(np.clip(confidence, 0.0, 1.0)))
    return None


def _locate_backpack_slots_fast(
    image: np.ndarray,
    panel: Rect,
    previous: BackpackObservation | None = None,
) -> tuple[Rect, ...]:
    # 上一帧的格子坐标是**绝对坐标**，用之前必须按面板的位移/缩放映射过来。
    # 直接用旧坐标会出事：面板整体挪 2 像素，裁出来的就是隔壁的边，指纹跟着全变，
    # 缓存命中率从 99.9% 掉到个位数（实机观测里「同一 slot 矩形」的跨帧指纹距离
    # mean=0.446，就是这么来的）。
    if previous is not None and previous.paints:
        base = previous.panel_rect
        if base.w > 0 and base.h > 0:
            scale_x = panel.w / float(base.w)
            scale_y = panel.h / float(base.h)
            mapped = tuple(
                _bounded_rect(
                    panel.x + (paint.slot_rect.x - base.x) * scale_x,
                    panel.y + (paint.slot_rect.y - base.y) * scale_y,
                    paint.slot_rect.w * scale_x,
                    paint.slot_rect.h * scale_y,
                    image.shape[1],
                    image.shape[0],
                )
                for paint in previous.paints
            )
        else:
            mapped = tuple(
                _bounded_rect(paint.slot_rect.x, paint.slot_rect.y, paint.slot_rect.w, paint.slot_rect.h, image.shape[1], image.shape[0])
                for paint in previous.paints
            )
        visible_slots = tuple(slot for slot in mapped if _slot_has_icon(image, slot))
        if visible_slots:
            return visible_slots
    calibrated = _calibrated_backpack_slots(image, panel)
    if len(calibrated) >= 5:
        return calibrated
    return _locate_backpack_slots(image, panel)


def _calibrated_backpack_slots(image: np.ndarray, panel: Rect) -> tuple[Rect, ...]:
    slot = max(42, int(round(panel.h * 0.36)))
    pitch = max(slot + 2, int(round(panel.h * 0.375)))
    x0 = int(round(panel.w * 0.12))
    candidates = tuple(
        _bounded_rect(
            panel.x + x0 + col * pitch,
            panel.y + row * pitch,
            slot,
            slot,
            image.shape[1],
            image.shape[0],
        )
        for row in range(3)
        for col in range(5)
    )
    return tuple(candidate for candidate in candidates if _slot_has_icon(image, candidate))


def _analyze_backpack_slot_fast(
    image: np.ndarray,
    slot: Rect,
    *,
    assume_visible: bool = False,
    fingerprint: str | None = None,
    identify_family: bool = True,
) -> BackpackPaintObservation | None:
    crop = image[slot.y : slot.bottom, slot.x : slot.right]
    if crop.size == 0 or (not assume_visible and not _slot_has_icon(image, slot)):
        return None

    if identify_family:
        subtype, family_category, family_confidence = _identify_backpack_family(crop)
    else:
        subtype, family_category, family_confidence = None, PaintCategory.UNKNOWN, 0.0
    if subtype is not None:
        # 家族认出来了，类别和等级口径都跟着家族走（框形/框色只用来读等级）。
        category, category_confidence = family_category, family_confidence
    elif identify_family:
        category, category_confidence = _classify_backpack_category(crop)
    else:
        # 这一帧没分到认家族的额度：**先不猜类别**，下一帧补认（指纹没变的会重试）。
        # 框形状那一层每格 2.7ms，而实测 15 格背包清一色判成 ``gameplay 0.64``
        # ——猜了等于没猜，不如空着让界面显示"待识别"。
        category, category_confidence = PaintCategory.UNKNOWN, 0.0
    level, level_confidence = _detect_backpack_level_fast(crop, category)
    is_selected, selection_confidence = _detect_selected_backpack_slot(crop)
    is_treasure, treasure_confidence = _resolve_backpack_treasure(
        crop,
        subtype,
        level,
        category,
        category_confidence,
    )
    if is_treasure:
        subtype = "宝箱"
        category = PaintCategory.GAMEPLAY
        category_confidence = max(category_confidence, treasure_confidence)
    return BackpackPaintObservation(
        slot_rect=slot,
        category=category,
        category_confidence=category_confidence,
        level=level,
        level_confidence=level_confidence,
        subtype=subtype,
        is_treasure=is_treasure,
        fingerprint=fingerprint or _backpack_slot_fingerprint(crop),
        is_selected=is_selected,
        selection_confidence=selection_confidence,
    )


def _detect_backpack_level_fast(
    crop: np.ndarray,
    category: PaintCategory = PaintCategory.UNKNOWN,
) -> tuple[int, float]:
    if crop.size == 0:
        return 1, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color_level, color_confidence = _detect_backpack_level_by_frame_color(hsv, category)
    if color_level is not None and color_confidence >= 0.38:
        return color_level, color_confidence
    return 1, 0.42


def _locate_backpack_slots(image: np.ndarray, panel: Rect) -> tuple[Rect, ...]:
    crop = image[panel.y : panel.bottom, panel.x : panel.right]
    if crop.size == 0:
        return ()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect = w / max(1, h)
        if 42 <= w <= 115 and 42 <= h <= 115 and 0.72 <= aspect <= 1.28 and area > 900:
            boxes.append((x, y, w, h))

    if boxes:
        boxes = _dedupe_slot_boxes(boxes)
        grid_boxes = _expand_backpack_slot_grid(boxes, crop.shape[1], crop.shape[0])
    else:
        grid_boxes = _fallback_backpack_slot_grid(crop)

    slots: list[Rect] = []
    for x, y, w, h in grid_boxes:
        rect = _bounded_rect(panel.x + x, panel.y + y, w, h, image.shape[1], image.shape[0])
        if _slot_has_icon(image, rect):
            slots.append(rect)
    return tuple(slots)


def _dedupe_slot_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    boxes = sorted(boxes, key=lambda item: item[2] * item[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for box in boxes:
        rect = Rect(box[0], box[1], box[2], box[3])
        if all(_iou(rect, Rect(other[0], other[1], other[2], other[3])) < 0.55 for other in kept):
            kept.append(box)
    return sorted(kept, key=lambda item: (item[1], item[0]))


def _expand_backpack_slot_grid(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    sizes = [max(w, h) for _, _, w, h in boxes]
    slot = int(np.median(sizes)) if sizes else 80
    centers_x = sorted(x + w / 2 for x, _, w, _ in boxes)
    centers_y = sorted(y + h / 2 for _, y, _, h in boxes)
    pitch_x = _median_positive_delta(centers_x, default=slot + 5)
    pitch_y = _median_positive_delta(centers_y, default=slot + 7)
    min_x = min(x for x, _, _, _ in boxes)
    min_y = min(y for _, y, _, _ in boxes)

    # The visible backpack page is a regular grid. Some colorful borders are
    # hard to contour, so infer rows/columns from detected slots, but do not
    # invent a leading column outside the detected panel; that can crop the
    # board/background and turn it into fake backpack paints.
    while min_x - pitch_x >= slot * 0.35:
        min_x -= pitch_x
    while min_y - pitch_y >= -slot * 0.55:
        min_y -= pitch_y

    result: list[tuple[int, int, int, int]] = []
    y = min_y
    while y <= height - slot * 0.40:
        x = min_x
        while x <= width - slot * 0.40:
            result.append((int(round(x)), int(round(y)), slot, slot))
            x += pitch_x
        y += pitch_y
    return result


def _fallback_backpack_slot_grid(crop: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = crop.shape[:2]
    slot = max(54, min(88, int(h * 0.34)))
    gap = max(4, int(slot * 0.08))
    result: list[tuple[int, int, int, int]] = []
    for y in range(0, max(1, h - slot + 1), slot + gap):
        for x in range(0, max(1, w - slot + 1), slot + gap):
            result.append((x, y, slot, slot))
    return result


def _median_positive_delta(values: list[float], default: float) -> float:
    if len(values) < 2:
        return default
    deltas = [
        b - a
        for a, b in zip(values, values[1:])
        if default * 0.55 <= b - a <= default * 1.60
    ]
    if not deltas:
        return default
    return float(np.median(deltas))


def _slot_has_icon(image: np.ndarray, slot: Rect) -> bool:
    crop = image[slot.y : slot.bottom, slot.x : slot.right]
    if crop.size == 0:
        return False
    # 下面五项全是**比例**（占比、方差），跟分辨率无关，没必要在 94px 上算：
    # 缩到 56px 后 Canny 和五个 mean 便宜三倍，判据值反而更稳
    # （边缘占比 0.17~0.22 → 0.22~0.28，阈值不用动）。背包一帧 15 格，
    # 省下来的是每帧实打实的几十毫秒。
    longest = max(crop.shape[:2])
    if longest > 56:
        scale = 56.0 / longest
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    yy, xx = np.indices(gray.shape)
    border = (
        (yy < gray.shape[0] * 0.18)
        | (yy > gray.shape[0] * 0.82)
        | (xx < gray.shape[1] * 0.18)
        | (xx > gray.shape[1] * 0.82)
    )
    icon_ratio = float(np.mean((s > 45) & (v > 55)))
    bright_ratio = float(np.mean(v > 95))
    edge_ratio = float(np.mean(edges > 0))
    border_edge_ratio = float(np.mean((edges > 0) & border))
    color_std = float(np.std(crop.reshape(-1, 3), axis=0).mean())
    return (
        icon_ratio > 0.10
        and bright_ratio > 0.12
        and edge_ratio > 0.055
        and border_edge_ratio > 0.025
        and color_std > 24.0
    )


def _family_category(family: str | None) -> PaintCategory:
    """家族名 → 颜料大类。不在官方目录里返回 ``UNKNOWN``。

    **类别由家族决定，不由框形状猜**：官方目录里每个家族只属于一类
    （``ColorMonster*`` 是怪物、``ColorPlay*`` 是玩法），而框形状判据在 90px
    的小格上很不稳——实测那张 15 格背包里，4 个圆形框（征战之神）全被判成
    菱形，15 格清一色 ``gameplay 0.64``。
    """
    known = paint_kind.category_of(family)
    if known == "monster":
        return PaintCategory.MONSTER
    if known == "gameplay":
        return PaintCategory.GAMEPLAY
    return PaintCategory.UNKNOWN


#: 家族识别结果的**跨帧缓存**：内容指纹 → ``(家族, 类别, 置信度)``。
#:
#: 背包里同一款颜料会反复出现在同一格（换列、换页、重开面板），每次重认要花
#: 40~60ms 的 ORB 匹配。指纹是 8×8 的 HSV 量化，实测「同一款颜料」距离 ≤0.16、
#: 「不同款」≥0.25（见 ``test_backpack_family``）。
#:
#: **必须按距离查、不能按字符串精确查**：捕获到的画面每帧都会有 1~3 像素的位移
#: 和亮度微变（组合器缩放、格内动画），同一格的指纹几乎从不逐字节相同——实机
#: ``observations.jsonl`` 里 3156 对「同一个 slot 矩形」的相邻读数，精确相同只有
#: 68.6%，但距离 ≤0.14 的占 **99.9%**。原先用 ``fingerprint in _FAMILY_CACHE``
#: 精确查，等于每帧都判「没认出」，后台认出来的结果永远接不上：3402 次读数里
#: ``subtype`` 非空 **0 次**，界面上就只剩等级框。
#:
#: 只缓存认出来的结果：认不出就下一轮重试，别让一次失败变成永久结论。
_FAMILY_CACHE: dict[str, tuple[str, "PaintCategory", float]] = {}
_FAMILY_CACHE_ORDER: list[str] = []
FAMILY_CACHE_MAX = 512

#: 指纹距离上限：同一格跨帧、同一款颜料之间的最大距离。
#: 实测「同一 slot 矩形」相邻读数 99.9% ≤0.14，「不同款」≥0.25，取 0.14 两边都留余量。
FAMILY_MATCH_DISTANCE = 0.14

#: 最近邻查找用的指纹矩阵（惰性重建）。逐格线性扫 512 条 hex 字符串太慢，
#: 转成 ``(N, 64) uint8`` 之后一次比较就是微秒级。
_FAMILY_INDEX: "np.ndarray | None" = None
_FAMILY_INDEX_KEYS: list[str] = []

#: 每帧「同步认一格」的额度。后台线程是主路，这条只是保险：万一 ``family-id``
#: 线程在某个环境里安静地死了（构造图标库失败、ORB 抛异常……），原先的表现是
#: **永远**认不出来。留一格额度，最坏情况也能几秒内补完一页 15 格。
FAMILY_SYNC_PER_FRAME = 1
#: 队列积压到这个量才动同步通道：正常情况后台 22 格/秒足够，没必要抢帧预算。
FAMILY_SYNC_PENDING_TRIGGER = 24
_FAMILY_SYNC_LEFT = 0

#: 后台线程最近几次异常。``except Exception: pass`` 会让「线程一直失败」
#: 伪装成「没认出来」，出问题时至少能从这里看到真因。
_FAMILY_WORKER_ERRORS: list[str] = []
FAMILY_ERROR_LOG_MAX = 8

#: 待认的格子队列。**认家族永远不在帧里做**：一格 40~60ms（27 个家族逐个 ORB
#: 匹配），15 格就是 0.9 秒，比增量帧 0.2 秒的整帧预算还长——分帧认也就变成
#: 「面板挂五六帧空框」。改成丢给后台线程后，帧内只查一次字典（微秒级），
#: 认出来的那一刻下一次取场景就有图案。
_FAMILY_QUEUE: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue()
_FAMILY_PENDING: set[str] = set()
_FAMILY_PENDING_LOCK = threading.Lock()
#: 队列上限。画面在快速变化（动画、翻页）时别把队列堆成内存泄漏。
FAMILY_QUEUE_MAX = 96
_FAMILY_WORKER: threading.Thread | None = None
_FAMILY_WORKER_GALLERY: "paint_icon.IconGallery | None" = None


def reset_family_cache() -> None:
    """清空家族缓存与待认队列（换游戏、换窗口尺寸、跑测试时用）。"""

    global _FAMILY_INDEX
    _FAMILY_CACHE.clear()
    _FAMILY_CACHE_ORDER.clear()
    _FAMILY_INDEX = None
    _FAMILY_INDEX_KEYS.clear()
    with _FAMILY_PENDING_LOCK:
        _FAMILY_PENDING.clear()


def _remember_family(fingerprint: str, family: str, category: "PaintCategory", confidence: float) -> None:
    if not fingerprint:
        return
    global _FAMILY_INDEX
    if fingerprint not in _FAMILY_CACHE:
        _FAMILY_CACHE_ORDER.append(fingerprint)
        # 追加而不是整表重建：后台线程每认出一格就会走到这里，而认一格只要 45ms，
        # 重建一次 512 行的矩阵（每个键都要 fromhex 一遍）反而比识别本身还贵，
        # 实测把一帧从 0.4s 顶到 1.6s（process_time 2.5s，多出来的都在重建上）。
        if _FAMILY_INDEX is not None:
            vector = _fingerprint_vector(fingerprint)
            if vector is not None and vector.shape[0] == _FAMILY_INDEX.shape[1]:
                _FAMILY_INDEX = np.vstack([_FAMILY_INDEX, vector[None, :]])
                _FAMILY_INDEX_KEYS.append(fingerprint)
            else:
                _FAMILY_INDEX = None
        while len(_FAMILY_CACHE_ORDER) > FAMILY_CACHE_MAX:
            _FAMILY_CACHE.pop(_FAMILY_CACHE_ORDER.pop(0), None)
            # 淘汰会让矩阵的行序跟键表错开，只能整表重建（很少发生）。
            _FAMILY_INDEX = None
    _FAMILY_CACHE[fingerprint] = (family, category, confidence)


def _family_cache_size() -> int:
    return len(_FAMILY_CACHE)


def _fingerprint_vector(fingerprint: str) -> "np.ndarray | None":
    try:
        raw = bytes.fromhex(fingerprint)
    except ValueError:
        return None
    if not raw:
        return None
    return np.frombuffer(raw, dtype=np.uint8).copy()


def _family_index() -> "np.ndarray | None":
    """把缓存里的指纹摊成 ``(N, 64)`` 矩阵，供最近邻查询。"""

    global _FAMILY_INDEX, _FAMILY_INDEX_KEYS
    if _FAMILY_INDEX is not None:
        return _FAMILY_INDEX
    vectors: list[np.ndarray] = []
    keys: list[str] = []
    for key in _FAMILY_CACHE:
        vector = _fingerprint_vector(key)
        if vector is None:
            continue
        vectors.append(vector)
        keys.append(key)
    if not vectors:
        _FAMILY_INDEX_KEYS = []
        return None
    _FAMILY_INDEX_KEYS = keys
    _FAMILY_INDEX = np.stack(vectors).astype(np.uint8)
    return _FAMILY_INDEX


def _family_lookup(fingerprint: str) -> "tuple[str, PaintCategory, float] | None":
    """按**距离**查缓存，不是按字符串相等。

    精确命中先走（同一张图重复取时占多数，零成本）；不中再拿最近邻比一次。
    这一步是整条链路的关键：帧间总有 1~3 像素的位移，指纹几乎从不逐字节相同，
    精确查等于每帧都判「没认出」。
    """

    if not fingerprint:
        return None
    cached = _FAMILY_CACHE.get(fingerprint)
    if cached is not None:
        return cached
    matrix = _family_index()
    if matrix is None:
        return None
    query = _fingerprint_vector(fingerprint)
    if query is None or query.shape[0] != matrix.shape[1]:
        return None
    distances = np.count_nonzero(matrix != query, axis=1) / float(matrix.shape[1])
    best = int(np.argmin(distances))
    if distances[best] > FAMILY_MATCH_DISTANCE:
        return None
    return _FAMILY_CACHE.get(_FAMILY_INDEX_KEYS[best])


def family_worker_status() -> dict[str, object]:
    """后台认家族线程的状态（诊断用：异常计数 + 最近一条异常 + 缓存规模）。"""

    with _FAMILY_PENDING_LOCK:
        pending = len(_FAMILY_PENDING)
    return {
        "cache": len(_FAMILY_CACHE),
        "pending": pending,
        "queued": _FAMILY_QUEUE.qsize(),
        "worker_alive": bool(_FAMILY_WORKER is not None and _FAMILY_WORKER.is_alive()),
        "errors": list(_FAMILY_WORKER_ERRORS),
    }


def _record_family_error(text: str) -> None:
    _FAMILY_WORKER_ERRORS.append(text)
    while len(_FAMILY_WORKER_ERRORS) > FAMILY_ERROR_LOG_MAX:
        _FAMILY_WORKER_ERRORS.pop(0)


def _worker_gallery() -> "paint_icon.IconGallery":
    """后台线程专用的图标库实例。

    **必须和主线程分开**：``IconGallery`` 里的 ORB 和 BFMatcher 都有内部状态，
    两个线程同时用同一个实例会拿到脏结果（而且主线程正在拿它认棋盘格）。
    多一个实例的代价只是描述子数组，从缓存的 npz 载入不到 10ms。
    """

    global _FAMILY_WORKER_GALLERY
    if _FAMILY_WORKER_GALLERY is None:
        _FAMILY_WORKER_GALLERY = paint_icon.IconGallery()
    return _FAMILY_WORKER_GALLERY


def _lower_current_thread_priority() -> None:
    """把「认家族」线程降到 ``THREAD_PRIORITY_BELOW_NORMAL``。

    认完 15 格要跑 0.9 秒的 ORB，而交互帧也在算 OpenCV。同进程两个线程抢同一个
    物理核时，帧延迟会被顶高——实测 11 张实机图里有 6 张因此超过 0.75 秒的
    首帧预算（降到低优先级后回到 0.4~0.7 秒）。识别晚几十毫秒没人看得出来，
    帧掉一下是能感觉到的。
    """

    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenThread(0x0020, False, kernel32.GetCurrentThreadId())
        if handle:
            kernel32.SetThreadPriority(handle, -1)
            kernel32.CloseHandle(handle)
    except Exception:
        pass


def _family_worker_loop() -> None:
    _lower_current_thread_priority()
    while True:
        fingerprint, crop = _FAMILY_QUEUE.get()
        try:
            if _family_lookup(fingerprint) is None:
                match = _worker_gallery().identify(crop, center_ratio=paint_frame.ICON_SCALE)
                if match is not None:
                    category = _family_category(match.family)
                    if category != PaintCategory.UNKNOWN:
                        _remember_family(fingerprint, match.family, category, ICON_FALLBACK_CONFIDENCE)
        except Exception as exc:  # 后台线程不能因为一张图崩掉就安静地死掉
            _record_family_error(f"{type(exc).__name__}: {exc}")
        finally:
            with _FAMILY_PENDING_LOCK:
                _FAMILY_PENDING.discard(fingerprint)
            _FAMILY_QUEUE.task_done()


def _ensure_family_worker() -> None:
    global _FAMILY_WORKER
    if _FAMILY_WORKER is not None and _FAMILY_WORKER.is_alive():
        return
    _FAMILY_WORKER = threading.Thread(target=_family_worker_loop, name="family-id", daemon=True)
    _FAMILY_WORKER.start()


def wait_for_family_identification(timeout: float = 10.0) -> bool:
    """等后台把队列认完（测试与「点开背包立刻要结果」的场景用）。"""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _FAMILY_PENDING_LOCK:
            pending = len(_FAMILY_PENDING)
        if pending == 0:
            return True
        time.sleep(0.02)
    return False


def _identify_backpack_family(
    crop: np.ndarray,
) -> tuple[str | None, PaintCategory, float]:
    """认背包格里的家族，返回 ``(家族, 类别, 置信度)``。

    认出来的走缓存直接给；认不出的**丢给后台线程**，本次返回「未知」，
    下一次取场景时就有结果了。

    为什么是「官方图标」而不是「模板库」：模板库（``paint_kind``）是 NCC，
    要求同一块图案、同一个取景比例；它的模板采自**棋盘格**，而背包格只有 90 来
    像素、外面套着品质框，取景比例对不上——实测 15 格只认出 7 格。官方图标
    （``paint_icon``）用 ORB 比局部纹理，按官方 ``ICON_SCALE`` 裁掉外框后
    15 格全中（good 55~93、领先亚军 5.4~11 倍）。数字见
    :func:`paint_icon._center_crop`。
    """
    if crop.size == 0:
        return None, PaintCategory.UNKNOWN, 0.0
    fingerprint = _backpack_slot_fingerprint(crop)
    cached = _family_lookup_for_crop(crop, fingerprint)
    if cached is not None:
        return cached
    # 保险通道：后台线程万一在这个环境里认不动，攒下来的额度每帧补认一格。
    # 只在「这一格从没认过」时消耗，认出来的结果照样进缓存，后面所有帧直接命中。
    if _take_family_sync_budget():
        identified = _identify_family_synchronously(crop)
        if identified is not None:
            family, category = identified
            _remember_family(fingerprint, family, category, ICON_FALLBACK_CONFIDENCE)
            return _FAMILY_CACHE[fingerprint]
    with _FAMILY_PENDING_LOCK:
        already_queued = fingerprint in _FAMILY_PENDING
        if not already_queued and len(_FAMILY_PENDING) < FAMILY_QUEUE_MAX:
            _FAMILY_PENDING.add(fingerprint)
        else:
            already_queued = True
    if not already_queued:
        _ensure_family_worker()
        # crop 是调用方图像的视图，入队必须拷贝：那一帧之后就会被回收/复写。
        _FAMILY_QUEUE.put((fingerprint, crop.copy()))
    return None, PaintCategory.UNKNOWN, 0.0


def _family_lookup_for_crop(
    crop: np.ndarray,
    fingerprint: str,
) -> "tuple[str, PaintCategory, float] | None":
    """查缓存。

    这里刻意**不做「挪一像素再试」的多变体查询**：实机 ``observations.jsonl``
    回放显示，只要 slot 坐标跟着面板一起平移（``_locate_backpack_slots_fast``
    已经按面板位移映射上一帧的格子），同一格的模糊命中率就是 99.2%，
    多变体只多出 8 倍指纹计算和最近邻扫描，收益不到一格。
    """

    return _family_lookup(fingerprint)


def _take_family_sync_budget() -> bool:
    """要不要在这一帧里直接认一格（抢 45ms 帧预算）。

    只有后台通道明显不正常时才放行——否则等于把识别偷偷搬回帧里，
    首帧就会因为认一格而变慢，而首帧不阻塞正是 ``test_backpack_family`` 锁住的语义。
    """

    global _FAMILY_SYNC_LEFT
    if _FAMILY_SYNC_LEFT <= 0:
        return False
    if not _family_sync_needed():
        return False
    _FAMILY_SYNC_LEFT -= 1
    return True


def _family_sync_needed() -> bool:
    if _FAMILY_WORKER_ERRORS:
        return True
    if _FAMILY_WORKER is not None and not _FAMILY_WORKER.is_alive():
        return True
    with _FAMILY_PENDING_LOCK:
        pending = len(_FAMILY_PENDING)
    return pending >= FAMILY_SYNC_PENDING_TRIGGER


def _reset_family_sync_budget() -> None:
    global _FAMILY_SYNC_LEFT
    _FAMILY_SYNC_LEFT = FAMILY_SYNC_PER_FRAME


def _identify_family_synchronously(crop: np.ndarray) -> "tuple[str, PaintCategory] | None":
    """帧内直接认一格（40~60ms），只在后台通道失灵时兜底用。"""

    try:
        match = paint_icon.gallery().identify(crop, center_ratio=paint_frame.ICON_SCALE)
    except Exception as exc:
        _record_family_error(f"sync {type(exc).__name__}: {exc}")
        return None
    if match is None:
        return None
    category = _family_category(match.family)
    if category == PaintCategory.UNKNOWN:
        return None
    return match.family, category



def _analyze_backpack_slot(
    image: np.ndarray,
    slot: Rect,
    *,
    identify_family: bool = True,
) -> BackpackPaintObservation | None:
    crop = image[slot.y : slot.bottom, slot.x : slot.right]
    if crop.size == 0 or not _slot_has_icon(image, slot):
        return None

    if identify_family:
        subtype, family_category, family_confidence = _identify_backpack_family(crop)
    else:
        subtype, family_category, family_confidence = None, PaintCategory.UNKNOWN, 0.0
    if subtype is not None:
        # 家族认出来了，类别和等级口径都跟着家族走（框形/框色只用来读等级）。
        category, category_confidence = family_category, family_confidence
    else:
        category, category_confidence = _classify_backpack_category(crop)
    level, level_confidence = _detect_backpack_level(crop, category)
    is_selected, selection_confidence = _detect_selected_backpack_slot(crop)
    is_treasure, treasure_confidence = _resolve_backpack_treasure(
        crop,
        subtype,
        level,
        category,
        category_confidence,
    )
    if is_treasure:
        subtype = "宝箱"
        category = PaintCategory.GAMEPLAY
        category_confidence = max(category_confidence, treasure_confidence)
    fingerprint = _backpack_slot_fingerprint(crop)
    return BackpackPaintObservation(
        slot_rect=slot,
        category=category,
        category_confidence=category_confidence,
        level=level,
        level_confidence=level_confidence,
        subtype=subtype,
        is_treasure=is_treasure,
        fingerprint=fingerprint,
        is_selected=is_selected,
        selection_confidence=selection_confidence,
    )


def _resolve_backpack_treasure(
    crop: np.ndarray,
    subtype: str | None,
    level: int | None,
    category: PaintCategory,
    category_confidence: float,
) -> tuple[bool, float]:
    """这一格是不是宝箱。

    家族认出来就只信家族（宝箱在官方目录里就是一个正常家族 ``ColorPlay1``）；
    认不出来才回头看颜色，而且必须过棋盘那条**同样的闸**
    （:func:`_color_treasure_allowed`）：Lv5 橙框本身就有 0.23 的金色占比、
    等级读不出来、已判成怪物——三种一律不采信。

    以前这里直接用 ``_detect_treasure_icon`` 的原始结论，没有闸也没有家族优先，
    实机上「Lv5 橙框的空格」和「金色家族」都会被翻成宝箱。
    """
    if subtype is not None:
        return subtype == "宝箱", 0.0
    colored, treasure_confidence = _detect_treasure_icon(crop)
    if colored and _color_treasure_allowed(level, category, category_confidence):
        return True, treasure_confidence
    return False, 0.0


def _detect_selected_backpack_slot(crop: np.ndarray) -> tuple[bool, float]:
    if crop.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    height, width = saturation.shape
    border = max(3, int(round(min(height, width) * 0.055)))
    white = (saturation < 55) & (value > 190)
    inset = max(1, border // 2)
    side_ratios = (
        float(np.mean(white[:border, inset : width - inset])),
        float(np.mean(white[height - border :, inset : width - inset])),
        float(np.mean(white[inset : height - inset, :border])),
        float(np.mean(white[inset : height - inset, width - border :])),
    )
    strong_sides = sum(ratio >= 0.28 for ratio in side_ratios)
    score = float(np.mean(sorted(side_ratios, reverse=True)[:3]))
    selected = strong_sides >= 3 and score >= 0.34
    confidence = float(np.clip((score - 0.18) / 0.45, 0.0, 1.0))
    return selected, confidence


def _classify_backpack_category(crop: np.ndarray) -> tuple[PaintCategory, float]:
    ui_category, ui_confidence = _classify_backpack_frame_shape(crop)
    if ui_category != PaintCategory.UNKNOWN:
        return ui_category, ui_confidence

    # Fallback only. Internal art often contains circular highlights, so this
    # must not override the explicit outer-frame UI shape.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, min(gray.shape[:2]) // 2),
        param1=80,
        param2=20,
        minRadius=max(10, min(gray.shape[:2]) // 4),
        maxRadius=max(14, min(gray.shape[:2]) // 2),
    )
    if circles is not None:
        return PaintCategory.MONSTER, 0.58
    return PaintCategory.UNKNOWN, 0.30


def _classify_backpack_frame_shape(crop: np.ndarray) -> tuple[PaintCategory, float]:
    if crop.size == 0:
        return PaintCategory.UNKNOWN, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = gray.shape[:2]
    # Ignore inner artwork. Monster icons frequently contain strong diagonals,
    # while the actual category evidence is the outer circular/diamond frame.
    edges[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)] = 0
    roi = edges[
        int(h * 0.05) : int(h * 0.92),
        int(w * 0.05) : int(w * 0.92),
    ]
    if roi.size == 0:
        return PaintCategory.UNKNOWN, 0.0

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=12,
        minLineLength=max(10, int(min(roi.shape[:2]) * 0.18)),
        maxLineGap=5,
    )
    diagonal_length = 0.0
    axial_length = 0.0
    if lines is not None:
        # OpenCV 4 commonly returns (N, 1, 4), while OpenCV 5 wheels may
        # return (N, 4). Normalize both public representations.
        normalized_lines = np.asarray(lines).reshape(-1, 4)
        for line in normalized_lines:
            x1, y1, x2, y2 = line
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            if angle > 90.0:
                angle = 180.0 - angle
            if 32.0 <= angle <= 58.0:
                diagonal_length += length
            elif angle <= 12.0 or angle >= 78.0:
                axial_length += length

    side = max(1.0, float(min(roi.shape[:2])))
    diagonal_score = diagonal_length / side
    axial_score = axial_length / side

    # Green/blue/silver/orange diamond borders are saturated along the diagonals;
    # circular monster frames have stronger vertical/horizontal ring edges.
    if diagonal_score >= 2.5 and diagonal_score >= axial_score * 0.75:
        return PaintCategory.GAMEPLAY, float(np.clip(diagonal_score / 10.0, 0.64, 0.92))
    if axial_score >= 1.8 and axial_score > diagonal_score * 1.50:
        return PaintCategory.MONSTER, float(np.clip(axial_score / 8.0, 0.58, 0.86))
    # 圆形框的边缘在 HoughLinesP 下角度很杂，轴向长度经常凑不到上面那条线
    # （实测 12 个圆形格 axial 最高只有 1.77），于是全掉进 UNKNOWN、被
    # HoughCircles 兜底成固定 0.58——「碰巧对」而不是「认出来」。
    # 但菱形的对角强度是圆形的 2.5 倍以上（实测圆形 ≤1.78、菱形 ≥4.40），
    # 所以反过来用「对角强度低」判圆形更稳。
    if diagonal_score < 2.0:
        return PaintCategory.MONSTER, float(np.clip(0.78 - diagonal_score * 0.08, 0.55, 0.78))

    return PaintCategory.UNKNOWN, 0.0


def _detect_backpack_level(
    crop: np.ndarray,
    category: PaintCategory = PaintCategory.UNKNOWN,
) -> tuple[int, float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    color_level, color_confidence = _detect_backpack_level_by_frame_color(hsv, category)

    # 角标 "LvN" 只能拿来**确认**，不能拿来翻案：
    # ``_detect_level_digit_from_label`` 的模板是 OpenCV 合成字体，而卡片用的是
    # 游戏自己的圆角粗体，实测同一格会读出 7、3 这种明显错值。框色（品质色）现在是
    # 15/15 准的，所以以框色为准，角标只用来把同一结论的置信度抬上去。
    label = crop[int(crop.shape[0] * 0.56) :, int(crop.shape[1] * 0.54) :]
    digit = _detect_level_digit_from_label(label)
    if digit is not None and color_level == digit:
        return digit, max(0.78, color_confidence)
    if color_level is not None:
        return color_level, color_confidence

    border = (
        (np.indices(h.shape)[0] < crop.shape[0] * 0.16)
        | (np.indices(h.shape)[0] > crop.shape[0] * 0.84)
        | (np.indices(h.shape)[1] < crop.shape[1] * 0.16)
        | (np.indices(h.shape)[1] > crop.shape[1] * 0.84)
    ) & (s > 60) & (v > 75)
    scores = {
        2: float(np.mean(border & (h >= 45) & (h <= 85))),
        3: float(np.mean(border & (h >= 90) & (h <= 115))),
        4: float(np.mean(border & (h >= 125) & (h <= 165))),
        5: float(np.mean(border & (h >= 8) & (h <= 28))),
        6: float(np.mean(border & (((h >= 0) & (h <= 8)) | (h >= 170)))),
    }
    best_level, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0.02:
        return best_level, float(np.clip(best_score / 0.12, 0.45, 0.82))
    return 1, 0.55


#: 卡片外框是**纯色品质框**，等级就看它的色相：银白 1 / 绿 2 / 青 3 / 紫 4 / 橙 5 / 红 6。
#:
#: 这套「色相=等级」判据只在**框线本体的像素**上成立。卡片底纹是半透明的深青
#: （h≈95~105、v 只有 60~80），一旦混进环带就会把绿框读成青框——也就是把 Lv2
#: 读成 Lv3。2026-11-17 实拍 15 格（其中 11 格右下角 "LvN" 文字可读，当真值）：
#:
#: - 旧口径（s>50 & v>60，环带 0.050）：矮人绿框 Lv2 读成 **3**（cyan 0.42 > green 0.38）
#: - 新口径（只收 v≥105，彩色要 s≥100、银白要 s<55）：15 格全部自洽，11/11 真值命中
#: - 矮人绿框 green 0.62、cyan 0.00；紫菱形 purple 0.96；青菱形 cyan 0.96；绿菱形 green 0.95
#:
#: 环带同时收窄到 0.038：0.050 会同时套住框线外侧的背景。
FRAME_BRIGHT_V = 105
FRAME_COLOR_S = 100
FRAME_GRAY_S = 55
FRAME_RING_HALF = 0.038
#: 橙/红读数的「成环」判据：主色必须覆盖这么多个角度段才算框线。
#: 12 段取 0.55（≈7 段）：整圈框线通常铺满 10~12 段，图案穿角只有 2~4 段。
RING_ANGULAR_BINS = 12
RING_SPREAD_MIN = 0.55
#: 色相 → 等级。区间左闭右开，单位是 OpenCV 的 0~179。
FRAME_HUE_LEVELS: dict[int, tuple[tuple[int, int], ...]] = {
    2: ((45, 86),),
    3: ((86, 119),),
    4: ((119, 166),),
    5: ((8, 31),),
    6: ((166, 180), (0, 8)),
}


def _ring_angular_spread(
    hsv: np.ndarray,
    ring: np.ndarray,
    level: int,
) -> float:
    """主色像素在这个环带上覆盖了多少个角度段（0~1）。

    用来区分「整圈的框线」和「穿过环带的图案」。机械之神的图案本身就是金橙色，
    菱形环带在四个角上会切进图案里，于是读到橙色 → 被当成 Lv5（实机报过
    「几个 5 级机械之神」）。整圈框线是均匀铺开的，图案穿角只覆盖少数几段。
    """
    if level not in FRAME_HUE_LEVELS:
        return 0.0
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    colored = ring & (v >= FRAME_BRIGHT_V) & (s >= FRAME_COLOR_S)
    hit = np.zeros(h.shape, dtype=bool)
    for low, high in FRAME_HUE_LEVELS[level]:
        hit |= colored & (h >= low) & (h < high)
    ys, xs = np.nonzero(hit)
    if ys.size == 0:
        return 0.0
    height, width = h.shape[:2]
    angles = np.arctan2(ys - (height - 1) / 2.0, xs - (width - 1) / 2.0)
    bins = ((angles + np.pi) / (2 * np.pi) * RING_ANGULAR_BINS).astype(np.int32) % RING_ANGULAR_BINS
    return float(np.unique(bins).size / RING_ANGULAR_BINS)


def _frame_color_reading(
    hsv: np.ndarray,
    ring: np.ndarray,
) -> tuple[int, float]:
    """在给定环带上读「主色 → 等级」，返回 ``(等级, 该色的占比)``。

    色相统计走一次 ``bincount`` 再按区间求和，而不是六个等级各扫一遍全图：
    背包一帧 15 格 × 2 种环带，这是每帧都要付的钱。
    """
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    frame = ring & (v > 45)
    total = int(np.count_nonzero(frame))
    if total <= 0:
        return 0, 0.0
    bright = frame & (v >= FRAME_BRIGHT_V)
    picked = int(np.count_nonzero(bright))
    # 框线本身很细：整圈里收不到一成半的亮像素，说明这一圈根本没压在框上。
    if picked < total * 0.15:
        return 0, 0.0

    gray = int(np.count_nonzero(bright & (s < FRAME_GRAY_S)))
    colored = bright & (s >= FRAME_COLOR_S)
    ratios: dict[int, float] = {1: gray / picked}
    if np.any(colored):
        hist = np.bincount(hsv[:, :, 0][colored].ravel(), minlength=180)
        for level, ranges in FRAME_HUE_LEVELS.items():
            hit = sum(int(hist[low:high].sum()) for low, high in ranges)
            ratios[level] = hit / picked
    else:
        for level in FRAME_HUE_LEVELS:
            ratios[level] = 0.0
    best_level, best_ratio = max(ratios.items(), key=lambda item: item[1])
    # 橙/红最容易被图案本身污染（机械之神、宝箱都是金橙色）。要求它们**成环**：
    # 主色覆盖不到六成角度段，说明那是图案穿进环带，不是框线，直接不采信这套环带。
    if best_level in (5, 6) and best_ratio >= 0.30:
        if _ring_angular_spread(hsv, ring, best_level) < RING_SPREAD_MIN:
            return 0, 0.0
    return best_level, best_ratio


def _detect_backpack_level_by_frame_color(
    hsv: np.ndarray,
    category: PaintCategory = PaintCategory.UNKNOWN,
) -> tuple[int | None, float]:
    """读卡片品质框的颜色定等级。

    **圆形和菱形两套环带都试，取主色占比高的那个**——不按 ``category`` 选框型。
    ``category`` 是从家族反推的，而实拍里就有反例：``jietu/初始和背包宝箱.png``
    15 格全是玩法（``category=gameplay``，该用菱形环带），其中 4 格的框其实是**圆形**，
    按菱形采样会把 Lv1 读成 Lv3（diamond 0.37 的主色是杂色，circle 0.56 才是银白）。
    两种都试时，那张图 circle 15/15、diamond 13/15，取大者正好 15/15；
    本机新截图两种各 15/15，取大者不变。
    """
    if hsv.size == 0:
        return None, 0.0
    height, width = hsv.shape[:2]
    yy, xx = np.indices((height, width))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    xnorm = np.abs(xx - cx) / max(1.0, width * 0.43)
    ynorm = np.abs(yy - cy) / max(1.0, height * 0.43)
    diamond_frame = np.abs(xnorm + ynorm - 1.0) < FRAME_RING_HALF
    circle_radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(1.0, min(width, height) * 0.40)
    circle_frame = np.abs(circle_radius - 1.0) < FRAME_RING_HALF
    # 角标那块的亮字会污染环带，先剔掉
    label_area = (xx > width * 0.56) & (yy > height * 0.58)

    best_level, best_ratio = 0, 0.0
    for ring in (diamond_frame, circle_frame):
        level, ratio = _frame_color_reading(hsv, ring & ~label_area)
        if ratio > best_ratio:
            best_level, best_ratio = level, ratio
    best_level_int = int(best_level)
    if best_level_int <= 0 or best_ratio < 0.30:
        return None, 0.0
    return best_level_int, float(np.clip(best_ratio / 0.75, 0.50, 0.92))


def _detect_level_digit_from_label(label_crop: np.ndarray) -> int | None:
    if label_crop.size == 0:
        return None
    gray = cv2.cvtColor(label_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < binary.shape[0] * 0.22 or w < 3:
            continue
        boxes.append((x, y, w, h))
    for x, y, w, h in sorted(boxes, key=lambda item: item[0] + item[2], reverse=True)[:3]:
        digit = _match_digit_template(binary[y : y + h, x : x + w])
        if digit is not None:
            return digit
    return None


#: 「金色占比」判宝箱的两个阈值。
#:
#: 2026-11-12 用 ``runtime/s8_paint_icons`` 里的官方图标实测（合成到深色底）：
#:
#: ==========  ==========
#: 家族        金色占比
#: ==========  ==========
#: 英灵        0.288
#: 宝箱        0.266
#: 机械之神    0.133
#: 玩偶        0.124
#: 迷城        0.087
#: ==========  ==========
#:
#: 旧的 0.13 正好卡在「机械之神」上面（0.133 > 0.13），会把金色玩法面料判成
#: 宝箱；而怪物「英灵」比真宝箱还金，靠颜色根本无法区分。抬到 0.20 后
#: 机械之神/玩偶/迷城都被排除，真宝箱仍有余量。**这条启发式只信"很金"，
#: 不负责认人**——认家族一律走 :mod:`paint_kind` 与 :mod:`paint_icon`。
TREASURE_GOLD_RATIO = 0.20
TREASURE_CENTER_GOLD = 0.15


def _detect_treasure_icon(crop: np.ndarray) -> tuple[bool, float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    gold_ratio = float(np.mean((h >= 13) & (h <= 38) & (s > 70) & (v > 100)))
    dark_center = crop[crop.shape[0] // 4 : crop.shape[0] * 3 // 4, crop.shape[1] // 4 : crop.shape[1] * 3 // 4]
    if dark_center.size == 0:
        return False, 0.0
    center_hsv = cv2.cvtColor(dark_center, cv2.COLOR_BGR2HSV)
    center_gold = float(np.mean((center_hsv[:, :, 0] >= 13) & (center_hsv[:, :, 0] <= 38) & (center_hsv[:, :, 1] > 70) & (center_hsv[:, :, 2] > 95)))
    confidence = float(np.clip(max(gold_ratio, center_gold) / TREASURE_GOLD_RATIO, 0.0, 0.85))
    return (gold_ratio > TREASURE_GOLD_RATIO and center_gold > TREASURE_CENTER_GOLD), confidence


def _backpack_slot_fingerprint(crop: np.ndarray) -> str:
    small = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    quantized = (hsv[:, :, 0] // 12).astype(np.uint8)
    # ``tobytes().hex()`` 与逐元素 f-string 等价，但快一个数量级——一帧要算几十次。
    return quantized.tobytes().hex()


def _backpack_fingerprint_distance(first: str, second: str) -> float:
    if not first or not second or len(first) != len(second):
        return 1.0
    try:
        left = bytes.fromhex(first)
        right = bytes.fromhex(second)
    except ValueError:
        return 1.0
    if not left or len(left) != len(right):
        return 1.0
    return sum(a != b for a, b in zip(left, right)) / len(left)


def _brush_stage1_crop(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    return crop[: max(1, int(crop.shape[0] * 0.34)), :]


def _brush_stage2_crop(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    return crop[max(0, int(crop.shape[0] * 0.22)) :, :]


def _brush_stage1_scores(crop: np.ndarray) -> dict[str, float]:
    scores = match_keywords(
        crop,
        ("转锋", "转峰", "堆砌", "扩散"),
        font_size_range=range(16, 29, 3),
        scale=1.0,
        include_binary=False,
        threshold=0.0,
    )
    return {
        BrushType.TRANSFORM.value: max(scores.get("转锋", 0.0), scores.get("转峰", 0.0)),
        BrushType.STACK.value: scores.get("堆砌", 0.0),
        BrushType.SPREAD.value: scores.get("扩散", 0.0),
    }


def _brush_card_pattern_scores(crop: np.ndarray) -> dict[str, float]:
    if crop.size == 0:
        return {}
    cards = _split_brush_cards(crop)
    scores = {
        BrushType.TRANSFORM.value: 0.0,
        BrushType.STACK.value: 0.0,
        BrushType.SPREAD.value: 0.0,
    }
    for card in cards:
        card_scores = _brush_pattern_score_single(card)
        for brush_type, score in card_scores.items():
            scores[brush_type] = max(scores[brush_type], score)
    return scores


def _brush_stage1_options_from_cards(crop: np.ndarray) -> tuple[tuple[str, ...], str | None]:
    cards = _split_brush_cards(crop)
    options: list[str] = []
    selected: str | None = None
    for card in cards:
        scores = _brush_pattern_score_single(card)
        if not scores:
            continue
        brush_type, score = max(scores.items(), key=lambda item: item[1])
        if score < 0.54 or brush_type in options:
            continue
        options.append(brush_type)
        if _brush_card_selected_score(card) >= 0.16:
            selected = brush_type
        if len(options) >= 2:
            break

    ordered = tuple(brush_type.value for brush_type in BRUSH_TYPE_ORDER if brush_type.value in options)
    if selected not in ordered:
        selected = None
    return ordered, selected


def _split_brush_cards(crop: np.ndarray) -> list[np.ndarray]:
    if crop.size == 0:
        return []
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    active = (hsv[:, :, 1] > 20) & (hsv[:, :, 2] < 105)
    row_activity = active.mean(axis=1)
    rows = np.where(row_activity > 0.25)[0]
    if len(rows) == 0:
        return [crop]

    segments: list[tuple[int, int]] = []
    start = int(rows[0])
    last = int(rows[0])
    for row in rows[1:]:
        row = int(row)
        if row - last > max(5, h * 0.035):
            if last - start > h * 0.06:
                segments.append((start, last + 1))
            start = row
        last = row
    if last - start > h * 0.06:
        segments.append((start, last + 1))

    cards: list[np.ndarray] = []
    for y0, y1 in segments:
        band = active[y0:y1, :]
        col_activity = band.mean(axis=0) if band.size else np.array([])
        cols = np.where(col_activity > 0.08)[0]
        if len(cols) > 0:
            x0 = max(0, int(cols[0]) - 6)
            x1 = min(w, int(cols[-1]) + 7)
        else:
            x0, x1 = 0, w
        cards.append(crop[max(0, y0 - 4) : min(h, y1 + 4), x0:x1])
    return cards or [crop]


def _brush_pattern_score_single(card: np.ndarray) -> dict[str, float]:
    h, w = card.shape[:2]
    if h < 12 or w < 32:
        return {}
    left = card[:, : max(1, int(w * 0.24))]
    hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat > 35) & (val > 35)
    if not np.any(mask):
        return {}

    hs = hue[mask]
    vs = val[mask]
    cyan = float(np.mean((hs >= 88) & (hs <= 101)))
    bright_blue = float(np.mean((hs >= 100) & (hs <= 108) & (vs > 130)))
    deep_blue = float(np.mean((hs >= 108) & (hs <= 122)))
    dim_blue = float(np.mean(vs < 115))
    median_hue = float(np.median(hs))

    # Workbook samples show stable families:
    # 转锋: teal/cyan leaf motif, 堆砌: bright blue swirl, 扩散: deeper blue/purple target motif.
    transform_score = _clamp01_vision((cyan - 0.28) / 0.55)
    stack_score = _clamp01_vision((bright_blue - 0.22) / 0.52 + (1.0 - abs(median_hue - 103.0) / 18.0) * 0.22)
    dark_stack_score = _clamp01_vision((dim_blue - 0.52) / 0.34) * _clamp01_vision(
        1.0 - abs(median_hue - 104.0) / 8.0
    )
    if cyan < 0.32 and deep_blue < 0.18:
        stack_score = max(stack_score, dark_stack_score)
    spread_score = _clamp01_vision((deep_blue - 0.32) / 0.60 + dim_blue * 0.18)

    if transform_score < 0.22:
        transform_score = 0.0
    if stack_score < 0.22:
        stack_score = 0.0
    if spread_score < 0.22:
        spread_score = 0.0

    return {
        BrushType.TRANSFORM.value: min(0.74, transform_score),
        BrushType.STACK.value: min(0.72, stack_score),
        BrushType.SPREAD.value: min(0.72, spread_score),
    }


def _brush_card_selected_score(card: np.ndarray) -> float:
    if card.size == 0:
        return 0.0
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    ice = (((h >= 88) & (h <= 112) & (s > 35) & (v > 150)) | ((s < 75) & (v > 200)))
    yy, xx = np.indices(h.shape)
    border = (yy < 9) | (yy > h.shape[0] - 10) | (xx < 9) | (xx > h.shape[1] - 10)
    if not np.any(border):
        return 0.0
    return float(np.mean(ice[border]))


def _clamp01_vision(value: float) -> float:
    return max(0.0, min(1.0, value))


def _detected_stage1_options(stage1_scores: dict[str, float]) -> tuple[str, ...]:
    if not stage1_scores:
        return ()
    top = max(stage1_scores.values(), default=0.0)
    if top < 0.28:
        return ()
    threshold = max(0.28, top * 0.68)
    selected = [
        brush_type.value
        for brush_type in BRUSH_TYPE_ORDER
        if stage1_scores.get(brush_type.value, 0.0) >= threshold
    ]
    if len(selected) <= 2:
        return tuple(selected)
    top_two = sorted(selected, key=lambda value: stage1_scores.get(value, 0.0), reverse=True)[:2]
    return tuple(brush_type.value for brush_type in BRUSH_TYPE_ORDER if brush_type.value in top_two)


def _brush_keyword_scores(crop: np.ndarray, stage1_options: tuple[str, ...] = ()) -> dict[str, float]:
    keywords = _brush_affix_keywords(stage1_options)
    scores = match_keywords(
        crop,
        keywords,
        font_size_range=range(14, 27, 3),
        scale=1.0,
        include_binary=False,
        threshold=0.0,
    )
    return {key: value for key, value in scores.items() if value > 0.12}


def _brush_stage2_scores(
    keyword_scores: dict[str, float],
    stage1_scores: dict[str, float],
    stage1_options: tuple[str, ...] = (),
) -> dict[str, float]:
    scored: dict[str, float] = {}
    allowed_types = set(stage1_options)
    for affix in build_affix_catalog():
        if allowed_types and affix.brush_type.value not in allowed_types:
            continue
        score = max(
            _affix_name_score(alias, keyword_scores)
            for alias in (affix.display_name, *affix.aliases)
        )
        score += min(0.08, stage1_scores.get(affix.brush_type.value, 0.0) * 0.10)
        if score > 0.10:
            scored[affix.display_name] = float(np.clip(score, 0.0, 1.0))
    return scored


def _detected_stage2_options(stage2_scores: dict[str, float]) -> tuple[str, ...]:
    if not stage2_scores:
        return ()
    ranked = sorted(stage2_scores.items(), key=lambda item: item[1], reverse=True)
    top = ranked[0][1]
    if top < 0.24:
        return ()
    threshold = max(0.24, top * 0.74)
    selected = [name for name, score in ranked if score >= threshold][:2]
    return tuple(selected)


def _infer_chosen_stage1(
    stage1_options: tuple[str, ...],
    stage2_options: tuple[str, ...],
) -> str | None:
    if stage2_options:
        by_name = {affix.name: affix for affix in build_affix_catalog()}
        affix = by_name.get(stage2_options[0])
        if affix is not None:
            return affix.brush_type.value
    if len(stage1_options) == 1:
        return stage1_options[0]
    return None


def _affix_name_score(name: str, scores: dict[str, float]) -> float:
    full_score = scores.get(name, 0.0)
    tokens = _affix_tokens(name)
    if not tokens:
        return full_score
    token_scores = [scores.get(token, 0.0) for token in tokens]
    strong = [score for score in token_scores if score >= 0.24]
    average = sum(token_scores) / len(token_scores)
    coverage = len(strong) / len(tokens)
    best = max(token_scores, default=0.0)
    return max(full_score, average * 0.48 + coverage * 0.34 + best * 0.18)


def _affix_tokens(name: str) -> tuple[str, ...]:
    tokens = [keyword for keyword in _brush_affix_token_keywords() if keyword in name]
    if "当前颜料等级随机改变" in name:
        tokens.extend(["当前颜料", "等级随机"])
    if "完成条件所需数量-2" in name:
        tokens.extend(["完成条件", "所需数量", "-2", "调色目标", "进度要求", "减少2"])
    if "调色目标的进度要求减少1" in name:
        tokens.extend(["调色目标", "进度要求", "减少1"])
    if "重新获得" in name:
        tokens.extend(["重新获得", "初始颜料"])
    if "同等级的随机怪物颜料" in name:
        tokens.extend(["转变为", "同等级", "随机怪物颜料"])
    if "可放置于失色地块中" in name:
        tokens.extend(["可放置于", "可放置", "失色地块", "失色地块中"])
    return tuple(dict.fromkeys(tokens))


def _brush_affix_keywords(stage1_options: tuple[str, ...] = ()) -> tuple[str, ...]:
    allowed_types = set(stage1_options)
    tokens: list[str] = []
    for affix in build_affix_catalog():
        if allowed_types and affix.brush_type.value not in allowed_types:
            continue
        for alias in (affix.display_name, *affix.aliases):
            tokens.extend(_affix_tokens(alias))
    return tuple(dict.fromkeys(tokens or _brush_affix_token_keywords()))


def _brush_affix_token_keywords() -> tuple[str, ...]:
    return (
        "作用方向",
        "作用对象",
        "作用范围",
        "改为",
        "横向",
        "纵向",
        "对角斜向",
        "相邻",
        "全图",
        "全图随机",
        "另一类型",
        "转变自身",
        "调色目标",
        "完成条件",
        "所需数量",
        "-2",
        "-1",
        "进度要求",
        "减少1",
        "减少2",
        "当前颜料",
        "等级随机",
        "内容不变",
        "作用方向",
        "内容，等级不变",
        "重新获得",
        "随机调色颜料",
        "初始颜料",
        "随机类型",
        "随机玩法",
        "随机怪物",
        "随机怪物颜料",
        "范围内",
        "随机1个",
        "随机2个",
        "随机1块",
        "距离最近",
        "最近1个",
        "随机增加",
        "调色进度",
        "要求增加2",
        "同等级",
        "转变为",
        "所有颜料",
        "提升1级",
        "提升2级",
        "提升3级",
        "自身提升",
        "全图玩法",
        "全图怪物",
        "自身相同",
        "类型与等级",
        "替换为自身",
        "玩法",
        "怪物",
        "阻挡地块",
        "失色地块",
        "虚空地块",
        "空白地块",
        "1块",
        "1或2块",
        "可放置",
        "可放置于",
        "已放置颜料",
        "失色地块中",
        "失去自身内容",
        "变为空白地块",
        "变成",
    )


def _save_current_paint_crops(image: np.ndarray, scene: SceneObservation | None) -> None:
    if scene is None or scene.current_paint is None:
        return
    paint = scene.current_paint
    output_dir = Path("runtime")
    write_image(output_dir / "current_paint_panel.png", image[paint.panel_rect.y : paint.panel_rect.bottom, paint.panel_rect.x : paint.panel_rect.right])
    write_image(output_dir / "current_paint_icon.png", image[paint.icon_rect.y : paint.icon_rect.bottom, paint.icon_rect.x : paint.icon_rect.right])
    write_image(output_dir / "current_paint_text.png", image[paint.text_rect.y : paint.text_rect.bottom, paint.text_rect.x : paint.text_rect.right])


def _save_brush_options_crops(image: np.ndarray, scene: SceneObservation | None) -> None:
    if scene is None or scene.brush_options is None:
        return
    panel = scene.brush_options.panel_rect
    output_dir = Path("runtime")
    write_image(output_dir / "brush_options_panel.png", image[panel.y : panel.bottom, panel.x : panel.right])


def _save_backpack_slot_crops(image: np.ndarray, scene: SceneObservation | None) -> None:
    if scene is None or scene.backpack is None:
        return
    output_dir = Path("runtime/backpack_slots")
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = scene.backpack.panel_rect
    write_image(output_dir / "panel.png", image[panel.y : panel.bottom, panel.x : panel.right])
    for index, paint in enumerate(scene.backpack.paints, start=1):
        rect = paint.slot_rect
        filename = (
            f"slot_{index:02d}_"
            f"{paint.category.value}_lv{paint.level}_"
            f"{'treasure' if paint.is_treasure else 'normal'}_"
            f"c{paint.category_confidence:.2f}_l{paint.level_confidence:.2f}.png"
        )
        write_image(output_dir / filename, image[rect.y : rect.bottom, rect.x : rect.right])


#: 笔触面板右边界相对棋盘宽度的倍数。**不要轻易收窄它。**
#:
#: 2026-11-22 有人按某一个窗口尺寸量出「卡区右缘 = b.right+1.04·b.w」，据此把右边界
#: 收到 1.04。用 jietu/ 下 4 张真实截图回归：ratio=1.04 时 stage2_options 全空
#: （0/4），ratio 退回「不限」时 4/4 全对——收窄会把卡片正文切掉，OCR 读不到。
#:
#: 右侧多出来的空白交给 :func:_trim_blank_right 自动裁，不靠常数。
BRUSH_ROI_RIGHT_RATIO = 99.0


def _estimate_regions(image: np.ndarray, board: BoardObservation) -> list[RegionObservation]:
    height, width = image.shape[:2]
    b = board.rect
    x0 = _clamp_int(b.right + b.w * 0.10, 0, width - 1)
    content_x0 = _clamp_int(b.right - b.w * 0.06, 0, width - 1)
    right = _clamp_int(max(x0 + 1, width - b.w * 0.06), 1, width)
    panel_w = max(1, right - x0)
    content_panel_w = max(1, right - content_x0)
    # 笔触面板**单独**算宽度，不能跟 ``right`` 共用。
    #
    # 2026-11-22 用户在 1917x1077 窗口下逐像素量过：词缀卡区真值是
    # ``x ∈ [b.right+0.13·b.w, b.right+1.04·b.w]``、``y ∈ [b.y+0.30·b.h, b.y+0.65·b.h]``。
    # 而 ``right = width - 0.06·b.w`` 这一屏右侧空出约 298px（≈0.47·b.w）雪地，
    # 于是 ROI 右边界往外多出约 260px（31%）——整块 ROI 混进大片空白，
    # ``card_bands_ok`` 的上下两张卡分带判据直接崩掉，**面板根本检测不出来**，
    # 实测 17:30 之后 ``brush_options`` 一条都没有（此前累计 390 条）。
    #
    # 这里只给笔触面板收窄，``right`` 仍留给背包含义不变（背包要用到窗口右缘）。
    brush_right = _clamp_int(min(right, b.right + b.w * BRUSH_ROI_RIGHT_RATIO + 1), 1, width)
    brush_panel_w = max(1, brush_right - x0)
    stage_mode = _detect_brush_options_mode(image, x0, b, brush_panel_w)

    regions: list[RegionObservation] = [
        RegionObservation(
            kind=RegionKind.CURRENT_PAINT,
            rect=_bounded_rect(
                x0,
                b.y - b.h * 0.02,
                panel_w,
                b.h * 0.24,
                width,
                height,
            ),
            confidence=board.score,
            label="current paint",
        ),
    ]
    finish_signal = _find_finish_drawing_signal(image)
    if finish_signal is not None:
        regions.append(
            RegionObservation(
                kind=RegionKind.FINISH_DRAWING,
                rect=finish_signal[1],
                confidence=finish_signal[0],
                label=finish_signal[2],
            )
        )

    if stage_mode:
        regions.extend(
            [
                RegionObservation(
                    kind=RegionKind.BRUSH_OPTIONS,
                    rect=_bounded_rect(
                        x0,
                        b.y + b.h * 0.23,
                        brush_panel_w,
                        b.h * 0.48,
                        width,
                        height,
                    ),
                    confidence=board.score * 0.88,
                    label="brush options",
                ),
                RegionObservation(
                    kind=RegionKind.CONFIRM_BUTTON,
                    rect=_estimate_confirm_button_rect(image, b, x0, brush_panel_w, stage_mode=True),
                    confidence=board.score * 0.70,
                    label="confirm button",
                ),
            ]
        )
    else:
        regions.extend(
            [
                RegionObservation(
                    kind=RegionKind.BACKPACK,
                    rect=_bounded_rect(
                        content_x0,
                        b.y + b.h * 0.29,
                        content_panel_w,
                        b.h * 0.50,
                        width,
                        height,
                    ),
                    confidence=board.score * 0.85,
                    label="backpack",
                ),
                RegionObservation(
                    kind=RegionKind.FORCE_COMPLETE,
                    rect=_bounded_rect(
                        x0 + panel_w * 0.50,
                        b.y + b.h * 0.80,
                        panel_w * 0.42,
                        b.h * 0.11,
                        width,
                        height,
                    ),
                    confidence=board.score * 0.75,
                    label="force complete",
                ),
                RegionObservation(
                    kind=RegionKind.REFRESH_AFFIX,
                    rect=_bounded_rect(
                        x0,
                        b.y + b.h * 0.80,
                        panel_w * 0.42,
                        b.h * 0.11,
                        width,
                        height,
                    ),
                    confidence=board.score * 0.55,
                    label="refresh affix",
                ),
                RegionObservation(
                    kind=RegionKind.CONFIRM_BUTTON,
                    rect=_estimate_confirm_button_rect(image, b, x0, panel_w, stage_mode=False),
                    confidence=board.score * 0.70,
                    label="confirm button",
                ),
            ]
        )
    return regions


def _estimate_regions_fast(image: np.ndarray, board: BoardObservation) -> list[RegionObservation]:
    height, width = image.shape[:2]
    b = board.rect
    x0 = _clamp_int(b.right + b.w * 0.10, 0, width - 1)
    content_x0 = _clamp_int(b.right - b.w * 0.06, 0, width - 1)
    right = _clamp_int(max(x0 + 1, width - b.w * 0.06), 1, width)
    panel_w = max(1, right - x0)
    content_panel_w = max(1, right - content_x0)
    # 与 ``_estimate_regions`` 同款：笔触面板右边界单独算，不占用背包的 ``right``。
    brush_right = _clamp_int(min(right, b.right + b.w * BRUSH_ROI_RIGHT_RATIO + 1), 1, width)
    brush_panel_w = max(1, brush_right - x0)
    stage_mode = _detect_brush_options_mode(image, x0, b, brush_panel_w)
    regions = [
        RegionObservation(
            kind=RegionKind.CURRENT_PAINT,
            rect=_bounded_rect(x0, b.y - b.h * 0.02, panel_w, b.h * 0.24, width, height),
            confidence=board.score,
            label="current paint fast",
        ),
    ]
    blank_count = sum(tile.state == TileState.BLANK for tile in board.tiles)
    if blank_count <= 1:
        enter_button = _find_raster_action_button_fast(image, "enter_painting.png")
        if enter_button is not None:
            regions.append(
                RegionObservation(
                    kind=RegionKind.FINISH_DRAWING,
                    rect=enter_button[1],
                    confidence=enter_button[0],
                    label="enter painting button fast",
                )
            )
    if stage_mode:
        regions.extend(
            [
                RegionObservation(
                    kind=RegionKind.BRUSH_OPTIONS,
                    rect=_bounded_rect(x0, b.y + b.h * 0.23, brush_panel_w, b.h * 0.48, width, height),
                    confidence=board.score * 0.88,
                    label="brush options fast",
                ),
                RegionObservation(
                    kind=RegionKind.CONFIRM_BUTTON,
                    rect=_bounded_rect(x0 + brush_panel_w * 0.35, b.y + b.h * 0.88, brush_panel_w * 0.38, b.h * 0.10, width, height),
                    confidence=board.score * 0.62,
                    label="confirm button fast",
                ),
            ]
        )
    else:
        regions.extend(
            [
                RegionObservation(
                    kind=RegionKind.BACKPACK,
                    rect=_bounded_rect(content_x0, b.y + b.h * 0.29, content_panel_w, b.h * 0.50, width, height),
                    confidence=board.score * 0.85,
                    label="backpack fast",
                ),
                RegionObservation(
                    kind=RegionKind.FORCE_COMPLETE,
                    rect=_bounded_rect(x0 + panel_w * 0.50, b.y + b.h * 0.80, panel_w * 0.42, b.h * 0.11, width, height),
                    confidence=board.score * 0.75,
                    label="force complete fast",
                ),
                RegionObservation(
                    kind=RegionKind.REFRESH_AFFIX,
                    rect=_bounded_rect(x0, b.y + b.h * 0.80, panel_w * 0.42, b.h * 0.11, width, height),
                    confidence=board.score * 0.55,
                    label="refresh affix fast",
                ),
                RegionObservation(
                    kind=RegionKind.CONFIRM_BUTTON,
                    rect=_bounded_rect(x0 + panel_w * 0.35, b.y + b.h * 0.88, panel_w * 0.38, b.h * 0.10, width, height),
                    confidence=board.score * 0.62,
                    label="confirm button fast",
                ),
            ]
        )
    return regions


def _find_finish_drawing_signal(image: np.ndarray) -> tuple[float, Rect, str] | None:
    candidates: list[tuple[float, Rect, str]] = []
    button = _find_finish_drawing_button(image)
    if button is not None:
        candidates.append((button[0], button[1], "finish drawing button"))
    enter_button = _find_enter_painting_button(image)
    if enter_button is not None:
        candidates.append((enter_button[0], enter_button[1], "enter painting button"))
    banner = _find_finish_ready_banner(image)
    if banner is not None:
        candidates.append((banner[0], banner[1], "finish ready banner"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def _find_start_drawing_button_fast(image: np.ndarray) -> tuple[float, Rect] | None:
    if image.size == 0:
        return None
    height, width = image.shape[:2]
    rect = _bounded_rect(width * 0.72, height * 0.86, width * 0.15, height * 0.10, width, height)
    crop = image[rect.y : rect.bottom, rect.x : rect.right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    ice = ((hue >= 82) & (hue <= 112) & (saturation > 55) & (value > 130)) | (
        (saturation < 70) & (value > 205)
    )
    orange = (hue <= 25) & (saturation > 75) & (value > 105)
    ice_ratio = float(np.mean(ice))
    orange_ratio = float(np.mean(orange))
    if ice_ratio < 0.035 or orange_ratio < 0.003:
        return None
    confidence = float(np.clip(0.55 + ice_ratio * 1.8 + orange_ratio * 2.2, 0.0, 0.96))
    return confidence, rect


def _find_finish_drawing_button(image: np.ndarray) -> tuple[float, Rect] | None:
    return _find_blue_action_button(image, ("完成绘制",))


def _find_enter_painting_button(image: np.ndarray) -> tuple[float, Rect] | None:
    return _find_blue_action_button(image, ("入画",)) or _find_raster_action_button(
        image,
        "enter_painting.png",
    )


def _find_raster_action_button(image: np.ndarray, filename: str) -> tuple[float, Rect] | None:
    template = _load_current_paint_template(filename)
    if template is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    roi_rect = _bounded_rect(width * 0.45, height * 0.55, width * 0.50, height * 0.38, width, height)
    roi = image[roi_rect.y : roi_rect.bottom, roi_rect.x : roi_rect.right]
    if roi.size == 0:
        return None
    best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
    for scale in (0.82, 0.91, 1.0, 1.10, 1.20):
        candidate = template if scale == 1.0 else cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        if candidate.shape[0] > roi.shape[0] or candidate.shape[1] > roi.shape[1]:
            continue
        result = cv2.matchTemplate(roi, candidate, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        matched = (float(max_value), max_location, (candidate.shape[1], candidate.shape[0]))
        if best is None or matched[0] > best[0]:
            best = matched
    if best is None or best[0] < 0.78:
        return None
    score, (match_x, match_y), (match_width, match_height) = best
    center_x = roi_rect.x + match_x + match_width / 2.0
    center_y = roi_rect.y + match_y + match_height / 2.0
    button_width = max(width * 0.14, match_width * 4.8)
    button_height = max(height * 0.05, match_height * 2.1)
    button_rect = _bounded_rect(
        center_x - button_width / 2.0,
        center_y - button_height / 2.0,
        button_width,
        button_height,
        width,
        height,
    )
    return score, button_rect


def _find_raster_action_button_fast(image: np.ndarray, filename: str) -> tuple[float, Rect] | None:
    template = _load_current_paint_template(filename)
    if template is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    roi_rect = _bounded_rect(width * 0.67, height * 0.73, width * 0.23, height * 0.16, width, height)
    roi = image[roi_rect.y : roi_rect.bottom, roi_rect.x : roi_rect.right]
    if roi.size == 0:
        return None
    scale = min(width / 1673.0, height / 878.0)
    candidate = template if abs(scale - 1.0) < 0.01 else cv2.resize(
        template,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    if candidate.shape[0] > roi.shape[0] or candidate.shape[1] > roi.shape[1]:
        return None
    result = cv2.matchTemplate(roi, candidate, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    if score < 0.78:
        return None
    match_width, match_height = candidate.shape[1], candidate.shape[0]
    center_x = roi_rect.x + location[0] + match_width / 2.0
    center_y = roi_rect.y + location[1] + match_height / 2.0
    button_rect = _bounded_rect(
        center_x - max(width * 0.14, match_width * 4.8) / 2.0,
        center_y - max(height * 0.05, match_height * 2.1) / 2.0,
        max(width * 0.14, match_width * 4.8),
        max(height * 0.05, match_height * 2.1),
        width,
        height,
    )
    return float(score), button_rect


def _find_blue_action_button(image: np.ndarray, keywords: tuple[str, ...]) -> tuple[float, Rect] | None:
    height, width = image.shape[:2]
    # Settlement UI places a large blue button in the lower-right half, but
    # other blue buttons exist. Require the red/orange "完成绘制" text too.
    roi_rect = _bounded_rect(width * 0.38, height * 0.58, width * 0.58, height * 0.34, width, height)
    roi = image[roi_rect.y : roi_rect.bottom, roi_rect.x : roi_rect.right]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    blue = ((h >= 88) & (h <= 112) & (s > 45) & (v > 120)).astype(np.uint8) * 255
    blue = cv2.morphologyEx(
        blue,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 9)),
        iterations=2,
    )
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, Rect]] = []
    roi_area = max(1, roi.shape[0] * roi.shape[1])
    for contour in contours:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < roi_area * 0.025:
            continue
        aspect = w_box / max(1, h_box)
        if not 2.0 <= aspect <= 7.5:
            continue
        if w_box < width * 0.14 or h_box < height * 0.045:
            continue
        button_rect = _bounded_rect(roi_rect.x + x, roi_rect.y + y, w_box, h_box, width, height)
        crop = image[button_rect.y : button_rect.bottom, button_rect.x : button_rect.right]
        confidence = _finish_drawing_button_confidence(crop, area / roi_area, aspect, keywords)
        if confidence >= 0.62:
            candidates.append((confidence, button_rect))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def _find_finish_ready_banner(image: np.ndarray) -> tuple[float, Rect] | None:
    height, width = image.shape[:2]
    # The end prompt appears near the upper-left/top-center as a long blue
    # banner. Keep this conservative so normal current-paint text does not
    # become a false settlement.
    roi_rect = _bounded_rect(0, height * 0.03, width * 0.86, height * 0.18, width, height)
    roi = image[roi_rect.y : roi_rect.bottom, roi_rect.x : roi_rect.right]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    blue = ((h >= 88) & (h <= 112) & (s > 35) & (v > 95)).astype(np.uint8) * 255
    blue = cv2.morphologyEx(
        blue,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)),
        iterations=2,
    )
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, Rect]] = []
    roi_area = max(1, roi.shape[0] * roi.shape[1])
    for contour in contours:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < roi_area * 0.04:
            continue
        aspect = w_box / max(1, h_box)
        if not 4.0 <= aspect <= 14.0:
            continue
        if w_box < width * 0.35 or h_box < height * 0.045:
            continue
        rect = _bounded_rect(roi_rect.x + x, roi_rect.y + y, w_box, h_box, width, height)
        crop = image[rect.y : rect.bottom, rect.x : rect.right]
        confidence = _finish_ready_banner_confidence(crop, area / roi_area, aspect)
        if confidence >= 0.66:
            candidates.append((confidence, rect))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def _finish_ready_banner_confidence(crop: np.ndarray, area_ratio: float, aspect: float) -> float:
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    blue_ratio = float(np.mean((h >= 88) & (h <= 112) & (s > 35) & (v > 95)))
    white_text_ratio = float(np.mean((s < 90) & (v > 150)))
    warm_icon_ratio = float(np.mean((h >= 10) & (h <= 35) & (s > 70) & (v > 115)))
    keyword_score = _finish_banner_keyword_score(crop)
    aspect_score = 1.0 - min(1.0, abs(aspect - 8.0) / 6.0)
    area_score = min(1.0, area_ratio / 0.12)
    if keyword_score < 0.38:
        return 0.0
    return float(
        np.clip(
            blue_ratio * 0.30
            + white_text_ratio * 0.45
            + warm_icon_ratio * 0.25
            + keyword_score * 0.55
            + aspect_score * 0.10
            + area_score * 0.08,
            0.0,
            1.0,
        )
    )


def _finish_drawing_button_confidence(
    crop: np.ndarray,
    area_ratio: float,
    aspect: float,
    keywords: tuple[str, ...] = ("完成绘制",),
) -> float:
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    blue_ratio = float(np.mean((h >= 88) & (h <= 112) & (s > 45) & (v > 115)))
    red_text_ratio = float(np.mean((((h <= 12) | (h >= 170)) & (s > 55) & (v > 95))))
    bright_ratio = float(np.mean(v > 145))
    # 「完成绘制」是红字按钮。没有红字就不可能通过，先判它再跑字形模板匹配：
    # 那一步是 45 次 matchTemplate（实测 0.7~1.4s），而「入画」是白字蓝底，
    # 每次都白跑一遍。条件与原来的 `keyword_score < 0.40 or red_text_ratio < 0.010`
    # 完全等价，只是把便宜的判断提到前面。
    if red_text_ratio < 0.010:
        return 0.0
    keyword_score = _finish_button_keyword_score(crop, keywords)
    if keyword_score < 0.40:
        return 0.0
    aspect_score = 1.0 - min(1.0, abs(aspect - 3.6) / 3.0)
    area_score = min(1.0, area_ratio / 0.08)
    return float(
        np.clip(
            blue_ratio * 0.26
            + red_text_ratio * 0.95
            + bright_ratio * 0.08
            + keyword_score * 0.72
            + aspect_score * 0.10
            + area_score * 0.08,
            0.0,
            1.0,
        )
    )


def _finish_button_keyword_score(crop: np.ndarray, keywords: tuple[str, ...] = ("完成绘制",)) -> float:
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((((h <= 14) | (h >= 170)) & (s > 50) & (v > 85))).astype(np.uint8) * 255
    if float(np.mean(mask > 0)) < 0.006:
        return 0.0
    text_crop = cv2.bitwise_and(crop, crop, mask=mask)
    scores = match_keywords(
        text_crop,
        keywords,
        font_size_range=range(14, 34, 2),
        scale=2.0,
        include_binary=True,
        threshold=0.0,
    )
    return max(scores.values(), default=0.0)


def _finish_banner_keyword_score(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    scores = match_keywords(
        crop,
        ("全部放置", "可完成绘制", "颜料已全部放置"),
        font_size_range=range(14, 30, 2),
        scale=1.8,
        include_binary=True,
        threshold=0.0,
    )
    return max(scores.values(), default=0.0)


def _estimate_confirm_button_rect(
    image: np.ndarray,
    board_rect: Rect,
    x0: int,
    panel_w: int,
    *,
    stage_mode: bool,
) -> Rect:
    height, width = image.shape[:2]
    if stage_mode:
        return _bounded_rect(
            x0 + panel_w * 0.62,
            board_rect.y + board_rect.h * 0.86,
            panel_w * 0.34,
            board_rect.h * 0.12,
            width,
            height,
        )
    return _bounded_rect(
        x0 + panel_w * 0.52,
        board_rect.y + board_rect.h * 0.80,
        panel_w * 0.42,
        board_rect.h * 0.11,
        width,
        height,
    )


def _detect_brush_options_mode(image: np.ndarray, x0: int, board_rect: Rect, panel_w: int) -> bool:
    height, width = image.shape[:2]
    roi_rect = _bounded_rect(
        x0,
        board_rect.y + board_rect.h * 0.24,
        panel_w,
        board_rect.h * 0.45,
        width,
        height,
    )
    roi = image[roi_rect.y : roi_rect.bottom, roi_rect.x : roi_rect.right]
    if roi.size == 0:
        return False
    # Real brush pages contain two wide horizontal cards. Keyword scores alone
    # are unsafe here: backpack art and the large pending-paint panel can both
    # weakly match brush words and previously entered the wrong phase.
    return _has_brush_card_layout(roi)


def _has_brush_card_layout(crop: np.ndarray) -> bool:
    """这块区域像不像「笔触选择面板」（上下两张宽卡片）。

    2026-11-20 换成 ``brush_panel.card_bands_ok``：原来按「宽高比 ≥3.8 的宽卡片
    至少两张」判，窗口尺寸一变（1607x822 下卡片 404x116，宽高比 3.48）就全判否，
    BRUSH_OPTIONS 区域根本不建，整局笔触读数恒为空。
    """
    return brush_panel.card_bands_ok(crop)


def _candidate_board_rects(image: np.ndarray) -> list[Rect]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 110)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Rect] = []
    min_side = int(min(width, height) * 0.34)
    max_side = int(min(width, height) * 0.90)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        side = max(w, h)
        if side < min_side or side > max_side:
            continue
        aspect = w / max(1, h)
        if not 0.82 <= aspect <= 1.18:
            continue
        expanded = _expand_to_square(Rect(x, y, w, h), width, height)
        if not _is_plausible_board_candidate(image.shape, expanded):
            continue
        candidates.append(expanded)

    candidates.extend(_estimated_board_layout_candidates(image))

    candidates.extend(_grid_scan_candidates(image))
    return _dedupe_rects(_preselect_board_candidates(image, candidates), image=image)


def _estimated_board_layout_candidate(image: np.ndarray) -> Rect | None:
    candidates = _estimated_board_layout_candidates(image)
    return candidates[0] if candidates else None


def _estimated_board_layout_candidates(image: np.ndarray) -> list[Rect]:
    height, width = image.shape[:2]
    short = min(width, height)
    candidates: list[Rect] = []
    # 先放实测标定出来的版式（``layout.py`` / ``runtime/layout.json``）。
    # 下面那组硬编码比例只是「大概在这个方向」的兜底，实测值能直接落到正确位置：
    # 1037x544 实测，兜底比例 (0.620, 0.215, 0.245) 会整体右偏整整一格，
    # 而标定值 (0.594, 0.187, 0.256) 一格不差。
    screen = _screen_layout(width, height)

    def _add(rect: Rect) -> None:
        if not _is_rect_inside_image(image.shape, rect):
            return
        if not _is_plausible_board_candidate(image.shape, rect):
            return
        # 这里不能直接套 ``_dedupe_rects``：它按面积从大到小排、再用 IoU 0.72 去重，
        # 而「偏一格的兜底框」和「标定框」的 IoU 恰好是 0.77——大框会把标定框吃掉，
        # 正好把要修的那个位置丢了。只在几乎重合时才去重，并保留插入顺序。
        if any(_iou(rect, kept) >= 0.93 for kept in candidates):
            return
        candidates.append(rect)

    for layout in (*screen.active, *screen.start):
        _add(layout.resolve(width, height))
    presets = (
        # Common centered-window layout.
        (0.600, 0.303, 0.373),
        (0.585, 0.295, 0.360),
        # Older narrow estimate kept as a fallback for previously working captures.
        (0.475, 0.245, 0.300),
        # More left/top layouts seen in direct game-window screenshots.
        (0.700, 0.170, 0.185),
        (0.750, 0.135, 0.145),
        (0.620, 0.215, 0.245),
    )
    for side_ratio, x_ratio, y_ratio in presets:
        side = int(round(short * side_ratio))
        _add(Rect(int(round(width * x_ratio)), int(round(height * y_ratio)), side, side))
    return candidates


#: 版式候选最多精修几个。
#:
#: ``_refine_board_rect`` 要跑全图 Canny + 上百次评分，1607x822 实测 0.6s/次。
#: 原来对**每个**候选都无差别精修（1607x822 上是 31 个），实测 18.5 秒，
#: 而且这 31 个候选里一个都不在真棋盘上（预设比例最小 x=0.475，真实棋盘在
#: x=0.075）—— 十几秒全花在一个注定失败的搜索上。用户看到的就是
#: 「开局识别卡十秒」。
#:
#: 综合分（``_score_board_rect``）很便宜（31 个共 0.29s），先粗排再精修头部，
#: 结果不变而耗时降到 1/8。留 4 个而不是 1 个：候选框偏 8px 时粗分会掉、
#: 精修才救得回来，只取 top-1 会漏掉这种。
BEST_LAYOUT_REFINE_LIMIT = 4


def _best_estimated_board_layout(image: np.ndarray) -> tuple[Rect, float] | None:
    candidates = _estimated_board_layout_candidates(image)
    if not candidates:
        return None
    if len(candidates) <= BEST_LAYOUT_REFINE_LIMIT:
        ordered = [(None, rect) for rect in candidates]
    else:
        ordered = [
            (score, rect)
            for score, rect in sorted(
                ((_score_board_rect(image, rect), rect) for rect in candidates),
                key=lambda item: item[0],
                reverse=True,
            )[:BEST_LAYOUT_REFINE_LIMIT]
        ]
    best: tuple[Rect, float] | None = None
    for _, rect in ordered:
        refined = _refine_board_rect(image, rect)
        score = _score_board_rect(image, refined)
        if best is None or score > best[1]:
            best = (refined, score)
    return best


def _preselect_board_candidates(image: np.ndarray, rects: list[Rect], limit: int = 60) -> list[Rect]:
    if len(rects) <= limit:
        return rects
    layout_candidates = _estimated_board_layout_candidates(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 110)
    scored = []
    for rect in rects:
        grid = _score_edges_grid(edges, rect)
        position = _board_position_score(image.shape, rect)
        size = _board_size_score(image.shape, rect)
        scored.append((grid * (0.55 + position * 0.35 + size * 0.10), rect))
    scored.sort(reverse=True, key=lambda item: item[0])
    selected = [rect for _, rect in scored[:limit]]
    for layout in layout_candidates:
        if all(_iou(layout, rect) < 0.90 for rect in selected):
            selected.append(layout)
    return selected


def _grid_scan_candidates(image: np.ndarray) -> list[Rect]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 110)
    candidates: list[Rect] = []
    min_side = int(min(width, height) * 0.38)
    max_side = int(min(width, height) * 0.82)
    step = max(8, min(width, height) // 80)

    # The playable board is the large grid left of the paint/backpack panels.
    # The right-side backpack can also look like a 6x6 grid, so scan only the
    # plausible board area and let scoring refine the exact rectangle.
    for side in range(min_side, max_side + 1, step * 4):
        y_limit = max(1, height - side)
        x_limit = max(1, min(int(width * 0.62), width - side))
        for y in range(0, y_limit, step * 5):
            for x in range(0, x_limit, step * 5):
                rect = Rect(x, y, side, side)
                if not _is_plausible_board_candidate(image.shape, rect):
                    continue
                score = _score_edges_grid(edges, rect)
                if score > 0.42:
                    candidates.append(rect)
    candidates.extend(_left_board_anchor_candidates(image, edges))
    return candidates


def _left_board_anchor_candidates(image: np.ndarray, edges: np.ndarray) -> list[Rect]:
    height, width = image.shape[:2]
    candidates: list[Rect] = []
    min_side = int(min(width, height) * 0.42)
    max_side = int(min(width, height) * 0.52)
    x_min = int(width * 0.18)
    x_max = int(width * 0.38)
    y_min = int(height * 0.22)
    y_max = int(height * 0.42)
    step = max(22, min(width, height) // 52)
    side_step = max(18, step)
    for side in range(min_side, max_side + 1, side_step):
        for y in range(y_min, min(y_max, height - side) + 1, step):
            for x in range(x_min, min(x_max, width - side) + 1, step):
                rect = Rect(x, y, side, side)
                if not _is_plausible_board_candidate(image.shape, rect):
                    continue
                score = _score_edges_grid(edges, rect)
                if score > 0.34:
                    candidates.append(rect)
    return candidates


def _board_grid_score(
    image: np.ndarray,
    rect: Rect,
    *,
    margin: int | None = None,
    outer_penalty: bool = True,
) -> float:
    """只在棋盘附近算边缘，不扫整屏。

    原先每评一个候选框就把整张画面灰度+Canny 一遍，而真正用到的只有棋盘那一块；
    实时链路一帧要评好几个候选框，这部分是纯浪费。

    margin 默认留出 1.25 格，因为 ``_outer_edge_penalty`` 要看框外一格有没有线。
    实时链路（布局直接给框、只判断"像不像棋盘"）传 ``outer_penalty=False``：
    那里只关心网格成不成立，不该因为框外恰好有一条 UI 线就被判成"不是棋盘"。
    """
    height, width = image.shape[:2]
    if margin is None:
        margin = int(rect.w / BOARD_SIZE * 1.3) + 4
    x1 = max(0, rect.x - margin)
    y1 = max(0, rect.y - margin)
    x2 = min(width, rect.right + margin)
    y2 = min(height, rect.bottom + margin)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 110)
    local = Rect(rect.x - x1, rect.y - y1, rect.w, rect.h)
    score = _score_edges_grid(edges, local)
    if outer_penalty:
        score *= _outer_edge_penalty(edges, local)
    return score


def _score_board_rect(image: np.ndarray, rect: Rect) -> float:
    if not _is_rect_inside_image(image.shape, rect):
        return 0.0
    position_score = _board_position_score(image.shape, rect)
    size_score = _board_size_score(image.shape, rect)
    if position_score <= 0.02 or size_score <= 0.02:
        return 0.0

    grid_score = _board_grid_score(image, rect)
    if grid_score < 0.16:
        return grid_score * position_score * size_score
    color_score, completeness_score = _board_tile_scores(image, rect)
    return (
        grid_score * 0.46
        + color_score * 0.22
        + size_score * 0.14
        + completeness_score * 0.18
    ) * position_score


def _board_scoring_arrays(image: np.ndarray, rect: Rect) -> tuple[np.ndarray, np.ndarray]:
    """整块棋盘只转一次 HSV、只做一次 Canny，36 个格子共用。

    原先每格各做一次 cvtColor（有的函数一格里还做两次），一帧下来一百多次，
    是实时链路里最大的浪费。裁剪时把边长裁成 6 的倍数，之后按下标切片即可。
    """
    height, width = image.shape[:2]
    x1 = max(0, min(width, rect.x))
    y1 = max(0, min(height, rect.y))
    x2 = max(x1, min(width, rect.right))
    y2 = max(y1, min(height, rect.bottom))
    crop = image[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    h -= h % BOARD_SIZE
    w -= w % BOARD_SIZE
    if h <= 0 or w <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8), np.empty((0, 0), dtype=np.uint8)
    crop = crop[:h, :w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 35, 100)
    return hsv, edges


def _tile_view(array: np.ndarray, index: int, pad_divisor: int = 0) -> np.ndarray:
    """按下标取第 index 格（0..35）的视图；``pad_divisor`` 与原来按格尺寸内缩的比例一致。"""
    row, col = divmod(index, BOARD_SIZE)
    height, width = array.shape[:2]
    tile_h = height // BOARD_SIZE
    tile_w = width // BOARD_SIZE
    if tile_h <= 0 or tile_w <= 0:
        return array[0:0, 0:0]
    y1, y2 = row * tile_h, (row + 1) * tile_h
    x1, x2 = col * tile_w, (col + 1) * tile_w
    if pad_divisor:
        pad_y = max(2, tile_h // pad_divisor)
        pad_x = max(2, tile_w // pad_divisor)
        if tile_h - 2 * pad_y > 0 and tile_w - 2 * pad_x > 0:
            y1, y2 = y1 + pad_y, y2 - pad_y
            x1, x2 = x1 + pad_x, x2 - pad_x
    return array[y1:y2, x1:x2]


def _tile_state_from_hsv(hsv: np.ndarray) -> TileState:
    """按 HSV 判断格子类型。调用方负责先转好 HSV，避免每格重复 cvtColor。"""
    if hsv.size == 0:
        return TileState.UNKNOWN

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mean_v = float(np.mean(v)) / 255.0
    mean_s = float(np.mean(s)) / 255.0
    dark_ratio = float(np.mean(v < 65))
    very_dark_ratio = float(np.mean(v < 38))
    bright_ratio = float(np.mean(v > 150))
    low_sat_ratio = float(np.mean(s < 45))
    blue_ratio = float(np.mean((h >= 82) & (h <= 112) & (s > 70) & (v > 80)))
    cyan_ratio = float(np.mean((h >= 75) & (h <= 100) & (s > 95) & (v > 120)))
    non_blue_saturated = float(np.mean((s > 70) & (v > 120) & ~((h >= 75) & (h <= 112))))
    purple_ratio = float(np.mean((h >= 120) & (h <= 170) & (s > 70) & (v > 90)))
    gold_ratio = float(np.mean((h >= 15) & (h <= 45) & (s > 70) & (v > 110)))
    gray_mid_ratio = float(np.mean((s < 55) & (v >= 55) & (v <= 145)))

    center = hsv[
        hsv.shape[0] // 4 : hsv.shape[0] * 3 // 4,
        hsv.shape[1] // 4 : hsv.shape[1] * 3 // 4,
    ]
    if center.size == 0:
        center = hsv
    center_blue = float(
        np.mean(
            (center[:, :, 0] >= 75)
            & (center[:, :, 0] <= 112)
            & (center[:, :, 1] > 80)
            & (center[:, :, 2] > 80)
        )
    )
    center_dark = float(np.mean(center[:, :, 2] < 70))

    if very_dark_ratio > 0.34 and mean_v < 0.33:
        return TileState.VOID
    if dark_ratio > 0.62 and mean_v < 0.38:
        return TileState.VOID
    if bright_ratio > 0.45 and low_sat_ratio > 0.44:
        return TileState.BLANK
    if cyan_ratio >= 0.34 and center_blue >= 0.58 and mean_s >= 0.50:
        return TileState.BLOCKED
    if non_blue_saturated > 0.20 or purple_ratio > 0.08 or gold_ratio > 0.06:
        return TileState.PAINT
    if center_dark > 0.18 and (cyan_ratio > 0.06 or mean_s > 0.26):
        return TileState.PAINT
    if gray_mid_ratio > 0.38 and bright_ratio < 0.42:
        return TileState.FADED
    if blue_ratio > 0.72 and center_blue > 0.70:
        return TileState.BLOCKED
    if dark_ratio > 0.35:
        return TileState.VOID
    return TileState.UNKNOWN


def _tile_texture_from_arrays(hsv: np.ndarray, edges: np.ndarray) -> bool:
    """用已经算好的 HSV / 边缘图判断这一格是否有棋盘纹理。"""
    if hsv.size == 0 or edges.size == 0:
        return False
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    edge_ratio = float(np.mean(edges > 0))
    contrast = float(np.std(v)) / 255.0
    bright_ratio = float(np.mean(v > 145))
    low_sat_ratio = float(np.mean(s < 55))
    dark_ratio = float(np.mean(v < 70))
    # Empty blue background is smooth and has little contrast/edge content.
    if edge_ratio < 0.018 and contrast < 0.085 and bright_ratio < 0.24 and dark_ratio < 0.24:
        return False
    if low_sat_ratio < 0.12 and edge_ratio < 0.012 and contrast < 0.06:
        return False
    return True


def _core_stacks(array: np.ndarray, pad_divisor: int = 10) -> np.ndarray:
    """把整块棋盘重排成 (36, H, W[, C])，一次算完所有格子。

    这是"能批量处理就不逐个处理"的具体做法：36 格各自做 cvtColor / 均值，
    改成整块做一次、再按格切片。
    """
    height, width = array.shape[:2]
    tile_h = height // BOARD_SIZE
    tile_w = width // BOARD_SIZE
    if tile_h <= 0 or tile_w <= 0:
        return np.empty((0, 0, 0), dtype=array.dtype)
    pad_y = max(2, tile_h // pad_divisor)
    pad_x = max(2, tile_w // pad_divisor)
    if tile_h - 2 * pad_y <= 0 or tile_w - 2 * pad_x <= 0:
        pad_y = pad_x = 0
    usable_h = tile_h * BOARD_SIZE
    usable_w = tile_w * BOARD_SIZE
    trimmed = array[:usable_h, :usable_w]
    if trimmed.ndim == 2:
        blocks = trimmed.reshape(BOARD_SIZE, tile_h, BOARD_SIZE, tile_w)
        inner = blocks[:, pad_y : tile_h - pad_y, :, pad_x : tile_w - pad_x]
        return inner.transpose(0, 2, 1, 3).reshape(
            BOARD_SIZE * BOARD_SIZE, tile_h - 2 * pad_y, tile_w - 2 * pad_x
        )
    channels = trimmed.shape[2]
    blocks = trimmed.reshape(BOARD_SIZE, tile_h, BOARD_SIZE, tile_w, channels)
    inner = blocks[:, pad_y : tile_h - pad_y, :, pad_x : tile_w - pad_x, :]
    return inner.transpose(0, 2, 1, 3, 4).reshape(
        BOARD_SIZE * BOARD_SIZE, tile_h - 2 * pad_y, tile_w - 2 * pad_x, channels
    )


def _tile_ratio_vectors(stack: np.ndarray) -> dict[str, np.ndarray]:
    """一次性算出 36 格的 HSV 比例特征。"""
    h = stack[..., 0].astype(np.int16)
    s = stack[..., 1].astype(np.int16)
    v = stack[..., 2].astype(np.int16)
    axis = (1, 2)
    return {
        "mean_v": v.mean(axis=axis) / 255.0,
        "mean_s": s.mean(axis=axis) / 255.0,
        "dark_ratio": (v < 65).mean(axis=axis),
        "very_dark_ratio": (v < 38).mean(axis=axis),
        "bright_ratio": (v > 150).mean(axis=axis),
        "low_sat_ratio": (s < 45).mean(axis=axis),
        "blue_ratio": ((h >= 82) & (h <= 112) & (s > 70) & (v > 80)).mean(axis=axis),
        "cyan_ratio": ((h >= 75) & (h <= 100) & (s > 95) & (v > 120)).mean(axis=axis),
        "non_blue_saturated": ((s > 70) & (v > 120) & ~((h >= 75) & (h <= 112))).mean(axis=axis),
        "purple_ratio": ((h >= 120) & (h <= 170) & (s > 70) & (v > 90)).mean(axis=axis),
        "gold_ratio": ((h >= 15) & (h <= 45) & (s > 70) & (v > 110)).mean(axis=axis),
        "gray_mid_ratio": ((s < 55) & (v >= 55) & (v <= 145)).mean(axis=axis),
    }


def _tile_center_vectors(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """格子中心区域的蓝占比和暗占比。"""
    height, width = stack.shape[1:3]
    center = stack[
        :,
        height // 4 : height * 3 // 4,
        width // 4 : width * 3 // 4,
        :,
    ]
    if center.size == 0:
        center = stack
    ch = center[..., 0].astype(np.int16)
    cs = center[..., 1].astype(np.int16)
    cv = center[..., 2].astype(np.int16)
    axis = (1, 2)
    center_blue = ((ch >= 75) & (ch <= 112) & (cs > 80) & (cv > 80)).mean(axis=axis)
    center_dark = (cv < 70).mean(axis=axis)
    return center_blue, center_dark


def _tile_texture_vector(stack: np.ndarray, edge_stack: np.ndarray) -> np.ndarray:
    """36 格是否有棋盘纹理（逐格判定，但特征一次算完）。"""
    axis = (1, 2)
    v = stack[..., 2].astype(np.int16)
    s = stack[..., 1].astype(np.int16)
    edge_ratio = (edge_stack > 0).mean(axis=axis)
    contrast = v.std(axis=axis) / 255.0
    bright_ratio = (v > 145).mean(axis=axis)
    low_sat_ratio = (s < 55).mean(axis=axis)
    dark_ratio = (v < 70).mean(axis=axis)
    empty = (
        (edge_ratio < 0.018)
        & (contrast < 0.085)
        & (bright_ratio < 0.24)
        & (dark_ratio < 0.24)
    )
    flat = (low_sat_ratio < 0.12) & (edge_ratio < 0.012) & (contrast < 0.06)
    return ~(empty | flat)


def _board_tile_scores(image: np.ndarray, rect: Rect) -> tuple[float, float]:
    """一次遍历算完"颜色多样性"和"格子完整度"，两份结果共用同一批数组。"""
    hsv, edges = _board_scoring_arrays(image, rect)
    if hsv.size == 0:
        return 0.0, 0.0
    stack = _core_stacks(hsv, 10)
    edge_stack = _core_stacks(edges, 10)
    if stack.size == 0 or edge_stack.size == 0:
        return 0.0, 0.0
    ratios = _tile_ratio_vectors(stack)
    center_blue, center_dark = _tile_center_vectors(stack)
    textured = _tile_texture_vector(stack, edge_stack)

    states: list[TileState] = []
    for index in range(BOARD_SIZE * BOARD_SIZE):
        states.append(
            _tile_state_from_ratios(
                mean_v=float(ratios["mean_v"][index]),
                mean_s=float(ratios["mean_s"][index]),
                dark_ratio=float(ratios["dark_ratio"][index]),
                very_dark_ratio=float(ratios["very_dark_ratio"][index]),
                bright_ratio=float(ratios["bright_ratio"][index]),
                low_sat_ratio=float(ratios["low_sat_ratio"][index]),
                blue_ratio=float(ratios["blue_ratio"][index]),
                cyan_ratio=float(ratios["cyan_ratio"][index]),
                non_blue_saturated=float(ratios["non_blue_saturated"][index]),
                purple_ratio=float(ratios["purple_ratio"][index]),
                gold_ratio=float(ratios["gold_ratio"][index]),
                gray_mid_ratio=float(ratios["gray_mid_ratio"][index]),
                center_blue=float(center_blue[index]),
                center_dark=float(center_dark[index]),
            )
        )

    valid_cells = int(textured.sum())
    bottom_valid = int(textured[-BOARD_SIZE:].sum())
    unique = len(set(states))
    known = sum(1 for state in states if state != TileState.UNKNOWN)
    color_score = float(np.clip((unique / 5.0) * 0.45 + (known / 36.0) * 0.55, 0.0, 1.0))
    completeness = float(np.clip((valid_cells / 36.0) * 0.50 + (bottom_valid / 6.0) * 0.50, 0.0, 1.0))
    return color_score, completeness


def _tile_state_from_ratios(
    *,
    mean_v: float,
    mean_s: float,
    dark_ratio: float,
    very_dark_ratio: float,
    bright_ratio: float,
    low_sat_ratio: float,
    blue_ratio: float,
    cyan_ratio: float,
    non_blue_saturated: float,
    purple_ratio: float,
    gold_ratio: float,
    gray_mid_ratio: float,
    center_blue: float,
    center_dark: float,
) -> TileState:
    """纯阈值判定，输入是已经算好的比例。"""
    if very_dark_ratio > 0.34 and mean_v < 0.33:
        return TileState.VOID
    if dark_ratio > 0.62 and mean_v < 0.38:
        return TileState.VOID
    if bright_ratio > 0.45 and low_sat_ratio > 0.44:
        return TileState.BLANK
    if cyan_ratio >= 0.34 and center_blue >= 0.58 and mean_s >= 0.50:
        return TileState.BLOCKED
    if non_blue_saturated > 0.20 or purple_ratio > 0.08 or gold_ratio > 0.06:
        return TileState.PAINT
    if center_dark > 0.18 and (cyan_ratio > 0.06 or mean_s > 0.26):
        return TileState.PAINT
    if gray_mid_ratio > 0.38 and bright_ratio < 0.42:
        return TileState.FADED
    if blue_ratio > 0.72 and center_blue > 0.70:
        return TileState.BLOCKED
    if dark_ratio > 0.35:
        return TileState.VOID
    return TileState.UNKNOWN


def _score_tile_color_diversity(image: np.ndarray, rect: Rect) -> float:
    return _board_tile_scores(image, rect)[0]


def _score_board_cell_completeness(image: np.ndarray, rect: Rect) -> float:
    return _board_tile_scores(image, rect)[1]


def _classify_tile_for_board_score(tile_bgr: np.ndarray) -> TileState:
    if tile_bgr.size == 0:
        return TileState.UNKNOWN
    pad_y = max(2, tile_bgr.shape[0] // 10)
    pad_x = max(2, tile_bgr.shape[1] // 10)
    core = tile_bgr[pad_y : tile_bgr.shape[0] - pad_y, pad_x : tile_bgr.shape[1] - pad_x]
    if core.size == 0:
        core = tile_bgr
    return _tile_state_from_hsv(cv2.cvtColor(core, cv2.COLOR_BGR2HSV))


def _tile_has_board_texture(tile_bgr: np.ndarray) -> bool:
    if tile_bgr.size == 0:
        return False
    pad_y = max(2, tile_bgr.shape[0] // 10)
    pad_x = max(2, tile_bgr.shape[1] // 10)
    core = tile_bgr[pad_y : tile_bgr.shape[0] - pad_y, pad_x : tile_bgr.shape[1] - pad_x]
    if core.size == 0:
        core = tile_bgr
    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    return _tile_texture_from_arrays(hsv, cv2.Canny(gray, 35, 100))


def _is_rect_inside_image(shape: tuple[int, ...], rect: Rect) -> bool:
    height, width = shape[:2]
    return rect.x >= 0 and rect.y >= 0 and rect.right <= width and rect.bottom <= height


def _is_plausible_board_candidate(shape: tuple[int, ...], rect: Rect) -> bool:
    height, width = shape[:2]
    if not _is_rect_inside_image(shape, rect):
        return False
    side_ratio = rect.w / max(1, min(width, height))
    center_x_ratio = (rect.x + rect.w * 0.5) / max(1, width)
    if side_ratio < 0.32 or side_ratio > 0.92:
        return False
    if rect.x > width * 0.68 or center_x_ratio > 0.74:
        return False
    # The paint/backpack panel starts just to the right of the 6x6 board. A
    # shifted candidate can align with the board's internal lines while crossing
    # into that right panel, which makes the bottom rows become background.
    if rect.right > width * 0.56 and side_ratio >= 0.38 and center_x_ratio > 0.52:
        return False
    if rect.right > width * 0.84 and side_ratio < 0.54:
        return False
    return True


def _board_size_score(shape: tuple[int, ...], rect: Rect) -> float:
    height, width = shape[:2]
    side_ratio = rect.w / max(1, min(width, height))
    if side_ratio < 0.32 or side_ratio > 0.92:
        return 0.0
    # Sub-regions inside the real board can also look like a grid, especially
    # when several strong internal lines are visible. The playable 6x6 board is
    # normally around 0.55-0.70 of the game-window short side; keep smaller
    # candidates alive, but make them lose to the full board unless their other
    # evidence is overwhelming.
    #
    # 0.44~0.50 这一段原来给到 0.65~1.00，等于放行了「棋盘内部一圈」的候选：
    # 1920x1080 全屏实测，529/1080=0.49 的内部框靠 grid/color 双满分以 0.719
    # 压过真正框住 6x6 的 632/1080=0.585（0.705），结果整块棋盘偏小一圈，
    # 连带 current_paint 区域往下滑，颜料面板文字读不到、方向与目标全成 UNKNOWN。
    # 压到 0.25~0.80，让偏小的框明显吃亏。
    if side_ratio < 0.44:
        return float(np.interp(side_ratio, [0.32, 0.44], [0.04, 0.25]))
    if side_ratio < 0.50:
        return float(np.interp(side_ratio, [0.44, 0.50], [0.25, 0.80]))
    if side_ratio < 0.64:
        return 1.0
    if side_ratio < 0.74:
        return float(np.interp(side_ratio, [0.64, 0.74], [0.95, 0.50]))
    return float(np.interp(side_ratio, [0.74, 0.92], [0.50, 0.08]))


def _board_position_score(shape: tuple[int, ...], rect: Rect) -> float:
    height, width = shape[:2]
    center_x_ratio = (rect.x + rect.w * 0.5) / max(1, width)
    left_ratio = rect.x / max(1, width)
    right_ratio = rect.right / max(1, width)
    side_ratio = rect.w / max(1, min(width, height))

    # In this minigame the real board is left of the current-paint/backpack
    # panels. Shifted candidates on the right panel can have stronger grid
    # lines, so penalize them earlier instead of letting edge score dominate.
    if center_x_ratio <= 0.42:
        score = 1.0
    elif center_x_ratio <= 0.50:
        score = float(np.interp(center_x_ratio, [0.42, 0.50], [1.0, 0.82]))
    elif center_x_ratio <= 0.56:
        score = float(np.interp(center_x_ratio, [0.50, 0.56], [0.82, 0.58]))
    elif center_x_ratio <= 0.68:
        score = float(np.interp(center_x_ratio, [0.56, 0.68], [0.58, 0.22]))
    else:
        score = float(np.interp(min(center_x_ratio, 0.86), [0.68, 0.86], [0.35, 0.04]))

    if left_ratio > 0.62:
        score *= 0.35
    if right_ratio > 0.84 and side_ratio < 0.54:
        score *= 0.25
    if right_ratio > 0.56 and side_ratio >= 0.38 and center_x_ratio > 0.52:
        score *= 0.08
    return float(np.clip(score, 0.0, 1.0))


def _refine_board_rect(image: np.ndarray, rect: Rect) -> Rect:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 110)
    # 积分图只算一次，精修里上千个候选共用（见 ``_grid_line_strengths``）。
    integral = cv2.integral(edges, sdepth=cv2.CV_64F)
    height, width = image.shape[:2]

    best_score = _refine_candidate_score(image, edges, rect, integral=integral)
    best_rect = rect

    side_min = max(BOARD_SIZE * 30, int(rect.w * 0.94))
    side_max = min(min(width, height), int(rect.w * 1.06))
    step = max(22, rect.w // 20)
    side_step = max(12, step)
    x_min = max(0, int(rect.x - rect.w * 0.08))
    x_max = min(width - side_min, int(rect.x + rect.w * 0.10))
    y_min = max(0, int(rect.y - rect.h * 0.08))
    y_max = min(height - side_min, int(rect.y + rect.h * 0.08))

    for side in range(side_min, side_max + 1, side_step):
        max_x_for_side = min(width - side, x_max)
        max_y_for_side = min(height - side, y_max)
        if max_x_for_side < x_min or max_y_for_side < y_min:
            continue
        for y in range(y_min, max_y_for_side + 1, step):
            for x in range(x_min, max_x_for_side + 1, step):
                candidate = Rect(x, y, side, side)
                score = _refine_candidate_score(image, edges, candidate, integral=integral)
                if score > best_score:
                    best_score = score
                    best_rect = candidate

    return _refine_board_rect_precise(image, edges, best_rect, width, height, integral=integral)


def _refine_board_rect_precise(
    image: np.ndarray,
    edges: np.ndarray,
    rect: Rect,
    width: int,
    height: int,
    *,
    integral: np.ndarray | None = None,
) -> Rect:
    best_score = _refine_candidate_score(image, edges, rect, integral=integral)
    best_rect = rect
    # 粗搜步长是 ``max(22, w//20)``，精修只需把 ±1 格内的取整误差收掉。
    # 半径 5→4：搜索点从 216 降到 125（-42%），实测结果一格没变
    # （1607x822 六张图 rect 完全相同），而这段是每次定位跑三遍的热点。
    precise_radius = 4
    precise_step = 2
    for side in range(max(BOARD_SIZE * 30, rect.w - precise_radius), min(width, height, rect.w + precise_radius) + 1, precise_step):
        max_x = width - side
        max_y = height - side
        for y in range(max(0, rect.y - precise_radius), min(max_y, rect.y + precise_radius) + 1, precise_step):
            for x in range(max(0, rect.x - precise_radius), min(max_x, rect.x + precise_radius) + 1, precise_step):
                candidate = Rect(x, y, side, side)
                score = _refine_candidate_score(image, edges, candidate, integral=integral)
                if score > best_score:
                    best_score = score
                    best_rect = candidate
    return best_rect


def _refine_candidate_score(
    image: np.ndarray,
    edges: np.ndarray,
    rect: Rect,
    *,
    integral: np.ndarray | None = None,
) -> float:
    if not _is_rect_inside_image(image.shape, rect):
        return 0.0
    # 14 条网格线的强度算一次就够：``_score_edges_grid`` 和 ``_outer_edge_penalty``
    # 各自调一次 ``_grid_line_strengths``，而后者每条线一个 ``ndarray.mean()``。
    # 精修一次要评 1200+ 个候选，等于几万次 mean —— 实测 3 秒，全花在重复计算上。
    strengths = _grid_line_strengths(edges, rect, integral=integral)
    grid = _score_edges_grid(edges, rect, strengths=strengths)
    position = _board_position_score(image.shape, rect)
    size = _board_size_score(image.shape, rect)
    return grid * (0.55 + position * 0.35 + size * 0.10) * _outer_edge_penalty(
        edges, rect, strengths=strengths
    )


def _grid_line_strengths(
    edges: np.ndarray,
    rect: Rect,
    *,
    integral: np.ndarray | None = None,
) -> list[float]:
    """返回棋盘 7 竖 7 横共 14 条网格线的边缘强度（0~1）。

    传了 ``integral``（``cv2.integral(edges, sdepth=cv2.CV_64F)``）就改用它做
    矩形求和：每条线常数时间，不再切一张小图调 ``ndarray.mean()``。
    精修一次要评 1200+ 个候选、每个 14 条线，实测 3 万次 ``mean()`` 占 2.3 秒 ——
    这是「窗口没标定时开局识别卡好几秒」的主要开销。
    """
    h, w = edges.shape[:2]
    line_band = max(2, rect.w // 120)
    strengths: list[float] = []
    for i in range(BOARD_SIZE + 1):
        x = round(rect.x + rect.w * i / BOARD_SIZE)
        x1 = max(0, x - line_band)
        x2 = min(w, x + line_band + 1)
        strengths.append(
            _band_mean(edges, integral, max(0, rect.y), min(h, rect.bottom), x1, x2)
        )

        y = round(rect.y + rect.h * i / BOARD_SIZE)
        y1 = max(0, y - line_band)
        y2 = min(h, y + line_band + 1)
        strengths.append(
            _band_mean(edges, integral, y1, y2, max(0, rect.x), min(w, rect.right))
        )
    return strengths


def _band_mean(
    edges: np.ndarray,
    integral: np.ndarray | None,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
) -> float:
    """一条网格带内的平均边缘强度（0~1）。"""

    if y2 <= y1 or x2 <= x1:
        return 0.0
    if integral is None:
        return float(edges[y1:y2, x1:x2].mean()) / 255.0
    total = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    return float(total) / ((y2 - y1) * (x2 - x1)) / 255.0


def _score_edges_grid(
    edges: np.ndarray,
    rect: Rect,
    *,
    strengths: list[float] | None = None,
) -> float:
    h, w = edges.shape[:2]
    if rect.x < 0 or rect.y < 0 or rect.right > w or rect.bottom > h:
        return 0.0
    values = _grid_line_strengths(edges, rect) if strengths is None else strengths
    return float(np.clip(np.mean(values) * 3.5, 0.0, 1.0))


def _outer_edge_penalty(
    edges: np.ndarray,
    rect: Rect,
    *,
    strengths: list[float] | None = None,
) -> float:
    """框的左/上再往外一格若还压着一条同样强的网格线，说明整体偏了一格。

    6x6 棋盘正好 7 条网格线，所以「棋盘内部任意 6x6 子区域」也能凑出 7 条线，
    而且周期完全一致——网格分数几乎一样。这正是 1024x576 下棋盘整体右偏一格、
    结束态最右列读不出东西的根因：候选框落在了棋盘内部偏右的位置。

    真正能区分的是「最外圈」：正确框的左/上外侧是界面背景，没有线；
    偏一格的框左/上外侧还压着棋盘自己的边框线。右/下不参与判断——那里紧邻
    颜料与背包面板，竖线本来就多，拿它当证据会把正确的框也判成偏移。
    """
    h, w = edges.shape[:2]
    values = _grid_line_strengths(edges, rect) if strengths is None else strengths
    inside = float(np.mean(values)) if values else 0.0
    if inside <= 1e-6:
        return 1.0

    cell = rect.w / BOARD_SIZE
    worst = 0.0
    # 左：竖线；上：横线。扫描 [0.45, 1.25] 格，容忍几像素的取整误差。
    # 用整列/整行均值再取最大，一次向量化算完——这里每帧都要跑，不能写双层循环。
    lo = max(0, int(rect.x - cell * 1.25))
    hi = min(w - 1, int(rect.x - cell * 0.45))
    if hi > lo:
        worst = float(edges[rect.y : rect.bottom, lo : hi + 1].mean(axis=0).max()) / 255.0
    lo = max(0, int(rect.y - cell * 1.25))
    hi = min(h - 1, int(rect.y - cell * 0.45))
    if hi > lo:
        worst = max(worst, float(edges[lo : hi + 1, rect.x : rect.right].mean(axis=1).max()) / 255.0)

    ratio = worst / inside
    # 实测：正确框的比值 ≤0.20，偏一格的框 ≥1.00，留足余量。
    return float(np.clip(1.0 - (ratio - 0.45) * 0.8, 0.35, 1.0))


def _classify_tiles(
    image: np.ndarray,
    board_rect: Rect,
    *,
    icon_fallback: bool = False,
) -> list[TileObservation]:
    """36 格全量分类。

    ``icon_fallback=True`` 才给官方图标兜底开配额。这个开关**默认关**：
    实时链路（``analyze_realtime``）每帧都可能重跑整盘，而 ORB 单格 26ms、
    整盘就是 0.9s，直接把 0.75s 的交互预算吃穿。图标兜底留给离线场景——
    ``analyze_scene`` 单张截图分析、批量采集、命令行体检——那里慢几百毫秒
    换「不认识也能认出来」完全值得。
    """
    paint_icon.reset_budget(allow=icon_fallback)
    tiles: list[TileObservation] = []
    cell_w = board_rect.w / BOARD_SIZE
    cell_h = board_rect.h / BOARD_SIZE
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x1 = round(board_rect.x + col * cell_w)
            y1 = round(board_rect.y + row * cell_h)
            x2 = round(board_rect.x + (col + 1) * cell_w)
            y2 = round(board_rect.y + (row + 1) * cell_h)
            rect = Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            crop = image[y1:y2, x1:x2]
            tiles.append(
                _classify_tile_observation(crop, row, col, rect, icon_fallback=icon_fallback)
            )
    return _resolve_board_scope_anchor(tiles)


def _classify_tiles_quick(
    image: np.ndarray,
    board_rect: Rect,
    previous_board: BoardObservation,
) -> list[TileObservation]:
    paint_icon.reset_budget(allow=False)
    tiles: list[TileObservation] = []
    cell_w = board_rect.w / BOARD_SIZE
    cell_h = board_rect.h / BOARD_SIZE
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x1 = round(board_rect.x + col * cell_w)
            y1 = round(board_rect.y + row * cell_h)
            x2 = round(board_rect.x + (col + 1) * cell_w)
            y2 = round(board_rect.y + (row + 1) * cell_h)
            rect = Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            crop = image[y1:y2, x1:x2]
            signature = _tile_visual_signature(crop)
            previous = previous_board.tile(row, col)
            previous_signature = _tile_signature_from_features(previous.features)
            visual_change = (
                float(np.mean(np.abs(np.asarray(signature) - np.asarray(previous_signature))))
                if previous_signature is not None
                else 1.0
            )
            fast_state = _classify_tile_for_board_score(crop)
            state_compatible = fast_state in {previous.state, TileState.UNKNOWN}
            visually_identical = visual_change <= 0.012
            stable_limit = 0.075 if previous.state == TileState.PAINT else 0.05
            if visually_identical or (visual_change <= stable_limit and state_compatible):
                features = dict(previous.features)
                _store_tile_visual_signature(features, signature)
                features["visual_change"] = visual_change
                tiles.append(replace(previous, row=row, col=col, rect=rect, features=features))
            else:
                observation = _classify_tile_observation(crop, row, col, rect)
                observation.features["visual_change"] = visual_change
                tiles.append(observation)
    return _resolve_board_scope_anchor(tiles)


def _classify_tile_observation(
    tile_bgr: np.ndarray,
    row: int,
    col: int,
    rect: Rect,
    *,
    icon_fallback: bool = False,
) -> TileObservation:
    state, confidence, features = classify_tile(tile_bgr)
    scope_score, anchor_score = _tile_scope_and_anchor_scores(tile_bgr)
    # 初始颜料的感叹号：静态 UI 标记，不受「金框在不在」影响。
    features["initial_marker_score"] = _tile_initial_marker_score(tile_bgr)
    is_scope_highlight = scope_score >= 0.24 and (
        state == TileState.BLANK or _features_look_like_blank_tile(features)
    )
    is_current_anchor = anchor_score >= 0.10 and state == TileState.PAINT
    features["scope_highlight_score"] = scope_score
    features["current_anchor_score"] = anchor_score
    # The gold frame is an overlay on a still-empty white cell. It must not
    # become paint or contribute to the current rule's match progress.
    if is_scope_highlight and _features_look_like_blank_tile(features):
        state = TileState.BLANK
        confidence = max(confidence, min(0.98, 0.62 + scope_score * 0.50))
    _store_tile_visual_signature(features, _tile_visual_signature(tile_bgr))
    # 棋盘上每枚颜料都会渲染一个 LvN 角标；空白、冰块、漩涡地块都没有。
    # 这是判断「这格到底有没有颜料」最可靠的一条证据（入画阶段的空白地块是纯灰蓝，
    # 没有作画阶段的白色画笔水印，光看颜色会被误判成颜料）。
    label = tile_reader.read_label(tile_bgr)
    features["label_present"] = 1.0 if label.present else 0.0
    features["label_density"] = label.density
    features["label_score"] = label.level_score
    if state in {TileState.PAINT, TileState.UNKNOWN} or label.present:
        (
            paint_category,
            paint_category_confidence,
            paint_level,
            paint_level_confidence,
            paint_subtype,
            paint_is_treasure,
        ) = _analyze_board_paint_tile(tile_bgr, label, icon_fallback=icon_fallback)
    else:
        paint_category = PaintCategory.UNKNOWN
        paint_category_confidence = 0.0
        paint_level = None
        paint_level_confidence = 0.0
        paint_subtype = None
        paint_is_treasure = False
    paint_detail_score = _board_paint_detail_score(
        paint_category,
        paint_category_confidence,
        paint_level,
        paint_level_confidence,
    )
    features["paint_detail_score"] = paint_detail_score
    features["paint_category_confidence"] = paint_category_confidence
    features["paint_level_confidence"] = paint_level_confidence
    if _paint_detail_can_override_tile_state(
        state,
        features,
        paint_detail_score,
        paint_category_confidence,
        paint_level_confidence,
        paint_is_treasure,
    ):
        state = TileState.PAINT
        confidence = max(confidence, min(0.90, paint_detail_score))
    # 角标做最终仲裁：有角标就是颜料；没有角标、连家族/等级都读不出来，就不是颜料。
    #
    # 旧条件只认 ``paint_category == UNKNOWN``，于是「无角标、无家族、无等级，
    # 但颜色被判成 monster 0.78」的深色空板会被当成颜料。实测新一局入画后
    # **18/36 格空板全部变成「怪物」**（v≈0.32、s≈0.36，没有任何图案），
    # 整盘 blank/void 归零 → 「满铺」判据失效 → 收尾后认不出新局，
    # 界面一直挂着上一局的规则与推荐。
    #
    # 判据换成「有没有读出东西」：家族图案和等级任意一项在，就仍是颜料
    # （动画帧会漏读角标，不该因此把已经认出来的颜料降级）；
    # 两者都没有，说明这格提供不了任何颜料信息，颜色像什么都不算数。
    if state in {TileState.PAINT, TileState.UNKNOWN} and not label.present:
        detail_known = paint_subtype is not None or paint_level is not None
        if not detail_known and not paint_is_treasure:
            # 这里要区分「作画阶段带白色画笔水印的可放置空白」和
            # 「入画阶段/不可放置地块的平灰底」——两者都没有角标，但只有前者能落子。
            # 实测：带画笔水印的空白 mean_v≈0.81、white_ratio≈0.07；
            # 平灰底 mean_v≈0.33、white_ratio≈0.0。
            looks_placeable = (
                float(features.get("white_ratio", 0.0)) > 0.03
                and float(features.get("mean_v", 0.0)) > 0.60
            )
            state = TileState.BLANK if looks_placeable else TileState.VOID
            confidence = max(confidence, 0.72)
    elif state in {TileState.BLANK, TileState.UNKNOWN} and label.present:
        if label.density >= tile_reader.PRESENCE_DENSITY_MIN * 1.25:
            state = TileState.PAINT
            confidence = max(confidence, 0.80)
    if state != TileState.PAINT:
        paint_category = PaintCategory.UNKNOWN
        paint_category_confidence = 0.0
        paint_level = None
        paint_level_confidence = 0.0
        paint_subtype = None
        paint_is_treasure = False
    return TileObservation(
        row=row,
        col=col,
        rect=rect,
        state=state,
        confidence=confidence,
        features=features,
        paint_category=paint_category,
        paint_category_confidence=paint_category_confidence,
        paint_level=paint_level,
        paint_level_confidence=paint_level_confidence,
        paint_subtype=paint_subtype,
        paint_is_treasure=paint_is_treasure,
        is_scope_highlight=is_scope_highlight,
        scope_highlight_confidence=scope_score,
        is_current_anchor=is_current_anchor,
        current_anchor_confidence=anchor_score,
    )


def _resolve_board_scope_anchor(tiles: list[TileObservation]) -> list[TileObservation]:
    highlighted = [tile for tile in tiles if tile.is_scope_highlight]
    candidates = [tile for tile in tiles if tile.is_current_anchor]
    if not highlighted or not candidates:
        return [
            replace(tile, is_current_anchor=False, current_anchor_confidence=0.0)
            if tile.is_current_anchor
            else tile
            for tile in tiles
        ]

    rows = {tile.row for tile in highlighted}
    cols = {tile.col for tile in highlighted}
    if len(rows) == 1:
        candidates = [tile for tile in candidates if tile.row in rows]
    elif len(cols) == 1:
        candidates = [tile for tile in candidates if tile.col in cols]
    else:
        candidates = []
    winner = max(candidates, key=lambda tile: tile.current_anchor_confidence) if candidates else None
    return [
        replace(
            tile,
            is_current_anchor=winner is not None and (tile.row, tile.col) == (winner.row, winner.col),
            current_anchor_confidence=(
                tile.current_anchor_confidence
                if winner is not None and (tile.row, tile.col) == (winner.row, winner.col)
                else 0.0
            ),
        )
        if tile.is_current_anchor
        else tile
        for tile in tiles
    ]


#: 「初始颜料」标记（格子左上角的感叹号）的判据区间。
#:
#: 标记形状是**深灰圆底 + 纯白感叹号**，实测 2026-11-15（1239x780，r2c2）：
#: 该区域白色占比 0.185、暗色占比 0.459。同盘其它 35 格分两类，都不在区间内：
#: 浅底格白色 0.42~0.87 且暗色 <0.25，虚空/阻挡格白色 ≈0 且暗色 0.76+。
#: 两个条件必须**同时**成立 —— 单看白色会命中所有浅底格，单看暗色会命中虚空。
INITIAL_MARKER_WHITE = (0.08, 0.40)
INITIAL_MARKER_DARK = (0.25, 0.75)


def _tile_initial_marker_score(tile_bgr: np.ndarray) -> float:
    """左上角有没有那颗感叹号：有则 1.0，没有则 0.0。

    与金色高亮（``_tile_scope_and_anchor_scores``）无关：感叹号是**静态 UI 标记**，
    新画纸预置的初始颜料一直带着它；金框只在选中作用范围时出现。
    2026-11-15 实机：金框消失的那一帧 scope/anchor 都归零，感叹号照旧在。
    """
    if tile_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    marker_height = max(1, int(height * 0.32))
    marker_width = max(1, int(width * 0.28))
    marker = hsv[:marker_height, :marker_width]
    value = marker[:, :, 2]
    saturation = marker[:, :, 1]
    white = float(np.mean((saturation < 80) & (value > 180)))
    white_lo, white_hi = INITIAL_MARKER_WHITE
    if not white_lo <= white <= white_hi:
        return 0.0
    dark = float(np.mean(value < 95))
    dark_lo, dark_hi = INITIAL_MARKER_DARK
    if not dark_lo <= dark <= dark_hi:
        return 0.0
    return 1.0


def _tile_scope_and_anchor_scores(tile_bgr: np.ndarray) -> tuple[float, float]:
    if tile_bgr.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    gold = (hue >= 12) & (hue <= 45) & (saturation > 75) & (value > 120)

    height, width = gold.shape
    band = max(3, min(height, width) // 14)
    perimeter = np.zeros_like(gold, dtype=bool)
    perimeter[:band, :] = True
    perimeter[-band:, :] = True
    perimeter[:, :band] = True
    perimeter[:, -band:] = True
    scope_score = float(np.mean(gold[perimeter])) if np.any(perimeter) else 0.0

    marker_height = max(1, int(height * 0.32))
    marker_width = max(1, int(width * 0.28))
    marker = gold[:marker_height, :marker_width]
    marker_dark = value[:marker_height, :marker_width] < 70
    marker_gold_ratio = float(np.mean(marker)) if marker.size else 0.0
    marker_dark_ratio = float(np.mean(marker_dark)) if marker_dark.size else 0.0
    anchor_score = marker_gold_ratio if marker_dark_ratio >= 0.018 else 0.0
    return scope_score, anchor_score


def _tile_visual_signature(tile_bgr: np.ndarray) -> tuple[float, ...]:
    if tile_bgr.size == 0:
        return (0.0,) * 48
    small = cv2.resize(tile_bgr, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return tuple(float(value) for value in small.reshape(-1))


def _store_tile_visual_signature(features: dict[str, float], signature: tuple[float, ...]) -> None:
    for index, value in enumerate(signature):
        features[f"visual_{index}"] = value


def _tile_signature_from_features(features: dict[str, float]) -> tuple[float, ...] | None:
    values: list[float] = []
    for index in range(48):
        key = f"visual_{index}"
        if key not in features:
            return None
        values.append(float(features[key]))
    return tuple(values)


#: 外框本身的金色占比就超过阈值的等级。
#:
#: Lv5 是橙色菱形框（``runtime/s8_frames/diamond_Lv5_橙.png`` 实测 0.232），
#: 一个**没有任何图案**的空 Lv5 框都会被判成宝箱、置信度还顶到 0.85。
#: 也就是说这一级颜色判据完全不具分辨力，直接不采信。
TREASURE_COLOR_UNTRUSTED_LEVELS = frozenset({5})

#: 判成怪物后还允许颜色翻成宝箱的最低置信度。与背包面板那条路径的守卫
#: 同值同义（见 :func:`_read_backpack_slot`），两条路径不能各自一套标准。
TREASURE_MONSTER_GUARD_CONFIDENCE = 0.55


def _color_treasure_allowed(
    level: int | None,
    category: "PaintCategory",
    category_confidence: float,
) -> bool:
    """颜色兜底判宝箱的最后两道闸。

    只有「模板库」和「官方 27 图标」**都没认出**这个格子时才会走到这里。
    颜色本身分不开宝箱和金色家族（英灵比宝箱还金），所以只排除两种
    "绝不可能是宝箱"的情形：

    * 该等级的外框自己就够金（Lv5 橙框），颜色没有分辨力；
    * 等级读不出来——外框占比无从扣除，这时候颜色说什么都不能信；
    * 已经明确判成怪物——升级不会把怪物变成宝箱。
    """
    if level is None:
        return False
    if level in TREASURE_COLOR_UNTRUSTED_LEVELS:
        return False
    if category == PaintCategory.MONSTER and category_confidence >= TREASURE_MONSTER_GUARD_CONFIDENCE:
        return False
    return True


def _analyze_board_paint_tile(
    tile_bgr: np.ndarray,
    label: "tile_reader.LabelReading | None" = None,
    *,
    icon_fallback: bool = False,
) -> tuple[PaintCategory, float, int | None, float, str | None, bool]:
    if tile_bgr.size == 0:
        return PaintCategory.UNKNOWN, 0.0, None, 0.0, None, False

    category, category_confidence = _classify_tile_paint_frame(tile_bgr)
    level, level_confidence = _detect_board_tile_level(tile_bgr, label)

    # 放进虚空的颜料：图案被整个抹掉，只剩一圈彩色框（``斜对角.png`` r3c1
    # 就是 Lv3 蓝环 + 空中心）。这类格子必须**先**认出来，否则模板库会
    # 拿它去匹配某个家族——历史上它就以 0.769 分匹配成「邪影」，把整个家族带偏。
    #
    # 注意框色代表的是**初始颜料的等级**，不是这格颜料自己的等级：初始颜料升到
    # Lv5，虚空格的框就跟着变橙。所以这里照读 level（它确实是界面上显示的东西），
    # 但不给它挂家族，类别留 unknown——图案已经没了，算效果时不能按家族算。
    # 判据只看中心平不平（拉普拉斯方差 25.6 vs 真图案最低 677.8），并强制要求
    # 有 LvN 角标：空白格和阻挡地块的中心一样平，但它们没有角标。
    if label is not None and label.present and paint_frame.looks_voided(tile_bgr):
        return PaintCategory.UNKNOWN, 0.0, level, level_confidence, paint_frame.VOIDED, False

    is_treasure, treasure_confidence = _detect_treasure_icon(tile_bgr)
    subtype = None
    # 种类（家族）靠中央图案认，和等级完全正交——同一个家族在 7 个等级里
    # 画的是同一张图，只是外框颜色变。认对了种类，类别也顺带确定：
    # 「鼠人」只可能是怪物颜料，「机械之神」只可能是玩法颜料。
    #
    # 顺序必须是「模板 → 官方图标 → 颜色」三层，颜色垫底。宝箱本身就是一个
    # 正常家族（官方 ColorPlay1），而「金色占比高」分不开宝箱和金色家族：
    # 实测官方图标里英灵 0.288、机械之神 0.133，都不低于真宝箱 0.266 多少。
    # 更致命的是**等级外框**——Lv5 的橙色菱形框单独就有 0.232 的金色占比，
    # 一个没有任何图案的空 Lv5 框都会被判成宝箱（conf 顶格 0.85）。
    # 2026-11-12 的实机记录就是这么错的：r1c1「玩法 Lv5」被翻成「玩法·宝箱」、
    # 升到 Lv6（红框，金色占比 0.000）又"变回"玩法；r3c4 甚至从「怪物 Lv4」
    # 被硬翻成「玩法·宝箱 Lv5」。所以官方 27 图标的兜底必须排在颜色之前，
    # 颜色只留作最后一道弱证据，并且要扣掉等级外框基线。
    kind = paint_kind.read_kind(tile_bgr)
    if kind.present and kind.subtype:
        subtype = kind.subtype
        known_category = paint_kind.category_of(kind.subtype)
        if known_category == "monster":
            category = PaintCategory.MONSTER
            category_confidence = max(category_confidence, kind.score)
        elif known_category == "gameplay":
            category = PaintCategory.GAMEPLAY
            category_confidence = max(category_confidence, kind.score)
        is_treasure = subtype == "宝箱"
    else:
        # 模板库只认识「见过并入库」的家族；剩下 11 个家族一直靠实机截图慢慢攒。
        # 现在有 27 张官方图标，模板认不出时可以直接对着标准照认人（见 paint_icon）。
        # 实测 56 个「模板也认对了」的格子上，图标与模板零冲突，所以这层兜底是
        # 加在模板之后的补充，不是替代。
        match = (
            paint_icon.identify(tile_bgr, budgeted=True)
            if icon_fallback and paint_kind.looks_like_paint_tile(tile_bgr)
            else None
        )
        if match is not None:
            subtype = match.family
            known_category = paint_kind.category_of(match.family)
            if known_category == "monster":
                category = PaintCategory.MONSTER
            elif known_category == "gameplay":
                category = PaintCategory.GAMEPLAY
            category_confidence = max(category_confidence, ICON_FALLBACK_CONFIDENCE)
            is_treasure = subtype == "宝箱"
        elif is_treasure and _color_treasure_allowed(level, category, category_confidence):
            # 图案两层都没认出来，只剩颜色。这里必须挡住两种情况：
            # Lv5 的橙色外框（基线 0.23，任何格子都会过线），以及已经明确
            # 判成怪物的格子（怪物不可能变宝箱，升级也不改类别）。
            subtype = "宝箱"
            category = PaintCategory.GAMEPLAY
            category_confidence = max(category_confidence, treasure_confidence)
        else:
            # 颜色说像宝箱但被守卫拦下：不挂"宝箱"，也别谎报真值。
            is_treasure = False
    return category, category_confidence, level, level_confidence, subtype, is_treasure


def _detect_board_tile_level(
    tile_bgr: np.ndarray,
    label: "tile_reader.LabelReading | None" = None,
) -> tuple[int | None, float]:
    """读取棋盘格子的颜料等级。

    两个来源，优先级明确：

    1. **菱形边框颜色**（``frame_level``）。等级 → 边框色是固定的：
       银=Lv1、绿=Lv2、青蓝=Lv3、紫=Lv4、橙=Lv5、红=Lv6。
       32 个带真值的格子实测 **100%**，而且不需要模板库、不受图标内容干扰。
    2. **右下角 ``LvN`` 角标**（``tile_reader``）。留一法只有 63.6%，
       降级为交叉校验和兜底（比如 Lv7 边框色还没样本时）。

    没角标 = 没颜料，直接返回 ``None``，不猜。
    **不再用「边框颜色反推背包等级」那条路**——它在实机上把 Lv3 读成 5、Lv5 读成 1。
    """
    if tile_bgr.size == 0:
        return None, 0.0
    reading = label if label is not None else tile_reader.read_label(tile_bgr)
    if not reading.present:
        return None, 0.0

    frame = frame_level.read_frame_level(tile_bgr)
    # 银框不是「正向证据」，只是「带内没扫到彩色」的兜底。历史上 Lv7 的彩虹环
    # 也落进银框判据，靠这里的「角标等级分差」翻案；现在 frame_level 能直接认出
    # Lv7（圆形彩虹环在菱形带内侧只漏一点饱和），这条路留作双保险：
    # 真 Lv1 的 Lv1 分是 1.00、别的等级 0.84；Lv7 宝箱的 Lv1 只有 0.73。
    silver_fallback = (
        frame.level == 1
        and frame.mean_saturation < frame_level.SILVER_MEAN_SAT_MAX
        and frame.sat_ratio < frame_level.SILVER_SAT_RATIO_MAX
    )
    if silver_fallback and reading.level not in (None, 1):
        margin = reading.level_score - reading.scores.get(1, 0.0)
        if margin >= SILVER_LABEL_MARGIN:
            return reading.level, reading.level_score
    # 角标自己清晰读出 7（模板分 ≥ LEVEL_TRUST_SCORE）就采信，**不看边框**。
    #
    # 7 是最高等级，边框是动画彩虹环：2026-11-18 那局 r2c3 黑帆真实 Lv7，
    # 逐帧边框读成 6 → 4 → 7 → 5 → 1（彩虹转起来时主色相偏橙就判 Lv5、
    # 饱和掉下去就判银框 Lv1），只有一帧读到 7。角标是像素级模板匹配，
    # Lv1~Lv6 的角标读出 7 的分数只有 0.48 左右，碰不到 0.95 这道门槛。
    if reading.level == 7 and reading.level_score >= tile_reader.LEVEL_TRUST_SCORE:
        return 7, reading.level_score
    if frame.level is not None and frame.confidence >= 0.70:
        confidence = frame.confidence
        # 角标和边框色打架时留个痕迹，并把置信度压下来（边框色更可信，仍然采用它）。
        if reading.level is not None and reading.level != frame.level:
            confidence = min(confidence, 0.70)
        return frame.level, confidence

    if reading.level is not None:
        # 角标确实存在但数字没把握。仍然把最佳猜测交出去，置信度压到一半，
        # 让它带着「低置信」的标记进入模型，而不是让这枚颜料直接从棋盘上消失。
        if reading.level_score >= tile_reader.LEVEL_TRUST_SCORE:
            return reading.level, reading.level_score
        return reading.level, reading.level_score * 0.5
    return None, 0.0


def _board_paint_detail_score(
    category: PaintCategory,
    category_confidence: float,
    level: int | None,
    level_confidence: float,
) -> float:
    if category == PaintCategory.UNKNOWN:
        return 0.0
    if level is None:
        return category_confidence * 0.72
    return float(np.clip(category_confidence * 0.58 + level_confidence * 0.42, 0.0, 1.0))


def _features_look_like_blank_tile(features: dict[str, float]) -> bool:
    return (
        float(features.get("bright_ratio", 0.0)) > 0.45
        and float(features.get("low_sat_ratio", 0.0)) > 0.44
        and float(features.get("saturated_bright_ratio", 0.0)) < 0.12
    ) or (
        float(features.get("white_ratio", 0.0)) > 0.52
        and float(features.get("mean_s", 0.0)) < 0.24
        and float(features.get("non_blue_saturated_bright_ratio", 0.0)) < 0.08
    )


def _features_look_like_void_tile(features: dict[str, float]) -> bool:
    return (
        float(features.get("very_dark_ratio", 0.0)) > 0.34
        and float(features.get("mean_v", 0.0)) < 0.33
    ) or (
        float(features.get("dark_ratio", 0.0)) > 0.62
        and float(features.get("mean_v", 0.0)) < 0.38
    ) or (
        float(features.get("center_dark", 0.0)) > 0.88
        and float(features.get("mean_v", 0.0)) < 0.30
    )


def _paint_detail_can_override_tile_state(
    state: TileState,
    features: dict[str, float],
    paint_detail_score: float,
    category_confidence: float,
    level_confidence: float,
    is_treasure: bool,
) -> bool:
    if state == TileState.PAINT:
        return False
    if paint_detail_score < 0.56:
        return False

    blank_like = _features_look_like_blank_tile(features)
    void_like = _features_look_like_void_tile(features)
    ice_like = bool(features.get("blocked_ice_like", 0.0) >= 0.5)
    if state in {TileState.BLANK, TileState.VOID, TileState.BLOCKED, TileState.FADED} and (blank_like or void_like or ice_like):
        return False
    if state == TileState.UNKNOWN:
        return category_confidence >= 0.60 and paint_detail_score >= 0.60
    if is_treasure and category_confidence >= 0.70:
        return True

    frame_score = float(features.get("paint_frame_score", 0.0))
    saturated = float(features.get("saturated_bright_ratio", 0.0))
    non_blue = float(features.get("non_blue_saturated_bright_ratio", 0.0))
    purple = float(features.get("purple_ratio", 0.0))
    gold = float(features.get("gold_ratio", 0.0))
    return (
        category_confidence >= 0.64
        and level_confidence >= 0.68
        and paint_detail_score >= 0.66
        and (
            frame_score >= 0.66
            or saturated >= 0.28
            or non_blue >= 0.10
            or purple >= 0.06
            or gold >= 0.04
        )
    )


def classify_tile(tile_bgr: np.ndarray) -> tuple[TileState, float, dict[str, float]]:
    if tile_bgr.size == 0:
        return TileState.UNKNOWN, 0.0, {}

    pad_y = max(2, tile_bgr.shape[0] // 12)
    pad_x = max(2, tile_bgr.shape[1] // 12)
    core = tile_bgr[pad_y : tile_bgr.shape[0] - pad_y, pad_x : tile_bgr.shape[1] - pad_x]
    if core.size == 0:
        core = tile_bgr

    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mean_v = float(np.mean(v)) / 255.0
    mean_s = float(np.mean(s)) / 255.0
    dark_ratio = float(np.mean(v < 65))
    very_dark_ratio = float(np.mean(v < 38))
    bright_ratio = float(np.mean(v > 150))
    low_sat_ratio = float(np.mean(s < 45))
    blue_ratio = float(np.mean(((h >= 82) & (h <= 112) & (s > 70) & (v > 80))))
    cyan_ratio = float(np.mean(((h >= 75) & (h <= 100) & (s > 95) & (v > 120))))
    saturated_bright_ratio = float(np.mean((s > 70) & (v > 120)))
    non_blue_saturated_bright_ratio = float(
        np.mean((s > 70) & (v > 120) & ~((h >= 75) & (h <= 112)))
    )
    purple_ratio = float(np.mean((h >= 120) & (h <= 170) & (s > 70) & (v > 90)))
    gold_ratio = float(np.mean((h >= 15) & (h <= 45) & (s > 70) & (v > 110)))
    gray_mid_ratio = float(np.mean((s < 55) & (v >= 55) & (v <= 145)))
    white_ratio = float(np.mean((s < 45) & (v > 170)))

    center = core[
        core.shape[0] // 4 : core.shape[0] * 3 // 4,
        core.shape[1] // 4 : core.shape[1] * 3 // 4,
    ]
    if center.size == 0:
        center = core
    center_hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    center_dark = float(np.mean(center_hsv[:, :, 2] < 70))
    center_blue = float(
        np.mean(
            (center_hsv[:, :, 0] >= 75)
            & (center_hsv[:, :, 0] <= 112)
            & (center_hsv[:, :, 1] > 80)
            & (center_hsv[:, :, 2] > 80)
        )
    )

    features = {
        "mean_v": mean_v,
        "mean_s": mean_s,
        "dark_ratio": dark_ratio,
        "very_dark_ratio": very_dark_ratio,
        "bright_ratio": bright_ratio,
        "low_sat_ratio": low_sat_ratio,
        "blue_ratio": blue_ratio,
        "cyan_ratio": cyan_ratio,
        "saturated_bright_ratio": saturated_bright_ratio,
        "non_blue_saturated_bright_ratio": non_blue_saturated_bright_ratio,
        "purple_ratio": purple_ratio,
        "gold_ratio": gold_ratio,
        "gray_mid_ratio": gray_mid_ratio,
        "white_ratio": white_ratio,
        "center_dark": center_dark,
        "center_blue": center_blue,
    }

    frame_category, frame_confidence = _classify_tile_paint_frame(tile_bgr)
    blocked_ice_like = _tile_looks_like_blocked_ice(
        cyan_ratio=cyan_ratio,
        blue_ratio=blue_ratio,
        center_blue=center_blue,
        mean_s=mean_s,
        low_sat_ratio=low_sat_ratio,
        white_ratio=white_ratio,
    )
    features["paint_frame_score"] = frame_confidence
    features["blocked_ice_like"] = 1.0 if blocked_ice_like else 0.0

    if very_dark_ratio > 0.34 and mean_v < 0.33:
        return TileState.VOID, _conf(very_dark_ratio, 0.34, 0.75), features

    if dark_ratio > 0.62 and mean_v < 0.38:
        return TileState.VOID, _conf(dark_ratio, 0.62, 0.90), features

    if bright_ratio > 0.45 and low_sat_ratio > 0.44:
        return TileState.BLANK, _conf(bright_ratio + low_sat_ratio, 0.90, 1.65), features

    if blocked_ice_like:
        return TileState.BLOCKED, _conf(cyan_ratio + center_blue + mean_s, 1.65, 2.55), features

    # 纯深蓝空板（棋盘底色/不可放置的空洞）：整格几乎同一块蓝，没有任何亮部。
    # 背包「框形状」分类器会把它当成怪物框（实测 0.78，正好越过下面那道 0.62），
    # 而棋盘上真正的颜料**一定有亮部**——图案或 Lv 角标，
    # 实测最低的真颜料 bright_ratio≥0.45，空板则是 0.00（v≈0.32、s≈0.36）。
    # 不拦的话：新一局入画后 17/36 格深色空板全被认成「怪物 Lv?」，
    # 整盘 blank/void 归零 → 「满铺」判据失效 → 收尾后认不出新局。
    if (
        blue_ratio > 0.90
        and center_blue > 0.88
        and bright_ratio < 0.05
        and white_ratio < 0.03
        and mean_v < 0.45
    ):
        return TileState.VOID, _conf(blue_ratio, 0.90, 1.00), features

    # Board paints can be blue/green and look close to ice by color alone, but
    # blank shovel highlights also create weak diamond-like lines. Only trust
    # frame evidence after strong terrain checks have had the first say.
    if frame_category != PaintCategory.UNKNOWN and frame_confidence >= 0.62:
        return TileState.PAINT, max(0.60, min(0.92, frame_confidence)), features

    if center_dark > 0.18 and (cyan_ratio > 0.06 or mean_s > 0.26):
        return TileState.PAINT, _conf(center_dark + mean_s, 0.42, 0.95), features

    if (
        non_blue_saturated_bright_ratio > 0.22
        and saturated_bright_ratio > 0.30
        and (purple_ratio > 0.08 or gold_ratio > 0.06 or mean_s > 0.34)
    ):
        return TileState.PAINT, _conf(non_blue_saturated_bright_ratio, 0.22, 0.62), features

    if gray_mid_ratio > 0.40 and 0.22 < mean_v < 0.62 and bright_ratio < 0.42:
        return TileState.FADED, _conf(gray_mid_ratio, 0.40, 0.85), features

    if gray_mid_ratio > 0.28 and mean_s < 0.32 and dark_ratio < 0.18 and bright_ratio < 0.38:
        return TileState.FADED, _conf(gray_mid_ratio, 0.28, 0.62), features

    if blue_ratio > 0.72 and center_blue > 0.70 and mean_s > 0.58 and low_sat_ratio < 0.12:
        return TileState.BLOCKED, _conf(blue_ratio, 0.10, 0.45), features

    if dark_ratio > 0.35:
        return TileState.VOID, _conf(dark_ratio, 0.35, 0.75), features

    return TileState.UNKNOWN, 0.35, features


def _classify_tile_paint_frame(tile_bgr: np.ndarray) -> tuple[PaintCategory, float]:
    if tile_bgr.size == 0:
        return PaintCategory.UNKNOWN, 0.0
    category, confidence = _classify_backpack_frame_shape(tile_bgr)
    if category == PaintCategory.UNKNOWN:
        return PaintCategory.UNKNOWN, 0.0
    return category, confidence


def _tile_looks_like_blocked_ice(
    *,
    cyan_ratio: float,
    blue_ratio: float,
    center_blue: float,
    mean_s: float,
    low_sat_ratio: float,
    white_ratio: float,
) -> bool:
    # True ice blocks cover nearly the whole cell with saturated cyan/blue.
    # Blue gameplay paints have a visible blue frame, but much lower cyan and
    # center coverage, plus more internal art/label variation.
    if cyan_ratio >= 0.42 and center_blue >= 0.72 and mean_s >= 0.62:
        return True
    return blue_ratio >= 0.86 and center_blue >= 0.82 and mean_s >= 0.66 and low_sat_ratio <= 0.08 and white_ratio <= 0.08


def _conf(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0.45, 0.98))


def _expand_to_square(rect: Rect, width: int, height: int) -> Rect:
    side = max(rect.w, rect.h)
    cx = rect.x + rect.w / 2
    cy = rect.y + rect.h / 2
    x = int(round(cx - side / 2))
    y = int(round(cy - side / 2))
    x = max(0, min(x, width - side))
    y = max(0, min(y, height - side))
    return Rect(x, y, side, side)


def _bounded_rect(x: float, y: float, w: float, h: float, width: int, height: int) -> Rect:
    left = _clamp_int(x, 0, max(0, width - 1))
    top = _clamp_int(y, 0, max(0, height - 1))
    right = _clamp_int(x + w, left + 1, width)
    bottom = _clamp_int(y + h, top + 1, height)
    return Rect(left, top, right - left, bottom - top)


def _clamp_int(value: float, min_value: int, max_value: int) -> int:
    if max_value < min_value:
        return min_value
    return int(max(min_value, min(round(value), max_value)))


def _dedupe_rects(rects: list[Rect], image: np.ndarray | None = None) -> list[Rect]:
    if image is not None:
        rects = sorted(rects, key=lambda r: _score_board_rect(image, r), reverse=True)
    else:
        rects = sorted(rects, key=lambda r: r.w * r.h, reverse=True)
    kept: list[Rect] = []
    for rect in rects:
        if all(_iou(rect, other) < 0.72 for other in kept):
            kept.append(rect)
    return kept[:80]


def _iou(a: Rect, b: Rect) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    if union <= 0:
        return 0.0
    return inter / union
