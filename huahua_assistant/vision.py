from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .capture import write_image
from .game_model import BRUSH_TYPE_ORDER, BrushType, build_affix_catalog
from .keyword_match import match_keywords
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


BOARD_SIZE = 6


class VisionTracker:
    def __init__(self) -> None:
        self.board_rect: Rect | None = None
        self.last_scene: SceneObservation | None = None

    def analyze(self, image: np.ndarray) -> SceneObservation | None:
        scene = analyze_scene(image, previous_board_rect=self.board_rect)
        if scene is not None:
            self.board_rect = scene.board.rect
            self.last_scene = scene
        return scene

    def analyze_quick(self, image: np.ndarray) -> SceneObservation | None:
        scene = analyze_scene_quick(image, previous_scene=self.last_scene, previous_board_rect=self.board_rect)
        if scene is not None:
            self.board_rect = scene.board.rect
            self.last_scene = scene
        return scene

    def reset(self) -> None:
        self.board_rect = None
        self.last_scene = None


def analyze_frame(image: np.ndarray) -> BoardObservation | None:
    scene = analyze_scene(image)
    return scene.board if scene is not None else None


def analyze_scene(image: np.ndarray, previous_board_rect: Rect | None = None) -> SceneObservation | None:
    if previous_board_rect is not None:
        scene = _analyze_scene_from_board_rect(image, previous_board_rect, fast=True)
        # A wrong-but-grid-like candidate can score well if it lands on the
        # backpack/debug grid. Only reuse the tracked board when it is still a
        # strong match; otherwise do a full search and recover.
        if scene is not None and scene.board.score >= 0.62:
            return scene

    layout = _estimated_board_layout_candidate(image)
    if layout is not None:
        refined_layout = _refine_board_rect(image, layout)
        layout_score = _score_board_rect(image, refined_layout)
        if layout_score >= 0.72:
            return _analyze_scene_from_board_rect(image, refined_layout, fast=False)

    rects = _candidate_board_rects(image)
    best: tuple[float, Rect] | None = None
    for rect in rects:
        score = _score_board_rect(image, rect)
        if best is None or score > best[0]:
            best = (score, rect)

    if best is None or best[0] < 0.20:
        return None

    rect = _choose_board_rect(image, best[1], best[0])
    return _analyze_scene_from_board_rect(image, rect, fast=False)


def _choose_board_rect(image: np.ndarray, best_rect: Rect, best_score: float) -> Rect:
    refined_best = _refine_board_rect(image, best_rect)
    refined_best_score = _score_board_rect(image, refined_best)
    layout = _estimated_board_layout_candidate(image)
    if layout is None:
        return refined_best
    refined_layout = _refine_board_rect(image, layout)
    layout_score = _score_board_rect(image, refined_layout)
    if layout_score >= max(best_score, refined_best_score) * 0.94:
        return refined_layout
    return refined_best


def analyze_scene_quick(
    image: np.ndarray,
    previous_scene: SceneObservation | None = None,
    previous_board_rect: Rect | None = None,
) -> SceneObservation | None:
    rect = previous_scene.board.rect if previous_scene is not None else previous_board_rect
    if rect is None:
        return analyze_scene(image)
    return _analyze_scene_from_board_rect(image, rect, fast=True, quick=True, previous_scene=previous_scene)


def _analyze_scene_from_board_rect(
    image: np.ndarray,
    rect: Rect,
    fast: bool,
    *,
    quick: bool = False,
    previous_scene: SceneObservation | None = None,
) -> SceneObservation | None:
    if not fast:
        rect = _refine_board_rect(image, rect)
    rect = _bounded_rect(rect.x, rect.y, rect.w, rect.h, image.shape[1], image.shape[0])
    score = _score_board_rect(image, rect)
    if score < 0.20:
        return None
    tiles = tuple(_classify_tiles(image, rect))
    board = BoardObservation(rect=rect, score=score, tiles=tiles)
    regions = tuple(_estimate_regions(image, board))
    if quick:
        current_paint = _analyze_current_paint_fast(image, regions, previous_scene.current_paint if previous_scene is not None else None)
        current_paint = _correct_current_paint_from_board_anchor(image, current_paint, tiles)
        backpack = _analyze_backpack_fast(image, regions, previous_scene.backpack if previous_scene is not None else None)
        return SceneObservation(
            board=board,
            regions=regions,
            current_paint=current_paint,
            brush_options=previous_scene.brush_options if previous_scene is not None else None,
            backpack=backpack,
        )
    current_paint = _analyze_current_paint(image, regions)
    current_paint = _correct_current_paint_from_board_anchor(image, current_paint, tiles)
    brush_options = _analyze_brush_options(image, regions)
    backpack = _analyze_backpack(image, regions)
    return SceneObservation(
        board=board,
        regions=regions,
        current_paint=current_paint,
        brush_options=brush_options,
        backpack=backpack,
    )


def _correct_current_paint_from_board_anchor(
    image: np.ndarray,
    current_paint: CurrentPaintObservation | None,
    tiles: tuple[TileObservation, ...],
) -> CurrentPaintObservation | None:
    if current_paint is None:
        return None
    if current_paint.category != PaintCategory.UNKNOWN and current_paint.category_confidence >= 0.58:
        return current_paint
    paint_tiles = [tile for tile in tiles if tile.state == TileState.PAINT]
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
    )


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
) -> tuple[SceneObservation | None, Path | None]:
    scene = tracker.analyze(image) if tracker is not None else analyze_scene(image)
    if save_crops:
        _save_current_paint_crops(image, scene)
        _save_brush_options_crops(image, scene)
        _save_backpack_slot_crops(image, scene)
    if debug_path is None:
        return scene, None
    debug = make_debug_image(image, scene)
    return scene, write_image(debug_path, debug)


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
    keyword_scores = _current_paint_keyword_scores(text_crop)
    match_current, match_required = _infer_match_count(keyword_scores)
    if match_required is None:
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
    raw_text_hint = _build_raw_text_hint(
        category,
        level,
        match_current,
        match_required,
        target_object,
        direction,
        upgrade_amount,
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
    )


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
    if level is None or level_confidence < 0.38:
        text_crop = image[text.y : text.bottom, text.x : text.right]
        hint_level, hint_confidence = _detect_level_hint(text_crop)
        if hint_level is not None and hint_confidence > level_confidence:
            level, level_confidence = hint_level, hint_confidence

    if previous is not None:
        if category == PaintCategory.UNKNOWN or category_confidence < 0.46:
            category = previous.category
            category_confidence = max(0.42, previous.category_confidence * 0.72)
        if level is None or level_confidence < 0.40:
            level = previous.level
            level_confidence = max(0.36, previous.level_confidence * 0.72)
        target_object = previous.target_object
        direction = previous.direction
        match_current = previous.match_current
        match_required = previous.match_required
        upgrade_amount = previous.upgrade_amount
        subtype = previous.subtype
        is_treasure = previous.is_treasure
    else:
        target_object = PaintObject.UNKNOWN
        direction = PaintDirection.UNKNOWN
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

    raw_text_hint = _build_raw_text_hint(
        category,
        level,
        match_current,
        match_required,
        target_object,
        direction,
        upgrade_amount,
    )
    if previous is not None:
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
        keyword_scores={},
        raw_text_hint=raw_text_hint,
    )


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
    return scores


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
    mapping = {
        "横向": PaintDirection.HORIZONTAL,
        "纵向": PaintDirection.VERTICAL,
        "对角斜向": PaintDirection.DIAGONAL,
        "相邻": PaintDirection.ADJACENT,
        "全图": PaintDirection.GLOBAL,
    }
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


def _analyze_backpack(
    image: np.ndarray,
    regions: tuple[RegionObservation, ...],
) -> BackpackObservation | None:
    region = next((item for item in regions if item.kind == RegionKind.BACKPACK), None)
    if region is None:
        return None

    panel = region.rect
    slots = _locate_backpack_slots(image, panel)
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
    paints = tuple(
        observation
        for slot in slots
        if (observation := _analyze_backpack_slot_fast(image, slot)) is not None
    )
    if paints:
        confidence = float(np.mean([paint.category_confidence * 0.55 + paint.level_confidence * 0.45 for paint in paints]))
        return BackpackObservation(panel_rect=panel, paints=paints, confidence=float(np.clip(confidence, 0.0, 1.0)))
    return None


def _locate_backpack_slots_fast(
    image: np.ndarray,
    panel: Rect,
    previous: BackpackObservation | None = None,
) -> tuple[Rect, ...]:
    if previous is not None and previous.paints:
        slots = tuple(
            _bounded_rect(
                paint.slot_rect.x,
                paint.slot_rect.y,
                paint.slot_rect.w,
                paint.slot_rect.h,
                image.shape[1],
                image.shape[0],
            )
            for paint in previous.paints
        )
        visible_slots = tuple(slot for slot in slots if _slot_has_icon(image, slot))
        if visible_slots:
            return visible_slots
    return _locate_backpack_slots(image, panel)


def _analyze_backpack_slot_fast(image: np.ndarray, slot: Rect) -> BackpackPaintObservation | None:
    crop = image[slot.y : slot.bottom, slot.x : slot.right]
    if crop.size == 0 or not _slot_has_icon(image, slot):
        return None

    category, category_confidence = _classify_backpack_category(crop)
    level, level_confidence = _detect_backpack_level_fast(crop)
    is_treasure, treasure_confidence = _detect_treasure_icon(crop)
    subtype = "宝箱" if is_treasure else None
    if is_treasure:
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
        fingerprint=_backpack_slot_fingerprint(crop),
    )


def _detect_backpack_level_fast(crop: np.ndarray) -> tuple[int, float]:
    if crop.size == 0:
        return 1, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color_level, color_confidence = _detect_backpack_level_by_frame_color(hsv)
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


def _analyze_backpack_slot(image: np.ndarray, slot: Rect) -> BackpackPaintObservation | None:
    crop = image[slot.y : slot.bottom, slot.x : slot.right]
    if crop.size == 0 or not _slot_has_icon(image, slot):
        return None

    category, category_confidence = _classify_backpack_category(crop)
    level, level_confidence = _detect_backpack_level(crop)
    is_treasure, treasure_confidence = _detect_treasure_icon(crop)
    subtype = "宝箱" if is_treasure else None
    if is_treasure:
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
    )


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
        threshold=14,
        minLineLength=max(12, int(min(roi.shape[:2]) * 0.22)),
        maxLineGap=5,
    )
    diagonal_length = 0.0
    axial_length = 0.0
    if lines is not None:
        for line in lines[:, 0, :]:
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
    if diagonal_score >= 2.8 and diagonal_score >= axial_score * 0.72:
        return PaintCategory.GAMEPLAY, float(np.clip(diagonal_score / 10.0, 0.64, 0.92))
    if axial_score >= 2.4 and axial_score > diagonal_score * 1.10:
        return PaintCategory.MONSTER, float(np.clip(axial_score / 8.0, 0.58, 0.86))

    return PaintCategory.UNKNOWN, 0.0


def _detect_backpack_level(crop: np.ndarray) -> tuple[int, float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Prefer the bottom-right label when visible.
    label = crop[int(crop.shape[0] * 0.56) :, int(crop.shape[1] * 0.54) :]
    digit = _detect_level_digit_from_label(label)
    if digit is not None:
        color_level, color_confidence = _detect_backpack_level_by_frame_color(hsv)
        if color_level == digit:
            return digit, max(0.78, color_confidence)
        return digit, 0.74

    color_level, color_confidence = _detect_backpack_level_by_frame_color(hsv)
    if color_level is not None and color_confidence >= 0.42:
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


def _detect_backpack_level_by_frame_color(hsv: np.ndarray) -> tuple[int | None, float]:
    if hsv.size == 0:
        return None, 0.0
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    height, width = h.shape[:2]
    yy, xx = np.indices(h.shape)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    xnorm = np.abs(xx - cx) / max(1.0, width * 0.43)
    ynorm = np.abs(yy - cy) / max(1.0, height * 0.43)
    diamond_frame = np.abs(xnorm + ynorm - 1.0) < 0.050
    circle_radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(1.0, min(width, height) * 0.40)
    circle_frame = np.abs(circle_radius - 1.0) < 0.050
    label_area = (xx > width * 0.56) & (yy > height * 0.58)
    frame = (diamond_frame | circle_frame) & ~label_area & (v > 45)
    if np.mean(frame) < 0.012:
        return None, 0.0

    total = max(1, int(np.sum(frame)))
    ratios = {
        1: float(np.sum(frame & (s < 78) & (v > 70)) / total),
        2: float(np.sum(frame & (h >= 45) & (h <= 85) & (s > 50) & (v > 60)) / total),
        3: float(np.sum(frame & (h >= 88) & (h <= 118) & (s > 50) & (v > 60)) / total),
        4: float(np.sum(frame & (h >= 120) & (h <= 165) & (s > 45) & (v > 60)) / total),
        5: float(np.sum(frame & (h >= 8) & (h <= 30) & (s > 50) & (v > 70)) / total),
        6: float(np.sum(frame & (((h <= 8) | (h >= 170)) & (s > 50) & (v > 60))) / total),
    }
    best_level, best_ratio = max(ratios.items(), key=lambda item: item[1])
    if best_ratio >= 0.18:
        return best_level, float(np.clip(best_ratio / 0.72, 0.50, 0.92))
    return None, 0.0


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
    confidence = float(np.clip(max(gold_ratio, center_gold) / 0.20, 0.0, 0.85))
    return (gold_ratio > 0.13 and center_gold > 0.10), confidence


def _backpack_slot_fingerprint(crop: np.ndarray) -> str:
    small = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    quantized = (hsv[:, :, 0] // 12).astype(np.uint8)
    return "".join(f"{value:02x}" for value in quantized.flatten())


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
    for card in cards[:2]:
        scores = _brush_pattern_score_single(card)
        if not scores:
            continue
        brush_type, score = max(scores.items(), key=lambda item: item[1])
        if score < 0.54 or brush_type in options:
            continue
        options.append(brush_type)
        if _brush_card_selected_score(card) >= 0.16:
            selected = brush_type

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
        score = _affix_name_score(affix.name, keyword_scores)
        score += min(0.08, stage1_scores.get(affix.brush_type.value, 0.0) * 0.10)
        if score > 0.10:
            scored[affix.name] = float(np.clip(score, 0.0, 1.0))
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
        tokens.extend(["完成条件", "所需数量", "-2"])
    if "重新获得" in name:
        tokens.extend(["重新获得", "初始颜料"])
    if "同等级的随机怪物颜料" in name:
        tokens.extend(["转变为", "同等级", "随机怪物颜料"])
    return tuple(dict.fromkeys(tokens))


def _brush_affix_keywords(stage1_options: tuple[str, ...] = ()) -> tuple[str, ...]:
    allowed_types = set(stage1_options)
    tokens: list[str] = []
    for affix in build_affix_catalog():
        if allowed_types and affix.brush_type.value not in allowed_types:
            continue
        tokens.extend(_affix_tokens(affix.name))
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
        "完成条件",
        "所需数量",
        "-2",
        "-1",
        "进度要求",
        "减少1",
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
        "已放置颜料",
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


def _estimate_regions(image: np.ndarray, board: BoardObservation) -> list[RegionObservation]:
    height, width = image.shape[:2]
    b = board.rect
    x0 = _clamp_int(b.right + b.w * 0.10, 0, width - 1)
    content_x0 = _clamp_int(b.right - b.w * 0.06, 0, width - 1)
    right = _clamp_int(max(x0 + 1, width - b.w * 0.06), 1, width)
    panel_w = max(1, right - x0)
    content_panel_w = max(1, right - content_x0)
    stage_mode = _detect_brush_options_mode(image, x0, b, panel_w)

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
                        panel_w,
                        b.h * 0.48,
                        width,
                        height,
                    ),
                    confidence=board.score * 0.88,
                    label="brush options",
                ),
                RegionObservation(
                    kind=RegionKind.CONFIRM_BUTTON,
                    rect=_estimate_confirm_button_rect(image, b, x0, panel_w, stage_mode=True),
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


def _find_finish_drawing_signal(image: np.ndarray) -> tuple[float, Rect, str] | None:
    candidates: list[tuple[float, Rect, str]] = []
    button = _find_finish_drawing_button(image)
    if button is not None:
        candidates.append((button[0], button[1], "finish drawing button"))
    banner = _find_finish_ready_banner(image)
    if banner is not None:
        candidates.append((banner[0], banner[1], "finish ready banner"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def _find_finish_drawing_button(image: np.ndarray) -> tuple[float, Rect] | None:
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
        confidence = _finish_drawing_button_confidence(crop, area / roi_area, aspect)
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


def _finish_drawing_button_confidence(crop: np.ndarray, area_ratio: float, aspect: float) -> float:
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    blue_ratio = float(np.mean((h >= 88) & (h <= 112) & (s > 45) & (v > 115)))
    red_text_ratio = float(np.mean((((h <= 12) | (h >= 170)) & (s > 55) & (v > 95))))
    bright_ratio = float(np.mean(v > 145))
    keyword_score = _finish_button_keyword_score(crop)
    aspect_score = 1.0 - min(1.0, abs(aspect - 3.6) / 3.0)
    area_score = min(1.0, area_ratio / 0.08)
    if keyword_score < 0.40 or red_text_ratio < 0.010:
        return 0.0
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


def _finish_button_keyword_score(crop: np.ndarray) -> float:
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
        ("完成绘制",),
        font_size_range=range(14, 34, 2),
        scale=2.0,
        include_binary=True,
        threshold=0.0,
    )
    return scores.get("完成绘制", 0.0)


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

    scores = _brush_stage1_scores(_brush_stage1_crop(roi))
    pattern_scores = _brush_card_pattern_scores(roi)
    for brush_type, score in pattern_scores.items():
        scores[brush_type] = max(scores.get(brush_type, 0.0), score)
    ranked = sorted(scores.values(), reverse=True)
    strong_count = sum(1 for score in ranked if score >= 0.52)
    # The first initial paint has backpack icons here; their texture can match
    # brush words weakly. Only enter brush mode when two real stage-1 labels are
    # present.
    return len(ranked) >= 2 and strong_count >= 2 and ranked[1] >= 0.52


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

    layout = _estimated_board_layout_candidate(image)
    if layout is not None:
        candidates.append(layout)

    candidates.extend(_grid_scan_candidates(image))
    return _dedupe_rects(_preselect_board_candidates(image, candidates), image=image)


def _estimated_board_layout_candidate(image: np.ndarray) -> Rect | None:
    height, width = image.shape[:2]
    side = int(round(min(width, height) * 0.475))
    x = int(round(width * 0.245))
    y = int(round(height * 0.300))
    rect = Rect(x, y, side, side)
    if not _is_rect_inside_image(image.shape, rect):
        return None
    return rect


def _preselect_board_candidates(image: np.ndarray, rects: list[Rect], limit: int = 60) -> list[Rect]:
    if len(rects) <= limit:
        return rects
    layout = _estimated_board_layout_candidate(image)
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
    if layout is not None and all(_iou(layout, rect) < 0.90 for rect in selected):
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


def _score_board_rect(image: np.ndarray, rect: Rect) -> float:
    if not _is_rect_inside_image(image.shape, rect):
        return 0.0
    position_score = _board_position_score(image.shape, rect)
    size_score = _board_size_score(image.shape, rect)
    if position_score <= 0.02 or size_score <= 0.02:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 110)
    grid_score = _score_edges_grid(edges, rect)
    if grid_score < 0.16:
        return grid_score * position_score * size_score
    color_score = _score_tile_color_diversity(image, rect)
    completeness_score = _score_board_cell_completeness(image, rect)
    return (
        grid_score * 0.46
        + color_score * 0.22
        + size_score * 0.14
        + completeness_score * 0.18
    ) * position_score


def _score_board_cell_completeness(image: np.ndarray, rect: Rect) -> float:
    valid_cells = 0
    bottom_valid = 0
    cell_w = rect.w / BOARD_SIZE
    cell_h = rect.h / BOARD_SIZE
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x1 = round(rect.x + col * cell_w)
            y1 = round(rect.y + row * cell_h)
            x2 = round(rect.x + (col + 1) * cell_w)
            y2 = round(rect.y + (row + 1) * cell_h)
            if _tile_has_board_texture(image[y1:y2, x1:x2]):
                valid_cells += 1
                if row == BOARD_SIZE - 1:
                    bottom_valid += 1
    # A one-row-down candidate often has good internal grid lines but the last
    # row becomes plain background. Penalize that hard.
    return float(np.clip((valid_cells / 36.0) * 0.50 + (bottom_valid / 6.0) * 0.50, 0.0, 1.0))


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
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    edges = cv2.Canny(gray, 35, 100)
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
    if rect.right > width * 0.56 and side_ratio >= 0.38:
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
    if side_ratio < 0.44:
        return float(np.interp(side_ratio, [0.32, 0.44], [0.04, 0.52]))
    if side_ratio < 0.50:
        return float(np.interp(side_ratio, [0.44, 0.50], [0.52, 1.0]))
    if side_ratio < 0.56:
        return float(np.interp(side_ratio, [0.50, 0.56], [1.0, 0.72]))
    if side_ratio < 0.68:
        return float(np.interp(side_ratio, [0.56, 0.68], [0.72, 0.35]))
    return float(np.interp(side_ratio, [0.68, 0.92], [0.35, 0.08]))


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
    if right_ratio > 0.56 and side_ratio >= 0.38:
        score *= 0.08
    return float(np.clip(score, 0.0, 1.0))


def _refine_board_rect(image: np.ndarray, rect: Rect) -> Rect:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 110)
    height, width = image.shape[:2]

    best_score = _refine_candidate_score(image, edges, rect)
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
                score = _refine_candidate_score(image, edges, candidate)
                if score > best_score:
                    best_score = score
                    best_rect = candidate

    return _refine_board_rect_precise(image, edges, best_rect, width, height)


def _refine_board_rect_precise(image: np.ndarray, edges: np.ndarray, rect: Rect, width: int, height: int) -> Rect:
    best_score = _refine_candidate_score(image, edges, rect)
    best_rect = rect
    precise_radius = 5
    precise_step = 2
    for side in range(max(BOARD_SIZE * 30, rect.w - precise_radius), min(width, height, rect.w + precise_radius) + 1, precise_step):
        max_x = width - side
        max_y = height - side
        for y in range(max(0, rect.y - precise_radius), min(max_y, rect.y + precise_radius) + 1, precise_step):
            for x in range(max(0, rect.x - precise_radius), min(max_x, rect.x + precise_radius) + 1, precise_step):
                candidate = Rect(x, y, side, side)
                score = _refine_candidate_score(image, edges, candidate)
                if score > best_score:
                    best_score = score
                    best_rect = candidate
    return best_rect


def _refine_candidate_score(image: np.ndarray, edges: np.ndarray, rect: Rect) -> float:
    if not _is_rect_inside_image(image.shape, rect):
        return 0.0
    grid = _score_edges_grid(edges, rect)
    position = _board_position_score(image.shape, rect)
    size = _board_size_score(image.shape, rect)
    return grid * (0.55 + position * 0.35 + size * 0.10)


def _score_edges_grid(edges: np.ndarray, rect: Rect) -> float:
    h, w = edges.shape[:2]
    if rect.x < 0 or rect.y < 0 or rect.right > w or rect.bottom > h:
        return 0.0

    line_band = max(2, rect.w // 120)
    strengths: list[float] = []
    for i in range(BOARD_SIZE + 1):
        x = round(rect.x + rect.w * i / BOARD_SIZE)
        x1 = max(0, x - line_band)
        x2 = min(w, x + line_band + 1)
        strengths.append(float(edges[rect.y : rect.bottom, x1:x2].mean()) / 255.0)

        y = round(rect.y + rect.h * i / BOARD_SIZE)
        y1 = max(0, y - line_band)
        y2 = min(h, y + line_band + 1)
        strengths.append(float(edges[y1:y2, rect.x : rect.right].mean()) / 255.0)

    return float(np.clip(np.mean(strengths) * 3.5, 0.0, 1.0))


def _score_tile_color_diversity(image: np.ndarray, rect: Rect) -> float:
    states = []
    cell_w = rect.w / BOARD_SIZE
    cell_h = rect.h / BOARD_SIZE
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x1 = round(rect.x + col * cell_w)
            y1 = round(rect.y + row * cell_h)
            x2 = round(rect.x + (col + 1) * cell_w)
            y2 = round(rect.y + (row + 1) * cell_h)
            states.append(_classify_tile_for_board_score(image[y1:y2, x1:x2]))
    unique = len(set(states))
    known = sum(1 for state in states if state != TileState.UNKNOWN)
    return float(np.clip((unique / 5.0) * 0.45 + (known / 36.0) * 0.55, 0.0, 1.0))


def _classify_tile_for_board_score(tile_bgr: np.ndarray) -> TileState:
    if tile_bgr.size == 0:
        return TileState.UNKNOWN

    pad_y = max(2, tile_bgr.shape[0] // 10)
    pad_x = max(2, tile_bgr.shape[1] // 10)
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
    blue_ratio = float(np.mean((h >= 82) & (h <= 112) & (s > 70) & (v > 80)))
    cyan_ratio = float(np.mean((h >= 75) & (h <= 100) & (s > 95) & (v > 120)))
    non_blue_saturated = float(np.mean((s > 70) & (v > 120) & ~((h >= 75) & (h <= 112))))
    purple_ratio = float(np.mean((h >= 120) & (h <= 170) & (s > 70) & (v > 90)))
    gold_ratio = float(np.mean((h >= 15) & (h <= 45) & (s > 70) & (v > 110)))
    gray_mid_ratio = float(np.mean((s < 55) & (v >= 55) & (v <= 145)))

    center = core[
        core.shape[0] // 4 : core.shape[0] * 3 // 4,
        core.shape[1] // 4 : core.shape[1] * 3 // 4,
    ]
    center_hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    center_blue = float(
        np.mean(
            (center_hsv[:, :, 0] >= 75)
            & (center_hsv[:, :, 0] <= 112)
            & (center_hsv[:, :, 1] > 80)
            & (center_hsv[:, :, 2] > 80)
        )
    )
    center_dark = float(np.mean(center_hsv[:, :, 2] < 70))

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


def _classify_tiles(image: np.ndarray, board_rect: Rect) -> list[TileObservation]:
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
            state, confidence, features = classify_tile(image[y1:y2, x1:x2])
            tiles.append(
                TileObservation(
                    row=row,
                    col=col,
                    rect=rect,
                    state=state,
                    confidence=confidence,
                    features=features,
                )
            )
    return tiles


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

    # Board paints can be blue/green and look close to ice by color alone. The
    # explicit paint frame/icon must be checked before the broad ice rule.
    if frame_category != PaintCategory.UNKNOWN and frame_confidence >= 0.58 and not blocked_ice_like:
        return TileState.PAINT, max(0.60, min(0.92, frame_confidence)), features

    if blocked_ice_like:
        return TileState.BLOCKED, _conf(cyan_ratio + center_blue + mean_s, 1.65, 2.55), features

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
