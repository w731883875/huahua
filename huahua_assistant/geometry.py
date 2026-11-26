"""Image-derived grid and card geometry, in capture pixels.

Window ratios supply search hints only. Every accepted rectangle is supported
by edges in the current frame; cached coordinates are never the evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .models import Rect


@dataclass(frozen=True)
class GridGeometry:
    rect: Rect
    confidence: float
    line_support: tuple[float, ...]


def capture_viewport(capture) -> tuple[float, float, float, float] | None:
    """The game's 16:9 UI fits inside the client; title bars never scale.

    This is a prediction model, not proof of a layout. Grid/card edges still
    validate the prediction, including when the game changes its UI rules.
    """
    window = capture.window
    if window is None or window.client_bounds is None or window.width <= 0 or window.height <= 0:
        return None
    h, w = capture.image.shape[:2]
    if capture.kind not in ("window_content", "window_region") or w < 100 or h < 100:
        return None
    l, t, r, b = window.client_bounds
    sx, sy = w / window.width, h / window.height
    x, y = (l-window.left)*sx, (t-window.top)*sy
    cw, ch = (r-l)*sx, (b-t)*sy
    scale = min(cw / 1600., ch / 900.)
    return x+(cw-1600*scale)/2, y+(ch-900*scale)/2, 1600*scale, 900*scale


def map_rect(rect: Rect, old, new) -> Rect:
    sx, sy = new[2]/old[2], new[3]/old[3]
    return Rect(round(new[0]+(rect.x-old[0])*sx), round(new[1]+(rect.y-old[1])*sy),
                round(rect.w*sx), round(rect.h*sy))


def _axis_fit(profile: np.ndarray, origin: float, side: float, radius: float):
    """Fit seven equally spaced boundaries, preserving subpixel spacing."""
    n = len(profile)
    origins = np.arange(max(0, int(origin - radius)), min(n, int(origin + radius) + 1))
    sizes = np.arange(max(60, int(side - radius * 2)), min(n, int(side + radius * 2) + 1))
    if not len(origins) or not len(sizes):
        return None
    starts, spans = np.meshgrid(origins, sizes, indexing="ij")
    indices = np.rint(starts[..., None] + spans[..., None] * np.arange(7) / 6).astype(int)
    valid = indices[..., -1] < n
    values = profile[np.clip(indices, 0, n - 1)]
    # A few missing segments are normal; a single strong decorative border is not.
    score = values.mean(axis=-1) * .55 + np.sort(values, axis=-1)[..., 1:5].mean(axis=-1) * .45
    score[~valid] = -1
    i, j = np.unravel_index(np.argmax(score), score.shape)
    return int(starts[i, j]), int(spans[i, j]), values[i, j], float(score[i, j])


def refine_grid(image: np.ndarray, hint: Rect) -> GridGeometry | None:
    """Snap a coarse square to the actual seven horizontal/vertical lines."""
    h, w = image.shape[:2]
    if hint.w < 90 or hint.h < 90:
        return None
    pad = max(8, round(max(hint.w, hint.h) * .12))
    x0, y0 = max(0, hint.x - pad), max(0, hint.y - pad)
    x1, y1 = min(w, hint.right + pad), min(h, hint.bottom + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 100)
    length = max(9, round(min(hint.w, hint.h) / 6 * .38))
    vertical = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vx = vertical.mean(axis=0) / 255
    hy = horizontal.mean(axis=1) / 255
    # One pixel of tolerance handles double edges and rounding at grid intersections.
    vx = cv2.dilate(vx.reshape(1, -1), np.ones((1, 3), np.uint8)).ravel()
    hy = cv2.dilate(hy.reshape(-1, 1), np.ones((3, 1), np.uint8)).ravel()
    fx = _axis_fit(vx, hint.x - x0, hint.w, pad)
    fy = _axis_fit(hy, hint.y - y0, hint.h, pad)
    if fx is None or fy is None:
        return None
    x, bw, vs, sx = fx
    y, bh, hs, sy = fy
    if abs(bw - bh) > max(5, min(bw, bh) * .025):
        return None
    support = np.concatenate((vs, hs))
    if min(sx, sy) < .12 or np.count_nonzero(vs > .08) < 6 or np.count_nonzero(hs > .08) < 6:
        return None
    rect = Rect(x + x0, y + y0, bw, bh)
    return GridGeometry(rect, float(min(sx, sy)), tuple(float(v) for v in support))


def locate_grid(image: np.ndarray) -> GridGeometry | None:
    """Find a lattice without requiring a calibration for this window size."""
    h, w = image.shape[:2]
    scale = min(1., 800. / w)
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 35, 100)
    length = max(9, round(min(sh, sw) * .035))
    v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    v = cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    hh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
    hh = cv2.morphologyEx(hh, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vx = v[round(sh * .08):round(sh * .96), :round(sw * .82)].mean(axis=0) / 255
    hy = hh[:, :round(sw * .72)].mean(axis=1) / 255
    vx = cv2.dilate(vx.reshape(1, -1), np.ones((1, 3), np.uint8)).ravel()
    hy = cv2.dilate(hy.reshape(-1, 1), np.ones((3, 1), np.uint8)).ravel()
    best = None
    # A tall client adds vertical letterboxing without enlarging the board.
    # Search relative to both dimensions, not just the total window height.
    for side in range(max(90, round(min(sh, sw) * .32)), round(min(sh * .86, sw * .82)), 2):
        origins_x = np.arange(0, max(0, len(vx) - side))
        origins_y = np.arange(round(sh * .08), max(round(sh * .08), len(hy) - side))
        if not len(origins_x) or not len(origins_y):
            continue
        offset = np.rint(np.arange(7) * side / 6).astype(int)
        xs = vx[origins_x[:, None] + offset]
        ys = hy[origins_y[:, None] + offset]
        sx = xs.mean(axis=1) * .5 + np.sort(xs, axis=1)[:, 1:5].mean(axis=1) * .5
        sy = ys.mean(axis=1) * .5 + np.sort(ys, axis=1)[:, 1:5].mean(axis=1) * .5
        ix, iy = int(np.argmax(sx)), int(np.argmax(sy))
        score = min(sx[ix], sy[iy])
        if best is None or score > best[0]:
            best = (score, origins_x[ix], origins_y[iy], side)
    if best is None or best[0] < .10:
        return None
    _, x, y, side = best
    return refine_grid(image, Rect(round(x / scale), round(y / scale), round(side / scale), round(side / scale)))


def brush_cards(image: np.ndarray, panel: Rect) -> tuple[Rect, ...]:
    """Measure two stacked cards, including their shared left/right edges."""
    from .brush_panel import gutter_bands, card_bands_ok
    crop = image[panel.y:panel.bottom, panel.x:panel.right]
    if not card_bands_ok(crop):
        return ()
    bands = gutter_bands(crop)
    profiles = []
    for top, bottom in bands:
        value = cv2.cvtColor(crop[top:bottom], cv2.COLOR_BGR2HSV)[:, :, 2]
        profiles.append(value.mean(axis=0))
    width = crop.shape[1]
    delta = max(2, round(width * .005))
    gradients = [np.abs(p[delta:] - p[:-delta]) for p in profiles]
    common = np.minimum(*gradients)
    left_end = max(1, round(width * .20))
    right_start = round(width * .48)
    left = int(np.argmax(common[:left_end])) + delta // 2
    right = right_start + int(np.argmax(common[right_start:])) + delta // 2
    if common[left - delta // 2] < 10 or common[right - delta // 2] < 10 or right - left < width * .42:
        return ()
    return tuple(Rect(panel.x + left, panel.y + top, right - left, bottom - top) for top, bottom in bands)


def locate_brush_panel(image: np.ndarray, board: Rect) -> tuple[Rect, tuple[Rect, ...]] | None:
    """Start beside the board, then search nearby when the UI reflows."""
    h, w = image.shape[:2]
    for dx, dy in ((.10, .23), (.06, .23), (.14, .23), (.10, .18), (.10, .28), (.02, .23), (.18, .23)):
        x = max(0, min(w - 1, round(board.right + board.w * dx)))
        y = max(0, min(h - 1, round(board.y + board.h * dy)))
        right = min(w, round(w - board.w * .06))
        bottom = min(h, round(y + board.h * .48))
        if right <= x or bottom <= y:
            continue
        panel = Rect(x, y, right - x, bottom - y)
        cards = brush_cards(image, panel)
        if len(cards) == 2:
            return panel, cards
    return None
