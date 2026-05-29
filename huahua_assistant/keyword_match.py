from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
)


def match_keywords(
    image_bgr: np.ndarray,
    keywords: tuple[str, ...],
    *,
    font_size_range: range = range(14, 24),
    scale: float = 2.0,
    include_binary: bool = True,
    threshold: float = 0.34,
) -> dict[str, float]:
    if image_bgr.size == 0:
        return {}

    target = _prepare_target(image_bgr, scale)
    binary_target = _prepare_binary_target(image_bgr, scale) if include_binary else None
    found: dict[str, float] = {}
    for keyword in keywords:
        score = _best_keyword_score(target, keyword, font_size_range, binary=False, scale=scale)
        if binary_target is not None:
            score = max(
                score,
                _best_keyword_score(binary_target, keyword, font_size_range, binary=True, scale=scale),
            )
        if score >= threshold:
            found[keyword] = score
    return found


def _prepare_target(image_bgr: np.ndarray, scale: float = 2.0) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if scale == 1.0:
        return gray
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _prepare_binary_target(image_bgr: np.ndarray, scale: float = 2.0) -> np.ndarray:
    gray = _prepare_target(image_bgr, scale)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(binary > 0)) > 0.55:
        binary = cv2.bitwise_not(binary)
    return binary


def _best_keyword_score(
    target: np.ndarray,
    keyword: str,
    font_size_range: range,
    *,
    binary: bool,
    scale: float,
) -> float:
    best = 0.0
    for font_size in font_size_range:
        template = _keyword_template(keyword, font_size, binary, scale)
        if template.shape[0] > target.shape[0] or template.shape[1] > target.shape[1]:
            continue
        result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        best = max(best, float(max_val))
    return best


@lru_cache(maxsize=512)
def _keyword_template(keyword: str, font_size: int, binary: bool = False, scale: float = 2.0) -> np.ndarray:
    font = ImageFont.truetype(_font_path(), font_size)
    bbox = font.getbbox(keyword)
    width = max(1, bbox[2] - bbox[0] + 8)
    height = max(1, bbox[3] - bbox[1] + 8)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.text((4 - bbox[0], 4 - bbox[1]), keyword, fill=255, font=font)
    arr = np.array(image)
    if scale != 1.0:
        arr = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if binary:
        _, arr = cv2.threshold(arr, 20, 255, cv2.THRESH_BINARY)
    return arr


def _font_path() -> str:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No Chinese font found in C:/Windows/Fonts")
