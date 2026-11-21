from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
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

#: 关键词之间互不依赖，``cv2.matchTemplate`` 会释放 GIL，所以按关键词并行。
#: 实测 14 个关键词一次匹配串行 3.7s、并行约 1.0s（8 核，外层 4~8 线程）。
#: 结果与串行完全一致——每个关键词独立取自己的最大值，没有归约顺序问题。
_MATCH_WORKERS = max(2, min(8, os.cpu_count() or 4))
_MATCH_POOL = ThreadPoolExecutor(max_workers=_MATCH_WORKERS, thread_name_prefix="kw-match")


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
    sizes = tuple(font_size_range)
    keyword_list = tuple(keywords)
    if len(keyword_list) < 2:
        scores = {
            keyword: _keyword_best(target, binary_target, keyword, sizes, scale)
            for keyword in keyword_list
        }
    else:
        scores = dict(
            zip(
                keyword_list,
                _MATCH_POOL.map(
                    lambda keyword: _keyword_best(target, binary_target, keyword, sizes, scale),
                    keyword_list,
                ),
            )
        )
    return {keyword: score for keyword, score in scores.items() if score >= threshold}


def _keyword_best(
    target: np.ndarray,
    binary_target: np.ndarray | None,
    keyword: str,
    font_sizes: tuple[int, ...],
    scale: float,
) -> float:
    score = _best_keyword_score(target, keyword, font_sizes, binary=False, scale=scale)
    if binary_target is not None:
        score = max(
            score,
            _best_keyword_score(binary_target, keyword, font_sizes, binary=True, scale=scale),
        )
    return score


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
    font_size_range: tuple[int, ...],
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


@lru_cache(maxsize=2048)
def _keyword_template(keyword: str, font_size: int, binary: bool = False, scale: float = 2.0) -> np.ndarray:
    font = _font(font_size)
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


@lru_cache(maxsize=1)
def _font_path() -> str:
    """字体路径。

    **必须缓存**：``_keyword_template`` 每次生成模板都会问一次，而冷启动第一帧
    要生成 500+ 个模板 —— 一次 ``Path(...).exists()`` 走四次磁盘 stat，
    522 次调用实测吃掉 2.4 秒，全在首帧预算里。
    """
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No Chinese font found in C:/Windows/Fonts")


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    """按字号复用 ``ImageFont`` 对象。

    ``ImageFont.truetype`` 每次都要重新读一遍字体文件（实测 522 次共 1.18s，
    其中 ``getfont`` 0.72s）。字号只有 ``range(14, 24)`` 这几个，缓存起来就没了。
    """
    return ImageFont.truetype(_font_path(), size)
