"""界面存在性判定：先用一个极便宜的梯度特征确认"这是不是作画界面"。

对应 TartarusSolver 的 ``SandboxPresence.cs``：

1. 把棋盘区域缩到 184x104；
2. 算水平和垂直方向的亮度梯度；
3. 归一化；
4. 和模板做点积。

棋盘每格的内容每局都不一样，但"6x6 格子的边框结构"是稳定的，
梯度特征抓的正是这个结构，所以一个模板就能跨局复用。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .layout import ScreenLayout
from .models import Rect

PRESENCE_WIDTH = 184
PRESENCE_HEIGHT = 104

# 阈值来自 11 张实机截图的实测分布：棋盘区域 0.27~0.84，同图其他区域 -0.05~0.01。
# 取 0.15 两侧都留足余量。
DEFAULT_THRESHOLD = 0.15


def gradient_signature(image_bgr: np.ndarray) -> np.ndarray:
    """把一块图像压成单位长度的梯度特征向量。"""
    if image_bgr.size == 0:
        return np.zeros(PRESENCE_WIDTH * PRESENCE_HEIGHT * 2, dtype=np.float32)
    small = cv2.resize(
        image_bgr,
        (PRESENCE_WIDTH, PRESENCE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    vector = np.concatenate([gx.ravel(), gy.ravel()]).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.zeros_like(vector)
    return vector / norm


def crop(image_bgr: np.ndarray, rect: Rect) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1 = max(0, min(width, rect.x))
    y1 = max(0, min(height, rect.y))
    x2 = max(x1, min(width, rect.right))
    y2 = max(y1, min(height, rect.bottom))
    return image_bgr[y1:y2, x1:x2]


def build_template(images: list[np.ndarray], layout: ScreenLayout | None = None) -> np.ndarray:
    """用若干张实机截图的棋盘区域平均出一个模板。"""
    layout = layout or ScreenLayout()
    vectors = []
    for image in images:
        height, width = image.shape[:2]
        board = layout.board_rect(width, height)
        signature = gradient_signature(crop(image, layout.presence_rect(board)))
        if float(np.linalg.norm(signature)) > 1e-6:
            vectors.append(signature)
    if not vectors:
        raise ValueError("没有可用于构建模板的图像")
    mean = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-6:
        raise ValueError("模板梯度接近零，图像可能全黑或全白")
    return (mean / norm).astype(np.float32)


def similarity(image_bgr: np.ndarray, template: np.ndarray, rect: Rect) -> float:
    signature = gradient_signature(crop(image_bgr, rect))
    return float(np.dot(signature, template))


def is_drawing_screen(
    image_bgr: np.ndarray,
    rect: Rect,
    *,
    template: np.ndarray | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, float]:
    """判断 ``rect`` 这块区域是不是目标界面。模板缺失时退化为"通过"。"""
    if template is None:
        template = load_template()
    if template is None:
        return True, 0.0
    score = similarity(image_bgr, template, rect)
    return score >= threshold, score


def load_template(path: str | Path | None = None) -> np.ndarray | None:
    file = Path(path) if path is not None else template_path()
    try:
        return np.load(file)
    except (OSError, ValueError):
        return None


def save_template(template: np.ndarray, path: str | Path | None = None) -> Path:
    file = Path(path) if path is not None else template_path()
    file.parent.mkdir(parents=True, exist_ok=True)
    np.save(file, template)
    return file


def template_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "layout" / "board_gradient.npy"
