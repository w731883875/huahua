"""棋盘格子「LvN」角标读取。

棋盘上每一枚颜料都会在格子右下角渲染一个 ``LvN`` 角标（Lv1 也会显示）。
空白地块、深色漩涡地块、冰块地块都没有角标。因此角标有两个用途：

1. **存在性判定**：有角标 == 该格子确实放了颜料。这比按颜色猜状态可靠得多，
   尤其在「入画」阶段（空白地块是纯灰蓝，没有作画阶段的白色画笔水印）。
2. **等级读取**：读角标上的数字。

实现方式对齐 TartarusSolver 的「梯度模板匹配」思路：先把角标区域转成
Sobel 梯度幅值图（文字边缘能量高、图标底图平滑），再用 ``TM_CCOEFF_NORMED``
做带平移容忍的模板匹配。角标在格子内的落点会随颜料图标尺寸轻微漂移，
所以模板比搜索窗小一圈，让 ``matchTemplate`` 自己找最佳对齐位置。

实测（``tests/fixtures/board_level_labels.json`` 的 32 个带标签格子）：

* 存在性判定：有标签最低分 0.70，无标签最高分 0.56 —— 阈值 0.63 可分。
* 等级数字：留一法 22/40（55%）。**不足以直接当作等级真值**，
  所以默认策略是「等级由边框色/模型跟踪，角标只做校验和兜底」：
  只有分数高于 ``LEVEL_TRUST_SCORE`` 时才覆盖边框色的判断。

Lv7 原本没有模板，``Lv7`` 角标只能匹配到最像的 ``Lv1``（0.38~0.46），
于是「7 级宝箱」被读成「1 级宝箱」。现在库里补了一枚 Lv7 样本，
实测自匹配 1.00、其它等级对它的最高分只有 0.69，可以放心用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

#: 角标搜索区域（相对格子左上角的比例）：从 y=0.62、x=0.34 一直到格子右下角。
LABEL_REGION_TOP = 0.62
LABEL_REGION_LEFT = 0.34
#: 归一化后的搜索窗尺寸（宽, 高）。所有截图缩放到这个尺寸后再匹配，
#: 这样 1673x878、1920x1080 等不同分辨率可以共用同一套模板。
LABEL_WINDOW = (56, 32)
#: 模板在搜索窗内的固定取景框 (x0, y0, x1, y1)，比搜索窗小一圈以容忍平移。
LABEL_TEMPLATE_BOX = (6, 4, 50, 28)
#: 每个等级最多保留多少个模板。模板越多越稳，但每格要跑的 matchTemplate 也越多。
MAX_TEMPLATES_PER_LEVEL = 3

#: 有角标的判定阈值。实测有标签最低 0.70、无标签最高 0.56，取中点。
PRESENCE_THRESHOLD = 0.63
#: 角标区域的梯度能量密度下限。文字边缘密集，空白/冰块/漩涡地块这一块是平滑的。
#: 实测有标签最低 75.6、无标签最高 53.6，取中点。
PRESENCE_DENSITY_MIN = 64.0
#: 等级数字可以直接采信的阈值。低于它只当作「有颜料」的证据，不采信数字。
LEVEL_TRUST_SCORE = 0.95
#: 棋盘上可能出现的最大等级（游戏上限）。
MAX_BOARD_LEVEL = 7

_TEMPLATE_PATH = Path(__file__).resolve().parent / "assets" / "layout" / "level_templates.npz"


@dataclass(frozen=True)
class LabelReading:
    """一次角标读取的结果。"""

    #: 读到的等级；读不准时为 None。
    level: int | None
    #: 数字匹配分数（0~1）。
    level_score: float
    #: 存在性分数（所有等级模板里的最高分，0~1）。
    presence_score: float
    #: 角标区域的梯度能量密度（文字边缘越密集越大）。
    density: float
    #: 角标是否存在。
    present: bool
    #: 每个等级的最高匹配分。只看绝对分数分不清「Lv7」和「读歪的 Lv1」，
    #: 看「最高分比 Lv1 高多少」就分得很开：真 Lv1 的 Lv1 分是 1.00、
    #: 别的等级 0.84；Lv7 宝箱的 Lv1 只有 0.73、Lv7 有 0.91~0.94。
    scores: dict[int, float] = field(default_factory=dict)

    @property
    def trusted_level(self) -> int | None:
        """只有分数足够高时才返回等级，否则 None（交给跟踪值）。"""
        if self.level is None:
            return None
        if self.level_score < LEVEL_TRUST_SCORE:
            return None
        return self.level


def label_window(tile_bgr: np.ndarray) -> np.ndarray:
    """把格子的角标区域裁出来并归一化成固定尺寸的梯度幅值图。"""
    if tile_bgr.size == 0:
        return np.zeros((LABEL_WINDOW[1], LABEL_WINDOW[0]), dtype=np.float32)
    h, w = tile_bgr.shape[:2]
    top = int(h * LABEL_REGION_TOP)
    left = int(w * LABEL_REGION_LEFT)
    crop = tile_bgr[top:h, left:w]
    if crop.size == 0:
        return np.zeros((LABEL_WINDOW[1], LABEL_WINDOW[0]), dtype=np.float32)
    crop = cv2.resize(crop, LABEL_WINDOW, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    magnitude = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    return cv2.GaussianBlur(magnitude, (3, 3), 0)


def label_template(tile_bgr: np.ndarray) -> np.ndarray | None:
    """从格子里取出用于建模板的固定取景框。"""
    window = label_window(tile_bgr)
    x0, y0, x1, y1 = LABEL_TEMPLATE_BOX
    patch = window[y0:y1, x0:x1]
    if patch.shape != (y1 - y0, x1 - x0):
        return None
    return patch


def _match_score(window: np.ndarray, template: np.ndarray) -> float:
    if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
        return 0.0
    result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    return float(result.max()) if result.size else 0.0


@dataclass
class LevelTemplateBank:
    """等级角标模板库。每个等级可以存多个样本，取最高分。"""

    templates: dict[int, list[np.ndarray]] = field(default_factory=dict)

    # ---------------------------------------------------------------- 构建

    def add(self, level: int, tile_bgr: np.ndarray) -> bool:
        patch = label_template(tile_bgr)
        if patch is None:
            return False
        items = self.templates.setdefault(int(level), [])
        if len(items) >= MAX_TEMPLATES_PER_LEVEL:
            return False
        items.append(patch.astype(np.float32))
        return True

    def extend_from_labels(
        self,
        cells: dict[tuple[int, int], np.ndarray],
        labels: dict[tuple[int, int], int],
    ) -> int:
        """按 {(row, col): level} 真值往库里加样本，返回成功条数。"""
        added = 0
        for position, level in labels.items():
            tile = cells.get(position)
            if tile is None:
                continue
            if self.add(level, tile):
                added += 1
        return added

    def __len__(self) -> int:
        return sum(len(items) for items in self.templates.values())

    def levels(self) -> tuple[int, ...]:
        return tuple(sorted(self.templates))

    # ---------------------------------------------------------------- 读取

    def read(self, tile_bgr: np.ndarray) -> LabelReading:
        window = label_window(tile_bgr)
        x0, y0, x1, y1 = LABEL_TEMPLATE_BOX
        patch = window[y0:y1, x0:x1]
        density = float(patch.mean()) if patch.size else 0.0
        best_score = 0.0
        best_level: int | None = None
        scores: dict[int, float] = {}
        for level, patches in self.templates.items():
            level_best = 0.0
            for template in patches:
                score = _match_score(window, template)
                if score > level_best:
                    level_best = score
            scores[level] = level_best
            if level_best > best_score:
                best_score = level_best
                best_level = level
        present = density >= PRESENCE_DENSITY_MIN and best_score >= PRESENCE_THRESHOLD
        return LabelReading(
            level=best_level if present else None,
            level_score=best_score,
            presence_score=best_score,
            density=density,
            present=present,
            scores=scores,
        )

    # ---------------------------------------------------------------- 持久化

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path is not None else _TEMPLATE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        for level in self.levels():
            patches = self.templates[level]
            if not patches:
                continue
            payload[f"level_{level}"] = np.stack(patches).astype(np.float32)
        np.savez_compressed(target, **payload)
        return target

    @classmethod
    def load(cls, path: Path | str | None = None) -> "LevelTemplateBank":
        target = Path(path) if path is not None else _TEMPLATE_PATH
        bank = cls()
        if not target.exists():
            return bank
        with np.load(target) as data:
            for key in data.files:
                if not key.startswith("level_"):
                    continue
                level = int(key.split("_", 1)[1])
                stack = data[key]
                bank.templates[level] = [
                    np.asarray(stack[index], dtype=np.float32) for index in range(stack.shape[0])
                ]
        return bank


_BANK_CACHE: LevelTemplateBank | None = None


def bank() -> LevelTemplateBank:
    """进程内共享的模板库（首次调用时从磁盘加载）。"""
    global _BANK_CACHE
    if _BANK_CACHE is None:
        _BANK_CACHE = LevelTemplateBank.load()
    return _BANK_CACHE


def reload_bank() -> LevelTemplateBank:
    global _BANK_CACHE
    _BANK_CACHE = LevelTemplateBank.load()
    return _BANK_CACHE


def read_label(tile_bgr: np.ndarray) -> LabelReading:
    """读取一个格子的角标。模板库为空时返回「不存在」。"""
    active = bank()
    if not active.templates:
        return LabelReading(
            level=None, level_score=0.0, presence_score=0.0, density=0.0, present=False
        )
    return active.read(tile_bgr)


def has_paint_label(tile_bgr: np.ndarray) -> bool:
    """该格子右下角是否有 ``LvN`` 角标（等价于「这格确实放了颜料」）。"""
    return read_label(tile_bgr).present
