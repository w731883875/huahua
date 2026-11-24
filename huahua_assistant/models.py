from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class TileState(str, Enum):
    BLANK = "blank"
    BLOCKED = "blocked"
    VOID = "void"
    FADED = "faded"
    PAINT = "paint"
    UNKNOWN = "unknown"


class RegionKind(str, Enum):
    CURRENT_PAINT = "current_paint"
    BACKPACK = "backpack"
    BRUSH_OPTIONS = "brush_options"
    CONFIRM_BUTTON = "confirm_button"
    START_DRAWING = "start_drawing"
    FINISH_DRAWING = "finish_drawing"
    FORCE_COMPLETE = "force_complete"
    REFRESH_AFFIX = "refresh_affix"


class PaintCategory(str, Enum):
    MONSTER = "monster"
    GAMEPLAY = "gameplay"
    UNKNOWN = "unknown"


class PaintDirection(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL = "diagonal"
    ADJACENT = "adjacent"
    GLOBAL = "global"
    UNKNOWN = "unknown"


class PaintObject(str, Enum):
    MONSTER = "monster"
    GAMEPLAY = "gameplay"
    ALL = "all"
    BLOCKED_TILE = "blocked_tile"
    BLANK_TILE = "blank_tile"
    VOID_TILE = "void_tile"
    FADED_TILE = "faded_tile"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class TileObservation:
    row: int
    col: int
    rect: Rect
    state: TileState
    confidence: float
    features: dict[str, float]
    paint_category: PaintCategory = PaintCategory.UNKNOWN
    paint_category_confidence: float = 0.0
    paint_level: int | None = None
    paint_level_confidence: float = 0.0
    paint_subtype: str | None = None
    paint_is_treasure: bool = False
    is_scope_highlight: bool = False
    scope_highlight_confidence: float = 0.0
    is_current_anchor: bool = False
    current_anchor_confidence: float = 0.0
    #: 「这格有残留笔触光环」的可信度 0~1。
    #:
    #: 颜料落到虚空后失去自身内容、退化成笔触效果，画面上只剩底色 + 一圈等级光环，
    #: 没有怪物/玩法图标、也就没有 LvN 角标。2026-11-24 实机 c5r6 就是这种格子，
    #: 当时被判成纯 VOID，笔触效果整个丢失。``features`` 里还有
    #: ``retained_aura_ring`` / ``retained_aura_center_edge`` 两个原始分量，
    #: 供后续标定阈值用。
    retained_aura: float = 0.0
    #: 等级光环颜色读出来的等级（用户实测一般 3-5 级）；读不出为 ``None``。
    retained_aura_level: int | None = None


@dataclass(frozen=True)
class BoardObservation:
    rect: Rect
    score: float
    tiles: tuple[TileObservation, ...]

    def tile(self, row: int, col: int) -> TileObservation:
        return self.tiles[row * 6 + col]


def is_enter_painting_region(region: object) -> bool:
    """这颗区域是不是右下角的「入画」按钮。

    它在**作画界面常驻**（实机每帧都检出，置信 0.92，位置 x=1263 y=779），
    意思是「可以进下一张画纸」，不是「本局已经画完」。判定结算时必须排除它，
    否则新一局的第一帧就会被当成可结算、自动收尾，整局落子都不进轨迹。

    ``label`` 在实时链路上带 ``" fast"`` 后缀（``enter painting button fast``），
    所以按前缀比，不要写全等。测试桩偶尔用 ``SimpleNamespace`` 伪造 region，
    这里按 ``getattr`` 取，缺 label 时当作「不是入画」。
    """

    return str(getattr(region, "label", "")).startswith("enter painting")


@dataclass(frozen=True)
class RegionObservation:
    kind: RegionKind
    rect: Rect
    confidence: float
    label: str

    @property
    def is_enter_painting(self) -> bool:
        """见 :func:`is_enter_painting_region`。"""

        return is_enter_painting_region(self)


@dataclass(frozen=True)
class CurrentPaintObservation:
    panel_rect: Rect
    icon_rect: Rect
    text_rect: Rect
    category: PaintCategory
    category_confidence: float
    subtype: str | None
    is_treasure: bool
    level: int | None
    level_confidence: float
    match_current: int | None
    match_required: int | None
    target_object: PaintObject
    direction: PaintDirection
    upgrade_amount: int | None
    keyword_scores: dict[str, float]
    raw_text_hint: str
    #: 面板第三行「调色效果」的正文 = 这枚颜料自带的**笔触条目**原文，
    #: 例如「使自身等级提升1级」「对全图中与自身内容相同的颜料额外提升1级」。
    #: 过去这一行解析完就扔，复盘只能看到大类（堆砌/扩散/转锋），
    #: 对不上游戏里那一条（官方 52 条 selectableeffect）。
    effect_text: str | None = None


@dataclass(frozen=True)
class BrushOptionsObservation:
    panel_rect: Rect
    stage1_options: tuple[str, ...]
    stage1_scores: dict[str, float]
    chosen_stage1: str | None
    stage2_options: tuple[str, ...]
    stage2_scores: dict[str, float]
    keyword_scores: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class BackpackPaintObservation:
    slot_rect: Rect
    category: PaintCategory
    category_confidence: float
    level: int
    level_confidence: float
    subtype: str | None = None
    is_treasure: bool = False
    fingerprint: str = ""
    is_selected: bool = False
    selection_confidence: float = 0.0


@dataclass(frozen=True)
class BackpackObservation:
    panel_rect: Rect
    paints: tuple[BackpackPaintObservation, ...]
    confidence: float


@dataclass(frozen=True)
class SceneObservation:
    board: BoardObservation
    regions: tuple[RegionObservation, ...]
    current_paint: CurrentPaintObservation | None = None
    brush_options: BrushOptionsObservation | None = None
    backpack: BackpackObservation | None = None

    def region(self, kind: RegionKind) -> RegionObservation | None:
        for region in self.regions:
            if region.kind == kind:
                return region
        return None


@dataclass(frozen=True)
class AnalysisResult:
    image_path: Path | None
    scene: SceneObservation | None
    debug_image_path: Path | None
