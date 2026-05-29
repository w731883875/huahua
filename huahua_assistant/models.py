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


@dataclass(frozen=True)
class BoardObservation:
    rect: Rect
    score: float
    tiles: tuple[TileObservation, ...]

    def tile(self, row: int, col: int) -> TileObservation:
        return self.tiles[row * 6 + col]


@dataclass(frozen=True)
class RegionObservation:
    kind: RegionKind
    rect: Rect
    confidence: float
    label: str


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
class SceneObservation:
    board: BoardObservation
    regions: tuple[RegionObservation, ...]
    current_paint: CurrentPaintObservation | None = None
    brush_options: BrushOptionsObservation | None = None

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
