from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PetGuidancePlacement:
    left: int
    top: int
    width: int
    height: int
    side: str

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def covers(self, point: tuple[int, int]) -> bool:
        x, y = point
        return self.left <= x < self.right and self.top <= y < self.bottom


def choose_pet_guidance_placement(
    target: tuple[int, int],
    pet_size: tuple[int, int],
    screen_bounds: tuple[int, int, int, int],
    *,
    gap: int = 30,
    margin: int = 8,
) -> PetGuidancePlacement | None:
    if not _valid_pair(target) or not _valid_pair(pet_size) or not _valid_bounds(screen_bounds):
        return None
    target_x, target_y = (int(target[0]), int(target[1]))
    width, height = (int(pet_size[0]), int(pet_size[1]))
    screen_left, screen_top, screen_width, screen_height = (int(value) for value in screen_bounds)
    if width <= 0 or height <= 0 or gap < 0 or margin < 0:
        return None

    usable_left = screen_left + margin
    usable_top = screen_top + margin
    usable_right = screen_left + screen_width - margin
    usable_bottom = screen_top + screen_height - margin
    if not (
        screen_left <= target_x < screen_left + screen_width
        and screen_top <= target_y < screen_top + screen_height
    ):
        return None

    centered_left = _clamp(target_x - width // 2, usable_left, usable_right - width)
    centered_top = _clamp(target_y - height // 2, usable_top, usable_bottom - height)
    candidates = (
        PetGuidancePlacement(centered_left, target_y - gap - height, width, height, "above"),
        PetGuidancePlacement(target_x + gap, centered_top, width, height, "right"),
        PetGuidancePlacement(target_x - gap - width, centered_top, width, height, "left"),
        PetGuidancePlacement(centered_left, target_y + gap, width, height, "below"),
    )
    for candidate in candidates:
        if (
            candidate.left >= usable_left
            and candidate.top >= usable_top
            and candidate.right <= usable_right
            and candidate.bottom <= usable_bottom
            and not candidate.covers((target_x, target_y))
        ):
            return candidate
    return None


def pointer_geometry(side: str, size: int = 76) -> tuple[tuple[int, int, int, int], tuple[int, ...]]:
    center = max(1, int(size)) // 2
    edge = 7
    shoulder = 7
    if side == "above":
        line = (center, edge, center, center)
        head = (center, center, center - shoulder, center - shoulder, center + shoulder, center - shoulder)
    elif side == "right":
        line = (size - edge, center, center, center)
        head = (center, center, center + shoulder, center - shoulder, center + shoulder, center + shoulder)
    elif side == "left":
        line = (edge, center, center, center)
        head = (center, center, center - shoulder, center - shoulder, center - shoulder, center + shoulder)
    else:
        line = (center, size - edge, center, center)
        head = (center, center, center - shoulder, center + shoulder, center + shoulder, center + shoulder)
    return line, head


def _valid_pair(values) -> bool:
    return (
        isinstance(values, (tuple, list))
        and len(values) == 2
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
    )


def _valid_bounds(values) -> bool:
    return (
        isinstance(values, (tuple, list))
        and len(values) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
        and values[2] > 0
        and values[3] > 0
    )


def _clamp(value: int, low: int, high: int) -> int:
    return low if high < low else max(low, min(high, int(value)))
