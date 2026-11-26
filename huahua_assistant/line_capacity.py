"""Whole-line capacity for recording, without simulation or recommendations.

2026-11-26 user definition: monster/blank/gameplay/blank/monster/blank
has capacity 5 for monsters and 4 for gameplay. Neither an occupied cell
nor a terrain hole splits a row, column or diagonal into separate segments.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

BOARD_SIZE = 6
SCHEMA = "huahua-line-capacity-v1"
KINDS = ("geometry", "monster", "gameplay")


def _line_definitions():
    lines = []
    for row in range(BOARD_SIZE):
        lines.append((f"r{row+1}", "horizontal", [row*BOARD_SIZE+c for c in range(BOARD_SIZE)]))
    for col in range(BOARD_SIZE):
        lines.append((f"c{col+1}", "vertical", [r*BOARD_SIZE+col for r in range(BOARD_SIZE)]))
    for direction in ("diagonal_down", "diagonal_up"):
        groups = {}
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                key = row-col if direction == "diagonal_down" else row+col
                groups.setdefault(key, []).append(row*BOARD_SIZE+col)
        for key, cells in sorted(groups.items()):
            if len(cells) >= 2:
                lines.append((f"{direction}:{key}", direction, cells))
    return tuple(lines)


LINE_DEFINITIONS = _line_definitions()


def analyze_lines(states: str, tiles: Sequence[Mapping | None] = ()) -> dict:
    """Count exact capacities, keeping unknown/conflicting cells explicit.

    Geometry counts normal land (blank or painted) regardless of category.
    Category capacity counts blanks plus installed paints of that category;
    treasure is a flag, never a third category. K/V/F are excluded, not barriers.
    Bounds describe recognition uncertainty, not future terrain conversion.
    """
    known = {kind: set() for kind in KINDS}
    unknown = {kind: set() for kind in KINDS}
    for index in range(BOARD_SIZE**2):
        char = states[index] if index < len(states) else "?"
        tile = tiles[index] if index < len(tiles) else None
        category = tile.get("c") if isinstance(tile, Mapping) else None
        conflict = char != "P" and bool(category)
        if char not in "BPKVF" or conflict:
            for kind in KINDS:
                unknown[kind].add(index)
        elif char in ("B", "P"):
            known["geometry"].add(index)
            if char == "B":
                known["monster"].add(index)
                known["gameplay"].add(index)
            elif category in ("monster", "gameplay"):
                known[category].add(index)
            else:
                unknown["monster"].add(index)
                unknown["gameplay"].add(index)

    counts = {kind: {"exact": {str(n): 0 for n in (4, 5, 6)},
                     "at_least": {str(n): 0 for n in (4, 5, 6)},
                     "uncertain_lines": 0} for kind in KINDS}
    lines = []
    for identity, direction, cells in LINE_DEFINITIONS:
        capacity = {}
        for kind in KINDS:
            usable = [i for i in cells if i in known[kind]]
            uncertain = [i for i in cells if i in unknown[kind]]
            lower, upper = len(usable), len(usable)+len(uncertain)
            capacity[kind] = {"known": lower, "possible": upper, "cells": usable,
                              "uncertain_cells": uncertain}
            if uncertain:
                counts[kind]["uncertain_lines"] += 1
            elif lower >= 4:
                counts[kind]["exact"][str(lower)] += 1
            for size in (4, 5, 6):
                if lower >= size:
                    counts[kind]["at_least"][str(size)] += 1
        lines.append({"id": identity, "direction": direction, "cells": list(cells),
                      "capacity": capacity})
    return {"schema": SCHEMA, "counts": counts, "lines": lines,
            "uncertain_cells": sorted(set().union(*unknown.values()))}


def summarize_lines(states: str, tiles: Sequence[Mapping | None] = ()) -> dict:
    """Small per-frame record; line coordinates can be rebuilt from raw cells."""
    report = analyze_lines(states, tiles)
    return {key: report[key] for key in ("schema", "counts", "uncertain_cells")}
