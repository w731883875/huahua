"""User-defined line capacity: a whole line, blanks plus matching category."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from huahua_assistant.line_capacity import analyze_lines, summarize_lines
from huahua_assistant import game_model as gm
from huahua_assistant import strategy_lv7 as strategy
from huahua_assistant.learning import compact_trajectory_payload, expand_trajectory_payload


def row_board(row="MBGBMB"):
    mapping = {"M": "monster", "G": "gameplay"}
    chars = row+"V"*30
    states = "".join("P" if char in mapping else char for char in chars)
    tiles = [{"c": mapping[c], "l": 3, "s": "家族"+str(i)} if c in mapping else None
             for i, c in enumerate(chars)]
    return states, tiles


class LineCapacityTests(unittest.TestCase):
    def test_user_example_monster_five_gameplay_four(self):
        states, tiles = row_board()
        report = analyze_lines(states, tiles)
        row = next(line for line in report["lines"] if line["id"] == "r1")
        self.assertEqual(row["capacity"]["monster"]["cells"], [0, 1, 3, 4, 5])
        self.assertEqual(row["capacity"]["gameplay"]["cells"], [1, 2, 3, 5])
        self.assertEqual(report["counts"]["monster"]["exact"], {"4": 0, "5": 1, "6": 0})
        self.assertEqual(report["counts"]["gameplay"]["exact"], {"4": 1, "5": 0, "6": 0})
        self.assertEqual(report["counts"]["geometry"]["exact"]["6"], 1)

    def test_holes_do_not_split_line(self):
        states, tiles = row_board("BBVBBK")
        report = analyze_lines(states, tiles)
        self.assertEqual(report["counts"]["monster"]["exact"]["4"], 1)
        self.assertEqual(report["lines"][0]["capacity"]["monster"]["cells"], [0, 1, 3, 4])

    def test_matching_paints_keep_capacity_despite_different_family_level(self):
        states, tiles = row_board("BBBBBB")
        before = summarize_lines(states, tiles)
        states, tiles = row_board("MBMBMB")
        tiles[2]["l"] = 7
        after = summarize_lines(states, tiles)
        self.assertEqual(before["counts"]["monster"]["exact"], after["counts"]["monster"]["exact"])
        self.assertEqual(after["counts"]["monster"]["exact"]["6"], 1)
        self.assertEqual(after["counts"]["gameplay"]["exact"]["6"], 0)

    def test_treasure_flag_does_not_introduce_third_category(self):
        states, tiles = row_board("GBBBBB")
        tiles[0]["t"] = 1
        report = analyze_lines(states, tiles)
        self.assertEqual(report["counts"]["gameplay"]["exact"]["6"], 1)
        self.assertEqual(report["counts"]["monster"]["exact"]["5"], 1)

    def test_both_diagonal_orientations_and_column_are_counted(self):
        for occupied, expected in [([r*6+2 for r in range(6)], "vertical"),
                                   ([r*7 for r in range(6)], "diagonal_down"),
                                   ([r*6+5-r for r in range(6)], "diagonal_up")]:
            chars = ["V"]*36
            for i in occupied:
                chars[i] = "B"
            lines = analyze_lines("".join(chars))["lines"]
            sixes = [line for line in lines if line["capacity"]["monster"]["known"] == 6]
            self.assertEqual(len(sixes), 1)
            self.assertEqual(sixes[0]["direction"], expected)

    def test_full_line_not_counted_as_smaller_subsegments(self):
        report = analyze_lines("B"*36)
        self.assertEqual(report["counts"]["geometry"]["exact"], {"4": 4, "5": 4, "6": 14})
        self.assertEqual(report["counts"]["monster"]["at_least"], {"4": 22, "5": 18, "6": 14})
        self.assertEqual(len({tuple(line["cells"]) for line in report["lines"]}), 30)

    def test_unknown_paint_yields_bounds_not_false_exact_count(self):
        states, tiles = row_board("BBPBBB")
        report = analyze_lines(states, tiles)
        capacity = report["lines"][0]["capacity"]["monster"]
        self.assertEqual((capacity["known"], capacity["possible"]), (5, 6))
        self.assertEqual(report["counts"]["monster"]["exact"], {"4": 0, "5": 0, "6": 0})
        self.assertEqual(report["counts"]["monster"]["at_least"]["5"], 1)
        self.assertEqual(report["counts"]["geometry"]["exact"]["6"], 1)
        self.assertEqual(report["uncertain_cells"], [2])

    def test_conflicting_blank_detail_is_not_assumed_empty(self):
        states, tiles = row_board("BBBBBB")
        tiles[0] = {"c": "monster", "l": 4}
        source = copy.deepcopy(tiles)
        report = analyze_lines(states, tiles)
        self.assertEqual(report["uncertain_cells"], [0])
        self.assertEqual(report["lines"][0]["capacity"]["gameplay"]["known"], 5)
        self.assertEqual(report["counts"]["geometry"]["exact"]["6"], 0)
        self.assertEqual(tiles, source)

    def test_missing_cells_are_unknown_not_synthetic_blanks(self):
        report = analyze_lines("B")
        self.assertEqual(report["uncertain_cells"], list(range(1, 36)))
        self.assertEqual(report["counts"]["monster"]["exact"], {"4": 0, "5": 0, "6": 0})

    def test_capture_and_initial_snapshot_store_same_definition(self):
        from huahua_assistant import app
        states, raw = row_board()
        tiles = tuple(gm.Tile(gm.TileState.PAINT, gm.Paint(gm.PaintCategory(t["c"]), t["l"], t["s"]))
                      if t else gm.Tile(gm.TileState.BLANK if char == "B" else gm.TileState.VOID)
                      for char, t in zip(states, raw))
        state = gm.PuzzleState(tiles=tiles)
        scene = SimpleNamespace(brush_options=None, current_paint=None, backpack=None)
        with patch.object(app, "_compact_board_summary", return_value={"states": states}):
            step = app._compact_real_capture_step(scene=scene, state=state, backpack=(),
                                                 recommendations=[], brush_scenario=None, resources=gm.ResourceState())
        self.assertEqual(step["line_capacity"]["counts"]["monster"]["exact"]["5"], 1)
        self.assertEqual(step["line_capacity"], app._initial_snapshot_from_step(step)["line_capacity"])
        payload = {"steps": [step, copy.deepcopy(step)]}
        compacted = compact_trajectory_payload(copy.deepcopy(payload))
        self.assertIn("#line_capacity", compacted["steps"][0])
        self.assertEqual(expand_trajectory_payload(compacted), payload)
        self.assertIn(5, strategy.line_runs(state, gm.PaintCategory.MONSTER))
        self.assertNotIn(6, strategy.line_runs(state, gm.PaintCategory.MONSTER))
        self.assertIn(4, strategy.line_runs(state, gm.PaintCategory.GAMEPLAY))


if __name__ == "__main__":
    unittest.main()
