"""``scripts/replay_auto_run.py`` 的校验器测试。

重点不是「能跑通」，而是**负例必须被抓到**：如果校验器对篡改过的记录照样
说「通过」，那它就是在盖章，不是在验证。
"""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "replay_auto_run",
    Path(__file__).resolve().parent.parent / "scripts" / "replay_auto_run.py",
)
assert _SPEC and _SPEC.loader
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)


def _board(states: str, paints: list[dict]) -> dict:
    return {
        "signature": "sig",
        "states": states,
        "paints": paints,
        "buffs": [],
        "score": 0.9,
        "terrain_counts": {"paint": len(paints), "blank": 0, "blocked": 0, "void": 0, "faded": 0},
    }


def _paint(row: int, col: int, level: int, subtype: str = "宝箱") -> dict:
    return {
        "row": row,
        "col": col,
        "category": "gameplay",
        "level": level,
        "subtype": subtype,
        "treasure": False,
        "aura_level": 0,
        "bonus_level": 0,
        "state": "paint",
    }


BLANK = "B" * 36
ONE_PAINT = "PB" + "B" * 34
TWO_PAINT = "PP" + "B" * 34


def _run() -> list[dict]:
    return [
        {
            "event": "run_started",
            "ts": 1000.0,
            "session": "s",
            "payload": {"attached_mid_run": False, "initial_board": _board(BLANK, [])},
        },
        {
            "event": "board_transition",
            "ts": 1005.0,
            "payload": {
                "before_signature": "sig",
                "after_signature": "sig2",
                "board": {**_board(ONE_PAINT, [_paint(0, 0, 3)]), "signature": "sig2"},
                "changes": [
                    {
                        "row": 0,
                        "col": 0,
                        "from": {"state": "blank", "category": None, "level": None, "subtype": None},
                        "to": {"state": "paint", "category": "gameplay", "level": 3, "subtype": "宝箱"},
                    }
                ],
            },
        },
        {
            "event": "board_transition",
            "ts": 1008.0,
            "payload": {
                "before_signature": "sig2",
                "after_signature": "sig3",
                "board": {**_board(TWO_PAINT, [_paint(0, 0, 3), _paint(0, 1, 4, "存活")]), "signature": "sig3"},
                "changes": [
                    {
                        "row": 0,
                        "col": 1,
                        "from": {"state": "blank", "category": None, "level": None, "subtype": None},
                        "to": {"state": "paint", "category": "gameplay", "level": 4, "subtype": "存活"},
                    }
                ],
            },
        },
        {
            "event": "run_finished",
            "ts": 1010.0,
            "payload": {
                "duration": 10.0,
                "reason": "no_blank_tiles",
                "final_board": {**_board(TWO_PAINT, [_paint(0, 0, 3), _paint(0, 1, 4, "存活")]), "signature": "sig3"},
            },
        },
    ]


class ReplayVerifyTest(unittest.TestCase):
    def test_clean_run_passes(self) -> None:
        self.assertEqual(replay.verify(_run()), [])

    def test_detects_broken_signature_chain(self) -> None:
        run = copy.deepcopy(_run())
        run[2]["payload"]["before_signature"] = "wrong"
        issues = replay.verify(run)
        self.assertTrue(any("签名断链" in i for i in issues), issues)

    def test_detects_truncated_states(self) -> None:
        run = copy.deepcopy(_run())
        run[1]["payload"]["board"]["states"] = "P" * 20
        issues = replay.verify(run)
        self.assertTrue(any("states 长度" in i for i in issues), issues)

    def test_detects_paint_count_mismatch(self) -> None:
        run = copy.deepcopy(_run())
        run[1]["payload"]["board"]["paints"] = []
        issues = replay.verify(run)
        self.assertTrue(any("terrain_counts" in i for i in issues), issues)

    def test_detects_missing_finish(self) -> None:
        run = [r for r in _run() if r["event"] != "run_finished"]
        issues = replay.verify(run)
        self.assertTrue(any("run_finished" in i for i in issues), issues)

    def test_detects_final_board_mismatch(self) -> None:
        run = copy.deepcopy(_run())
        run[-1]["payload"]["final_board"]["signature"] = "other"
        issues = replay.verify(run)
        self.assertTrue(any("final_board" in i for i in issues), issues)


class ReplayStepsTest(unittest.TestCase):
    def test_steps_and_labels(self) -> None:
        steps = replay.build_steps(_run())
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["index"], 0)
        self.assertIn("落子", steps[1]["labels"][0])
        self.assertIn("r1c1", steps[1]["labels"][0])

    def test_grid_layout_is_row_major(self) -> None:
        steps = replay.build_steps(_run())
        table = replay.grid(steps[-1]["board"])
        self.assertIsNotNone(table[0][0]["paint"])
        self.assertIsNotNone(table[0][1]["paint"])
        self.assertIsNone(table[0][2]["paint"])

    def test_flip_flop_is_flagged(self) -> None:
        """同一格「颜料→虚空→颜料」，第二步必须被标成读数抖动（本局 r1c2 就是这个形态）。"""
        run = _run()
        paint_cell = {"state": "paint", "category": "gameplay", "level": 3, "subtype": "宝箱"}
        void_cell = {"state": "void", "category": None, "level": None, "subtype": None}
        synthetic = [
            run[0],
            {
                "event": "board_transition",
                "ts": 1006.0,
                "payload": {
                    "before_signature": "sig",
                    "after_signature": "sigv",
                    "board": {**_board("VB" + "B" * 34, []), "signature": "sigv"},
                    "changes": [{"row": 0, "col": 0, "from": paint_cell, "to": void_cell}],
                },
            },
            {
                "event": "board_transition",
                "ts": 1007.0,
                "payload": {
                    "before_signature": "sigv",
                    "after_signature": "sigp",
                    "board": {**_board(ONE_PAINT, [_paint(0, 0, 3)]), "signature": "sigp"},
                    "changes": [{"row": 0, "col": 0, "from": void_cell, "to": paint_cell}],
                },
            },
        ]
        steps = replay.build_steps(synthetic)
        self.assertTrue(steps[1]["flip_flops"], "翻回原状的变化没有被标记")
        self.assertIn("读数抖动", steps[1]["flip_flops"][0])
        self.assertIn("r1c1", steps[1]["flip_flops"][0])


class ReplayBrushEventWordingTest(unittest.TestCase):
    """词缀相关事件的措辞不能凭猜测。

    2026-11-19：用户明确「我的宠物是 2 级，所以并没有所谓的刷新词缀这个按钮」
    （设计文档也写着该功能 5 级宠物解锁）。而 ``affix_options_refreshed`` 只是
    「阶段二候选集变了」——换一张笔触大类卡也会触发它。早先 replay 把它标成
    「刷新词缀：A → B」，等于在复盘里凭空记录了一次用户不可能做的操作。
    """

    def test_options_refreshed_is_not_reported_as_refresh_button(self) -> None:
        line = replay._brush_event_line(
            "affix_options_refreshed", {"before": ["甲", "乙"], "after": ["丙", "丁"]}
        )
        self.assertIsNotNone(line)
        self.assertIn("候选变化", line)
        self.assertNotIn("刷新词缀", line)
        self.assertIn("丙", line)

    def test_real_refresh_click_is_flagged_as_level5_only(self) -> None:
        line = replay._brush_event_line("affix_refresh_clicked", {})
        self.assertIsNotNone(line)
        self.assertIn("刷新词缀", line)
        self.assertIn("5 级", line)


class ReplayFrameReliabilityTest(unittest.TestCase):
    """录制器对每一帧的「信不信」必须在复盘里看得见。

    2026-11-20：上一局日志里 8 条 ``unreliable_frame_dropped``、19 次落笔一条没记下，
    但复盘页对此只字未提 —— 用户看到的是「这一局我怎么没落笔」。所以把地形冲突、
    整帧作废、强制采信四类事件单独汇总出来。
    """

    def _run(self, *events: tuple[str, dict]) -> list[dict]:
        return [
            {"ts": 100.0 + index, "event": name, "payload": payload, "sequence": index}
            for index, (name, payload) in enumerate(events)
        ]

    def test_held_and_resolved_terrain_conflicts_are_summarized(self) -> None:
        run = self._run(
            ("terrain_conflict_held", {"row": 1, "col": 3, "held": "blocked", "observed": "void"}),
            ("terrain_conflict_held", {"row": 1, "col": 3, "held": "blocked", "observed": "void"}),
            ("terrain_conflict_resolved", {"row": 1, "col": 3, "from": "blocked", "to": "void"}),
        )
        lines = replay.audit_frame_reliability(run)
        text = "\n".join(lines)
        self.assertIn("r2c4", text)
        self.assertIn("×2", text)
        self.assertIn("改判", text)

    def test_dropped_and_forced_frames_are_counted(self) -> None:
        run = self._run(
            ("unreliable_frame_dropped", {"count": 4}),
            ("unreliable_frame_dropped", {"count": 4}),
            ("unreliable_frame_forced", {"frames": 5}),
        )
        text = "\n".join(replay.audit_frame_reliability(run))
        self.assertIn("整帧作废：2 帧", text)
        self.assertIn("强制采信：1 次", text)

    def test_clean_run_reports_nothing(self) -> None:
        """没有冲突事件时不能凭空造提示，否则等于噪音。"""
        run = self._run(("board_transition", {"changes": []}))
        self.assertEqual([], replay.audit_frame_reliability(run))


class ReplayInterruptedRunTest(unittest.TestCase):
    """中断局（没有 ``run_finished``）也必须能出 HTML。

    2026-11-20：录制器冻结那局就是中断局 —— ``payload`` 是空的，``duration`` 是
    ``None``，``{duration:.1f}`` 直接 TypeError，整个复盘页生不出来。最需要复盘的时候
    恰恰是这种局，不能崩。
    """

    def test_interrupted_run_still_renders_html(self) -> None:
        board = _board("P" + "B" * 35, [_paint(0, 0, 1, "伊吉")])
        run = [
            {"ts": 100.0, "event": "run_started", "payload": {"initial_board": board}},
            {
                "ts": 101.0,
                "event": "board_transition",
                "payload": {
                    "changes": [
                        {
                            "row": 0,
                            "col": 0,
                            "from": {"state": "blank", "category": None, "level": None,
                                     "subtype": None, "treasure": False,
                                     "bonus_level": 0, "aura_level": 0},
                            "to": _paint(0, 0, 1, "伊吉"),
                        }
                    ],
                    "before_signature": "a",
                    "after_signature": "b",
                    "board": board,
                },
            },
        ]
        steps = replay.build_steps(run)
        html = replay.build_html(run, steps, [], Path("runtime/x.jsonl"), None)
        self.assertIn("0.0s", html)


if __name__ == "__main__":
    unittest.main()
