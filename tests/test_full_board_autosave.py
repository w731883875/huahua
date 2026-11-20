"""满铺自动存档（用户 2026-11-15 定）。

两条需求：

1. **满铺就要自动存**——实机玩法是画满就点「入画」开下一张，没人会去点
   「结束本局」，所以以前 ``_finish_real_session_internal`` 从来没被调用过，
   整局的轨迹全丢。现在第一次看到 ``blank=0 且 paint>=12`` 就自动收尾一次。
2. **无高分不加**——高分口径是「至少一枚 7 级颜料」。旧门槛里
   「内部评分 ≥3000」「6 级宝箱」也能单独过线，于是 ``lv7=0`` 的普通局照样进库
   （现存 14 条里就有 3 条）。

另外还有「按得分降序只保留前 500 局」的上限，见 :func:`learning._prune_trajectories`。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huahua_assistant import learning
from huahua_assistant.app import _high_score_gate


def _terminal(*, lv7: int = 0, treasure6: int = 0, treasure7: int = 0, total: int = 20) -> dict:
    return {
        "known": {
            "lv7": lv7,
            "treasure6": treasure6,
            "treasure7": treasure7,
            "total": total,
        }
    }


class HighScoreGateTests(unittest.TestCase):
    """入库口径（用户 2026-11-15 两轮确认）。

    1. 库没满 → 全部入库（先攒样本量）；
    2. 库满了 → 至少一枚 7 级颜料的局直接收；
    3. 库满且没有 7 级 → 只有内部评分高于库内最低分才替换入库。
    """

    def _gate(
        self,
        terminal: dict,
        *,
        count: int = 0,
        lowest: float | None = None,
        limit: int = 500,
        require_level7: bool = True,
        score: float = 100.0,
    ) -> dict:
        def fake_config(key, default=None):
            if key == "autosave.max_sessions":
                return limit
            if key == "autosave.require_level7":
                return require_level7
            return default

        with patch("huahua_assistant.app.config_value", side_effect=fake_config), \
             patch("huahua_assistant.app.trajectory_count", return_value=count), \
             patch("huahua_assistant.app.lowest_trajectory_score", return_value=lowest), \
             patch("huahua_assistant.app._internal_run_score", return_value=score):
            return _high_score_gate(terminal)

    def test_library_not_full_accepts_anything(self) -> None:
        """库没满时普通局也进——这是这次改口的地方。"""
        gate = self._gate(_terminal(lv7=0), count=7)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["primary_reason"], "library_not_full")
        self.assertEqual(gate["library"]["count"], 7)
        self.assertFalse(gate["library"]["full"])
        self.assertIn("未满", gate["summary"])

    def test_lv7_passes_when_library_full(self) -> None:
        gate = self._gate(_terminal(lv7=1), count=500, lowest=10.0)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["primary_reason"], "level7_paint")

    def test_full_library_rejects_low_score_without_lv7(self) -> None:
        gate = self._gate(_terminal(lv7=0), count=500, lowest=500.0, score=100.0)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["primary_reason"], "below_high_score_gate")
        self.assertIn("不高于最低分", gate["summary"])
        self.assertEqual(gate["library"]["lowest_score"], 500.0)

    def test_full_library_replaces_lowest_when_score_higher(self) -> None:
        gate = self._gate(_terminal(lv7=0), count=500, lowest=50.0, score=80.0)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["primary_reason"], "beats_lowest_score")
        self.assertIn("替换入库", gate["summary"])

    def test_equal_score_does_not_replace(self) -> None:
        """打平不替换：既省一次全量重写，也避免与已有那局重复。"""
        gate = self._gate(_terminal(lv7=0), count=500, lowest=80.0, score=80.0)
        self.assertFalse(gate["passed"])

    def test_disabling_require_level7_falls_back_to_score_only(self) -> None:
        """``require_level7=false`` 表示库满后 7 级局也不再直通，一律比分数。"""
        gate = self._gate(
            _terminal(lv7=1), count=500, lowest=500.0, score=100.0, require_level7=False
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["library"]["require_level7"], False)


class FullBoardAutosaveTriggerTests(unittest.TestCase):
    """满铺只触发一次存档尝试，且不重复评分。"""

    def _app(self, *, full: bool, attempted: bool = False, state=object()):
        from huahua_assistant.app import HuahuaApp

        calls: list[str] = []
        notices: list[tuple[str, str]] = []

        class _Fake:
            board_reached_full = full
            board_full_autosave_attempted = attempted
            current_state = state
            last_finish_skip_reason = None
            real_session_id = "run-test"
            real_trajectory_steps = [{}]

            def _finish_real_session_internal(self, *, reason: str):
                calls.append(reason)
                return True, 1234.0

            def _set_notice(self, text: str, level: str = "info") -> None:
                notices.append((text, level))

        fake = _Fake()
        return fake, calls, notices, lambda scene: HuahuaApp._maybe_autosave_full_board(fake, scene)

    def test_full_board_triggers_one_autosave(self) -> None:
        from tests.test_new_run_detection import ENTRY_SCREEN_STATES, _scene

        fake, calls, notices, call = self._app(full=True)
        with patch("huahua_assistant.app.record_observation"):
            call(_scene(ENTRY_SCREEN_STATES))
            call(_scene(ENTRY_SCREEN_STATES))
        self.assertEqual(calls, ["board_full_autosave"])
        self.assertTrue(fake.board_full_autosave_attempted)
        self.assertTrue(any("自动存档" in text for text, _level in notices))

    def test_not_full_does_nothing(self) -> None:
        from tests.test_new_run_detection import NEW_DEAL_STATES, _scene

        fake, calls, _notices, call = self._app(full=False)
        with patch("huahua_assistant.app.record_observation"):
            call(_scene(NEW_DEAL_STATES))
        self.assertEqual(calls, [])
        self.assertFalse(fake.board_full_autosave_attempted)

    def test_missing_state_marks_attempted_without_scoring(self) -> None:
        """没有可评分局面时不要让每一帧都重试。"""
        from tests.test_new_run_detection import ENTRY_SCREEN_STATES, _scene

        fake, calls, _notices, call = self._app(full=True, state=None)
        with patch("huahua_assistant.app.record_observation"):
            call(_scene(ENTRY_SCREEN_STATES))
        self.assertEqual(calls, [])
        self.assertTrue(fake.board_full_autosave_attempted)


class TrajectoryPruneTests(unittest.TestCase):
    """高分库按内部评分降序只保留前 500 局。"""

    def test_counts_lines_without_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "high_score_trajectories.jsonl"
            path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
            with patch.object(learning, "TRAJECTORY_LOG", path):
                self.assertEqual(learning.trajectory_count(), 2)

    def test_count_is_zero_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.jsonl"
            with patch.object(learning, "TRAJECTORY_LOG", path):
                self.assertEqual(learning.trajectory_count(), 0)
                self.assertIsNone(learning.lowest_trajectory_score())

    def test_lowest_score_reads_payload_internal_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "high_score_trajectories.jsonl"
            events = [
                {"ts": 1.0, "payload": {"internal_score": 30.0}},
                {"ts": 2.0, "payload": {"internal_score": 5.5}},
                {"ts": 3.0, "payload": {"internal_score": 90.0}},
            ]
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            with patch.object(learning, "TRAJECTORY_LOG", path):
                self.assertEqual(learning.lowest_trajectory_score(), 5.5)

    def test_keeps_top_scored_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "high_score_trajectories.jsonl"
            events = [
                {
                    "ts": float(index),
                    "event": "high_score_trajectory",
                    "source": "real_capture",
                    "payload": {"internal_score": float(index), "steps": [{}]},
                }
                for index in range(600)
            ]
            path.write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
                encoding="utf-8",
            )
            with patch.object(learning, "TRAJECTORY_LOG", path):
                dropped = learning._prune_trajectories(500)
                kept = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(dropped, 100)
        self.assertEqual(len(kept), 500)
        scores = [float(event["payload"]["internal_score"]) for event in kept]
        self.assertEqual(min(scores), 100.0)  # 0~99 分那 100 局被丢
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_under_limit_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "high_score_trajectories.jsonl"
            original = json.dumps({"ts": 1.0, "payload": {"internal_score": 5.0}}) + "\n"
            path.write_text(original, encoding="utf-8")
            with patch.object(learning, "TRAJECTORY_LOG", path):
                self.assertEqual(learning._prune_trajectories(500), 0)
                self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
