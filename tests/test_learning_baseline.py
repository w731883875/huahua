from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import huahua_assistant.learning as learning
from huahua_assistant.codes import canonical_affix_id
from huahua_assistant.learning import (
    _build_stats,
    _dedupe_events,
    _trajectory_payload_is_trusted,
)


def _trajectory(*quality_flags: bool, gate_passed: bool = True) -> dict:
    return {
        "version": 2,
        "high_score_gate": {"passed": gate_passed},
        "steps": [{"quality": {"ok": flag}} for flag in quality_flags],
    }


class LearningTrustBaselineTests(unittest.TestCase):
    def test_learning_stats_cache_reuses_and_invalidates_by_log_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "observations.jsonl", root / "corrections.jsonl", root / "trajectories.jsonl"]
            for path in paths:
                path.write_text("", encoding="utf-8")
            with (
                patch.object(learning, "OBSERVATION_LOG", paths[0]),
                patch.object(learning, "CORRECTION_LOG", paths[1]),
                patch.object(learning, "TRAJECTORY_LOG", paths[2]),
                patch.object(learning, "_LEARNING_STATS_CACHE_SIGNATURE", None),
                patch.object(learning, "_LEARNING_STATS_CACHE", None),
                patch.object(learning, "_build_complete_stats", side_effect=[{"generation": 1}, {"generation": 2}]) as build,
            ):
                first = learning.load_learning_stats()
                second = learning.load_learning_stats()
                paths[0].write_text("{}\n", encoding="utf-8")
                third = learning.load_learning_stats()

            self.assertIs(first, second)
            self.assertEqual({"generation": 2}, third)
            self.assertEqual(2, build.call_count)

    def test_trusted_trajectory_requires_steps_and_all_quality_gates(self) -> None:
        self.assertTrue(_trajectory_payload_is_trusted(_trajectory(True, True)))
        self.assertFalse(_trajectory_payload_is_trusted(_trajectory()))
        self.assertFalse(_trajectory_payload_is_trusted(_trajectory(True, False)))
        self.assertFalse(_trajectory_payload_is_trusted(_trajectory(True, gate_passed=False)))

    def test_legacy_payload_without_explicit_high_score_gate_remains_readable(self) -> None:
        payload = {"steps": [{"quality": {"ok": True}}]}

        self.assertTrue(_trajectory_payload_is_trusted(payload))

    def test_dedupe_rejects_simulation_and_untrusted_trajectories(self) -> None:
        trusted = {
            "source": "real_capture",
            "event": "high_score_trajectory",
            "payload": _trajectory(True),
        }
        duplicate = dict(trusted)
        simulated = {
            "source": "simulation",
            "event": "high_score_trajectory",
            "payload": _trajectory(True),
        }
        untrusted = {
            "source": "real_capture",
            "event": "high_score_trajectory",
            "payload": _trajectory(False),
        }

        self.assertEqual([trusted], _dedupe_events([trusted, duplicate, simulated, untrusted]))

    def test_stats_builder_ignores_non_real_sources_without_runtime_io(self) -> None:
        events = [
            {"source": "simulation", "event": "future_event", "payload": {}},
            {"source": "real_capture", "event": "future_event", "payload": {}},
        ]

        stats = _build_stats(events, corrections=[], trajectories=[])

        self.assertEqual(1, stats["total_events"])
        self.assertEqual(1, stats["event_counts"]["future_event"])
        self.assertEqual(0, stats["trajectories"])

    def test_legacy_affix_aliases_aggregate_under_canonical_identity(self) -> None:
        events = [
            {
                "source": "real_capture",
                "event": "brush_options",
                "payload": {
                    "stage2_options": ["范围内随机1个颜料提升1级"],
                    "stage2_codes": ["2-2"],
                },
            },
            {
                "source": "real_capture",
                "event": "brush_options",
                "payload": {
                    "stage2_options": ["对调色范围内的随机1个颜料额外提升1级"],
                    "stage2_codes": ["2-6"],
                },
            },
        ]

        stats = _build_stats(events, corrections=[], trajectories=[])
        identity = canonical_affix_id("2-2")

        self.assertIsNotNone(identity)
        self.assertEqual({identity: 2}, stats["affix_id_seen"])
        self.assertEqual(2, stats["affix_id_seen_total"])


if __name__ == "__main__":
    unittest.main()
