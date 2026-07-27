from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
