from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huahua_assistant.batch_evaluation import BATCH_REPORT_SCHEMA, build_batch_evaluation_report
from huahua_assistant.cli import main


class BatchEvaluationTests(unittest.TestCase):
    def test_report_is_deterministic_and_explicitly_proxy_only(self) -> None:
        first = build_batch_evaluation_report([3, 7, 11], code_revision="test-rev")
        second = build_batch_evaluation_report([3, 7, 11], code_revision="test-rev")

        self.assertEqual(first, second)
        self.assertEqual(BATCH_REPORT_SCHEMA, first["schema"])
        self.assertEqual("simulation_proxy", first["score_type"])
        self.assertIsNone(first["observed_outcome"])
        self.assertEqual(3, first["seed_set"]["count"])
        self.assertEqual(64, len(first["configuration"]["sha256"]))
        self.assertEqual(64, len(first["code"]["source_sha256"]))

    def test_policy_and_resource_reports_use_every_seed(self) -> None:
        report = build_batch_evaluation_report(range(5))

        self.assertEqual({"stable", "balanced", "gamble"}, set(report["policies"]))
        for policy in report["policies"].values():
            self.assertEqual(5, policy["total_runs"])
            self.assertEqual(5, len(policy["runs"]))
            self.assertIn("bottom20_mean", policy["distribution"])
            self.assertIn("top1_hit_rate", policy)
            self.assertIn("top3_hit_rate", policy)

        for comparison in report["resource_ab"].values():
            self.assertEqual(5, comparison["paired_runs"])
            self.assertTrue(comparison["same_seed_set"])
            self.assertEqual(list(range(5)), [pair["seed"] for pair in comparison["pairs"]])
            self.assertIn("p10", comparison["delta_distribution"])
            self.assertIn("p90", comparison["delta_distribution"])

    def test_seed_validation_rejects_empty_duplicate_and_non_integer_sets(self) -> None:
        with self.assertRaises(ValueError):
            build_batch_evaluation_report([])
        with self.assertRaises(ValueError):
            build_batch_evaluation_report([1, 1])
        with self.assertRaises(TypeError):
            build_batch_evaluation_report([1, 1.5])

    def test_cli_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "batch.json"
            with patch(
                "sys.argv",
                [
                    "huahua-assistant",
                    "batch-evaluate",
                    "--start-seed",
                    "20",
                    "--runs",
                    "2",
                    "--output",
                    str(output),
                ],
            ):
                main()

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([20, 21], payload["seed_set"]["values"])
            self.assertEqual(2, payload["policies"]["balanced"]["total_runs"])


if __name__ == "__main__":
    unittest.main()
