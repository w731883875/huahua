from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huahua_assistant import learning
from huahua_assistant.app import _compact_recommendation
from huahua_assistant.game_model import Paint, PlacementRecommendation, RecommendationExplanation
from huahua_assistant.learning import _build_complete_stats
from huahua_assistant.models import PaintCategory
from huahua_assistant.outcome_learning import (
    DECISION_SCHEMA,
    RISK_SCHEMA,
    RUN_SCHEMA,
    build_outcome_risk_report,
    normalize_decision_payload,
    normalize_run_payload,
    normalize_trajectory_payload,
)


def _run_event(
    session: str,
    score: float,
    *,
    mode: str = "stable",
    event: str = "run_session_finished",
    actual: bool = True,
    high: bool = False,
) -> dict:
    payload = {
        "session": session,
        "strategy_mode": mode,
        "internal_score": score + 1000.0,
        "score_model": "proxy-v1",
        "finish_seen": True,
        "terminal": {"known": {"total": 12}},
        "initial_snapshot": {
            "board": {"blank": 18, "paint": 2, "blocked": 6, "void": 10, "states": "B" * 36}
        },
        "versions": {"policy": "policy-v1"},
        "high_score_gate": {"passed": high},
        "resource_usage": {"refresh_used": 1, "force_used": 2},
    }
    if actual:
        payload["actual_score"] = score
    return {"source": "real_capture", "event": event, "ts": score, "payload": payload}


class OutcomeContractTests(unittest.TestCase):
    def test_internal_score_never_becomes_observed_score(self) -> None:
        run = normalize_run_payload(
            {
                "session": "run-1",
                "internal_score": 4321.0,
                "score_model": "proxy-v1",
                "terminal": {"known": {"treasure7": 1}},
            },
            event_type="run_session_finished",
        )

        self.assertEqual(RUN_SCHEMA, run["schema"])
        self.assertIsNone(run["observed_outcome"]["actual_score"])
        self.assertEqual(4321.0, run["proxy_score"]["value"])

    def test_only_explicit_actual_score_populates_observed_outcome(self) -> None:
        run = normalize_run_payload(
            {"actual_score": 88, "internal_score": 9999, "score": 777},
            event_type="run_session_finished",
        )

        self.assertEqual(88.0, run["observed_outcome"]["actual_score"])
        self.assertEqual(9999.0, run["proxy_score"]["value"])

    def test_decision_keeps_all_candidates_and_separates_policy_score(self) -> None:
        candidates = [
            {
                "r": index,
                "c": index + 1,
                "paint": {"code": f"P-{index}"},
                "score": 100 - index,
                "explanation": {"layers": {"value": 20 + index, "risk": -index}},
            }
            for index in range(5)
        ]
        decision = normalize_decision_payload(
            {
                "session": "run-2",
                "board": {"states": "B" * 36},
                "quality": {"ok": True},
                "candidates": candidates,
                "chosen_recommendation": {"matched": True, "rank": 3, "cell": [2, 3]},
                "outcome": {"score_delta": 25, "changed_cells": [{"r": 2, "c": 3}]},
            }
        )

        self.assertEqual(DECISION_SCHEMA, decision["schema"])
        self.assertEqual(5, len(decision["candidates"]))
        self.assertEqual(98.0, decision["candidates"][2]["policy_score"]["value"])
        self.assertNotIn("score", decision["candidates"][2])
        self.assertEqual(-2.0, decision["candidates"][2]["risk"]["value"])
        self.assertFalse(decision["top1_hit"])
        self.assertTrue(decision["top3_hit"])
        self.assertEqual(25.0, decision["transition"]["proxy_score_delta"])
        self.assertTrue(decision["attribution"]["eligible_for_policy_evaluation"])

        low_confidence = normalize_decision_payload(
            {
                "quality": {"ok": True},
                "action": {"type": "install_paint", "confidence": 0.4},
                "candidates": candidates,
                "chosen_recommendation": {"matched": True, "rank": 1},
            }
        )
        self.assertFalse(low_confidence["attribution"]["eligible_for_policy_evaluation"])

    def test_trajectory_retains_full_steps_and_derives_decisions(self) -> None:
        trajectory = normalize_trajectory_payload(
            {
                "session": "run-3",
                "internal_score": 500,
                "steps": [
                    {
                        "step_index": 0,
                        "quality": {"ok": True},
                        "board": {"states": "B" * 36},
                        "resources": {"force_used": 0},
                        "candidates": [{"r": 1, "c": 2, "score": 90}],
                    },
                    {
                        "step_index": 1,
                        "quality": {"ok": True},
                        "inferred_action": {"type": "placement", "confidence": 0.9},
                        "chosen_recommendation": {"matched": True, "rank": 1},
                        "outcome": {"score_delta": 10},
                    },
                ],
            }
        )

        self.assertEqual("trusted_full_trajectory", trajectory["retention"])
        self.assertEqual(2, len(trajectory["steps"]))
        self.assertEqual(1, len(trajectory["decision_records"]))
        self.assertTrue(trajectory["trajectory_trust"]["trusted"])

    def test_ordinary_run_contains_lightweight_failure_summary(self) -> None:
        run = normalize_run_payload(
            {
                "session": "run-4",
                "skip_reason": "recognition_unreliable",
                "terminal": {"known": {"total": 5}, "quality": 0.4},
                "initial_snapshot": {"board": {"blank": 20, "paint": 2, "blocked": 5, "void": 9}},
                "resources": {"force_used": 1, "refresh_used": 2},
                "decision_records": [
                    {
                        "quality": {"ok": True},
                        "candidates": [{"r": 1, "c": 1, "score": 20}],
                        "chosen_recommendation": {"matched": True, "rank": 1},
                    }
                ],
            },
            event_type="run_session_finish_skipped",
        )

        summary = run["ordinary_summary"]
        self.assertEqual("lightweight_summary", run["retention"])
        self.assertIn("initial_map_bucket", summary)
        self.assertEqual("recognition_unreliable", summary["failure_reason"])
        self.assertEqual(1.0, summary["resource_usage"]["force_used"])
        self.assertEqual(1, len(summary["major_decisions"]))
        self.assertNotIn("decision_records", run)

    def test_compact_candidate_emits_explicit_policy_contract(self) -> None:
        recommendation = PlacementRecommendation(
            row=2,
            col=4,
            paint=Paint(PaintCategory.GAMEPLAY, 3),
            score=123.4,
            completes_match=False,
            affected=((2, 4),),
            reason="test",
            explanation=RecommendationExplanation(value=40, risk=-12, tags=("terrain_gain",)),
        )

        compact = _compact_recommendation(recommendation, rank=4)

        self.assertEqual(4, compact["rank"])
        self.assertEqual(123.4, compact["policy_score"]["value"])
        self.assertEqual(-12.0, compact["policy_score"]["components"]["risk"])
        self.assertTrue(compact["candidate_id"].startswith("placement:2:4:"))

    def test_learning_entry_points_write_versioned_contracts_in_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "RUNTIME_DIR": root,
                "OBSERVATION_LOG": root / "observations.jsonl",
                "CORRECTION_LOG": root / "corrections.jsonl",
                "TRAJECTORY_LOG": root / "trajectories.jsonl",
                "STATS_PATH": root / "stats.json",
            }
            with patch.multiple(learning, **paths):
                learning.record_observation(
                    "run_session_finished",
                    {"session": "isolated", "internal_score": 12, "terminal": {"known": {"total": 1}}},
                    source="real_capture",
                )
                payload = {
                    "session": "trusted",
                    "internal_score": 50,
                    "steps": [{"quality": {"ok": True}}],
                }
                self.assertTrue(learning.record_high_score_trajectory(payload, source="real_capture"))
                self.assertFalse(learning.record_high_score_trajectory(payload, source="real_capture"))
                # stats.json 由后台线程写：必须在这里等它写完，原因有两个——
                # 一是下面要读文件；二是补丁一退出，后台线程用的就是真实
                # STATS_PATH，会把测试数据写进 runtime/。
                self.assertTrue(learning.flush_pending_stats(timeout=60))

            observation = json.loads(paths["OBSERVATION_LOG"].read_text(encoding="utf-8").strip())
            trajectories = paths["TRAJECTORY_LOG"].read_text(encoding="utf-8").splitlines()
            stats = json.loads(paths["STATS_PATH"].read_text(encoding="utf-8"))

        self.assertEqual(RUN_SCHEMA, observation["payload"]["schema"])
        self.assertIsNone(observation["payload"]["observed_outcome"]["actual_score"])
        self.assertEqual(1, len(trajectories))
        self.assertEqual(RUN_SCHEMA, json.loads(trajectories[0])["payload"]["schema"])
        self.assertEqual(RISK_SCHEMA, stats["outcome_risk"]["schema"])


class ObservationSourceGateTests(unittest.TestCase):
    """学习样本只收真实截图，诊断事件不受这条限制。

    之前这条门对所有事件一律生效，于是游戏日志和自动跟随状态机的现场记录
    全被静默丢掉——工具「不出声」，事后也没有任何痕迹可查。
    """

    def test_only_real_capture_reaches_the_learning_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "observations.jsonl"
            with patch.multiple(learning, RUNTIME_DIR=root, OBSERVATION_LOG=log):
                learning.record_observation("quick_capture_step", {"a": 1}, source="replay")
                learning.record_observation("quick_capture_step", {"a": 1}, source="real_capture")

            events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(1, len(events))
        self.assertEqual("real_capture", events[0]["source"])

    def test_diagnostic_events_get_through_from_any_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "observations.jsonl"
            with patch.multiple(learning, RUNTIME_DIR=root, OBSERVATION_LOG=log):
                learning.record_observation("game_log_event", {"name": "pigment"}, source="game_log")
                learning.record_observation("auto_follow_pause", {"wgc": "x"}, source="auto_follow")

            events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(["game_log_event", "auto_follow_pause"], [item["event"] for item in events])
        self.assertEqual("game_log", events[0]["source"])


class RiskReportTests(unittest.TestCase):
    def test_report_computes_required_distribution_and_resource_metrics(self) -> None:
        events = [_run_event(f"run-{index}", float(index), high=index > 10) for index in range(1, 21)]

        report = build_outcome_risk_report(events, min_samples=20)
        stable = report["modes"]["stable"]
        metrics = stable["metrics"]

        self.assertEqual(RISK_SCHEMA, report["schema"])
        self.assertEqual({"stable", "balanced", "gamble"}, set(report["modes"]))
        self.assertEqual("ready", stable["status"])
        self.assertEqual("observed_outcome", stable["metric_basis"])
        self.assertEqual(10.5, metrics["mean"])
        self.assertEqual(10.5, metrics["median"])
        self.assertEqual(2.9, metrics["p10"])
        self.assertEqual(4.8, metrics["p20"])
        self.assertEqual(16.2, metrics["p80"])
        self.assertEqual(18.1, metrics["p90"])
        self.assertEqual(2.5, metrics["bottom_20_mean"])
        self.assertEqual(0.5, metrics["high_score_rate"])
        self.assertEqual(1.0, metrics["refresh_used_mean"])
        self.assertEqual(2.0, metrics["force_used_mean"])

    def test_below_threshold_and_proxy_only_data_stay_observing(self) -> None:
        observed = [_run_event(f"observed-{index}", float(index)) for index in range(19)]
        proxy_only = [_run_event(f"proxy-{index}", float(index), mode="gamble", actual=False) for index in range(30)]

        report = build_outcome_risk_report(observed + proxy_only, min_samples=20)

        self.assertEqual("observing", report["modes"]["stable"]["status"])
        self.assertEqual("proxy_score", report["modes"]["gamble"]["metric_basis"])
        self.assertEqual("proxy_only", report["modes"]["gamble"]["outcome_status"])
        self.assertEqual("observing", report["modes"]["gamble"]["status"])
        self.assertFalse(report["modes"]["gamble"]["personalization_ready"])
        self.assertIsNone(report["style_label"])

        legacy = _run_event("legacy", 7.0, actual=False)
        legacy["payload"].pop("strategy_mode")
        unclassified = build_outcome_risk_report([legacy], min_samples=20)["unclassified"]
        self.assertEqual("proxy_score", unclassified["metric_basis"])
        self.assertEqual(1007.0, unclassified["metrics"]["mean"])

    def test_session_deduplication_prefers_finished_observation(self) -> None:
        finished = _run_event("same-session", 9.0, event="run_session_finished")
        trajectory = _run_event("same-session", 100.0, event="high_score_trajectory")

        report = build_outcome_risk_report([finished], [trajectory], min_samples=1)

        self.assertEqual(1, report["runs_total"])
        self.assertEqual(9.0, report["modes"]["stable"]["metrics"]["mean"])

    def test_complete_learning_stats_exposes_risk_report_without_io(self) -> None:
        stats = _build_complete_stats([_run_event("run-1", 10.0)], corrections=[], trajectories=[])

        self.assertEqual(RISK_SCHEMA, stats["outcome_risk"]["schema"])
        self.assertEqual(1, stats["outcome_risk"]["runs_total"])


if __name__ == "__main__":
    unittest.main()
