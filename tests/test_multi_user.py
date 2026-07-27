from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from huahua_assistant.multi_user import (
    AGGREGATE_SCHEMA,
    BUNDLE_SCHEMA,
    QUEUE_SCHEMA,
    BundleValidationError,
    aggregate_multi_user_bundles,
    create_multi_user_bundle,
    disabled_consent,
    enqueue_local_bundle,
    granted_consent,
    installation_pseudonym,
    validate_multi_user_bundle,
    validate_queue_envelope,
)


VERSIONS = {
    "client": "0.1.0",
    "game": "test-game",
    "recognizer": "test-recognizer",
    "policy": "test-policy",
}


def _decision(session: str, *, rank: int = 1, event_id: str | None = None) -> dict:
    event = {
        "source": "real_capture",
        "event": "recommendation_decision",
        "payload": {
            "session": session,
            "state_signature": f"state-{session}",
            "quality": {"ok": True, "score": 0.95},
            "action": {"type": "install_paint", "confidence": 0.9},
            "candidates": [
                {"r": 1, "c": 2, "score": 100.0},
                {"r": 2, "c": 3, "score": 90.0},
                {"r": 3, "c": 4, "score": 80.0},
            ],
            "actual_choice": {"matched": True, "rank": rank},
            "transition": {"changed_cells": [{"r": 1, "c": 2}]},
        },
    }
    if event_id is not None:
        event["event_id"] = event_id
    return event


def _run(session: str, score: float = 120.0, *, event_id: str | None = None) -> dict:
    event = {
        "source": "real_capture",
        "event": "run_session_finished",
        "payload": {
            "session": session,
            "strategy_mode": "balanced",
            "actual_score": score,
            "internal_score": score * 10,
            "versions": {"game": "test-game", "recognizer": "test-recognizer", "policy": "test-policy"},
        },
    }
    if event_id is not None:
        event["event_id"] = event_id
    return event


def _trajectory(session: str, *, trusted: bool) -> dict:
    return {
        "source": "real_capture",
        "event": "high_score_trajectory",
        "payload": {
            "session": session,
            "high_score_gate": {"passed": True},
            "steps": [{"quality": {"ok": trusted}, "action": "capture_update"}],
        },
    }


def _bundle(events: list[dict], secret: str = "installation-secret-0001") -> dict:
    return create_multi_user_bundle(
        events,
        consent=granted_consent(granted_at=100.0),
        installation_secret=secret,
        rotation_id="2026-07",
        versions=VERSIONS,
        created_at=101.0,
    )


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resign(bundle: dict) -> None:
    base = {key: value for key, value in bundle.items() if key not in {"bundle_id", "content_sha256"}}
    digest = _stable_hash(base)
    bundle["bundle_id"] = digest[:24]
    bundle["content_sha256"] = digest


class MultiUserBundleTests(unittest.TestCase):
    def test_sharing_is_disabled_until_explicit_versioned_consent(self) -> None:
        with self.assertRaises(BundleValidationError):
            create_multi_user_bundle(
                [_decision("s1")],
                consent=disabled_consent(),
                installation_secret="installation-secret-0001",
                rotation_id="2026-07",
                versions=VERSIONS,
            )

    def test_bundle_is_versioned_deduplicated_and_screenshot_free(self) -> None:
        event = _decision("s1")
        event["payload"]["screenshot"] = "C:/private/capture.png"
        bundle = _bundle([event, deepcopy(event), _run("s1")])

        self.assertEqual(BUNDLE_SCHEMA, bundle["schema"])
        self.assertEqual(2, len(bundle["events"]))
        self.assertFalse(bundle["privacy"]["screenshots_included"])
        self.assertNotIn("C:/private/capture.png", json.dumps(bundle, ensure_ascii=False))
        self.assertTrue(all("screenshot" not in event["payload"] for event in bundle["events"]))
        self.assertEqual(bundle, validate_multi_user_bundle(bundle))

    def test_installation_pseudonym_rotates_and_never_contains_secret(self) -> None:
        first = installation_pseudonym("installation-secret-0001", "2026-07")
        second = installation_pseudonym("installation-secret-0001", "2026-08")

        self.assertNotEqual(first, second)
        self.assertEqual(64, len(first))
        self.assertNotIn("installation-secret", first)

    def test_non_real_and_untrusted_trajectory_events_are_rejected(self) -> None:
        simulated = _decision("s1")
        simulated["source"] = "simulation"
        with self.assertRaises(BundleValidationError):
            _bundle([simulated])
        with self.assertRaises(BundleValidationError):
            _bundle([_trajectory("s1", trusted=False)])

    def test_tampering_unknown_fields_and_prohibited_payloads_are_rejected(self) -> None:
        original = _bundle([_decision("s1")])
        tampered = deepcopy(original)
        tampered["unknown"] = True
        with self.assertRaises(BundleValidationError):
            validate_multi_user_bundle(tampered)

        prohibited = deepcopy(original)
        payload = prohibited["events"][0]["payload"]
        payload["window_title"] = "private title"
        prohibited["events"][0]["payload_sha256"] = _stable_hash(payload)
        _resign(prohibited)
        with self.assertRaises(BundleValidationError):
            validate_multi_user_bundle(prohibited)

        unknown_payload = deepcopy(original)
        payload = unknown_payload["events"][0]["payload"]
        payload["harmless_but_unknown"] = True
        unknown_payload["events"][0]["payload_sha256"] = _stable_hash(payload)
        _resign(unknown_payload)
        with self.assertRaises(BundleValidationError):
            validate_multi_user_bundle(unknown_payload)

        with self.assertRaises(BundleValidationError):
            validate_multi_user_bundle(b"{" + b"x" * (2 * 1024 * 1024) + b"}")

    def test_local_queue_is_validated_deduplicated_and_remote_disabled(self) -> None:
        bundle = _bundle([_decision("s1")])
        with tempfile.TemporaryDirectory() as directory:
            consent = granted_consent(granted_at=100.0)
            first = enqueue_local_bundle(bundle, directory, consent=consent, enqueued_at=200.0)
            second = enqueue_local_bundle(bundle, directory, consent=consent, enqueued_at=300.0)
            envelope = validate_queue_envelope(Path(first["path"]).read_bytes())

        self.assertTrue(first["added"])
        self.assertFalse(second["added"])
        self.assertEqual(QUEUE_SCHEMA, envelope["schema"])
        self.assertEqual("pending_local_only", envelope["state"])
        self.assertFalse(envelope["remote_upload_enabled"])

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(BundleValidationError):
                enqueue_local_bundle(bundle, directory, consent=disabled_consent(), enqueued_at=200.0)

    def test_consent_time_and_event_versions_must_match_bundle_contract(self) -> None:
        with self.assertRaises(BundleValidationError):
            create_multi_user_bundle(
                [_decision("s1")],
                consent=granted_consent(granted_at=102.0),
                installation_secret="installation-secret-0001",
                rotation_id="2026-07",
                versions=VERSIONS,
                created_at=101.0,
            )

        mismatched = dict(VERSIONS)
        mismatched["policy"] = "different-policy"
        with self.assertRaises(BundleValidationError):
            create_multi_user_bundle(
                [_run("s1")],
                consent=granted_consent(granted_at=100.0),
                installation_secret="installation-secret-0001",
                rotation_id="2026-07",
                versions=mismatched,
                created_at=101.0,
            )


class MultiUserAggregateTests(unittest.TestCase):
    def test_aggregator_weights_installations_and_sessions_not_raw_decisions(self) -> None:
        frequent = _bundle(
            [_decision("a1", rank=1), _decision("a2", rank=2)],
            secret="installation-secret-A001",
        )
        sparse = _bundle([_decision("b1", rank=1)], secret="installation-secret-B001")

        aggregate = aggregate_multi_user_bundles([frequent, sparse], created_at=500.0)

        self.assertEqual(AGGREGATE_SCHEMA, aggregate["schema"])
        self.assertEqual(2, aggregate["counts"]["installations"])
        self.assertEqual(0.75, aggregate["metrics"]["decision"]["weighted_top1_rate"])
        self.assertEqual(1.0, aggregate["metrics"]["decision"]["weighted_top3_rate"])
        self.assertFalse(aggregate["weighting"]["raw_step_weighting"])

    def test_deduplication_conflicts_and_anomalous_scores_are_quarantined(self) -> None:
        first = _bundle([_run("s1", 100.0, event_id="shared")], secret="installation-secret-A001")
        conflict = _bundle([_run("s2", 200.0, event_id="shared")], secret="installation-secret-B001")
        anomalous = _bundle([_run("s3", 2_000_000_000.0)], secret="installation-secret-C001")

        aggregate = aggregate_multi_user_bundles([first, first, conflict, anomalous], created_at=600.0)

        self.assertEqual(1, aggregate["counts"]["accepted_events"])
        self.assertEqual(2, aggregate["counts"]["quarantined_events"])
        self.assertEqual(1, aggregate["quarantine"]["reason_counts"]["event_id_payload_conflict"])
        self.assertEqual(1, aggregate["quarantine"]["reason_counts"]["anomalous_observed_score"])

    def test_invalid_input_cannot_partially_replace_existing_aggregate(self) -> None:
        valid = _bundle([_decision("s1")])
        invalid = deepcopy(valid)
        invalid["events"][0]["payload"]["top1_hit"] = False
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "aggregate.json"
            output.write_text("original", encoding="utf-8")
            with self.assertRaises(BundleValidationError):
                aggregate_multi_user_bundles([valid, invalid], output_path=output)
            self.assertEqual("original", output.read_text(encoding="utf-8"))

    def test_single_installation_session_flood_is_capped_and_quarantined(self) -> None:
        flood = _bundle([_run(f"session-{index}") for index in range(101)])

        aggregate = aggregate_multi_user_bundles([flood], created_at=700.0)

        self.assertEqual(100, aggregate["counts"]["sessions"])
        self.assertEqual(100, aggregate["counts"]["accepted_events"])
        self.assertEqual(1, aggregate["quarantine"]["reason_counts"]["installation_session_rate_limit"])


if __name__ == "__main__":
    unittest.main()
