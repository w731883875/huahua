from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .outcome_learning import (
    DECISION_SCHEMA,
    RUN_SCHEMA,
    normalize_decision_payload,
    normalize_run_payload,
    normalize_trajectory_payload,
)


BUNDLE_SCHEMA = "huahua-multi-user-bundle-v1"
QUEUE_SCHEMA = "huahua-local-upload-queue-v1"
AGGREGATE_SCHEMA = "huahua-multi-user-aggregate-v1"
CONSENT_VERSION = "huahua-structured-sharing-v1"
REMOTE_UPLOAD_ENABLED = False

MAX_BUNDLE_BYTES = 2 * 1024 * 1024
MAX_EVENTS = 500
MAX_EVENT_BYTES = 128 * 1024
MAX_STRING_LENGTH = 512
MAX_CONTAINER_ITEMS = 512
MAX_DEPTH = 12
MAX_SESSIONS_PER_INSTALLATION = 100
MIN_EXPERT_INSTALLATIONS = 5
MIN_EXPERT_SESSIONS = 20

_BUNDLE_KEYS = {
    "schema",
    "bundle_id",
    "created_at",
    "consent",
    "installation",
    "versions",
    "data_categories",
    "privacy",
    "events",
    "content_sha256",
}
_EVENT_KEYS = {"event_id", "session_id", "kind", "source", "payload_schema", "payload", "payload_sha256"}
_QUEUE_KEYS = {
    "schema",
    "state",
    "remote_upload_enabled",
    "enqueued_at",
    "bundle_sha256",
    "bundle",
}
_SHAREABLE_EVENTS = {
    "recommendation_decision",
    "run_session_finished",
    "compressed_run_summary",
    "run_session_failed",
    "run_session_finish_skipped",
    "high_score_trajectory",
}
_RUN_EVENTS = _SHAREABLE_EVENTS - {"recommendation_decision"}
_PROHIBITED_KEY_PARTS = {
    "account",
    "chat",
    "clipboard",
    "debug_image",
    "email",
    "file_path",
    "image",
    "nickname",
    "note",
    "ocr_text",
    "password",
    "path",
    "phone",
    "raw_text",
    "screenshot",
    "token",
    "title",
    "user_id",
    "username",
    "window_title",
}
_RUN_PROJECTION = {
    "schema",
    "session",
    "versions",
    "strategy_mode",
    "observed_outcome",
    "proxy_score",
    "resource_usage",
    "trajectory_trust",
    "retention",
    "decision_records",
    "ordinary_summary",
    "initial_snapshot",
    "terminal",
    "high_score_gate",
}
_DECISION_PROJECTION = {
    "schema",
    "decision_id",
    "session",
    "state_signature",
    "phase",
    "recognition_quality",
    "candidates",
    "actual_choice",
    "top1_hit",
    "top3_hit",
    "transition",
    "resources",
    "attribution",
}


class BundleValidationError(ValueError):
    pass


def disabled_consent() -> dict[str, Any]:
    return {"enabled": False, "version": CONSENT_VERSION, "granted_at": None}


def granted_consent(*, granted_at: float | None = None) -> dict[str, Any]:
    return {
        "enabled": True,
        "version": CONSENT_VERSION,
        "granted_at": float(time.time() if granted_at is None else granted_at),
    }


def installation_pseudonym(secret: str | bytes, rotation_id: str) -> str:
    raw = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    if len(raw) < 16:
        raise BundleValidationError("installation secret must contain at least 16 bytes")
    rotation = _bounded_string(rotation_id, "rotation_id", maximum=64)
    return hashlib.sha256(b"huahua-share-v1\0" + rotation.encode("utf-8") + b"\0" + raw).hexdigest()


def create_multi_user_bundle(
    events: Iterable[Mapping[str, Any]],
    *,
    consent: Mapping[str, Any],
    installation_secret: str | bytes,
    rotation_id: str,
    versions: Mapping[str, Any],
    created_at: float | None = None,
) -> dict[str, Any]:
    normalized_consent = _validate_consent(consent, require_enabled=True)
    normalized_versions = _validate_versions(versions)
    pseudonym = installation_pseudonym(installation_secret, rotation_id)
    shareable: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    seen_hashes: set[str] = set()

    for event in events:
        if not isinstance(event, Mapping):
            raise BundleValidationError("every source event must be an object")
        kind = str(event.get("event") or "")
        if kind not in _SHAREABLE_EVENTS:
            continue
        normalized = _normalize_shareable_event(event)
        event_id = normalized["event_id"]
        payload_hash = normalized["payload_sha256"]
        previous_hash = seen_ids.get(event_id)
        if previous_hash is not None and previous_hash != payload_hash:
            raise BundleValidationError(f"conflicting payloads use event_id {event_id!r}")
        if previous_hash is not None or payload_hash in seen_hashes:
            continue
        seen_ids[event_id] = payload_hash
        seen_hashes.add(payload_hash)
        shareable.append(normalized)
        if len(shareable) > MAX_EVENTS:
            raise BundleValidationError(f"bundle exceeds {MAX_EVENTS} events")

    if not shareable:
        raise BundleValidationError("bundle contains no eligible structured events")

    base = {
        "schema": BUNDLE_SCHEMA,
        "created_at": _finite_timestamp(time.time() if created_at is None else created_at, "created_at"),
        "consent": normalized_consent,
        "installation": {"pseudonym": pseudonym, "rotation_id": str(rotation_id)},
        "versions": normalized_versions,
        "data_categories": sorted({_event_category(event["kind"]) for event in shareable}),
        "privacy": {
            "structured_events_only": True,
            "screenshots_included": False,
            "window_titles_included": False,
            "account_identifiers_included": False,
            "remote_upload_enabled": False,
        },
        "events": shareable,
    }
    content_hash = _stable_hash(base)
    bundle = {
        **base,
        "bundle_id": content_hash[:24],
        "content_sha256": content_hash,
    }
    return validate_multi_user_bundle(bundle)


def validate_multi_user_bundle(bundle: Mapping[str, Any] | bytes | str) -> dict[str, Any]:
    data, encoded_size = _decode_bounded_json(bundle, MAX_BUNDLE_BYTES, "bundle")
    _require_exact_keys(data, _BUNDLE_KEYS, "bundle")
    if data.get("schema") != BUNDLE_SCHEMA:
        raise BundleValidationError("incompatible bundle schema")
    consent = _validate_consent(data.get("consent"), require_enabled=True)
    versions = _validate_versions(data.get("versions"))
    installation = _require_mapping(data.get("installation"), "installation")
    _require_exact_keys(installation, {"pseudonym", "rotation_id"}, "installation")
    pseudonym = _hex_digest(installation.get("pseudonym"), "installation.pseudonym")
    _bounded_string(installation.get("rotation_id"), "installation.rotation_id", maximum=64)
    created_at = _finite_timestamp(data.get("created_at"), "created_at")
    if float(consent["granted_at"]) > created_at:
        raise BundleValidationError("consent cannot be granted after bundle creation")
    _bounded_string(data.get("bundle_id"), "bundle_id", maximum=24, minimum=24)
    _hex_digest(data.get("content_sha256"), "content_sha256")

    categories = data.get("data_categories")
    if not isinstance(categories, list) or not categories or set(categories) - {"decisions", "run_outcomes", "trajectories"}:
        raise BundleValidationError("data_categories contains unsupported values")
    if categories != sorted(set(categories)):
        raise BundleValidationError("data_categories must be sorted and unique")
    privacy = _require_mapping(data.get("privacy"), "privacy")
    expected_privacy = {
        "structured_events_only": True,
        "screenshots_included": False,
        "window_titles_included": False,
        "account_identifiers_included": False,
        "remote_upload_enabled": False,
    }
    if dict(privacy) != expected_privacy:
        raise BundleValidationError("privacy declaration is incompatible or unsafe")

    events = data.get("events")
    if not isinstance(events, list) or not events:
        raise BundleValidationError("events must be a non-empty array")
    if len(events) > MAX_EVENTS:
        raise BundleValidationError(f"bundle exceeds {MAX_EVENTS} events")
    seen_ids: dict[str, str] = {}
    seen_hashes: set[str] = set()
    for index, event in enumerate(events):
        _validate_bundle_event(event, index=index)
        if event["kind"] != "recommendation_decision":
            payload_versions = event["payload"]["versions"]
            for key in ("game", "recognizer", "policy"):
                if payload_versions.get(key) != versions[key]:
                    raise BundleValidationError(f"event version {key!r} does not match bundle versions")
        event_id = str(event["event_id"])
        payload_hash = str(event["payload_sha256"])
        if event_id in seen_ids and seen_ids[event_id] != payload_hash:
            raise BundleValidationError(f"event_id conflict at events[{index}]")
        if event_id in seen_ids or payload_hash in seen_hashes:
            raise BundleValidationError(f"duplicate event at events[{index}]")
        seen_ids[event_id] = payload_hash
        seen_hashes.add(payload_hash)

    expected_categories = sorted({_event_category(str(event["kind"])) for event in events})
    if categories != expected_categories:
        raise BundleValidationError("data_categories does not match events")
    base = {key: data[key] for key in _BUNDLE_KEYS - {"bundle_id", "content_sha256"}}
    expected_hash = _stable_hash(base)
    if data.get("content_sha256") != expected_hash or data.get("bundle_id") != expected_hash[:24]:
        raise BundleValidationError("bundle content hash mismatch")
    if encoded_size > MAX_BUNDLE_BYTES:
        raise BundleValidationError("bundle exceeds maximum encoded size")
    return _json_copy(data)


def enqueue_local_bundle(
    bundle: Mapping[str, Any] | bytes | str,
    queue_directory: str | Path,
    *,
    consent: Mapping[str, Any],
    enqueued_at: float | None = None,
) -> dict[str, Any]:
    _validate_consent(consent, require_enabled=True)
    validated = validate_multi_user_bundle(bundle)
    queued_at = _finite_timestamp(time.time() if enqueued_at is None else enqueued_at, "enqueued_at")
    if queued_at < float(validated["created_at"]):
        raise BundleValidationError("bundle cannot be enqueued before it was created")
    envelope = {
        "schema": QUEUE_SCHEMA,
        "state": "pending_local_only",
        "remote_upload_enabled": REMOTE_UPLOAD_ENABLED,
        "enqueued_at": queued_at,
        "bundle_sha256": validated["content_sha256"],
        "bundle": validated,
    }
    validate_queue_envelope(envelope)
    directory = Path(queue_directory)
    path = directory / f"{validated['bundle_id']}.queue.json"
    if path.exists():
        existing = validate_queue_envelope(path.read_bytes())
        if existing["bundle_sha256"] != envelope["bundle_sha256"]:
            raise BundleValidationError("queue filename collision")
        return {"path": str(path), "added": False, "state": existing["state"]}
    _atomic_write_json(path, envelope)
    return {"path": str(path), "added": True, "state": envelope["state"]}


def validate_queue_envelope(envelope: Mapping[str, Any] | bytes | str) -> dict[str, Any]:
    data, _size = _decode_bounded_json(envelope, MAX_BUNDLE_BYTES + MAX_EVENT_BYTES, "queue envelope")
    _require_exact_keys(data, _QUEUE_KEYS, "queue envelope")
    if data.get("schema") != QUEUE_SCHEMA:
        raise BundleValidationError("incompatible queue schema")
    if data.get("state") != "pending_local_only":
        raise BundleValidationError("unsupported queue state")
    if data.get("remote_upload_enabled") is not False or REMOTE_UPLOAD_ENABLED:
        raise BundleValidationError("remote upload must remain disabled")
    _finite_timestamp(data.get("enqueued_at"), "enqueued_at")
    bundle_hash = _hex_digest(data.get("bundle_sha256"), "bundle_sha256")
    bundle = validate_multi_user_bundle(data.get("bundle"))
    if bundle_hash != bundle["content_sha256"]:
        raise BundleValidationError("queue bundle hash mismatch")
    return _json_copy(data)


def aggregate_multi_user_bundles(
    bundles: Iterable[Mapping[str, Any] | bytes | str | Path],
    *,
    created_at: float | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    loaded = [_load_bundle_input(item) for item in bundles]
    if not loaded:
        raise BundleValidationError("at least one bundle is required")
    validated = list({bundle["content_sha256"]: bundle for bundle in loaded}.values())

    candidates: list[dict[str, Any]] = []
    quarantined: list[dict[str, str]] = []
    seen_ids: dict[str, str] = {}
    seen_hashes: set[str] = set()
    repeated_states: set[tuple[str, str, str]] = set()
    for bundle in sorted(validated, key=lambda item: item["content_sha256"]):
        installation = bundle["installation"]["pseudonym"]
        for event in bundle["events"]:
            event_id = event["event_id"]
            payload_hash = event["payload_sha256"]
            if event_id in seen_ids and seen_ids[event_id] != payload_hash:
                quarantined.append(_quarantine(event_id, "event_id_payload_conflict"))
                continue
            if event_id in seen_ids or payload_hash in seen_hashes:
                continue
            seen_ids[event_id] = payload_hash
            seen_hashes.add(payload_hash)
            reason = _event_anomaly(event)
            state_key = _repeated_state_key(installation, event)
            if reason is None and state_key is not None and state_key in repeated_states:
                reason = "repeated_decision_state"
            if reason is not None:
                quarantined.append(_quarantine(event_id, reason))
                continue
            if state_key is not None:
                repeated_states.add(state_key)
            candidates.append({"installation": installation, **event})

    sessions_by_installation: dict[str, set[str]] = defaultdict(set)
    for event in candidates:
        sessions_by_installation[event["installation"]].add(event["session_id"])
    allowed_sessions = {
        installation: set(sorted(sessions)[:MAX_SESSIONS_PER_INSTALLATION])
        for installation, sessions in sessions_by_installation.items()
    }
    accepted: list[dict[str, Any]] = []
    for event in candidates:
        if event["session_id"] not in allowed_sessions[event["installation"]]:
            quarantined.append(_quarantine(event["event_id"], "installation_session_rate_limit"))
        else:
            accepted.append(event)

    aggregate = _aggregate_accepted_events(accepted)
    result = {
        "schema": AGGREGATE_SCHEMA,
        "created_at": _finite_timestamp(time.time() if created_at is None else created_at, "created_at"),
        "source_bundle_sha256": sorted(bundle["content_sha256"] for bundle in validated),
        "counts": {
            "bundles": len(validated),
            "installations": len({event["installation"] for event in accepted}),
            "sessions": len({(event["installation"], event["session_id"]) for event in accepted}),
            "accepted_events": len(accepted),
            "quarantined_events": len(quarantined),
        },
        "weighting": {
            "installation": "equal",
            "session_within_installation": "equal",
            "decision_within_session": "equal",
            "raw_step_weighting": False,
            "max_sessions_per_installation": MAX_SESSIONS_PER_INSTALLATION,
        },
        "metrics": aggregate,
        "expert_gate": {
            "minimum_installations": MIN_EXPERT_INSTALLATIONS,
            "minimum_sessions": MIN_EXPERT_SESSIONS,
            "ready": (
                len({event["installation"] for event in accepted}) >= MIN_EXPERT_INSTALLATIONS
                and len({(event["installation"], event["session_id"]) for event in accepted})
                >= MIN_EXPERT_SESSIONS
            ),
        },
        "quarantine": {
            "count": len(quarantined),
            "reason_counts": dict(sorted(Counter(item["reason"] for item in quarantined).items())),
            "items": quarantined[:100],
        },
    }
    result["content_sha256"] = _stable_hash(result)
    if output_path is not None:
        _atomic_write_json(Path(output_path), result)
    return result


def _normalize_shareable_event(event: Mapping[str, Any]) -> dict[str, Any]:
    if event.get("source") != "real_capture":
        raise BundleValidationError("shareable events must come from real_capture")
    kind = str(event.get("event") or "")
    payload = _require_mapping(event.get("payload"), f"{kind}.payload")
    if kind == "recommendation_decision":
        normalized = normalize_decision_payload(payload)
        projected = {key: normalized.get(key) for key in _DECISION_PROJECTION}
        payload_schema = DECISION_SCHEMA
    elif kind == "high_score_trajectory":
        normalized = normalize_trajectory_payload(payload)
        if (normalized.get("trajectory_trust") or {}).get("trusted") is not True:
            raise BundleValidationError("trajectory is not trusted")
        projected = {key: normalized.get(key) for key in _RUN_PROJECTION if key in normalized}
        payload_schema = RUN_SCHEMA
    else:
        normalized = normalize_run_payload(payload, event_type=kind)
        projected = {key: normalized.get(key) for key in _RUN_PROJECTION if key in normalized}
        payload_schema = RUN_SCHEMA
    projected = _validated_payload(projected, f"{kind}.payload")
    session = _bounded_string(projected.get("session"), f"{kind}.session", maximum=128)
    payload_hash = _stable_hash(projected)
    source_id = event.get("event_id") or projected.get("decision_id") or f"{session}:{kind}:{payload_hash[:16]}"
    event_id = _bounded_string(source_id, f"{kind}.event_id", maximum=160)
    result = {
        "event_id": event_id,
        "session_id": session,
        "kind": kind,
        "source": "real_capture",
        "payload_schema": payload_schema,
        "payload": projected,
        "payload_sha256": payload_hash,
    }
    _validate_bundle_event(result, index=0)
    return result


def _validate_bundle_event(event: Any, *, index: int) -> None:
    value = _require_mapping(event, f"events[{index}]")
    _require_exact_keys(value, _EVENT_KEYS, f"events[{index}]")
    _bounded_string(value.get("event_id"), f"events[{index}].event_id", maximum=160)
    _bounded_string(value.get("session_id"), f"events[{index}].session_id", maximum=128)
    kind = _bounded_string(value.get("kind"), f"events[{index}].kind", maximum=64)
    if kind not in _SHAREABLE_EVENTS:
        raise BundleValidationError(f"unsupported event kind at events[{index}]")
    if value.get("source") != "real_capture":
        raise BundleValidationError(f"non-real source at events[{index}]")
    expected_schema = DECISION_SCHEMA if kind == "recommendation_decision" else RUN_SCHEMA
    if value.get("payload_schema") != expected_schema:
        raise BundleValidationError(f"payload schema mismatch at events[{index}]")
    payload = _validated_payload(value.get("payload"), f"events[{index}].payload")
    if payload.get("schema") != expected_schema:
        raise BundleValidationError(f"embedded schema mismatch at events[{index}]")
    if kind == "recommendation_decision":
        _require_exact_keys(payload, _DECISION_PROJECTION, f"events[{index}].payload")
    else:
        unknown = set(payload) - _RUN_PROJECTION
        required = {
            "schema",
            "session",
            "versions",
            "strategy_mode",
            "observed_outcome",
            "proxy_score",
            "resource_usage",
            "trajectory_trust",
            "retention",
        }
        missing = required - set(payload)
        if unknown or missing:
            raise BundleValidationError(
                f"events[{index}].payload fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        payload_versions = _require_mapping(payload.get("versions"), f"events[{index}].payload.versions")
        _require_exact_keys(
            payload_versions,
            {"schema", "game", "recognizer", "policy"},
            f"events[{index}].payload.versions",
        )
        if payload_versions.get("schema") != RUN_SCHEMA:
            raise BundleValidationError(f"run version schema mismatch at events[{index}]")
    if payload.get("session") != value.get("session_id"):
        raise BundleValidationError(f"session mismatch at events[{index}]")
    if kind == "high_score_trajectory" and (payload.get("trajectory_trust") or {}).get("trusted") is not True:
        raise BundleValidationError(f"untrusted trajectory at events[{index}]")
    payload_hash = _hex_digest(value.get("payload_sha256"), f"events[{index}].payload_sha256")
    if payload_hash != _stable_hash(payload):
        raise BundleValidationError(f"payload hash mismatch at events[{index}]")
    if len(_stable_json(value).encode("utf-8")) > MAX_EVENT_BYTES:
        raise BundleValidationError(f"event exceeds maximum encoded size at events[{index}]")


def _event_anomaly(event: Mapping[str, Any]) -> str | None:
    payload = event["payload"]
    if event["kind"] == "recommendation_decision":
        candidates = payload.get("candidates") or []
        if not isinstance(candidates, list) or len(candidates) > 128:
            return "invalid_candidate_count"
        for candidate in candidates:
            target = candidate.get("target") if isinstance(candidate, Mapping) else None
            if isinstance(target, Mapping):
                row, col = target.get("row"), target.get("col")
                if row is not None and (not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < 6):
                    return "impossible_candidate_target"
                if col is not None and (not isinstance(col, int) or isinstance(col, bool) or not 0 <= col < 6):
                    return "impossible_candidate_target"
        changed = (payload.get("transition") or {}).get("changed_cells") or []
        if not isinstance(changed, list) or len(changed) > 36:
            return "impossible_transition"
    observed = payload.get("observed_outcome") if isinstance(payload.get("observed_outcome"), Mapping) else {}
    actual_score = _finite_number(observed.get("actual_score"))
    if actual_score is not None and not 0 <= actual_score <= 1_000_000_000:
        return "anomalous_observed_score"
    proxy = payload.get("proxy_score") if isinstance(payload.get("proxy_score"), Mapping) else {}
    proxy_score = _finite_number(proxy.get("value"))
    if proxy_score is not None and not -1_000_000 <= proxy_score <= 1_000_000_000:
        return "anomalous_proxy_score"
    return None


def _aggregate_accepted_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_installation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_installation[str(event["installation"])].append(event)

    decision_top1 = 0.0
    decision_top3 = 0.0
    decision_weight = 0.0
    mode_score_sum: Counter[str] = Counter()
    mode_score_weight: Counter[str] = Counter()
    mode_run_count: Counter[str] = Counter()
    if not by_installation:
        return {
            "decision": {"weighted_top1_rate": None, "weighted_top3_rate": None, "weight": 0.0},
            "observed_outcome_by_mode": {},
        }

    installation_weight = 1.0 / len(by_installation)
    for installation_events in by_installation.values():
        sessions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for event in installation_events:
            sessions[str(event["session_id"])].append(event)
        decision_sessions = [items for items in sessions.values() if any(item["kind"] == "recommendation_decision" for item in items)]
        for items in decision_sessions:
            decisions = [item for item in items if item["kind"] == "recommendation_decision"]
            event_weight = installation_weight / len(decision_sessions) / len(decisions)
            for event in decisions:
                payload = event["payload"]
                decision_top1 += event_weight * float(payload.get("top1_hit") is True)
                decision_top3 += event_weight * float(payload.get("top3_hit") is True)
                decision_weight += event_weight

        run_sessions: list[Mapping[str, Any]] = []
        for items in sessions.values():
            runs = [item for item in items if item["kind"] in _RUN_EVENTS]
            if runs:
                run_sessions.append(max(runs, key=lambda item: _run_priority(str(item["kind"]))))
        for event in run_sessions:
            payload = event["payload"]
            actual = _finite_number((payload.get("observed_outcome") or {}).get("actual_score"))
            if actual is None:
                continue
            mode = str(payload.get("strategy_mode") or "unknown")
            weight = installation_weight / len(run_sessions)
            mode_score_sum[mode] += actual * weight
            mode_score_weight[mode] += weight
            mode_run_count[mode] += 1

    return {
        "decision": {
            "weighted_top1_rate": _rounded(decision_top1 / decision_weight) if decision_weight else None,
            "weighted_top3_rate": _rounded(decision_top3 / decision_weight) if decision_weight else None,
            "weight": _rounded(decision_weight),
        },
        "observed_outcome_by_mode": {
            mode: {
                "runs": int(mode_run_count[mode]),
                "weight": _rounded(mode_score_weight[mode]),
                "weighted_mean_actual_score": _rounded(mode_score_sum[mode] / mode_score_weight[mode]),
            }
            for mode in sorted(mode_score_weight)
            if mode_score_weight[mode] > 0
        },
    }


def _load_bundle_input(item: Mapping[str, Any] | bytes | str | Path) -> dict[str, Any]:
    if isinstance(item, Path):
        raw: Mapping[str, Any] | bytes | str = item.read_bytes()
    elif isinstance(item, str) and not item.lstrip().startswith("{") and len(item) < 1024:
        try:
            candidate = Path(item)
            raw = candidate.read_bytes() if candidate.is_file() else item
        except OSError as exc:
            raise BundleValidationError("bundle input path is invalid or unreadable") from exc
    else:
        raw = item
    try:
        data, _size = _decode_bounded_json(raw, MAX_BUNDLE_BYTES + MAX_EVENT_BYTES, "bundle input")
    except BundleValidationError:
        raise
    if data.get("schema") == QUEUE_SCHEMA:
        return validate_queue_envelope(data)["bundle"]
    return validate_multi_user_bundle(data)


def _validated_payload(payload: Any, label: str) -> dict[str, Any]:
    value = _require_mapping(payload, label)
    _validate_json_tree(value, label, depth=0)
    return _json_copy(value)


def _validate_json_tree(value: Any, label: str, *, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise BundleValidationError(f"{label} exceeds maximum nesting depth")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise BundleValidationError(f"{label} contains a non-finite number")
        return
    if isinstance(value, str):
        _bounded_string(value, label)
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise BundleValidationError(f"{label} contains too many fields")
        for key, item in value.items():
            key_text = _bounded_string(key, f"{label}.key", maximum=96)
            normalized_key = key_text.casefold().replace("-", "_")
            if any(part in normalized_key for part in _PROHIBITED_KEY_PARTS):
                raise BundleValidationError(f"{label} contains prohibited field {key_text!r}")
            _validate_json_tree(item, f"{label}.{key_text}", depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise BundleValidationError(f"{label} contains too many items")
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{label}[{index}]", depth=depth + 1)
        return
    raise BundleValidationError(f"{label} contains unsupported value type")


def _validate_consent(value: Any, *, require_enabled: bool) -> dict[str, Any]:
    consent = _require_mapping(value, "consent")
    _require_exact_keys(consent, {"enabled", "version", "granted_at"}, "consent")
    if not isinstance(consent.get("enabled"), bool):
        raise BundleValidationError("consent.enabled must be boolean")
    enabled = consent.get("enabled") is True
    if require_enabled and not enabled:
        raise BundleValidationError("structured sharing requires explicit opt-in consent")
    if consent.get("version") != CONSENT_VERSION:
        raise BundleValidationError("incompatible consent version")
    granted_at = consent.get("granted_at")
    if enabled:
        granted_at = _finite_timestamp(granted_at, "consent.granted_at")
    elif granted_at is not None:
        raise BundleValidationError("disabled consent cannot have granted_at")
    return {"enabled": enabled, "version": CONSENT_VERSION, "granted_at": granted_at}


def _validate_versions(value: Any) -> dict[str, str]:
    versions = _require_mapping(value, "versions")
    required = {"client", "game", "recognizer", "policy"}
    _require_exact_keys(versions, required, "versions")
    return {key: _bounded_string(versions.get(key), f"versions.{key}", maximum=96) for key in sorted(required)}


def _decode_bounded_json(value: Mapping[str, Any] | bytes | str, maximum: int, label: str) -> tuple[dict[str, Any], int]:
    if isinstance(value, Mapping):
        try:
            encoded = _stable_json(value).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise BundleValidationError(f"{label} is not JSON-compatible") from exc
        data = dict(value)
    else:
        encoded = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        if len(encoded) > maximum:
            raise BundleValidationError(f"{label} exceeds maximum encoded size")
        try:
            decoded = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BundleValidationError(f"{label} is not valid UTF-8 JSON") from exc
        if not isinstance(decoded, dict):
            raise BundleValidationError(f"{label} must be a JSON object")
        data = decoded
    if len(encoded) > maximum:
        raise BundleValidationError(f"{label} exceeds maximum encoded size")
    return data, len(encoded)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BundleValidationError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise BundleValidationError(f"{label} fields mismatch; missing={missing}, unknown={unknown}")


def _bounded_string(value: Any, label: str, *, maximum: int = MAX_STRING_LENGTH, minimum: int = 1) -> str:
    if not isinstance(value, str):
        raise BundleValidationError(f"{label} must be a string")
    if not minimum <= len(value) <= maximum:
        raise BundleValidationError(f"{label} length must be between {minimum} and {maximum}")
    return value


def _finite_timestamp(value: Any, label: str) -> float:
    number = _finite_number(value)
    if number is None or number < 0:
        raise BundleValidationError(f"{label} must be a finite non-negative timestamp")
    return number


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _hex_digest(value: Any, label: str) -> str:
    text = _bounded_string(value, label, maximum=64, minimum=64)
    if any(character not in "0123456789abcdef" for character in text):
        raise BundleValidationError(f"{label} must be a lowercase SHA256 digest")
    return text


def _event_category(kind: str) -> str:
    if kind == "recommendation_decision":
        return "decisions"
    if kind == "high_score_trajectory":
        return "trajectories"
    return "run_outcomes"


def _repeated_state_key(installation: str, event: Mapping[str, Any]) -> tuple[str, str, str] | None:
    if event["kind"] != "recommendation_decision":
        return None
    signature = str(event["payload"].get("state_signature") or "")
    choice = event["payload"].get("actual_choice") or {}
    candidate = str(choice.get("candidate_id") or "") if isinstance(choice, Mapping) else ""
    if not signature:
        return None
    return installation, str(event["session_id"]), f"{signature}:{candidate}"


def _run_priority(kind: str) -> int:
    return {
        "run_session_finished": 50,
        "compressed_run_summary": 40,
        "run_session_failed": 30,
        "run_session_finish_skipped": 20,
        "high_score_trajectory": 10,
    }.get(kind, 0)


def _quarantine(event_id: str, reason: str) -> dict[str, str]:
    return {"event_id": event_id, "reason": reason}


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_copy(value: Any) -> Any:
    return json.loads(_stable_json(value))


def _rounded(value: float) -> float:
    return round(float(value), 6)
