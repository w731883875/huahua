"""Bounded, asynchronous visual evidence for actual brush choices.

The sampler does no board recognition/OCR. It retains card pixels first; a
single worker reads those exact pixels later. Each result belongs to a durable
choice UUID, never to whichever game happens to be open when OCR finishes.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import queue
import re
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import windows_ocr
from .brush_panel import BRUSH_DESCRIPTIONS, _affix_catalog_index, dice_similarity, lcs_ratio
from .codes import brush_effect_identity, brush_code, resolve_affix
from .models import Rect

ROOT = Path("runtime/brush_evidence")
VERSION = "brush-evidence-v1"
DEFAULT_MAX_BYTES = 128 * 1024 * 1024


def _clean(text):
    return re.sub(r"[^\w]", "", str(text)).replace("斜对角", "对角斜向")


@lru_cache(maxsize=1)
def recognizer_revision():
    # A changed matcher/catalog needs a new audit, even with the same schema.
    try:
        source = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    except OSError:  # Frozen executable has no loose source file.
        import marshal
        source = marshal.dumps((_match.__wrapped__.__code__, read_card.__code__, detect_cards.__code__))
    return hashlib.sha256(source + repr(_affix_catalog_index()).encode()).hexdigest()[:16]


@lru_cache(maxsize=512)
def _match(text):
    text = _clean(text)
    empty = {"name": None, "stage": "unknown", "score": 0.0, "margin": 0.0,
             **brush_effect_identity(None)}
    if not text:
        return empty
    for brush, description in BRUSH_DESCRIPTIONS:
        if lcs_ratio(description, text) >= .88:
            return {**empty, "name": brush.value, "stage": "direction",
                    "score": round(lcs_ratio(description, text), 4)}
    ranks = {}
    for name, alias in _affix_catalog_index():
        alias = _clean(alias)
        # Do not infer a missing number/direction from a near-identical effect.
        tokens = r"\d+|横向|纵向|相邻|全图|斜对角|对角斜向|阻挡|虚空|随机|最近"
        if sorted(re.findall(tokens, alias)) != sorted(re.findall(tokens, text)):
            continue
        ranks[name] = max(ranks.get(name, 0.0), dice_similarity(alias, text))
    ordered = sorted(ranks.items(), key=lambda pair: pair[1], reverse=True)
    if not ordered:
        return empty
    name, score = ordered[0]
    margin = score - (ordered[1][1] if len(ordered) > 1 else 0)
    accepted = score >= .90 and margin >= .08
    return {"name": name if accepted else None, "stage": "effect" if accepted else "unknown",
            "score": round(score, 4), "margin": round(margin, 4),
            **brush_effect_identity(name if accepted else None)}


def match_card(lines):
    """Preserve one result per physical slot, including unread and duplicate cards."""
    return dict(_match("".join(lines or ())))


def read_card(card):
    crop = card[:, round(card.shape[1]*.14):round(card.shape[1]*.98)]
    scale = min(4., 2200/max(crop.shape[:2]))
    value = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2], cv2.COLOR_GRAY2BGR)
    variants = [(crop, scale), (value, scale), (value, min(3., scale))]
    attempts = []
    for variant, factor in variants:
        lines = windows_ocr.recognize_chinese_lines(cv2.resize(
            variant, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC))
        result = match_card(lines)
        attempts.append(list(lines))
        if result.get("name"):
            return result, attempts
    return result, attempts


def audit_report(records, reviews):
    """Human-labelled choices, including missed/unresolved choices in denominator."""
    revision = recognizer_revision()
    # Editing a label replaces its previous value; repeated labels cannot inflate n.
    labels = {r.get("choice_id"): r for r in reviews if r.get("choice_id") and r.get("reviewer")
              and r.get("recognizer_revision", revision) == revision}
    exclusions = {key for key, label in labels.items() if label.get("expected_kind") == "not_effect"}
    correct = 0
    sessions, sizes = set(), set()
    for key, label in labels.items():
        if key in exclusions:
            continue
        record = records.get(key, {})
        if record.get("session"):
            sessions.add(record["session"])
        if record.get("frame_size"):
            sizes.add(tuple(record["frame_size"]))
        ok = (record.get("recognizer_revision") == revision
              and label.get("recognizer_revision") == revision
              and record.get("status") == "confirmed"
              and bool(label.get("expected_canonical_id"))
              and record.get("canonical_affix_id") == label.get("expected_canonical_id")
              and record.get("selected_index") == label.get("expected_index")
              and record.get("brush_effect_ids") == label.get("expected_official_ids"))
        correct += bool(ok)
    n = len(labels)-len(exclusions)
    rate = correct / n if n else None
    z = 1.96
    lower = ((rate + z*z/(2*n) - z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)))
             / (1 + z*z/n)) if n else 0.0
    # Whole recorded sessions must be reviewed: cherry-picking only successful
    # predictions would conceal unread choices. Truly missed choices are labelled
    # explicitly as missed-<unique> by the reviewer.
    unreviewed = sum(1 for key, r in records.items() if r.get("session") in sessions
                     and r.get("stage") != "direction" and key not in labels
                     and r.get("recognizer_revision") == revision)
    return {"reviewed": n, "correct": correct, "accuracy": rate, "excluded_non_choices": len(exclusions),
            "unreviewed_in_sampled_sessions": unreviewed,
            "wilson_lower_95": lower, "sessions": len(sessions), "window_sizes": len(sizes),
            "recognizer_revision": revision,
            "eligible": bool(n >= 300 and lower >= .99 and len(sessions) >= 10 and len(sizes) >= 3
                             and unreviewed == 0)}


def choice_at_time(record, timestamp):
    """Late OCR may fill the observed cards, but future actions cannot fill old frames."""
    result = json.loads(json.dumps(record))
    if (result.get("selection_at", result.get("clicked_at")) or float("inf")) > timestamp:
        result.update(selected_index=None, selection_at=None, selection_source=None)
    if (result.get("clicked_at") or float("inf")) > timestamp:
        result["clicked_at"] = None
    if (result.get("confirmed_at") or float("inf")) > timestamp:
        result["confirmed_at"] = None
    if (result.get("closed_at") or float("inf")) > timestamp:
        result.update(status="observed", selected=None, closed_at=None)
        result.update(brush_effect_identity(None))
        result.update(brush_type=None, brush_code=None)
    if (result.get("resolved_at") or result.get("closed_at") or float("inf")) > timestamp:
        result.update(status="observed", selected=None, brush_type=None, brush_code=None)
        result.update(brush_effect_identity(None))
    for key in ("placement", "placement_click", "paint_observation"):
        value = result.get(key)
        if value and value.get("ts", float("inf")) > timestamp:
            result[key] = None
    return result


class EvidenceStore:
    def __init__(self, root=ROOT, *, max_bytes=DEFAULT_MAX_BYTES, mode="validation"):
        self.root = Path(root)
        self.max_bytes = max(0, int(max_bytes))
        self.lock = threading.RLock()
        self.image_bytes = sum(p.stat().st_size for p in (self.root / "images").glob("*.webp"))
        self.last_error = None
        self.requested_mode = mode
        self.mode = "validation"
        if mode == "id_only" and self.audit()["eligible"]:
            self.mode = "id_only"
        elif mode == "id_only":
            self.last_error = "id_only_requires_verified_accuracy"

    def records(self):
        result = {}
        for path in (self.root / "choices").glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                result[record["choice_id"]] = record
            except (OSError, ValueError, KeyError):
                continue
        return result

    def audit(self):
        path = self.root / "reviews.json"
        try:
            reviews = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            reviews = []
        return audit_report(self.records(), reviews if isinstance(reviews, list) else [])

    def write(self, record):
        with self.lock:
            directory = self.root / "choices"
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / (record["choice_id"] + ".json")
            temp = target.with_suffix(".tmp")
            persisted = json.loads(json.dumps(record))
            if self.mode == "id_only":
                persisted.pop("selected", None)
                persisted["samples"] = []
                for card in persisted.get("cards", []):
                    if card and card.get("stage") == "effect":
                        card.pop("name", None)
            temp.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(target)

    def save_crop(self, image):
        if self.mode == "id_only":
            return None
        ok, data = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, 90])
        if not ok:
            self.last_error = "image_encode_failed"
            return None
        blob = data.tobytes()
        name = hashlib.sha256(blob).hexdigest()[:24] + ".webp"
        with self.lock:
            target = self.root / "images" / name
            if target.exists():
                return "images/" + name
            if self.image_bytes + len(blob) > self.max_bytes:
                self.last_error = "image_budget_exhausted"
                return None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            self.image_bytes += len(blob)
        return "images/" + name


def _inside(rect, x, y):
    return rect is not None and rect.x <= x < rect.right and rect.y <= y < rect.bottom


def _rect(rect):
    return [rect.x, rect.y, rect.w, rect.h] if rect is not None else None


def _signatures(cards):
    # Ignore the icon and selection border; only text distinguishes the episodes.
    return tuple(windows_ocr.text_signature(c[round(c.shape[0]*.23):round(c.shape[0]*.77),
                                            round(c.shape[1]*.22):round(c.shape[1]*.94)],
                                          light_on_dark=True) for c in cards)


def selected_card(cards):
    """Recognize the continuous bright selection frame, not the blue artwork.

    Require three separated edges; text, ice decoration and a bright icon alone
    cannot select a card. Ratios are measured in each freshly located rectangle.
    """
    hits = []
    for index, card in enumerate(cards):
        h, w = card.shape[:2]
        if h < 20 or w < 60:
            continue
        b, g, r = cv2.split(card)
        bright = (b > 205) & (g > 205) & (r > 145)
        top = bright[:round(h*.16), round(w*.2):round(w*.95)].any(axis=0).mean()
        bottom = bright[round(h*.8):, round(w*.2):round(w*.95)].any(axis=0).mean()
        right = bright[round(h*.15):round(h*.8), round(w*.97):].any(axis=1).mean()
        if top >= .72 and bottom >= .35 and right >= .4:
            hits.append(index)
    return hits[0] if len(hits) == 1 else None


def _same_card_text(left, right):
    if windows_ocr.same_text(left, right):
        return True
    # Selected cards grow vertically to include the glowing frame. A changed
    # crop height alone must not split the offer. Allow the extra blank rows,
    # but reject an unmatched second text line (a genuinely different card).
    if left.shape[0] < right.shape[0]:
        left, right = right, left
    if left.shape[0] - right.shape[0] > 30 or min(left.std(), right.std()) < 1:
        return False
    padded = cv2.copyMakeBorder(left, 6, 6, 4, 4, cv2.BORDER_CONSTANT, value=0)
    _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(padded, right, cv2.TM_CCOEFF_NORMED))
    energy = padded.astype(np.float32) ** 2
    covered = energy[y:y+right.shape[0], x:x+right.shape[1]].sum()
    return score >= .93 and covered >= .8 * energy.sum()


class ChoiceTracker:
    """Selection state machine; independent of OCR latency and board stabilization."""
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.active = None
        self.records = {}
        self.signature = None
        self.missing_at = None
        self.geometry = None
        self.rects = ()
        self.confirm = None
        self.last_seen = 0.0
        self.geometry_since = 0.0
        self.suspended = False
        self.visual_candidate = None
        self.visual_count = 0
        self.click_diagnostics = {}

    def observe(self, cards, rects, confirm, *, geometry, session, ts):
        """Return (choice_id, should_retain_sample). Called before any OCR."""
        signature = _signatures(cards)
        with self.lock:
            previous = self.records.get(self.active)
            same = (previous is not None and previous["session"] == session and self.signature is not None
                    and len(signature) == len(self.signature)
                    and all(_same_card_text(a, b) for a, b in zip(signature, self.signature)))
            if previous is not None and not same:
                self._finish(ts, "cards_changed")
            if not same:
                key = uuid.uuid4().hex
                self.active = key
                self.records[key] = {
                    "schema": VERSION, "recognizer_revision": recognizer_revision(),
                    "choice_id": key, "session": session, "observed_at": ts,
                    "frame_size": list(geometry[-2:]), "card_rects": [_rect(r) for r in rects],
                    "stage": "unknown", "status": "observed", "selected_index": None,
                    "clicked_at": None, "confirmed_at": None, "closed_at": None,
                    "selection_at": None, "selection_source": None,
                    "previous_choice_id": previous["choice_id"] if previous and previous["session"] == session else None,
                    "source": "retained_card_ocr", "samples": [], "cards": [None, None],
                    "placement": None, "paint_observation": None, "samples_queued": 0,
                }
                self.signature = signature
                self.visual_candidate, self.visual_count = None, 0
                # Old worker results remain durable on disk; only retain a small active set.
                if len(self.records) > 128:
                    oldest = next(iter(self.records))
                    del self.records[oldest]
            if self.geometry != geometry:
                self.geometry_since = ts
            self.geometry, self.rects, self.confirm = geometry, rects, confirm
            self.last_seen, self.missing_at = ts, None
            self.suspended = False
            record = self.records[self.active]
            record["card_rects"] = [_rect(r) for r in rects]
            record["frame_size"] = list(geometry[-2:])
            self._visual_selection(record, cards, ts)
            retain = record["samples_queued"] == 0 or (
                record["samples_queued"] < 2 and ts - record["observed_at"] >= .4)
            if retain:
                record["samples_queued"] += 1
                self._write(record)
            return self.active, retain

    def _visual_selection(self, record, cards, ts):
        index = selected_card(cards)
        self.visual_count = self.visual_count + 1 if index is not None and index == self.visual_candidate else 1
        self.visual_candidate = index
        if index is None or self.visual_count < 2 or record.get("confirmed_at") is not None:
            return
        record["visual_selection"] = {"index": index, "ts": ts, "source": "three_edge_selection_frame"}
        if index != record["selected_index"]:
            record.update(selected_index=index, selection_at=ts, selection_source="selection_frame",
                          clicked_at=None)
            self._write(record)

    def suspend(self, ts):
        # Losing foreground is not a cancellation or panel transition. Preserve
        # the provisional slot; verify the same cards when sampling resumes.
        with self.lock:
            self.suspended = True
            self.missing_at = None
            self.visual_candidate, self.visual_count = None, 0

    def _click_result(self, reason):
        self.click_diagnostics[reason] = self.click_diagnostics.get(reason, 0) + 1
        return reason

    def click(self, ts, x, y, geometry):
        with self.lock:
            if (self.active is None or geometry != self.geometry or ts < self.geometry_since
                    or ts < self.records[self.active]["observed_at"]):
                return self._click_result("wrong_episode_or_geometry")
            if not self.suspended and ts - self.last_seen > 3:
                return self._click_result("stale_frame")
            record = self.records[self.active]
            hits = [i for i, rect in enumerate(self.rects) if _inside(rect, x, y)]
            if len(hits) == 1:
                if record["confirmed_at"] is not None:
                    return
                record.update(selected_index=hits[0], clicked_at=ts, confirmed_at=None,
                              selection_at=ts, selection_source="card_click")
                outcome = "card"
            elif _inside(self.confirm, x, y) and record["selected_index"] is not None:
                record.update(confirmed_at=ts, confirmation_source="confirm_click")
                outcome = "confirm"
            else:
                return self._click_result("outside_controls_or_no_selection")
            self._write(record)
            return self._click_result(outcome)

    def absent(self, ts, *, reason="panel_closed"):
        with self.lock:
            if reason == "foreground_lost":
                self.suspend(ts)
                return
            if self.active is None:
                return
            if self.missing_at is None:
                self.missing_at = ts
            elif ts - self.missing_at >= .35:
                self._finish(ts, reason)

    def boundary(self, ts, reason="session_changed"):
        with self.lock:
            self._finish(ts, reason)

    def _finish(self, ts, reason):
        if self.active is None:
            return
        record = self.records[self.active]
        record.update(closed_at=ts, close_reason=reason)
        # Timestamp the first absent frame, not the delayed debounce decision.
        if reason == "panel_closed" and self.missing_at is not None:
            record["closed_at"] = min(ts, self.missing_at)
        self._resolve(record)
        self._write(record)
        self.active, self.signature, self.missing_at = None, None, None

    def reading(self, key, cards, sample=None):
        with self.lock:
            record = self.records.get(key)
            if record is None:
                path = self.store.root / "choices" / (key + ".json")
                if not path.exists():
                    return
                record = json.loads(path.read_text(encoding="utf-8"))
            for index, card in enumerate(cards):
                previous = record["cards"][index]
                if previous and previous.get("name") and card.get("name") != previous["name"]:
                    if card.get("name"):
                        record["ocr_conflict"] = True
                if card.get("name") or previous is None:
                    record["cards"][index] = card
            known = {c["stage"] for c in record["cards"] if c and c.get("name")}
            record["stage"] = next(iter(known)) if len(known) == 1 else "unknown"
            if sample:
                record["samples"] = [s for s in record["samples"] if s["captured_at"] != sample["captured_at"]]
                record["samples"].append(sample)
            self._resolve(record)
            self._write(record)
            self._resolve_previous(record)
            for child in self.records.values():
                if child.get("previous_choice_id") == key:
                    self._resolve_previous(child)

    def _resolve_previous(self, next_record):
        """A recognized effect page proves the preceding category was accepted.

        Both effect slots must belong to the same brush type. This also works
        when the click hook missed the category click; generic disappearance
        and focus changes can never satisfy it.
        """
        previous = self.records.get(next_record.get("previous_choice_id"))
        if previous and previous.get("confirmation_source") == "stage2_cards":
            previous.update(confirmed_at=None, confirmation_source=None)
            self._resolve(previous)
            self._write(previous)
        if (not previous or previous.get("stage") != "direction"
                or next_record.get("stage") != "effect" or next_record.get("ocr_conflict")
                or previous.get("close_reason") != "cards_changed"):
            return
        rules = [resolve_affix(c.get("name")) if c else None for c in next_record["cards"]]
        if len(rules) != 2 or any(r is None for r in rules):
            return
        kinds = {r.brush_type.value for r in rules}
        indices = [i for i, c in enumerate(previous["cards"]) if c and c.get("name") in kinds]
        if len(kinds) != 1 or len(indices) != 1:
            return
        index = indices[0]
        if previous.get("selected_index") != index:
            previous.update(selection_at=previous["closed_at"], selection_source="stage2_cards", clicked_at=None)
        previous.update(selected_index=index, confirmed_at=previous["closed_at"],
                        confirmation_source="stage2_cards", next_choice_id=next_record["choice_id"])
        if previous.get("selection_at") is None:
            previous.update(selection_at=previous["closed_at"], selection_source="stage2_cards")
        self._resolve(previous)
        self._write(previous)

    def _resolve(self, record):
        record.update(brush_effect_identity(None))
        record["selected"] = None
        record.update(brush_type=None, brush_code=None)
        if record["closed_at"] is None:
            return
        record["status"] = "unconfirmed"
        index = record["selected_index"]
        if (index is None or record["confirmed_at"] is None
                or not 0 <= record["closed_at"] - record["confirmed_at"] <= 5
                or record.get("ocr_conflict")
                or record.get("close_reason") not in {"panel_closed", "cards_changed"}):
            return
        card = record["cards"][index]
        if not card or not card.get("name"):
            record["status"] = "unread"
            return
        source = record.get("confirmation_source") or "confirm_click"
        record.update(status="confirmed", selected=card["name"],
                      source=("card_click_and_confirm" if source == "confirm_click" and record.get("clicked_at")
                              else "selection_frame_and_confirm" if source == "confirm_click" else source),
                      resolved_at=record.get("resolved_at") or record["closed_at"])
        if card["stage"] == "effect":
            record.update(brush_effect_identity(card["name"]))
        elif card["stage"] == "direction":
            record.update(brush_type=card["name"], brush_code=brush_code(card["name"]))

    def _write(self, record):
        record["storage_mode"] = self.store.mode
        record["storage_warning"] = self.store.last_error
        self.store.write(record)

    def for_session(self, session):
        with self.lock:
            return json.loads(json.dumps([r for r in self.records.values() if r["session"] == session]))

    def board_click(self, session, board, ts, x, y):
        with self.lock:
            if not _inside(board, x, y):
                return
            # A fast placement can happen while the panel-close debounce is
            # pending. The board click after a confirmed card closes that episode.
            if self.active is not None:
                active = self.records[self.active]
                if self.missing_at is None or active["confirmed_at"] is None:
                    return
                self._finish(ts, "panel_closed")
            candidates = [r for r in self.records.values() if r["session"] == session
                          and r.get("confirmed_at") is not None and r.get("closed_at") is not None
                          and 0 <= ts-r["closed_at"] <= 30 and r["stage"] != "direction"]
            if not candidates:
                return
            record = candidates[-1]
            if record.get("placement") is not None or record.get("placement_click") is not None:
                return
            record["placement_click"] = {
                "row": min(5, int((y-board.y)*6/board.h)),
                "col": min(5, int((x-board.x)*6/board.w)), "ts": ts,
                "source": "board_click", "status": "unverified"}
            self._write(record)

    def observe_paint(self, session, snapshot, ts):
        """Independent text corroboration, explicitly not proof of paint identity."""
        current = snapshot.current if snapshot is not None else None
        if current is None:
            return
        reading = match_card([current.brush_effect_text]) if current.brush_effect_text else match_card([])
        with self.lock:
            matches = [r for r in self.records.values() if r["session"] == session
                       and r["stage"] == "effect" and r["closed_at"] is not None
                       and 0 <= ts - r["closed_at"] <= 30]
            if not matches:
                return
            record = matches[-1]
            # A later offer starts a different paint/decision episode.
            if any(r["session"] == session and r["observed_at"] > record["observed_at"]
                   for r in self.records.values()):
                return
            click = record.get("placement_click")
            if click and snapshot.current_anchor == (click["row"], click["col"]):
                tile = next((t for t in snapshot.tiles if (t.row, t.col) == snapshot.current_anchor), None)
                if (tile is not None and current.subtype and tile.paint_subtype == current.subtype
                        and tile.paint_category == current.category):
                    record["placement"] = {**click, "status": "associated",
                                           "source": "board_click_and_anchor_and_paint",
                                           "identity_proven": False}
            if not current.brush_effect_text:
                if record.get("placement"):
                    self._write(record)
                return
            observation = {"subtype": current.subtype, "category": current.category,
                           "level": current.level, "anchor": snapshot.current_anchor,
                           "canonical_affix_id": reading.get("canonical_affix_id"),
                           "agreement": (reading["name"] == record.get("selected"))
                           if reading["name"] and record.get("selected") else None,
                           "source": "independent_brush_section", "identity_proven": False}
            if self.store.mode == "validation":
                observation["brush_text"] = current.brush_effect_text
            if observation != record.get("paint_observation"):
                # Compare content without timestamp to avoid writing every frame.
                previous = dict(record.get("paint_observation") or {})
                previous.pop("ts", None)
                if observation == previous:
                    return
                record["paint_observation"] = {**observation, "ts": ts}
                self._write(record)


def _foreground_matches(window):
    try:
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = wintypes.HWND
        return int(user32.GetForegroundWindow() or 0) == int(window.hwnd)
    except (AttributeError, OSError):
        return False


def _confirm_rect(image, board, panel):
    # Locate the blue confirmation button below both cards. Geometry is a search
    # hint only: a click in empty space must never count as confirmation.
    x0, y0 = panel.x + round(panel.w * .40), board.y + round(board.h * .88)
    x1, y1 = min(image.shape[1], panel.right), min(image.shape[0], board.bottom + round(board.h*.015))
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (80, 110, 180), (120, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= board.w*.20 and h >= board.h*.045 and 2.2 < w/max(1,h) < 7:
            candidates.append(Rect(x+x0, y+y0, w, h))
    return max(candidates, key=lambda r:r.w*r.h) if candidates else None


def detect_cards(frame, hint=None):
    """Measure edges at a consistent scale, then map to original capture pixels."""
    from .geometry import locate_grid, refine_grid, locate_brush_panel
    scale = min(1., 1280 / frame.shape[1])
    small = cv2.resize(frame, (round(frame.shape[1]*scale), round(frame.shape[0]*scale)),
                       interpolation=cv2.INTER_AREA)
    fitted = refine_grid(small, hint) if hint is not None else None
    if fitted is None:
        fitted = locate_grid(small)
    if fitted is None:
        return None, None
    # A six-line partial grid can fit one cell above/below the real seven-line
    # board. Check adjacent lattice origins when an outer boundary is missing.
    support = fitted.line_support
    if min(support[0], support[6], support[7], support[13]) < .08:
        candidates = [fitted]
        r = fitted.rect
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            other = refine_grid(small, Rect(round(r.x+dx*r.w/6), round(r.y+dy*r.h/6), r.w, r.h))
            if other is not None:
                candidates.append(other)
        fitted = max(candidates, key=lambda g:(min(g.line_support[0], g.line_support[6],
                                                  g.line_support[7], g.line_support[13]) >= .08,
                                              g.confidence))
    found = locate_brush_panel(small, fitted.rect)
    if found is None:
        return fitted.rect, None
    panel, cards = found
    confirm = _confirm_rect(small, fitted.rect, panel)
    project = lambda r:Rect(*(round(v/scale) for v in (r.x, r.y, r.w, r.h))) if r else None
    return fitted.rect, (project(fitted.rect), tuple(project(r) for r in cards), project(confirm))


class BrushEvidenceService:
    def __init__(self, monitor, *, store=None):
        self.store = store or EvidenceStore()
        self.tracker = ChoiceTracker(self.store)
        self.monitor = monitor
        self.session = None
        self.stop_event = threading.Event()
        self.close_event = threading.Event()
        self.jobs = queue.Queue(maxsize=16)
        self.thread = None
        self.worker = None
        self.last_error = None
        self.last_click = 0.0
        self._board = None
        self._board_geometry = None
        self._recovered_samples = set()
        self._recovery_pending = True

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.clear()
            return
        self.stop_event.clear()
        self.close_event.clear()
        if self.worker is None or not self.worker.is_alive():
            self._recover_pending()
            self.worker = threading.Thread(target=self._work, name="BrushEvidenceOCR", daemon=True)
            self.worker.start()
        self.thread = threading.Thread(target=self._sample, name="BrushEvidenceCapture", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=.6)
        try:
            self.tracker.boundary(time.time(), "capture_stopped")
        except OSError as exc:
            self.last_error = f"storage:{type(exc).__name__}:{exc}"

    def close(self):
        self.stop()
        self.close_event.set()
        if self.worker is not None:
            self.worker.join(timeout=5)

    def context(self, session, snapshot):
        with self.tracker.lock:
            try:
                if self.session != session:
                    self.tracker.boundary(time.time())
                    self.session = session
                self.tracker.observe_paint(session, snapshot, time.time())
            except OSError as exc:
                self.last_error = f"storage:{type(exc).__name__}:{exc}"
                # A full/unwritable evidence disk cannot stall the board recorder
                # or leave future frames attached to the previous session.
                self.session = session
                self.tracker.active, self.tracker.signature = None, None

    def evidence(self, session):
        # Crops/raw OCR stay in the sidecar once; do not copy them to every frame.
        fields = ("choice_id", "session", "stage", "status", "source", "observed_at", "closed_at",
                  "selected_index", "selected", "canonical_affix_id", "brush_effect_id", "brush_effect_ids",
                  "clicked_at", "confirmed_at",
                  "selection_at", "selection_source", "confirmation_source", "resolved_at",
                  "brush_type", "brush_code", "cards", "next_choice_id",
                  "brush_effect_id_status", "brush_effect_id_source", "brush_effect_table",
                  "paint_observation", "placement", "placement_click")
        choices = [{k:v for k,v in r.items() if k in fields}
                   for r in self.tracker.for_session(session)]
        if self.store.mode == "id_only":
            for record in choices:
                record.pop("selected", None)
        return {"schema": VERSION, "choices": choices,
                "sidecar": str(self.store.root / "choices"),
                "mode": self.store.mode, "requested_mode": self.store.requested_mode,
                "image_bytes": self.store.image_bytes, "max_image_bytes": self.store.max_bytes,
                "input": self.monitor.diagnostics(),
                "click_handling": dict(self.tracker.click_diagnostics),
                "error": self.last_error or self.store.last_error}

    def _sample(self):
        from .capture import find_capture_window, capture_screen, physical_pixel_coordinates
        from .frame_pump import same_window_geometry
        hint = None
        board = None
        last_geometry = None
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                if self.session is None:
                    self.stop_event.wait(.2)
                    continue
                session = self.session
                window = find_capture_window()
                if window is None or not window.captureable or not _foreground_matches(window):
                    self.tracker.suspend(time.time())
                    self.stop_event.wait(.2)
                    continue
                geometry = (window.hwnd, window.left, window.top, window.width, window.height)
                # Independent MSS context, no concurrent access to WGC/DXCam sessions.
                with physical_pixel_coordinates():
                    frame = capture_screen({"left": window.left, "top": window.top,
                                            "width": window.width, "height": window.height})
                ts = time.time()
                if not _foreground_matches(window) or not same_window_geometry(window, find_capture_window()):
                    continue
                if geometry != last_geometry:
                    hint, board = None, None
                last_geometry = geometry
                hint, found = detect_cards(frame, hint)
                # Non-consuming click feed; board recorder keeps its own queue.
                clicks = [c for c in self.monitor.since(self.last_click) if c.ts <= ts]
                if hint is not None:
                    scale = min(1., 1280/frame.shape[1])
                    board = Rect(*(round(v/scale) for v in (hint.x, hint.y, hint.w, hint.h)))
                else:
                    board = None
                self._accept_sample(frame, found, geometry, ts, session, clicks, board=board)
            except Exception as exc:
                self.last_error = f"capture:{type(exc).__name__}:{exc}"
            self.stop_event.wait(max(.01, .12-(time.monotonic()-started)))

    def _accept_sample(self, frame, found, geometry, ts, session, clicks, *, board=None):
        """Process a capture only under its original session and window geometry."""
        with self.tracker.lock:
            if session != self.session or session is None or self.stop_event.is_set():
                return
            if found:
                board = found[0]
            if board is not None:
                self._board, self._board_geometry = board, geometry
            else:
                self._board, self._board_geometry = None, None
            if not found and self.tracker.active is not None and self.tracker.missing_at is None:
                self.tracker.absent(ts)
            # Re-establish the card identity after returning to the game before
            # accepting a click that happened between focus gain and this frame.
            preobserved = None
            cards = None
            if found and self.tracker.suspended:
                _, rects, confirm = found
                cards = tuple(frame[r.y:r.bottom, r.x:r.right].copy() for r in rects)
                preobserved = self.tracker.observe(cards, rects, confirm, geometry=geometry, session=session, ts=ts)
            for click in clicks:
                self.last_click = max(self.last_click, click.ts)
                if click.window_geometry != geometry:
                    self.tracker._click_result("capture_geometry_mismatch")
                    continue
                x, y = click.x-geometry[1], click.y-geometry[2]
                # click() also compares against the geometry of the last seen
                # cards. Resize-period clicks never reuse old rectangles.
                self.tracker.click(click.ts, x, y, geometry)
                if self._board_geometry == geometry:
                    self.tracker.board_click(session, self._board, click.ts, x, y)
            if not found:
                self.tracker.absent(ts)
                return
            _, rects, confirm = found
            if cards is None:
                cards = tuple(frame[r.y:r.bottom, r.x:r.right].copy() for r in rects)
            key, retain = preobserved or self.tracker.observe(cards, rects, confirm, geometry=geometry, session=session, ts=ts)
            if retain:
                images = [self.store.save_crop(card) for card in cards]
                self.tracker.reading(key, [], {"captured_at": ts, "images": images, "ocr_pending": True})
                try:
                    self.jobs.put_nowait((key, cards, ts, images))
                except queue.Full:
                    self.last_error = "ocr_queue_full"
                    self._recovery_pending = True

    def _work(self):
        while not self.close_event.is_set() or not self.jobs.empty():
            try:
                key, cards, ts, images = self.jobs.get(timeout=.2)
            except queue.Empty:
                if not self.close_event.is_set() and self._recovery_pending:
                    self._recover_pending()
                continue
            try:
                results, texts = [], []
                for card in cards:
                    result, attempts = read_card(card)
                    results.append(result)
                    texts.append(attempts)
                sample = {"captured_at": ts, "images": images}
                if self.store.mode == "validation":
                    sample["ocr_lines"] = texts
                self.tracker.reading(key, results, sample)
            except Exception as exc:
                self.last_error = f"ocr:{type(exc).__name__}:{exc}"
            finally:
                self.jobs.task_done()

    def _recover_pending(self):
        self._recovery_pending = False
        for record in self.store.records().values():
            if record.get("recognizer_revision") != recognizer_revision():
                continue
            for sample in record.get("samples", []):
                images = sample.get("images") or []
                sample_key = (record["choice_id"], sample.get("captured_at"))
                if not sample.get("ocr_pending") or len(images) != 2 or not all(images):
                    continue
                if sample_key in self._recovered_samples:
                    continue
                try:
                    cards = tuple(cv2.imdecode(np.fromfile(str(self.store.root / name), np.uint8),
                                               cv2.IMREAD_COLOR) for name in images)
                    if any(card is None for card in cards):
                        continue
                    self.jobs.put_nowait((record["choice_id"], cards, sample["captured_at"], images))
                    self._recovered_samples.add(sample_key)
                except queue.Full:
                    self.last_error = "pending_recovery_queue_full"
                    self._recovery_pending = True
                    return
                except (OSError, ValueError):
                    self.last_error = "pending_crop_unreadable"
