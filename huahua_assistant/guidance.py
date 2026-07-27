from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class GamePhase(str, Enum):
    WAITING = "waiting"
    INITIAL_MAP = "initial_map"
    BRUSH_STAGE1 = "brush_stage1"
    BRUSH_STAGE2 = "brush_stage2"
    BACKPACK_SELECT = "backpack_select"
    BOARD_PLACE = "board_place"
    FINISH_READY = "finish_ready"
    UNSTABLE = "unstable"


class TargetKind(str, Enum):
    START = "start"
    BOARD = "board"
    BACKPACK = "backpack"
    BRUSH_STAGE1 = "brush_stage1"
    BRUSH_STAGE2 = "brush_stage2"
    FINISH = "finish"


@dataclass(frozen=True)
class PhaseEvidence:
    frame_id: str
    board_detected: bool
    board_signature: str = ""
    board_confidence: float = 0.0
    quality_ok: bool = True
    conflicting: bool = False
    current_rule_present: bool = False
    current_rule_pending: bool = False
    brush_visible: bool = False
    brush_stage2_visible: bool = False
    brush_type_selected: bool = False
    affix_selected: bool = False
    backpack_target: bool = False
    board_target: bool = False
    start_visible: bool = False
    finish_signals: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PhaseState:
    phase: GamePhase
    observed_phase: GamePhase
    stable: bool
    confidence: float
    source_frame_id: str
    reason: str = ""
    last_trusted_phase: GamePhase = GamePhase.WAITING


@dataclass(frozen=True)
class TargetProposal:
    key: str
    kind: TargetKind
    priority: int
    screen_point: tuple[int, int]
    confidence: float
    reason_short: str
    source_frame_id: str


@dataclass(frozen=True)
class RecommendationTarget:
    key: str
    kind: TargetKind
    priority: int
    screen_point: tuple[int, int]
    confidence: float
    reason_short: str
    source_frame_id: str
    stable_since: float
    expires_at: float

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "kind": self.kind.value,
            "priority": self.priority,
            "screen_point": [self.screen_point[0], self.screen_point[1]],
            "confidence": self.confidence,
            "reason_short": self.reason_short,
            "source_frame_id": self.source_frame_id,
            "stable_since": self.stable_since,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class GuidanceState:
    phase: PhaseState
    target: RecommendationTarget | None


_LEGAL_TRANSITIONS: dict[GamePhase, frozenset[GamePhase]] = {
    GamePhase.WAITING: frozenset(
        {
            GamePhase.INITIAL_MAP,
            GamePhase.BRUSH_STAGE1,
            GamePhase.BRUSH_STAGE2,
            GamePhase.BACKPACK_SELECT,
            GamePhase.BOARD_PLACE,
            GamePhase.FINISH_READY,
        }
    ),
    GamePhase.INITIAL_MAP: frozenset(
        {
            GamePhase.BRUSH_STAGE1,
            GamePhase.BACKPACK_SELECT,
            GamePhase.BOARD_PLACE,
            GamePhase.FINISH_READY,
        }
    ),
    GamePhase.BRUSH_STAGE1: frozenset(
        {GamePhase.BRUSH_STAGE2, GamePhase.INITIAL_MAP, GamePhase.FINISH_READY}
    ),
    GamePhase.BRUSH_STAGE2: frozenset(
        {
            GamePhase.BRUSH_STAGE1,
            GamePhase.BACKPACK_SELECT,
            GamePhase.BOARD_PLACE,
            GamePhase.FINISH_READY,
        }
    ),
    GamePhase.BACKPACK_SELECT: frozenset(
        {
            GamePhase.BOARD_PLACE,
            GamePhase.INITIAL_MAP,
            GamePhase.BRUSH_STAGE1,
            GamePhase.FINISH_READY,
        }
    ),
    GamePhase.BOARD_PLACE: frozenset(
        {
            GamePhase.BACKPACK_SELECT,
            GamePhase.INITIAL_MAP,
            GamePhase.BRUSH_STAGE1,
            GamePhase.FINISH_READY,
        }
    ),
    GamePhase.FINISH_READY: frozenset({GamePhase.WAITING, GamePhase.INITIAL_MAP}),
    GamePhase.UNSTABLE: frozenset(),
}


class GamePhaseStateMachine:
    def __init__(self, *, stable_frames: int = 2, min_board_confidence: float = 0.5) -> None:
        self.stable_frames = max(1, int(stable_frames))
        self.min_board_confidence = float(min_board_confidence)
        self._confirmed_phase = GamePhase.WAITING
        self._confirmed_key: tuple[object, ...] | None = None
        self._candidate_phase: GamePhase | None = None
        self._candidate_key: tuple[object, ...] | None = None
        self._candidate_frames: set[str] = set()

    @property
    def last_trusted_phase(self) -> GamePhase:
        return self._confirmed_phase

    def update(self, evidence: PhaseEvidence) -> PhaseState:
        observed, reason = _phase_from_evidence(evidence, self.min_board_confidence)
        confidence = max(0.0, min(1.0, float(evidence.board_confidence)))
        if observed == GamePhase.UNSTABLE:
            self._clear_candidate()
            return PhaseState(
                phase=GamePhase.UNSTABLE,
                observed_phase=observed,
                stable=False,
                confidence=confidence,
                source_frame_id=evidence.frame_id,
                reason=reason,
                last_trusted_phase=self._confirmed_phase,
            )

        key = _phase_stability_key(observed, evidence)
        if observed == self._confirmed_phase and key == self._confirmed_key:
            self._clear_candidate()
            return PhaseState(
                phase=observed,
                observed_phase=observed,
                stable=True,
                confidence=confidence,
                source_frame_id=evidence.frame_id,
                last_trusted_phase=self._confirmed_phase,
            )

        if not self._transition_is_legal(self._confirmed_phase, observed):
            self._clear_candidate()
            return PhaseState(
                phase=GamePhase.UNSTABLE,
                observed_phase=observed,
                stable=False,
                confidence=confidence,
                source_frame_id=evidence.frame_id,
                reason=f"illegal_transition:{self._confirmed_phase.value}->{observed.value}",
                last_trusted_phase=self._confirmed_phase,
            )

        if observed != self._candidate_phase or key != self._candidate_key:
            self._candidate_phase = observed
            self._candidate_key = key
            self._candidate_frames = {evidence.frame_id}
        else:
            self._candidate_frames.add(evidence.frame_id)

        if len(self._candidate_frames) >= self.stable_frames:
            self._confirmed_phase = observed
            self._confirmed_key = key
            self._clear_candidate()
            return PhaseState(
                phase=observed,
                observed_phase=observed,
                stable=True,
                confidence=confidence,
                source_frame_id=evidence.frame_id,
                last_trusted_phase=observed,
            )

        return PhaseState(
            phase=GamePhase.UNSTABLE,
            observed_phase=observed,
            stable=False,
            confidence=confidence,
            source_frame_id=evidence.frame_id,
            reason="phase_transition_pending",
            last_trusted_phase=self._confirmed_phase,
        )

    def _transition_is_legal(self, previous: GamePhase, current: GamePhase) -> bool:
        return current == previous or current in _LEGAL_TRANSITIONS.get(previous, frozenset())

    def _clear_candidate(self) -> None:
        self._candidate_phase = None
        self._candidate_key = None
        self._candidate_frames.clear()


class RecommendationTargetStabilizer:
    def __init__(
        self,
        *,
        stable_frames: int = 2,
        min_stable_seconds: float = 0.3,
        ttl_seconds: float = 2.0,
        min_confidence: float = 0.75,
    ) -> None:
        self.stable_frames = max(1, int(stable_frames))
        self.min_stable_seconds = max(0.0, float(min_stable_seconds))
        self.ttl_seconds = max(0.1, float(ttl_seconds))
        self.min_confidence = float(min_confidence)
        self._context_signature: tuple[object, ...] | None = None
        self._candidate: TargetProposal | None = None
        self._candidate_since = 0.0
        self._candidate_frames: set[str] = set()
        self._current: RecommendationTarget | None = None

    def update(
        self,
        proposal: TargetProposal | None,
        *,
        phase: GamePhase,
        context_signature: tuple[object, ...],
        now: float,
    ) -> RecommendationTarget | None:
        if context_signature != self._context_signature:
            self._context_signature = context_signature
            self._candidate = None
            self._candidate_frames.clear()
            self._current = None

        if self._current is not None and self._current.is_expired(now):
            self._current = None

        if phase in {GamePhase.WAITING, GamePhase.UNSTABLE}:
            self._candidate = None
            self._candidate_frames.clear()
            return None
        if proposal is None or proposal.confidence < self.min_confidence:
            self._candidate = None
            self._candidate_frames.clear()
            return None
        if not _target_matches_phase(proposal.kind, phase):
            self._candidate = None
            self._candidate_frames.clear()
            return None

        if self._current is not None and _same_target(self._current, proposal):
            if self._current.source_frame_id == proposal.source_frame_id:
                return self._current
            self._current = replace(
                self._current,
                confidence=max(self._current.confidence, proposal.confidence),
                source_frame_id=proposal.source_frame_id,
                expires_at=now + self.ttl_seconds,
            )
            return self._current

        if self._candidate is None or not _same_proposal(self._candidate, proposal):
            self._candidate = proposal
            self._candidate_since = now
            self._candidate_frames = {proposal.source_frame_id}
            return None

        self._candidate = proposal
        self._candidate_frames.add(proposal.source_frame_id)
        stable_for = now - self._candidate_since
        if len(self._candidate_frames) < self.stable_frames or stable_for < self.min_stable_seconds:
            return None

        self._current = RecommendationTarget(
            key=proposal.key,
            kind=proposal.kind,
            priority=proposal.priority,
            screen_point=proposal.screen_point,
            confidence=proposal.confidence,
            reason_short=proposal.reason_short,
            source_frame_id=proposal.source_frame_id,
            stable_since=self._candidate_since,
            expires_at=now + self.ttl_seconds,
        )
        self._candidate = None
        self._candidate_frames.clear()
        return self._current

    def active(
        self,
        *,
        context_signature: tuple[object, ...],
        now: float,
    ) -> RecommendationTarget | None:
        if context_signature != self._context_signature:
            return None
        if self._current is None or self._current.is_expired(now):
            return None
        return self._current


class GuidanceController:
    def __init__(self) -> None:
        self.phases = GamePhaseStateMachine()
        self.targets = RecommendationTargetStabilizer()

    def update(
        self,
        evidence: PhaseEvidence,
        proposal: TargetProposal | None,
        *,
        context_signature: tuple[object, ...],
        now: float,
    ) -> GuidanceState:
        phase = self.phases.update(evidence)
        target = self.targets.update(
            proposal,
            phase=phase.observed_phase,
            context_signature=context_signature,
            now=now,
        )
        if not phase.stable:
            target = None
        return GuidanceState(phase=phase, target=target)


def _phase_from_evidence(
    evidence: PhaseEvidence,
    min_board_confidence: float,
) -> tuple[GamePhase, str]:
    if not evidence.board_detected:
        return GamePhase.WAITING, "board_missing"
    if not evidence.quality_ok or evidence.conflicting:
        return GamePhase.UNSTABLE, "recognition_unstable"
    if evidence.board_confidence < min_board_confidence:
        return GamePhase.UNSTABLE, "board_confidence_low"
    if len(evidence.finish_signals) >= 2:
        return GamePhase.FINISH_READY, ""
    if evidence.start_visible:
        return GamePhase.INITIAL_MAP, ""
    if evidence.current_rule_pending or evidence.brush_visible:
        if not evidence.brush_type_selected and not evidence.brush_stage2_visible:
            if evidence.brush_visible:
                return GamePhase.BRUSH_STAGE1, ""
            return GamePhase.INITIAL_MAP, ""
        if not evidence.affix_selected:
            return GamePhase.BRUSH_STAGE2, ""
    if evidence.backpack_target:
        return GamePhase.BACKPACK_SELECT, ""
    if evidence.board_target:
        return GamePhase.BOARD_PLACE, ""
    if evidence.current_rule_present:
        return GamePhase.INITIAL_MAP, ""
    return GamePhase.WAITING, "no_actionable_state"


def _phase_stability_key(phase: GamePhase, evidence: PhaseEvidence) -> tuple[object, ...]:
    return (
        phase,
        evidence.board_signature,
        evidence.current_rule_pending,
        evidence.brush_visible,
        evidence.brush_stage2_visible,
        evidence.brush_type_selected,
        evidence.affix_selected,
        evidence.backpack_target,
        evidence.board_target,
        evidence.start_visible,
        tuple(sorted(evidence.finish_signals)),
    )


def _target_matches_phase(kind: TargetKind, phase: GamePhase) -> bool:
    if kind == TargetKind.START:
        return phase == GamePhase.INITIAL_MAP
    if kind == TargetKind.BOARD:
        return phase in {GamePhase.INITIAL_MAP, GamePhase.BOARD_PLACE}
    return {
        TargetKind.BACKPACK: GamePhase.BACKPACK_SELECT,
        TargetKind.BRUSH_STAGE1: GamePhase.BRUSH_STAGE1,
        TargetKind.BRUSH_STAGE2: GamePhase.BRUSH_STAGE2,
        TargetKind.FINISH: GamePhase.FINISH_READY,
    }[kind] == phase


def _same_target(current: RecommendationTarget, proposal: TargetProposal) -> bool:
    return current.key == proposal.key and current.kind == proposal.kind


def _same_proposal(current: TargetProposal, proposal: TargetProposal) -> bool:
    return current.key == proposal.key and current.kind == proposal.kind
