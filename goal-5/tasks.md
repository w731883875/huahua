# Goal 5 Tasks

## Task 1: Create The Optimization Design Document

Status: completed

Create a standalone Chinese optimization document grounded in the current code, workbook assets, runtime sample quality, target architecture, data schema, learning stages, privacy, acceptance criteria, and rollout order.

Work performed:

- Added the standalone Chinese design document `桌宠辅助项目优化设计.md`.
- Documented product boundaries, the current implementation/data baseline, target architecture, explicit game phases, recognition confidence rules, canonical affix identity, layered recommendation utility, Top1 desktop-pet teleport behavior, outcome/proxy/policy score separation, risk metrics, batch evaluation, multi-user privacy/security, engineering tests, rollout stages, acceptance criteria, and rollback expectations.
- Kept new screenshot collection as the final diagnostic stage and specified that existing workbook/PDF/runtime assets must be exhausted first.

Validation evidence:

- Document inspection: 424 lines, 15 main sections, with explicit sections for `RecommendationTarget`, three separate score types, upload disabled by default, and final screenshot gap assessment.
- Static code evidence confirmed the documented active entry points in `vision.py`, `game_model.py`, `desktop_pet.py`, and `learning.py`.
- Runtime rebuild evidence confirmed 4 trusted trajectories, 14 imitation actions, 0 brush samples, 0 affix samples, 0 recommendation decisions, and 0 compressed summaries.
- Rule/workbook evidence confirmed 55 current catalog entries and 30 embedded workbook images.
- `python scripts\check_encoding.py .` checked 48 text files successfully.

Remaining risk:

- The document describes the target behavior; most of the new target-state, pet movement, risk, evaluation, and multi-user contracts are not implemented yet.
- The workbook image count is verified, but canonical identity for its 28 affix screenshots will be established in Task 4.
- Real-game visual accuracy still needs final-stage screenshot regression evidence after non-image-dependent work is complete.

Next step:

- Task 2: establish the automated test and reproducible environment baseline before changing business logic.

## Task 2: Establish Test And Environment Baseline

Status: pending

Add a lightweight automated test foundation for rule catalogs, learning trust gates, recommendation target behavior, and compatibility helpers; document a reproducible Python environment without overwriting the user's runtime data.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 3: Major Checkpoint 1

Status: pending

Review requirement drift, documentation accuracy, environment reproducibility, test health, dirty-worktree safety, build implications, and rollback before business logic changes.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 4: Canonicalize Affix IDs And Recognition Aliases

Status: pending

Separate canonical affix rules from OCR/template aliases so equivalent workbook descriptions aggregate under one stable code while old logs and exported bundles remain readable.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 5: Add Stable Game Phase And Recommendation Target Model

Status: pending

Introduce explicit phases and a structured Top1 target with confidence, coordinates, source frame, stability/expiry metadata, and hysteresis so recognition changes cannot make the pet oscillate.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 6: Major Checkpoint 2

Status: pending

Review canonical data compatibility, phase transitions, target stability, tests, event/data consistency, UI integration risk, and rollback before changing desktop-pet movement.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 7: Implement Top1 Desktop-Pet Teleport Guidance

Status: pending

Make the desktop-pet actor move beside the stable Top1 board/backpack/brush/finish target, point precisely without covering the click area, remain click-through during play, and retain lightweight Top2/Top3 markers.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 8: Add Outcome And Risk Learning Schema

Status: pending

Version decision and run records to distinguish observed game outcomes from proxy scores, retain ordinary/failure summaries, log candidate sets and chosen actions, and compute stable/balanced/gamble risk metrics without premature labeling.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 9: Major Checkpoint 3

Status: pending

Review real-data-only boundaries, privacy, legacy JSONL compatibility, target/pet behavior, outcome integrity, risk metrics, tests, build, and rollback.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 10: Add Reproducible Batch And Resource Evaluation

Status: pending

Implement deterministic batch simulation reports for rule policies, Top1/Top3 behavior, score distributions, downside metrics, and force/refresh A/B comparisons using fixed seeds and configuration snapshots.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 11: Add Privacy-Ready Multi-User Data Contract

Status: pending

Implement a versioned, validated, deduplicated, opt-in data bundle/upload queue contract plus a local reference aggregator; keep remote upload disabled and exclude screenshots by default.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 12: Major Checkpoint 4

Status: pending

Review evaluation determinism, multi-user validation/security, consent defaults, data weighting, expert thresholds, policy compatibility, docs, tests, build, and rollback.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 13: Final Review And Screenshot Gap Assessment

Status: pending

Perform the largest end-to-end review, fix remaining high-risk issues, update user-facing documentation and completion records, and only then produce a minimal prioritized screenshot capture list for recognition cases that existing assets cannot resolve.

Work performed:

Validation evidence:

Remaining risk:

Next step:
