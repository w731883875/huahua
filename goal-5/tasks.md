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

Status: completed

Add a lightweight automated test foundation for rule catalogs, learning trust gates, recommendation target behavior, and compatibility helpers; document a reproducible Python environment without overwriting the user's runtime data.

Work performed:

- Added a standard-library `unittest` suite covering rule catalog/legacy-code round trips, compact payload compatibility, real-only learning trust gates, deduplication, legacy trajectory readability, pet target precedence, and unstable-capture freezing.
- Added `scripts/check_environment.py` with strict and report-only modes, JSON output, supported Python checks, Tk detection, dependency minimums, and required-project-file checks. The checker does not import the heavy packages or read/write runtime learning data.
- Added `开发与测试指南.md` with a reproducible Python 3.12/3.13 virtual-environment workflow, test commands, strict environment verification, source/build consistency, and runtime data safety rules.
- Added a minimal `.gitignore` for `.venv`, Python caches, and test coverage artifacts without ignoring runtime data or screenshots.
- Created `.venv` with Python 3.12 and installed the existing `requirements.txt`; the environment is local and ignored by Git.
- Code commit: `1d40bd8 Add test and environment baseline`.

Validation evidence:

- `.\.venv\Scripts\python.exe -B scripts\check_environment.py` passed with Python 3.12.10, Tk, NumPy 2.5.1, OpenCV 5.0.0.93, MSS 10.2.0, Pillow 12.3.0, and PyInstaller 6.21.0.
- `.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v` ran 18 tests successfully in 0.037 seconds.
- Core imports succeeded for OpenCV, NumPy, MSS, Pillow, `HuahuaApp`, and the desktop-pet patch.
- `.\.venv\Scripts\python.exe -B -m huahua_assistant simulate --seed 42 --brush --refresh` completed and produced board, install, placement, brush, force, and refresh recommendations.
- The global Python diagnostic intentionally returned strict exit code 1 when dependencies were missing; `--report-only` returned 0. This proves the checker distinguishes reporting from enforcement.
- AST checks passed for all new Python files, and `python scripts\check_encoding.py .` checked 54 text files successfully.
- `git diff --cached --check` passed before the code commit.

Remaining risk:

- Tests characterize pure decision/data behavior; they do not yet prove live Windows capture, Tk click-through, desktop-pet movement, or recognition accuracy.
- OpenCV 5 satisfies the declared lower bound but is newer than the previously used 4.x line; full visual regression and packaging checks remain necessary.
- Existing tracked `__pycache__` changes and other dirty files predate this task and were intentionally not modified or committed.

Next step:

- Task 3: run the first major checkpoint before changing affix identities and business logic.

## Task 3: Major Checkpoint 1

Status: completed

Review requirement drift, documentation accuracy, environment reproducibility, test health, dirty-worktree safety, build implications, and rollback before business logic changes.

Work performed:

- Reviewed the original request, Goal 5 plan, optimization design, Task 1/2 outputs, current code entry points, runtime data boundaries, and dirty worktree before business-logic changes.
- Confirmed there is no requirement drift: passive assistance, Top1 pet guidance, risk modes, opt-in multi-user data, and screenshot-last sequencing remain intact.
- Clarified in `桌宠辅助项目优化设计.md` that the old v2.1 PyQt5/PaddleOCR/SQLite document is a mechanism/legacy-plan reference, not the current implemented stack.
- Added `requirements-lock.txt` for the exact Windows CPython 3.12 environment validated at this checkpoint and changed the development guide to prefer the lock over loose minimum ranges.
- Extended strict environment validation to require the lock file and verify every direct/transitive locked distribution at its exact version.
- Fixed the existing encoding checker to skip `.venv` and common test caches; added regression tests for encoding exclusions and lock parsing.
- Audited passive/safe behavior: no input-injection or network/upload calls were found in application code, and three learning write entry points reject non-`real_capture` sources.
- Built the application with PyInstaller into a unique system temporary directory so the project's existing `build/`, `dist/`, spec, exe, and runtime data were not overwritten.
- Checkpoint fix commit: `1237f2e Harden checkpoint environment validation`.

Validation evidence:

- Strict environment check passed on Python 3.12.10 with all 11 lock packages matching exactly; `pip check` reported no broken requirements and lock dry-run found all packages satisfied.
- Full suite passed: 20 tests in 0.048 seconds after checkpoint fixes.
- Fixed-seed simulation still produced placement, brush, force, and refresh recommendations.
- Encoding check inspected 56 project text files successfully after `.venv` exclusion; script/test AST parsing passed.
- Temporary PyInstaller build succeeded with PyInstaller 6.21.0 and Python 3.12.10, producing a 75,738,817-byte `HuahuaAssistant.exe` with SHA256 `1122E82447B99104C85914D8654F965FE251DB71EABFE5FE66028E4F0FE96DCD`.
- Build TOC contained pet asset evidence, and the warning audit found zero missing `cv2`, `mss`, `PIL`, `tkinter`, or `huahua_assistant` modules.
- SHA256 values for all runtime JSON/JSONL files were unchanged across tests and the temporary build.
- `git diff --check` reported no whitespace errors in task changes; only the pre-existing CRLF conversion warning for `runtime/desktop_pet_position.json` remains.
- Dirty-worktree audit confirmed checkpoint commits excluded existing changes in README, spec, `app.py`, `config.py`, `game_model.py`, `learning.py`, runtime position, build/install scripts, caches, and workspace helpers.

Remaining risk:

- The temporary exe was inspected but not launched because doing so would open/steal focus from a GUI; live startup remains a later manual/runtime gate.
- Live game capture, recognition accuracy, Tk click-through, and pet visibility have not yet been proven on the user's display configuration.
- Task 4 overlaps `game_model.py`, which already has substantial pre-existing uncommitted changes; implementation must preserve and test those changes rather than replace the file.
- Legacy install/build scripts still install loose `requirements.txt` ranges. The authoritative reproducible workflow is currently the explicit lock command in `开发与测试指南.md`.
- Existing tracked Python 3.13 cache changes remain dirty and were intentionally left untouched.
- The unique temporary checkpoint build directory remains under the system temp directory because recursive cleanup was rejected by the execution safety policy; it does not affect the project worktree.

Next step:

- Task 4: introduce canonical affix identities and aliases with backward-compatible legacy codes and logs.

## Task 4: Canonicalize Affix IDs And Recognition Aliases

Status: completed

Separate canonical affix rules from OCR/template aliases so equivalent workbook descriptions aggregate under one stable code while old logs and exported bundles remain readable.

Work performed:

- Extended `AffixRule` with a stable `canonical_affix_id`, a compatibility `display_name`, recognition aliases, and all historical `1-n`/`2-n`/`3-n` codes.
- Canonicalized the 55 legacy text entries into 48 semantic rules by full effect parameters. Seven duplicate long/short descriptions remain available as aliases and all 55 historical codes remain resolvable.
- Pinned all 48 canonical identities in an explicit registry instead of deriving them from implementation names; `stack.upgrade_scope.random_1.lv1` now remains stable across internal enum/field refactors.
- Added canonical lookup helpers that accept canonical IDs, legacy codes, display names, aliases, normalized punctuation/spacing, and optional brush-type qualification.
- Kept existing `stage2_codes` fields and added `stage2_ids` plus `canonical_affix_id` to new structured records; no runtime JSONL migration or rewrite is required.
- Updated brush recognition to score every known alias but emit only the canonical display name, preventing duplicate semantic candidates.
- Updated learning statistics and policy priors to aggregate by canonical identity first while retaining legacy name/code counters for existing reports and exports.

Validation evidence:

- Full suite passed: 26 tests, including all 55 legacy-code round trips, unique and explicitly pinned canonical identities, normalized OCR alias lookup, known long/short alias merging, payload compatibility, and legacy-event aggregation.
- Strict environment verification passed on Python 3.12.10 with all 11 locked packages matching exactly.
- Fixed-seed simulation with brush and refresh recommendations completed successfully after catalog canonicalization.
- Read-only rebuild of current runtime logs produced 295 real events and 4 trusted trajectories; 4 observed affix entries normalized to canonical IDs while policy-affix samples remained 0, so no experience prior was enabled prematurely.
- `python scripts\check_encoding.py .` checked 56 project text files successfully; package and test compile checks passed.
- `git diff --check` found no whitespace errors in task changes; only the pre-existing runtime CRLF warning remains.

Remaining risk:

- Canonical grouping is based on the currently modeled effect parameters. If future workbook evidence proves two identical parameter sets have distinct hidden behavior, they must receive separate explicit identities.
- Alias recognition remains heuristic and has not been measured against a representative live screenshot regression set; this stays deferred to the final screenshot-gap audit.
- Existing runtime records contain no trusted affix-choice decisions, so compatibility is proven structurally but experience scoring remains intentionally inactive.

Next step:

- Task 5: add the explicit game-phase state machine and stable `RecommendationTarget` model with confidence, hysteresis, and expiry.

## Task 5: Add Stable Game Phase And Recommendation Target Model

Status: completed

Introduce explicit phases and a structured Top1 target with confidence, coordinates, source frame, stability/expiry metadata, and hysteresis so recognition changes cannot make the pet oscillate.

Work performed:

- Added `guidance.py` with explicit `WAITING`, `INITIAL_MAP`, `BRUSH_STAGE1`, `BRUSH_STAGE2`, `BACKPACK_SELECT`, `BOARD_PLACE`, `FINISH_READY`, and `UNSTABLE` phases plus documented legal transitions.
- Added frame-based phase confirmation: phase and key evidence require two distinct frames; low-confidence/conflicting frames expose `UNSTABLE` without overwriting the last trusted phase, and illegal jumps are rejected.
- Added the structured `RecommendationTarget` contract with target kind, screen point, priority, confidence, short reason, source frame, stable time, expiry time, and JSON-compatible serialization.
- Added a target stabilizer requiring two distinct frames, at least 300ms, confidence >= 0.75, matching phase, and an unchanged window/capture/board context. Stable targets expire after two seconds, same-key coordinate jitter does not move them, and context changes invalidate them immediately.
- Integrated phase evidence and Top1 proposal generation into `pet_brain.py` for board, backpack, brush stage 1/2, and finish targets. Existing immediate pet modes remain compatible; Task 5 only exposes stable target data and does not move the desktop-pet window.
- Added deterministic frame IDs per snapshot object so repeated UI renders cannot masquerade as multiple capture frames.
- Code commit: `4e42aa4 Add stable game phase and recommendation targets`.

Validation evidence:

- Full suite passed: 35 tests in 0.073 seconds.
- Fifteen focused guidance/pet tests passed, covering two-frame confirmation, key-field stability, illegal transitions, low-confidence freeze/recovery, two-category finish evidence, 300ms target latency, expiry, context invalidation, same-key jitter suppression, serialization, and a two-snapshot `pet_brain` integration flow.
- Strict environment validation passed on Python 3.12.10 with all 11 locked packages exact; encoding validation checked 58 text files successfully.
- Core imports passed for the Tk app, desktop-pet patch, guidance controller, and recommendation target types.
- Fixed-seed CLI simulation with brush and refresh recommendations exited 0.
- `git diff --cached --check` passed, and the Task 5 commit contained only `guidance.py`, `pet_brain.py`, and their tests; pre-existing dirty files remained outside the commit.

Remaining risk:

- Live capture has not yet proven phase/target timing across real window movement, DPI changes, and brush/finish transitions; pure tests cover the intended contracts but visual/runtime evidence remains for later tasks.
- `FINISH_READY` intentionally needs two independent evidence categories. If existing recognition only supplies one category in a real layout, the final visual audit must identify the missing signal instead of weakening the safety rule prematurely.
- The desktop pet does not consume `RecommendationTarget` coordinates yet; that behavior is deliberately deferred until Task 7 after the Task 6 checkpoint.

Next step:

- Task 6: run the second major checkpoint across canonical data, phase legality, target stability, event consistency, UI integration risk, tests, build, and rollback.

## Task 6: Major Checkpoint 2

Status: completed

Review canonical data compatibility, phase transitions, target stability, tests, event/data consistency, UI integration risk, and rollback before changing desktop-pet movement.

Work performed:

- Audited the 48 canonical affix identities, all 55 historical codes, legacy aliases, current runtime logs, phase legality, target hysteresis/expiry, UI coordinate boundaries, passive-assistance constraints, and rollback isolation before desktop-pet movement changes.
- Fixed a target lifetime bug where repeated UI rendering of the same capture frame could refresh `expires_at` indefinitely; only a distinct source frame can now renew an otherwise unchanged target.
- Fixed `FINISH_READY` stabilization by deriving persistent terminal-map evidence from the current snapshot instead of relying only on edge-triggered events. Terminal evidence now requires zero blanks and at least 36 known cells, and malformed/partial counts are rejected.
- Added target safety validation for board, backpack, brush, and finish proposals: points must be finite, fall inside a valid positive-size capture window, and, for the real Tk app, land on an actual Windows monitor. Windows `MonitorFromPoint` supports negative multi-monitor coordinates; Tk virtual-root bounds remain the non-Windows/API-failure fallback.
- Added regression coverage for same-frame TTL behavior, stable two-frame finish guidance, incomplete terminal evidence, window/screen rejection, negative-coordinate secondary monitors, and malformed/non-finite inputs.
- Kept desktop-pet movement unchanged as required by the checkpoint boundary; structured `RecommendationTarget` consumption remains isolated to Task 7.
- Checkpoint code commit: `f313b7b Harden guidance checkpoint behavior`.

Validation evidence:

- Full suite passed: 41 tests in 0.111 seconds; the 21 focused guidance/pet tests passed independently.
- Canonical compatibility tests cover every historical code and explicit canonical identity. A read-only rebuild of current logs produced 295 real events, 4 trusted trajectories, 14 action/placement samples, 0 brush samples, 0 affix-policy samples, and 0 recommendation decisions; canonical observed IDs remained `transform.randomize_scope.monster` (2), `transform.randomize_scope.any` (1), and `spread.place_on.blocked` (1).
- Strict environment validation passed on Python 3.12.10 with all 11 locked distributions exact; `pip check` reported no broken requirements, and encoding validation checked 58 project text files.
- AST parsing and core imports passed for 26 Python files/modules. The fixed-seed brush/refresh simulation exited 0 and produced install, placement, brush, force, and refresh recommendations.
- Passive safety scan found no network or input-injection API usage in application, script, or test Python files.
- Final temporary PyInstaller build from the committed code succeeded in `C:\Users\m'm'm'm'm'm'm'm'm'm'\AppData\Local\Temp\huahua-goal5-task6-final-fea0fe054cda4adf85c34f7778bf49f3`; the EXE is 75,756,533 bytes with SHA256 `9A1CB95655562B0980F2DC4587E545B576CBCC8E161A6508276E12075B0534B7`. Warning audit found 0 critical missing modules and the Analysis TOC contains `huahua_assistant.guidance`.
- SHA256 values for corrected observations, desktop-pet position, trajectories, observations, and stats were unchanged after the full test/read-only rebuild/build cycle.
- `git diff --cached --check` passed and the checkpoint commit contained only `guidance.py`, `pet_brain.py`, and their two tests; all pre-existing dirty files remained outside the commit.

Remaining risk:

- The packaged GUI was not launched because it would open a window and potentially steal focus. Live game capture, per-monitor DPI changes, game-window movement, Tk click-through, and exact pet/marker placement still require Task 7 runtime behavior plus the final screenshot audit.
- The legacy immediate `finish_hint` message can still react to a single `finish_candidate` event, but no structured finish movement target is released until two independent evidence categories stabilize across two frames.
- If Windows monitor lookup is unavailable, safety falls back to Tk virtual-root bounds; a non-Tk test/integration object without screen metadata is constrained only by its capture-window bounds.
- Recognition accuracy for aliases and phase evidence remains heuristic. Current structural/runtime compatibility is proven, while representative live visual accuracy stays intentionally deferred to Task 13.

Next step:

- Task 7: make the desktop-pet actor consume only stable, unexpired `RecommendationTarget` values, teleport beside Top1 without covering the click point, and preserve lightweight Top2/Top3 markers and click-through behavior.

## Task 7: Implement Top1 Desktop-Pet Teleport Guidance

Status: completed

Make the desktop-pet actor move beside the stable Top1 board/backpack/brush/finish target, point precisely without covering the click area, remain click-through during play, and retain lightweight Top2/Top3 markers.

Work performed:

- Added `pet_layout.py` with a pure, deterministic placement policy: keep a 30-pixel gap from the click point, try above/right/left/below in order, clamp on the perpendicular axis, stay fully inside the target monitor, and refuse movement when no non-covering placement fits.
- Made `DesktopPetActor` consume only live structured `RecommendationTarget` values. A stable target temporarily overrides the saved manual position; expiry, instability, explicit drag mode, or an impossible layout removes the pointer and restores the normal saved/observation position.
- Added a dedicated transparent Top1 pointer window whose arrow tip and target ring are centered on the real click coordinate. Actor and pointer windows use exact Win32 `SetWindowPos` coordinates, including negative-coordinate monitors, with a Tk geometry fallback.
- Added separate render keys for actor and pointer so the same stable target does not repeatedly move or redraw either window. The target's short structured reason is shown in the pet bubble.
- Changed desktop-pet dragging to default off, so game operation is click-through by default. Explicitly enabling drag temporarily suspends target teleport/pointer behavior, makes the pet interactive, and retains the existing saved-position workflow.
- Reworked coordinate markers so Top1 is represented only by the pet/pointer and stable board phases retain lightweight Top2/Top3 markers. Markers are suppressed for expired/unstable targets, non-board phases without structured alternatives, malformed window data, and off-window/off-screen points.
- Kept the existing capture safety path: capture, quick capture, and auto-follow hide the pet, Top1 pointer, and Top2/Top3 markers together before pixels are acquired, then restore from the latest stable state.
- Code commit: `935d5ca Guide desktop pet to stable top target`.

Validation evidence:

- Full suite passed: 53 tests in 0.132 seconds; 33 focused layout/guidance/pet tests passed independently.
- New tests cover placement priority and 30-pixel separation, target non-coverage, negative monitor coordinates, impossible layouts, exact pointer tip geometry, expiry at render time, saved-position override, no repeated movement/redraw, default click-through, explicit drag behavior, capture hiding of both pet and pointer, stable Top2/Top3 ranking, and off-screen marker rejection.
- Real Win32 monitor probing returned `(0, 0, 1536, 864)` for the current display and no monitor for `(-100000, -100000)`, confirming the target-monitor boundary path without opening a GUI.
- Strict environment validation passed on Python 3.12.10 with all 11 locked packages exact; encoding validation checked 60 text files, and AST/core imports passed for 28 Python files.
- Fixed-seed brush/refresh simulation exited 0, and a passive scan found no network or mouse/keyboard injection APIs.
- Isolated PyInstaller build succeeded in `C:\Users\m'm'm'm'm'm'm'm'm'm'\AppData\Local\Temp\huahua-goal5-task7-6a61399e37f24d2c9622d1ca285b56b8`; the EXE is 75,767,597 bytes with SHA256 `93C445593799AFC55B16D69770B4AC76684966CB32896A7450DB7B6C1A96B04E`. Critical missing-module count was 0 and `huahua_assistant.pet_layout` appeared in the Analysis TOC.
- SHA256 values for corrected observations, desktop-pet position, trajectories, observations, and stats stayed unchanged through tests and build.
- `git diff --cached --check` passed, and the code commit contains only `desktop_pet.py`, `pet_layout.py`, and their dedicated test file; pre-existing dirty files remain uncommitted.

Remaining risk:

- The GUI/EXE was not launched because doing so could steal focus. Real per-monitor DPI scaling, Tk color-key transparency, topmost ordering, click-through styles, and visual arrow alignment still need a later live/manual visual gate and the final screenshot audit.
- Only board placement currently has a ranked alternative list, so Top2/Top3 markers are shown only for stable board targets. Backpack, brush, and finish phases correctly show Top1 but need structured alternative targets before meaningful Top2/Top3 markers can exist.
- The Win32 placement path is verified at the API and pure-layout levels. If Win32 positioning fails at runtime, Tk geometry fallback remains, but exact negative-coordinate behavior of that fallback is platform dependent.
- Explicit drag mode intentionally suspends automatic teleport until drag is turned off; this preserves a usable manual-position control while keeping normal gameplay click-through by default.

Next step:

- Task 8: version decision/run records, separate observed outcomes from proxy/policy scores, retain ordinary/failure summaries, and compute the three risk-mode metric sets without enabling premature personalization.

## Task 8: Add Outcome And Risk Learning Schema

Status: completed

Version decision and run records to distinguish observed game outcomes from proxy scores, retain ordinary/failure summaries, log candidate sets and chosen actions, and compute stable/balanced/gamble risk metrics without premature labeling.

Work performed:

- Added `outcome_learning.py` with versioned `huahua-decision-v2`, `huahua-run-outcome-v2`, and `huahua-risk-summary-v1` contracts while preserving legacy payload fields.
- Separated `observed_outcome`, `proxy_score`, and candidate-level `policy_score`; internal treasure-equivalence values never populate an actual observed score.
- Recorded complete bounded placement candidate sets with canonical action/paint IDs, rank, legality, layered policy components, risk decomposition, actual choice, Top1/Top3 hits, transition deltas, resource state, and attribution trust gates.
- Retained trusted high-value trajectories with full decision records and converted ordinary/failure terminal records into lightweight summaries with initial-map buckets, major decisions, resources, terminal distribution, failure reason, and recognition quality.
- Added deduplicated stable/balanced/gamble risk reports with mean, median, population standard deviation, IQR, P10/P20/P80/P90, bottom-20% mean, high/low/failure rates, and force/refresh usage.
- Required at least 20 comparable same-bucket/same-policy runs and explicit observed scores before trend or personalization readiness; proxy-only and legacy unclassified data remain observing with no style label.
- Integrated normalization and risk reports into real-capture observation/trajectory writes and learning stats without rewriting existing runtime JSONL files.
- Code commit: `4a804e9 Add outcome and risk learning schema`.

Validation evidence:

- Full suite passed: 64 tests; 11 focused outcome/risk tests cover score separation, full candidate normalization, attribution gates, trajectory/summary retention, percentiles, resource metrics, sample thresholds, proxy-only behavior, session deduplication, and isolated file writes.
- Strict Python 3.12.10 environment check passed with all 11 locked distributions exact; encoding validation checked 62 project text files.
- Read-only rebuild of current logs produced 7 deduplicated legacy runs. All remain `unclassified`, `proxy_score`, and `observing`; personalization is disabled and no style label is emitted.
- Isolated PyInstaller build succeeded with a 75,788,621-byte EXE and SHA256 `C6EC8F9AC6B9085CE726260FB23AD413CE8715D275E5210923FD084165015D5C`; critical missing-module count was 0.
- Runtime observation, trajectory, correction, position, and stats SHA256 values were unchanged across tests, read-only reports, and packaging.
- `git diff --cached --check` passed and the Task 8 commit excluded pre-existing README, config, game-model, learning-stat extensions, runtime, script, cache, and workspace-helper changes.

Remaining risk:

- Historical runs have no explicit strategy mode or actual game score, so they are intentionally unclassified/proxy-only and cannot establish a personal trend.
- Actual outcome score support is contract-ready but depends on future reliable recognition of a game-provided final score; terminal distributions remain observed facts without being relabeled as an actual score.
- Full candidates are available for new captures; old logs retain only their historical Top3 fields and cannot be reconstructed into unseen alternatives.
- Live GUI capture was not launched; Task 9 must re-audit privacy, compatibility, pet behavior, packaging, and rollback as an integrated checkpoint.

Next step:

- Task 9: run the third major checkpoint across real-data-only writes, privacy, legacy compatibility, outcome integrity, risk metrics, UI guidance, build, and rollback.

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
