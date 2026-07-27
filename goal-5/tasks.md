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
