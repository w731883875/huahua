# Goal 2 Tasks

## Task 1 - Baseline Current Pipeline
- Status: completed
- Goal: Baseline current capture, quick vision, desktop pet, timing, and learning-record pipeline; identify insertion points.
- Work performed: Inspected the current capture, quick vision, desktop pet, timing, and learning-record paths. Current capture path is `capture_target -> capture_window_content(PrintWindow) -> mss region/screen fallback` in `huahua_assistant/capture.py`. Current quick path is `HuahuaApp._quick_capture_worker/_auto_follow_worker -> VisionTracker.analyze_quick -> analyze_scene_quick`, which skips full OCR except stage-2 refresh. Desktop pet is installed through `run_app.py -> install_desktop_pet_markers(HuahuaApp)` and renders click-through marker windows from current recommendations, but it has no independent pet-brain state machine yet. Timing fields already exist in `_fast_capture_perf` for capture, analysis, debug-write, and full-refresh. Learning records already gate on `source="real_capture"` and track `quick_capture_step` / `quick_capture_ignored`.
- Validation evidence: Located key insertion points at `capture.py:21,52,66,80,253`; `vision.py:33,38,45,62,103,117,442,1175`; `app.py:662,712,733,894,1298,2988,4557`; `desktop_pet.py:48,177,445,807,829,855,1014`; `learning.py:19,49,281,301,450`. Ran `python -B -m py_compile run_app.py huahua_assistant\capture.py huahua_assistant\vision.py huahua_assistant\app.py huahua_assistant\desktop_pet.py huahua_assistant\learning.py` successfully.
- Remaining risk: Existing auto-follow starts a new worker/thread per tick and does not maintain a latest-frame pump, so it can feel bursty instead of continuously intelligent. Current capture abstraction is implicit in `capture_target`, not yet a pluggable backend. Running `py_compile` updated a `__pycache__` file in the working tree.
- Next step: Task 2 should introduce a capture backend abstraction that keeps the current PrintWindow/mss path as default while preparing a safe optional high-speed backend and rollback path.

## Task 2 - Capture Backend Abstraction
- Status: completed
- Goal: Abstract capture backend interface, keep the current backend, and provide optional high-speed backend entry and fallback.
- Work performed: Added a pluggable capture backend layer in `huahua_assistant/capture.py` with `CaptureBackend`, `CaptureBackendPlan`, `LegacyCaptureBackend`, and optional `DxcamRegionCaptureBackend`. Kept the current `PrintWindow -> mss window region -> full screen fallback` behavior as the default legacy/auto path. Added `capture.backend`, `capture.highspeed_enabled`, and `capture.allow_composited_highspeed` config defaults. Added CLI `capture --backend auto|legacy|dxcam|highspeed` for backend debugging without changing app calls. Source strings now include backend/method/attempt details for timing and failure diagnosis.
- Validation evidence: Ran `python -m compileall huahua_assistant run_app.py` successfully. Ran `git diff --check` successfully with only existing CRLF normalization warnings. Reviewed the diff to confirm default config keeps `auto` on legacy-only unless high-speed composited capture is explicitly enabled.
- Remaining risk: `dxcam` is optional and not installed by default; when enabled it captures composed desktop pixels, so it can be polluted by overlays/desktop pet unless the pet is hidden or the capture region is known safe. The default path intentionally avoids this by keeping `PrintWindow` first and not enabling high-speed composited capture automatically.
- Next step: Task 3 should perform the first major check/debug cycle to verify the abstraction did not drift from requirements, the old complete/quick recognition path still compiles, and dependency/fallback risk is documented.

## Task 3 - Major Check 1
- Status: completed
- Goal: Check whether Tasks 1-2 drifted from requirements; verify complete/quick recognition still runs; inspect dependency and fallback risk.
- Work performed: Performed the first major check/debug cycle after the baseline and capture-backend abstraction. Confirmed the app still calls `capture_target(... activate=False)` from full capture, quick capture, and auto-follow without requiring the new backend parameter. Confirmed default backend resolution is `auto` with only `legacy` content/region backends, so optional high-speed composited capture is not silently enabled. Checked desktop-pet marker patch still hides marker surfaces before full/quick/auto-follow captures. Checked real learning writes still pass through `record_observation`, `record_correction`, and `record_high_score_trajectory` gates that ignore non-`real_capture` sources, and quick capture still records low-confidence frames as `quick_capture_ignored` instead of trajectory steps.
- Validation evidence: Ran `python -B -c "from huahua_assistant.capture import resolve_capture_backend; p=resolve_capture_backend(); print(p.name, [b.name for b in p.content_backends], [b.name for b in p.region_backends])"` and got `auto ['legacy'] ['legacy']`. Ran `python -m compileall huahua_assistant run_app.py` successfully. Ran `git diff --check` successfully. `rg "capture_target\(" -n huahua_assistant run_app.py` shows only the expected app/CLI/capture definition call sites. Inspected `desktop_pet.py` patched capture hooks and `learning.py` real-capture guards.
- Remaining risk: The objective is not complete yet. Auto-follow still starts a worker per tick and can feel bursty; there is no latest-frame pump yet. `dxcam` remains optional and, if explicitly enabled later, can capture desktop composition including overlays unless coordinated with hiding/region safety. No live game-window runtime test was performed in this check because the task was code-path validation.
- Next step: Task 4 should add a realtime frame-pump prototype that keeps only the newest frame, records capture timing, and gives the later fast-vision/event/state-machine layers a stable 0.5-second observation loop.

## Task 4 - Frame Pump Prototype
- Status: completed
- Goal: Add realtime observation loop prototype that keeps only the latest frame, avoids frame queue buildup, and records capture timing.
- Work performed: Added `huahua_assistant/frame_pump.py` with a background `FramePump` loop that captures, runs quick analysis, optionally performs the existing stage-2 full refresh, records capture/analysis/write/full-refresh/pump interval timing, and stores only the latest `FramePumpResult`. Integrated auto-follow in `HuahuaApp` so UI ticks now consume the newest pump result instead of launching a new worker thread per tick. Auto-follow starts the pump only after the tool window overlap check passes, stops it when overlap pause occurs, stops it on manual full/quick capture, and stops it on app close. Removed the old auto-follow worker path to avoid two competing realtime routes.
- Validation evidence: Ran `python -m compileall huahua_assistant run_app.py` successfully. Ran `git diff --check` successfully with only CRLF normalization warnings for `app.py`. Confirmed `rg "_auto_follow_worker|threading.Thread\(target=lambda: self._auto_follow_worker" -n huahua_assistant\app.py huahua_assistant\frame_pump.py` no longer finds old auto-follow worker usage, while `FramePumpResult` remains in the pump module. Reviewed the diff to confirm runtime/debug/pycache files are not part of the intended source changes.
- Remaining risk: This is still a prototype loop. It reuses current `VisionTracker.analyze_quick`, so real 0.5-second feedback depends on capture and quick-analysis cost on the user's machine. Desktop-pet marker hiding before every individual pump frame is not yet coordinated; the pump starts only after overlap checks, but future pet-state work must keep overlays/pets from polluting composed capture modes. Live in-game timing was not measured in this task.
- Next step: Task 5 should define a lightweight fast-vision result model so later event detection and pet state can consume stable, OCR-light board/backpack/brush summaries instead of the full scene object.

## Task 5 - FastVision Result Model
- Status: completed
- Goal: Build lightweight fast-vision result structure that reuses board/backpack/brush ROI and skips OCR.
- Work performed: Added `huahua_assistant/fast_vision.py` with lightweight immutable models: `FastVisionSnapshot`, `FastTile`, `FastBackpackPaint`, `FastCurrentPaint`, and `FastBrush`. Added `snapshot_from_scene()` to compress a `SceneObservation` into board/paint/backpack signatures, terrain counts, compact ROI rectangles, current-paint summary, brush-stage summary, and reliability reasons. Integrated snapshots into `FramePumpResult` so the realtime pump emits both the existing full scene and a compact OCR-light snapshot. Added `HuahuaApp.last_fast_snapshot` and update points for quick capture and auto-follow.
- Validation evidence: Ran `python -m compileall huahua_assistant run_app.py` successfully. Ran `git diff --check` successfully with only CRLF normalization warnings. Ran a minimal in-memory snapshot construction command and got a stable signature with `blank=36`, `reliable=True`, and no reasons. `rg "FastVisionSnapshot|snapshot_from_scene|last_fast_snapshot|snapshot" -n huahua_assistant` shows the expected model, pump, and app integration points.
- Remaining risk: The snapshot currently wraps existing quick-scene recognition; it does not yet perform independent ROI-only CV extraction, so recognition speed is still governed by `VisionTracker.analyze_quick`. Reliability thresholds are conservative starting points and may need tuning against real captures. Event detection is not implemented yet; this task only provides the compact data object for it.
- Next step: Task 6 should run the second major check/debug cycle to verify the realtime pump plus FastVision snapshot do not break old complete/quick paths, then Task 7 can build event detection on top of snapshot signature changes.

## Task 6 - Major Check 2
- Status: completed
- Goal: Check realtime loop, fast-vision data structure, and old-path compatibility; run compile/basic regression.
- Work performed: Performed the second major check/debug cycle over the realtime frame pump and FastVision snapshot work. Verified full capture and manual quick capture still use their existing worker paths, while auto-follow uses `_start_auto_follow_pump` and `_poll_auto_follow_pump`. Verified `last_fast_snapshot` is only held in app memory and not written into learning records or trajectories. Reviewed learning gates to confirm `record_observation`, `record_correction`, and `record_high_score_trajectory` still reject non-`real_capture` sources, and quick low-confidence frames still go to `quick_capture_ignored`.
- Validation evidence: Ran `python -m compileall huahua_assistant run_app.py` successfully. Ran `git diff --check` successfully. Ran a pure in-memory `FramePump` start/stop test that produced `True 36 0.1`, proving the pump can emit a latest result with a `FastVisionSnapshot` and stop cleanly. `rg` checks confirmed expected auto-follow/quick/full capture call sites, `last_fast_snapshot` locations, and real-capture learning gates. `git diff --stat` showed only runtime/pycache changes before this task-record update.
- Remaining risk: No live game-window timing test was performed, so the actual 0.5-second target remains unproven on the user's machine. The fast snapshot still wraps existing quick recognition, not a fully independent ROI-only CV path. Desktop-pet state and event detection are still missing.
- Next step: Task 7 should implement an event detector over consecutive `FastVisionSnapshot` objects for board changes, backpack appearance/disappearance, current paint changes, brush-stage changes, and finish-drawing candidates.

## Task 7 - Event Detector
- Status: completed
- Goal: Detect events from fast frame changes: board change, backpack appearance, selected state, new current paint, brush stage, finish-drawing candidate.
- Work performed: Added `huahua_assistant/fast_events.py` with `FastVisionEvent` and `detect_fast_events(previous, current)`. The detector emits snapshot-ready, board-changed, paint-changed, backpack-appeared/disappeared/changed, conservative selection-state-changed, current-paint appeared/disappeared/changed, brush-stage appeared/disappeared/changed, and finish-candidate events. Integrated event state into `HuahuaApp` through `previous_fast_snapshot`, `last_fast_snapshot`, `last_fast_events`, and `_update_fast_snapshot()`. Reset fast-event state when starting a new real session. Kept events in memory only; they are not written into learning trajectories.
- Validation evidence: Ran `python -m compileall huahua_assistant run_app.py` successfully. Ran `git diff --check` successfully with only CRLF normalization warnings. Ran an in-memory board-change test and got `['board_changed', 'paint_changed', 'finish_candidate']`. Ran an in-memory backpack fingerprint-only change test and got `['selection_state_changed']`. `rg "FastVisionEvent|detect_fast_events|last_fast_events|previous_fast_snapshot|_update_fast_snapshot" -n huahua_assistant` shows the expected model and app integration points.
- Remaining risk: Selection detection is deliberately conservative because the current backpack model has no explicit selected-slot field; it infers a possible selection-state change from backpack fingerprint changes when logical backpack contents and board state remain stable. Finish candidates are hints, not authoritative settlement proof. Events are not yet consumed by a desktop-pet state machine.
- Next step: Task 8 should add the desktop-pet state machine that consumes current recommendations plus `last_fast_events` to output observing, recommend-backpack, recommend-board-cell, recommend-brush, wait-for-drop, and finish-hint states.

## Task 8 - Pet Brain State Machine
- Status: pending
- Goal: Add desktop-pet state machine that outputs observing, recommend backpack, recommend board cell, recommend brush, wait for drop, and finish hint states.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 9 - Major Check 3
- Status: pending
- Goal: Check event/state machine consistency with real recording, safety boundary, mouse passthrough, and desktop pet flicker risks.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 10 - Performance Panel
- Status: pending
- Goal: Show capture, fast CV, recommendation, desktop-pet update timing, and latest state in the UI to support the 0.5s target.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 11 - Trust Gate And Learning Safety
- Status: pending
- Goal: Ensure untrusted fast-loop results do not pollute real high-score trajectories, while recording explainable observation events.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 12 - Final Review
- Status: pending
- Goal: Review UX, performance, data safety, rollback, packaging impact, and next-stage OCR throttling strategy.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:
