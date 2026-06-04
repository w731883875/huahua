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
- Status: pending
- Goal: Check whether Tasks 1-2 drifted from requirements; verify complete/quick recognition still runs; inspect dependency and fallback risk.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 4 - Frame Pump Prototype
- Status: pending
- Goal: Add realtime observation loop prototype that keeps only the latest frame, avoids frame queue buildup, and records capture timing.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 5 - FastVision Result Model
- Status: pending
- Goal: Build lightweight fast-vision result structure that reuses board/backpack/brush ROI and skips OCR.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 6 - Major Check 2
- Status: pending
- Goal: Check realtime loop, fast-vision data structure, and old-path compatibility; run compile/basic regression.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 7 - Event Detector
- Status: pending
- Goal: Detect events from fast frame changes: board change, backpack appearance, selected state, new current paint, brush stage, finish-drawing candidate.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

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
