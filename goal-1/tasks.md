# Goal Tasks

## Task 1 - Map Existing UI/Overlay/Capture Flow

Status: completed

Work performed:
- Read the goal input, plan, and task list.
- Inspected the current capture flow in `huahua_assistant/capture.py`.
- Inspected UI state, overlay variables, toggle logic, disabled overlay refresh path, side-panel coordinate output, and recommendation rendering in `huahua_assistant/app.py`.
- Inspected where real capture and quick capture update `last_capture_window`, `current_scene`, `current_state`, `recommendations`, and `install_plan`.
- Confirmed recommendation computation already flows through `state_from_observation()`, `recommend_placements()`, and `recommend_install_plan()` after full/quick captures.
- Confirmed side-panel coordinate helpers already exist for board cells and visible backpack slots:
  - `_screen_point_for_board_cell()`
  - `_screen_point_for_backpack_slot()`
  - `_visible_backpack_index_for_paint()`
- Confirmed old broad overlay drawing methods still exist but `_refresh_overlay()` intentionally returns immediately to avoid covering the game and blocking clicks.
- Confirmed auto-follow currently pauses when the main tool window overlaps the game, which protects recognition but also prevents using the main UI as the in-game presentation layer.

Validation evidence:
- `capture_target()` finds `torchlight_infinite`/Torchlight windows and captures a rectangle based on `WindowInfo.left/top/width/height`.
- Full capture calls `analyze_and_save_debug()`, quick/auto-follow call `VisionTracker.analyze_quick()` and optionally full analysis for stage-2 brush refresh.
- `_render_result()` and `_render_fast_scene_result()` both compute recommendations and call `_refresh_overlay()`.
- `_refresh_overlay()` currently returns before creating/drawing a game-window overlay, so no visible in-game marker is expected.
- Existing coordinate helpers use detected scene rects plus `last_capture_window` offsets, not hard-coded absolute screen positions.
- No business code was changed in this task.

Remaining risk:
- `mss` captures the desktop rectangle corresponding to the game window; if a future marker is drawn over that rectangle, it can still contaminate recognition unless hidden during capture or moved to a true window-capture method.
- Tk click-through support is already attempted in `_make_overlay_clickthrough()`, but previous user testing showed broad overlays can still block clicks. The next implementation should use small marker windows plus a fail-safe destroy/toggle path.
- Current UI strings contain mojibake in several places; this task did not address text cleanup.

Next step:
- Implement a safe coordinate marker prototype that reuses the existing screen-point helpers and recommendation data, while avoiding a broad full-window overlay.

## Task 2 - Implement Safe Coordinate Marker Prototype

Status: completed

Work performed:
- Added `huahua_assistant/desktop_pet.py` with a small coordinate-marker presentation layer.
- Implemented `DesktopPetMarkerLayer`, which manages multiple tiny `tk.Toplevel` marker windows instead of one broad full-window overlay.
- Markers are topmost, transparent-background windows and attempt Windows click-through with `WS_EX_LAYERED`, `WS_EX_TRANSPARENT`, `WS_EX_NOACTIVATE`, `WS_EX_TOOLWINDOW`, and `WM_NCHITTEST -> HTTRANSPARENT`.
- Added marker types for:
  - recommended board cells, using detected tile rects plus `last_capture_window` offsets
  - visible backpack slot for the first desired paint
  - stage-1/stage-2 brush choice recommendations, using detected brush panel rects
- Added capture safety for this prototype by hiding marker windows before full capture, quick capture, and auto-follow capture ticks.
- Patched `HuahuaApp` at startup from `run_app.py` via `install_desktop_pet_markers(HuahuaApp)`, avoiding direct edits to the already large `app.py`.
- Reused the existing “坐标提示” checkbox as the feature toggle.

Validation evidence:
- `python -B` AST parse succeeded for `run_app.py` and `huahua_assistant/desktop_pet.py`.
- Import check succeeded:
  - `patch_installed= True`
  - `overlay_callable= True`
- `python -m compileall huahua_assistant run_app.py` completed successfully.
- `git diff` shows startup wiring in `run_app.py`; `desktop_pet.py` is isolated as a new module.
- Marker coordinate construction uses `last_capture_window`, scene tile rects, backpack slot rects, and brush panel rects rather than hard-coded absolute screen pixels.

Remaining risk:
- Runtime click-through behavior still needs real Windows/game testing, because some games/window modes can ignore or reorder ordinary topmost windows.
- `mss` window-rectangle capture can still include markers if capture happens before the hidden windows visually disappear; Task 8 will strengthen capture safety.
- The prototype uses simple drawn circular markers rather than final pet art assets.
- Brush option coordinates are approximate within the detected brush panel because `BrushOptionsObservation` currently does not expose per-card rects.

Next step:
- Run Major Checkpoint 1 to review requirements, click safety, capture contamination risk, build health, and whether the marker prototype should be refined before adding structured recommendation explanations.

## Task 3 - Major Checkpoint 1

Status: completed

Work performed:
- Reviewed requirement drift after Tasks 1-2. The marker prototype still aligns with the confirmed plan: small coordinate-anchored markers, no broad overlay, no input control, and reuse of game-window-derived coordinates.
- Reviewed UI/UX behavior. The existing “坐标提示” checkbox is reused as the safe toggle; markers are hidden when no valid scene/window/recommendation exists.
- Reviewed click safety. Markers are small `Toplevel` windows and attempt click-through with Windows extended styles and `WM_NCHITTEST -> HTTRANSPARENT`; broad full-game overlay is not used.
- Reviewed capture contamination risk. Full capture, quick capture, and auto-follow capture ticks hide marker windows before screenshot; remaining race risk is deferred to Task 8 for stronger capture safety.
- Reviewed data consistency. Task 2 did not alter learning writes or simulation behavior.
- Reviewed rollback needs. The feature is isolated in `desktop_pet.py` and startup wiring in `run_app.py`; disabling the checkbox or removing the install call rolls it back.
- Found and fixed one implementation risk: `install_desktop_pet_markers()` now returns early if the class is already patched, preventing repeated imports/hot reload from wrapping methods multiple times.

Validation evidence:
- `python -m compileall huahua_assistant run_app.py` completed successfully.
- Import/patch check confirmed:
  - `installed= True`
  - `same_after_repeat= True`
  - `refresh_name= patched_refresh_overlay`
- `git status --short` after cache cleanup showed only `huahua_assistant/desktop_pet.py` modified before task-record update.
- No learning-log behavior was changed in this checkpoint.

Remaining risk:
- Real game runtime testing is still needed to prove click-through behavior under the user's exact Windows/game display mode.
- `mss` can still capture topmost marker windows if Windows has not visually hidden them before the grab; Task 8 should add stronger marker-hide/settle handling or true window capture.
- Brush marker placement remains approximate because per-card/per-affix rectangles are not yet exposed by `BrushOptionsObservation`.

Next step:
- Add structured recommendation explanations so the side panel and desktop-pet markers can show why a position/paint/affix is recommended, not just where to point.

## Task 4 - Add Structured Recommendation Explanation

Status: completed

Work performed:
- Added `RecommendationExplanation` to `huahua_assistant/game_model.py`.
- Added optional `explanation` data to `PlacementRecommendation` while keeping the existing `reason` string intact.
- Populated structured explanations for normal backpack placement recommendations.
- Populated structured explanations for pending current-paint anchor placement recommendations.
- Split recommendation explanation into:
  - `immediate`
  - `terrain`
  - `risk`
  - `resource`
  - `expected`
  - `value`
  - `tags`
  - `short`
- Added compact Chinese `short` labels for key cases such as treasure-to-7, match completion, level-7 push, overwrite risk, terrain gain, and anchor placement.

Validation evidence:
- `python -m compileall huahua_assistant run_app.py` completed successfully.
- `python -B` smoke test constructed a normal placement state and confirmed:
  - `normal_count= 3`
  - `normal_explanation= True`
  - `normal_short= 推进1格`
  - `normal_tags= ('treasure', 'high_value_gain')`
- `python -B` smoke test constructed a pending initial-paint anchor state and confirmed:
  - `anchor_count= 3`
  - `anchor_explanation= True`
  - `anchor_short= 可完成地形`
  - `anchor_tags= ('anchor', 'can_complete')`
- `git diff --check -- huahua_assistant/game_model.py` found no whitespace errors.
- Code change committed as `488d1be Add structured recommendation explanations`.

Remaining risk:
- The structured explanation is available in the recommendation objects but is not yet shown in the side-panel UI or desktop-pet marker labels.
- The numeric breakdown mirrors the current scoring model; future tuning should still use real high-value trajectory data once Task 7 strengthens logging.
- Current UI mojibake remains outside this task's scope.

Next step:
- Connect the explanation output to the existing recommendation panel and desktop-pet marker labels so the user can see why a cell/backpack/affix is recommended.

## Task 5 - Connect Explanation Output To UI And Marker Labels

Status: completed

Work performed:
- Connected `PlacementRecommendation.explanation` to the side-panel recommendation output in `huahua_assistant/app.py`.
- Added readable explanation lines under each placement recommendation:
  - compact conclusion from `explanation.short`
  - layered score breakdown for immediate/value/terrain/risk/resource/expected
  - translated tags for key recommendation reasons
- Added marker captions to `huahua_assistant/desktop_pet.py` so desktop-pet markers can show short reasons instead of generic `cell`/`bag`/`affix` labels.
- Board and backpack markers now use the best recommendation's `explanation.short` when available.
- Brush markers now use a short caption derived from the best brush/affix recommendation reason.

Validation evidence:
- `python -m compileall huahua_assistant run_app.py` completed successfully.
- `python -B` smoke test verified `_recommendation_explanation_lines()` emits:
  - a short conclusion line
  - a layered score line
  - translated tag text
- `python -B` smoke test verified desktop-pet helper functions preserve and fit the short marker caption.
- `git diff --check -- huahua_assistant/app.py huahua_assistant/desktop_pet.py` found no whitespace errors.
- Code change committed as `d53d675 Show recommendation explanations in UI markers`.

Remaining risk:
- Real UI/runtime testing is still needed to confirm the small marker captions remain legible at the user's exact game resolution and scaling.
- Brush marker coordinates remain approximate because per-option rectangles are not yet exposed by the recognition model.
- Some existing UI text still contains mojibake from older code; this task avoided broad text cleanup to reduce regression risk.

Next step:
- Run Major Checkpoint 2 to review requirement drift, build health, UI safety, data consistency, and whether the explanation/marker integration introduces any regression before strengthening real-only trajectory logging.

## Task 6 - Major Checkpoint 2

Status: completed

Work performed:
- Reviewed requirement drift after Tasks 4-5. The implementation still aligns with the plan: structured recommendation explanations exist in the decision model and are now surfaced in both the side panel and small desktop-pet markers.
- Reviewed UI behavior. The old broad overlay remains effectively disabled in `HuahuaApp._refresh_overlay()`, while `desktop_pet.py` patches the app to use small marker windows only when the coordinate hint toggle is enabled.
- Reviewed marker safety. The desktop-pet patch remains idempotent, uses click-through Windows styles, and hides marker windows before full capture, quick capture, and auto-follow capture ticks.
- Reviewed data consistency. Learning entry points still reject non-`real_capture` sources; simulation mode remains separated and does not write learning records.
- Reviewed rollback needs. Presentation changes are isolated in `huahua_assistant/desktop_pet.py` plus startup patch wiring in `run_app.py`; disabling/removing the patch can roll back marker presentation without removing recommendation logic.
- Cleaned generated Python cache changes after validation so no `pyc` files are left dirty.

Validation evidence:
- `git status --short` was clean before the checkpoint checks.
- `python -m compileall huahua_assistant run_app.py` completed successfully.
- `python -B` smoke test confirmed `install_desktop_pet_markers(HuahuaApp)` is idempotent:
  - `patch_idempotent= True`
- `python -B` smoke test confirmed recommendation explanation UI helpers produce three explanation lines and preserve the short marker caption:
  - `line_count= 3`
  - `caption= \u5b9d\u7bb1\u52307`
  - `fit= \u5b9d\u7bb1\u52307`
- `python -B` smoke test confirmed learning filters reject simulation data and accept real-capture data:
  - `sim_observation_exists= False`
  - `sim_trajectory_saved= False`
  - `real_observation_exists= True`
  - `real_trajectory_saved= True`
- `git diff --check` found no whitespace errors.

Remaining risk:
- Real game runtime testing is still required to prove the small topmost marker windows are visible and click-through under the user's exact display mode.
- `mss` can still theoretically capture marker windows if Windows has not hidden them visually before screenshot; Task 8 should strengthen hide/settle capture safety.
- Brush option marker coordinates are still approximate because the recognition observation does not expose per-card rectangles.
- Existing mojibake strings remain a readability issue but were not broadened in this checkpoint.

Next step:
- Strengthen real-only trajectory and high-value logging so the tool records useful real-operation data for later probability-based recommendation tuning without accepting simulation or manually polluted scores.

## Task 7 - Strengthen Real-Only Trajectory And High-Value Logging

Status: completed

Work performed:
- Enhanced real trajectory capture payloads so compact placement recommendations now include structured explanation data:
  - `short`
  - `tags`
  - numeric layer breakdown for immediate/value/terrain/risk/resource/expected
- Moved full-capture high-value logging to run after recommendation calculation, so high-value observations can include the active top recommendation context.
- Added top recommendation snapshots and internal score breakdown to `high_value_seen` observations.
- Added recommendation semantic aggregation in `huahua_assistant/learning.py`:
  - `recommendation_cells`
  - `recommendation_shorts`
  - `recommendation_tags`
  - `high_value_recommendation_shorts`
  - `high_value_recommendation_tags`
- Kept recommendation semantic stats separate from `placement_cells` and `policy_prior_samples`, so recommended positions do not contaminate actual user-placement priors.
- Added a visible learning-stats summary line for recommendation semantics in the side panel.

Validation evidence:
- `python -m compileall huahua_assistant run_app.py` completed successfully.
- `python -B` smoke test confirmed `_compact_recommendation()` writes explanation data:
  - `has_explanation= True`
  - `short_ok= True`
  - `tags_ok= True`
  - `layers_value= 5000.0`
- `python -B` smoke test confirmed learning stats aggregate only real-capture recommendation semantics and keep placement priors clean:
  - `high_value_seen= 1`
  - `treasure7= 1`
  - `recommendation_cells= {'2,3': 2}`
  - `recommendation_shorts= {'treasure7': 2}`
  - `recommendation_tags= {'complete_match': 2, 'treasure_to_7': 2}`
  - `high_value_tags= {'complete_match': 1, 'treasure_to_7': 1}`
  - `policy_placements= 0`
- A source-filter smoke test confirmed simulation high-value observations are still ignored and only `real_capture` contributes to stats.
- `git diff --check -- huahua_assistant/app.py huahua_assistant/learning.py` found no whitespace errors.
- Code change committed as `af63f06 Strengthen real trajectory recommendation logging`.

Remaining risk:
- The enriched recommendation context records what the tool recommended at the time, not necessarily what the user actually clicked. It is intentionally kept separate from user-placement priors until stronger click/action inference is available.
- Real runtime data is still needed to verify whether recommendation semantic tags correlate with later high-value outcomes.
- Existing mojibake strings remain in some UI text; this task only added the data fields and did not perform a broad text cleanup.

Next step:
- Add stronger recognition/capture safety for the marker layer, especially hiding and settling desktop-pet markers before screenshots so marker windows cannot contaminate game-window captures.

## Task 8 - Add Recognition/Capture Safety For Marker Layer

Status: completed

Work performed:
- Added a capture-specific marker hiding path in `huahua_assistant/desktop_pet.py`.
- Changed `DesktopPetMarkerLayer.hide()` to return the number of withdrawn marker windows.
- Added `hide_for_capture()`, which withdraws active marker windows and waits for Tk/DWM compositor settling before a screenshot is started.
- Added Tk idle flushing, `DwmFlush()`, and a short 80 ms settle window to reduce the chance that topmost desktop-pet marker windows remain visible in `mss` game-window rectangle captures.
- Updated the patched full capture, quick capture, and auto-follow capture paths to call `hide_for_capture()` before delegating to the original app methods.
- Cleaned generated Python cache files before committing so no `pyc` files were included.

Validation evidence:
- `python -m compileall huahua_assistant run_app.py` completed successfully.
- `git diff --check -- huahua_assistant/desktop_pet.py` found no whitespace errors.
- Smoke test confirmed `hide()` withdraws marker windows and returns the hidden count.
- Smoke test confirmed `hide_for_capture(settle_ms=0)` withdraws marker windows and flushes Tk idle work.
- Smoke test confirmed patched capture methods call marker hiding before the original methods:
  - `hide,capture,hide,quick,hide,auto`
- Smoke test confirmed `install_desktop_pet_markers()` remains idempotent.
- `git status --short` showed only `huahua_assistant/desktop_pet.py` modified before the code commit.
- Code change committed as `a72b42f Strengthen marker capture hiding`.

Remaining risk:
- Real game runtime testing is still needed because Windows compositor timing and game display mode can vary.
- `mss` still captures a desktop rectangle; hiding and settling greatly reduces marker contamination risk, but a true window-capture path would be the stronger future solution if contamination still appears.
- The 80 ms settle delay is intentionally small for responsiveness; it may need tuning if the user's machine or display mode still captures stale marker frames.

Next step:
- Run Major Checkpoint 3 to review requirement drift, build health, UI safety, data consistency, capture safety, and remaining packaging-readiness risks.

## Task 9 - Major Checkpoint 3

Status: completed

Work performed:
- Re-read the goal input, plan, and task list before starting the checkpoint.
- Reviewed requirement drift after Tasks 7-8:
  - The implementation still uses small coordinate-anchored desktop-pet marker windows instead of a broad full-screen overlay.
  - The marker layer still does not inject clicks, control the game, read memory, or take over user input.
  - Recommendation explanations are still present in the model, side-panel compact output, marker captions, and learning snapshots.
  - Learning writes still reject non-`real_capture` sources and keep recommendation semantics separate from actual placement priors.
- Reviewed capture safety:
  - `run_app.py` installs `install_desktop_pet_markers(HuahuaApp)` at startup.
  - Full capture, quick capture, and auto-follow capture are patched to hide desktop-pet markers before entering the original capture paths.
  - The broad legacy overlay in `HuahuaApp._refresh_overlay()` remains effectively disabled unless patched by the desktop-pet marker module.
- Reviewed UI/UX safety:
  - The coordinate hint checkbox remains the feature toggle.
  - Marker windows are small, topmost, transparent-background, and attempt Windows click-through/no-activate styles.
  - The marker layer can be disabled by toggling the coordinate hint off.
- Reviewed data consistency and rollback:
  - Presentation changes remain isolated in `huahua_assistant/desktop_pet.py` plus the startup patch in `run_app.py`.
  - Recommendation data added to learning stats is semantic recommendation context, not assumed user-click data.
  - Simulation mode remains separate from real learning logs.
- During validation, one first-pass test harness accidentally targeted real `runtime` learning paths. Restored the affected runtime files and re-ran the learning test with all learning paths redirected to an isolated temporary directory.

Validation evidence:
- `git status --short` was clean at the start of the checkpoint.
- `python -m compileall huahua_assistant run_app.py` completed successfully.
- `git diff --check` completed successfully.
- `rg` checks confirmed:
  - `run_app.py` imports and installs `install_desktop_pet_markers`.
  - desktop-pet patch wraps `capture_async`, `quick_capture_async`, and `_auto_follow_tick`.
  - learning entry points and stats filters require `source == "real_capture"`.
  - recommendation explanation fields are compacted into real-capture steps and high-value observations.
- `python -B` smoke test confirmed marker patch idempotency and capture-hide behavior:
  - `marker_patch_idempotent=True`
  - `hide_for_capture_order=withdraw,flush,flush`
- `python -B` smoke test confirmed compact recommendation explanations preserve short text, tags, and numeric layers:
  - `explanation_compact=True`
  - `explanation_lines=3`
- Isolated learning smoke test confirmed:
  - simulation observations and trajectories are ignored
  - real observations and trajectories are accepted
  - high-value recommendation tags aggregate from real `recommendations`
  - temporary test data was deleted afterward
- `python -B` startup smoke test confirmed importing `run_app.py` installs the desktop-pet patch:
  - `run_app_patch_installed=True`
  - `refresh_name=patched_refresh_overlay`
- After restoring validation artifacts and deleting generated cache files, `git status --short` was clean before this task-record update.

Remaining risk:
- Real game runtime testing is still needed to prove marker visibility and click-through behavior under the user's exact display mode.
- Ordinary topmost Tk marker windows may not appear above true exclusive fullscreen; borderless/windowed fullscreen remains the supported first target.
- `mss` still captures a desktop rectangle; marker hiding and DWM/Tk flushing reduce contamination risk, but true window capture remains the stronger future option if stale marker frames still appear.
- Brush-stage marker coordinates are still approximate because recognition currently exposes the brush panel rect rather than exact per-card option rects.
- Existing mojibake UI strings remain a readability issue and should be cleaned only in a focused future task to avoid broad regressions.

Next step:
- Run the final integration review and packaging-readiness notes, including user-facing behavior, remaining runtime risks, rollback path, and whether any final small fixes are required before marking the goal complete.

## Task 10 - Final Integration Review And Packaging Readiness Notes

Status: completed

Work performed:
- Re-read the goal input, plan, task list, and current git state before starting the final review.
- Performed a requirement-by-requirement completion audit against the active objective:
  - Independent desktop-pet/hint layer is implemented in `huahua_assistant/desktop_pet.py` and installed from `run_app.py`.
  - Markers are small coordinate-anchored windows, not a broad full-screen overlay.
  - Marker coordinates come from detected board/backpack/brush panel geometry plus `last_capture_window` offsets.
  - Marker windows attempt topmost display, transparent background, no-activate behavior, and Windows click-through styles.
  - Full capture, quick capture, and auto-follow capture hide markers before screenshots.
  - Structured recommendation explanations exist in `RecommendationExplanation`, placement recommendations, side-panel lines, compact learning payloads, and desktop-pet captions.
  - Real learning records reject non-`real_capture` sources and keep recommendation semantics separate from actual placement priors.
  - Export/merge support remains present for combining real learning bundles.
- Reviewed compliance and interaction boundaries:
  - No input injection, auto-clicking, memory reading, or device-control path was added.
  - GUI capture paths use `activate=False`, so normal desktop-app capture does not intentionally steal focus.
  - The coordinate hint checkbox remains the user-facing enable/disable switch for the marker layer.
- Reviewed packaging readiness:
  - `HuahuaAssistant.spec` points at `run_app.py`, which installs the desktop-pet marker patch before `main()`.
  - `scripts/build_exe.ps1` still builds `HuahuaAssistant.exe` from `run_app.py`.
  - No new third-party dependency was introduced by the desktop-pet marker layer.
- Cleaned generated Python cache files after final validation so no `pyc` files were left dirty.

Validation evidence:
- `git status --short` was clean at the start of the final review.
- Code scans confirmed:
  - `run_app.py` imports and installs `install_desktop_pet_markers(HuahuaApp)`.
  - `desktop_pet.py` defines `DesktopPetMarkerLayer`, `PetMarker`, click-through styles, marker building, and `hide_for_capture()`.
  - `desktop_pet.py` wraps `capture_async`, `quick_capture_async`, and `_auto_follow_tick`.
  - `game_model.py` defines `RecommendationExplanation` and attaches it to placement recommendations.
  - `app.py` compacts explanation data, renders explanation lines, and records high-value recommendations.
  - `learning.py` filters by `source == "real_capture"` and aggregates recommendation semantics separately.
  - `HuahuaAssistant.spec` and `scripts/build_exe.ps1` build from `run_app.py`.
- `python -B` final smoke test confirmed:
  - startup patch is installed
  - recommendation explanation compacts into short/tags/layers
  - side-panel explanation lines are generated
  - board and backpack desktop-pet markers are generated from scene/window-derived coordinates
- `python -m compileall huahua_assistant run_app.py` completed successfully.
- Isolated learning/export smoke test confirmed:
  - simulation data is ignored
  - real capture data is accepted
  - recommendation semantic tags aggregate
  - high-value recommendation semantic tags aggregate
  - export bundle contains only the isolated real test records
  - temporary test directory was deleted afterward
- `git diff --check` completed successfully.
- Final `git status --short` was clean before this task-record update.

Remaining risk:
- Real game runtime testing is still required to prove the marker windows are visible and click-through in the user's exact Windows display mode.
- True exclusive fullscreen may still hide normal topmost desktop windows; borderless/windowed fullscreen is the supported first runtime target.
- `mss` still captures desktop rectangles, so stale compositor frames remain theoretically possible even after marker hiding and DWM/Tk flushing.
- Brush recommendation marker placement remains approximate until recognition exposes exact per-option/card rectangles.
- Existing mojibake UI/documentation text remains a readability issue, but it is not a blocker for this goal's functional integration and should be handled in a focused cleanup task later.
- Final exe packaging was reviewed for readiness but not executed in this goal, because the user's current workflow prefers script iteration until behavior is stable.

Next step:
- Goal complete. Next practical work outside this goal should be real game runtime testing of marker visibility/click-through and, if needed, a focused follow-up task for exact brush-option rectangles or true window capture.
