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

Status: pending

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 6 - Major Checkpoint 2

Status: pending

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 7 - Strengthen Real-Only Trajectory And High-Value Logging

Status: pending

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 8 - Add Recognition/Capture Safety For Marker Layer

Status: pending

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 9 - Major Checkpoint 3

Status: pending

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 10 - Final Integration Review And Packaging Readiness Notes

Status: pending

Work performed:

Validation evidence:

Remaining risk:

Next step:
