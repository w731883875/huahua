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

Status: pending

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 3 - Major Checkpoint 1

Status: pending

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 4 - Add Structured Recommendation Explanation

Status: pending

Work performed:

Validation evidence:

Remaining risk:

Next step:

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
