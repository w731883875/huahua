# Goal Plan

## Objective

完成用户确认的“桌宠/提示层 + 可解释推荐 + 真实学习记录 + 识别加速 + 稳定后打包准备”计划。优先从 M1 独立提示层原型开始，逐步把推荐、记录、识别和 UI 状态串成可用流程。

## Confirmed User Direction

- 第一版优先支持无边框全屏/窗口化全屏；独占全屏不作为主要验收标准。
- 提示层采用“坐标锚定型桌宠”：小宠物/小占位标记移动到推荐地块、背包或词缀位置，而不是大面积蒙版。
- 识别必须基于游戏窗口内部截图，不能因为桌宠覆盖桌面而污染识别。
- 桌宠/提示层必须置顶、鼠标穿透、只显示推荐，不接管输入。
- 推荐体系要输出可解释的分层评分，而不是只有总分。
- 学习记录只保存真实识别和真实操作轨迹，不保存模拟数据，不允许手动填分污染数据。

## Existing Project Context

- Main entry: `run_app.py`
- UI and workflow: `huahua_assistant/app.py`
- Recognition: `huahua_assistant/vision.py`
- Capture: `huahua_assistant/capture.py`
- Scoring/recommendation: `huahua_assistant/game_model.py`
- Learning logs/export/merge: `huahua_assistant/learning.py`
- Models/enums: `huahua_assistant/models.py`

The current Tk overlay code exists but is effectively disabled for safety because previous transparent overlays blocked game clicks. The next implementation should avoid broad overlays and favor small click-through marker windows anchored to game-window coordinates.

## Architecture Approach

1. Keep the current Python/OpenCV/Tk code as the control and decision core.
2. Add a separate lightweight presentation layer for coordinate markers. It should be small, click-through, and derived from existing game-window and detected-region coordinates.
3. Convert recommendation outputs into a structured explanation model that can be displayed both in the existing side panel and by the desktop-pet markers.
4. Continue using real-capture-only learning logs. Add/adjust fields only when they are supported by recognized real data.
5. Improve speed through caching and region-based updates after presentation and explanation structure are stable.

## Technical Risks

- True exclusive fullscreen may hide normal desktop windows; support is best-effort only. Windowed fullscreen/borderless fullscreen remains the intended mode.
- Tk click-through behavior on Windows can be inconsistent. If a small marker window still blocks clicks, add a fail-safe toggle and destroy marker windows when unsafe.
- Current source files show some mojibake text in UI strings. Avoid broad string rewrites unless task-specific and tested.
- Window-region screenshot via `mss` may still include topmost overlay if the overlay lies inside the captured desktop region. Mitigation: use small marker windows, hide markers during capture when necessary, or later move to a window-specific capture path.
- Recommendation explanations must not overfit to incomplete recognition data; include confidence/risk fields.

## Validation Strategy

- Run `python -m compileall huahua_assistant run_app.py` after code edits.
- Validate that overlay/marker code is off by default or safely controlled by checkbox.
- Verify marker coordinate calculations use `last_capture_window` and scene rects, not hard-coded screen pixels.
- Verify real learning writes still reject non-`real_capture` sources.
- Inspect diffs after each task and avoid reverting unrelated user changes.
- At major checkpoints, review requirements, UI safety, data integrity, and recognition performance risk.

## Rollback Strategy

- New presentation code should be isolated enough to disable by checkbox or feature flag.
- If click-through is unreliable, fallback to existing in-tool coordinate prompt without deleting the decision logic.
- If structured scoring destabilizes recommendations, keep old total score while adding explanation fields in parallel.

## Default Assumptions

- The user will run the game in borderless/windowed fullscreen for the first usable version.
- The existing `torchlight_infinite` window title remains the target.
- First deliverable does not need final art assets; simple image/emoji-like marker placeholders are acceptable until the behavior is correct.
- No device control, auto-clicking, memory reading, or input injection will be implemented.
- Simulation mode remains separate and does not feed learning logs.
