# Goal 3 Plan

## Objective
Implement the high-score trajectory learning direction discussed with the user: high-score trajectories should become complete state-action-result training data, not just terminal summaries. Add real-data-only collection for new/unknown affixes and cumulative affix appearance percentages so future brush/affix recommendations can use empirical frequencies.

## Context
- Project root is `j:\tools\huahua`; entrypoint is `run_app.py`; core modules are under `huahua_assistant/`.
- Goal 2 completed realtime capture, fast vision, desktop-pet state feedback, performance panel, and trust gates.
- Existing learning storage is in `huahua_assistant/learning.py`, writing JSONL files under `runtime/` and filtering to `source="real_capture"`.
- Existing app recording paths are in `huahua_assistant/app.py`, including quick/full capture steps, finish saving, high-value observation, brush option observation, and learning stats rendering.
- Existing game rules and recommendation scoring are in `huahua_assistant/game_model.py`.

## Requirements
- Define and implement a clearer high-score trajectory mission: save complete trusted per-step trajectories when terminal value crosses meaningful high-score gates.
- Initial high-score gates should include at least: Lv6+ treasure/box or Lv7+ paint. Later threshold escalation should be supported by stats, not hard-coded assumptions only.
- Every saved high-score trajectory should preserve enough state-action-result data for later imitation: board, paints, backpack, current paint, brush choices, inferred action, recommendations, resources, and quality.
- Do not let simulation data enter real learning records.
- Collect new/unknown affixes from real recognition when brush options appear but are not in the catalog or do not map to known codes.
- Count each affix appearance cumulatively and expose percentages for stage1 brush types and stage2 affix names/codes.
- Keep data compact enough to export/merge and compatible with existing JSONL logs.
- Add UI/stat visibility so the user can see high-score gate counts, affix appearance counts, percentages, and unknown/new affixes.
- Keep trust gates: only trusted real-capture data should enter high-score trajectories.

## Default Assumptions
- A high-score trajectory is saved only from trusted real captures and only when terminal value meets the configured/high-value gate.
- Existing `record_high_score_trajectory()` will remain the storage choke point and should continue rejecting untrusted payloads.
- Existing `runtime/observations.jsonl` can store lightweight affix/new-word observations; no separate database is required for this phase.
- Affix appearance percentages can be computed from raw observations at stats-build time to avoid schema migration.
- Unknown affix collection can initially use recognized strings not found in `affix_code()` plus raw brush keyword/score data; deeper OCR/template capture can be added later.
- The first implementation should be deterministic stats/rules, not a machine-learning model.

## Execution Approach
1. Audit current high-score save gates, trajectory payload, brush observation payload, and learning stats/export/merge.
2. Add a compact high-score gate model with explicit reasons and threshold stats.
3. Ensure saved high-score trajectory payloads include complete trusted steps and gate metadata.
4. Add affix appearance percentage aggregation in learning stats, including stage1 and stage2 counts, code counts, and unknown/new affixes.
5. Record unknown/new affix observations from real brush recognition without polluting simulation.
6. Surface the new stats in the right-side UI and correction/stats dialog.
7. Add smoke tests/inline verification for trajectory gate, affix percentages, unknown affix filtering, export/merge trust behavior, and compile.
8. Perform major checks after every 3 tasks and a final review.

## Validation
- Run `python -m compileall huahua_assistant run_app.py` after code changes.
- Run targeted smoke commands for high-score gates and learning stats.
- Verify `record_high_score_trajectory(..., source="simulation")` still returns false.
- Verify low-quality trajectory payloads still fail the storage gate.
- Verify affix percentage stats are computed from only `source="real_capture"` events.
- Verify unknown affix stats ignore known coded affixes and simulation events.
- Run `git diff --check`.

## Risks
- Existing saved trajectories may have older payload shapes; stats must handle them defensively.
- OCR/keyword recognition may produce noisy unknown affix names. Unknown collection should use counts/percentages and not automatically affect recommendations until enough samples exist.
- Recording every low-value full step would bloat data. Full trajectories should stay gated; lower-value map summaries can be added later if needed.
- If high-score gates are too strict early, data collection may be sparse. Configurable thresholds should allow tuning.
- UI text encoding must stay readable in source and runtime.

## Rollback
- New stats should be derived from existing logs and safe to ignore if bad.
- New payload fields should be additive and backward-compatible.
- Recommendation scoring should not depend on new affix percentages until enough samples exist; if needed, it can be disabled by config.
- Storage gates remain centralized in `learning.py` to prevent accidental pollution.
