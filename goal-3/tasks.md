# Goal 3 Tasks

## Task 1 - Audit Learning And Trajectory Baseline
- Status: completed
- Goal: Inspect current high-score trajectory gates, payload shape, brush/affix observations, learning stats, export/merge, and UI stats insertion points.
- Work performed: Audited the current high-score and learning pipeline. High-score saves currently flow through `HuahuaApp._finish_real_session_internal()` and `_auto_record_real_trajectory()`, then through `learning.record_high_score_trajectory()`. Storage already rejects non-`real_capture` sources and requires every saved step to have `quality.ok is True`. Current trajectory steps from `_compact_real_capture_step()` include board state counts/signature, current rule, anchor/pending state, backpack summary, brush stage1/stage2 plus codes, resources, tile paint compact list, quality, top recommendations, and inferred actions after later stamping. Current auto-save gate is `_should_save_auto_trajectory()`, which supports score, treasure7, treasure6, many lv7, mixed lv7, and treasure5+lv7 reasons, but it does not yet expose structured gate metadata or statistics for the user's explicit initial threshold. Manual finish currently saves any terminal that passes `_terminal_reliability_for_finish()`, not a strict high-score gate. Brush observations are recorded in `_record_scene_observation()` as `brush_options` with stage1/stage2 options and codes. `learning._build_stats()` already counts `stage1_seen`, `stage1_code_seen`, `affix_seen`, `affix_code_seen`, plus trajectory brush/affix counts, but does not compute percentages or unknown/new affix lists. UI insertion points are `HuahuaApp._write_learning_stats()`, `_format_correction_stats()`, and `_top_affix_display_items()`. Recommendation already consumes `trajectory_brushes`/`trajectory_affixes` via policy priors in `game_model.py`, so future percentage stats can feed recommendation without replacing the existing deterministic scoring.
- Validation evidence: `rg` located the relevant high-score gates and write paths in `huahua_assistant/app.py` at `_finish_real_session_internal`, `_auto_record_real_trajectory`, `_compact_real_capture_step`, `_should_save_auto_trajectory`, and `_summary_has_high_value`; learning storage/stat paths in `huahua_assistant/learning.py` at `record_high_score_trajectory`, `_trajectory_payload_is_trusted`, export/merge, and `_build_stats`; UI paths in `HuahuaApp._write_learning_stats()` and `_format_correction_stats()`; recommendation policy-prior paths in `game_model.py` for placement/brush/affix sample thresholds. Smoke command `load_learning_stats()` returned a dict with 4 trusted trajectories and existing `affix_seen`/`trajectory_affixes` keys. `_should_save_auto_trajectory()` returned `(True, 'score_3006')` for Lv6 treasure and `(True, 'many_level7')` for three Lv7 paints. `record_high_score_trajectory(..., source='simulation')` returned `False`. `affix_code('不存在的新词条') is None` returned `True`, confirming an unknown-affix detector can key off missing codes.
- Remaining risk: Current terminal summary counts `treasure6` and `treasure7` as cumulative `>=6` / `>=7`, so future gate metadata must be careful to label them as Lv6+ and Lv7+ rather than exact counts. Manual finish can still save reliable but lower-value endings if terminal reliability passes; Task 2 should align manual and auto save behavior with explicit high-score gates. Unknown/new affix collection must handle noisy recognized strings and should not affect recommendation until enough real samples exist.
- Next step: Task 2 should implement an explicit high-score gate model with structured gate reasons for Lv6+ treasure/box and Lv7+ paints, add configurable thresholds, and prepare stats fields for later threshold escalation.

## Task 2 - High-Score Gate Model
- Status: completed
- Goal: Implement explicit high-score gate reasons for Lv6+ treasure/box, Lv7+ paint, and configurable future threshold escalation.
- Work performed: Added explicit high-score gate configuration in `huahua_assistant/config.py` with `autosave.min_high_score_treasure6` and `autosave.min_high_score_lv7`. Added `_high_score_gate()` in `huahua_assistant/app.py`, returning structured metadata: `passed`, `primary_reason`, all `reasons`, human summary, score, thresholds, and terminal counts. The gate treats Lv6+ treasure/box and any Lv7+ paint as initial high-score gates, while preserving score, treasure7, many-level7, mixed-level7, and treasure5+level7 reasons for escalation/analysis. Updated manual finish to skip high-score trajectory storage when a reliable terminal does not pass the high-score gate, recording `run_session_finish_skipped` with gate diagnostics instead. Updated auto trajectory saving to use the same gate and include `high_score_gate` in saved payloads. Added storage-level defense in `learning._trajectory_payload_is_trusted()` so a payload explicitly marked `high_score_gate.passed=False` cannot be written even if all steps are trusted. Added `high_score_gate_primary` and `high_score_gate_reasons` aggregation in learning stats.
- Validation evidence: Ran `python -m compileall huahua_assistant run_app.py` successfully. Ran `git diff --check` successfully with only CRLF normalization warnings. Smoke-tested `_high_score_gate()` returning `False/below_high_score_gate` for low-value terminals, `True/treasure6_plus` with reasons `['treasure6_plus', 'score_3006']` for a Lv6 treasure terminal, and `True/level7_paint` for a Lv7 paint terminal. Smoke-tested `_should_save_auto_trajectory()` returning `(True, 'treasure6_plus')` for Lv6 treasure, preserving old call compatibility. Smoke-tested learning stats aggregating `{'treasure6_plus': 1}` primary gate and both `treasure6_plus` / `score_3006` reasons. Smoke-tested `record_high_score_trajectory()` rejecting a real-capture payload with `high_score_gate.passed=False` and accepting one with `passed=True` plus trusted steps.
- Remaining risk: Older saved trajectories without `high_score_gate` remain accepted for backward compatibility and are counted using their legacy `reason`; future tasks should display both legacy and new gate stats clearly. Manual reliable but low-value endings now no longer enter high-score trajectories, which matches the new mission but may reduce early sample volume until real high-value captures accumulate.
- Next step: Task 3 should perform the first major check to verify the new gate model does not break trust boundaries, old trajectory compatibility, compile, stats loading, or UI assumptions.

## Task 3 - Major Check 1
- Status: pending
- Goal: Verify gate changes do not break trust boundaries, old trajectory compatibility, compile, or stats loading.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 4 - Complete Trajectory Payload Metadata
- Status: pending
- Goal: Ensure saved high-score trajectories carry gate metadata and complete trusted per-step state-action-result context needed for imitation analysis.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 5 - Affix Appearance Percentages
- Status: pending
- Goal: Aggregate stage1 brush and stage2 affix appearance counts, code counts, and percentages from real observations.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 6 - Major Check 2
- Status: pending
- Goal: Verify trajectory metadata and affix percentage stats are real-data-only, compact, export/merge-safe, and backward-compatible.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 7 - Unknown/New Affix Collection
- Status: pending
- Goal: Record and report recognized brush/affix strings that do not map to known catalog codes, with counts and confidence samples.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 8 - UI Stats Surface
- Status: pending
- Goal: Show high-score gate counts, affix percentages, and unknown/new affixes in the right-side learning panel and stats dialog.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 9 - Major Check 3
- Status: pending
- Goal: Broad check for UI readability, data safety, stats correctness, compile, smoke tests, and rollback readiness.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 10 - Imitation Analysis Hooks
- Status: pending
- Goal: Add lightweight derived stats for later imitation: top actions/placements/brush choices among saved high-score trajectories and sample thresholds for when to trust them.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:

## Task 11 - Final Review
- Status: pending
- Goal: Final review of high-score trajectory mission, new affix collection, percentage stats, safety gates, docs impact, and next live-data validation steps.
- Work performed:
- Validation evidence:
- Remaining risk:
- Next step:
