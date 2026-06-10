# Goal 4 Tasks

## Task 1: Add Top-Level Initial Aliases

Status: completed

Implement top-level `initial_board` and `initial_anchor` compatibility fields for high-score trajectories and compressed summaries while retaining `initial_snapshot`.

Work performed:

- Added `_with_initial_aliases()` in `huahua_assistant/app.py`.
- Applied the alias helper to manual high-score trajectory payloads, auto-captured high-score trajectory payloads, `run_session_finished` observations, and `compressed_run_summary` payloads.
- Updated `三层高分轨迹方案完成清单.md` to mark the top-level initial alias item and suggested step complete.

Validation evidence:

- `rg -n "_with_initial_aliases|initial_board|initial_anchor|initial_snapshot = self\._initial_snapshot_from_steps" huahua_assistant\app.py`
- `python -m py_compile huahua_assistant\app.py`
- staged-index syntax check: `git show :huahua_assistant/app.py` compiled successfully before commit.
- `python scripts\check_encoding.py .`
- `git diff --cached --check`
- Commit: `0dfc295 Add initial aliases to trajectory payloads`

Remaining risk:

- Existing historical JSONL payloads are not migrated; consumers should handle missing `initial_board` / `initial_anchor` on older records.

Next step:

- Task 2: Add per-step compact backpack details.

## Task 2: Add Per-Step Backpack Compact Details

Status: completed

Add compact visible/cumulative backpack detail fields to real trajectory steps, keeping payload size modest and preserving existing `backpack` summary.

Work performed:

- Added `backpack_visible` and `backpack_cumulative` fields to `_compact_real_capture_step()` payloads while preserving the existing `backpack` summary.
- Added compact visible backpack details with paint code, category, level, treasure flag, slot rectangle, fingerprint, and confidence fields.
- Extended the detected backpack cache to retain compact detail metadata so cumulative backpack entries can carry recent slot/fingerprint/confidence data when available.
- Kept cumulative fallback compact when metadata is missing, so old or partial states still emit code/category/level/treasure data.

Validation evidence:

- `python -m py_compile huahua_assistant\app.py`
- staged-index syntax check: `git show :huahua_assistant/app.py` compiled successfully before commit.
- Smoke test for `_compact_observed_backpack_paint()` and `_compact_cumulative_backpack()`.
- `python scripts\check_encoding.py .`
- `git diff --cached --check`
- Commit: `39dcb88 Add compact backpack details to trajectory steps`

Remaining risk:

- Existing historical trajectory steps are not migrated; consumers should treat `backpack_visible` and `backpack_cumulative` as optional for older records.
- Cumulative entries can only include slot/fingerprint/confidence when the paint came from the current session cache; fallback entries remain compact paint-only records.

Next step:

- Task 3: Major Checkpoint 1.

## Task 3: Major Checkpoint 1

Status: completed

Review requirement drift, data compatibility, encoding, py_compile, stats rebuild safety, and checklist sync after Tasks 1-2.

Work performed:

- Reviewed the first two completed task commits and confirmed they are recorded:
  - `0dfc295 Add initial aliases to trajectory payloads`
  - `39dcb88 Add compact backpack details to trajectory steps`
  - `2e88e3a Record goal task 2 completion`
- Reviewed current worktree drift. `huahua_assistant/app.py` still has unrelated terrain/stats UI display changes left in the working tree; they were not included in Task 1 or Task 2 commits.
- Confirmed `三层高分轨迹方案完成清单.md` marks top-level initial aliases and per-step backpack compact details complete, with remaining items still open.
- Confirmed the next planned task remains candidate/low-weight trajectory classification, so the goal has not drifted into simulation or expert-filter work early.

Validation evidence:

- `python -m py_compile huahua_assistant\app.py huahua_assistant\learning.py huahua_assistant\game_model.py`
- `python scripts\check_encoding.py .`
- `load_learning_stats()` smoke test returned stats including `expert_sample_stage` and `terrain_prior`.
- Compatibility smoke test for `_with_initial_aliases()`, `_compact_observed_backpack_paint()`, and `_compact_cumulative_backpack()`.
- `rg -n "initial_board|initial_anchor|backpack_visible|backpack_cumulative|compressed_run_summary|candidate_low_weight|candidate_level6_cluster|expert_sample_stage|top20_threshold" huahua_assistant goal-4 "三层高分轨迹方案完成清单.md"`
- `git diff --check` exited successfully; it reported only the existing CRLF warning for `runtime/desktop_pet_position.json`.
- The first attempted stats smoke test used bash heredoc syntax and failed in PowerShell; rerun with a PowerShell here-string passed.

Remaining risk:

- The worktree remains dirty with pre-existing or out-of-scope changes, including `huahua_assistant/app.py` terrain/stats UI display changes and generated `__pycache__` files. These should not be treated as completed goal tasks until explicitly reviewed or committed under the right task.
- Current runtime stats show no compressed summaries yet (`compressed_run_summaries: 0`) and no trajectory initial snapshots in existing historical data (`trajectory_initial_snapshots: 0`), so runtime benefits depend on future newly captured records.
- `scripts/check_encoding.py` is available locally and was used for validation, but the repository still has several untracked workspace/encoding helper files from earlier work; this checkpoint does not claim ownership of those files.

Next step:

- Task 4: Make candidate sample classification explicit.

## Task 4: Make Candidate Sample Classification Explicit

Status: completed

Strengthen `candidate_low_weight` / `candidate_level6_cluster` handling with explicit export/stat fields or a lightweight candidate event while avoiding high-score trajectory pollution.

Work performed:

- Kept candidate samples inside the existing `compressed_run_summary` observation path instead of writing them into `high_score_trajectories.jsonl`.
- Added `_compressed_summary_sample_metadata()` in `huahua_assistant/app.py`.
- Added explicit exported payload fields for compressed summaries: `training_weight`, `sample_class`, `sample_role`, `candidate_sample`, and `candidate_metadata`.
- Classified `candidate_high_score`, `candidate_low_weight`, and `candidate_level6_cluster` as candidate samples with low/non-expert roles and explicit `use_for` / `exclude_from` metadata.
- Preserved `weight` as a compatibility alias for existing consumers.

Validation evidence:

- `python -c "import subprocess; data=subprocess.check_output(['git','show',':huahua_assistant/app.py']); compile(data.decode('utf-8'), 'staged huahua_assistant/app.py', 'exec'); print('staged app.py syntax OK')"`
- `python -m py_compile huahua_assistant\app.py`
- Smoke test for `_compressed_summary_sample_metadata()` across `candidate_high_score`, `candidate_low_weight`, `candidate_level6_cluster`, and `low_value_terminal`.
- `python scripts\check_encoding.py .`
- `git diff --cached --check`
- Staged diff review confirmed no terrain UI or special-action recommendation hunks were included.
- Commit: `757aa5b Add explicit candidate sample metadata`

Remaining risk:

- Historical `compressed_run_summary` records do not have the new metadata fields; consumers should fall back to `kind` and `weight` for old data.
- Dedicated aggregate candidate counters were not added in this task to avoid mixing in unrelated pending terrain/statistics work. The explicit fields are available in exported observations through the existing learning bundle export.

Next step:

- Task 5: Add Top3 Hit Statistics.

## Task 5: Add Top3 Hit Statistics

Status: completed

Extend recommendation hit stats with Top3 hit counts/rates while preserving existing Top1 fields.

Work performed:

- Extended `_recommendation_hit_summary()` in `huahua_assistant/learning.py` with `top3`, `top3_rate`, and `top3_pct`.
- Computed Top3 as the sum of `rank_1`, `rank_2`, and `rank_3` recommendation hits from existing real trajectory rank counters.
- Preserved existing `top1`, `top1_rate`, `top1_pct`, `hit_rate`, `hit_pct`, and `rank_counts` fields.
- Kept the change scoped to stats output; existing trajectory capture and rank counters did not need schema changes.

Validation evidence:

- `python -c "import subprocess; data=subprocess.check_output(['git','show',':huahua_assistant/learning.py']); compile(data.decode('utf-8'), 'staged huahua_assistant/learning.py', 'exec'); print('staged learning.py syntax OK')"`
- `python -m py_compile huahua_assistant\learning.py`
- Smoke test for `_recommendation_hit_summary()` confirmed `top3=rank_1+rank_2+rank_3`, `top3_rate`, `top3_pct`, and unchanged rank counts.
- `python scripts\check_encoding.py .`
- `git diff --cached --check`
- Staged diff review confirmed only the Top3 summary fields were included.
- Commit: `6836cd2 Add top3 recommendation hit stats`

Remaining risk:

- Existing historical stats files are rebuilt from raw logs when loaded, but any externally exported old bundle will not contain Top3 until stats are regenerated or exported again.
- UI/report text still primarily displays Top1; Task 6 checkpoint will review whether to surface Top3 in the panel/report without mixing it into this task.

Next step:

- Task 6: Major Checkpoint 2.

## Task 6: Major Checkpoint 2

Status: completed

Review stats correctness, UI/report display, data compatibility, py_compile, encoding, and checklist sync after Tasks 4-5.

Work performed:

- Reviewed Task 4 and Task 5 commits:
  - `757aa5b Add explicit candidate sample metadata`
  - `6836cd2 Add top3 recommendation hit stats`
- Confirmed candidate samples remain in `compressed_run_summary` observations and do not pollute `high_score_trajectories.jsonl`.
- Confirmed Top3 stats are produced by `_recommendation_hit_summary()` as `top3`, `top3_rate`, and `top3_pct` while preserving Top1 fields.
- Found and fixed the UI/report gap where imitation policy details still displayed Top1 only; the report line now displays Top1 and Top3.
- Reviewed worktree drift and kept existing out-of-scope terrain/stats UI, generated cache files, build script, runtime, and workspace helper changes out of this checkpoint's code commit.

Validation evidence:

- `python -m py_compile huahua_assistant\app.py huahua_assistant\learning.py huahua_assistant\game_model.py`
- `python scripts\check_encoding.py .`
- `checkpoint task4 task5 stats smoke OK` smoke test covered `_compressed_summary_sample_metadata()`, `_recommendation_hit_summary()`, `_format_imitation_policy_detail_lines()`, and `load_learning_stats()`.
- `git diff --check` passed with only the existing CRLF warning for `runtime/desktop_pet_position.json`.
- Staged diff review confirmed the Task 6 code commit only changed the Top3 display line.
- Commit: `c948dbe Show top3 recommendation hit stats`

Remaining risk:

- The worktree still contains pre-existing or out-of-scope uncommitted changes, including terrain/stats UI work, generated `__pycache__`, runtime position JSON, build scripts, and workspace encoding helpers.
- Existing historical exported bundles need stats regeneration/export to include the new Top3 fields.
- Runtime data volume is still low for some new stats paths; benefits depend on future real captured trajectories and summaries.

Next step:

- Task 7: Expert Stage Weighting / Selection Metadata.

## Task 7: Expert Stage Weighting / Selection Metadata

Status: pending

Implement stage-aware expert metadata such as `expert_rank_pct`, `expert_selected`, or `expert_weight` in stats/export paths without deleting old trajectories.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 8: Batch Simulation Evaluation Foundation

Status: pending

Add a reproducible batch simulation evaluation command or helper that can run fixed seeds and report baseline policy metrics.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 9: Major Checkpoint 3

Status: pending

Review simulation metrics, determinism, resource use, stats/export compatibility, py_compile, encoding, and checklist sync after Tasks 7-8.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 10: Force/Refresh A/B Evaluation

Status: pending

Extend simulation evaluation to compare force/refresh resource choices against baseline under fixed seeds.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 11: Final Review and Checklist Closure

Status: pending

Perform final review across code, data compatibility, docs, tests, build/compile, encoding, and mark the active goal complete if all planned tasks are done.

Work performed:

Validation evidence:

Remaining risk:

Next step:
