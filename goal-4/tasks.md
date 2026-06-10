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

Status: pending

Add compact visible/cumulative backpack detail fields to real trajectory steps, keeping payload size modest and preserving existing `backpack` summary.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 3: Major Checkpoint 1

Status: pending

Review requirement drift, data compatibility, encoding, py_compile, stats rebuild safety, and checklist sync after Tasks 1-2.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 4: Make Candidate Sample Classification Explicit

Status: pending

Strengthen `candidate_low_weight` / `candidate_level6_cluster` handling with explicit export/stat fields or a lightweight candidate event while avoiding high-score trajectory pollution.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 5: Add Top3 Hit Statistics

Status: pending

Extend recommendation hit stats with Top3 hit counts/rates while preserving existing Top1 fields.

Work performed:

Validation evidence:

Remaining risk:

Next step:

## Task 6: Major Checkpoint 2

Status: pending

Review stats correctness, UI/report display, data compatibility, py_compile, encoding, and checklist sync after Tasks 4-5.

Work performed:

Validation evidence:

Remaining risk:

Next step:

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
