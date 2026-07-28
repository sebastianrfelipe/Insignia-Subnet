# Cycle 38 Duplicate Cleanup — 2026-07-28

## Summary

Duplicate cycle 38 documents were created by two concurrent coder subtasks:
- **subtask-3b022dfe** (original — KEPT)
- **subtask-159b828a** (duplicate — CANCELLED)

## Cleanup Actions

### 1. experiment_configs (6 → 3 active)

3 duplicate experiment configs marked `status: "duplicate_cancelled"`:
- AUTORES-GBDT-094 (duplicate from 159b828a)
- AUTORES-GBDT-095 (duplicate from 159b828a)
- AUTORES-GBDT-096 (duplicate from 159b828a)

3 originals from subtask-3b022dfe remain with `status: "queued"`.

### 2. researcher_state

Updated `_id: "gbdt_autoresearch"`:
- **Before**: `status: "cycle_37_analyzed"`
- **After**: `status: "cycle_38_queued"`

### 3. researcher_insights (2 → 1 active)

1 duplicate from subtask-159b828a marked `status: "duplicate_cancelled"`.
1 original from subtask-3b022dfe preserved.

### 4. convergence_metrics (2 → 1 active)

1 duplicate from subtask-159b828a marked `status: "duplicate_cancelled"`.
1 original from subtask-3b022dfe preserved.

### 5. audit_log (6 → 2 active + 1 cleanup entry)

4 duplicates cancelled:
- 2 from subtask-159b828a (cycle_38_configured + cycle_37_diagnosis_complete)
- 2 from gbdt_cycle_orchestrator_manual (autoresearch_cycle_triggered)

2 originals from subtask-3b022dfe preserved (cycle_38_configured + cycle_37_diagnosis_complete).

1 new audit entry added documenting this cleanup.

## Verification Counts

| Collection | Active | Cancelled | Total |
|---|---|---|---|
| experiment_configs | 3 (queued) | 3 (duplicate_cancelled) | 6 |
| researcher_insights | 1 | 1 (duplicate_cancelled) | 2 |
| convergence_metrics | 1 | 1 (duplicate_cancelled) | 2 |
| audit_log | 3* | 4 (duplicate_cancelled) | 7* |
| researcher_state | 1 (cycle_38_queued) | — | 1 |

*audit_log active count includes the new cleanup entry.

## Root Cause

Two coder subtasks ran concurrently for cycle 38 configuration, both inserting the same experiment configs, insights, metrics, and audit entries. The orchestrator triggered the cycle twice (two `autoresearch_cycle_triggered` events), causing both subtasks to execute.

## Prevention

Future cycles should use a lock/check in researcher_state before inserting cycle documents, e.g. check `status != "cycle_N_configured"` before proceeding with cycle N configuration.