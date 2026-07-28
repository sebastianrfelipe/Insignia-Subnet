# GBDT Autoresearch Cycle 38 — Duplicate Document Cleanup

**Date:** 2025-07-28  
**Procedure:** `gbdt_autoresearch`  
**Root Cause:** Step 5-6 persistence was accidentally run twice, creating duplicate documents across 4 MongoDB collections.

---

## Summary of Changes

### TASK 1: experiment_configs — 6 → 3 documents
- **Deleted 3 earlier duplicates** (`_insertedBy: subtask-159b828a-4fd2-4983-b605-5f0b8a90676f`, timestamp `2025-07-18T06:45:00Z`)
- **Kept 3 later documents** (`_insertedBy: subtask-3b022dfe-a351-4f37-aa13-7d8101b2e1c8`, timestamp `2026-07-28T12:00:00Z`)
- Remaining experiment_ids: `AUTORES-GBDT-094`, `AUTORES-GBDT-095`, `AUTORES-GBDT-096`

### TASK 2: convergence_metrics — 2 → 1 document
- **Deleted 1 earlier duplicate** (`_insertedBy: subtask-159b828a...`)
- **Kept 1 later document** (`_insertedBy: subtask-3b022dfe...`)

### TASK 3: researcher_insights — 2 → 1 document
- **Deleted 1 earlier duplicate** (`_insertedBy: subtask-159b828a...`)
- **Kept 1 later document** (`_insertedBy: subtask-3b022dfe...`)

### TASK 4: audit_log — 6 → 2 documents
- **Deleted 4 documents:**
  - 2 earlier duplicates of `cycle_38_configured` and `cycle_37_diagnosis_complete` (`_insertedBy: subtask-159b828a...`)
  - 2 `autoresearch_cycle_triggered` event documents (orphaned trigger records)
- **Kept 2 valid documents:** `cycle_38_configured` and `cycle_37_diagnosis_complete` (`_insertedBy: subtask-3b022dfe...`)

### TASK 5: researcher_state — status fix
- **Updated** `_id: "gbdt_autoresearch"`:
  - `status`: `"cycle_37_analyzed"` → `"cycle_38_queued"`
  - `last_cycle`: 37 (unchanged)
  - `cycle`: 38 (unchanged)
  - `queued_experiments`: `["AUTORES-GBDT-094", "AUTORES-GBDT-095", "AUTORES-GBDT-096"]` (confirmed)

---

## Verification Results

| Collection | Before | After | Expected |
|---|---|---|---|
| experiment_configs (cycle=38) | 6 | 3 | 3 ✅ |
| convergence_metrics (cycle=38, TSLA) | 2 | 1 | 1 ✅ |
| researcher_insights (cycle=38, TSLA) | 2 | 1 | 1 ✅ |
| audit_log (cycle=38, TSLA) | 6 | 2 | 2 ✅ |
| researcher_state status | cycle_37_analyzed | cycle_38_queued | cycle_38_queued ✅ |

All retained documents have `_insertedBy: subtask-3b022dfe-a351-4f37-aa13-7d8101b2e1c8` and timestamp `2026-07-28T12:00:00Z`.

---

## Prevention Recommendations

1. **Idempotency keys**: Before inserting cycle documents, check for existing documents with the same `cycle + procedure + symbol + experiment_id` (or `action`) combination.
2. **Upsert pattern**: Use `mongodb_update_one` with `upsert: true` keyed on a unique compound filter instead of `mongodb_insert_one` for Step 5-6 persistence.
3. **Atomic step guards**: Add a `step_persisted` flag to `researcher_state` to prevent re-running Steps 5-6 if already completed for the current cycle.