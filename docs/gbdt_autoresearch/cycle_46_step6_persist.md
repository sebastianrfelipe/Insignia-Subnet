# GBDT Autoresearch Cycle 46 (TSLA) — Step 6 Persist & Hand Off

**Date:** 2025-07-18
**Symbol:** TSLA
**Cycle:** 46
**Procedure:** gbdt_autoresearch

## Summary

Cycle 45 produced an HV breakthrough via AUTORES-GBDT-115 (max_leaf_nodes ceiling 50): hypervolume jumped from 0.1851 → 0.8442 (4.6× champion 105's HV), but best_f1=0.2433 (-6% vs champion 105's 0.2586). The F1 champion AUTORES-GBDT-105 was retained. Cycle 46 parents three new experiments on 115 to recover F1 while holding HV gains.

## Champion

| Field | Value |
|---|---|
| Champion Experiment | AUTORES-GBDT-105 |
| Best F1 | 0.2586 |
| Hypervolume | 0.1851 |
| Gen Gap | 0.449 |
| Mean Sharpe | 0.604 |
| Max Drawdown | 0.193 |
| N Pareto | 14 |

## Cycle 45 Results

| Experiment | Description | Best F1 | Hypervolume | Notes |
|---|---|---|---|---|
| AUTORES-GBDT-115 | max_leaf_nodes ceiling 50 | 0.2433 | 0.8442 | HV breakthrough 4.6× champion, F1 -6% |
| AUTORES-GBDT-116 | lr ceiling 0.03 | 0.1999 | 0.4724 | Underperformed |
| AUTORES-GBDT-117 | minority_class_boost ceiling 6 | 0.2264 | 0.1795 | Mid pack |

## New Experiments Queued (parent: AUTORES-GBDT-115)

| Experiment | Description | Search Space Override | Env Overrides |
|---|---|---|---|
| AUTORES-GBDT-118 | max_leaf_nodes ceiling relax to 65 | max_leaf_nodes.ceiling=65 | — |
| AUTORES-GBDT-119 | 115 space + minority_class_boost ceiling 6 | minority_class_boost.ceiling=6 | — |
| AUTORES-GBDT-120 | 115 space replicate | — | GBDT_RANDOM_STATE=137 |

## Diagnosis

- **Binding metric:** generalization_gap (0.449 / 0.5 constraint = 89.8%)
- **Primary bottleneck:** overfitting_gen_gap_0.449_at_89.8pct_of_constraint
- **Asymmetry:** bearish_sharpe structurally below 0.4 all cycles
- **HV trend:** plateau broken by 115 max_leaf_nodes ceiling
- **Dead zones:** 7
- **Saturation:** 105 narrow optimum, 9 consecutive regressions cycles 42-44

## Escalation

- **Level:** 3 (hold)
- **Consecutive no improvement:** 3 (<< 12 threshold for level 4)
- **Action:** Hold level 3; cycle 46 parents on 115 to recover F1 while holding HV gains

## MongoDB Deliverables Persisted

| Collection | Operation | _id |
|---|---|---|
| convergence_metrics | insert | 6a7fc4247cceb137f804eae0 |
| researcher_insights | insert (cycle_summary) | 6a7fc4247cceb137f804eae2 |
| researcher_insights | insert (diagnosis) | 6a7fc4247cceb137f804eae4 |
| researcher_state | update (_id: gbdt_autoresearch) | gbdt_autoresearch |
| cycle_snapshots | insert (cycle_046) | 6a7fc4247cceb137f804eae7 |
| audit_log | insert (cycle_complete) | 6a7fc4247cceb137f804eae9 |

All documents include `procedure: "gbdt_autoresearch"`. All inserts used `mongodb_insert_one`. The researcher_state update used `mongodb_update_one` with `$set` + `$push` to escalation.history.

## Stop Gate

The `audit_log` entry with `event_type: "cycle_complete"` marks the completion of cycle 46. No advancement to cycle 47.
